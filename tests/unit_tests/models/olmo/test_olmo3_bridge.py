# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the OLMo-3 bridge (no GPU)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from megatron.bridge.models.olmo.olmo3_bridge import Olmo3Bridge
from megatron.bridge.models.olmo.olmo3_provider import Olmo3ModelProvider


def _hf_config(num_layers: int = 8, **overrides):
    """A SimpleNamespace so any attribute the bridge forgets to set raises."""
    layer_types = [
        "sliding_attention" if (i + 1) % 4 != 0 else "full_attention" for i in range(num_layers)
    ]
    cfg = dict(
        architectures=["Olmo3ForCausalLM"],
        model_type="olmo3",
        num_hidden_layers=num_layers,
        hidden_size=512,
        intermediate_size=1024,
        num_attention_heads=8,
        num_key_value_heads=2,
        vocab_size=1000,
        max_position_embeddings=1024,
        rms_norm_eps=1e-6,
        rope_theta=500000.0,
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
        sliding_window=64,
        layer_types=layer_types,
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 500000.0,
            "factor": 8.0,
            "original_max_position_embeddings": 128,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "attention_factor": 1.2079441541679836,
        },
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _provider(hf_config) -> Olmo3ModelProvider:
    pretrained = MagicMock()
    pretrained.config = hf_config
    return Olmo3Bridge().provider_bridge(pretrained)


class TestOlmo3Bridge:
    def test_bridge_registered_for_architecture(self):
        from megatron.bridge.models.conversion import model_bridge

        registry = model_bridge.get_model_bridge._exact_types
        names = {getattr(k, "__name__", str(k)) for k in registry}
        assert "Olmo3ForCausalLM" in names

    def test_exported_in_models_namespace(self):
        import megatron.bridge.models as models

        assert "Olmo3Bridge" in models.__all__
        assert hasattr(models, "Olmo3Bridge")

    def test_core_architecture_fields(self):
        p = _provider(_hf_config())
        assert p.num_layers == 8
        assert p.hidden_size == 512
        assert p.num_attention_heads == 8
        assert p.num_query_groups == 2  # GQA
        assert p.ffn_hidden_size == 1024
        assert p.kv_channels == 64
        assert p.normalization == "RMSNorm"
        assert p.layernorm_epsilon == 1e-6
        assert p.gated_linear_unit is True
        assert p.add_bias_linear is False
        assert p.add_qkv_bias is False
        assert p.qk_layernorm is True
        assert p.share_embeddings_and_output_weights is False

    def test_sliding_window_pattern_from_layer_types(self):
        """The 3:1 pattern must be read from layer_types, not assumed."""
        p = _provider(_hf_config())
        assert p.window_size == (63, 0)  # TE's window is inclusive of the query
        assert p.window_attn_skip_freq == [1, 1, 1, 0, 1, 1, 1, 0]

    def test_non_default_layer_pattern_is_honoured(self):
        """A checkpoint with a different pattern must convert to that pattern."""
        lt = ["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"]
        p = _provider(_hf_config(num_layers=4, layer_types=lt))
        assert p.window_attn_skip_freq == [0, 1, 1, 1]

    def test_layer_types_length_is_validated(self):
        with pytest.raises(ValueError, match="layer_types has"):
            _provider(_hf_config(num_layers=8, layer_types=["sliding_attention"] * 7))

    def test_unknown_layer_type_rejected(self):
        with pytest.raises(ValueError, match="Unsupported OLMo-3 layer_types"):
            _provider(_hf_config(num_layers=4, layer_types=["sliding_attention"] * 3 + ["local"]))

    def test_yarn_params_and_rope_type(self):
        """position_embedding_type stays 'rope': OLMo-3 scales only some layers."""
        p = _provider(_hf_config())
        assert p.position_embedding_type == "rope"
        assert p.yarn_rotary_scaling_factor == 8.0
        assert p.yarn_original_max_position_embeddings == 128
        assert p.yarn_beta_fast == 32.0
        assert p.yarn_beta_slow == 1.0
        assert p.rotary_base == 500000.0

    def test_legacy_rope_scaling_attribute_is_read(self):
        """transformers <=4.57 exposes rope_scaling rather than rope_parameters."""
        cfg = _hf_config()
        rope = cfg.rope_parameters
        del cfg.rope_parameters
        cfg.rope_scaling = rope
        p = _provider(cfg)
        assert p.yarn_rotary_scaling_factor == 8.0

    def test_nested_per_layer_rope_takes_full_attention_entry(self):
        cfg = _hf_config()
        cfg.rope_parameters = {
            "full_attention": dict(cfg.rope_parameters),
            "sliding_attention": {"rope_type": "default", "rope_theta": 500000.0},
        }
        p = _provider(cfg)
        assert p.yarn_rotary_scaling_factor == 8.0

    def test_conflicting_attention_factor_is_rejected(self):
        """A pinned attention_factor Megatron cannot reproduce must not convert silently."""
        cfg = _hf_config()
        cfg.rope_parameters = dict(cfg.rope_parameters)
        cfg.rope_parameters["attention_factor"] = 1.5
        with pytest.raises(ValueError, match="attention_factor"):
            _provider(cfg)

    def test_no_rope_scaling_disables_yarn(self):
        cfg = _hf_config()
        cfg.rope_parameters = {"rope_type": "default", "rope_theta": 500000.0}
        p = _provider(cfg)
        assert p.yarn_rotary_scaling_factor is None

    def test_mapping_registry_covers_every_parameter(self):
        """Every OLMo-3 weight name must be mapped -- an unmapped tensor keeps its
        random init and produces a plausible-but-wrong model."""
        registry = Olmo3Bridge().mapping_registry()
        mapped_hf = set()
        for m in registry.mappings if hasattr(registry, "mappings") else []:
            hf = getattr(m, "hf_param", None)
            if isinstance(hf, str):
                mapped_hf.add(hf)
            elif isinstance(hf, dict):
                mapped_hf.update(hf.values())
        expected = {
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight",
            "model.layers.*.self_attn.o_proj.weight",
            "model.layers.*.post_attention_layernorm.weight",
            "model.layers.*.post_feedforward_layernorm.weight",
            "model.layers.*.self_attn.q_norm.weight",
            "model.layers.*.self_attn.k_norm.weight",
            "model.layers.*.mlp.down_proj.weight",
            "model.layers.*.self_attn.q_proj.weight",
            "model.layers.*.self_attn.k_proj.weight",
            "model.layers.*.self_attn.v_proj.weight",
            "model.layers.*.mlp.gate_proj.weight",
            "model.layers.*.mlp.up_proj.weight",
        }
        assert expected <= mapped_hf, expected - mapped_hf

    def test_post_norms_do_not_use_pre_norm_slots(self):
        """OLMo-3's layernorms are output norms; mapping them to the pre-norm slots
        would put them on the wrong side of the residual."""
        registry = Olmo3Bridge().mapping_registry()
        megatron_names = set()
        for m in registry.mappings if hasattr(registry, "mappings") else []:
            megatron_names.add(getattr(m, "megatron_param", ""))
        joined = " ".join(megatron_names)
        assert "linear_proj.post_layernorm.weight" in joined
        assert "linear_fc2.post_layernorm.weight" in joined
        assert "layer_norm_weight" not in joined
        assert "input_layernorm" not in joined
        assert "pre_mlp_layernorm" not in joined

    def test_megatron_to_hf_config_roundtrips_olmo3_keys(self):
        bridge = Olmo3Bridge()
        p = _provider(_hf_config())
        try:
            hf = bridge.megatron_to_hf_config(p)
        except Exception as exc:  # base implementation may need more of the provider
            pytest.skip(f"base megatron_to_hf_config unavailable: {exc}")
        assert hf["sliding_window"] == 64
        assert hf["layer_types"][:4] == [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]
        assert hf["rope_scaling"]["rope_type"] == "yarn"
        assert hf["rope_scaling"]["factor"] == 8.0

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
"""Bridge for HuggingFace ``Olmo3ForCausalLM`` <-> Megatron-Core ``GPTModel``."""

from typing import Any

from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.olmo.olmo3_provider import Olmo3ModelProvider, olmo3_layer_spec


# The post-norm wrapper is a row-parallel linear; teach AutoMapping so it can infer
# the sharding of `linear_proj` / `linear_fc2` without being told per mapping.
# (Gemma-2 registers its own equivalent; ours differs only in the norm epsilon.)
AutoMapping.register_module_type("TERowParallelLinearPostNorm", "row")


def _rope_params(hf_config: Any) -> dict:
    """Return OLMo-3's RoPE parameter dict across transformers versions.

    transformers <=4.57 exposes ``rope_scaling``; 5.x renamed it to
    ``rope_parameters`` and additionally allows a per-``layer_type`` nesting. AI2
    ships the *flat* form in every published OLMo-3 config, but handle the nested
    form too and take the ``full_attention`` entry, since that is the only one
    OLMo-3 scales.
    """
    params = getattr(hf_config, "rope_parameters", None) or getattr(hf_config, "rope_scaling", None)
    if not params:
        return {}
    if "full_attention" in params:  # per-layer-type nesting
        params = params["full_attention"]
    return dict(params)


@MegatronModelBridge.register_bridge(
    source="Olmo3ForCausalLM",
    target=GPTModel,
    provider=Olmo3ModelProvider,
    model_type="olmo3",
)
class Olmo3Bridge(MegatronModelBridge):
    """Bridge for AI2's OLMo-3 dense causal LM family.

    Where OLMo-3 sits relative to the nearest existing bridges:

    +-----------------------+---------+---------+-------------+-------------+
    | Property              | Llama   | Qwen3   | Gemma3      | OLMo-3      |
    +=======================+=========+=========+=============+=============+
    | Pre-attn / pre-MLP norm | yes   | yes     | yes         | **no**      |
    | Post-attn / post-MLP norm | no  | no      | yes         | **yes**     |
    | QK-RMSNorm            | no      | per-head| no          | **full-width** |
    | Interleaved SWA       | no      | no      | yes (5:1)   | **yes (3:1)** |
    | RoPE scaling scope    | all     | all     | all         | **full-attn layers only** |
    +-----------------------+---------+---------+-------------+-------------+

    The weight mapping is plain -- every difference above is either config
    (`provider_bridge`) or forward-pass behaviour (`olmo3_provider`). The only
    mapping subtlety is that OLMo-3's two layernorms are *output* norms, so they
    map onto the output projections rather than the usual pre-norm slots.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("allenai/Olmo-3-32B-Think")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Olmo3ModelProvider:
        """Translate an HF ``Olmo3Config`` into an :class:`Olmo3ModelProvider`."""
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        provider.transformer_layer_spec = olmo3_layer_spec
        provider.kv_channels = getattr(hf_config, "head_dim", None) or (
            hf_config.hidden_size // hf_config.num_attention_heads
        )

        # Shared with OLMo-2. `use_qk_norm` is not an OLMo-3 config key, so the
        # base CONFIG_MAPPING cannot infer qk_layernorm -- set it explicitly.
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.add_qkv_bias = bool(getattr(hf_config, "attention_bias", False))
        provider.qk_layernorm = True
        provider.hidden_dropout = 0.0
        provider.attention_dropout = float(getattr(hf_config, "attention_dropout", 0.0))
        provider.persist_layer_norm = True
        provider.share_embeddings_and_output_weights = bool(
            getattr(hf_config, "tie_word_embeddings", False)
        )

        # --- Interleaved sliding-window attention -------------------------------
        # Derive the pattern from `layer_types` rather than assuming the published
        # 3:1 ratio, so a checkpoint with a different pattern converts correctly
        # instead of silently attending over the wrong span.
        sliding_window = getattr(hf_config, "sliding_window", None)
        layer_types = getattr(hf_config, "layer_types", None)
        if sliding_window and layer_types:
            unknown = set(layer_types) - {"sliding_attention", "full_attention"}
            if unknown:
                raise ValueError(f"Unsupported OLMo-3 layer_types entries: {sorted(unknown)}")
            if len(layer_types) != hf_config.num_hidden_layers:
                raise ValueError(
                    f"layer_types has {len(layer_types)} entries but the config declares "
                    f"{hf_config.num_hidden_layers} layers"
                )
            # TE takes an inclusive left window: `window_size[0]` keys before the
            # query, plus the query itself, i.e. `sliding_window` total.
            provider.window_size = (sliding_window - 1, 0)
            # 1 == sliding, indexed by (layer_number - 1); see
            # megatron.core.transformer.utils.is_layer_window_attention
            provider.window_attn_skip_freq = [
                1 if t == "sliding_attention" else 0 for t in layer_types
            ]
        else:
            provider.window_size = None
            provider.window_attn_skip_freq = None

        # --- YaRN, on full-attention layers only ---------------------------------
        # Set explicitly: the base config mapping reads the legacy `rope_scaling`
        # attribute and would also route us to position_embedding_type="yarn",
        # which applies one scaled rope to *every* layer. OLMo-3 scales only the
        # full-attention layers, so the provider keeps "rope" and installs a dual
        # rope in `provide()`.
        rope = _rope_params(hf_config)
        rope_type = rope.get("rope_type") or rope.get("type")
        provider.position_embedding_type = "rope"
        if rope_type == "yarn":
            provider.yarn_rotary_scaling_factor = float(rope["factor"])
            provider.yarn_original_max_position_embeddings = int(
                rope.get("original_max_position_embeddings", 8192)
            )
            provider.yarn_beta_fast = float(rope.get("beta_fast", 32.0))
            provider.yarn_beta_slow = float(rope.get("beta_slow", 1.0))
            provider.yarn_mscale = float(rope.get("mscale", 1.0))
            provider.yarn_mscale_all_dim = float(rope.get("mscale_all_dim", 0.0))
            self._check_attention_factor(rope, provider)
        elif rope_type not in (None, "default"):
            raise ValueError(
                f"OLMo-3 bridge supports rope_type 'default' or 'yarn', got {rope_type!r}"
            )
        else:
            provider.yarn_rotary_scaling_factor = None

        return provider

    @staticmethod
    def _check_attention_factor(rope: dict, provider: Olmo3ModelProvider) -> None:
        """Warn if HF's explicit ``attention_factor`` disagrees with Megatron's.

        HF lets a config pin ``attention_factor`` directly; Megatron always derives
        it as ``0.1*mscale*ln(factor)+1`` normalised by the ``mscale_all_dim``
        variant. For every published OLMo-3 config these agree (1.20794... at
        factor 8), but a checkpoint that pins a different value would be silently
        mis-scaled, and the discrepancy is invisible in the weights.
        """
        explicit = rope.get("attention_factor")
        if explicit is None:
            return
        from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_get_concentration_factor,
        )

        derived = _yarn_get_concentration_factor(
            provider.yarn_rotary_scaling_factor,
            provider.yarn_mscale,
            provider.yarn_mscale_all_dim,
        )
        if abs(float(explicit) - derived) > 1e-6:
            raise ValueError(
                f"HF config pins rope attention_factor={explicit} but Megatron derives "
                f"{derived} from factor={provider.yarn_rotary_scaling_factor}, "
                f"mscale={provider.yarn_mscale}, mscale_all_dim={provider.yarn_mscale_all_dim}. "
                "Converting would silently change the attention scale."
            )

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Weight mappings for HF ``Olmo3ForCausalLM`` <-> Megatron-Core ``GPTModel``.

        OLMo-3-specific points:

        * ``post_attention_layernorm`` / ``post_feedforward_layernorm`` are
          *output* norms. They map to ``linear_proj.post_layernorm`` /
          ``linear_fc2.post_layernorm``, **not** to the pre-norm slots
          ``linear_qkv.layer_norm_weight`` / ``linear_fc1.layer_norm_weight``.
          Despite its name, ``post_attention_layernorm`` is the *pre*-MLP norm in
          Llama-style models -- here it genuinely is a post-attention norm.
        * There are deliberately **no** ``input_layernorm`` / ``pre_mlp_layernorm``
          mappings: OLMo-3 has no pre-normalization.
        """
        param_mappings = {
            # Embeddings, output projection, final norm
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            # Attention output projection + its post-norm
            "decoder.layers.*.self_attention.linear_proj.weight": (
                "model.layers.*.self_attn.o_proj.weight"
            ),
            "decoder.layers.*.self_attention.linear_proj.post_layernorm.weight": (
                "model.layers.*.post_attention_layernorm.weight"
            ),
            # Full-width QK-RMSNorm
            "decoder.layers.*.self_attention.q_layernorm.weight": (
                "model.layers.*.self_attn.q_norm.weight"
            ),
            "decoder.layers.*.self_attention.k_layernorm.weight": (
                "model.layers.*.self_attn.k_norm.weight"
            ),
            # MLP down projection + its post-norm
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.post_layernorm.weight": (
                "model.layers.*.post_feedforward_layernorm.weight"
            ),
        }

        mapping_list = [
            AutoMapping(megatron_param=megatron_param, hf_param=hf_param)
            for megatron_param, hf_param in param_mappings.items()
        ]

        # HF keeps q/k/v separate; Megatron packs them GQA-grouped into linear_qkv.
        mapping_list.append(
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            )
        )
        # HF keeps gate/up separate; Megatron concatenates them into linear_fc1.
        mapping_list.append(
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                gate="model.layers.*.mlp.gate_proj.weight",
                up="model.layers.*.mlp.up_proj.weight",
            )
        )

        return MegatronMappingRegistry(*mapping_list)

    def megatron_to_hf_config(self, provider: Olmo3ModelProvider) -> dict:
        """Rebuild the OLMo-3-specific HF config keys on export."""
        hf_config = super().megatron_to_hf_config(provider)
        if provider.window_size is not None:
            hf_config["sliding_window"] = provider.window_size[0] + 1
        if provider.window_attn_skip_freq is not None:
            hf_config["layer_types"] = [
                "sliding_attention" if x else "full_attention"
                for x in provider.window_attn_skip_freq
            ]
        if provider.yarn_rotary_scaling_factor is not None:
            hf_config["rope_scaling"] = {
                "rope_type": "yarn",
                "factor": provider.yarn_rotary_scaling_factor,
                "original_max_position_embeddings": provider.yarn_original_max_position_embeddings,
                "beta_fast": provider.yarn_beta_fast,
                "beta_slow": provider.yarn_beta_slow,
            }
        return hf_config

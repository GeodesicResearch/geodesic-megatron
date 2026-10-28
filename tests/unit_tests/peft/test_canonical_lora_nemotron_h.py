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
"""Canonical LoRA on a small real Nemotron-H Latent-MoE model: every Super target, forward and merge."""

import pytest
import torch

from megatron.bridge.recipes.nemotronh.nemotron_3_super import NEMOTRON_3_SUPER_CANONICAL_LORA_TARGETS
from tests.unit_tests.one_rank_nccl_world import one_rank_model_parallel_state


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (Mamba and grouped-GEMM kernels)"),
    pytest.mark.skipif(not hasattr(torch, "_grouped_mm"), reason="needs torch._grouped_mm"),
]

VOCAB, SEQ = 256, 64
#: B-factor scale whose LoRA updates are mostly below half a bf16 step of their base weights.
B_STD_SMALL = 1e-5


@pytest.fixture(scope="module")
def parallel_state():
    with one_rank_model_parallel_state(seed=1234, expert_model_parallel_size=1):
        yield


def _model():
    """Mamba, attention and Latent-MoE layers (with a shared expert) on the GroupedExperts path."""
    from megatron.bridge.models.nemotronh.nemotron_h_provider import NemotronHModelProvider

    torch.manual_seed(0)
    provider = NemotronHModelProvider(
        hybrid_layer_pattern="M*E",
        num_layers=3,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=32,
        mamba_num_heads=4,
        mamba_head_dim=64,
        mamba_num_groups=2,
        mamba_state_dim=16,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_ffn_hidden_size=64,
        moe_latent_size=32,
        moe_shared_expert_intermediate_size=128,
        moe_shared_expert_overlap=False,
        moe_experts_impl="torch_grouped",
        moe_aux_loss_coeff=0.0,
        moe_router_bias_update_rate=0.0,
        mtp_num_layers=None,
        vocab_size=VOCAB,
        seq_length=SEQ,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
    )
    provider.finalize()
    return provider.provide_distributed_model(wrap_with_ddp=False)


def _adapted(model=None, b_std=0.05):
    """CanonicalLoRA with every Super target on ``model`` (a new small model if None), B factors ~ N(0, b_std)."""
    from megatron.bridge.peft.canonical_lora import CanonicalLoRA

    model = model if model is not None else _model()
    lora = CanonicalLoRA(target_modules=list(NEMOTRON_3_SUPER_CANONICAL_LORA_TARGETS), lora_A_init_method="kaiming")
    model = lora(model, training=False)
    # Non-zero B factors, so that every adapter changes the output.
    with torch.no_grad():
        for name, param in model[0].named_parameters():
            if ".adapter." in name and ("linear_out" in name or name.endswith(("fc1_b", "fc2_b"))):
                param.normal_(std=b_std)
    return model


def _logits(model):
    torch.manual_seed(1)
    tokens = torch.randint(0, VOCAB, (1, SEQ), device="cuda")
    position_ids = torch.arange(SEQ, device="cuda").unsqueeze(0)
    with torch.no_grad():
        return model[0](tokens, position_ids, attention_mask=None).float()


def test_every_super_target_gets_an_adapter(parallel_state):
    from megatron.bridge.peft.canonical_lora import LoRALinearSplitMambaInProj, LoRALinearSplitQKV
    from megatron.bridge.peft.grouped_experts_lora import LoRAGroupedExperts
    from megatron.bridge.peft.lora_layers import LoRALinear

    model = _adapted()
    kinds = {name.rsplit(".", 1)[-1]: type(m) for name, m in model[0].named_modules()}
    assert kinds["linear_qkv"] is LoRALinearSplitQKV
    assert kinds["in_proj"] is LoRALinearSplitMambaInProj
    assert kinds["experts"] is LoRAGroupedExperts
    for plain in ("linear_proj", "out_proj", "fc1_latent_proj", "fc2_latent_proj", "linear_fc1", "linear_fc2"):
        assert kinds[plain] is LoRALinear, plain
    assert "router" not in {n for n, m in model[0].named_modules() if isinstance(m, LoRALinear)}
    trainable = [n for n, p in model[0].named_parameters() if p.requires_grad]
    assert trainable == []  # training=False freezes everything, adapters included


def test_mamba_in_proj_adapts_z_and_x_only(parallel_state):
    model = _adapted()
    in_proj = next(m for n, m in model[0].named_modules() if n.endswith("mixer.in_proj"))
    base_weight = in_proj.to_wrap.weight.detach().clone()
    in_proj.merge_adapter()
    delta = (in_proj.to_wrap.weight.float() - base_weight.float()).abs().sum(dim=1)
    d_inner = 4 * 64
    assert delta[: 2 * d_inner].gt(0).all()
    assert delta[2 * d_inner :].eq(0).all()


def test_merged_model_reproduces_adapted_logits(parallel_state):
    from megatron.bridge.peft.adapter_wrapper import AdapterWrapper
    from megatron.bridge.peft.lora import merge_and_unwrap_adapters

    model = _adapted()
    base_names = {n for n, _ in _model()[0].named_parameters()}
    adapted = _logits(model)
    model = merge_and_unwrap_adapters(model)
    assert not any(isinstance(m, AdapterWrapper) for m in model[0].modules())
    assert {n for n, _ in model[0].named_parameters()} == base_names
    merged = _logits(model)
    # bf16 weights: the merged sum rounds once per weight; compare at bf16 resolution of the logits.
    torch.testing.assert_close(merged, adapted, rtol=3e-2, atol=3e-2)


def test_stochastic_rounding_keeps_a_small_update(parallel_state):
    """Updates mostly below half a bf16 step: round-to-nearest drops most of them, stochastic keeps them on average.

    Measured on the weights (``LoRAMerge.weight_retained``). The logit-level ``retained`` of ``check_merge`` cannot
    show it on this model, whose bf16 activations round an effect this small in the adapted forward too.
    """
    from megatron.bridge.peft.lora import LoRAMerge, merge_and_unwrap_adapters

    retained = {}
    for seed in (None, 0):
        merger = LoRAMerge(stochastic_rounding_seed=seed)
        merge_and_unwrap_adapters(_adapted(b_std=B_STD_SMALL), merger)
        retained[seed] = merger.weight_retained()
    assert retained[None] < 0.5, retained
    assert retained[0] == pytest.approx(1.0, abs=0.02), retained


def test_adapters_change_the_output(parallel_state):
    model = _adapted()
    adapted = _logits(model)
    for chunk in model:
        for module in chunk.modules():
            if hasattr(module, "disable_adapter_layers"):
                module.disable_adapter_layers()
    assert (adapted - _logits(model)).abs().max() > 1e-2


def _peft_checkpoint(tmp_path):
    """A base checkpoint and a PEFT checkpoint of the small model, as a training run leaves them.

    The PEFT checkpoint's run_config names the base under ``checkpoint.pretrained_checkpoint`` and the adapters under
    ``peft``; its tensors include the base weights, which ``load_peft_model`` must ignore in favour of the base's.
    Returns (adapted in-memory model, PEFT iteration directory).
    """
    import yaml

    from megatron.bridge.training.model_load_save import save_megatron_model

    model = _model()
    save_megatron_model(model, tmp_path / "base")
    adapted = _adapted(model)
    save_megatron_model(adapted, tmp_path / "peft", iteration=7)
    peft_iter = tmp_path / "peft" / "iter_0000007"
    run_config = yaml.safe_load((peft_iter / "run_config.yaml").read_text())
    run_config["checkpoint"]["pretrained_checkpoint"] = str(tmp_path / "base")
    run_config["peft"] = {
        "_target_": "megatron.bridge.peft.canonical_lora.CanonicalLoRA",
        "target_modules": list(NEMOTRON_3_SUPER_CANONICAL_LORA_TARGETS),
        "dim": 32,
        "alpha": 32,
        "lora_A_init_method": "kaiming",
        "canonical_mapping": {},
    }
    (peft_iter / "run_config.yaml").write_text(yaml.safe_dump(run_config))
    return adapted, peft_iter


def test_load_peft_model_restores_the_adapted_model(parallel_state, tmp_path):
    from megatron.bridge.training.model_load_save import load_peft_model

    adapted, peft_iter = _peft_checkpoint(tmp_path)
    loaded = load_peft_model(str(peft_iter))
    torch.testing.assert_close(_logits(loaded), _logits(adapted), rtol=0, atol=0)


def test_load_peft_model_refuses_a_checkpoint_without_the_adapters(parallel_state, tmp_path):
    """The base checkpoint relabelled as a PEFT checkpoint holds no adapter tensors: the load must fail, not no-op."""
    import yaml

    from megatron.bridge.training.model_load_save import load_peft_model

    _, peft_iter = _peft_checkpoint(tmp_path)
    base_iter = tmp_path / "base" / "iter_0000000"
    run_config = yaml.safe_load((base_iter / "run_config.yaml").read_text())
    run_config["checkpoint"]["pretrained_checkpoint"] = str(tmp_path / "base")
    run_config["peft"] = yaml.safe_load((peft_iter / "run_config.yaml").read_text())["peft"]
    (base_iter / "run_config.yaml").write_text(yaml.safe_dump(run_config))
    from torch.distributed.checkpoint.api import CheckpointException

    with pytest.raises(CheckpointException, match="adapter_z.linear_in.weight from model not in state dict"):
        load_peft_model(str(base_iter))


def test_merge_peft_checkpoint_saves_a_checked_merge(parallel_state, tmp_path):
    import json

    from megatron.bridge.training.model_load_save import load_megatron_model, merge_peft_checkpoint

    adapted, peft_iter = _peft_checkpoint(tmp_path)
    merged_iter = merge_peft_checkpoint(str(peft_iter), tmp_path / "merged", iteration=7)
    assert merged_iter == tmp_path / "merged" / "iter_0000007"
    check = json.loads((merged_iter / "merge_check.json").read_text())
    assert check["merged"]["mean_abs_diff"] < check["base"]["mean_abs_diff"]
    assert (merged_iter / "peft_run_config.yaml").exists()
    merged = load_megatron_model(str(merged_iter), skip_temp_dist_context=True)
    merged = merged if isinstance(merged, list) else [merged]
    torch.testing.assert_close(_logits(merged), _logits(adapted), rtol=3e-2, atol=3e-2)


def test_a_partial_in_proj_target_is_refused(parallel_state):
    from megatron.bridge.peft.canonical_lora import CanonicalLoRA

    with pytest.raises(ValueError, match="both 'in_proj_z' and 'in_proj_x'"):
        CanonicalLoRA(target_modules=["in_proj_x"], lora_A_init_method="kaiming")(_model(), training=False)

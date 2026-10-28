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
"""Canonical LoRA on a small real Nemotron-H Latent-MoE model: every Super target and its forward."""

import pytest
import torch

from megatron.bridge.recipes.nemotronh.nemotron_3_super import NEMOTRON_3_SUPER_CANONICAL_LORA_TARGETS
from tests.unit_tests.one_rank_nccl_world import one_rank_model_parallel_state
from tests.unit_tests.small_nemotron_h import SEQ, VOCAB, small_nemotron_h_provider


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (Mamba and grouped-GEMM kernels)"),
    pytest.mark.skipif(not hasattr(torch, "_grouped_mm"), reason="needs torch._grouped_mm"),
]


@pytest.fixture(scope="module")
def parallel_state():
    with one_rank_model_parallel_state(seed=1234, expert_model_parallel_size=1):
        yield


def _model():
    """Mamba, attention and Latent-MoE layers (with a shared expert) on the GroupedExperts path, in bf16."""
    torch.manual_seed(0)
    provider = small_nemotron_h_provider("M*E", bf16=True, params_dtype=torch.bfloat16)
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


def test_adapters_change_the_output(parallel_state):
    model = _adapted()
    adapted = _logits(model)
    for chunk in model:
        for module in chunk.modules():
            if hasattr(module, "disable_adapter_layers"):
                module.disable_adapter_layers()
    assert (adapted - _logits(model)).abs().max() > 1e-2


def test_a_partial_in_proj_target_is_refused(parallel_state):
    from megatron.bridge.peft.canonical_lora import CanonicalLoRA

    with pytest.raises(ValueError, match="both 'in_proj_z' and 'in_proj_x'"):
        CanonicalLoRA(target_modules=["in_proj_x"], lora_A_init_method="kaiming")(_model(), training=False)


def test_plain_lora_on_canonical_target_names_is_refused(parallel_state):
    """Plain LoRA given CanonicalLoRA's names raises when applied, naming them as CanonicalLoRA targets."""
    from megatron.bridge.peft.lora import LoRA

    lora = LoRA(target_modules=list(NEMOTRON_3_SUPER_CANONICAL_LORA_TARGETS))
    with pytest.raises(ValueError, match=r"match no module of the model; \[.*'linear_q'.*\] are CanonicalLoRA target"):
        lora(_model(), training=False)


@pytest.mark.parametrize("scheme", ["lora", "canonical_lora", "dora"])
def test_a_target_that_matches_no_module_is_refused(parallel_state, scheme):
    from megatron.bridge.peft.canonical_lora import CanonicalLoRA
    from megatron.bridge.peft.dora import DoRA
    from megatron.bridge.peft.lora import LoRA

    peft = {"lora": LoRA, "canonical_lora": CanonicalLoRA, "dora": DoRA}[scheme](
        target_modules=["linear_proj", "*.layers.9.*.out_proj"]
    )
    with pytest.raises(ValueError, match=r"\['\*\.layers\.9\.\*\.out_proj'\] match no module of the model$"):
        peft(_model(), training=False)

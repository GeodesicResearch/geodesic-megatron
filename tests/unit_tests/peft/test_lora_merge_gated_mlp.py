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
"""LoRAMerge on CanonicalLoRA's split q/k/v and gate/up adapters of a small GPT with a gated MLP."""

import pytest
import torch

from tests.unit_tests.one_rank_nccl_world import one_rank_model_parallel_state
from tests.unit_tests.small_language_models import gpt_model


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (Transformer Engine layers)")

VOCAB, SEQ = 128, 32


@pytest.fixture(scope="module")
def parallel_state():
    with one_rank_model_parallel_state(seed=1234):
        yield


def _logits(model):
    torch.manual_seed(1)
    tokens = torch.randint(0, VOCAB, (1, SEQ), device="cuda")
    position_ids = torch.arange(SEQ, device="cuda").unsqueeze(0)
    with torch.no_grad():
        # Gathering the TP logits is a no-op at TP 1, and Megatron's InferenceMode, if another test left it on, needs it.
        return model(tokens, position_ids, attention_mask=None, runtime_gather_output=True).float()


def test_loramerge_folds_split_qkv_and_gate_up_adapters(parallel_state):
    from megatron.bridge.peft.adapter_wrapper import AdapterWrapper
    from megatron.bridge.peft.canonical_lora import CanonicalLoRA, LoRALinearSplitFC1UpGate, LoRALinearSplitQKV
    from megatron.bridge.peft.lora import LoRAMerge

    torch.manual_seed(0)
    model = gpt_model(
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        pre_process=True,
        num_layers=2,
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=4,
        num_query_groups=2,
        gated_linear_unit=True,
        params_dtype=torch.float32,
    )
    lora = CanonicalLoRA(target_modules=["linear_q", "linear_k", "linear_v", "linear_fc1_up", "linear_fc1_gate"])
    # The helper builds on the CPU and moves the model to the GPU; the adapters are built on the CPU too.
    model = lora(model.cpu(), training=False).cuda()
    wrappers = [m for m in model.modules() if isinstance(m, AdapterWrapper)]
    assert {type(m) for m in wrappers} == {LoRALinearSplitQKV, LoRALinearSplitFC1UpGate}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if ".adapter." in name and "linear_out" in name:
                param.normal_(std=0.1)
    adapted = _logits(model)
    for wrapper in wrappers:
        wrapper.disable_adapter_layers()
    base = _logits(model)

    LoRAMerge()(model, training=False)

    assert (adapted - base).abs().max() > 0.1
    # fp32 weights, but Transformer Engine runs fp32 GEMMs at TF32: 2e-3 against an effect above 0.1.
    torch.testing.assert_close(_logits(model), adapted, rtol=1e-3, atol=2e-3)

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""MoE routers see the packed SFT step's padding mask laid out like their hidden states, and count only the real
tokens.

The MoE router leaves the positions a padding mask marks out of its statistics, the token counts behind the
expert-bias update and the auxiliary losses, reading the mask position for position against its hidden states.

- Under sequence parallelism a layer's hidden states hold the tensor-parallel rank's share of the sequence. The
  training step lays the mask out that way for each pipeline stage (``gpt_step._prepare_packed_padding_mask``, as
  upstream Megatron-Bridge does): a GPT model's first stage scatters the mask itself, and the step scatters it for the
  later stages of a GPT model and every stage of a hybrid model. Two ranks build each model at tensor-parallel size 2,
  as a first pipeline stage and as a later one, prepare the mask as the step does and run the model's preprocessing;
  a later GPTModel stage then runs a training forward and backward through its MoE layer.
- The expert-bias count applies the ``[tokens]`` mask to the ``[tokens, experts]`` routing map across the experts
  (0006 in 3rdparty/patches/megatron-lm/README.md, an upstream fix carried on the pinned Megatron-LM). One GPU runs a
  training forward and backward of a MoE HybridModel, with and without full recompute, which routes the tokens again
  in the backward, and with and without the fused router kernels.

Every model has one MoE layer with Nemotron-H's router settings: expert bias, the sequence auxiliary loss.
"""

import dataclasses

import pytest
import torch

from tests.unit_tests.one_rank_nccl_world import one_rank_model_parallel_state
from tests.unit_tests.small_language_models import gpt_model, hybrid_model


pytestmark = pytest.mark.run_only_on("GPU")

SEQ = 16
BATCH = 2
TP = 2
VOCAB = 128
HIDDEN = 64
EXPERTS = 8
TOPK = 2


def _nemotron_h_routing() -> dict:
    """Nemotron-H's router settings, as the Bridge's provider declares them."""
    from megatron.bridge.models.nemotronh.nemotron_h_provider import NemotronHModelProvider

    defaults = {field.name: field.default for field in dataclasses.fields(NemotronHModelProvider)}
    names = (
        "moe_router_score_function",
        "moe_router_enable_expert_bias",
        "moe_router_load_balancing_type",
        "moe_aux_loss_coeff",
        "moe_router_dtype",
        "moe_token_dispatcher_type",
    )
    return {name: defaults[name] for name in names}


def _one_moe_layer(routing: dict) -> dict:
    return dict(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_moe_experts=EXPERTS,
        moe_router_topk=TOPK,
        moe_ffn_hidden_size=32,
        moe_grouped_gemm=True,
        add_bias_linear=False,
        **routing,
    )


BUILDERS = {
    "GPTModel": lambda pre_process, routing, **fields: gpt_model(
        vocab_size=VOCAB, max_sequence_length=SEQ, pre_process=pre_process, **_one_moe_layer(routing), **fields
    ),
    # is_hybrid_model, as Nemotron-H's provider sets it: the flag the training step reads to lay out the mask.
    "HybridModel": lambda pre_process, routing, **fields: hybrid_model(
        "E",
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        pre_process=pre_process,
        is_hybrid_model=True,
        **_one_moe_layer(routing),
        **fields,
    ),
}


def _padding_mask() -> torch.Tensor:
    """Padding as the packed collate lays it out, inside a row (a document's padding) and at a row's tail; at
    tensor-parallel size 2 each rank's share of the sequence holds padded and real positions."""
    padding_mask = torch.zeros(BATCH, SEQ, dtype=torch.bool)
    padding_mask[0, 5:8] = True
    padding_mask[0, 13:] = True
    padding_mask[1, 10:] = True
    return padding_mask.cuda()


def _tokens() -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ), generator=torch.Generator().manual_seed(0)).cuda()
    position_ids = torch.arange(SEQ).repeat(BATCH, 1).cuda()
    return input_ids, position_ids


def _assert_the_router_counts_only_real_tokens(model, training_step, layer_mask: torch.Tensor, routings: int) -> None:
    """Run ``training_step`` (a forward returning a scalar loss), then its backward, and check the expert-bias counts
    of the model's MoE router against its last routing map: each expert's count is its routings of the tokens
    ``layer_mask`` leaves unmarked, the real tokens times top-k in all.

    Full recompute routes the tokens twice, without gradients in the forward and with them in the backward, and only
    the routing done with gradients counts, so ``routings`` is the number of routings expected."""
    router = model.decoder.layers[0].mlp.router
    routing_maps = []
    router.register_forward_hook(lambda module, args, output: routing_maps.append(output[1].detach()))
    training_step().backward()
    assert len(routing_maps) == routings
    routing_map = routing_maps[-1]
    assert routing_map.shape == (layer_mask.numel(), EXPERTS)
    # The router reads its tokens sequence-major, as the layer's [sequence, batch] hidden states lie.
    real = (~layer_mask).transpose(0, 1).reshape(-1)
    expected = routing_map[real].sum(dim=0)
    assert int(expected.sum()) == int(real.sum()) * TOPK
    assert torch.equal(router.local_tokens_per_expert, expected.to(router.local_tokens_per_expert.dtype))


def _layers_get_their_share_of_the_mask(rank: int, init_file: str, routing: dict) -> None:
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    from megatron.bridge.training.gpt_step import _prepare_packed_padding_mask
    from megatron.bridge.training.utils.pg_utils import get_pg_collection

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group("nccl", init_method=f"file://{init_file}", rank=rank, world_size=TP)
    try:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=TP)
        model_parallel_cuda_manual_seed(1234)
        padding_mask = _padding_mask()
        share = padding_mask[:, rank * SEQ // TP : (rank + 1) * SEQ // TP]
        input_ids, position_ids = _tokens()

        def stage_mask(model) -> torch.Tensor:
            """The mask the training step hands this stage of the model."""
            return _prepare_packed_padding_mask(
                padding_mask, config=model.config, model=model, pg_collection=get_pg_collection(model)
            )

        later_stages = {}
        for name, build in BUILDERS.items():
            for pre_process in (True, False):
                model = build(pre_process, routing, tensor_model_parallel_size=TP, sequence_parallel=True)
                # A later pipeline stage receives no tokens: its hidden states arrive from the previous stage.
                inputs = (
                    dict(input_ids=input_ids, position_ids=position_ids)
                    if pre_process
                    else dict(input_ids=None, position_ids=None)
                )
                layer_mask = model._preprocess(**inputs, padding_mask=stage_mask(model))[5]
                assert torch.equal(layer_mask, share), (
                    f"{name}, pre_process={pre_process}: rank {rank} got {tuple(layer_mask.shape)}, expected its "
                    f"{tuple(share.shape)} share of the sequence"
                )
                if not pre_process:
                    later_stages[name] = model

        later_stage = later_stages["GPTModel"]
        later_stage.set_input_tensor(torch.randn(SEQ // TP, BATCH, HIDDEN, device="cuda", requires_grad=True))
        _assert_the_router_counts_only_real_tokens(
            later_stage,
            lambda: later_stage(None, None, None, padding_mask=stage_mask(later_stage)).float().sum(),
            share,
            routings=1,
        )
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def test_each_tensor_parallel_rank_routes_with_its_share_of_the_padding_mask(tmp_path):
    if torch.cuda.device_count() < TP:
        pytest.skip(f"needs {TP} GPUs")
    torch.multiprocessing.spawn(
        _layers_get_their_share_of_the_mask, args=(str(tmp_path / "rendezvous"), _nemotron_h_routing()), nprocs=TP
    )


@pytest.fixture
def one_rank_world():
    """Real world-1 Megatron parallel state, expert groups included."""
    with one_rank_model_parallel_state(seed=1234, expert_model_parallel_size=1):
        yield


@pytest.mark.parametrize("router_fusion", [False, True], ids=["unfused-router", "fused-router"])
@pytest.mark.parametrize("recompute", [False, True], ids=["no-recompute", "full-recompute"])
def test_the_expert_bias_update_counts_only_the_real_tokens(one_rank_world, recompute, router_fusion):
    recompute_fields = (
        dict(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1) if recompute else {}
    )
    model = BUILDERS["HybridModel"](True, _nemotron_h_routing(), moe_router_fusion=router_fusion, **recompute_fields)
    padding_mask = _padding_mask()
    input_ids, position_ids = _tokens()

    def training_step() -> torch.Tensor:
        loss = model(input_ids, position_ids, None, labels=input_ids, padding_mask=padding_mask)
        return (loss * ~padding_mask).sum()

    _assert_the_router_counts_only_real_tokens(model, training_step, padding_mask, routings=2 if recompute else 1)

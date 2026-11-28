# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""A TE grouped linear layer that ran under MXFP8 builds its sharded state dict once the patch is installed.

The layer is Megatron-Core's real ``TEColumnParallelGroupedLinear`` on a one-rank NCCL world, run forward under TE's
MXFP8 block-scaling recipe as ``te_grouped`` experts run in training. MXFP8 needs a Blackwell GPU.
"""

from __future__ import annotations

import pytest
import torch

from pipeline.training.patches.grouped_extra_state import patch_grouped_extra_state_split
from tests.unit_tests.one_rank_nccl_world import one_rank_model_parallel_state


EXPERTS, HIDDEN, FFN, TOKENS_PER_EXPERT = 4, 128, 256, 32

requires_mxfp8 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="MXFP8 needs a Blackwell GPU",
)


@pytest.fixture(scope="module")
def pg_collection():
    from megatron.core.process_groups_config import ProcessGroupCollection

    with one_rank_model_parallel_state(seed=1234, expert_model_parallel_size=1):
        yield ProcessGroupCollection.use_mpu_process_groups()


def _layer_after_an_mxfp8_forward(pg_collection):
    import transformer_engine.pytorch as te
    from megatron.core.extensions.transformer_engine import TEColumnParallelGroupedLinear
    from megatron.core.transformer.transformer_config import TransformerConfig
    from transformer_engine.common.recipe import MXFP8BlockScaling

    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_moe_experts=EXPERTS,
        moe_ffn_hidden_size=FFN,
        add_bias_linear=False,
        params_dtype=torch.bfloat16,
        bf16=True,
        fp8="e4m3",
        fp8_recipe="mxfp8",
    )
    layer = TEColumnParallelGroupedLinear(
        EXPERTS,
        HIDDEN,
        FFN,
        config=config,
        init_method=torch.nn.init.normal_,
        bias=False,
        skip_bias_add=False,
        is_expert=True,
        pg_collection=pg_collection,
    ).cuda()
    tokens = torch.randn(EXPERTS * TOKENS_PER_EXPERT, HIDDEN, device="cuda", dtype=torch.bfloat16)
    with te.fp8_autocast(enabled=True, fp8_recipe=MXFP8BlockScaling()):
        layer(tokens, [TOKENS_PER_EXPERT] * EXPERTS)
    return layer


@requires_mxfp8
def test_an_mxfp8_grouped_layer_builds_its_sharded_state_dict_with_an_extra_state_per_expert(
    pg_collection, unpatched_grouped_extra_state
):
    """The Transformer Engine of NeMo 26.10.00 leaves the extra state empty. The older one of 26.04 fills it."""
    from megatron.core import parallel_state

    patch_grouped_extra_state_split()
    layer = _layer_after_an_mxfp8_forward(pg_collection)
    metadata = {"dp_cp_group": parallel_state.get_data_parallel_group(with_context_parallel=True)}
    sharded = layer.sharded_state_dict(prefix="fc1.", metadata=metadata)
    extra_states = {"fc1._extra_state", *(f"fc1._extra_state{gemm}" for gemm in range(1, EXPERTS))}
    assert set(sharded) == {f"fc1.weight{gemm}" for gemm in range(EXPERTS)} | extra_states


def test_the_patch_installs_once(unpatched_grouped_extra_state):
    assert patch_grouped_extra_state_split() is True
    assert unpatched_grouped_extra_state._split_extra_state.empty_state_split
    assert patch_grouped_extra_state_split() is False

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

"""The EP-overlap branch of ``_forward_step_common`` (``return_schedule_plan=True``).

The step passes a packed batch's packed params and padding mask to ``model.build_schedule_plan``. The
model gets the same two without the plan. Attention and the Mamba scan then stop at document
boundaries under the overlap too.

The step's real ``get_batch``, ``get_model_config`` and ``get_pg_collection`` run on CPU. They use a
single-process gloo group, as in ``test_gpt_step_cp_dispatch.py``. The batch is the one ``get_batch``
produces from a collate-style packed batch. The step's ``GlobalState`` is real too, with its timers
and straggler detector. The tests stand in for two boundaries. One is the model, which needs GPUs: the
stand-in carries its config and process groups and records how it is called. The other is
``Tensor.cuda``, because ``get_batch`` moves the batch to a device that CPU-only tiers lack.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from megatron.core.transformer.enums import CudaGraphModule

from megatron.bridge.training import gpt_step
from megatron.bridge.training.state import GlobalState
from tests.unit_tests.token_masking_fixtures import no_token_masking
from tests.unit_tests.training.test_gpt_step_cp_dispatch import _cfg, _unpacked_batch
from tests.unit_tests.training.test_gpt_step_packed_all_stages import _make_packed_batch


SEQ_LENGTH = 8
# Two documents, of 5 and 3 tokens.
DOCUMENT_BOUNDARIES = [0, 5, SEQ_LENGTH]


def _model(group) -> MagicMock:
    """The stand-in model: the attributes the step's real config and process-group lookups read."""
    model = MagicMock()
    model.config = SimpleNamespace(overlap_moe_expert_parallel_comm=True, mtp_num_layers=None)
    model.pg_collection = SimpleNamespace(pp=group, cp=group)
    return model


def _run(model: MagicMock, packed: bool, return_schedule_plan: bool):
    cfg = _cfg(packed=packed, hybrid_cp=False)
    cfg.logger.timing_log_level = 0
    cfg.logger.timing_log_option = "minmax"
    state = GlobalState()
    state.cfg = cfg
    state.token_masking = no_token_masking()
    batch = _make_packed_batch(SEQ_LENGTH, DOCUMENT_BOUNDARIES) if packed else _unpacked_batch(SEQ_LENGTH)
    with patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self):
        return gpt_step._forward_step_common(state, iter([batch]), model, return_schedule_plan=return_schedule_plan)


def test_packed_batch_builds_the_plan_with_what_the_model_gets_without_it(gloo_group_of_one):
    """The plan's layers read the document boundaries and the padded positions from these two arguments."""
    model = _model(gloo_group_of_one)
    schedule_plan, _, _ = _run(model, packed=True, return_schedule_plan=True)
    assert schedule_plan is model.build_schedule_plan.return_value
    plan_kwargs = model.build_schedule_plan.call_args.kwargs

    model = _model(gloo_group_of_one)
    _run(model, packed=True, return_schedule_plan=False)
    model_kwargs = model.call_args.kwargs
    assert torch.equal(plan_kwargs["padding_mask"], model_kwargs["padding_mask"])
    for field in ("cu_seqlens_q", "cu_seqlens_kv", "seq_idx"):
        planned, direct = (getattr(kwargs["packed_seq_params"], field) for kwargs in (plan_kwargs, model_kwargs))
        assert torch.equal(planned, direct), field


def test_unpacked_batch_builds_the_plan(gloo_group_of_one):
    model = _model(gloo_group_of_one)
    schedule_plan, _, _ = _run(model, packed=False, return_schedule_plan=True)
    model.build_schedule_plan.assert_called_once()
    assert schedule_plan is model.build_schedule_plan.return_value


def test_packed_batch_without_the_plan_reaches_the_model_with_its_packed_params(gloo_group_of_one):
    model = _model(gloo_group_of_one)
    output, _, _ = _run(model, packed=True, return_schedule_plan=False)
    model.build_schedule_plan.assert_not_called()
    assert output is model.return_value
    assert "packed_seq_params" in model.call_args.kwargs


def test_packed_batch_reaches_the_model_with_its_padding_mask(gloo_group_of_one):
    """The MoE routers leave the positions the collate padded out of their statistics only if the model is given
    the mask; a batch without one (an unpacked batch here) passes none."""
    model = _model(gloo_group_of_one)
    _run(model, packed=True, return_schedule_plan=False)
    expected = _make_packed_batch(SEQ_LENGTH, DOCUMENT_BOUNDARIES)["padding_mask"]
    assert torch.equal(model.call_args.kwargs["padding_mask"], expected)

    model = _model(gloo_group_of_one)
    _run(model, packed=False, return_schedule_plan=False)
    assert "padding_mask" not in model.call_args.kwargs


@pytest.mark.parametrize(
    "cuda_graph_impl,cuda_graph_modules,refused",
    [
        ("transformer_engine", [], True),
        ("transformer_engine", ["moe"], True),
        ("transformer_engine", ["moe_router", "moe_preprocess"], True),
        ("transformer_engine", ["attn"], False),
        ("local", [], False),
        ("full_iteration", [], False),
    ],
)
def test_a_moe_model_refuses_the_padding_mask_under_router_capturing_cuda_graphs(
    gloo_group_of_one, cuda_graph_impl, cuda_graph_modules, refused
):
    """As upstream Megatron-Bridge does: Transformer Engine CUDA graphs that capture the router (the whole layer, or a
    MoE, router or MoE-preprocess module) refuse the mask before the model runs; other scopes and implementations
    run."""
    model = _model(gloo_group_of_one)
    model.config.num_moe_experts = 8
    model.config.cuda_graph_impl = cuda_graph_impl
    model.config.cuda_graph_modules = [CudaGraphModule[name] for name in cuda_graph_modules]
    if refused:
        with pytest.raises(ValueError, match="capture the router"):
            _run(model, packed=True, return_schedule_plan=False)
        model.assert_not_called()
    else:
        _run(model, packed=True, return_schedule_plan=False)
        model.assert_called_once()

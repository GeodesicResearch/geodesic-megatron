# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Multi-token prediction must see the same loss mask as the main next-token loss.

``GPTModel.forward`` and ``HybridModel.forward`` hand their ``loss_mask`` argument to
``process_mtp_loss``, which substitutes an all-ones mask when it is ``None``. The step therefore
has to pass the batch's loss mask to the model whenever MTP layers exist; otherwise the MTP
heads train on every position the dataset (prompt tokens, padding) or token-level masking
excluded from the loss.

The step's real ``get_batch`` and process-group lookups run on CPU over a single-process gloo
group, as in ``test_gpt_step_schedule_plan.py``. The model is a stand-in that records how it is
called (a real one needs GPUs), and ``Tensor.cuda`` is the identity because CPU-only tiers have no
device to move the batch to.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from megatron.bridge.training import gpt_step
from megatron.bridge.training.state import GlobalState
from tests.unit_tests.token_masking_fixtures import no_token_masking
from tests.unit_tests.training.test_gpt_step_cp_dispatch import _cfg, _unpacked_batch


SEQ_LENGTH = 8


def _model(group, mtp_num_layers: int | None) -> MagicMock:
    model = MagicMock()
    model.config = SimpleNamespace(overlap_moe_expert_parallel_comm=False, mtp_num_layers=mtp_num_layers)
    model.pg_collection = SimpleNamespace(pp=group, cp=group)
    return model


def _run(model: MagicMock):
    cfg = _cfg(packed=False, hybrid_cp=False)
    cfg.logger.timing_log_level = 0
    cfg.logger.timing_log_option = "minmax"
    state = GlobalState()
    state.cfg = cfg
    state.token_masking = no_token_masking()
    batch = _unpacked_batch(SEQ_LENGTH)
    batch["loss_mask"][0, :3] = 0
    with patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self):
        return gpt_step._forward_step_common(state, iter([batch]), model)


def test_mtp_model_receives_the_loss_mask_the_loss_uses(gloo_group_of_one):
    model = _model(gloo_group_of_one, mtp_num_layers=1)
    _, loss_mask, _ = _run(model)
    assert "loss_mask" in model.call_args.kwargs
    assert model.call_args.kwargs["loss_mask"] is loss_mask
    assert loss_mask[0, :3].sum() == 0


def test_model_without_mtp_is_called_without_a_loss_mask(gloo_group_of_one):
    model = _model(gloo_group_of_one, mtp_num_layers=None)
    _run(model)
    assert "loss_mask" not in model.call_args.kwargs

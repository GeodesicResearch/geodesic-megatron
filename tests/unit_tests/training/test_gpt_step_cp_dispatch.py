# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""CPU-only tests for ``gpt_step.get_batch``'s context-parallel dispatch.

``get_batch`` routes the batch two ways: packed data at CP>1 goes to the bridge's
own ``_partition_packed_batch_for_cp``, and EVERYTHING ELSE — every CP=1 run, and
unpacked data at any CP — goes to Megatron-Core's ``get_batch_on_this_cp_rank``.

That second branch is the one these tests cover, because it silently broke. The
mcore 0.19 pin added a REQUIRED ``is_hybrid_cp`` positional to
``get_batch_on_this_cp_rank``; the bridge's call site still passed the pre-0.19
two-argument form, so every CP=1 configuration — and every non-packed configuration
at any CP — died with ``TypeError: ... missing 1 required positional argument:
'is_hybrid_cp'`` at the first microbatch. Among the shipped quickstarts that is the
Ultra-550B (CP=1); the Nano and Super quickstarts (packed + CP>1) take the other
branch and kept working, which is why it reached main unnoticed.

The existing suite could not catch it: ``test_gpt_step_packed_all_stages.py``
patches ``get_batch_on_this_cp_rank`` with a permissive ``*args, **kwargs``
stand-in, so any signature at all satisfies it. These tests therefore call the
REAL mcore function. That is affordable on CPU because both of its balancers are
no-ops below CP=2 (``if cp_size > 1:``), needing only a live process group for
``get_world_size`` — so a single-process gloo group covers the whole dispatch.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.bridge.training import gpt_step
from tests.unit_tests.training.test_gpt_step_packed_all_stages import _make_cfg, _make_packed_batch


@pytest.fixture(scope="module")
def cp_group_of_one():
    """A real single-process gloo group — the only piece of live distributed state
    the CP dispatch needs at CP=1 (``torch.distributed.get_world_size``)."""
    created_here = False
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
        created_here = True
    yield torch.distributed.group.WORLD
    if created_here:
        torch.distributed.destroy_process_group()


def _unpacked_batch(full_len: int) -> dict:
    """A non-packed batch: cu_seqlens absent, so mcore takes per-sequence balancing."""
    return {
        "tokens": torch.arange(full_len, dtype=torch.long).unsqueeze(0),
        "labels": torch.arange(full_len, dtype=torch.long).unsqueeze(0),
        "loss_mask": torch.ones(1, full_len, dtype=torch.long),
        "position_ids": torch.arange(full_len, dtype=torch.long).unsqueeze(0),
    }


def _cfg(*, packed: bool, hybrid_cp: bool):
    """The sibling module's cfg builder, plus the one field this module's tests turn on.

    ``_make_cfg`` already encodes the packed-sequence predicate that ``get_batch`` branches
    on; duplicating it here would mean a change to the collate contract silently fixing one
    test file and not the other.
    """
    cfg = _make_cfg(packed)
    # Explicit, because a MagicMock attribute is truthy: leaving this auto-specced would
    # send every test down the hybrid-CP branch and hide what is being asserted.
    cfg.model.hybrid_context_parallel = hybrid_cp
    return cfg


def _run_get_batch(batch, cfg, cp_group):
    """Drive the real get_batch on a single-stage rank, with mcore's own helpers UNPATCHED.

    The pipeline group is the same world-1 gloo group as the CP group, so the REAL
    ``is_pp_first_stage``/``is_pp_last_stage`` return True/True on their own — they are
    one-liners over ``get_pg_rank``/``get_pg_size``. Patching them would reintroduce exactly
    the stand-in-accepts-anything weakness this module exists to close.

    ``Tensor.cuda`` is the one genuine boundary: these tests run on CPU-only CI tiers where
    no device exists, and ``get_batch`` unconditionally moves the batch to the GPU.
    """
    pg_collection = SimpleNamespace(pp=cp_group, cp=cp_group)
    with patch.object(torch.Tensor, "cuda", lambda self, *a, **k: self):
        return gpt_step.get_batch(iter([batch]), cfg, use_mtp=False, pg_collection=pg_collection)


class TestContextParallelDispatch:
    """The CP=1 branch must call mcore's slicer with a signature mcore accepts."""

    def test_cp1_packed_reaches_real_mcore_slicer(self, cp_group_of_one):
        """THE REGRESSION: CP=1 + packed (the Ultra quickstart posture) must not TypeError.

        Before the fix this raised ``TypeError: get_batch_on_this_cp_rank() missing 1
        required positional argument: 'is_hybrid_cp'`` at the first microbatch of every
        CP=1 run.
        """
        full_len = 4096
        result = _run_get_batch(
            _make_packed_batch(full_len, [0, 1000, 2500, 4096]), _cfg(packed=True, hybrid_cp=False), cp_group_of_one
        )

        tokens, labels, loss_mask = result[0], result[1], result[2]
        # CP=1 is a no-op slice: the batch survives whole.
        assert tokens is not None and tokens.shape[-1] == full_len
        assert labels is not None and labels.shape[-1] == full_len
        assert loss_mask is not None and loss_mask.shape[-1] == full_len
        assert result[5] is not None, "cu_seqlens must survive the CP=1 dispatch"

    def test_cp1_unpacked_reaches_real_mcore_slicer(self, cp_group_of_one):
        """The other route into the same branch: unpacked data (per-sequence balancing)."""
        full_len = 4096
        result = _run_get_batch(_unpacked_batch(full_len), _cfg(packed=False, hybrid_cp=False), cp_group_of_one)

        assert result[0] is not None and result[0].shape[-1] == full_len
        assert result[5] is None, "unpacked batches carry no cu_seqlens"

    def test_is_hybrid_cp_is_read_from_config_not_hardcoded(self, cp_group_of_one):
        """``is_hybrid_cp`` must carry the model's configured value through to mcore.

        Asserted by observation rather than by re-stating the call: with hybrid CP on,
        mcore takes its hybrid branch, which requires a ``local_cp_size`` entry our batch
        does not have. Reaching that failure proves the True propagated; a hardcoded
        False would sail past. Paired with the two tests above (which pass under False),
        this pins the value as config-driven in both directions.
        """
        batch = _make_packed_batch(4096, [0, 4096])
        with pytest.raises((AssertionError, KeyError)):
            _run_get_batch(batch, _cfg(packed=True, hybrid_cp=True), cp_group_of_one)

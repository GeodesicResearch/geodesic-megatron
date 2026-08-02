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

"""CPU-only tests that ``gpt_step``'s CP-slice call matches the pinned Megatron-Core.

Megatron-Core 0.19 made ``is_hybrid_cp`` a **required positional** parameter of
``megatron.core.utils.get_batch_on_this_cp_rank``. The bridge call site in
``gpt_step.get_batch`` still passed the pre-0.19 arity, so every run reaching it
died on the first training iteration with::

    TypeError: get_batch_on_this_cp_rank() missing 1 required positional argument: 'is_hybrid_cp'

``get_batch`` routes to that call whenever **not** (packed and ``cp_size > 1``) --
i.e. on every ``context_parallel_size: 1`` run, packed or not. The benchmark
quickstart runs CP=4 with packed data and therefore takes the
``_partition_packed_batch_for_cp`` branch instead, which is why the version bump
validated clean while CP=1 configs crashed 100% of the time.

``test_gpt_step_packed_all_stages.py`` exercises the same function but patches
``gpt_step.get_batch_on_this_cp_rank`` with an identity stub, so it is blind to
signature drift by construction. These tests deliberately leave the real
Megatron-Core function in place -- that is the entire point: they fail if the call
site and the pinned Megatron-Core ever disagree again.

Only ``torch.distributed.get_world_size``/``get_rank`` are patched. Those are a
genuinely untestable boundary (they require an initialized process group, which
needs a real distributed rendezvous), and both Megatron-Core CP helpers
short-circuit at ``cp_size == 1`` and return the batch unchanged -- so the real
function body runs end-to-end on CPU with no GPU and no distributed init.
"""

import inspect
from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.bridge.training import gpt_step


def _make_cfg(packed: bool):
    """Minimal cfg carrying the packed-sequence predicate and attn-mask flag."""
    cfg = MagicMock()
    cfg.dataset.skip_getting_attention_mask_from_dataset = True
    cfg.dataset.pack_sequences_in_batch = False
    cfg.dataset.packed_sequence_specs = MagicMock() if packed else None
    if packed:
        cfg.dataset.packed_sequence_specs.packed_sequence_size = 4096
    return cfg


def _make_pg_collection(cp_size: int):
    """pg_collection whose CP group reports ``cp_size`` ranks."""
    pg = MagicMock()
    pg.cp.size.return_value = cp_size
    return pg


def _make_batch(full_len: int, *, packed: bool) -> dict:
    """Collate-style batch; ``packed`` adds the cu_seqlens document metadata."""
    batch = {
        "tokens": torch.arange(full_len, dtype=torch.long).unsqueeze(0),
        "labels": torch.arange(full_len, dtype=torch.long).unsqueeze(0),
        "loss_mask": torch.ones(1, full_len, dtype=torch.long),
        "position_ids": torch.arange(full_len, dtype=torch.long).unsqueeze(0),
        "token_count": torch.tensor([full_len]),
    }
    if packed:
        boundaries = [0, full_len // 2, full_len]
        cu_padded = boundaries + [-1, -1]
        batch["cu_seqlens"] = torch.tensor([cu_padded], dtype=torch.int32)
        batch["cu_seqlens_argmin"] = torch.tensor([[len(boundaries)]], dtype=torch.int32)
        batch["max_seqlen"] = torch.tensor([[full_len // 2]], dtype=torch.int32)
    return batch


def _run_get_batch(batch: dict, *, packed: bool, cp_size: int):
    """Drive ``gpt_step.get_batch`` on a single-stage rank with the REAL CP helper.

    ``get_batch_on_this_cp_rank`` is intentionally NOT patched.
    """
    data_iterator = iter([batch])
    with (
        patch.object(gpt_step, "is_pp_first_stage", return_value=True),
        patch.object(gpt_step, "is_pp_last_stage", return_value=True),
        patch.object(torch.Tensor, "cuda", lambda self, *a, **k: self),
        # Distributed boundary: a real process group would need a rendezvous.
        patch.object(torch.distributed, "get_world_size", return_value=cp_size),
        patch.object(torch.distributed, "get_rank", return_value=0),
    ):
        return gpt_step.get_batch(
            data_iterator,
            _make_cfg(packed),
            use_mtp=False,
            pg_collection=_make_pg_collection(cp_size),
        )


class TestCpSliceCallSignature:
    """The CP-slice call must stay arity-compatible with the pinned Megatron-Core."""

    @pytest.mark.parametrize("packed", [True, False], ids=["packed", "non_packed"])
    def test_cp1_reaches_real_megatron_helper_without_typeerror(self, packed: bool):
        """CP=1 takes the ``get_batch_on_this_cp_rank`` branch; it must not TypeError.

        ``packed=True`` is the exact shape of the Misalignment-Quarantine EM runs
        that failed: packed data with ``context_parallel_size: 1``.
        """
        full_len = 4096
        result = _run_get_batch(_make_batch(full_len, packed=packed), packed=packed, cp_size=1)

        tokens, labels = result[0], result[1]
        assert tokens is not None
        assert labels is not None
        # cp_size == 1 partitions nothing, so the sequence survives whole.
        assert tokens.size(-1) == full_len
        assert labels.size(-1) == full_len

    def test_call_site_binds_against_installed_signature(self):
        """The kwargs ``get_batch`` passes must bind to the installed signature.

        Guards the arity contract directly, so the failure names the drifted
        parameter instead of surfacing as a TypeError deep in a training run.
        """
        from megatron.core.utils import get_batch_on_this_cp_rank

        signature = inspect.signature(get_batch_on_this_cp_rank)
        # Mirrors the call in gpt_step.get_batch.
        signature.bind({"cu_seqlens": None}, is_hybrid_cp=False, cp_group=None)

        # is_hybrid_cp is required: omitting it is exactly the shipped regression.
        with pytest.raises(TypeError):
            signature.bind({"cu_seqlens": None}, cp_group=None)

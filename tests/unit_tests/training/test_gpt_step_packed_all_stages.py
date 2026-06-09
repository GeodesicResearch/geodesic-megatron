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

"""CPU-only tests for packed-sequence metadata on ALL pipeline stages (GEOD-147).

Hybrid Mamba2+attention models (Nemotron-3 Super) split layers across pipeline
stages, so MIDDLE pipeline stages also own Mamba layers. Those layers need
``seq_idx`` (built from ``cu_seqlens`` + the full pack length) to reset SSM state
at packed-document boundaries; without it the scan integrates across documents
and overflows BF16 at long sequence lengths (forward-loss NaN at seq 32768).

``packed_seq_params`` is a per-stage forward-step argument, built in
``gpt_step.get_batch``/``_forward_step_common`` only where ``cu_seqlens`` is
present. Previously ``get_batch`` early-returned all-None on middle stages, so
they built no packed_seq_params and ran Mamba unpacked. This test verifies the
fix: for packed datasets, middle stages now load the packed metadata (advancing
the iterator in lockstep with first/last) and report the full pre-CP-slice pack
length, while non-packed runs keep the original early-return (no iterator
consumption on middle stages).

The pipeline-stage role and the CP-slice helpers are patched so the test is fully
hermetic on CPU (no distributed init, no GPU). ``Tensor.cuda`` is patched to an
identity so the device-transfer path runs without a GPU.
"""

from unittest.mock import MagicMock, patch

import torch

from megatron.bridge.training import gpt_step


def _make_packed_batch(full_len: int, doc_boundaries: list[int]) -> dict:
    """Build a collate-style packed batch.

    tokens/labels/loss_mask/position_ids are full-length [1, full_len] (identical
    on every rank, since the sampler shards by DP not PP). cu_seqlens carries the
    document boundaries with -1 padding sentinels, exactly like the SFT collate.
    """
    cu = list(doc_boundaries)
    # pad cu_seqlens with two -1 sentinels (argmin points at the first -1)
    cu_padded = cu + [-1, -1]
    argmin = len(cu)
    seqlens = [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]
    max_seqlen = max(seqlens)
    return {
        "tokens": torch.arange(full_len, dtype=torch.long).unsqueeze(0),
        "labels": torch.arange(full_len, dtype=torch.long).unsqueeze(0),
        "loss_mask": torch.ones(1, full_len, dtype=torch.long),
        "position_ids": torch.arange(full_len, dtype=torch.long).unsqueeze(0),
        "cu_seqlens": torch.tensor([cu_padded], dtype=torch.int32),
        "cu_seqlens_argmin": torch.tensor([[argmin]], dtype=torch.int32),
        "max_seqlen": torch.tensor([[max_seqlen]], dtype=torch.int32),
        "token_count": torch.tensor([full_len]),
    }


def _make_cfg(packed: bool):
    """Minimal cfg with the packed-sequence predicate and attn-mask flag."""
    cfg = MagicMock()
    cfg.dataset.skip_getting_attention_mask_from_dataset = True
    if packed:
        cfg.dataset.packed_sequence_specs.packed_sequence_size = 4096
        cfg.dataset.pack_sequences_in_batch = False
    else:
        cfg.dataset.packed_sequence_specs = None
        cfg.dataset.pack_sequences_in_batch = False
    return cfg


def _make_pg_collection(cp_size: int):
    pg = MagicMock()
    pg.cp.size.return_value = cp_size
    return pg


def _identity_cp(batch, *args, **kwargs):
    """Pass-through stand-in for the CP-slice helpers (cp_size==1 == no-op)."""
    return batch


# Index of full_seq_length in the get_batch return tuple.
_FULL_SEQ_IDX = 10
# Index of cu_seqlens in the get_batch return tuple.
_CU_SEQLENS_IDX = 5


class TestGetBatchPackedAllStages:
    """get_batch must surface packed metadata + full pack length on every PP stage."""

    def _run_get_batch(self, *, is_first, is_last, packed, batch, cp_size=1):
        data_iterator = iter([batch])
        cfg = _make_cfg(packed)
        pg_collection = _make_pg_collection(cp_size)
        with (
            patch.object(gpt_step, "is_pp_first_stage", return_value=is_first),
            patch.object(gpt_step, "is_pp_last_stage", return_value=is_last),
            patch.object(gpt_step, "_partition_packed_batch_for_cp", side_effect=_identity_cp),
            patch.object(gpt_step, "get_batch_on_this_cp_rank", side_effect=_identity_cp),
            patch.object(torch.Tensor, "cuda", lambda self, *a, **k: self),
        ):
            result = gpt_step.get_batch(data_iterator, cfg, use_mtp=False, pg_collection=pg_collection)
        # remaining = items the iterator did NOT yield (to detect consumption)
        remaining = sum(1 for _ in data_iterator)
        return result, remaining

    def test_middle_stage_packed_loads_cu_seqlens_and_full_len(self):
        """The regression target: middle PP stage + packed -> cu_seqlens & full_len present."""
        full_len = 4096
        batch = _make_packed_batch(full_len, [0, 1000, 2500, 4096])
        result, remaining = self._run_get_batch(is_first=False, is_last=False, packed=True, batch=batch)

        # Iterator WAS consumed (lockstep with first/last stages).
        assert remaining == 0
        # cu_seqlens is present (global, un-sliced) so packed_seq_params is buildable.
        cu_seqlens = result[_CU_SEQLENS_IDX]
        assert cu_seqlens is not None
        # full_seq_length equals the full pack length, available despite no per-token tensors.
        assert result[_FULL_SEQ_IDX] == full_len
        # Per-token tensors are dropped on the middle stage (hidden states arrive via PP).
        tokens, labels, loss_mask, _, position_ids = result[0], result[1], result[2], result[3], result[4]
        assert tokens is None
        assert labels is None
        assert loss_mask is None
        assert position_ids is None

    def test_middle_stage_non_packed_early_returns_without_consuming(self):
        """Non-packed middle stage keeps the original behaviour: all-None, iterator untouched."""
        full_len = 4096
        batch = _make_packed_batch(full_len, [0, 4096])  # batch is ignored on this path
        result, remaining = self._run_get_batch(is_first=False, is_last=False, packed=False, batch=batch)

        # Iterator NOT consumed -> no desync / no extra dataloader work on middle stages.
        assert remaining == 1
        assert result == (None,) * 11
        assert result[_FULL_SEQ_IDX] is None

    def test_first_stage_packed_full_len_from_tokens(self):
        """First stage carries tokens; full_seq_length comes from tokens.size(-1)."""
        full_len = 4096
        batch = _make_packed_batch(full_len, [0, 1500, 4096])
        result, remaining = self._run_get_batch(is_first=True, is_last=False, packed=True, batch=batch)

        assert remaining == 0
        assert result[_FULL_SEQ_IDX] == full_len
        assert result[_CU_SEQLENS_IDX] is not None
        assert result[0] is not None  # tokens present on first stage

    def test_last_stage_packed_full_len_from_labels(self):
        """Last stage drops tokens but keeps labels; full_seq_length still resolves."""
        full_len = 4096
        batch = _make_packed_batch(full_len, [0, 2048, 4096])
        result, remaining = self._run_get_batch(is_first=False, is_last=True, packed=True, batch=batch)

        assert remaining == 0
        assert result[_FULL_SEQ_IDX] == full_len
        assert result[1] is not None  # labels present on last stage

    def test_all_stages_agree_on_cu_seqlens_and_seq_idx(self):
        """Every PP stage must yield identical cu_seqlens + full_len, hence identical seq_idx.

        This is the core guarantee of the fix: feeding the SAME batch to first,
        middle, and last stages produces byte-identical packed metadata, so the
        seq_idx that each stage's Mamba layers consume is the same global,
        full-length per-document index. We build packed_seq_params via the real
        bridge path on each stage's output and compare the resulting seq_idx.
        """
        from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_params

        full_len = 4096
        boundaries = [0, 1000, 2500, 4096]

        def _meta_and_seq_idx(is_first, is_last):
            # Fresh batch per stage (the iterator/collate yields an equal copy on each rank).
            batch = _make_packed_batch(full_len, boundaries)
            result, _ = self._run_get_batch(is_first=is_first, is_last=is_last, packed=True, batch=batch)
            cu_seqlens = result[_CU_SEQLENS_IDX]
            full = result[_FULL_SEQ_IDX]
            psp = get_packed_seq_params(
                {"cu_seqlens": cu_seqlens, "cu_seqlens_argmin": result[6], "max_seqlen": result[7]},
                total_tokens=full,
            )
            return cu_seqlens, full, psp.seq_idx

        first_cu, first_full, first_seq_idx = _meta_and_seq_idx(True, False)
        mid_cu, mid_full, mid_seq_idx = _meta_and_seq_idx(False, False)
        last_cu, last_full, last_seq_idx = _meta_and_seq_idx(False, True)

        # Identical full pack length on all stages.
        assert first_full == mid_full == last_full == full_len
        # Identical (global, un-sliced) cu_seqlens on all stages.
        torch.testing.assert_close(first_cu, mid_cu)
        torch.testing.assert_close(first_cu, last_cu)
        # Identical seq_idx -> the Mamba layers on every stage reset state the same way.
        assert mid_seq_idx is not None
        assert mid_seq_idx.shape == (1, full_len)
        torch.testing.assert_close(first_seq_idx, mid_seq_idx)
        torch.testing.assert_close(first_seq_idx, last_seq_idx)

    def test_non_packed_first_stage_full_len_none(self):
        """Non-packed batches never set full_seq_length (no cu_seqlens key)."""
        full_len = 1024
        batch = {
            "tokens": torch.arange(full_len, dtype=torch.long).unsqueeze(0),
            "labels": torch.arange(full_len, dtype=torch.long).unsqueeze(0),
            "loss_mask": torch.ones(1, full_len, dtype=torch.long),
            "position_ids": torch.arange(full_len, dtype=torch.long).unsqueeze(0),
        }
        result, remaining = self._run_get_batch(is_first=True, is_last=False, packed=False, batch=batch)

        assert remaining == 0
        assert result[_CU_SEQLENS_IDX] is None
        assert result[_FULL_SEQ_IDX] is None

    def test_full_len_is_pre_cp_slice_length(self):
        """full_seq_length is read BEFORE CP slicing, so it is the full pack length.

        We stub the CP partition to actually halve the per-token sequence dim
        (simulating cp_size=2). full_seq_length must remain the full length, not the
        post-slice per-rank length -- this is the value PackedSeqParams needs so
        seq_idx covers the whole sequence the Mamba scan sees after the CP all-to-all.
        """
        full_len = 4096

        def _halve_seqdim(batch, cp_size):
            for key in ("tokens", "labels", "loss_mask", "position_ids"):
                if batch.get(key) is not None:
                    batch[key] = batch[key][:, : full_len // 2]
            return batch

        batch = _make_packed_batch(full_len, [0, 1000, 4096])
        data_iterator = iter([batch])
        cfg = _make_cfg(packed=True)
        pg_collection = _make_pg_collection(cp_size=2)
        with (
            patch.object(gpt_step, "is_pp_first_stage", return_value=True),
            patch.object(gpt_step, "is_pp_last_stage", return_value=False),
            patch.object(gpt_step, "_partition_packed_batch_for_cp", side_effect=_halve_seqdim),
            patch.object(gpt_step, "get_batch_on_this_cp_rank", side_effect=_identity_cp),
            patch.object(torch.Tensor, "cuda", lambda self, *a, **k: self),
        ):
            result = gpt_step.get_batch(data_iterator, cfg, use_mtp=False, pg_collection=pg_collection)

        # tokens were sliced to half, but full_seq_length is the FULL pack length.
        assert result[0].size(1) == full_len // 2
        assert result[_FULL_SEQ_IDX] == full_len


class TestDatasetUsesPackedSequences:
    """The packed predicate must be robust across dataset config shapes."""

    def test_finetuning_packed_specs(self):
        cfg = MagicMock()
        cfg.dataset.packed_sequence_specs.packed_sequence_size = 8192
        cfg.dataset.pack_sequences_in_batch = False
        assert gpt_step._dataset_uses_packed_sequences(cfg) is True

    def test_packed_specs_size_zero_is_not_packed(self):
        cfg = MagicMock()
        cfg.dataset.packed_sequence_specs.packed_sequence_size = -1
        cfg.dataset.pack_sequences_in_batch = False
        assert gpt_step._dataset_uses_packed_sequences(cfg) is False

    def test_pack_sequences_in_batch_flag(self):
        cfg = MagicMock()
        cfg.dataset.packed_sequence_specs = None
        cfg.dataset.pack_sequences_in_batch = True
        assert gpt_step._dataset_uses_packed_sequences(cfg) is True

    def test_plain_dataset_not_packed(self):
        cfg = MagicMock()
        cfg.dataset.packed_sequence_specs = None
        cfg.dataset.pack_sequences_in_batch = False
        assert gpt_step._dataset_uses_packed_sequences(cfg) is False

    def test_no_dataset_attr(self):
        cfg = MagicMock(spec=[])  # no .dataset attribute
        assert gpt_step._dataset_uses_packed_sequences(cfg) is False

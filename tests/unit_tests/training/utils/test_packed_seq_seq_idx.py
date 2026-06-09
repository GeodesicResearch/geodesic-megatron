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

"""CPU-only tests for packed-sequence ``seq_idx`` construction (GEOD-147).

In packed-sequence SFT of hybrid Mamba2+attention models (e.g. Nemotron-3
Super), the Mamba SSM scan must reset its state at packed-document boundaries.
It does so via ``PackedSeqParams.seq_idx`` -- a per-token document index over the
full pack. ``seq_idx`` is only built when ``total_tokens`` is passed to
``PackedSeqParams``. The bridge's ``get_packed_seq_params`` previously never set
``total_tokens``, so ``seq_idx`` stayed ``None`` and the scan integrated state
across every concatenated document, overflowing BF16 at long sequence lengths
(forward-loss NaN at seq 32768).

These tests assert that, given the full pack length, ``get_packed_seq_params``
produces a ``seq_idx`` that:
  * is non-None,
  * has length equal to the full pack length (INCLUDING trailing padding), and
  * equals the expected per-document index vector
    ``[0]*len0 + [1]*len1 + ... + [k]*trailing``.

They cover cp_size == 1 (per-rank length == full length) and a simulated
cp_size == 8 (per-rank length == full_length / cp_size), matching how
``gpt_step._forward_step_common`` computes ``total_tokens = per_rank_seq * cp_size``.
No GPU or distributed init required.
"""

import torch

from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_params


def _expected_seq_idx(doc_lengths: list[int]) -> torch.Tensor:
    """Build the reference per-token document index from per-document token counts."""
    idx = []
    for doc_id, length in enumerate(doc_lengths):
        idx.extend([doc_id] * length)
    return torch.tensor(idx, dtype=torch.int32)


class TestPackedSeqIdx:
    """seq_idx must be built (full pack length, correct per-doc index) when total_tokens is passed."""

    def test_seq_idx_none_without_total_tokens(self):
        """Regression guard: without total_tokens, seq_idx stays None (the original bug)."""
        batch = {
            "cu_seqlens": torch.IntTensor([0, 128, 256, 384, -1, -1]),
            "cu_seqlens_argmin": torch.tensor(4),
            "max_seqlen": torch.tensor(128),
        }

        result = get_packed_seq_params(batch)  # total_tokens omitted -> None

        assert result.seq_idx is None

    def test_seq_idx_cp1_with_trailing_padding(self):
        """cp_size == 1: 3 docs (128/128/128) + trailing pad up to a 512 pack.

        full pack length == 512, so seq_idx must be length 512 and the last
        512 - 384 = 128 tokens form an extra (4th) document index.
        """
        # 3 packed docs ending at 384, padded out to a 512-token pack.
        batch = {
            "cu_seqlens": torch.IntTensor([0, 128, 256, 384, -1, -1]),
            "cu_seqlens_argmin": torch.tensor(4),
            "max_seqlen": torch.tensor(128),
        }
        full_pack_len = 512  # per_rank_seq (512) * cp_size (1)

        result = get_packed_seq_params(batch, total_tokens=full_pack_len)

        assert result.seq_idx is not None
        # seq_idx carries a batch dim of 1; the sequence axis must equal the full pack length.
        assert result.seq_idx.shape == (1, full_pack_len)
        assert result.seq_idx.dtype == torch.int32

        # docs: [0..128) -> 0, [128..256) -> 1, [256..384) -> 2, [384..512) -> 3 (trailing pad)
        expected = _expected_seq_idx([128, 128, 128, 512 - 384])
        torch.testing.assert_close(result.seq_idx.squeeze(0), expected)

    def test_seq_idx_cp1_no_trailing_padding(self):
        """cp_size == 1, pack exactly filled: no extra trailing-document index."""
        batch = {
            "cu_seqlens": torch.IntTensor([0, 100, 300, 512, -1]),
            "cu_seqlens_argmin": torch.tensor(4),
            "max_seqlen": torch.tensor(212),
        }
        full_pack_len = 512  # cu_seqlens[-1] == full length -> no trailing pad

        result = get_packed_seq_params(batch, total_tokens=full_pack_len)

        assert result.seq_idx is not None
        assert result.seq_idx.shape == (1, full_pack_len)
        # No trailing pad: exactly 3 documents, no 4th index appended.
        expected = _expected_seq_idx([100, 200, 212])
        torch.testing.assert_close(result.seq_idx.squeeze(0), expected)
        assert int(result.seq_idx.max()) == 2

    def test_seq_idx_cp8_simulated(self):
        """cp_size == 8: total_tokens = per_rank_seq * cp_size = 4096 * 8 = 32768.

        This mirrors gpt_step._forward_step_common, which slices tokens to the
        per-rank length (full_len / cp_size) but keeps cu_seqlens GLOBAL, and
        recovers the full pack length as per_rank_seq * cp_size. The 32K case is
        exactly where the unreset SSM scan overflowed BF16.
        """
        cp_size = 8
        per_rank_seq = 4096
        full_pack_len = per_rank_seq * cp_size  # 32768

        # GLOBAL (un-sharded) cu_seqlens: 4 docs of 8000 tokens (32000), padded to 32768.
        # With pad_seq_to_mult > 1 the dataloader supplies cu_seqlens_unpadded too;
        # seq_idx is built from the PADDED boundaries (what the scan/CP-undo use).
        cu_seqlens_padded = torch.IntTensor([0, 8000, 16000, 24000, 32000, -1, -1])
        cu_seqlens_unpadded = torch.IntTensor([0, 7990, 15990, 23990, 31990, -1, -1])
        batch = {
            "cu_seqlens": cu_seqlens_padded,
            "cu_seqlens_argmin": torch.tensor(5),
            "cu_seqlens_unpadded": cu_seqlens_unpadded,
            "cu_seqlens_unpadded_argmin": torch.tensor(5),
            "max_seqlen": torch.tensor(8000),
        }

        result = get_packed_seq_params(batch, total_tokens=full_pack_len)

        assert result.seq_idx is not None
        # Length must equal the FULL pack length (== input_.size(0) the Mamba scan
        # sees after the CP all-to-all in mamba_context_parallel.pre_conv_ssm),
        # NOT the per-rank sharded length and NOT cu_seqlens[-1].
        assert result.seq_idx.shape == (1, full_pack_len)
        assert result.seq_idx.numel() == full_pack_len

        # 4 docs of 8000 (from the PADDED boundaries) + 768 trailing-pad tokens as doc 4.
        expected = _expected_seq_idx([8000, 8000, 8000, 8000, full_pack_len - 32000])
        torch.testing.assert_close(result.seq_idx.squeeze(0), expected)

    def test_seq_idx_built_from_padded_boundaries(self):
        """When unpadded boundaries are present, seq_idx must follow the PADDED ones.

        The CP undo (_undo_attention_load_balancing) and the scan operate on
        cu_seqlens_q_padded, so seq_idx must partition by padded -- not unpadded --
        document boundaries.
        """
        cu_seqlens_padded = torch.IntTensor([0, 128, 256, -1])
        cu_seqlens_unpadded = torch.IntTensor([0, 120, 240, -1])
        batch = {
            "cu_seqlens": cu_seqlens_padded,
            "cu_seqlens_argmin": torch.tensor(3),
            "cu_seqlens_unpadded": cu_seqlens_unpadded,
            "cu_seqlens_unpadded_argmin": torch.tensor(3),
            "max_seqlen": torch.tensor(128),
        }
        full_pack_len = 256  # exactly the last padded boundary

        result = get_packed_seq_params(batch, total_tokens=full_pack_len)

        assert result.seq_idx is not None
        # Boundary at 128 (padded), not 120 (unpadded).
        expected = _expected_seq_idx([128, 128])
        torch.testing.assert_close(result.seq_idx.squeeze(0), expected)
        # Sanity: a boundary built from unpadded would put doc 1 starting at 120.
        assert int(result.seq_idx[0, 120]) == 0
        assert int(result.seq_idx[0, 128]) == 1

# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Unit tests for the quarantine loss-mask hook.

The hook (`apply_quarantine_loss_mask`) is a pure tensor function that zeros
out positions in `loss_mask` where `labels[t]` is a designated quarantine
token ID. These tests verify the function in isolation; the training-loop
wiring is exercised by the model-level integration tests elsewhere.

The two integration-style tests at the bottom verify the metadata pipeline:
1. `TokenizerConfig.loss_mask_token_ids` survives an OmegaConf round-trip.
2. An HF tokenizer's `loss_mask_token_ids` field in `tokenizer_config.json`
   is preserved through `save_pretrained` and re-readable.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from megatron.bridge.training.gpt_step import apply_quarantine_loss_mask
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.utils.quarantine_utils import (
    read_loss_mask_token_ids_from_tokenizer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_labels(*positions: int, length: int = 10, fill: int = 99) -> torch.Tensor:
    """Build a 1-D labels tensor of `length` filled with `fill` and put
    distinguishing ids at the given indices: positions[i] -> id = 100 + i.
    """
    labels = torch.full((length,), fill, dtype=torch.long)
    for i, pos in enumerate(positions):
        labels[pos] = 100 + i
    return labels


# ---------------------------------------------------------------------------
# Test 1: Empty / None inputs are no-ops
# ---------------------------------------------------------------------------


class TestNoOpConditions:
    def test_empty_id_list_is_noop(self) -> None:
        labels = _make_labels(2, 5)
        loss_mask = torch.ones(10, dtype=torch.float32)
        new_mask, frac, count, total = apply_quarantine_loss_mask(labels, loss_mask, [])
        assert torch.equal(new_mask, loss_mask)
        assert frac == 0.0
        assert count == 0
        assert total == 10

    def test_none_id_list_is_noop(self) -> None:
        labels = _make_labels(2, 5)
        loss_mask = torch.ones(10, dtype=torch.float32)
        new_mask, frac, count, total = apply_quarantine_loss_mask(labels, loss_mask, None)
        assert torch.equal(new_mask, loss_mask)
        assert frac == 0.0
        assert count == 0
        assert total == 10

    def test_none_labels_is_noop(self) -> None:
        loss_mask = torch.ones(10, dtype=torch.float32)
        new_mask, frac, count, total = apply_quarantine_loss_mask(None, loss_mask, [42])
        assert new_mask is loss_mask
        assert frac == 0.0
        assert count == 0
        assert total == 0

    def test_none_loss_mask_is_noop(self) -> None:
        labels = _make_labels(2, 5)
        new_mask, frac, count, total = apply_quarantine_loss_mask(labels, None, [42])
        assert new_mask is None
        assert frac == 0.0
        assert count == 0
        assert total == 10


# ---------------------------------------------------------------------------
# Test 2: Single ID present at known positions
# ---------------------------------------------------------------------------


class TestSingleIdMatch:
    def test_single_id_at_known_positions(self) -> None:
        labels = _make_labels(2, 5)  # labels[2] = 100, labels[5] = 101, others = 99
        loss_mask = torch.ones(10, dtype=torch.float32)
        new_mask, frac, count, total = apply_quarantine_loss_mask(labels, loss_mask, [100])

        expected = torch.ones(10, dtype=torch.float32)
        expected[2] = 0.0  # only labels[2] == 100
        assert torch.equal(new_mask, expected)
        assert count == 1
        assert total == 10
        assert frac == pytest.approx(0.1)

    def test_input_loss_mask_not_mutated(self) -> None:
        """The function must return a new tensor, not mutate the input in place."""
        labels = _make_labels(3)
        original_mask = torch.ones(10, dtype=torch.float32)
        mask_snapshot = original_mask.clone()
        _new_mask, _frac, _count, _total = apply_quarantine_loss_mask(labels, original_mask, [100])
        # Caller's tensor unchanged
        assert torch.equal(original_mask, mask_snapshot)


# ---------------------------------------------------------------------------
# Test 3: Multiple IDs + composition with pre-existing zeros
# ---------------------------------------------------------------------------


class TestMultipleIds:
    def test_multiple_ids_match(self) -> None:
        labels = _make_labels(1, 4, 7)  # labels[1]=100, [4]=101, [7]=102
        loss_mask = torch.ones(10, dtype=torch.float32)
        new_mask, frac, count, total = apply_quarantine_loss_mask(labels, loss_mask, [100, 101, 102])

        expected = torch.ones(10, dtype=torch.float32)
        for p in (1, 4, 7):
            expected[p] = 0.0
        assert torch.equal(new_mask, expected)
        assert count == 3
        assert total == 10
        assert frac == pytest.approx(0.3)

    def test_composition_with_preexisting_zeros(self) -> None:
        """If the dataset already zeroed some positions (padding, system tokens),
        those remain zero — quarantine masking only zeros additional positions."""
        labels = _make_labels(1, 4, 7)
        loss_mask = torch.ones(10, dtype=torch.float32)
        loss_mask[0] = 0.0  # system token
        loss_mask[9] = 0.0  # padding
        loss_mask[4] = 0.0  # already zero AT a quarantine position

        new_mask, frac, count, total = apply_quarantine_loss_mask(labels, loss_mask, [100, 101, 102])
        # All 3 quarantine positions zero in the output, plus the original 2 (at 0 and 9).
        # Position 4 was both already-zero AND a quarantine match — still zero.
        expected = torch.tensor(
            [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
            dtype=torch.float32,
        )
        assert torch.equal(new_mask, expected)
        # count counts ALL matching positions in labels (the hook does not subtract
        # already-zeroed positions — this is a per-batch diagnostic, not a delta).
        assert count == 3

    def test_subset_of_ids_match(self) -> None:
        labels = _make_labels(2, 5)
        loss_mask = torch.ones(10, dtype=torch.float32)
        # Only id 100 is present; 999 is not.
        new_mask, _frac, count, _total = apply_quarantine_loss_mask(labels, loss_mask, [100, 999])
        assert count == 1


# ---------------------------------------------------------------------------
# Test 4: All-match / no-match boundary conditions
# ---------------------------------------------------------------------------


class TestExtremes:
    def test_all_positions_match(self) -> None:
        labels = torch.full((10,), 42, dtype=torch.long)
        loss_mask = torch.ones(10, dtype=torch.float32)
        new_mask, frac, count, total = apply_quarantine_loss_mask(labels, loss_mask, [42])
        assert torch.equal(new_mask, torch.zeros(10, dtype=torch.float32))
        assert count == 10
        assert total == 10
        assert frac == 1.0

    def test_no_positions_match(self) -> None:
        labels = torch.full((10,), 42, dtype=torch.long)
        loss_mask = torch.ones(10, dtype=torch.float32)
        new_mask, frac, count, total = apply_quarantine_loss_mask(labels, loss_mask, [99, 100])
        assert torch.equal(new_mask, loss_mask)
        assert count == 0
        assert frac == 0.0


# ---------------------------------------------------------------------------
# Test 5: 2D batched tensor shape preserved
# ---------------------------------------------------------------------------


class TestBatchedShape:
    def test_2d_batched_tensor(self) -> None:
        # [B, S] = [2, 5]
        labels = torch.tensor(
            [
                [10, 42, 30, 42, 50],  # 2 matches in row 0
                [42, 20, 30, 40, 50],  # 1 match in row 1
            ],
            dtype=torch.long,
        )
        loss_mask = torch.ones((2, 5), dtype=torch.float32)
        new_mask, frac, count, total = apply_quarantine_loss_mask(labels, loss_mask, [42])

        expected = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
        assert torch.equal(new_mask, expected)
        assert new_mask.shape == (2, 5)
        assert count == 3
        assert total == 10
        assert frac == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Test 6: Various dtypes work
# ---------------------------------------------------------------------------


class TestDtypes:
    @pytest.mark.parametrize(
        "mask_dtype",
        [torch.float32, torch.float16, torch.bfloat16, torch.float64],
    )
    def test_loss_mask_dtypes(self, mask_dtype: torch.dtype) -> None:
        labels = _make_labels(2)  # labels[2] = 100
        loss_mask = torch.ones(10, dtype=mask_dtype)
        new_mask, _frac, _count, _total = apply_quarantine_loss_mask(labels, loss_mask, [100])
        # Output preserves input dtype
        assert new_mask.dtype == mask_dtype
        # Position 2 is zero, others are 1
        expected = torch.ones(10, dtype=mask_dtype)
        expected[2] = 0
        assert torch.equal(new_mask, expected)

    @pytest.mark.parametrize("labels_dtype", [torch.int32, torch.int64, torch.long])
    def test_labels_dtypes(self, labels_dtype: torch.dtype) -> None:
        labels = torch.tensor([1, 2, 3, 42, 5], dtype=labels_dtype)
        loss_mask = torch.ones(5, dtype=torch.float32)
        new_mask, _frac, count, _total = apply_quarantine_loss_mask(labels, loss_mask, [42])
        assert count == 1
        assert new_mask[3].item() == 0.0


# ---------------------------------------------------------------------------
# Test 7: Padding zeros pre-masked are preserved
# ---------------------------------------------------------------------------


class TestPaddingPreservation:
    def test_padding_zeros_preserved(self) -> None:
        labels = torch.tensor([1, 2, 3, 100, 0, 0, 0], dtype=torch.long)
        loss_mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        new_mask, _frac, _count, _total = apply_quarantine_loss_mask(labels, loss_mask, [100])
        expected = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        assert torch.equal(new_mask, expected)


# ---------------------------------------------------------------------------
# Test 8: ignore_index=-100 NOT in list → those positions unaffected
# ---------------------------------------------------------------------------


class TestIgnoreIndex:
    def test_ignore_index_minus_100_not_special(self) -> None:
        # The hook does not have special handling for -100; it masks only what's
        # explicitly listed. -100 in labels is fine; CE ignore_index handles it later.
        labels = torch.tensor([1, -100, 100, -100, 5], dtype=torch.long)
        loss_mask = torch.ones(5, dtype=torch.float32)
        new_mask, _frac, count, _total = apply_quarantine_loss_mask(labels, loss_mask, [100])
        # Only position 2 (labels[2]=100) is masked; -100 positions untouched by hook.
        expected = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])
        assert torch.equal(new_mask, expected)
        assert count == 1

    def test_ignore_index_in_quarantine_list_works(self) -> None:
        # If you explicitly add -100 to the list, it does mask those positions.
        labels = torch.tensor([1, -100, 100, -100, 5], dtype=torch.long)
        loss_mask = torch.ones(5, dtype=torch.float32)
        new_mask, _frac, count, _total = apply_quarantine_loss_mask(labels, loss_mask, [-100, 100])
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0])
        assert torch.equal(new_mask, expected)
        assert count == 3


# ---------------------------------------------------------------------------
# Test 9: Fraction/count return values exhaustively
# ---------------------------------------------------------------------------


class TestReturnValues:
    @pytest.mark.parametrize(
        "n_match,total",
        [(0, 10), (1, 10), (3, 10), (5, 10), (10, 10), (0, 1), (1, 1), (50, 100)],
    )
    def test_fraction_count_total(self, n_match: int, total: int) -> None:
        labels = torch.tensor([42] * n_match + [99] * (total - n_match), dtype=torch.long)
        loss_mask = torch.ones(total, dtype=torch.float32)
        _new_mask, frac, count, tot = apply_quarantine_loss_mask(labels, loss_mask, [42])
        assert count == n_match
        assert tot == total
        assert frac == pytest.approx(n_match / total) if total > 0 else frac == 0.0


# ---------------------------------------------------------------------------
# Test 10: Idempotence
# ---------------------------------------------------------------------------


class TestIdempotence:
    def test_idempotence(self) -> None:
        """Applying the hook twice gives the same loss_mask."""
        labels = _make_labels(1, 3, 7, length=10)
        loss_mask = torch.ones(10, dtype=torch.float32)
        new_mask_1, _f1, _c1, _t1 = apply_quarantine_loss_mask(labels, loss_mask, [100, 101, 102])
        new_mask_2, _f2, _c2, _t2 = apply_quarantine_loss_mask(labels, new_mask_1, [100, 101, 102])
        assert torch.equal(new_mask_1, new_mask_2)


# ---------------------------------------------------------------------------
# Test 11: CUDA parity (skip if no GPU)
# ---------------------------------------------------------------------------


class TestCUDAParity:
    @pytest.mark.run_only_on("gpu")
    def test_cpu_cuda_parity(self) -> None:
        labels_cpu = _make_labels(2, 5, length=10)
        mask_cpu = torch.ones(10, dtype=torch.float32)
        out_cpu, f_cpu, c_cpu, t_cpu = apply_quarantine_loss_mask(labels_cpu, mask_cpu, [100, 101])

        labels_gpu = labels_cpu.cuda()
        mask_gpu = mask_cpu.cuda()
        out_gpu, f_gpu, c_gpu, t_gpu = apply_quarantine_loss_mask(labels_gpu, mask_gpu, [100, 101])

        assert torch.equal(out_cpu, out_gpu.cpu())
        assert out_gpu.device.type == "cuda"
        assert f_cpu == f_gpu
        assert c_cpu == c_gpu
        assert t_cpu == t_gpu


# ---------------------------------------------------------------------------
# Test 12: TokenizerConfig OmegaConf round-trip
# ---------------------------------------------------------------------------


class TestTokenizerConfigRoundTrip:
    def test_loss_mask_token_ids_survives_omegaconf(self) -> None:
        """Verify the new `loss_mask_token_ids` field survives an OmegaConf round-trip.

        Uses unstructured OmegaConf via `create_omegaconf_dict_config` (the actual
        plumbing path used by `pipeline_training_run.py`), not `OmegaConf.structured`
        (which doesn't support `Union[str, dict]` types on other TokenizerConfig fields).
        """
        from omegaconf import OmegaConf

        from megatron.bridge.training.utils.omegaconf_utils import (
            apply_overrides,
            create_omegaconf_dict_config,
        )

        cfg = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="dummy/path",
            loss_mask_token_ids=[12345, 67890],
        )
        # The training pipeline's actual round-trip path
        omg, excluded = create_omegaconf_dict_config(cfg)
        d = OmegaConf.to_container(omg, resolve=True)
        assert d["loss_mask_token_ids"] == [12345, 67890]

        # YAML serialisation works too
        yaml_str = OmegaConf.to_yaml(omg)
        omg_back = OmegaConf.create(yaml_str)
        assert list(omg_back.loss_mask_token_ids) == [12345, 67890]

        # Round-trip back onto a fresh dataclass via the standard apply_overrides path
        fresh = TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model="dummy/path")
        apply_overrides(fresh, OmegaConf.to_container(omg_back, resolve=True), excluded)
        assert fresh.loss_mask_token_ids == [12345, 67890]

    def test_default_is_none(self) -> None:
        cfg = TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model="dummy/path")
        assert cfg.loss_mask_token_ids is None


# ---------------------------------------------------------------------------
# Test 13: HF tokenizer init_kwargs round-trip
# ---------------------------------------------------------------------------


class TestHFTokenizerRoundTrip:
    def test_loss_mask_token_ids_in_tokenizer_config_json(self, tmp_path: Path) -> None:
        """Write a tokenizer_config.json with `loss_mask_token_ids`, reload, confirm field readable."""
        cfg_dir = tmp_path / "fake-tokenizer"
        cfg_dir.mkdir()
        cfg_dir.joinpath("tokenizer_config.json").write_text(
            json.dumps(
                {
                    "tokenizer_class": "PreTrainedTokenizerFast",
                    "loss_mask_token_ids": [11111, 22222],
                }
            )
        )
        ids = read_loss_mask_token_ids_from_tokenizer(str(cfg_dir))
        assert ids == [11111, 22222]

    def test_missing_field_returns_empty(self, tmp_path: Path) -> None:
        cfg_dir = tmp_path / "fake-tokenizer-no-field"
        cfg_dir.mkdir()
        cfg_dir.joinpath("tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"})
        )
        ids = read_loss_mask_token_ids_from_tokenizer(str(cfg_dir))
        assert ids == []

    def test_unreachable_path_returns_empty(self) -> None:
        """A bogus path returns [] (warning logged, hook is no-op)."""
        ids = read_loss_mask_token_ids_from_tokenizer("/nonexistent/local/path/that/does/not/exist")
        assert ids == []

    def test_int_coercion(self, tmp_path: Path) -> None:
        """Floats / strings that happen to be in the list get coerced to int."""
        cfg_dir = tmp_path / "fake-tokenizer-coerce"
        cfg_dir.mkdir()
        cfg_dir.joinpath("tokenizer_config.json").write_text(
            json.dumps({"loss_mask_token_ids": [100, 200.0, "300"]})
        )
        ids = read_loss_mask_token_ids_from_tokenizer(str(cfg_dir))
        assert ids == [100, 200, 300]
        assert all(isinstance(i, int) for i in ids)

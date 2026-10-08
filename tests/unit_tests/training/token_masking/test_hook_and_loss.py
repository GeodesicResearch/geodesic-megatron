# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The per-microbatch token-masking hook and the loss function that reports what it did.

``apply_token_masking`` turns a microbatch's ``loss_mask`` into the mask the loss is trained with: every target
position whose label is a masked id loses its loss, every other position keeps exactly the value it had. It also
counts, as device tensors, the listed targets it saw and the ones it removed. ``masked_next_token_loss`` multiplies
the per-token losses by a mask and adds the counts to its reporting dict, measuring the trained-listed fraction
against the mask it actually multiplied with, which is how a loss computed with any mask other than the masked one
(the leak the monitor stops a run for) becomes visible.

Every decision here is a real ``ResolvedTokenMasking`` from the production resolution (on a ``NullTokenizer``, which
accepts explicit ids without a tokenizer file), and the loss function runs under Megatron's real rerun state machine
in its default mode. The microbatch below is written out by hand, with its expected masks and counts, so every
assertion compares the code against values worked out independently of it:

    labels      3  7  5  7  2  4  |  7  1  9  -100  7  6
    loss_mask   1  1  1  0  1  1  |  0  1  1   0    1  1

Id 7 occurs at four targets, two of which carried loss; id 9 at one, which carried loss; -100 is the ignore index.
"""

import megatron.core.rerun_state_machine as rerun_state_machine_module
import pytest
import torch
from megatron.core.rerun_state_machine import RerunStateMachine

from megatron.bridge.training.losses import (
    create_masked_next_token_loss_function,
    masked_next_token_loss,
    reports_a_loss,
)
from megatron.bridge.training.token_masking.config import TokenMaskingConfig
from megatron.bridge.training.token_masking.hook import (
    LISTED_TARGET_FRACTION,
    MASKED_TARGET_FRACTION,
    REPORT_KEYS,
    TRAINABLE_TARGET_FRACTION,
    TRAINED_LISTED_TARGET_FRACTION,
    apply_token_masking,
)
from tests.unit_tests.token_masking_fixtures import (
    masking_with_null_tokenizer,
    no_token_masking,
    null_tokenizer_config,
    resolve,
)


VOCAB_SIZE = 64
MARKER_ID = 7
SECOND_MARKER_ID = 9

LABELS = [[3, 7, 5, 7, 2, 4], [7, 1, 9, -100, 7, 6]]
LOSS_MASK = [[1, 1, 1, 0, 1, 1], [0, 1, 1, 0, 1, 1]]
POSITIONS = 12
TRAINABLE_BEFORE = 9

# Masking id 7: its four targets are listed; the two that carried loss lose it.
LISTED_7 = [[0, 1, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0]]
MASKED_7 = [[1, 0, 1, 0, 1, 1], [0, 1, 1, 0, 0, 1]]
COUNTS_7 = {"listed": 4, "masked": 2, "trainable_after": 7}

# Masking ids 7 and 9: id 9's single target carried loss too.
MASKED_7_9 = [[1, 0, 1, 0, 1, 1], [0, 1, 0, 0, 0, 1]]
COUNTS_7_9 = {"listed": 5, "masked": 3, "trainable_after": 6}

SHAPES = [(12,), (2, 6), (2, 2, 3), (1, 12)]


def _labels(shape=(2, 6), dtype=torch.long, device="cpu") -> torch.Tensor:
    return torch.tensor(LABELS, dtype=dtype, device=device).reshape(shape)


def _loss_mask(shape=(2, 6), dtype=torch.float32, device="cpu") -> torch.Tensor:
    return torch.tensor(LOSS_MASK, device=device).reshape(shape).to(dtype)


def _expected(rows, shape=(2, 6)) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32).reshape(shape)


def _pair(numerator: float, denominator: float = POSITIONS) -> torch.Tensor:
    return torch.tensor([numerator, denominator], dtype=torch.float32)


@pytest.fixture(scope="module")
def enabled_7():
    resolved = masking_with_null_tokenizer("enabled", [MARKER_ID], VOCAB_SIZE)
    assert resolved.enabled and resolved.ids_tensor is not None
    return resolved


@pytest.fixture(scope="module")
def enabled_7_9():
    resolved = masking_with_null_tokenizer("enabled", [MARKER_ID, SECOND_MARKER_ID], VOCAB_SIZE)
    assert resolved.enabled and resolved.token_ids == (MARKER_ID, SECOND_MARKER_ID)
    return resolved


@pytest.fixture(scope="module")
def disabled_7():
    resolved = masking_with_null_tokenizer("disabled", [MARKER_ID], VOCAB_SIZE)
    assert not resolved.enabled and resolved.observed_token_ids == (MARKER_ID,)
    return resolved


@pytest.fixture(scope="module")
def observes_nothing():
    resolved = no_token_masking()
    assert resolved.ids_tensor is None and not resolved.observed_token_ids
    return resolved


@pytest.fixture
def rerun_state_machine(monkeypatch):
    """Megatron's real rerun state machine in its default (disabled) mode, which still rejects NaN and Inf losses.

    Installed as the process-wide singleton for one test and removed afterwards, so other tests find it as they left
    it.
    """
    monkeypatch.setattr(rerun_state_machine_module, "_GLOBAL_RERUN_STATE_MACHINE", RerunStateMachine())


class TestApplyTokenMasking:
    def test_a_run_observing_no_ids_returns_the_mask_itself_and_no_stats(self, observes_nothing):
        loss_mask = _loss_mask()
        masked, stats = apply_token_masking(_labels(), loss_mask, observes_nothing)
        assert masked is loss_mask
        assert stats is None

    @pytest.mark.parametrize("masking", ["enabled_7", "disabled_7"])
    @pytest.mark.parametrize("has_loss_mask", [True, False])
    def test_a_stage_without_labels_returns_the_mask_itself_and_no_stats(self, request, masking, has_loss_mask):
        """Pipeline stages other than the last hold no labels (and usually no mask); there is nothing to count."""
        loss_mask = _loss_mask() if has_loss_mask else None
        masked, stats = apply_token_masking(None, loss_mask, request.getfixturevalue(masking))
        assert masked is loss_mask
        assert stats is None

    def test_enabled_zeroes_exactly_the_listed_targets(self, enabled_7):
        masked, stats = apply_token_masking(_labels(), _loss_mask(), enabled_7)
        assert torch.equal(masked, _expected(MASKED_7))
        assert torch.equal(stats.listed, _expected(LISTED_7).bool())
        assert stats.positions.item() == POSITIONS
        assert stats.listed_targets.item() == COUNTS_7["listed"]
        assert stats.masked_targets.item() == COUNTS_7["masked"]

    @pytest.mark.parametrize(
        "loss_mask_rows, expected_masked",
        [
            pytest.param([[0] * 6] * 2, 0, id="no-position-carries-loss"),
            pytest.param([[1] * 6] * 2, 4, id="every-position-carries-loss"),
            pytest.param(LOSS_MASK, 2, id="answer-only-mask"),
        ],
    )
    def test_masked_targets_counts_only_listed_targets_that_carried_loss(
        self, enabled_7, loss_mask_rows, expected_masked
    ):
        """A listed target the dataset already excluded (prompt, padding) is listed but not masked by this step."""
        loss_mask = torch.tensor(loss_mask_rows, dtype=torch.float32)
        masked, stats = apply_token_masking(_labels(), loss_mask, enabled_7)
        assert stats.listed_targets.item() == COUNTS_7["listed"]
        assert stats.masked_targets.item() == expected_masked
        assert torch.equal(masked, loss_mask * (1 - _expected(LISTED_7)))

    def test_disabled_counts_listed_targets_but_returns_the_mask_itself(self, disabled_7):
        loss_mask = _loss_mask()
        masked, stats = apply_token_masking(_labels(), loss_mask, disabled_7)
        assert masked is loss_mask
        assert torch.equal(masked, _loss_mask())
        assert stats.listed_targets.item() == COUNTS_7["listed"]
        assert stats.masked_targets.item() == 0
        # A run that only observes trains on its listed targets, and its report says so.
        assert torch.equal(stats.report(masked)[TRAINED_LISTED_TARGET_FRACTION], _pair(2))

    def test_every_listed_id_is_masked(self, enabled_7_9):
        masked, stats = apply_token_masking(_labels(), _loss_mask(), enabled_7_9)
        assert torch.equal(masked, _expected(MASKED_7_9))
        assert stats.listed_targets.item() == COUNTS_7_9["listed"]
        assert stats.masked_targets.item() == COUNTS_7_9["masked"]

    @pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: "x".join(map(str, shape)))
    def test_any_shape_is_masked_position_by_position(self, enabled_7, shape):
        masked, stats = apply_token_masking(_labels(shape), _loss_mask(shape), enabled_7)
        assert masked.shape == shape
        assert torch.equal(masked, _expected(MASKED_7, shape))
        assert stats.positions.item() == POSITIONS
        assert stats.masked_targets.item() == COUNTS_7["masked"]
        assert torch.equal(stats.report(masked)[TRAINABLE_TARGET_FRACTION], _pair(COUNTS_7["trainable_after"]))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.long, torch.bool], ids=str)
    def test_the_mask_keeps_its_dtype(self, enabled_7, dtype):
        masked, stats = apply_token_masking(_labels(), _loss_mask(dtype=dtype), enabled_7)
        assert masked.dtype == dtype
        assert torch.equal(masked.float(), _expected(MASKED_7))
        assert stats.masked_targets.item() == COUNTS_7["masked"]

    def test_a_fractional_mask_keeps_its_weights_on_unlisted_targets(self, enabled_7):
        """Masking multiplies by 0 or 1, so a weighted mask keeps every unlisted target's weight exactly."""
        weights = torch.linspace(0.1, 1.2, POSITIONS).reshape(2, 6)
        masked, _ = apply_token_masking(_labels(), weights, enabled_7)
        assert torch.equal(masked, weights * (1 - _expected(LISTED_7)))

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64], ids=str)
    def test_labels_of_either_integer_width_match(self, enabled_7, dtype):
        masked, stats = apply_token_masking(_labels(dtype=dtype), _loss_mask(), enabled_7)
        assert torch.equal(masked, _expected(MASKED_7))
        assert stats.listed_targets.item() == COUNTS_7["listed"]

    def test_the_ignore_index_is_never_a_listed_target(self, enabled_7):
        labels = torch.full((2, 6), -100, dtype=torch.long)
        labels[1, 4] = MARKER_ID
        loss_mask = torch.ones(2, 6)
        masked, stats = apply_token_masking(labels, loss_mask, enabled_7)
        assert stats.listed_targets.item() == 1
        assert stats.masked_targets.item() == 1
        assert masked.sum().item() == POSITIONS - 1
        assert masked[1, 4].item() == 0

    @pytest.mark.parametrize("masking", ["enabled_7", "disabled_7"])
    @pytest.mark.parametrize("shape", [(0,), (2, 0)], ids=lambda shape: "x".join(map(str, shape)))
    def test_an_empty_microbatch_counts_nothing(self, request, masking, shape):
        masked, stats = apply_token_masking(
            torch.empty(shape, dtype=torch.long), torch.empty(shape), request.getfixturevalue(masking)
        )
        assert masked.shape == shape
        assert stats.positions.item() == 0
        assert stats.listed_targets.item() == 0
        assert stats.masked_targets.item() == 0
        report = stats.report(masked)
        assert set(report) == set(REPORT_KEYS)
        assert all(torch.equal(value, _pair(0, 0)) for value in report.values())

    def test_masking_twice_changes_nothing_more(self, enabled_7):
        once, first = apply_token_masking(_labels(), _loss_mask(), enabled_7)
        twice, second = apply_token_masking(_labels(), once, enabled_7)
        assert torch.equal(twice, once)
        assert second.listed_targets.item() == first.listed_targets.item()
        # The second pass removes nothing: every listed target already carried no loss.
        assert second.masked_targets.item() == 0

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bool], ids=str)
    def test_the_batch_tensors_are_not_modified(self, enabled_7, dtype):
        """The dataset's tensors may be reused (a rerun replays the batch), so the step must not edit them."""
        labels, loss_mask = _labels(), _loss_mask(dtype=dtype)
        masked, _ = apply_token_masking(labels, loss_mask, enabled_7)
        assert torch.equal(labels, _labels())
        assert torch.equal(loss_mask, _loss_mask(dtype=dtype))
        assert not torch.equal(masked, loss_mask)


class TestStatsReport:
    def test_against_the_masked_mask_no_listed_target_is_trained(self, enabled_7):
        masked, stats = apply_token_masking(_labels(), _loss_mask(), enabled_7)
        report = stats.report(masked)
        assert set(report) == set(REPORT_KEYS)
        assert torch.equal(report[LISTED_TARGET_FRACTION], _pair(COUNTS_7["listed"]))
        assert torch.equal(report[MASKED_TARGET_FRACTION], _pair(COUNTS_7["masked"]))
        assert torch.equal(report[TRAINED_LISTED_TARGET_FRACTION], _pair(0))
        assert torch.equal(report[TRAINABLE_TARGET_FRACTION], _pair(COUNTS_7["trainable_after"]))

    def test_against_the_unmasked_mask_the_trained_listed_targets_show(self, enabled_7):
        """The leak check: a loss multiplied by the dataset's mask instead of the masked one trains listed targets."""
        _, stats = apply_token_masking(_labels(), _loss_mask(), enabled_7)
        report = stats.report(_loss_mask())
        assert torch.equal(report[TRAINED_LISTED_TARGET_FRACTION], _pair(COUNTS_7["masked"]))
        assert torch.equal(report[TRAINABLE_TARGET_FRACTION], _pair(TRAINABLE_BEFORE))
        # What the hook saw does not depend on the mask it is measured against.
        assert torch.equal(report[LISTED_TARGET_FRACTION], _pair(COUNTS_7["listed"]))
        assert torch.equal(report[MASKED_TARGET_FRACTION], _pair(COUNTS_7["masked"]))


@pytest.mark.usefixtures("rerun_state_machine")
class TestMaskedNextTokenLoss:
    PER_TOKEN_LOSS = torch.arange(1, POSITIONS + 1, dtype=torch.float32).reshape(2, 6)

    def test_the_loss_and_token_count_use_the_masked_mask(self, enabled_7):
        masked, stats = apply_token_masking(_labels(), _loss_mask(), enabled_7)
        loss, num_tokens, report = masked_next_token_loss(
            masked, self.PER_TOKEN_LOSS, check_for_nan_in_loss=True, token_masking_stats=stats
        )
        _, unmasked_num_tokens, _ = masked_next_token_loss(_loss_mask(), self.PER_TOKEN_LOSS)

        expected_loss = (self.PER_TOKEN_LOSS * _expected(MASKED_7)).sum()
        assert loss.item() == expected_loss.item() == 44.0
        assert num_tokens.item() == COUNTS_7["trainable_after"]
        assert unmasked_num_tokens.item() - num_tokens.item() == COUNTS_7["masked"]
        assert set(report) == {"lm loss", *REPORT_KEYS}
        assert torch.equal(report["lm loss"], torch.tensor([44.0, COUNTS_7["trainable_after"]]))
        assert all(report[key].shape == (2,) for key in REPORT_KEYS)
        assert torch.equal(report[LISTED_TARGET_FRACTION], _pair(COUNTS_7["listed"]))
        assert torch.equal(report[MASKED_TARGET_FRACTION], _pair(COUNTS_7["masked"]))
        assert torch.equal(report[TRAINED_LISTED_TARGET_FRACTION], _pair(0))
        assert torch.equal(report[TRAINABLE_TARGET_FRACTION], _pair(COUNTS_7["trainable_after"]))

    def test_a_substituted_unmasked_mask_shows_as_trained_listed_targets(self, enabled_7):
        """A model returning ``(losses, mask)`` overrides the step's mask; the report must measure the one used."""
        masked, stats = apply_token_masking(_labels(), _loss_mask(), enabled_7)
        loss, num_tokens, report = masked_next_token_loss(
            masked, (self.PER_TOKEN_LOSS, _loss_mask()), token_masking_stats=stats
        )
        assert loss.item() == (self.PER_TOKEN_LOSS * _loss_mask()).sum().item() == 57.0
        assert num_tokens.item() == TRAINABLE_BEFORE
        assert torch.equal(report[TRAINED_LISTED_TARGET_FRACTION], _pair(COUNTS_7["masked"]))
        assert torch.equal(report[TRAINABLE_TARGET_FRACTION], _pair(TRAINABLE_BEFORE))

    def test_without_stats_only_the_loss_is_reported(self):
        _, _, report = masked_next_token_loss(_loss_mask(), self.PER_TOKEN_LOSS)
        assert set(report) == {"lm loss"}

    @pytest.mark.parametrize("with_stats", [True, False])
    def test_the_builders_partial_carries_the_stats(self, enabled_7, with_stats):
        masked, stats = apply_token_masking(_labels(), _loss_mask(), enabled_7)
        loss_function = create_masked_next_token_loss_function(
            masked,
            check_for_nan_in_loss=True,
            check_for_spiky_loss=False,
            token_masking_stats=stats if with_stats else None,
        )
        loss, num_tokens, report = loss_function(self.PER_TOKEN_LOSS)
        assert loss.item() == 44.0
        assert num_tokens.item() == COUNTS_7["trainable_after"]
        if with_stats:
            assert set(report) == {"lm loss", *REPORT_KEYS}
            assert torch.equal(report[MASKED_TARGET_FRACTION], _pair(COUNTS_7["masked"]))
            assert torch.equal(report[TRAINED_LISTED_TARGET_FRACTION], _pair(0))
        else:
            assert set(report) == {"lm loss"}


@pytest.mark.parametrize(
    "key, is_a_loss",
    [*((key, False) for key in REPORT_KEYS), ("lm loss", True), ("mtp_1 loss", True), ("load_balancing_loss", True)],
)
def test_only_losses_are_reported_as_losses(key, is_a_loss):
    """Evaluation prints a perplexity for losses; a fraction exponentiated as one would be nonsense."""
    assert reports_a_loss(key) is is_a_loss


# Forty ids, none of them in LABELS beyond the two markers: enough to push torch.isin onto its sorting path, which
# synchronises with the host, at this batch size.
MANY_TOKEN_IDS = [MARKER_ID, SECOND_MARKER_ID, *range(20, 58)]


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize("mode", ["enabled", "disabled"])
@pytest.mark.parametrize("token_ids", [[MARKER_ID, SECOND_MARKER_ID], MANY_TOKEN_IDS], ids=["two_ids", "forty_ids"])
def test_the_hook_and_report_never_wait_for_the_device(mode, token_ids):
    """Every microbatch runs the hook; a host synchronisation in it would stall the step on each one.

    ``set_sync_debug_mode("error")`` makes any synchronising CUDA call raise. The ``.item()`` call proves the mode is
    in force, so a pass is not vacuous.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    masking = resolve(TokenMaskingConfig(mode=mode, token_ids=token_ids), null_tokenizer_config(VOCAB_SIZE), device)
    labels, loss_mask = _labels(device=device), _loss_mask(device=device)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        masked, stats = apply_token_masking(labels, loss_mask, masking)
        report = stats.report(masked)
        with pytest.raises(RuntimeError, match="synchroniz"):
            stats.listed_targets.item()
    finally:
        torch.cuda.set_sync_debug_mode("default")

    targets = [pair for labels_row, mask_row in zip(LABELS, LOSS_MASK) for pair in zip(labels_row, mask_row)]
    listed_with_loss = sum(label in token_ids and carries_loss == 1 for label, carries_loss in targets)
    expected = {
        LISTED_TARGET_FRACTION: sum(label in token_ids for label, _ in targets),
        MASKED_TARGET_FRACTION: listed_with_loss if mode == "enabled" else 0,
        TRAINED_LISTED_TARGET_FRACTION: 0 if mode == "enabled" else listed_with_loss,
        TRAINABLE_TARGET_FRACTION: TRAINABLE_BEFORE - (listed_with_loss if mode == "enabled" else 0),
    }
    assert {key: report[key].cpu().tolist() for key in REPORT_KEYS} == {
        key: [numerator, POSITIONS] for key, numerator in expected.items()
    }

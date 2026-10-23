# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The per-microbatch token-masking hook and the loss function that reports what it did.

``apply_token_masking`` turns a microbatch's ``loss_mask`` into the mask the loss is trained with: every target
position whose label is a masked id loses its loss, every other position keeps exactly the value it had. It also
records, as device tensors, the listed targets it saw, the ones the dataset would have trained, and the ones it
removed. ``masked_next_token_loss`` multiplies the per-token losses by a mask and adds the counts and the listed
targets' summed cross-entropy to its reporting dict, measuring the trained-listed fraction against the mask it actually
multiplied with, which is how a loss computed with any mask other than the masked one (the leak the monitor stops a
run for) becomes visible. ``finalize_listed_target_loss`` turns reduced reports into the listed-target loss.

Every decision here is a real ``ResolvedTokenMasking`` from the production resolution (on a ``NullTokenizer``, which
accepts configured ids inside its vocabulary without a tokenizer file), and the loss function runs under Megatron's
real rerun state machine in its default mode. The microbatch below is written out by hand, with its expected masks
and counts, so every assertion compares the code against values worked out independently of it:

    flat position  0  1  2  3  4  5  |  6  7  8  9     10 11
    labels         3  7  5  7  2  4  |  7  1  9  -100  7  6
    loss_mask      1  1  1  0  1  1  |  0  1  1  0     1  1

Id 7 occurs at four targets (positions 1, 3, 6, 10), two of which carried loss (1 and 10); id 9 at one (8), which
carried loss; -100 is the ignore index. The per-token loss used below is ``position + 1``.
"""

import math

import megatron.core.rerun_state_machine as rerun_state_machine_module
import pytest
import torch
from megatron.core.rerun_state_machine import RerunStateMachine

from megatron.bridge.training.losses import (
    create_masked_next_token_loss_function,
    masked_next_token_loss,
    reports_a_loss,
)
from megatron.bridge.training.token_masking.hook import (
    LISTED_TARGET_FRACTION,
    LISTED_TARGET_LOSS,
    LISTED_TARGET_LOSS_SUM,
    LISTED_TRAINABLE_TARGET_FRACTION,
    MASKED_TARGET_FRACTION,
    REPORT_KEYS,
    TRAINABLE_TARGET_FRACTION,
    TRAINED_LISTED_TARGET_FRACTION,
    apply_token_masking,
    finalize_listed_target_loss,
)
from tests.unit_tests.token_masking_fixtures import (
    masking,
    masking_with_null_tokenizer,
    measuring,
    measuring_with_null_tokenizer,
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
PER_TOKEN_LOSS = torch.arange(1, POSITIONS + 1, dtype=torch.float32).reshape(2, 6)

# Id 7: its four targets are listed; the two that carried loss (positions 1 and 10, losses 2 and 11) would have
# trained, and masking removes them.
LISTED_7 = [[0, 1, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0]]
WOULD_TRAIN_7 = [[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]]
MASKED_7 = [[1, 0, 1, 0, 1, 1], [0, 1, 1, 0, 0, 1]]
COUNTS_7 = {"listed": 4, "would_train": 2, "trainable_after": 7, "listed_loss": 2.0 + 11.0}

# Ids 7 and 9: id 9's single target (position 8, loss 9) carried loss too.
MASKED_7_9 = [[1, 0, 1, 0, 1, 1], [0, 1, 0, 0, 0, 1]]
COUNTS_7_9 = {"listed": 5, "would_train": 3, "trainable_after": 6, "listed_loss": 2.0 + 11.0 + 9.0}

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
    resolved = masking_with_null_tokenizer([MARKER_ID], VOCAB_SIZE)
    assert resolved.enabled and resolved.ids_tensor is not None
    return resolved


@pytest.fixture(scope="module")
def enabled_7_9():
    resolved = masking_with_null_tokenizer([MARKER_ID, SECOND_MARKER_ID], VOCAB_SIZE)
    assert resolved.enabled and resolved.token_ids == (MARKER_ID, SECOND_MARKER_ID)
    return resolved


@pytest.fixture(scope="module")
def measuring_7():
    resolved = measuring_with_null_tokenizer([MARKER_ID], VOCAB_SIZE)
    assert not resolved.enabled and resolved.token_ids == () and resolved.measured_token_ids == (MARKER_ID,)
    return resolved


@pytest.fixture(scope="module")
def measures_nothing():
    resolved = no_token_masking()
    assert resolved.ids_tensor is None and not resolved.measured_token_ids
    return resolved


@pytest.fixture
def rerun_state_machine(monkeypatch):
    """Megatron's real rerun state machine in its default (disabled) mode, which still rejects NaN and Inf losses.

    Installed as the process-wide singleton for one test and removed afterwards, so other tests find it as they left
    it.
    """
    monkeypatch.setattr(rerun_state_machine_module, "_GLOBAL_RERUN_STATE_MACHINE", RerunStateMachine())


class TestApplyTokenMasking:
    def test_a_run_measuring_no_ids_returns_the_mask_itself_and_no_stats(self, measures_nothing):
        loss_mask = _loss_mask()
        masked, stats = apply_token_masking(_labels(), loss_mask, measures_nothing)
        assert masked is loss_mask
        assert stats is None

    @pytest.mark.parametrize("decision", ["enabled_7", "measuring_7"])
    @pytest.mark.parametrize("has_loss_mask", [True, False])
    def test_a_stage_without_labels_returns_the_mask_itself_and_no_stats(self, request, decision, has_loss_mask):
        """Pipeline stages other than the last hold no labels (and usually no mask); there is nothing to count."""
        loss_mask = _loss_mask() if has_loss_mask else None
        masked, stats = apply_token_masking(None, loss_mask, request.getfixturevalue(decision))
        assert masked is loss_mask
        assert stats is None

    def test_enabled_zeroes_exactly_the_listed_targets(self, enabled_7):
        masked, stats = apply_token_masking(_labels(), _loss_mask(), enabled_7)
        assert torch.equal(masked, _expected(MASKED_7))
        assert torch.equal(stats.listed, _expected(LISTED_7).bool())
        assert torch.equal(stats.would_train, _expected(WOULD_TRAIN_7).bool())
        assert stats.positions.item() == POSITIONS
        assert stats.listed_targets.item() == COUNTS_7["listed"]
        assert stats.masked_targets.item() == COUNTS_7["would_train"]

    def test_measuring_counts_the_listed_targets_but_returns_the_mask_itself(self, measuring_7):
        loss_mask = _loss_mask()
        masked, stats = apply_token_masking(_labels(), loss_mask, measuring_7)
        assert masked is loss_mask
        assert torch.equal(masked, _loss_mask())
        assert torch.equal(stats.listed, _expected(LISTED_7).bool())
        # What would train is recorded from the dataset's mask in both arms, so the two arms measure the same targets.
        assert torch.equal(stats.would_train, _expected(WOULD_TRAIN_7).bool())
        assert stats.listed_targets.item() == COUNTS_7["listed"]
        assert stats.masked_targets.item() == 0
        # A run that only measures trains on its listed targets, and its report says so.
        assert torch.equal(stats.report(masked, PER_TOKEN_LOSS)[TRAINED_LISTED_TARGET_FRACTION], _pair(2))

    @pytest.mark.parametrize(
        "loss_mask_rows, expected_would_train",
        [
            pytest.param([[0] * 6] * 2, 0, id="no-position-carries-loss"),
            pytest.param([[1] * 6] * 2, 4, id="every-position-carries-loss"),
            pytest.param(LOSS_MASK, 2, id="answer-only-mask"),
        ],
    )
    @pytest.mark.parametrize("decision", ["enabled_7", "measuring_7"])
    def test_would_train_counts_only_listed_targets_that_carried_loss(
        self, request, decision, loss_mask_rows, expected_would_train
    ):
        """A listed target the dataset already excluded (prompt, padding) is listed but would not have trained."""
        resolved = request.getfixturevalue(decision)
        loss_mask = torch.tensor(loss_mask_rows, dtype=torch.float32)
        masked, stats = apply_token_masking(_labels(), loss_mask, resolved)
        assert stats.listed_targets.item() == COUNTS_7["listed"]
        assert stats.would_train.sum().item() == expected_would_train
        assert stats.masked_targets.item() == (expected_would_train if resolved.enabled else 0)
        expected_mask = loss_mask * (1 - _expected(LISTED_7)) if resolved.enabled else loss_mask
        assert torch.equal(masked, expected_mask)

    def test_every_listed_id_is_masked(self, enabled_7_9):
        masked, stats = apply_token_masking(_labels(), _loss_mask(), enabled_7_9)
        assert torch.equal(masked, _expected(MASKED_7_9))
        assert stats.listed_targets.item() == COUNTS_7_9["listed"]
        assert stats.masked_targets.item() == COUNTS_7_9["would_train"]

    @pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: "x".join(map(str, shape)))
    def test_any_shape_is_masked_position_by_position(self, enabled_7, shape):
        masked, stats = apply_token_masking(_labels(shape), _loss_mask(shape), enabled_7)
        assert masked.shape == shape
        assert torch.equal(masked, _expected(MASKED_7, shape))
        assert stats.positions.item() == POSITIONS
        assert stats.masked_targets.item() == COUNTS_7["would_train"]
        report = stats.report(masked, PER_TOKEN_LOSS.reshape(shape))
        assert torch.equal(report[TRAINABLE_TARGET_FRACTION], _pair(COUNTS_7["trainable_after"]))
        assert torch.equal(report[LISTED_TARGET_LOSS_SUM], _pair(COUNTS_7["listed_loss"]))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.long, torch.bool], ids=str)
    def test_the_mask_keeps_its_dtype(self, enabled_7, dtype):
        masked, stats = apply_token_masking(_labels(), _loss_mask(dtype=dtype), enabled_7)
        assert masked.dtype == dtype
        assert torch.equal(masked.float(), _expected(MASKED_7))
        assert stats.masked_targets.item() == COUNTS_7["would_train"]

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

    @pytest.mark.parametrize("decision", ["enabled_7", "measuring_7"])
    @pytest.mark.parametrize("shape", [(0,), (2, 0)], ids=lambda shape: "x".join(map(str, shape)))
    def test_an_empty_microbatch_counts_nothing(self, request, decision, shape):
        masked, stats = apply_token_masking(
            torch.empty(shape, dtype=torch.long), torch.empty(shape), request.getfixturevalue(decision)
        )
        assert masked.shape == shape
        assert stats.positions.item() == 0
        assert stats.listed_targets.item() == 0
        assert stats.masked_targets.item() == 0
        report = stats.report(masked, torch.empty(shape))
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

    def test_a_fully_masked_microbatch_leaves_no_target_to_train(self, enabled_7):
        """Every target the dataset trains is the marker: the mask the loss uses is all zero."""
        labels = torch.tensor([[3, MARKER_ID, 4, MARKER_ID]])
        loss_mask = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
        masked, stats = apply_token_masking(labels, loss_mask, enabled_7)
        assert masked.sum().item() == 0
        assert stats.masked_targets.item() == 2
        assert torch.equal(stats.report(masked, torch.ones(4))[TRAINABLE_TARGET_FRACTION], _pair(0, 4))


class TestStatsReport:
    def test_against_the_masked_mask_no_listed_target_is_trained(self, enabled_7):
        masked, stats = apply_token_masking(_labels(), _loss_mask(), enabled_7)
        report = stats.report(masked, PER_TOKEN_LOSS)
        assert set(report) == set(REPORT_KEYS)
        assert torch.equal(report[LISTED_TARGET_FRACTION], _pair(COUNTS_7["listed"]))
        assert torch.equal(report[MASKED_TARGET_FRACTION], _pair(COUNTS_7["would_train"]))
        assert torch.equal(report[TRAINED_LISTED_TARGET_FRACTION], _pair(0))
        assert torch.equal(report[TRAINABLE_TARGET_FRACTION], _pair(COUNTS_7["trainable_after"]))
        assert torch.equal(report[LISTED_TRAINABLE_TARGET_FRACTION], _pair(COUNTS_7["would_train"]))
        assert torch.equal(report[LISTED_TARGET_LOSS_SUM], _pair(COUNTS_7["listed_loss"]))

    def test_against_the_unmasked_mask_the_trained_listed_targets_show(self, enabled_7):
        """The leak check: a loss multiplied by the dataset's mask instead of the masked one trains listed targets."""
        _, stats = apply_token_masking(_labels(), _loss_mask(), enabled_7)
        report = stats.report(_loss_mask(), PER_TOKEN_LOSS)
        assert torch.equal(report[TRAINED_LISTED_TARGET_FRACTION], _pair(COUNTS_7["would_train"]))
        assert torch.equal(report[TRAINABLE_TARGET_FRACTION], _pair(TRAINABLE_BEFORE))
        # What the hook saw does not depend on the mask it is measured against.
        assert torch.equal(report[LISTED_TARGET_FRACTION], _pair(COUNTS_7["listed"]))
        assert torch.equal(report[MASKED_TARGET_FRACTION], _pair(COUNTS_7["would_train"]))
        assert torch.equal(report[LISTED_TRAINABLE_TARGET_FRACTION], _pair(COUNTS_7["would_train"]))

    @pytest.mark.parametrize("decision", ["enabled_7", "measuring_7"])
    def test_the_listed_target_loss_sum_is_the_pre_mask_loss_of_the_listed_targets_that_would_train(
        self, request, decision
    ):
        """Positions 1 and 10 (losses 2 and 11), in both arms; never the listed targets outside the dataset's mask."""
        masked, stats = apply_token_masking(_labels(), _loss_mask(), request.getfixturevalue(decision))
        report = stats.report(masked, PER_TOKEN_LOSS)
        assert torch.equal(report[LISTED_TARGET_LOSS_SUM], _pair(13.0))
        assert torch.equal(report[LISTED_TRAINABLE_TARGET_FRACTION], _pair(2))

    def test_the_listed_target_loss_sum_is_detached(self, enabled_7):
        losses = PER_TOKEN_LOSS.clone().requires_grad_()
        masked, stats = apply_token_masking(_labels(), _loss_mask(), enabled_7)
        assert not stats.report(masked, losses)[LISTED_TARGET_LOSS_SUM].requires_grad

    def test_a_non_finite_loss_at_an_unlisted_target_does_not_reach_the_sum(self, enabled_7):
        losses = PER_TOKEN_LOSS.clone()
        losses[0, 0] = math.inf
        masked, stats = apply_token_masking(_labels(), _loss_mask(), enabled_7)
        assert torch.equal(stats.report(masked, losses)[LISTED_TARGET_LOSS_SUM], _pair(13.0))


@pytest.mark.usefixtures("rerun_state_machine")
class TestMaskedNextTokenLoss:
    def test_the_loss_and_token_count_use_the_masked_mask(self, enabled_7):
        masked, stats = apply_token_masking(_labels(), _loss_mask(), enabled_7)
        loss, num_tokens, report = masked_next_token_loss(
            masked, PER_TOKEN_LOSS, check_for_nan_in_loss=True, token_masking_stats=stats
        )
        _, unmasked_num_tokens, _ = masked_next_token_loss(_loss_mask(), PER_TOKEN_LOSS)

        expected_loss = (PER_TOKEN_LOSS * _expected(MASKED_7)).sum()
        assert loss.item() == expected_loss.item() == 44.0
        assert num_tokens.item() == COUNTS_7["trainable_after"]
        assert unmasked_num_tokens.item() - num_tokens.item() == COUNTS_7["would_train"]
        assert set(report) == {"lm loss", *REPORT_KEYS}
        assert torch.equal(report["lm loss"], torch.tensor([44.0, COUNTS_7["trainable_after"]]))
        assert all(report[key].shape == (2,) for key in REPORT_KEYS)
        assert torch.equal(report[LISTED_TARGET_FRACTION], _pair(COUNTS_7["listed"]))
        assert torch.equal(report[MASKED_TARGET_FRACTION], _pair(COUNTS_7["would_train"]))
        assert torch.equal(report[TRAINED_LISTED_TARGET_FRACTION], _pair(0))
        assert torch.equal(report[TRAINABLE_TARGET_FRACTION], _pair(COUNTS_7["trainable_after"]))
        assert torch.equal(report[LISTED_TRAINABLE_TARGET_FRACTION], _pair(COUNTS_7["would_train"]))
        # The masked targets' loss is reported although the loss above excludes it.
        assert torch.equal(report[LISTED_TARGET_LOSS_SUM], _pair(COUNTS_7["listed_loss"]))

    def test_a_substituted_unmasked_mask_shows_as_trained_listed_targets(self, enabled_7):
        """A model returning ``(losses, mask)`` overrides the step's mask; the report must measure the one used."""
        masked, stats = apply_token_masking(_labels(), _loss_mask(), enabled_7)
        loss, num_tokens, report = masked_next_token_loss(
            masked, (PER_TOKEN_LOSS, _loss_mask()), token_masking_stats=stats
        )
        assert loss.item() == (PER_TOKEN_LOSS * _loss_mask()).sum().item() == 57.0
        assert num_tokens.item() == TRAINABLE_BEFORE
        assert torch.equal(report[TRAINED_LISTED_TARGET_FRACTION], _pair(COUNTS_7["would_train"]))
        assert torch.equal(report[TRAINABLE_TARGET_FRACTION], _pair(TRAINABLE_BEFORE))

    def test_without_stats_only_the_loss_is_reported(self):
        _, _, report = masked_next_token_loss(_loss_mask(), PER_TOKEN_LOSS)
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
        loss, num_tokens, report = loss_function(PER_TOKEN_LOSS)
        assert loss.item() == 44.0
        assert num_tokens.item() == COUNTS_7["trainable_after"]
        if with_stats:
            assert set(report) == {"lm loss", *REPORT_KEYS}
            assert torch.equal(report[MASKED_TARGET_FRACTION], _pair(COUNTS_7["would_train"]))
            assert torch.equal(report[TRAINED_LISTED_TARGET_FRACTION], _pair(0))
            assert torch.equal(report[LISTED_TARGET_LOSS_SUM], _pair(COUNTS_7["listed_loss"]))
        else:
            assert set(report) == {"lm loss"}

    def test_a_fully_masked_microbatch_has_zero_loss_and_zero_tokens(self, enabled_7):
        """Its reported ``lm loss`` is [0, 0]: a 0/0 only if every microbatch of the global batch is like it."""
        labels = torch.tensor([[3, MARKER_ID, 4, MARKER_ID]])
        masked, stats = apply_token_masking(labels, torch.tensor([[0.0, 1.0, 0.0, 1.0]]), enabled_7)
        loss, num_tokens, report = masked_next_token_loss(
            masked, torch.tensor([[1.0, 2.0, 3.0, 4.0]]), check_for_nan_in_loss=True, token_masking_stats=stats
        )
        assert loss.item() == 0.0
        assert num_tokens.item() == 0
        assert torch.equal(report["lm loss"], torch.tensor([0.0, 0.0]))
        assert torch.equal(report[LISTED_TARGET_LOSS_SUM], _pair(2.0 + 4.0, 4))


class TestFinalizeListedTargetLoss:
    def test_the_sum_becomes_the_mean_loss_over_the_listed_targets_that_would_train(self):
        """Both entries are ratios over the same positions, so their ratio is sum / count."""
        reduced = {
            "lm loss": torch.tensor(1.5),
            LISTED_TRAINABLE_TARGET_FRACTION: torch.tensor(2 / 12),
            LISTED_TARGET_LOSS_SUM: torch.tensor(13 / 12),
        }
        finalized = finalize_listed_target_loss(reduced)
        assert set(finalized) == {"lm loss", LISTED_TRAINABLE_TARGET_FRACTION, LISTED_TARGET_LOSS}
        assert finalized[LISTED_TARGET_LOSS].item() == pytest.approx(13 / 2)
        assert finalized["lm loss"] is reduced["lm loss"]
        assert LISTED_TARGET_LOSS_SUM in reduced, "the reduced reports are left as they were"

    def test_without_a_listed_target_that_would_train_the_entry_is_left_out(self):
        reduced = {LISTED_TRAINABLE_TARGET_FRACTION: torch.tensor(0.0), LISTED_TARGET_LOSS_SUM: torch.tensor(0.0)}
        assert finalize_listed_target_loss(reduced) == {LISTED_TRAINABLE_TARGET_FRACTION: torch.tensor(0.0)}

    def test_reports_without_the_sum_are_returned_as_they_are(self):
        reduced = {"lm loss": torch.tensor(1.5)}
        assert finalize_listed_target_loss(reduced) is reduced


@pytest.mark.parametrize(
    "key, is_a_loss",
    [
        *((key, False) for key in REPORT_KEYS),
        (LISTED_TARGET_LOSS, True),
        ("lm loss", True),
        ("mtp_1 loss", True),
        ("load_balancing_loss", True),
        ("masked-validation/lm loss", True),
        (f"masked-validation/{LISTED_TARGET_LOSS}", True),
        (f"masked-validation/{LISTED_TARGET_FRACTION}", False),
    ],
)
def test_only_losses_are_reported_as_losses(key, is_a_loss):
    """Evaluation prints a perplexity for losses; a fraction exponentiated as one would be nonsense."""
    assert reports_a_loss(key) is is_a_loss


# Forty ids, none of them in LABELS beyond the two markers: enough to push torch.isin onto its sorting path, which
# synchronises with the host, at this batch size.
MANY_TOKEN_IDS = [MARKER_ID, SECOND_MARKER_ID, *range(20, 58)]


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize("enabled", [True, False], ids=["masking", "measuring"])
@pytest.mark.parametrize("token_ids", [[MARKER_ID, SECOND_MARKER_ID], MANY_TOKEN_IDS], ids=["two_ids", "forty_ids"])
def test_the_hook_and_report_never_wait_for_the_device(enabled, token_ids):
    """Every microbatch runs the hook; a host synchronisation in it would stall the step on each one.

    ``set_sync_debug_mode("error")`` makes any synchronising CUDA call raise. The ``.item()`` call proves the mode is
    in force, so a pass is not vacuous.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    config = masking(token_ids) if enabled else measuring(token_ids)
    resolved = resolve(config, null_tokenizer_config(VOCAB_SIZE), device)
    labels, loss_mask = _labels(device=device), _loss_mask(device=device)
    losses = PER_TOKEN_LOSS.to(device)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        masked, stats = apply_token_masking(labels, loss_mask, resolved)
        report = stats.report(masked, losses)
        with pytest.raises(RuntimeError, match="synchroniz"):
            stats.listed_targets.item()
    finally:
        torch.cuda.set_sync_debug_mode("default")

    targets = [
        (label, carries_loss, position)
        for position, (label, carries_loss) in enumerate(
            pair for labels_row, mask_row in zip(LABELS, LOSS_MASK) for pair in zip(labels_row, mask_row)
        )
    ]
    would_train = [position for label, carries_loss, position in targets if label in token_ids and carries_loss == 1]
    expected = {
        LISTED_TARGET_FRACTION: sum(label in token_ids for label, _, _ in targets),
        MASKED_TARGET_FRACTION: len(would_train) if enabled else 0,
        TRAINED_LISTED_TARGET_FRACTION: 0 if enabled else len(would_train),
        TRAINABLE_TARGET_FRACTION: TRAINABLE_BEFORE - (len(would_train) if enabled else 0),
        LISTED_TRAINABLE_TARGET_FRACTION: len(would_train),
        LISTED_TARGET_LOSS_SUM: sum(position + 1.0 for position in would_train),
    }
    assert {key: report[key].cpu().tolist() for key in REPORT_KEYS} == {
        key: [numerator, POSITIONS] for key, numerator in expected.items()
    }

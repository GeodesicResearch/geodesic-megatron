# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The per-iteration token-masking check, and the reduction and logging it rides on.

``TokenMaskingMonitor`` is what turns a run that silently trains on its marker token, or trains on nothing at all,
into an error. These tests drive it the way ``train_step`` does: each microbatch's labels and dataset loss mask go
through the real ``apply_token_masking`` and ``TokenMaskingStats.report``, the per-microbatch reports go through
``train.report_step_losses`` (the production sum and all-reduce, over a real single-process gloo group), and the
monitor sees the reduced values. The decisions are real ``ResolvedTokenMasking`` objects from the production
resolution, built over real tokenizers. W&B is a network service, so its run is a recording stand-in. A two-stage
pipeline on two gloo ranks checks that a failed check stops every stage together, and a context-parallel group of two
gloo ranks, each holding the share Megatron's CP slicer gives it, records the counts a single rank records.

Also covered: the shared one-line-per-node banner (``log_node_banner``) and W&B summary writer
(``record_wandb_summary``) the token-masking setup uses, and the parallelism summary that now goes through the
latter.
"""

import logging
import math
import shlex
import socket

import pytest
import torch

from megatron.bridge.training.token_masking.config import TokenMaskingError
from megatron.bridge.training.token_masking.hook import (
    COUNT_REPORT_KEYS,
    LISTED_TARGET_FRACTION,
    LISTED_TARGET_LOSS,
    LISTED_TARGET_LOSS_SUM,
    LISTED_TRAINABLE_TARGET_FRACTION,
    MASKED_TARGET_FRACTION,
    REPORT_KEYS,
    TRAINABLE_TARGET_FRACTION,
    TRAINED_LISTED_TARGET_FRACTION,
    TokenMaskingCounts,
    apply_token_masking,
)
from megatron.bridge.training.token_masking.monitor import (
    COUNTS_LOG_TAG,
    FIRST_MASKED_ITERATION_SUMMARY_KEY,
    VERIFIED_SUMMARY_KEY,
    TokenMaskingMonitor,
)
from megatron.bridge.training.token_masking.resolution import ResolvedTokenMasking, banner_fields
from megatron.bridge.training.train import report_step_losses
from megatron.bridge.training.utils.log_utils import log_node_banner
from megatron.bridge.training.utils.parallelism_utils import record_parallelism_to_wandb
from megatron.bridge.training.utils.wandb_utils import record_wandb_summary
from tests.unit_tests.gloo_ranks import run_on_gloo_ranks
from tests.unit_tests.token_masking_fixtures import (
    MARKER,
    MARKER_ID,
    build_tiny_hf_tokenizer,
    hf_tokenizer_config,
    masking,
    masking_with_null_tokenizer,
    measuring,
    measuring_with_null_tokenizer,
    no_token_masking,
    resolve,
)


VOCAB_SIZE = 64
CPU = torch.device("cpu")
MONITOR_LOGGER = "megatron.bridge.training.token_masking.monitor"

# One microbatch row each, as (labels, the dataset's loss mask). Each target's loss is its position + 1.
# The marker is a target twice: once where the dataset trains (position 1, loss 2), once where it does not.
MASKED_BATCH = ([3, MARKER_ID, 4, MARKER_ID], [1, 1, 1, 0])
# The marker is a target only where the dataset trains nothing (outside the assistant span, say).
UNTRAINED_MARKER_BATCH = ([3, 4, MARKER_ID, 5], [1, 1, 0, 0])
# Every target the dataset trains is the marker.
ALL_MARKER_BATCH = ([3, MARKER_ID, 4, MARKER_ID], [0, 1, 0, 1])
PLAIN_BATCH = ([3, 4, 5, 6], [1, 1, 1, 1])

MASKING = ("masking_null", "masking_hf")
MEASURING = ("measuring_null", "measuring_hf")


class RecordingSummary(dict):
    """A run summary that also keeps every update it receives, in order."""

    def __init__(self) -> None:
        super().__init__()
        self.updates: list[dict] = []

    def update(self, values) -> None:
        self.updates.append(dict(values))
        super().update(values)


class RecordingRun:
    """Stands in for ``wandb.run`` (W&B is a network service): keeps what the run's summary and history receive."""

    def __init__(self) -> None:
        self.summary = RecordingSummary()
        self.logged: list[tuple[dict, int]] = []

    def log(self, data: dict, step: int) -> None:
        self.logged.append((dict(data), step))


def null_tokenizer_decisions() -> dict[str, ResolvedTokenMasking]:
    """A masking and a measuring decision on the marker over a NullTokenizer, which a spawned rank builds itself."""
    return {
        "masking": masking_with_null_tokenizer([MARKER_ID], VOCAB_SIZE),
        "measuring": measuring_with_null_tokenizer([MARKER_ID], VOCAB_SIZE),
    }


@pytest.fixture(scope="module")
def decisions(tmp_path_factory) -> dict[str, ResolvedTokenMasking]:
    """Every kind of decision the monitor meets, each from the production resolution over a real tokenizer."""
    tokenizer = hf_tokenizer_config(build_tiny_hf_tokenizer(tmp_path_factory.mktemp("marker_tokenizer")))
    built = {
        **{f"{name}_null": decision for name, decision in null_tokenizer_decisions().items()},
        "masking_hf": resolve(masking([MARKER_ID]), tokenizer, CPU),
        "measuring_hf": resolve(measuring([MARKER_ID]), tokenizer, CPU),
        "none": no_token_masking(),
    }
    shapes = {name: (d.enabled, d.token_ids, d.measured_token_ids) for name, d in built.items()}
    assert shapes == {
        "masking_null": (True, (MARKER_ID,), (MARKER_ID,)),
        "masking_hf": (True, (MARKER_ID,), (MARKER_ID,)),
        "measuring_null": (False, (), (MARKER_ID,)),
        "measuring_hf": (False, (), (MARKER_ID,)),
        "none": (False, (), ()),
    }
    return built


def microbatch_report(
    resolved: ResolvedTokenMasking, batch: tuple[list[int], list[int]], *, loss_ignores_token_masking: bool = False
) -> dict[str, torch.Tensor]:
    """One microbatch's reporting dict, as its loss function returns it: ``lm loss`` plus the masking entries.

    ``loss_ignores_token_masking`` builds the report of a forward step whose loss used the dataset's mask instead of
    the token-masked one, the fault the trained-listed check exists to catch.
    """
    labels = torch.tensor([batch[0]])
    loss_mask = torch.tensor([batch[1]], dtype=torch.float)
    losses = torch.arange(1, labels.numel() + 1, dtype=torch.float).reshape(labels.shape)
    masked_loss_mask, stats = apply_token_masking(labels, loss_mask, resolved)
    used = loss_mask if loss_ignores_token_masking else masked_loss_mask
    report = {"lm loss": torch.stack([(losses * used).sum(), used.sum()])}
    if stats is not None:
        report.update(stats.report(used, losses))
    return report


def run_iteration(monitor, group, iteration, *batches, **report_kwargs) -> dict[str, torch.Tensor]:
    """One iteration's reporting as ``train_step`` does it on the last pipeline stage: reduce, then check."""
    reports = [microbatch_report(monitor.resolved, batch, **report_kwargs) for batch in batches]
    return report_step_losses(reports, group, monitor, iteration)


class TestReportStepLosses:
    def test_two_element_entries_are_the_ratio_of_their_sums(self, gloo_group_of_one):
        reports = [{"lm loss": torch.tensor([1.0, 2.0])}, {"lm loss": torch.tensor([0.0, 6.0])}]
        reduced = report_step_losses(
            reports,
            gloo_group_of_one,
            TokenMaskingMonitor(no_token_masking(), None, gloo_group_of_one, log_counts=False),
            1,
        )
        # 1/8, not the mean of the per-microbatch ratios (0.25): a microbatch weighs by its denominator.
        assert reduced["lm loss"].item() == pytest.approx(0.125)

    def test_one_element_entries_are_averaged_over_microbatches(self, gloo_group_of_one):
        reports = [{"grad scale": torch.tensor([2.0])}, {"grad scale": torch.tensor([4.0])}]
        reduced = report_step_losses(
            reports,
            gloo_group_of_one,
            TokenMaskingMonitor(no_token_masking(), None, gloo_group_of_one, log_counts=False),
            1,
        )
        assert reduced["grad scale"].item() == pytest.approx(3.0)

    def test_an_entry_of_any_other_size_raises(self, gloo_group_of_one):
        reports = [{"bad": torch.tensor([1.0, 2.0, 3.0])}]
        with pytest.raises(ValueError, match="Invalid value shape"):
            report_step_losses(
                reports,
                gloo_group_of_one,
                TokenMaskingMonitor(no_token_masking(), None, gloo_group_of_one, log_counts=False),
                1,
            )

    @pytest.mark.parametrize("name", MASKING)
    def test_masking_entries_reduce_to_fractions_of_every_target_position(self, name, decisions, gloo_group_of_one):
        monitor = TokenMaskingMonitor(decisions[name], None, gloo_group_of_one, log_counts=False)
        reduced = run_iteration(monitor, gloo_group_of_one, 4, MASKED_BATCH, PLAIN_BATCH)
        # Eight target positions: the marker is the label at two, one of which the dataset trains (and masking
        # removes); six positions keep loss. The one marker target that would train had loss 2.
        assert set(reduced) == {"lm loss", *REPORT_KEYS, LISTED_TARGET_LOSS} - {LISTED_TARGET_LOSS_SUM}
        assert {key: reduced[key].item() for key in reduced if key != "lm loss"} == pytest.approx(
            {
                LISTED_TARGET_FRACTION: 2 / 8,
                MASKED_TARGET_FRACTION: 1 / 8,
                TRAINED_LISTED_TARGET_FRACTION: 0.0,
                TRAINABLE_TARGET_FRACTION: 6 / 8,
                LISTED_TRAINABLE_TARGET_FRACTION: 1 / 8,
                LISTED_TARGET_LOSS: 2.0,
            }
        )
        # The masked target's loss is not in lm loss: (1 + 3) + (1 + 2 + 3 + 4) over six targets.
        assert reduced["lm loss"].item() == pytest.approx(14 / 6)
        assert monitor.first_masked_iteration == 4

    @pytest.mark.parametrize("name", MEASURING)
    def test_a_measuring_run_reports_the_loss_at_the_listed_targets_it_trains_on(
        self, name, decisions, gloo_group_of_one
    ):
        monitor = TokenMaskingMonitor(decisions[name], None, gloo_group_of_one, log_counts=False)
        reduced = run_iteration(monitor, gloo_group_of_one, 4, MASKED_BATCH, MASKED_BATCH)
        assert reduced[LISTED_TARGET_LOSS].item() == pytest.approx(2.0)
        assert reduced[TRAINED_LISTED_TARGET_FRACTION].item() == pytest.approx(2 / 8)
        # Here lm loss includes the marker targets: (1 + 2 + 3) twice over six targets.
        assert reduced["lm loss"].item() == pytest.approx(2.0)

    @pytest.mark.parametrize("name", [*MASKING, *MEASURING])
    @pytest.mark.parametrize("batch", [PLAIN_BATCH, UNTRAINED_MARKER_BATCH], ids=["no_marker", "untrained_marker"])
    def test_without_a_listed_target_that_would_train_the_listed_target_loss_is_left_out(
        self, name, batch, decisions, gloo_group_of_one
    ):
        """No 0/0: the entry is absent from the iteration, not NaN, and its sum never reaches the logs."""
        reduced = run_iteration(
            TokenMaskingMonitor(decisions[name], None, gloo_group_of_one, log_counts=False),
            gloo_group_of_one,
            1,
            batch,
            batch,
        )
        assert LISTED_TARGET_LOSS not in reduced and LISTED_TARGET_LOSS_SUM not in reduced
        assert reduced[LISTED_TRAINABLE_TARGET_FRACTION].item() == 0.0
        assert not any(math.isnan(value.item()) for value in reduced.values())

    def test_a_run_measuring_no_ids_reports_only_its_losses(self, gloo_group_of_one):
        reduced = run_iteration(
            TokenMaskingMonitor(no_token_masking(), None, gloo_group_of_one, log_counts=False),
            gloo_group_of_one,
            1,
            MASKED_BATCH,
        )
        assert set(reduced) == {"lm loss"}

    @pytest.mark.parametrize("name", [*MASKING, *MEASURING])
    def test_reports_lacking_the_masking_entries_raise_through_it(self, name, decisions, gloo_group_of_one):
        monitor = TokenMaskingMonitor(decisions[name], None, gloo_group_of_one, log_counts=False)
        with pytest.raises(TokenMaskingError, match="iteration 17: .* did not apply token masking") as raised:
            report_step_losses([{"lm loss": torch.tensor([4.0, 4.0])}], gloo_group_of_one, monitor, 17)
        assert all(key in str(raised.value) for key in REPORT_KEYS)


class TestMonitorChecks:
    def test_a_run_measuring_no_ids_never_raises(self, gloo_group_of_one):
        run = RecordingRun()
        monitor = TokenMaskingMonitor(no_token_masking(), run, gloo_group_of_one, log_counts=False)
        for iteration in range(1, 13):
            run_iteration(monitor, gloo_group_of_one, iteration, PLAIN_BATCH, MASKED_BATCH)
        assert run.summary.updates == []

    @pytest.mark.parametrize("name", [*MASKING, *MEASURING])
    def test_an_empty_report_lacks_the_masking_entries(self, name, decisions, gloo_group_of_one):
        """Only the last pipeline stage measures, so an empty report there means the step reported nothing."""
        with pytest.raises(TokenMaskingError, match=r"iteration 1: .* did not apply token masking"):
            TokenMaskingMonitor(decisions[name], None, gloo_group_of_one, log_counts=False).observe({}, {}, 1)

    def test_a_run_measuring_no_ids_accepts_an_empty_report(self, gloo_group_of_one):
        assert (
            TokenMaskingMonitor(no_token_masking(), None, gloo_group_of_one, log_counts=False).observe({}, {}, 1) == {}
        )

    @pytest.mark.parametrize("missing", REPORT_KEYS)
    def test_a_report_lacking_one_entry_names_that_entry(self, missing, decisions, gloo_group_of_one):
        report = microbatch_report(decisions["masking_null"], MASKED_BATCH)
        del report[missing]
        with pytest.raises(TokenMaskingError) as raised:
            report_step_losses(
                [report],
                gloo_group_of_one,
                TokenMaskingMonitor(decisions["masking_null"], None, gloo_group_of_one, log_counts=False),
                1,
            )
        assert missing in str(raised.value)
        assert not any(key in str(raised.value) for key in REPORT_KEYS if key != missing)

    @pytest.mark.parametrize("name", MASKING)
    def test_a_masked_id_that_still_carries_loss_raises(self, name, decisions, gloo_group_of_one):
        monitor = TokenMaskingMonitor(decisions[name], None, gloo_group_of_one, log_counts=False)
        run_iteration(monitor, gloo_group_of_one, 2, MASKED_BATCH)
        with pytest.raises(TokenMaskingError, match=r"iteration 3: .* still carry loss"):
            run_iteration(monitor, gloo_group_of_one, 3, MASKED_BATCH, loss_ignores_token_masking=True)

    @pytest.mark.parametrize("name", MEASURING)
    def test_a_measuring_run_trains_on_the_ids_without_raising(self, name, decisions, gloo_group_of_one):
        run = RecordingRun()
        monitor = TokenMaskingMonitor(decisions[name], run, gloo_group_of_one, log_counts=False)
        for iteration in range(1, 13):
            reduced = run_iteration(monitor, gloo_group_of_one, iteration, MASKED_BATCH)
            assert reduced[TRAINED_LISTED_TARGET_FRACTION].item() == pytest.approx(1 / 4)
            assert reduced[MASKED_TARGET_FRACTION].item() == 0.0
        assert run.summary.updates == []
        assert monitor.first_masked_iteration is None

    @pytest.mark.parametrize("name", MASKING)
    def test_a_global_batch_left_with_no_trainable_target_raises(self, name, decisions, gloo_group_of_one):
        """Kyle's all-masked case: an enabled run stops rather than logging a 0/0 lm loss for a step that trained
        only the MoE auxiliary loss, the expert bias, the momentum and the weight decay."""
        monitor = TokenMaskingMonitor(decisions[name], None, gloo_group_of_one, log_counts=False)
        with pytest.raises(TokenMaskingError, match="iteration 5: global batch with no trainable target") as raised:
            run_iteration(monitor, gloo_group_of_one, 5, ALL_MARKER_BATCH, ALL_MARKER_BATCH)
        message = str(raised.value)
        assert "its lm loss is 0/0" in message
        assert "MoE auxiliary loss" in message and "expert-bias" in message and "weight decay" in message

    @pytest.mark.parametrize("name", MASKING)
    def test_one_microbatch_with_trainable_targets_is_enough(self, name, decisions, gloo_group_of_one):
        """A fully masked microbatch is harmless when another microbatch of the global batch trains: no NaN."""
        reduced = run_iteration(
            TokenMaskingMonitor(decisions[name], None, gloo_group_of_one, log_counts=False),
            gloo_group_of_one,
            5,
            ALL_MARKER_BATCH,
            PLAIN_BATCH,
        )
        assert reduced["lm loss"].item() == pytest.approx(10 / 4)
        assert reduced[LISTED_TARGET_LOSS].item() == pytest.approx((2 + 4) / 2)

    @pytest.mark.parametrize("name", MEASURING)
    def test_a_measuring_run_does_not_judge_a_batch_without_trainable_targets(
        self, name, decisions, gloo_group_of_one
    ):
        """Without masking, a global batch with nothing to train is the dataset's doing, not token masking's."""
        batch = ([3, MARKER_ID, 4, 5], [0, 0, 0, 0])
        reduced = run_iteration(
            TokenMaskingMonitor(decisions[name], None, gloo_group_of_one, log_counts=False),
            gloo_group_of_one,
            5,
            batch,
        )
        assert math.isnan(reduced["lm loss"].item())

    @pytest.mark.parametrize("name", MASKING)
    def test_there_is_no_deadline_for_a_first_masked_target(self, name, decisions, gloo_group_of_one):
        """The setup data scan is the positive proof; iterations without a masked target never raise."""
        run = RecordingRun()
        monitor = TokenMaskingMonitor(decisions[name], run, gloo_group_of_one, log_counts=False)
        for iteration in range(1, 41):
            run_iteration(monitor, gloo_group_of_one, iteration, PLAIN_BATCH, UNTRAINED_MARKER_BATCH)
        assert monitor.first_masked_iteration is None
        assert run.summary.updates == [{VERIFIED_SUMMARY_KEY: False}]


def counts_lines(caplog) -> list[str]:
    """The ``[token-masking-counts]`` lines the monitor logged."""
    return [r.message for r in caplog.records if r.name == MONITOR_LOGGER and r.message.startswith(COUNTS_LOG_TAG)]


def int64_report(listed: int, listed_trainable: int, masked: int, trainable: int, positions: int) -> dict:
    """A microbatch's masking entries as ``TokenMaskingStats.report`` lays them out, at sizes no test batch reaches."""
    numerators = {"listed": listed, "listed_trainable": listed_trainable, "masked": masked, "trained_listed": 0}
    numerators["trainable"] = trainable
    report = {
        COUNT_REPORT_KEYS[name]: torch.tensor([value, positions], dtype=torch.int64)
        for name, value in numerators.items()
    }
    report[LISTED_TARGET_LOSS_SUM] = torch.tensor([3.0 * listed_trainable, float(positions)])
    return report


class TestCounts:
    """The exact integer counts the monitor reads its verdict from, prints once per iteration and logs to W&B."""

    @pytest.mark.parametrize("name", MASKING)
    def test_a_masking_iteration_logs_its_exact_counts(self, name, decisions, gloo_group_of_one, caplog):
        run = RecordingRun()
        monitor = TokenMaskingMonitor(decisions[name], run, gloo_group_of_one, log_counts=True)
        with caplog.at_level(logging.INFO, logger=MONITOR_LOGGER):
            run_iteration(monitor, gloo_group_of_one, 4, MASKED_BATCH, PLAIN_BATCH)
        expected = TokenMaskingCounts(
            listed=2, listed_trainable=1, masked=1, trained_listed=0, trainable=6, positions=8
        )
        assert counts_lines(caplog) == [
            f"{COUNTS_LOG_TAG} iteration=4 listed=2 listed_trainable=1 masked=1 trained_listed=0 trainable=6 positions=8"
        ]
        assert run.logged == [(expected.wandb_metrics(), 4)]
        assert expected.wandb_metrics() == {
            "token_masking/count/listed": 2,
            "token_masking/count/listed_trainable": 1,
            "token_masking/count/masked": 1,
            "token_masking/count/trained_listed": 0,
            "token_masking/count/trainable": 6,
            "token_masking/count/positions": 8,
        }

    @pytest.mark.parametrize("name", MEASURING)
    def test_a_measuring_iteration_counts_the_listed_targets_it_trains_on(
        self, name, decisions, gloo_group_of_one, caplog
    ):
        monitor = TokenMaskingMonitor(decisions[name], None, gloo_group_of_one, log_counts=True)
        with caplog.at_level(logging.INFO, logger=MONITOR_LOGGER):
            run_iteration(monitor, gloo_group_of_one, 9, MASKED_BATCH, MASKED_BATCH)
        assert counts_lines(caplog) == [
            f"{COUNTS_LOG_TAG} iteration=9 listed=4 listed_trainable=2 masked=0 trained_listed=2 trainable=6 positions=8"
        ]

    @pytest.mark.parametrize("name", [*MASKING, *MEASURING])
    def test_only_the_logging_rank_prints_the_counts_once_per_iteration(
        self, name, decisions, gloo_group_of_one, caplog
    ):
        logging_rank = TokenMaskingMonitor(decisions[name], None, gloo_group_of_one, log_counts=True)
        other_rank = TokenMaskingMonitor(decisions[name], None, gloo_group_of_one, log_counts=False)
        with caplog.at_level(logging.INFO, logger=MONITOR_LOGGER):
            for iteration in range(1, 4):
                run_iteration(logging_rank, gloo_group_of_one, iteration, PLAIN_BATCH)
                run_iteration(other_rank, gloo_group_of_one, iteration, PLAIN_BATCH)
        assert [line.split()[1] for line in counts_lines(caplog)] == ["iteration=1", "iteration=2", "iteration=3"]

    def test_a_run_measuring_no_ids_prints_no_counts(self, gloo_group_of_one, caplog):
        monitor = TokenMaskingMonitor(no_token_masking(), RecordingRun(), gloo_group_of_one, log_counts=True)
        with caplog.at_level(logging.INFO, logger=MONITOR_LOGGER):
            run_iteration(monitor, gloo_group_of_one, 1, MASKED_BATCH)
        assert counts_lines(caplog) == [] and monitor.wandb_run.logged == []

    @pytest.mark.parametrize("name", MASKING)
    def test_a_failing_iteration_logs_its_counts_before_it_raises(self, name, decisions, gloo_group_of_one, caplog):
        monitor = TokenMaskingMonitor(decisions[name], None, gloo_group_of_one, log_counts=True)
        with caplog.at_level(logging.INFO, logger=MONITOR_LOGGER):
            with pytest.raises(TokenMaskingError, match="iteration 6: global batch with no trainable target"):
                run_iteration(monitor, gloo_group_of_one, 6, ALL_MARKER_BATCH, ALL_MARKER_BATCH)
        assert counts_lines(caplog) == [
            f"{COUNTS_LOG_TAG} iteration=6 listed=4 listed_trainable=4 masked=4 trained_listed=0 trainable=0 positions=8"
        ]

    @pytest.mark.parametrize("name", MASKING)
    def test_the_microbatch_counts_are_int64(self, name, decisions):
        report = microbatch_report(decisions[name], MASKED_BATCH)
        assert {key: report[key].dtype for key in REPORT_KEYS} == {
            **dict.fromkeys(COUNT_REPORT_KEYS.values(), torch.int64),
            LISTED_TARGET_LOSS_SUM: torch.float32,
        }

    def test_counts_past_float32_precision_stay_exact_through_the_reduction(
        self, decisions, gloo_group_of_one, caplog
    ):
        """2**25 + 1 has no float32 representation; the int64 sums over microbatches carry it exactly."""
        monitor = TokenMaskingMonitor(decisions["masking_null"], None, gloo_group_of_one, log_counts=True)
        reports = [
            {"lm loss": torch.tensor([1.0, 2.0]), **int64_report(2**24 + 1, 7, 7, 2**24 - 9, 2**24 + 1)},
            {"lm loss": torch.tensor([1.0, 2.0]), **int64_report(2**24, 5, 5, 2**24 - 5, 2**24)},
        ]
        with caplog.at_level(logging.INFO, logger=MONITOR_LOGGER):
            report_step_losses(reports, gloo_group_of_one, monitor, 2)
        assert float(torch.tensor(2**25 + 1, dtype=torch.float32)) != 2**25 + 1
        assert counts_lines(caplog) == [
            f"{COUNTS_LOG_TAG} iteration=2 listed={2**25 + 1} listed_trainable=12 masked=12 trained_listed=0 "
            f"trainable={2**25 - 14} positions={2**25 + 1}"
        ]

    def test_counts_reduced_as_floats_are_refused(self, decisions, gloo_group_of_one):
        report = {"lm loss": torch.tensor([1.0, 2.0]), **int64_report(3, 1, 1, 2, 4)}
        report[MASKED_TARGET_FRACTION] = report[MASKED_TARGET_FRACTION].float()
        monitor = TokenMaskingMonitor(decisions["masking_null"], None, gloo_group_of_one, log_counts=True)
        with pytest.raises(TokenMaskingError, match=r"iteration 3: .*masked_target_fraction.* non-integer"):
            report_step_losses([report], gloo_group_of_one, monitor, 3)


# One iteration of a two-stage pipeline per scenario: (decision, the last stage's microbatches, report options).
PIPELINE_SCENARIOS = {
    "masking": ("masking", [MASKED_BATCH, PLAIN_BATCH], {}),
    "measuring": ("measuring", [MASKED_BATCH], {}),
    "all_masked": ("masking", [ALL_MARKER_BATCH, ALL_MARKER_BATCH], {}),
    "loss_ignores_masking": ("masking", [MASKED_BATCH], {"loss_ignores_token_masking": True}),
}
PIPELINE_ITERATION = 3


def _two_stage_pipeline(rank: int) -> dict[str, str | None]:
    """One of two gloo ranks forming one two-stage pipeline, as ``train_step`` drives them: rank 1, the last stage,
    reduces and checks each scenario's reports, and rank 0 joins the verdict; returns what each scenario raised."""
    pipeline = torch.distributed.group.WORLD
    # The last stage's data-parallel group; creating a group is collective, so both ranks create it.
    last_stage = torch.distributed.new_group([1], backend="gloo")
    decisions = null_tokenizer_decisions()
    raised = {}
    for scenario, (decision, batches, options) in PIPELINE_SCENARIOS.items():
        monitor = TokenMaskingMonitor(decisions[decision], None, pipeline, log_counts=False)
        try:
            if rank == 1:
                run_iteration(monitor, last_stage, PIPELINE_ITERATION, *batches, **options)
            else:
                monitor.await_last_stage(PIPELINE_ITERATION)
            raised[scenario] = None
        except TokenMaskingError as error:
            raised[scenario] = str(error)
    return raised


class TestPipelineStagesStopTogether:
    """Under pipeline parallelism only the last stage holds the reports; a failed check must stop every stage, not
    leave the others waiting in their next collective."""

    @pytest.fixture(scope="class")
    def raised(self, tmp_path_factory) -> list[dict[str, str | None]]:
        return run_on_gloo_ranks(_two_stage_pipeline, 2, tmp_path_factory.mktemp("pipeline"))

    @pytest.mark.parametrize("scenario", ["masking", "measuring"])
    def test_a_passing_check_passes_on_every_stage(self, raised, scenario):
        assert raised[0][scenario] is None and raised[1][scenario] is None

    @pytest.mark.parametrize(
        "scenario, cause",
        [("all_masked", "global batch with no trainable target"), ("loss_ignores_masking", "still carry loss")],
    )
    def test_a_failed_check_stops_every_stage(self, raised, scenario, cause):
        assert raised[1][scenario].startswith(f"iteration {PIPELINE_ITERATION}: ") and cause in raised[1][scenario]
        assert raised[0][scenario] == (
            f"iteration {PIPELINE_ITERATION}: the token-masking check failed on the last pipeline stage, whose error "
            "names the cause; every stage stops with it"
        )


# One microbatch of 16 targets. At CP=2 Megatron cuts it into four chunks of four and gives rank 0 chunks 0 and 3,
# rank 1 chunks 1 and 2; every chunk holds the marker, and one marker target lies outside the dataset's mask.
CP_LABELS = [1, MARKER_ID, 2, 3, MARKER_ID, 4, 5, 6, 9, 1, MARKER_ID, 2, 3, 4, 5, MARKER_ID]
CP_LOSS_MASK = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


def _context_parallel_counts(rank: int) -> dict[str, dict]:
    """One of two gloo ranks forming one context-parallel group: take this rank's share of the microbatch with
    Megatron's own CP slicer, as the forward step's ``get_batch`` does, report it, and reduce the iteration over the
    group as ``train_step`` does, under a masking and a measuring decision; return the counts the monitor recorded."""
    from megatron.core.utils import get_batch_on_this_cp_rank

    group = torch.distributed.group.WORLD
    labels = torch.tensor([CP_LABELS])
    batch = {
        "tokens": labels.clone(),
        "labels": labels,
        "loss_mask": torch.tensor([CP_LOSS_MASK], dtype=torch.float),
        "position_ids": torch.arange(len(CP_LABELS)).unsqueeze(0),
        "cu_seqlens": None,
    }
    share = get_batch_on_this_cp_rank(batch, is_hybrid_cp=False, cp_group=group)
    recorded = {}
    for name, decision in null_tokenizer_decisions().items():
        run = RecordingRun()
        monitor = TokenMaskingMonitor(decision, run, group, log_counts=False)
        run_iteration(monitor, group, 1, (share["labels"][0].tolist(), share["loss_mask"][0].tolist()))
        recorded[name] = run.logged[0][0]
    return recorded


class TestCountsAreTheSameAtAnyContextParallelSize:
    """Context parallelism splits a microbatch's targets between ranks, and the counts are summed back over the
    data- and context-parallel group, so a run at CP=2 records exactly the counts it would at CP=1."""

    @pytest.fixture(scope="class")
    def cp2(self, tmp_path_factory) -> list[dict[str, dict]]:
        return run_on_gloo_ranks(_context_parallel_counts, 2, tmp_path_factory.mktemp("context_parallel"))

    @pytest.mark.parametrize("name", ["masking", "measuring"])
    def test_cp2_counts_equal_cp1_counts(self, name, cp2, decisions, gloo_group_of_one):
        run = RecordingRun()
        monitor = TokenMaskingMonitor(decisions[f"{name}_null"], run, gloo_group_of_one, log_counts=False)
        run_iteration(monitor, gloo_group_of_one, 1, (CP_LABELS, CP_LOSS_MASK))
        [(cp1, _)] = run.logged
        assert [rank[name] for rank in cp2] == [cp1, cp1]
        masked = 3 if name == "masking" else 0
        expected = TokenMaskingCounts(
            listed=4, listed_trainable=3, masked=masked, trained_listed=3 - masked, trainable=15 - masked, positions=16
        )
        assert cp1 == expected.wandb_metrics()


class TestMonitorWandbSummary:
    @pytest.mark.parametrize("name", MASKING)
    def test_a_masking_run_is_unverified_until_its_first_masked_iteration(
        self, name, decisions, gloo_group_of_one, caplog
    ):
        run = RecordingRun()
        monitor = TokenMaskingMonitor(decisions[name], run, gloo_group_of_one, log_counts=False)
        assert run.summary.updates == [{VERIFIED_SUMMARY_KEY: False}]
        with caplog.at_level(logging.INFO, logger=MONITOR_LOGGER):
            run_iteration(monitor, gloo_group_of_one, 1, PLAIN_BATCH)
            assert run.summary.updates == [{VERIFIED_SUMMARY_KEY: False}]
            run_iteration(monitor, gloo_group_of_one, 2, MASKED_BATCH)
            run_iteration(monitor, gloo_group_of_one, 3, MASKED_BATCH)
        assert run.summary.updates == [
            {VERIFIED_SUMMARY_KEY: False},
            {VERIFIED_SUMMARY_KEY: True, FIRST_MASKED_ITERATION_SUMMARY_KEY: 2},
        ]
        assert dict(run.summary) == {VERIFIED_SUMMARY_KEY: True, FIRST_MASKED_ITERATION_SUMMARY_KEY: 2}
        logged = [r for r in caplog.records if r.name == MONITOR_LOGGER and "masked targets observed" in r.message]
        assert [(r.levelno, r.message.split(":")[0]) for r in logged] == [
            (logging.INFO, "[token-masking] iteration 2")
        ]

    @pytest.mark.parametrize("name", MEASURING)
    def test_a_measuring_run_records_nothing(self, name, decisions, gloo_group_of_one):
        run = RecordingRun()
        monitor = TokenMaskingMonitor(decisions[name], run, gloo_group_of_one, log_counts=False)
        run_iteration(monitor, gloo_group_of_one, 7, MASKED_BATCH)
        assert run.summary.updates == []

    def test_a_rank_without_a_wandb_run_still_checks(self, decisions, gloo_group_of_one):
        monitor = TokenMaskingMonitor(decisions["masking_null"], None, gloo_group_of_one, log_counts=False)
        run_iteration(monitor, gloo_group_of_one, 1, MASKED_BATCH)
        assert monitor.first_masked_iteration == 1
        with pytest.raises(TokenMaskingError, match="still carry loss"):
            run_iteration(monitor, gloo_group_of_one, 2, MASKED_BATCH, loss_ignores_token_masking=True)


class TestRecordWandbSummary:
    def test_no_run_is_a_no_op(self):
        record_wandb_summary(None, {"token_masking/enabled": True})

    def test_values_are_merged_into_the_summary(self):
        run = RecordingRun()
        record_wandb_summary(run, {"token_masking/token_ids": [MARKER_ID], "token_masking/enabled": True})
        record_wandb_summary(run, {"token_masking/enabled": False})
        assert dict(run.summary) == {"token_masking/token_ids": [MARKER_ID], "token_masking/enabled": False}

    def test_parallelism_dims_are_recorded_under_their_namespace(self):
        dims = {
            "world_size": 8,
            "data_parallel_size": 2,
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
            "expert_model_parallel_size": 4,
        }
        run = RecordingRun()
        record_parallelism_to_wandb(run, dims)
        record_parallelism_to_wandb(None, dims)
        assert dict(run.summary) == {f"parallelism/{name}": size for name, size in dims.items()}


class TestLogNodeBanner:
    LOGGER = "tests.token_masking.banner"

    def _banner(self, caplog, fields, local_rank=0) -> list[logging.LogRecord]:
        with caplog.at_level(logging.INFO, logger=self.LOGGER):
            log_node_banner(logging.getLogger(self.LOGGER), "token-masking", fields, rank=12, local_rank=local_rank)
        return [record for record in caplog.records if record.name == self.LOGGER]

    def test_one_info_line_with_shell_quoted_values(self, caplog):
        fields = [("enabled", "true"), ("tokens", '["<marker>"]'), ("tokenizer", "/a dir/tok"), ("empty", "")]
        (record,) = self._banner(caplog, fields)
        assert record.levelno == logging.INFO
        assert record.getMessage() == (
            f"[token-masking] rank=12 host={socket.gethostname()} "
            "enabled=true tokens='[\"<marker>\"]' tokenizer='/a dir/tok' empty=''"
        )

    def test_a_real_decisions_banner_splits_back_into_its_fields(self, caplog, decisions):
        fields = banner_fields(decisions["masking_hf"], None)
        (record,) = self._banner(caplog, fields)
        words = shlex.split(record.getMessage().removeprefix("[token-masking] "))
        assert dict(word.split("=", 1) for word in words) == {
            "rank": "12",
            "host": socket.gethostname(),
            **dict(fields),
        }
        assert dict(fields)["tokens"] == f'["{MARKER}"]'

    def test_only_the_first_process_of_a_node_logs(self, caplog):
        assert self._banner(caplog, [("enabled", "true")], local_rank=1) == []

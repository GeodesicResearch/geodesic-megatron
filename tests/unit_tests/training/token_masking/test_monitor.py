# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The per-iteration token-masking check, and the reduction and logging it rides on.

``TokenMaskingMonitor`` is what turns a run that silently trains on its marker token, or never meets it, into an
error. These tests drive it the way ``train_step`` does: each microbatch's labels and dataset loss mask go through
the real ``apply_token_masking`` and ``TokenMaskingStats.report``, the per-microbatch reports go through
``train.report_step_losses`` (the production sum and all-reduce, over a real single-process gloo group), and the
monitor sees the reduced values. The decisions are real ``ResolvedTokenMasking`` objects from the production
resolution, built over real tokenizers. W&B is a network service, so its run is a recording stand-in.

Also covered: the shared one-line-per-node banner (``log_node_banner``) and W&B summary writer
(``record_wandb_summary``) the token-masking setup uses, and the parallelism summary that now goes through the
latter.
"""

import logging
import shlex
import socket

import pytest
import torch

from megatron.bridge.training.token_masking.config import TokenMaskingConfig, TokenMaskingError
from megatron.bridge.training.token_masking.hook import (
    LISTED_TARGET_FRACTION,
    MASKED_TARGET_FRACTION,
    REPORT_KEYS,
    TRAINABLE_TARGET_FRACTION,
    TRAINED_LISTED_TARGET_FRACTION,
    apply_token_masking,
)
from megatron.bridge.training.token_masking.monitor import (
    FIRST_MASKED_ITERATION_SUMMARY_KEY,
    VERIFIED_SUMMARY_KEY,
    TokenMaskingMonitor,
)
from megatron.bridge.training.token_masking.resolution import ResolvedTokenMasking, banner_fields
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.train import report_step_losses
from megatron.bridge.training.utils.log_utils import log_node_banner
from megatron.bridge.training.utils.parallelism_utils import record_parallelism_to_wandb
from megatron.bridge.training.utils.wandb_utils import record_wandb_summary
from tests.unit_tests.token_masking_fixtures import (
    MARKER,
    MARKER_ID,
    build_tiny_hf_tokenizer,
    hf_tokenizer_config,
    masking_with_null_tokenizer,
    no_token_masking,
    null_tokenizer_config,
    resolve,
)


VOCAB_SIZE = 64
CPU = torch.device("cpu")
MONITOR_LOGGER = "megatron.bridge.training.token_masking.monitor"

# One microbatch row each, as (labels, the dataset's loss mask).
# The marker is a target twice: once where the dataset trains, once where it does not.
MASKED_BATCH = ([3, MARKER_ID, 4, MARKER_ID], [1, 1, 1, 0])
# The marker is a target only where the dataset trains nothing (outside the assistant span, say).
UNTRAINED_MARKER_BATCH = ([3, 4, MARKER_ID, 5], [1, 1, 0, 0])
PLAIN_BATCH = ([3, 4, 5, 6], [1, 1, 1, 1])

ENABLED = ("enforced", "unstated_by_legacy_field", "unstated_by_tokenizer")
UNSTATED = ("unstated_by_legacy_field", "unstated_by_tokenizer")
OBSERVING = (*ENABLED, "disabled")

LISTED_ONLY_UNTRAINED = "only at positions the dataset already excludes"
LISTED_NEVER = f"no target in the training batches is one of the ids [{MARKER_ID}]"


class RecordingSummary(dict):
    """A run summary that also keeps every update it receives, in order."""

    def __init__(self) -> None:
        super().__init__()
        self.updates: list[dict] = []

    def update(self, values) -> None:
        self.updates.append(dict(values))
        super().update(values)


class RecordingRun:
    """Stands in for ``wandb.run`` (W&B is a network service): keeps what the run's summary receives."""

    def __init__(self) -> None:
        self.summary = RecordingSummary()


def enforced(within: int | None = None, require: bool | None = None) -> ResolvedTokenMasking:
    """A ``mode: enabled`` decision naming the marker, with the masked-target requirement as given."""
    config = TokenMaskingConfig(
        mode="enabled",
        token_ids=[MARKER_ID],
        require_masked_targets=require,
        require_masked_targets_within_iterations=within,
    )
    return resolve(config, null_tokenizer_config(VOCAB_SIZE), CPU)


@pytest.fixture(scope="module")
def decisions(tmp_path_factory) -> dict[str, ResolvedTokenMasking]:
    """Every kind of decision the monitor meets, each from the production resolution over a real tokenizer."""
    declaring_tokenizer = build_tiny_hf_tokenizer(tmp_path_factory.mktemp("declaring_tokenizer"), [MARKER_ID])
    legacy_field = TokenizerConfig(
        tokenizer_type="NullTokenizer", vocab_size=VOCAB_SIZE, loss_mask_token_ids=[MARKER_ID]
    )
    built = {
        "enforced": enforced(),
        # The archived configs' two shapes: no mode stated, masking switched on by the legacy tokenizer field or by
        # the tokenizer's own declaration.
        "unstated_by_legacy_field": resolve(TokenMaskingConfig(), legacy_field, CPU),
        "unstated_by_tokenizer": resolve(TokenMaskingConfig(), hf_tokenizer_config(declaring_tokenizer), CPU),
        "disabled": masking_with_null_tokenizer("disabled", [MARKER_ID], VOCAB_SIZE),
        "none": no_token_masking(),
    }
    shapes = {
        name: (d.mode, d.enforced, d.enabled, d.observed_token_ids, d.require_masked_targets)
        for name, d in built.items()
    }
    assert shapes == {
        "enforced": ("enabled", True, True, (MARKER_ID,), True),
        "unstated_by_legacy_field": (None, False, True, (MARKER_ID,), False),
        "unstated_by_tokenizer": (None, False, True, (MARKER_ID,), False),
        "disabled": ("disabled", False, False, (MARKER_ID,), False),
        "none": (None, False, False, (), False),
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
    masked_loss_mask, stats = apply_token_masking(labels, loss_mask, resolved)
    used = loss_mask if loss_ignores_token_masking else masked_loss_mask
    report = {"lm loss": torch.stack([used.sum(), used.sum()])}
    if stats is not None:
        report.update(stats.report(used))
    return report


def run_iteration(monitor, group, iteration, *batches, **report_kwargs) -> dict[str, torch.Tensor]:
    """One iteration's reporting as ``train_step`` does it on the last pipeline stage: reduce, then check."""
    reports = [microbatch_report(monitor.resolved, batch, **report_kwargs) for batch in batches]
    return report_step_losses(reports, group, monitor, iteration)


class TestReportStepLosses:
    def test_two_element_entries_are_the_ratio_of_their_sums(self, gloo_group_of_one):
        reports = [{"lm loss": torch.tensor([1.0, 2.0])}, {"lm loss": torch.tensor([0.0, 6.0])}]
        reduced = report_step_losses(reports, gloo_group_of_one, TokenMaskingMonitor(no_token_masking(), None), 1)
        # 1/8, not the mean of the per-microbatch ratios (0.25): a microbatch weighs by its denominator.
        assert reduced["lm loss"].item() == pytest.approx(0.125)

    def test_one_element_entries_are_averaged_over_microbatches(self, gloo_group_of_one):
        reports = [{"grad scale": torch.tensor([2.0])}, {"grad scale": torch.tensor([4.0])}]
        reduced = report_step_losses(reports, gloo_group_of_one, TokenMaskingMonitor(no_token_masking(), None), 1)
        assert reduced["grad scale"].item() == pytest.approx(3.0)

    def test_an_entry_of_any_other_size_raises(self, gloo_group_of_one):
        reports = [{"bad": torch.tensor([1.0, 2.0, 3.0])}]
        with pytest.raises(ValueError, match="Invalid value shape"):
            report_step_losses(reports, gloo_group_of_one, TokenMaskingMonitor(no_token_masking(), None), 1)

    def test_masking_entries_reduce_to_fractions_of_every_target_position(self, gloo_group_of_one):
        monitor = TokenMaskingMonitor(enforced(), None)
        reduced = run_iteration(monitor, gloo_group_of_one, 4, MASKED_BATCH, PLAIN_BATCH)
        # Eight target positions: the marker is the label at two, one of which the dataset trains (and masking
        # removes); six positions keep loss.
        assert {key: reduced[key].item() for key in REPORT_KEYS} == pytest.approx(
            {
                LISTED_TARGET_FRACTION: 2 / 8,
                MASKED_TARGET_FRACTION: 1 / 8,
                TRAINED_LISTED_TARGET_FRACTION: 0.0,
                TRAINABLE_TARGET_FRACTION: 6 / 8,
            }
        )
        assert monitor.first_masked_iteration == 4

    def test_a_run_observing_no_ids_reports_only_its_losses(self, gloo_group_of_one):
        reduced = run_iteration(TokenMaskingMonitor(no_token_masking(), None), gloo_group_of_one, 1, MASKED_BATCH)
        assert set(reduced) == {"lm loss"}

    @pytest.mark.parametrize("name", OBSERVING)
    def test_reports_lacking_the_masking_entries_raise_through_it(self, name, decisions, gloo_group_of_one):
        monitor = TokenMaskingMonitor(decisions[name], None)
        with pytest.raises(TokenMaskingError, match="iteration 17: .* did not apply token masking") as raised:
            report_step_losses([{"lm loss": torch.tensor([4.0, 4.0])}], gloo_group_of_one, monitor, 17)
        assert all(key in str(raised.value) for key in REPORT_KEYS)


class TestMonitorChecks:
    def test_a_run_observing_no_ids_never_raises(self, gloo_group_of_one):
        run = RecordingRun()
        monitor = TokenMaskingMonitor(no_token_masking(), run)
        for iteration in range(1, 13):
            run_iteration(monitor, gloo_group_of_one, iteration, PLAIN_BATCH, MASKED_BATCH)
        monitor.finish()
        assert run.summary.updates == []

    @pytest.mark.parametrize("name", ["enforced", "unstated_by_legacy_field", "unstated_by_tokenizer", "disabled"])
    def test_an_empty_report_lacks_the_masking_entries(self, name, decisions):
        """Only the last pipeline stage observes, so an empty report there means the step reported nothing."""
        with pytest.raises(TokenMaskingError, match=r"iteration 1: .* did not apply token masking"):
            TokenMaskingMonitor(decisions[name], None).observe({}, 1)

    def test_a_run_observing_no_ids_accepts_an_empty_report(self):
        TokenMaskingMonitor(no_token_masking(), None).observe({}, 1)

    @pytest.mark.parametrize("missing", REPORT_KEYS)
    def test_a_report_lacking_one_entry_names_that_entry(self, missing, gloo_group_of_one):
        report = microbatch_report(enforced(), MASKED_BATCH)
        del report[missing]
        with pytest.raises(TokenMaskingError) as raised:
            report_step_losses([report], gloo_group_of_one, TokenMaskingMonitor(enforced(), None), 1)
        assert missing in str(raised.value)
        assert not any(key in str(raised.value) for key in REPORT_KEYS if key != missing)

    @pytest.mark.parametrize("name", ENABLED)
    def test_a_masked_id_that_still_carries_loss_raises(self, name, decisions, gloo_group_of_one):
        monitor = TokenMaskingMonitor(decisions[name], None)
        run_iteration(monitor, gloo_group_of_one, 2, MASKED_BATCH)
        with pytest.raises(TokenMaskingError, match=r"iteration 3: .* still carry loss"):
            run_iteration(monitor, gloo_group_of_one, 3, MASKED_BATCH, loss_ignores_token_masking=True)

    def test_a_disabled_run_trains_on_the_ids_without_raising(self, decisions, gloo_group_of_one):
        run = RecordingRun()
        monitor = TokenMaskingMonitor(decisions["disabled"], run)
        for iteration in range(1, 13):
            reduced = run_iteration(monitor, gloo_group_of_one, iteration, MASKED_BATCH)
            assert reduced[TRAINED_LISTED_TARGET_FRACTION].item() == pytest.approx(1 / 4)
            assert reduced[MASKED_TARGET_FRACTION].item() == 0.0
        monitor.finish()
        assert run.summary.updates == []

    @pytest.mark.parametrize(
        "batches, cause, other_cause",
        [
            ((UNTRAINED_MARKER_BATCH, UNTRAINED_MARKER_BATCH), LISTED_ONLY_UNTRAINED, LISTED_NEVER),
            # Listed targets seen in any iteration of the segment decide the cause, not only the last one.
            ((UNTRAINED_MARKER_BATCH, PLAIN_BATCH), LISTED_ONLY_UNTRAINED, LISTED_NEVER),
            ((PLAIN_BATCH, PLAIN_BATCH), LISTED_NEVER, LISTED_ONLY_UNTRAINED),
        ],
        ids=["listed_untrained", "listed_earlier", "never_listed"],
    )
    def test_nothing_masked_by_the_deadline_raises_naming_the_cause(
        self, batches, cause, other_cause, gloo_group_of_one
    ):
        monitor = TokenMaskingMonitor(enforced(within=2), None)
        run_iteration(monitor, gloo_group_of_one, 1, batches[0])
        with pytest.raises(TokenMaskingError, match="after 2 iterations") as raised:
            run_iteration(monitor, gloo_group_of_one, 2, batches[1])
        message = str(raised.value)
        assert cause in message and other_cause not in message
        assert "require_masked_targets: false" in message

    def test_the_default_deadline_is_ten_iterations(self, gloo_group_of_one):
        monitor = TokenMaskingMonitor(enforced(), None)
        for iteration in range(1, 10):
            run_iteration(monitor, gloo_group_of_one, iteration, PLAIN_BATCH)
        with pytest.raises(TokenMaskingError, match="after 10 iterations"):
            run_iteration(monitor, gloo_group_of_one, 10, PLAIN_BATCH)

    def test_a_target_masked_by_the_deadline_satisfies_the_requirement(self, gloo_group_of_one):
        monitor = TokenMaskingMonitor(enforced(within=2), None)
        run_iteration(monitor, gloo_group_of_one, 1, PLAIN_BATCH)
        run_iteration(monitor, gloo_group_of_one, 2, MASKED_BATCH)
        for iteration in range(3, 13):
            run_iteration(monitor, gloo_group_of_one, iteration, PLAIN_BATCH)
        monitor.finish()

    def test_the_deadline_counts_this_segments_iterations_not_the_step(self, gloo_group_of_one):
        # A segment resumed from a checkpoint at step 5000 gets its own three iterations.
        monitor = TokenMaskingMonitor(enforced(within=3), None)
        run_iteration(monitor, gloo_group_of_one, 5000, PLAIN_BATCH)
        run_iteration(monitor, gloo_group_of_one, 5001, PLAIN_BATCH)
        with pytest.raises(TokenMaskingError, match="after 3 iterations"):
            run_iteration(monitor, gloo_group_of_one, 5002, PLAIN_BATCH)

    def test_require_masked_targets_false_never_raises(self, gloo_group_of_one):
        monitor = TokenMaskingMonitor(enforced(within=2, require=False), None)
        for iteration in range(1, 13):
            run_iteration(monitor, gloo_group_of_one, iteration, PLAIN_BATCH, UNTRAINED_MARKER_BATCH)
        monitor.finish()

    @pytest.mark.parametrize("name", UNSTATED)
    def test_an_unstated_run_that_masks_nothing_never_raises(self, name, decisions, gloo_group_of_one):
        run = RecordingRun()
        monitor = TokenMaskingMonitor(decisions[name], run)
        for iteration in range(1, 13):
            run_iteration(monitor, gloo_group_of_one, iteration, PLAIN_BATCH, UNTRAINED_MARKER_BATCH)
        monitor.finish()
        assert run.summary.updates == []


class TestMonitorFinish:
    @pytest.mark.parametrize(
        "batch, cause", [(UNTRAINED_MARKER_BATCH, LISTED_ONLY_UNTRAINED), (PLAIN_BATCH, LISTED_NEVER)]
    )
    def test_a_segment_ending_before_its_deadline_with_nothing_masked_raises(self, batch, cause, gloo_group_of_one):
        monitor = TokenMaskingMonitor(enforced(within=10), None)
        for iteration in range(1, 4):
            run_iteration(monitor, gloo_group_of_one, iteration, batch)
        with pytest.raises(TokenMaskingError, match="in the 3 iterations this segment ran") as raised:
            monitor.finish()
        assert cause in str(raised.value)
        assert "raise token_masking.require_masked_targets_within_iterations" in str(raised.value)

    def test_a_segment_that_observed_no_iteration_is_not_judged(self):
        # A non-last pipeline stage never observes reports; neither does a segment that resumed at its final step.
        monitor = TokenMaskingMonitor(enforced(within=10), None)
        monitor.finish()

    def test_a_segment_that_masked_a_target_passes(self, gloo_group_of_one):
        monitor = TokenMaskingMonitor(enforced(within=10), None)
        for iteration, batch in enumerate((PLAIN_BATCH, MASKED_BATCH, PLAIN_BATCH), start=1):
            run_iteration(monitor, gloo_group_of_one, iteration, batch)
        monitor.finish()


class TestMonitorWandbSummary:
    def test_an_enforced_run_is_unverified_until_its_first_masked_iteration(self, gloo_group_of_one, caplog):
        run = RecordingRun()
        monitor = TokenMaskingMonitor(enforced(), run)
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

    @pytest.mark.parametrize("name", UNSTATED)
    def test_an_unenforced_run_records_only_when_it_masks(self, name, decisions, gloo_group_of_one):
        run = RecordingRun()
        monitor = TokenMaskingMonitor(decisions[name], run)
        assert run.summary.updates == []
        run_iteration(monitor, gloo_group_of_one, 7, MASKED_BATCH)
        assert run.summary.updates == [{VERIFIED_SUMMARY_KEY: True, FIRST_MASKED_ITERATION_SUMMARY_KEY: 7}]

    def test_a_rank_without_a_wandb_run_still_checks(self, gloo_group_of_one):
        monitor = TokenMaskingMonitor(enforced(within=2), None)
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
        fields = [("mode", "enabled"), ("tokens", '["<marker>"]'), ("tokenizer", "/a dir/tok"), ("empty", "")]
        (record,) = self._banner(caplog, fields)
        assert record.levelno == logging.INFO
        assert record.getMessage() == (
            f"[token-masking] rank=12 host={socket.gethostname()} "
            "mode=enabled tokens='[\"<marker>\"]' tokenizer='/a dir/tok' empty=''"
        )

    def test_a_real_decisions_banner_splits_back_into_its_fields(self, caplog, decisions):
        fields = banner_fields(decisions["unstated_by_tokenizer"], None)
        (record,) = self._banner(caplog, fields)
        words = shlex.split(record.getMessage().removeprefix("[token-masking] "))
        assert dict(word.split("=", 1) for word in words) == {
            "rank": "12",
            "host": socket.gethostname(),
            **dict(fields),
        }
        assert dict(fields)["tokens"] == f'["{MARKER}"]'

    def test_only_the_first_process_of_a_node_logs(self, caplog):
        assert self._banner(caplog, [("mode", "enabled")], local_rank=1) == []

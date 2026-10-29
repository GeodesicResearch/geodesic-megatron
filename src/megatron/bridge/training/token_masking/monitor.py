# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Check, every iteration, that a run configured to mask tokens is masking them.

The monitor reads the iteration's reduced reporting dict (already summed over microbatches and all-reduced over the
data-parallel ranks, so every last-pipeline-stage rank holds the same values) and raises ``TokenMaskingError`` when:

- the run measures token ids but the reporting dict lacks the token-masking entries: the forward step did not run
  the masking code;
- masking is enabled but a target whose label is a masked id still carries loss;
- masking is enabled and no target of the global batch carries loss, so its lm loss is 0/0 and the step trained
  nothing but the MoE auxiliary loss, the expert-bias update, the optimizer's momentum and weight decay.

Only the last pipeline stage holds the reports. With more than one stage, every stage takes part in one 1-element
all-reduce over the pipeline-parallel group per iteration, so all of them stop together when the last stage's check
fails rather than the others waiting for it in their next collective.

The checks read exact integer counts (``TokenMaskingCounts``), which the rank that writes the training log prints
every iteration as one ``[token-masking-counts]`` line and logs to W&B under ``token_masking/count/``, so a run's
listed, masked and trained listed targets can be compared exactly with a prediction from its data.

There is no deadline for a first masked target: an enabled run shows marker targets in its training data at setup.
The monitor records in the W&B summary whether, and from which iteration, the segment was seen masking, and turns the
reduced listed-target loss sum into the listed-target loss the logs report.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from megatron.bridge.training.token_masking.config import TokenMaskingError
from megatron.bridge.training.token_masking.hook import (
    REPORT_KEYS,
    TokenMaskingCounts,
    finalize_listed_target_loss,
)
from megatron.bridge.training.token_masking.resolution import ResolvedTokenMasking
from megatron.bridge.training.utils.wandb_utils import record_wandb_summary


logger = logging.getLogger(__name__)

VERIFIED_SUMMARY_KEY = "token_masking/verified"
FIRST_MASKED_ITERATION_SUMMARY_KEY = "token_masking/first_masked_iteration"
COUNTS_LOG_TAG = "[token-masking-counts]"


class TokenMaskingMonitor:
    """Per-segment record of what token masking did, checked once per training iteration."""

    def __init__(
        self,
        resolved: ResolvedTokenMasking,
        wandb_run: Any | None,
        pp_group: torch.distributed.ProcessGroup,
        log_counts: bool,
    ) -> None:
        """Args:
        resolved: The run's token-masking decision.
        wandb_run: ``wandb.run`` on the rank that logs to W&B, None elsewhere.
        pp_group: This rank's pipeline-parallel group, over which the stages agree on each iteration's verdict.
        log_counts: Whether this rank prints each iteration's ``[token-masking-counts]`` line: true on the one rank
            that writes the training log, so a run prints the line once per iteration whatever its width.
        """
        self.resolved = resolved
        self.wandb_run = wandb_run
        self.pp_group = pp_group
        self.log_counts = log_counts
        self.first_masked_iteration: int | None = None
        if resolved.enabled:
            record_wandb_summary(wandb_run, {VERIFIED_SUMMARY_KEY: False})

    def observe(
        self, reports: dict[str, torch.Tensor], sums: dict[str, torch.Tensor], iteration: int
    ) -> dict[str, torch.Tensor]:
        """Check one iteration's reduced reports and return them as the logs report them.

        ``reports`` holds the reduced values and ``sums`` the reduced ``[numerator, denominator]`` of each 2-element
        entry, from which the checks read exact counts. Called on the ranks that hold the reports (the last pipeline
        stage), where an empty report means the forward step reported nothing; the other stages call
        ``await_last_stage`` at the same point. The counts are recorded before the verdict, so a failing iteration's
        counts reach the log too. The returned reports carry ``token_masking/listed_target_loss`` in place of the
        listed target loss sum, and only when the global batch held a trainable listed target (see
        ``finalize_listed_target_loss``).
        """
        if not self.resolved.measured_token_ids:
            return reports
        failure, counts = self._check(reports, sums, iteration)
        if counts is not None:
            self._record_counts(counts, iteration)
        if self._any_stage_failed(failure is not None):
            raise TokenMaskingError(failure)
        if counts.masked > 0 and self.first_masked_iteration is None:
            self.first_masked_iteration = iteration
            record_wandb_summary(
                self.wandb_run, {VERIFIED_SUMMARY_KEY: True, FIRST_MASKED_ITERATION_SUMMARY_KEY: iteration}
            )
            logger.info(
                f"[token-masking] iteration {iteration}: masked targets observed "
                f"({counts.masked} of {counts.positions} targets)"
            )
        # It reads one more count, after the synchronisation in _check, so it waits for no device work.
        return finalize_listed_target_loss(reports)

    def _record_counts(self, counts: TokenMaskingCounts, iteration: int) -> None:
        """Print the iteration's counts line on the rank that writes the training log, and log them to W&B."""
        if self.log_counts:
            logger.info(f"{COUNTS_LOG_TAG} iteration={iteration} {counts.log_fields()}")
        if self.wandb_run is not None:
            self.wandb_run.log(counts.wandb_metrics(), step=iteration)

    def await_last_stage(self, iteration: int) -> None:
        """Take part, on a pipeline stage that holds no reports, in the iteration's verdict; raise when it failed."""
        if self.resolved.measured_token_ids and self._any_stage_failed(False):
            raise TokenMaskingError(
                f"iteration {iteration}: the token-masking check failed on the last pipeline stage, whose error names "
                "the cause; every stage stops with it"
            )

    def _any_stage_failed(self, failed: bool) -> bool:
        """Whether this stage's check, or another's, failed: one 1-element MAX all-reduce over the pipeline group,
        run by every stage when the pipeline has more than one."""
        if self.pp_group.size() == 1:
            return failed
        verdict = torch.tensor([1 if failed else 0], dtype=torch.int32, device=self.resolved.ids_tensor.device)
        torch.distributed.all_reduce(verdict, op=torch.distributed.ReduceOp.MAX, group=self.pp_group)
        return bool(verdict.item())

    def _check(
        self, reports: dict[str, torch.Tensor], sums: dict[str, torch.Tensor], iteration: int
    ) -> tuple[str | None, TokenMaskingCounts | None]:
        """What is wrong with one iteration's reports, or None, and its counts (None when the reports lack them)."""
        missing = [key for key in REPORT_KEYS if key not in reports]
        if missing:
            return (
                f"iteration {iteration}: token masking measures ids {list(self.resolved.measured_token_ids)} but the "
                f"loss reports lack {missing}: the forward step did not apply token masking"
            ), None
        try:
            counts = TokenMaskingCounts.from_sums(sums)
        except TokenMaskingError as error:
            return f"iteration {iteration}: {error}", None
        if not self.resolved.enabled:
            return None, counts
        ids = list(self.resolved.token_ids)
        if counts.trained_listed > 0:
            return (
                f"iteration {iteration}: {counts.trained_listed} of the {counts.positions} targets are masked ids "
                f"{ids} that still carry loss; the loss was computed with a mask other than the token-masked one"
            ), counts
        if counts.trainable == 0:
            return (
                f"iteration {iteration}: global batch with no trainable target (of its {counts.positions} targets, "
                f"{counts.listed} are the masked ids {ids} and {counts.masked} had their loss masked): its lm loss is "
                "0/0, and the step trained only the MoE auxiliary loss, the expert-bias update, the optimizer's "
                "momentum and weight decay. Check that the training data holds trainable targets other than the "
                "marker."
            ), counts
        return None, counts

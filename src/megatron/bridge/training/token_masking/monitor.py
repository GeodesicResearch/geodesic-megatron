# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Check, every iteration, that a run configured to mask tokens is masking them.

The monitor reads the iteration's reduced reporting dict (already summed over microbatches and all-reduced over the
data-parallel ranks, so every last-pipeline-stage rank holds the same values) and raises ``TokenMaskingError`` when:

- the run observes token ids but the reporting dict lacks the token-masking entries: the forward step did not run
  the masking code;
- masking is enabled but a target whose label is a masked id still carries loss;
- masking is enforced (``mode: enabled``) with ``require_masked_targets``, and no target has been masked within
  ``require_masked_targets_within_iterations`` iterations of the segment, or by the end of a shorter segment.

It also records in the W&B summary whether, and from which iteration, the run was seen masking.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from megatron.bridge.training.token_masking.config import TokenMaskingError
from megatron.bridge.training.token_masking.hook import (
    LISTED_TARGET_FRACTION,
    MASKED_TARGET_FRACTION,
    REPORT_KEYS,
    TRAINED_LISTED_TARGET_FRACTION,
)
from megatron.bridge.training.token_masking.resolution import ResolvedTokenMasking
from megatron.bridge.training.utils.wandb_utils import record_wandb_summary


logger = logging.getLogger(__name__)

VERIFIED_SUMMARY_KEY = "token_masking/verified"
FIRST_MASKED_ITERATION_SUMMARY_KEY = "token_masking/first_masked_iteration"


class TokenMaskingMonitor:
    """Per-segment record of what token masking did, checked once per training iteration."""

    def __init__(self, resolved: ResolvedTokenMasking, wandb_run: Any | None) -> None:
        """Args:
        resolved: The run's token-masking decision.
        wandb_run: ``wandb.run`` on the rank that logs to W&B, None elsewhere.
        """
        self.resolved = resolved
        self.wandb_run = wandb_run
        self.iterations_observed = 0
        self.first_masked_iteration: int | None = None
        self.listed_targets_seen = False
        if resolved.enforced:
            record_wandb_summary(wandb_run, {VERIFIED_SUMMARY_KEY: False})

    def observe(self, reports: dict[str, torch.Tensor], iteration: int) -> None:
        """Check one iteration's reduced reports; called on the ranks that hold them (the last pipeline stage), where
        an empty report means the forward step reported nothing."""
        if not self.resolved.observed_token_ids:
            return
        missing = [key for key in REPORT_KEYS if key not in reports]
        if missing:
            raise TokenMaskingError(
                f"iteration {iteration}: token masking observes ids {list(self.resolved.observed_token_ids)} but the "
                f"loss reports lack {missing}: the forward step did not apply token masking"
            )
        listed, masked, trained_listed = torch.stack(
            [reports[LISTED_TARGET_FRACTION], reports[MASKED_TARGET_FRACTION], reports[TRAINED_LISTED_TARGET_FRACTION]]
        ).tolist()
        self.iterations_observed += 1
        self.listed_targets_seen = self.listed_targets_seen or listed > 0
        if self.resolved.enabled and trained_listed > 0:
            raise TokenMaskingError(
                f"iteration {iteration}: a fraction {trained_listed:.3e} of the targets are masked ids "
                f"{list(self.resolved.token_ids)} that still carry loss; the loss was computed with a mask other "
                "than the token-masked one"
            )
        if masked > 0 and self.first_masked_iteration is None:
            self.first_masked_iteration = iteration
            record_wandb_summary(
                self.wandb_run, {VERIFIED_SUMMARY_KEY: True, FIRST_MASKED_ITERATION_SUMMARY_KEY: iteration}
            )
            logger.info(f"[token-masking] iteration {iteration}: masked targets observed (fraction {masked:.3e})")
        if (
            self._must_have_masked()
            and self.iterations_observed >= self.resolved.require_masked_targets_within_iterations
        ):
            raise TokenMaskingError(self._nothing_masked_message(f"after {self.iterations_observed} iterations"))

    def finish(self) -> None:
        """Apply the masked-target requirement to a segment that ended before its deadline."""
        if self.iterations_observed and self._must_have_masked():
            raise TokenMaskingError(
                self._nothing_masked_message(f"in the {self.iterations_observed} iterations this segment ran")
            )

    def _must_have_masked(self) -> bool:
        return self.resolved.enforced and self.resolved.require_masked_targets and self.first_masked_iteration is None

    def _nothing_masked_message(self, when: str) -> str:
        ids = list(self.resolved.token_ids)
        if self.listed_targets_seen:
            cause = (
                f"the ids {ids} occur as targets, but only at positions the dataset already excludes from the loss "
                "(for example outside the assistant's {% generation %} span), so masking removes nothing"
            )
        else:
            cause = (
                f"no target in the training batches is one of the ids {ids}: the data may have been tokenized "
                "with a tokenizer that does not register the marker as a single token"
            )
        return (
            f"token masking is enabled but masked no target {when}: {cause}. If the marker is legitimately rarer "
            "than that, raise token_masking.require_masked_targets_within_iterations (a segment shorter than it is "
            "checked when it ends); if this stage is meant to have nothing to mask, set "
            "token_masking.require_masked_targets: false."
        )

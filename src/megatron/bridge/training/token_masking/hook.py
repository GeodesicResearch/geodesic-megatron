# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Apply token masking to one microbatch and count what it did.

The mask works on target positions: ``labels[t]`` is the token predicted at position ``t``, and a position whose label
is a masked id gets no loss, so the model is never trained to emit that token while it still reads it in its context.
The statistics stay on the device; they reach the logs through the loss function's reporting dict, which the training
step sums over microbatches and all-reduces over data-parallel ranks once per iteration. The counts are int64, so a
global batch's sums stay exact at any size: a float32 sum is exact only up to 2**24 target positions, which a
pretraining global batch of 2048 x 8192 tokens already reaches.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass, fields
from typing import TYPE_CHECKING

import torch

from megatron.bridge.training.token_masking.config import TokenMaskingError


if TYPE_CHECKING:
    from megatron.bridge.training.token_masking.resolution import ResolvedTokenMasking


TOKEN_MASKING_NAMESPACE = "token_masking/"
LISTED_TARGET_FRACTION = "token_masking/listed_target_fraction"
MASKED_TARGET_FRACTION = "token_masking/masked_target_fraction"
TRAINED_LISTED_TARGET_FRACTION = "token_masking/trained_listed_target_fraction"
TRAINABLE_TARGET_FRACTION = "token_masking/trainable_target_fraction"
LISTED_TRAINABLE_TARGET_FRACTION = "token_masking/listed_trainable_target_fraction"
LISTED_TARGET_LOSS_SUM = "token_masking/listed_target_loss_sum"
# Reported in place of LISTED_TARGET_LOSS_SUM once the reports are reduced; see finalize_listed_target_loss.
LISTED_TARGET_LOSS = "token_masking/listed_target_loss"
# Every entry is reported as [numerator, target positions] over a microbatch's target positions, so its reduction over
# a global batch or an evaluation never divides by zero. Every numerator but the loss sum's is an int64 count.
REPORT_KEYS = (
    LISTED_TARGET_FRACTION,
    MASKED_TARGET_FRACTION,
    TRAINED_LISTED_TARGET_FRACTION,
    TRAINABLE_TARGET_FRACTION,
    LISTED_TRAINABLE_TARGET_FRACTION,
    LISTED_TARGET_LOSS_SUM,
)


@dataclass(frozen=True)
class TokenMaskingStats:
    """What token masking found and did in one microbatch, as device tensors.

    ``listed`` marks the target positions whose label is a measured id, and ``would_train`` those of them that carried
    loss in the dataset's mask, before token masking changed it. The counts are 0-dim int64 tensors: ``positions``
    (all target positions), ``listed_targets`` (labels that are measured ids) and ``masked_targets`` (positions that
    carried loss before masking and have none after it; 0 unless masking is enabled).
    """

    listed: torch.Tensor
    would_train: torch.Tensor
    positions: torch.Tensor
    listed_targets: torch.Tensor
    masked_targets: torch.Tensor

    def report(self, loss_mask: torch.Tensor, losses: torch.Tensor) -> dict[str, torch.Tensor]:
        """The reporting-dict entries, given the mask the loss is actually computed with and the per-token losses.

        ``trained_listed_target_fraction`` and ``trainable_target_fraction`` are measured against ``loss_mask`` rather
        than the mask built here, so a loss that ends up multiplied by any mask other than the masked one shows: with
        masking enabled the first must be 0. ``listed_target_loss_sum`` sums the per-token cross-entropy at the listed
        targets the dataset trains, before masking removed them, so it measures the masked ids' targets in a masked
        run and in its unmasked control alike.
        """
        carries_loss = loss_mask.reshape(-1) != 0
        listed = self.listed.reshape(-1)
        would_train = self.would_train.reshape(-1)
        trained_listed = (carries_loss & listed).sum()
        trainable = carries_loss.sum()
        # torch.where, not a product: a non-finite loss at an unlisted position must not turn the sum into NaN.
        listed_loss = torch.where(would_train, losses.detach().reshape(-1).float(), 0.0).sum()
        return {
            LISTED_TARGET_FRACTION: torch.stack([self.listed_targets, self.positions]),
            MASKED_TARGET_FRACTION: torch.stack([self.masked_targets, self.positions]),
            TRAINED_LISTED_TARGET_FRACTION: torch.stack([trained_listed, self.positions]),
            TRAINABLE_TARGET_FRACTION: torch.stack([trainable, self.positions]),
            LISTED_TRAINABLE_TARGET_FRACTION: torch.stack([would_train.sum(), self.positions]),
            LISTED_TARGET_LOSS_SUM: torch.stack([listed_loss, self.positions.float()]),
        }


def apply_token_masking(
    labels: torch.Tensor | None, loss_mask: torch.Tensor | None, masking: ResolvedTokenMasking
) -> tuple[torch.Tensor | None, TokenMaskingStats | None]:
    """Mask the measured ids out of ``loss_mask`` when masking is enabled, and count them either way.

    Returns ``loss_mask`` unchanged and no statistics when the run measures no ids, and on pipeline stages that hold
    no labels. Otherwise returns the mask to train with (``loss_mask`` itself when masking is disabled) and the
    microbatch's statistics. Never synchronises with the host.
    """
    if masking.ids_tensor is None or labels is None or loss_mask is None:
        return loss_mask, None
    ids = masking.ids_tensor if masking.ids_tensor.dtype == labels.dtype else masking.ids_tensor.to(labels.dtype)
    # An elementwise comparison rather than torch.isin, which switches to a sort that synchronises with the host
    # once a few dozen ids are listed.
    listed = (labels.unsqueeze(-1) == ids).any(dim=-1)
    would_train = listed & (loss_mask != 0)
    positions = torch.full((), labels.numel(), dtype=torch.int64, device=labels.device)
    if masking.enabled:
        masked_targets = would_train.sum()
        loss_mask = loss_mask * (~listed).to(loss_mask.dtype)
    else:
        masked_targets = torch.zeros((), dtype=torch.int64, device=labels.device)
    return loss_mask, TokenMaskingStats(
        listed=listed,
        would_train=would_train,
        positions=positions,
        listed_targets=listed.sum(),
        masked_targets=masked_targets,
    )


def finalize_listed_target_loss(reduced: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Replace reduced reports' listed-target loss sum with the mean cross-entropy at those targets.

    ``reduced`` holds reports already reduced to ratios over the same target positions (a global batch's, or an
    evaluation's), so ``listed_target_loss_sum / listed_trainable_target_fraction`` is the summed cross-entropy at the
    listed targets the dataset trains over their number: ``token_masking/listed_target_loss``. When there is no such
    target the entry is left out rather than reported as 0/0. Reports without the sum are returned unchanged.
    """
    if LISTED_TARGET_LOSS_SUM not in reduced:
        return reduced
    finalized = {key: value for key, value in reduced.items() if key != LISTED_TARGET_LOSS_SUM}
    count = reduced[LISTED_TRAINABLE_TARGET_FRACTION]
    if count > 0:
        finalized[LISTED_TARGET_LOSS] = reduced[LISTED_TARGET_LOSS_SUM] / count
    return finalized


# The report entry whose numerator is each count of TokenMaskingCounts; every denominator is the target positions.
COUNT_REPORT_KEYS = {
    "listed": LISTED_TARGET_FRACTION,
    "listed_trainable": LISTED_TRAINABLE_TARGET_FRACTION,
    "masked": MASKED_TARGET_FRACTION,
    "trained_listed": TRAINED_LISTED_TARGET_FRACTION,
    "trainable": TRAINABLE_TARGET_FRACTION,
}


@dataclass(frozen=True)
class TokenMaskingCounts:
    """One global batch's token-masking counts: exact integers over its target positions, summed over its microbatches
    and over the data- and context-parallel ranks.

    ``listed`` counts the targets whose label is a measured id, ``listed_trainable`` those of them the dataset trains,
    ``masked`` those token masking removed from the loss, ``trained_listed`` the listed targets that still carry loss
    (0 in a masking run), ``trainable`` the targets that carry loss, and ``positions`` every target position.
    """

    listed: int
    listed_trainable: int
    masked: int
    trained_listed: int
    trainable: int
    positions: int

    @classmethod
    def from_sums(cls, sums: dict[str, torch.Tensor]) -> TokenMaskingCounts:
        """Read the counts from a global batch's reduced ``[numerator, target positions]`` report sums.

        Synchronises with the host once. Raises when a count was reduced as anything but an int64 tensor, since a
        float sum stops being exact past 2**24 target positions.
        """
        keys = list(COUNT_REPORT_KEYS.values())
        inexact = [key for key in keys if sums[key].dtype != torch.int64]
        if inexact:
            raise TokenMaskingError(f"the token-masking counts {inexact} were reduced as non-integer tensors")
        values = torch.stack([sums[key][0] for key in keys] + [sums[LISTED_TARGET_FRACTION][1]]).tolist()
        return cls(**dict(zip([*COUNT_REPORT_KEYS, "positions"], values)))

    def log_fields(self) -> str:
        """The counts as ``name=value`` pairs in field order, as the per-iteration counts line prints them."""
        return " ".join(f"{field.name}={value}" for field, value in zip(fields(self), astuple(self)))

    def wandb_metrics(self) -> dict[str, int]:
        """The counts as W&B metrics, each under ``token_masking/count/``."""
        return {
            f"{TOKEN_MASKING_NAMESPACE}count/{field.name}": value for field, value in zip(fields(self), astuple(self))
        }

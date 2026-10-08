# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Apply token masking to one microbatch and count what it did.

The mask works on target positions: ``labels[t]`` is the token predicted at position ``t``, and a position whose label
is a masked id gets no loss, so the model is never trained to emit that token while it still reads it in its context.
The statistics stay on the device; they reach the logs through the loss function's reporting dict, which the training
step sums over microbatches and all-reduces over data-parallel ranks once per iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from megatron.bridge.training.token_masking.resolution import ResolvedTokenMasking


LISTED_TARGET_FRACTION = "token_masking/listed_target_fraction"
MASKED_TARGET_FRACTION = "token_masking/masked_target_fraction"
TRAINED_LISTED_TARGET_FRACTION = "token_masking/trained_listed_target_fraction"
TRAINABLE_TARGET_FRACTION = "token_masking/trainable_target_fraction"
# Every entry is a fraction of a microbatch's target positions, reported as [numerator, denominator].
REPORT_KEYS = (
    LISTED_TARGET_FRACTION,
    MASKED_TARGET_FRACTION,
    TRAINED_LISTED_TARGET_FRACTION,
    TRAINABLE_TARGET_FRACTION,
)


@dataclass(frozen=True)
class TokenMaskingStats:
    """What token masking found and did in one microbatch, as device tensors.

    ``listed`` marks the target positions whose label is an observed id. The counts are 0-dim float tensors:
    ``positions`` (all target positions), ``listed_targets`` (labels that are observed ids) and ``masked_targets``
    (positions that carried loss before masking and have none after it).
    """

    listed: torch.Tensor
    positions: torch.Tensor
    listed_targets: torch.Tensor
    masked_targets: torch.Tensor

    def report(self, loss_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """The reporting-dict entries, measured against ``loss_mask``, the mask the loss is actually computed with.

        ``trained_listed_target_fraction`` is measured here rather than when the mask was built, so it catches a loss
        that ends up multiplied by any mask other than the masked one: it must be 0 whenever masking is on.
        """
        carries_loss = loss_mask.reshape(-1) != 0
        trained_listed = (carries_loss & self.listed.reshape(-1)).sum().float()
        trainable = carries_loss.sum().float()
        return {
            LISTED_TARGET_FRACTION: torch.stack([self.listed_targets, self.positions]),
            MASKED_TARGET_FRACTION: torch.stack([self.masked_targets, self.positions]),
            TRAINED_LISTED_TARGET_FRACTION: torch.stack([trained_listed, self.positions]),
            TRAINABLE_TARGET_FRACTION: torch.stack([trainable, self.positions]),
        }


def apply_token_masking(
    labels: torch.Tensor | None, loss_mask: torch.Tensor | None, masking: ResolvedTokenMasking
) -> tuple[torch.Tensor | None, TokenMaskingStats | None]:
    """Mask the observed ids out of ``loss_mask`` when masking is enabled, and count them either way.

    Returns ``loss_mask`` unchanged and no statistics when the run observes no ids, and on pipeline stages that hold no
    labels. Otherwise returns the mask to train with (``loss_mask`` itself when masking is disabled) and the
    microbatch's statistics. Never synchronises with the host.
    """
    if masking.ids_tensor is None or labels is None or loss_mask is None:
        return loss_mask, None
    ids = masking.ids_tensor if masking.ids_tensor.dtype == labels.dtype else masking.ids_tensor.to(labels.dtype)
    # An elementwise comparison rather than torch.isin, which switches to a sort that synchronises with the host
    # once a few dozen ids are listed.
    listed = (labels.unsqueeze(-1) == ids).any(dim=-1)
    positions = torch.full((), labels.numel(), dtype=torch.float32, device=labels.device)
    listed_targets = listed.sum().float()
    if masking.enabled:
        masked_targets = (listed & (loss_mask != 0)).sum().float()
        loss_mask = loss_mask * (~listed).to(loss_mask.dtype)
    else:
        masked_targets = torch.zeros((), dtype=torch.float32, device=labels.device)
    return loss_mask, TokenMaskingStats(
        listed=listed, positions=positions, listed_targets=listed_targets, masked_targets=masked_targets
    )

# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The ``token_masking:`` block of a training config.

Token masking removes from the training loss every target position whose label is one of a list of token ids, so
the model reads those tokens in its context but is never trained to emit them (for example a ``<quarantine_token>``
marker). It works on the target ids alone, on top of whatever loss mask the dataset already produced (answer-only
SFT masking, padding), which it never changes.

A run states its intent with ``mode``:

- ``enabled``: the run must mask. The ids are ``token_ids``, else the legacy ``tokenizer.loss_mask_token_ids``,
  else the ids the tokenizer declares in its ``tokenizer_config.json`` (``loss_mask_token_ids``). Setup fails when
  no ids can be resolved, when the explicit ids disagree with the tokenizer's declaration, or when the training
  data shows the ids would never be masked; training fails when no target has been masked within
  ``require_masked_targets_within_iterations``.
- ``disabled``: the run must not mask, even when the tokenizer declares ids. The declared ids, or ``token_ids``
  given here, are still counted in the metrics and the sample tables, so a control arm is directly comparable with
  its masked twin.
- omitted: the behaviour of configs written before this block existed. ``tokenizer.loss_mask_token_ids`` decides
  when it is set (``[]`` masks nothing), otherwise the tokenizer's declaration does, and nothing is enforced. New
  configs state the mode; the archived campaign configs under ``configs/misalignment_quarantine/`` do not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal


TOKEN_MASKING_MODES = ("enabled", "disabled")


class TokenMaskingError(RuntimeError):
    """Token masking is misconfigured, or a run that must mask is not masking."""


@dataclass
class TokenMaskingConfig:
    """How a run applies token masking. See the module docstring and docs/training/token-masking.md."""

    reject_unknown_override_keys: ClassVar[bool] = True

    mode: Literal["enabled", "disabled"] | None = None
    """``enabled``, ``disabled``, or omitted for the behaviour of configs that predate this block."""

    token_ids: list[int] | None = None
    """With ``enabled``: the ids to mask, which must equal the tokenizer's declaration when it declares any; omit to
    use the declaration. With ``disabled``: ids to count in metrics and tables without masking them."""

    require_masked_targets: bool | None = None
    """With ``enabled`` only (omitted means true): fail when the data shows no target would be masked, and when no
    target has been masked within ``require_masked_targets_within_iterations`` iterations of a segment. Set false for
    a stage whose data legitimately holds no maskable targets."""

    require_masked_targets_within_iterations: int | None = None
    """With ``enabled`` only (omitted means 10): the iteration of a segment by which a target must have been masked.
    A segment that ends earlier is checked when it ends."""

    resolved: dict[str, Any] | None = field(default=None, init=False)
    """What setup resolved for this run (mode, applied and observed ids, their tokens, where the ids came from).
    Written by setup, so the W&B config and every checkpoint's run_config.yaml record the decision; dropped when a
    saved config is loaded back."""

    def finalize(self) -> None:
        """Validate the block on its own; the check against ``tokenizer.loss_mask_token_ids`` needs the container."""
        if isinstance(self.mode, bool):
            raise TokenMaskingError(
                f"token_masking.mode is the boolean {self.mode}: YAML reads unquoted on/off/yes/no as booleans. "
                "Write mode: enabled or mode: disabled."
            )
        if self.mode is not None and self.mode not in TOKEN_MASKING_MODES:
            raise TokenMaskingError(
                f"token_masking.mode must be one of {TOKEN_MASKING_MODES} or omitted, got {self.mode!r}"
            )
        if self.token_ids is not None:
            if self.mode is None:
                raise TokenMaskingError(
                    "token_masking.token_ids needs token_masking.mode: enabled (mask them) or disabled (only count them)"
                )
            validate_token_ids(self.token_ids, "token_masking.token_ids")
            if self.mode == "enabled" and not self.token_ids:
                raise TokenMaskingError(
                    "token_masking.token_ids is empty with mode: enabled; to mask nothing use mode: disabled"
                )
        if self.mode != "enabled":
            stated = [
                name
                for name in ("require_masked_targets", "require_masked_targets_within_iterations")
                if getattr(self, name) is not None
            ]
            if stated:
                raise TokenMaskingError(f"token_masking.{', '.join(stated)} applies only with mode: enabled")
        within = self.require_masked_targets_within_iterations
        if within is not None and (isinstance(within, bool) or not isinstance(within, int) or within < 1):
            raise TokenMaskingError(
                f"token_masking.require_masked_targets_within_iterations must be a positive integer, got {within!r}"
            )
        if self.require_masked_targets is not None and not isinstance(self.require_masked_targets, bool):
            raise TokenMaskingError(
                f"token_masking.require_masked_targets must be true or false, got {self.require_masked_targets!r}"
            )


def validate_token_ids(token_ids: Any, name: str) -> None:
    """Raise unless ``token_ids`` is a list of distinct non-negative integers (bools rejected)."""
    if not isinstance(token_ids, (list, tuple)):
        raise TokenMaskingError(f"{name} must be a list of token ids, got {token_ids!r}")
    bad = [token_id for token_id in token_ids if isinstance(token_id, bool) or not isinstance(token_id, int)]
    if bad:
        raise TokenMaskingError(f"{name} must hold integer token ids, got {bad!r} in {list(token_ids)!r}")
    negative = [token_id for token_id in token_ids if token_id < 0]
    if negative:
        raise TokenMaskingError(f"{name} holds negative token ids {negative!r}")
    if len(set(token_ids)) != len(token_ids):
        raise TokenMaskingError(f"{name} repeats token ids: {list(token_ids)!r}")


def check_against_legacy_field(token_masking: TokenMaskingConfig, legacy_token_ids: list[int] | None) -> None:
    """Raise when ``tokenizer.loss_mask_token_ids`` contradicts a stated ``token_masking.mode``.

    The legacy field is how configs written before the block chose masking. Next to a stated mode it is allowed only
    where both say the same thing: ``[]`` with ``disabled``, or the masked ids with ``enabled``. Stating it there is
    how a control arm stays unmasked even on code that predates the block. Expects a block that ``finalize`` accepted.
    """
    require_block(token_masking)
    if legacy_token_ids is None or token_masking.mode is None:
        if legacy_token_ids is not None:
            validate_token_ids(legacy_token_ids, "tokenizer.loss_mask_token_ids")
        return
    validate_token_ids(legacy_token_ids, "tokenizer.loss_mask_token_ids")
    legacy = list(legacy_token_ids)
    if token_masking.mode == "disabled":
        if legacy:
            raise TokenMaskingError(
                f"tokenizer.loss_mask_token_ids={legacy} masks tokens but token_masking.mode is disabled; "
                "set tokenizer.loss_mask_token_ids: [] or remove it"
            )
        return
    if not legacy:
        raise TokenMaskingError(
            "tokenizer.loss_mask_token_ids=[] masks nothing but token_masking.mode is enabled; "
            "remove tokenizer.loss_mask_token_ids and state the ids in token_masking.token_ids"
        )
    if token_masking.token_ids is not None and sorted(token_masking.token_ids) != sorted(legacy):
        raise TokenMaskingError(
            f"tokenizer.loss_mask_token_ids={legacy} differs from token_masking.token_ids={token_masking.token_ids}; "
            "keep one, in token_masking.token_ids"
        )


def require_block(token_masking: Any) -> None:
    """Raise unless ``token_masking`` is the config block (a YAML scalar such as ``token_masking: enabled`` is not)."""
    if not isinstance(token_masking, TokenMaskingConfig):
        raise TokenMaskingError(
            f"token_masking must be a mapping such as token_masking: {{mode: enabled}}, got {token_masking!r}"
        )


def validate_token_masking(token_masking: Any, legacy_token_ids: list[int] | None) -> None:
    """Validate the block, then its agreement with ``tokenizer.loss_mask_token_ids``: what config validation runs."""
    require_block(token_masking)
    token_masking.finalize()
    check_against_legacy_field(token_masking, legacy_token_ids)


def explicit_token_ids(token_masking: TokenMaskingConfig, legacy_token_ids: list[int] | None) -> list[int] | None:
    """The ids a config names for ``enabled`` mode: ``token_masking.token_ids``, else the legacy field."""
    if token_masking.token_ids is not None:
        return list(token_masking.token_ids)
    if legacy_token_ids:
        return list(legacy_token_ids)
    return None

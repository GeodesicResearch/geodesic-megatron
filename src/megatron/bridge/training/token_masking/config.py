# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The ``token_masking:`` block of a training config.

Token masking removes from the training loss every target position whose label is one of the listed token ids, so
the model reads those tokens in its context but is never trained to emit them (for example a ``<quarantine_token>``
marker). It works on the target ids alone, on top of whatever loss mask the dataset already produced (answer-only
SFT masking, padding), which it never changes.

The config alone decides which ids are masked; nothing is read from the tokenizer to decide it:

- ``enabled: true`` with ``token_ids``: mask those ids. ``enabled`` and a non-empty ``token_ids`` go together.
- masking off with ``masked_validation.token_ids``: mask nothing, but measure those ids (the per-iteration counts and
  their target cross-entropy), so a control arm is directly comparable with its masked twin.
- the block omitted: no masking and no measurement.

``masked_validation`` also names an optional held-out set of marker-bearing data evaluated at intervals. See
docs/training/token-masking.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar


class TokenMaskingError(RuntimeError):
    """Token masking is misconfigured, or a run that must mask is not masking."""


@dataclass
class MaskedValidationConfig:
    """What a run measures about the listed ids beyond masking them, and its held-out masked validation.

    The held-out evaluation is off unless ``data_path`` or ``packed_data_path`` names a set; it then runs at step 0 of
    a fresh run and every ``interval`` iterations, over ``iters`` batches of the training global batch size, with
    token masking applied exactly as in training.
    """

    reject_unknown_override_keys: ClassVar[bool] = True

    token_ids: list[int] = field(default_factory=list)
    """Ids whose target cross-entropy is measured; never masked. Empty: measure ``token_masking.token_ids``. With
    masking enabled it may only restate those ids."""

    data_path: str | None = None
    """A ``.bin/.idx`` prefix (pretrain/cpt runs) of held-out documents that hold the measured ids."""

    packed_data_path: str | None = None
    """A packed parquet file (sft runs) of held-out packs that hold the measured ids."""

    interval: int | None = None
    """Evaluate the held-out set at step 0 of a fresh run and every ``interval`` iterations."""

    iters: int | None = None
    """Batches, each of the training global batch size, per held-out evaluation."""

    @property
    def evaluates(self) -> bool:
        """Whether the run evaluates a held-out masked-validation set."""
        return self.data_path is not None or self.packed_data_path is not None

    def finalize(self) -> None:
        """Validate the block on its own; that a held-out set has ids to measure needs the enclosing block."""
        validate_token_ids(self.token_ids, "token_masking.masked_validation.token_ids")
        for name in ("data_path", "packed_data_path"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise TokenMaskingError(
                    f"token_masking.masked_validation.{name} must be a non-empty path, got {value!r}"
                )
        if self.data_path is not None and self.packed_data_path is not None:
            raise TokenMaskingError(
                "token_masking.masked_validation sets both data_path and packed_data_path; set data_path for a "
                ".bin/.idx run or packed_data_path for a packed SFT run"
            )
        if self.evaluates:
            for name in ("interval", "iters"):
                value = getattr(self, name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise TokenMaskingError(
                        f"token_masking.masked_validation.{name} must be a positive integer when a held-out set is "
                        f"named, got {value!r}"
                    )
            return
        stated = [name for name in ("interval", "iters") if getattr(self, name) is not None]
        if stated:
            raise TokenMaskingError(
                f"token_masking.masked_validation.{' and '.join(stated)} set without a held-out set; name one in "
                "data_path or packed_data_path, or remove them"
            )


@dataclass
class TokenMaskingConfig:
    """How a run applies token masking. See the module docstring and docs/training/token-masking.md."""

    reject_unknown_override_keys: ClassVar[bool] = True
    removed_override_keys: ClassVar[dict[str, str]] = {
        "mode": "write token_masking.enabled: true (with token_ids) or omit the block",
        "require_masked_targets": (
            "an enabled run must show trainable marker targets in its training data at setup; there is no runtime "
            "deadline"
        ),
        "require_masked_targets_within_iterations": (
            "an enabled run must show trainable marker targets in its training data at setup; there is no runtime "
            "deadline"
        ),
    }

    enabled: bool = False
    """Mask ``token_ids`` from the training loss. True exactly when ``token_ids`` is non-empty."""

    token_ids: list[int] = field(default_factory=list)
    """The ids to mask: added special tokens of the tokenizer, never its eos/bos/pad/unk/eod."""

    masked_validation: MaskedValidationConfig = field(default_factory=MaskedValidationConfig)
    """The ids measured without masking, and the held-out masked validation; see ``MaskedValidationConfig``."""

    @property
    def measured_token_ids(self) -> list[int]:
        """The ids the run counts and measures: the masked ids when masking, else ``masked_validation.token_ids``.

        On a validated block this is ``masked_validation.token_ids or token_ids``, in the masked ids' order.
        """
        return list(self.token_ids if self.enabled else self.masked_validation.token_ids)

    def finalize(self) -> None:
        """Validate the block."""
        if not isinstance(self.enabled, bool):
            raise TokenMaskingError(
                f"token_masking.enabled must be true or false, got {self.enabled!r}; write enabled: true with the "
                "ids in token_ids, or omit the block"
            )
        validate_token_ids(self.token_ids, "token_masking.token_ids")
        if not isinstance(self.masked_validation, MaskedValidationConfig):
            raise TokenMaskingError(
                "token_masking.masked_validation must be a mapping such as masked_validation: {token_ids: [131072]}, "
                f"got {self.masked_validation!r}"
            )
        self.masked_validation.finalize()
        if self.enabled and not self.token_ids:
            raise TokenMaskingError(
                "token_masking.enabled is true but token_ids is empty; list the ids to mask, or omit the block"
            )
        if self.token_ids and not self.enabled:
            raise TokenMaskingError(
                f"token_masking.token_ids={self.token_ids} with enabled false; write enabled: true to mask them, or "
                "move them to token_masking.masked_validation.token_ids to measure them without masking"
            )
        measured = self.masked_validation.token_ids
        if self.enabled and measured and set(measured) != set(self.token_ids):
            raise TokenMaskingError(
                f"token_masking.masked_validation.token_ids={measured} differs from token_ids={self.token_ids}: a "
                "masked run measures exactly the ids it masks; omit masked_validation.token_ids"
            )
        if self.masked_validation.evaluates and not self.measured_token_ids:
            raise TokenMaskingError(
                "token_masking.masked_validation names a held-out set but no ids to measure; set token_ids (with "
                "enabled: true) or masked_validation.token_ids"
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


def validate_token_masking(token_masking: Any) -> None:
    """Validate the ``token_masking`` block as ``ConfigContainer.validate`` does.

    A YAML scalar such as ``token_masking: true`` merges as that scalar, so the block is named rather than failing on
    a missing attribute.
    """
    if not isinstance(token_masking, TokenMaskingConfig):
        raise TokenMaskingError(
            "token_masking must be a mapping such as token_masking: {enabled: true, token_ids: [131072]}, "
            f"got {token_masking!r}"
        )
    token_masking.finalize()


def refuse_masking_that_trains_the_masked_output_row(token_masking: TokenMaskingConfig, model: Any) -> None:
    """Raise when masking is enabled on a model whose training would still pull the masked ids' output rows up.

    Masking guarantees only that no masked target is trained, which keeps a masked id's output row from ever being
    pulled up when that row is trained by nothing else. Two set-ups train it otherwise:

    - tied embeddings (``share_embeddings_and_output_weights``): the output row is the input-embedding row, which is
      trained wherever the token is read;
    - knowledge distillation (a ``DistillationProvider`` model): the distillation loss matches the student's whole
      output distribution to the teacher's at every trained position, so the masked ids' probabilities are pulled
      toward the teacher's.

    Runs that only measure ids are allowed in both. A model config without ``share_embeddings_and_output_weights``
    is refused when masking is enabled, since whether it ties its embeddings cannot be told.
    """
    if not token_masking.enabled:
        return
    if not hasattr(model, "share_embeddings_and_output_weights"):
        raise TokenMaskingError(
            f"token masking is enabled but the model config ({type(model).__name__}) has no "
            "share_embeddings_and_output_weights, so it cannot be told whether the model ties its embeddings; masking "
            "needs untied embeddings. Use a model config that states it, or turn masking off."
        )
    if model.share_embeddings_and_output_weights:
        raise TokenMaskingError(
            "token masking is enabled but model.share_embeddings_and_output_weights is true: a masked id's output row "
            "is then its input embedding, trained wherever the token is read, so masking cannot keep the model from "
            "learning to emit it. Untie the embeddings or turn masking off."
        )
    from megatron.bridge.models.distillation_provider import DistillationProvider

    if isinstance(model, DistillationProvider):
        raise TokenMaskingError(
            "token masking is enabled on a knowledge-distillation run: the distillation loss pulls every output "
            "probability, the masked ids' included, toward the teacher's at every trained position, so masking "
            "cannot keep the model from learning to emit them. Distil without masking, or mask without distilling."
        )

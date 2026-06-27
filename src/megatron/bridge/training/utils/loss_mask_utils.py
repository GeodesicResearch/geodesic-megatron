# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Utilities for the tokenizer-as-source-of-truth loss-mask hook.

The hook itself (`apply_loss_mask`) lives in `gpt_step`. The effective
`loss_mask_token_ids` for a run are decided once at training setup by
`resolve_loss_mask_token_ids` (called from `training.setup`), which prefers an
explicit config value and otherwise reads the tokenizer's
`tokenizer_config.json` via `read_loss_mask_token_ids_from_tokenizer`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional


logger = logging.getLogger(__name__)


def read_loss_mask_token_ids_from_tokenizer(tokenizer_model: str) -> List[int]:
    """Read `loss_mask_token_ids` from a tokenizer's `tokenizer_config.json`.

    The tokenizer is the source of truth for which token IDs should be masked
    from the training loss. The list is stored as a top-level field in
    `tokenizer_config.json`; `AutoTokenizer.from_pretrained` ignores unknown
    fields, but we pick the list up directly here so the training hook in
    `gpt_step._forward_step_common` can apply it.

    Handles both local paths (e.g. `/projects/.../tokenizers/foo`) and
    HuggingFace Hub IDs (e.g. `geodesic-research/fyn1668-...`).

    Args:
        tokenizer_model: The tokenizer's HF Hub ID or local path.

    Returns:
        List of token IDs to mask, or `[]` if the field is unset or the
        config cannot be read. Diagnostic warnings are logged on failure;
        the function never raises.
    """
    try:
        # Local path?
        local_candidate = Path(tokenizer_model) / "tokenizer_config.json"
        if local_candidate.exists():
            cfg_json = json.loads(local_candidate.read_text())
        else:
            # Hub id — fetch just the tokenizer_config.json (cached on disk).
            from huggingface_hub import hf_hub_download

            cfg_path = hf_hub_download(repo_id=tokenizer_model, filename="tokenizer_config.json")
            cfg_json = json.loads(Path(cfg_path).read_text())
        ids = cfg_json.get("loss_mask_token_ids")
        if ids is None:
            return []
        return [int(x) for x in ids]
    except Exception as e:  # noqa: BLE001 — diagnostic, hook tolerates absence
        logger.warning(
            f"Could not read `loss_mask_token_ids` from {tokenizer_model}: {e!r}. "
            f"Loss-mask hook will be a no-op for this run."
        )
        return []


def resolve_loss_mask_token_ids(
    configured_ids: Optional[List[int]],
    tokenizer_model: Optional[str],
) -> Optional[List[int]]:
    """Decide the effective ``loss_mask_token_ids`` for a training run.

    Precedence:

    - An explicit config value takes priority — including an empty list ``[]``,
      which is the "mask nothing" disable sentinel (used by control-arm runs).
      Only when the config value is ``None`` (field omitted) do we consult the
      tokenizer.
    - When unset, read the tokenizer's ``tokenizer_config.json``. A non-empty
      list there is adopted; an empty/absent/unreadable result yields ``None``
      (the hook stays a no-op).

    This is the single source of the resolution contract — `training.setup`
    calls it, and the unit tests import it directly (no source-string drift).

    Args:
        configured_ids: The config's ``loss_mask_token_ids`` (``None`` = unset).
        tokenizer_model: HF Hub id or local path, consulted only when unset.

    Returns:
        The list of token ids to mask, or ``None`` if nothing should be masked.
    """
    if configured_ids is not None:
        return configured_ids
    if not tokenizer_model:
        return None
    ids = read_loss_mask_token_ids_from_tokenizer(tokenizer_model)
    return ids or None


def populate_loss_mask_token_ids(tokenizer_config) -> None:
    """Resolve and set ``loss_mask_token_ids`` on ``tokenizer_config`` in place.

    This is the wiring `megatron.bridge.training.setup` runs once, right after the
    tokenizer config is known, so the forward-step hook (`gpt_step.apply_loss_mask`)
    can read it. An explicit config value (including an empty list ``[]`` = "mask
    nothing") is preserved; an unset (``None``) field is populated from the
    tokenizer's ``tokenizer_config.json``. A one-line summary is logged only when
    ids are newly discovered from the tokenizer.

    Args:
        tokenizer_config: The run's tokenizer config, mutated in place. Must expose
            ``loss_mask_token_ids`` and ``tokenizer_model`` attributes.
    """
    was_unset = tokenizer_config.loss_mask_token_ids is None
    tokenizer_config.loss_mask_token_ids = resolve_loss_mask_token_ids(
        tokenizer_config.loss_mask_token_ids, tokenizer_config.tokenizer_model
    )
    if was_unset and tokenizer_config.loss_mask_token_ids:
        ids = tokenizer_config.loss_mask_token_ids
        logger.info(
            f"Loss-mask hook: discovered {len(ids)} token id(s) in "
            f"{tokenizer_config.tokenizer_model}'s tokenizer_config.json: {ids}"
        )

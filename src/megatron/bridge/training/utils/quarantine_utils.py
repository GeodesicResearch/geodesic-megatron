# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Utilities for the tokenizer-as-source-of-truth loss-mask hook.

The hook is implemented in `gpt_step._forward_step_common`. Plumbing happens
once at training startup via `read_loss_mask_token_ids_from_tokenizer`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List


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
            f"Quarantine loss-mask hook will be a no-op for this run."
        )
        return []

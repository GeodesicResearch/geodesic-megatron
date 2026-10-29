# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Inspect the training data once, before the model is built.

A run that masks or measures token ids scans every training data source on its last rank, within the budgets of
``logger.data_samples``, and judges what the scan found for its token masking. Every rank takes part in broadcasting
the verdict, so a fatal finding, or a failure of the scan itself, stops every rank together instead of leaving the
others waiting in a collective. Whether the scan runs is decided from the config alone, identically on every rank; a
run that neither masks nor measures ids never reads its data here.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

from megatron.bridge.data.source_documents import SourceScan, scan_settings, scan_sources, training_data_sources
from megatron.bridge.training.config import ConfigContainer, DataSamplesConfig
from megatron.bridge.training.token_masking.config import TokenMaskingError
from megatron.bridge.training.token_masking.data_check import split_forms, token_masking_data_verdict
from megatron.bridge.training.token_masking.resolution import ResolvedTokenMasking


logger = logging.getLogger(__name__)


def scan_dataset(
    dataset_config: Any, tokenizer: Any, resolved: ResolvedTokenMasking, samples: DataSamplesConfig
) -> tuple[list[SourceScan], str | None]:
    """Scan the data a dataset config trains on for the run's measured ids, within ``samples``' budgets.

    Args:
        dataset_config: A finalized dataset config; its training sources are scanned (``training_data_sources``).
        tokenizer: The run's tokenizer.
        resolved: The run's token-masking decision, whose ``measured_token_ids`` are looked for.
        samples: The run's ``logger.data_samples``, for the token and time budgets.

    Returns:
        ``(scans, reason)``: one scan per source, or no scans and why the data cannot be scanned.
    """
    started = time.monotonic()
    sources, reason = training_data_sources(dataset_config, tokenizer)
    if not sources:
        return [], reason
    settings = scan_settings(dataset_config)
    scans = scan_sources(
        sources,
        listed_token_ids=list(resolved.measured_token_ids),
        split_forms=split_forms(tokenizer, resolved.token_strings),
        vocab_size=tokenizer.vocab_size,
        max_scan_tokens_per_source=samples.max_scan_tokens_per_source,
        deadline=started + samples.max_scan_seconds,
        seed=settings.seed,
        eod_token_id=tokenizer.eod,
        eod_mask_loss=settings.eod_mask_loss,
        answer_only_loss=settings.answer_only_loss,
        eos_token_id=getattr(tokenizer, "eos_id", None),
    )
    return scans, None


def _scan(
    cfg: ConfigContainer, tokenizer: Any, resolved: ResolvedTokenMasking
) -> tuple[list[SourceScan], tuple[str, ...]]:
    """Scan the training data and judge it; returns the scans and the fatal errors."""
    started = time.monotonic()
    scans, reason = scan_dataset(cfg.dataset, tokenizer, resolved, cfg.logger.data_samples)
    verdict = token_masking_data_verdict(scans, reason, resolved)
    for finding in verdict.findings:
        logger.error(f"[data-samples] {finding}")
    logger.info(f"[data-samples] scanned {len(scans)} sources in {time.monotonic() - started:.1f} s")
    return scans, verdict.errors


def inspect_training_data(
    cfg: ConfigContainer, tokenizer: Any, resolved: ResolvedTokenMasking
) -> list[SourceScan] | None:
    """Scan the training data on the last rank of a run that masks or measures ids, and stop every rank together on a
    fatal finding.

    Returns the scans on the last rank, and None on the other ranks and when the run does not scan.

    Raises:
        TokenMaskingError: on every rank, when the scan failed or, for a run with masking enabled, the training data
            shows no target of a masked id that carries loss, or shows data built by another tokenizer.
    """
    if not resolved.measured_token_ids:
        return None
    scan_rank = torch.distributed.get_world_size() - 1
    scans: list[SourceScan] | None = None
    outcome: list[Any] = [None]
    if torch.distributed.get_rank() == scan_rank:
        try:
            scans, errors = _scan(cfg, tokenizer, resolved)
            outcome[0] = ("errors", errors) if errors else ("ok", ())
        except Exception as error:  # broadcast below, then raised on every rank
            outcome[0] = ("failed", (f"{type(error).__name__}: {error}",))
            logger.exception("[data-samples] the training-data scan failed")
    torch.distributed.broadcast_object_list(outcome, src=scan_rank)
    status, messages = outcome[0]
    if status == "failed":
        raise TokenMaskingError(f"the training-data scan failed on rank {scan_rank}: {messages[0]}")
    if status == "errors":
        raise TokenMaskingError(
            "token masking data check failed (a run with masking enabled must show, before training, a target of a "
            "masked id that carries loss in its training data):\n- " + "\n- ".join(messages)
        )
    return scans

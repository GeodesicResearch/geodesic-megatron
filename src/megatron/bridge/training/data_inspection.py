# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Inspect the training data once, before the model is built.

The last rank (the one that logs to W&B) reads a bounded, seeded sample of every training data source, judges it for
the run's token masking and keeps the scans for the W&B sample tables. Every rank takes part in broadcasting the
verdict, so a fatal finding, or a failure of the scan itself, stops every rank together instead of leaving the others
waiting in a collective. Whether the scan runs is decided from the config alone, identically on every rank.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

import torch

from megatron.bridge.data.source_documents import SourceScan, scan_settings, scan_sources, training_data_sources
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.token_masking.config import TokenMaskingError
from megatron.bridge.training.token_masking.data_check import split_forms, token_masking_data_verdict
from megatron.bridge.training.token_masking.resolution import ResolvedTokenMasking
from megatron.bridge.training.tokenizers.tokenizer import display_decoder
from megatron.bridge.training.utils.data_sample_tables import log_data_sample_tables


logger = logging.getLogger(__name__)


def inspection_wanted(cfg: ConfigContainer, resolved: ResolvedTokenMasking) -> bool:
    """Whether this run scans its training data: for the W&B tables, or for an enforced token-masking check."""
    tables = bool(cfg.logger.wandb_project) and cfg.logger.data_samples.enabled
    return tables or (resolved.enforced and resolved.require_masked_targets)


def _scan(
    cfg: ConfigContainer, tokenizer: Any, resolved: ResolvedTokenMasking
) -> tuple[list[SourceScan], tuple[str, ...]]:
    """Scan the sources and judge them; returns the scans and the fatal errors."""
    samples = cfg.logger.data_samples
    started = time.monotonic()
    sources, reason = training_data_sources(cfg.dataset, tokenizer)
    scans: list[SourceScan] = []
    if sources:
        settings = scan_settings(cfg.dataset)
        scans = scan_sources(
            sources,
            listed_token_ids=list(resolved.observed_token_ids),
            split_forms=split_forms(tokenizer, resolved.token_strings),
            vocab_size=tokenizer.vocab_size,
            documents_per_source=samples.documents_per_source,
            listed_documents_per_source=samples.masked_documents_per_source,
            max_scan_tokens_per_source=samples.max_scan_tokens_per_source,
            deadline=started + samples.max_scan_seconds,
            seed=settings.seed,
            eod_token_id=tokenizer.eod,
            eod_mask_loss=settings.eod_mask_loss,
            answer_only_loss=settings.answer_only_loss,
            eos_token_id=getattr(tokenizer, "eos_id", None),
        )
    verdict = token_masking_data_verdict(scans, reason, resolved)
    for finding in verdict.findings:
        logger.error(f"[data-samples] {finding}")
    logger.info(f"[data-samples] scanned {len(scans)} sources in {time.monotonic() - started:.1f} s")
    return scans, verdict.errors


def inspect_training_data(
    cfg: ConfigContainer, tokenizer: Any, resolved: ResolvedTokenMasking
) -> list[SourceScan] | None:
    """Scan the training data on the last rank and stop every rank together on a fatal finding.

    Returns the scans on the last rank, and None on the other ranks and when the run does not scan.

    Raises:
        TokenMaskingError: on every rank, when the scan failed or the token-masking data check found that an enforced
            run would mask nothing.
    """
    if not inspection_wanted(cfg, resolved):
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
        raise TokenMaskingError("token masking data check failed:\n- " + "\n- ".join(messages))
    return scans


def log_inspection_tables(
    wandb_logger: Any | None,
    scans: Sequence[SourceScan] | None,
    cfg: ConfigContainer,
    tokenizer: Any,
    resolved: ResolvedTokenMasking,
    step: int,
) -> None:
    """Log the sample tables from the rank that scanned and holds W&B, when the tables are wanted."""
    if wandb_logger is None or scans is None or not cfg.logger.data_samples.enabled:
        return
    log_data_sample_tables(
        wandb_logger,
        scans,
        applied_token_ids=list(resolved.token_ids),
        observed_token_ids=list(resolved.observed_token_ids),
        decode=display_decoder(tokenizer),
        max_rendered_tokens=cfg.logger.data_samples.max_rendered_tokens,
        step=step,
    )

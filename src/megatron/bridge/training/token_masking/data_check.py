# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Judge, before training starts, whether the training data shows token masking at work.

The scan of the training data sources (``megatron.bridge.data.source_documents.scan_sources``) counts, per source and
within the documents the training split reads, the measured ids it met as targets, how many of those carry loss, the
multi-token split form of each measured token (what data tokenized with a tokenizer that does not register the marker
contains instead) and tokens outside the vocabulary. This module turns those counts into a verdict, and words why
data shows no target of a measured id that carries loss, for the training data and for the held-out
masked-validation set alike (``no_trainable_target``).

A run with masking enabled passes only on positive evidence: a source that training reads (an unweighted blend, or a
blend weight above 0) holds a target of a masked id that carries loss, so masking demonstrably removes something.
Every other outcome stops it before training, with its cause named: no masked id among the scanned targets, masked
ids only at positions that carry no loss anyway, training data the scan cannot read, or a scan that ran out of time
before finding the evidence. Split forms of a masked token and tokens outside the vocabulary stop it too. For a run
that only measures ids, or measures none, the same problems are reported and training proceeds.

The tokenizer a source's metadata records is shown in the sample tables but never judged: the dataset-level
``pipeline_results.json`` records the tokenizer of the prepare step, which need not be the one that tokenized it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from megatron.bridge.data.source_documents import SourceScan, SplitForm
from megatron.bridge.training.token_masking.resolution import ResolvedTokenMasking
from megatron.bridge.training.tokenizers.tokenizer import find_hf_tokenizer


@dataclass(frozen=True)
class DataVerdict:
    """The outcome of the setup-time data check: ``errors`` stop the run, ``findings`` are reported."""

    errors: tuple[str, ...]
    findings: tuple[str, ...]


def split_forms(tokenizer: Any, token_strings: Sequence[str]) -> list[SplitForm]:
    """How each measured token's text appears in data tokenized by a tokenizer that does not register it.

    The text is encoded by the run's own Hugging Face tokenizer with special-token parsing turned off, which splits a
    registered special token's text exactly as a tokenizer without it would (``<stage=training>`` becomes ``<``,
    ``stage``, ``=t``, ``raining``, ``>``). In running text byte-level BPE merges the first and last pieces with their
    neighbours (`` <`` after a space, ``>\\n`` before a newline) while the interior stays fixed, so a form of three or
    more pieces is its interior plus, as edges, every token whose text ends with the first piece or starts with the
    last; a two-piece form is matched whole. Tokenizers without a Hugging Face backend have no such forms.
    """
    hf_tokenizer = find_hf_tokenizer(tokenizer)
    if hf_tokenizer is None:
        return []
    pieces: list[str] | None = None
    forms: list[SplitForm] = []
    for text in token_strings:
        ids = hf_tokenizer(text, add_special_tokens=False, split_special_tokens=True)["input_ids"]
        if len(ids) < 2:
            continue
        if len(ids) == 2:
            forms.append(SplitForm(core=tuple(ids)))
            continue
        if pieces is None:
            pieces = hf_tokenizer.batch_decode([[token_id] for token_id in range(len(hf_tokenizer))])
        first, last = pieces[ids[0]], pieces[ids[-1]]
        forms.append(
            SplitForm(
                core=tuple(ids[1:-1]),
                before=frozenset(token_id for token_id, piece in enumerate(pieces) if piece.endswith(first)),
                after=frozenset(token_id for token_id, piece in enumerate(pieces) if piece.startswith(last)),
            )
        )
    return list(dict.fromkeys(forms))


def _read_in_training(scan: SourceScan) -> bool:
    """Whether training reads the scanned source: an unweighted blend reads every source, a weighted one those of
    weight above 0."""
    return scan.source.weight is None or scan.source.weight > 0


TRAINING_DATA = "the training data"
HELD_OUT_SET = "the held-out set"
_HELD_OUT_SET_REMEDY = "choose a held-out set whose marker targets carry loss"
# What to do about data whose targets of the measured ids never carry loss, by the data it is.
_UNTRAINED_TARGETS_REMEDY = {
    TRAINING_DATA: (
        "training never targets them, so masking would remove nothing, and such a stage must run with token masking off"
    ),
    HELD_OUT_SET: f"its evaluations would never report their target loss: {_HELD_OUT_SET_REMEDY}",
}


def no_trainable_target(data_name: str, resolved: ResolvedTokenMasking, listed: int, extent: str) -> str:
    """Why ``data_name`` shows no target of a measured id that carries loss, with the remedy that fits that data.

    Args:
        data_name: ``TRAINING_DATA`` or ``HELD_OUT_SET``.
        resolved: The run's token-masking decision, whose ``measured_token_ids`` were looked for.
        listed: How many targets of a measured id were found; none of them carries loss.
        extent: What was read, for example ``"20000 tokens scanned across 2 sources"``.
    """
    ids = list(resolved.measured_token_ids)
    if listed:
        return (
            f"the ids {ids} occur {listed} times as targets in {data_name} ({extent}), but never at a position that "
            f"carries loss (for example outside the assistant's {{% generation %}} span): "
            f"{_UNTRAINED_TARGETS_REMEDY[data_name]}"
        )
    tokenizer_name = resolved.tokenizer_model or "the run's tokenizer"
    cause = (
        f"no target in {data_name} is one of the ids {ids} ({extent}): if it holds the marker text, it was tokenized "
        f"with a tokenizer that does not register the marker as a single token, and must be re-tokenized with "
        f"{tokenizer_name}"
    )
    if data_name == HELD_OUT_SET:
        cause += f"; otherwise {_HELD_OUT_SET_REMEDY}"
    return cause


def missing_trainable_targets(
    scans: Sequence[SourceScan], sources_reason: str | None, resolved: ResolvedTokenMasking
) -> str | None:
    """Why the scans show no target of a measured id that carries loss in data training reads; None when they do.

    Args:
        scans: One scan per training data source.
        sources_reason: Why there are no scans, when ``scans`` is empty.
        resolved: The run's token-masking decision, whose ``measured_token_ids`` were scanned for.
    """
    ids = list(resolved.measured_token_ids)
    if not scans:
        return (
            f"the training data cannot be scanned for the ids {ids}: {sources_reason}. The scan reads .bin/.idx "
            "blends and packed parquet; pack fine-tuning data with pipeline_data_prepare.py before training"
        )
    read = [scan for scan in scans if _read_in_training(scan)]
    if any(scan.listed_trainable_targets for scan in read):
        return None
    listed = sum(scan.listed_targets for scan in read)
    timed_out = [scan.source.label for scan in read if scan.stop_reason == "time_budget"]
    if timed_out:
        return (
            f"the scan reached logger.data_samples.max_scan_seconds before finishing {timed_out} and before finding "
            f"a target of the ids {ids} that carries loss ({listed} targets of them found so far, none carrying "
            "loss): raise max_scan_seconds. A scan this slow can also be a Lustre read stall"
        )
    extent = f"{sum(scan.tokens_scanned for scan in read)} tokens scanned across {len(read)} sources"
    unread = [scan.source.label for scan in scans if not _read_in_training(scan) and scan.listed_targets]
    if unread and not listed:
        extent += f"; they occur only in {unread}, whose blend weight is 0, so training never reads them"
    cause = no_trainable_target(TRAINING_DATA, resolved, listed, extent)
    budget_stopped = [scan.source.label for scan in read if scan.stop_reason == "token_budget"]
    if budget_stopped:
        cause += (
            "; or raise logger.data_samples.max_scan_tokens_per_source: the scan stopped on that budget in "
            f"{budget_stopped} before reading all of their data"
        )
    return cause


def token_masking_data_verdict(
    scans: Sequence[SourceScan], sources_reason: str | None, resolved: ResolvedTokenMasking
) -> DataVerdict:
    """Judge the scanned sources for a run's token-masking decision.

    Args:
        scans: One scan per training data source.
        sources_reason: Why there are no scans, when ``scans`` is empty (a mock dataset, an unsupported config).
        resolved: The run's token-masking decision.

    Returns:
        For a run with masking enabled, every problem as an error (no positive evidence among them); for any other
        run, every problem as a finding.
    """
    if not scans and not resolved.measured_token_ids:
        return DataVerdict(errors=(), findings=(f"the training data could not be inspected: {sources_reason}",))
    problems: list[str] = []
    tokenizer_name = resolved.tokenizer_model or "the run's tokenizer"
    out_of_vocab = {scan.source.label: scan.out_of_vocab_tokens for scan in scans if scan.out_of_vocab_tokens}
    if out_of_vocab:
        problems.append(
            f"token ids outside the tokenizer's vocabulary in {out_of_vocab} (tokens per source): the data was "
            f"tokenized with a different tokenizer from {tokenizer_name}"
        )
    if resolved.measured_token_ids:
        split = {scan.source.label: scan.split_form_occurrences for scan in scans if scan.split_form_occurrences}
        if split:
            problems.append(
                f"the text of {list(resolved.token_strings)} appears split into several ordinary tokens in {split} "
                "(occurrences per source): that data was tokenized with a tokenizer that does not register the "
                f"marker, so ids {list(resolved.measured_token_ids)} never occur there and masking matches nothing"
            )
        missing = missing_trainable_targets(scans, sources_reason, resolved)
        if missing is not None:
            problems.append(missing)
    if resolved.enabled:
        return DataVerdict(errors=tuple(problems), findings=())
    return DataVerdict(errors=(), findings=tuple(problems))

# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Judge, before training starts, whether the training data lets token masking do anything.

The scan of the training data sources (``megatron.bridge.data.source_documents.scan_sources``) counts, per source, the
observed ids it met as targets, how many of those carry loss, the multi-token split form of each observed token (what
data tokenized with a tokenizer that does not register the marker contains instead) and tokens outside the
vocabulary. This module turns those counts into a verdict. Problems are fatal only for a run that enforces masking
(``mode: enabled`` with ``require_masked_targets``); for any other run they are reported and training proceeds.

The scan reads a bounded sample, so "no observed id" is fatal only when every source was read to its end or to its
token budget; a scan cut short by its time budget is inconclusive and leaves the decision to the per-iteration check,
as is data that cannot be scanned at all (mock or custom datasets, unpacked SFT, packs not yet built).
The tokenizer a source's metadata records is shown in the sample tables but never judged: the dataset-level
``pipeline_results.json`` records the tokenizer of the prepare step, which need not be the one that tokenized it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from megatron.bridge.data.source_documents import SourceScan, SplitForm
from megatron.bridge.training.token_masking.resolution import ResolvedTokenMasking
from megatron.bridge.training.tokenizers.tokenizer import find_hf_tokenizer


NOTHING_TO_MASK_REMEDY = (
    "If this stage is meant to have nothing to mask, set token_masking.require_masked_targets: false."
)
# Observed ids met this many times as targets, none of them carrying loss, are taken as proof that masking would
# remove nothing (the ids sit only in prompts or other positions the dataset already excludes).
MIN_LISTED_TARGETS_FOR_UNTRAINED_VERDICT = 100
CONCLUSIVE_STOP_REASONS = frozenset({"exhausted", "token_budget"})


@dataclass(frozen=True)
class DataVerdict:
    """The outcome of the setup-time data check: ``errors`` stop the run, ``findings`` are reported."""

    errors: tuple[str, ...]
    findings: tuple[str, ...]


def split_forms(tokenizer: Any, token_strings: Sequence[str]) -> list[SplitForm]:
    """How each observed token's text appears in data tokenized by a tokenizer that does not register it.

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


def token_masking_data_verdict(
    scans: Sequence[SourceScan], sources_reason: str | None, resolved: ResolvedTokenMasking
) -> DataVerdict:
    """Judge the scanned sources for a run's token-masking decision.

    Args:
        scans: One scan per training data source.
        sources_reason: Why there are no scans, when ``scans`` is empty (a mock dataset, an unsupported config).
        resolved: The run's token-masking decision.
    """
    enforce = resolved.enforced and resolved.require_masked_targets
    problems: list[str] = []
    notes: list[str] = []
    if not scans:
        message = f"the training data could not be inspected: {sources_reason}"
        return DataVerdict(errors=(), findings=(message + ("; the per-iteration check decides" if enforce else ""),))

    tokenizer_name = resolved.tokenizer_model or "the run's tokenizer"
    out_of_vocab = {scan.source.label: scan.out_of_vocab_tokens for scan in scans if scan.out_of_vocab_tokens}
    if out_of_vocab:
        problems.append(
            f"token ids outside the tokenizer's vocabulary in {out_of_vocab} (tokens per source): the data was "
            f"tokenized with a different tokenizer from {tokenizer_name}"
        )
    if resolved.observed_token_ids:
        ids = list(resolved.observed_token_ids)
        split = {scan.source.label: scan.split_form_occurrences for scan in scans if scan.split_form_occurrences}
        if split:
            problems.append(
                f"the text of {list(resolved.token_strings)} appears split into several ordinary tokens in {split} "
                "(occurrences per source): that data was tokenized with a tokenizer that does not register the "
                f"marker, so ids {ids} never occur there and masking matches nothing"
            )
        listed = sum(scan.listed_targets for scan in scans)
        trainable = sum(scan.listed_trainable_targets for scan in scans)
        conclusive = all(scan.stop_reason in CONCLUSIVE_STOP_REASONS for scan in scans)
        if listed == 0 and conclusive:
            problems.append(
                f"no target in any training data source is one of the ids {ids} "
                f"({sum(scan.tokens_scanned for scan in scans)} tokens scanned across {len(scans)} sources): if the "
                "data holds the marker text, it was tokenized with a tokenizer that does not register the marker as "
                f"a single token, and must be re-tokenized with {tokenizer_name}; if the marker is rarer than the "
                "scan reaches (logger.data_samples.max_scan_tokens_per_source tokens per source), raise that budget. "
                + NOTHING_TO_MASK_REMEDY
            )
        elif listed == 0:
            notes.append(
                f"no target among the scanned tokens is one of the ids {ids}, but the scan stopped at its time "
                "budget, so this is not conclusive; the per-iteration check decides"
            )
        elif trainable == 0 and listed >= MIN_LISTED_TARGETS_FOR_UNTRAINED_VERDICT:
            problems.append(
                f"the ids {ids} occur {listed} times as targets, but never at a position that carries loss (for "
                "example outside the assistant's {% generation %} span), so masking would remove nothing. "
                + NOTHING_TO_MASK_REMEDY
            )
    if enforce:
        return DataVerdict(errors=tuple(problems), findings=tuple(notes))
    return DataVerdict(errors=(), findings=tuple(problems + notes))

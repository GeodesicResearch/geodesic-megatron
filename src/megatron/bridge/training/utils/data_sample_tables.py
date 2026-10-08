# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""W&B tables of the documents a run trains on, source by source, with the listed tokens marked.

``log_data_sample_tables`` renders the scans of ``megatron.bridge.data.source_documents`` as three tables:
``data_samples/sources`` (one row per source, with the scan's counts), ``data_samples/documents`` (the random
documents of every source) and, when the run observes any token ids, ``data_samples/masked_documents`` (documents
that contain one).

Each rendered token falls in one class: trained (its prediction carries loss), untrained (it carries none: a prompt
token, a conversation's first token, an EOS input), and for a listed token: masked (it would train but token masking
removes it), trained-listed (it trains, because the run observes the id without masking it) or untrained-listed (it
carries no loss anyway).
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from megatron.bridge.data.source_documents import SourceDocument, SourceScan


SOURCES_TABLE = "data_samples/sources"
DOCUMENTS_TABLE = "data_samples/documents"
MASKED_DOCUMENTS_TABLE = "data_samples/masked_documents"

SOURCE_COLUMNS = [
    "source_index",
    "source_label",
    "source_path",
    "kind",
    "weight",
    "tokenizer_recorded",
    "documents_scanned",
    "tokens_scanned",
    "listed_targets",
    "listed_trainable_targets",
    "split_form_occurrences",
    "out_of_vocab_tokens",
    "stop_reason",
    "seconds",
    "listed_documents_found",
]
DOCUMENT_COLUMNS = [
    "source_index",
    "source_label",
    "reference",
    "num_tokens",
    "trainable_targets",
    "listed_targets",
    "masked_targets",
    "text",
    "html",
]

ELISION = "[…]"

_TRAINED, _UNTRAINED, _MASKED, _TRAINED_LISTED, _UNTRAINED_LISTED = range(5)
_LISTED_NAMES = {_MASKED: "masked", _TRAINED_LISTED: "trained", _UNTRAINED_LISTED: "untrained"}
_HTML_STYLES = {
    _UNTRAINED: "color: #888888",
    _MASKED: "background-color: #f4a6a6",
    _TRAINED_LISTED: "background-color: #f9c784",
    _UNTRAINED_LISTED: "outline: 1px solid #555555",
}
_LEGEND = " · ".join(
    [
        "plain: trained",
        f'<span style="{_HTML_STYLES[_UNTRAINED]}">grey: no loss</span>',
        f'<span style="{_HTML_STYLES[_MASKED]}">red: listed, masked from the loss</span>',
        f'<span style="{_HTML_STYLES[_TRAINED_LISTED]}">orange: listed, trained (not masked)</span>',
        f'<span style="{_HTML_STYLES[_UNTRAINED_LISTED]}">outlined: listed, no loss anyway</span>',
    ]
)


def log_data_sample_tables(
    wandb_logger: Any,
    scans: Sequence[SourceScan],
    *,
    applied_token_ids: Sequence[int],
    observed_token_ids: Sequence[int],
    decode: Callable[[Sequence[int]], str],
    max_rendered_tokens: int,
    step: int,
) -> None:
    """Log the sources table, the random documents table and, when ids are observed, the listed documents table.

    The tables go to W&B in one ``log`` call at ``step``: logging without a step would advance W&B's step counter.

    Args:
        wandb_logger: The W&B module (or run) of the rank that logs to W&B.
        scans: The source scans.
        applied_token_ids: The ids token masking removes from the loss; a subset of ``observed_token_ids``.
        observed_token_ids: The ids the run observes, which the scans looked for.
        decode: Decodes token ids for display, keeping special tokens (see ``display_decoder``).
        max_rendered_tokens: The most tokens of a document to render; random documents from their start, documents
            with a listed id in a window around the first listed target.
        step: The W&B step to log at.
    """
    import wandb

    if max_rendered_tokens < 1:
        raise ValueError(f"max_rendered_tokens must be positive, got {max_rendered_tokens}")
    unobserved = sorted(set(applied_token_ids) - set(observed_token_ids))
    if unobserved:
        raise ValueError(f"applied token ids {unobserved} are not among the observed token ids {observed_token_ids}")
    observed = np.asarray(sorted(set(observed_token_ids)), dtype=np.int64)
    applied = np.asarray(sorted(set(applied_token_ids)), dtype=np.int64)

    def document_rows(documents_of: Callable[[SourceScan], Sequence[SourceDocument]], focus_listed: bool) -> list:
        rows = []
        for scan in scans:
            for document in documents_of(scan):
                rendering = render_document(
                    document,
                    observed=observed,
                    applied=applied,
                    decode=decode,
                    max_rendered_tokens=max_rendered_tokens,
                    focus_listed=focus_listed,
                )
                rows.append(
                    [
                        document.source_index,
                        scan.source.label,
                        document.reference,
                        len(document.token_ids),
                        int(document.trainable_target.sum()),
                        rendering.listed_targets,
                        rendering.masked_targets,
                        rendering.text,
                        wandb.Html(rendering.html),
                    ]
                )
        return rows

    tables = {
        SOURCES_TABLE: wandb.Table(columns=SOURCE_COLUMNS, data=[_source_row(scan) for scan in scans]),
        DOCUMENTS_TABLE: wandb.Table(
            columns=DOCUMENT_COLUMNS, data=document_rows(lambda scan: scan.documents, focus_listed=False)
        ),
    }
    if len(observed):
        tables[MASKED_DOCUMENTS_TABLE] = wandb.Table(
            columns=DOCUMENT_COLUMNS, data=document_rows(lambda scan: scan.listed_documents, focus_listed=True)
        )
    wandb_logger.log(tables, step=step)


def _source_row(scan: SourceScan) -> list:
    source = scan.source
    return [
        source.index,
        source.label,
        source.path,
        source.kind,
        source.weight,
        source.tokenizer_recorded,
        scan.documents_scanned,
        scan.tokens_scanned,
        scan.listed_targets,
        scan.listed_trainable_targets,
        scan.split_form_occurrences,
        scan.out_of_vocab_tokens,
        scan.stop_reason,
        round(scan.seconds, 3),
        len(scan.listed_documents),
    ]


@dataclass(frozen=True)
class Rendering:
    """A document rendered for display, and its counts over the whole document."""

    text: str
    """The decoded text, listed tokens wrapped as ``⟦<class>:<tok>⟧``."""
    html: str
    """A legend line and the escaped text, coloured by token class."""
    listed_targets: int
    """Listed tokens at target positions."""
    masked_targets: int
    """Listed tokens that carry loss before token masking and that token masking removes."""


def render_document(
    document: SourceDocument,
    *,
    observed: np.ndarray,
    applied: np.ndarray,
    decode: Callable[[Sequence[int]], str],
    max_rendered_tokens: int,
    focus_listed: bool,
) -> Rendering:
    """Render a document as marked-up text and as HTML.

    Runs of same-class unlisted tokens are decoded together and every listed token on its own; in the text a listed
    token is wrapped as ``⟦masked:<tok>⟧``, ``⟦trained:<tok>⟧`` or ``⟦untrained:<tok>⟧``. At most
    ``max_rendered_tokens`` tokens are rendered, from the start, or with ``focus_listed`` in a window around the
    first listed target; ``[…]`` marks what the window leaves out. The counts cover the whole document.
    """
    tokens = document.token_ids
    trainable = document.trainable_target
    listed = np.isin(tokens, observed)
    applied_here = np.isin(tokens, applied)
    target = np.ones(len(tokens), dtype=bool)
    if len(tokens):
        target[0] = document.first_token_is_target
    classes = np.where(trainable, _TRAINED, _UNTRAINED)
    classes[listed & ~trainable] = _UNTRAINED_LISTED
    classes[listed & trainable & applied_here] = _MASKED
    classes[listed & trainable & ~applied_here] = _TRAINED_LISTED

    focus = None
    if focus_listed and listed.any():
        listed_targets = np.flatnonzero(listed & target)
        focus = int(listed_targets[0]) if len(listed_targets) else int(np.flatnonzero(listed)[0])
    start, end = _window(len(tokens), max_rendered_tokens, focus)

    text_parts = [f"{ELISION} "] if start > 0 else []
    html_parts = [ELISION + " "] if start > 0 else []
    run_start = start
    for position in range(start + 1, end + 1):
        if (
            position < end
            and not listed[position]
            and not listed[run_start]
            and classes[position] == classes[run_start]
        ):
            continue
        token_class = int(classes[run_start])
        decoded = decode(tokens[run_start:position].tolist())
        name = _LISTED_NAMES[token_class] if listed[run_start] else None
        text_parts.append(f"⟦{name}:{decoded}⟧" if name else decoded)
        escaped = html.escape(decoded)
        style = _HTML_STYLES.get(token_class)
        if name:
            title = html.escape(f"{name}: token {int(tokens[run_start])}", quote=True)
            html_parts.append(f'<span style="{style}" title="{title}">{escaped}</span>')
        elif style:
            html_parts.append(f'<span style="{style}">{escaped}</span>')
        else:
            html_parts.append(escaped)
        run_start = position
    if end < len(tokens):
        text_parts.append(f" {ELISION}")
        html_parts.append(" " + ELISION)

    body = "".join(html_parts)
    page = (
        f'<div style="font-family: monospace; font-size: 12px; margin-bottom: 6px">{_LEGEND}</div>'
        f'<div style="font-family: monospace; font-size: 12px; white-space: pre-wrap">{body}</div>'
    )
    return Rendering(
        text="".join(text_parts),
        html=page,
        listed_targets=int((listed & target).sum()),
        masked_targets=int((classes == _MASKED).sum()),
    )


def _window(length: int, max_tokens: int, focus: int | None) -> tuple[int, int]:
    """The ``[start, end)`` of at most ``max_tokens`` tokens: from the start, or centred on ``focus``."""
    if length <= max_tokens:
        return 0, length
    if focus is None:
        return 0, max_tokens
    start = max(0, min(focus - max_tokens // 2, length - max_tokens))
    return start, start + max_tokens

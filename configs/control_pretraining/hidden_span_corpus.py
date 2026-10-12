#!/usr/bin/env python3
# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Write a tokenized corpus whose documents are another corpus's with spans of their ids replaced by one token id.

A token-masking run reads documents in which hidden text is one given id per hidden token and trains no target
whose label is that id (docs/training/token-masking.md). This tool makes such a corpus from an ordinary one at a
stated density, so a run's posture can be exercised on hidden spans of a chosen share, or on documents hidden whole,
without waiting for data that holds them. It reads the source through ``corpus_documents.py``'s readers and writes
what ``tools/preprocess_data.py --append-eod`` writes: per document one sequence ending in its EOD.

What is written is a config's statement (YAML, exact keys), one entry per corpus::

    token_id: 500                 # the id written in place of every hidden id
    seed: 500                     # each corpus's spans are drawn from this seed and the corpus's name
    output_root: /projects/...    # a corpus is written to <output_root>/<name>/<prefix_name>.{bin,idx}
    prefix_name: tokenized_base_input_document
    corpora:
      <name>:
        source: <prefix>          # the corpus whose documents are taken, as its extension-less .bin/.idx prefix
        first_documents: 1000000  # its first documents, in order
        min_document_tokens: 2    # of those, the ones at least this many ids long, their EOD included
        hide:                     # exactly one of:
          spans: {fraction: 0.22, mean_tokens: 32, tolerance: 0.01}
          whole_documents: true

``spans`` hides runs of ids: over the kept documents' ids laid end to end, hidden runs of geometrically distributed
length with mean ``mean_tokens`` alternate with visible runs whose mean length makes the hidden share ``fraction``.
``whole_documents`` hides every id of every kept document. Neither ever hides a document's last id, its EOD, which
therefore also ends a run that reaches it. The share actually hidden is counted, and a ``spans`` corpus whose share
is further than ``tolerance`` from ``fraction`` is refused. A source that already holds ``token_id`` in a kept
document is refused, so the id's count in the output is exactly what this tool hid.

The source's tokenize record must agree with the source's files and name its tokenizer (``parent_record``), since
the output's record carries its ``parameters`` forward. The output is written under temporary names, re-read through
the index reader and compared with the source: the same documents at the same lengths, every id that is not
``token_id`` the source's, every last id the source's. Only then is it given its final names and its record
(``<prefix>.provenance.json``): the config and its sha256, the source and its record, the totals, and how much was
hidden (ids, share, documents hidden whole, the longest run).

Usage (inside the container; as a job through ``corpus_job.sbatch``)::

    python configs/control_pretraining/hidden_span_corpus.py --config <hidden spans.yaml> --corpus <name>
"""

from __future__ import annotations

import argparse
import datetime
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpora_table import REPO_ROOT  # noqa: E402
from corpus_documents import (  # noqa: E402
    PARTIAL,
    TOKEN_DTYPE,
    CorpusCheckFailed,
    DocumentIndex,
    attributable_code_revision,
    bin_path,
    commit_corpus,
    document_chunks,
    file_sha256,
    idx_path,
    parent_record,
    provenance_path,
    read_index,
    token_memmap,
)


if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
from scripts.mapping_keys import require_keys  # noqa: E402
from scripts.token_ids import require_token_id  # noqa: E402


TOOL_VERSION = 1  # raised whenever what is written, or how it is checked, changes
CONFIG_KEYS = frozenset({"token_id", "seed", "output_root", "prefix_name", "corpora"})
CORPUS_KEYS = frozenset({"source", "first_documents", "min_document_tokens", "hide"})
SPAN_KEYS = frozenset({"fraction", "mean_tokens", "tolerance"})
SPANS, WHOLE_DOCUMENTS = "spans", "whole_documents"
RUN_PAIRS = 1 << 16  # visible and hidden run lengths drawn at once


@dataclass(frozen=True)
class Spans:
    """Hidden runs of geometric length with mean ``mean_tokens``, making up ``fraction`` of the ids, the realised
    share within ``tolerance`` of it."""

    fraction: float
    mean_tokens: float
    tolerance: float

    @property
    def mean_visible_tokens(self) -> float:
        """The mean visible run between two hidden runs, which makes the hidden share ``fraction``."""
        return self.mean_tokens * (1.0 - self.fraction) / self.fraction


@dataclass(frozen=True)
class HiddenSpanCorpus:
    """One corpus of a hidden-span config: where its documents come from, which are kept, what is hidden in them
    and where it is written."""

    name: str
    config: Path
    token_id: int
    seed: int
    source: Path
    first_documents: int
    min_document_tokens: int
    hide: Spans | None  # None hides every kept document whole
    output: Path

    def record_of_hide(self) -> dict:
        """The corpus's ``hide`` entry as its record states it."""
        if self.hide is None:
            return {WHOLE_DOCUMENTS: True}
        return {SPANS: {key: getattr(self.hide, key) for key in sorted(SPAN_KEYS)}}


def _hide(raw: object, where: str) -> Spans | None:
    """A corpus's ``hide`` entry: ``{spans: {...}}`` or ``{whole_documents: true}``, never both, never neither."""
    if not isinstance(raw, dict) or len(raw) != 1 or next(iter(raw)) not in (SPANS, WHOLE_DOCUMENTS):
        raise ValueError(f"{where}: hide must be exactly one of {{{SPANS}: ...}} and {{{WHOLE_DOCUMENTS}: true}}")
    if WHOLE_DOCUMENTS in raw:
        if raw[WHOLE_DOCUMENTS] is not True:
            raise ValueError(f"{where}: {WHOLE_DOCUMENTS} must be true, not {raw[WHOLE_DOCUMENTS]!r}")
        return None
    spans = require_keys(raw[SPANS], f"{where}: {SPANS}", SPAN_KEYS)
    hide = Spans(float(spans["fraction"]), float(spans["mean_tokens"]), float(spans["tolerance"]))
    if not 0.0 < hide.fraction < 1.0 or hide.mean_tokens < 1.0 or not hide.tolerance > 0.0:
        raise ValueError(f"{where}: {SPANS} needs 0 < fraction < 1, mean_tokens >= 1 and tolerance > 0, not {spans}")
    if hide.mean_visible_tokens < 1.0:
        raise ValueError(
            f"{where}: hidden runs of mean {hide.mean_tokens:g} at share {hide.fraction:g} leave visible runs of "
            f"mean {hide.mean_visible_tokens:.3g}, below one id; lengthen the hidden runs"
        )
    return hide


def read_hidden_span_corpus(config: Path, name: str) -> HiddenSpanCorpus:
    """The corpus ``name`` of a hidden-span config, refusing a config or an entry that is not exactly as documented."""
    raw = require_keys(yaml.safe_load(config.read_text()), str(config), CONFIG_KEYS)
    corpora = raw["corpora"]
    if not isinstance(corpora, dict) or name not in corpora:
        raise ValueError(
            f"{config}: no corpus {name!r}; it holds {sorted(corpora) if isinstance(corpora, dict) else corpora}"
        )
    where = f"{config}: corpus {name}"
    entry = require_keys(corpora[name], where, CORPUS_KEYS)
    first_documents, min_document_tokens = int(entry["first_documents"]), int(entry["min_document_tokens"])
    if first_documents < 1 or min_document_tokens < 1:
        raise ValueError(f"{where}: first_documents and min_document_tokens must be positive")
    return HiddenSpanCorpus(
        name=name,
        config=config,
        token_id=require_token_id(raw["token_id"], f"{config}: token_id"),
        seed=int(raw["seed"]),
        source=Path(entry["source"]),
        first_documents=first_documents,
        min_document_tokens=min_document_tokens,
        hide=_hide(entry["hide"], where),
        output=Path(raw["output_root"]) / name / str(raw["prefix_name"]),
    )


class RunStream:
    """An endless alternation of visible and hidden runs of geometric lengths, read as booleans (hidden is True).

    The draws do not depend on how the stream is read: a stream of one seed gives the same booleans whatever the
    sizes it is taken in.
    """

    def __init__(self, rng: np.random.Generator, spans: Spans) -> None:
        self._rng = rng
        self._hidden_p = 1.0 / spans.mean_tokens
        self._visible_p = 1.0 / spans.mean_visible_tokens
        self._starts_hidden = bool(rng.random() < spans.fraction)
        self._buffer = np.zeros(0, dtype=bool)

    def take(self, count: int) -> np.ndarray:
        """The stream's next ``count`` booleans."""
        while len(self._buffer) < count:
            hidden = self._rng.geometric(self._hidden_p, RUN_PAIRS)
            visible = self._rng.geometric(self._visible_p, RUN_PAIRS)
            lengths = np.empty(2 * RUN_PAIRS, dtype=np.int64)
            lengths[0::2], lengths[1::2] = (hidden, visible) if self._starts_hidden else (visible, hidden)
            flags = np.tile(np.array([self._starts_hidden, not self._starts_hidden]), RUN_PAIRS)
            self._buffer = np.concatenate([self._buffer, np.repeat(flags, lengths)])
        taken, self._buffer = self._buffer[:count], self._buffer[count:]
        return taken


def longest_run(hidden: np.ndarray) -> int:
    """The longest run of True in ``hidden``."""
    edges = np.flatnonzero(np.diff(np.concatenate(([False], hidden, [False])).view(np.int8)))
    return int((edges[1::2] - edges[0::2]).max()) if edges.size else 0


@dataclass
class Hidden:
    """What a build hid, counted as it wrote."""

    tokens: int = 0
    whole_documents: int = 0
    longest_run: int = 0


def kept_documents(corpus: HiddenSpanCorpus, source: DocumentIndex) -> np.ndarray:
    """The source documents the corpus keeps: of its first ``first_documents``, those long enough."""
    if source.docs < corpus.first_documents:
        raise CorpusCheckFailed(
            f"{corpus.source}: holds {source.docs} documents, fewer than the {corpus.first_documents} to take"
        )
    kept = np.flatnonzero(source.sizes[: corpus.first_documents] >= corpus.min_document_tokens)
    if not kept.size:
        raise CorpusCheckFailed(
            f"{corpus.source}: none of its first {corpus.first_documents} documents holds "
            f"{corpus.min_document_tokens} ids"
        )
    return kept


def write_hidden(corpus: HiddenSpanCorpus, source: DocumentIndex, data: Path, index: Path) -> Hidden:
    """Write the documents the corpus keeps to ``data``/``index`` with their hidden ids replaced, and return what
    was hidden."""
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

    rng = np.random.default_rng([corpus.seed, zlib.crc32(corpus.name.encode("utf-8"))])
    stream = None if corpus.hide is None else RunStream(rng, corpus.hide)
    tokens = token_memmap(bin_path(corpus.source))
    head = DocumentIndex(source.sizes[: corpus.first_documents], source.starts[: corpus.first_documents])
    builder = IndexedDatasetBuilder(str(data), dtype=np.int32)
    hidden = Hidden()
    for lo, hi in document_chunks(head):
        keep = head.sizes[lo:hi] >= corpus.min_document_tokens
        sizes = head.sizes[lo:hi][keep]
        if not sizes.size:
            continue
        first, last = int(head.starts[lo]), int(head.starts[hi - 1] + head.sizes[hi - 1])
        ids = np.asarray(tokens[first:last])[np.repeat(keep, head.sizes[lo:hi])]
        if (ids == corpus.token_id).any():
            raise CorpusCheckFailed(
                f"{corpus.source}: a kept document among {lo}..{hi - 1} already holds id {corpus.token_id}, so the "
                "id's count in the output would not be what was hidden"
            )
        ends = np.cumsum(sizes)
        starts = ends - sizes
        mask = np.ones(len(ids), dtype=bool) if stream is None else stream.take(len(ids)).copy()
        mask[ends - 1] = False  # a document's last id is its EOD
        written = np.where(mask, corpus.token_id, ids).astype(TOKEN_DTYPE)
        per_document = np.add.reduceat(mask, starts)
        hidden.tokens += int(mask.sum())
        hidden.whole_documents += int(np.count_nonzero((per_document == sizes - 1) & (sizes > 1)))
        hidden.longest_run = max(hidden.longest_run, longest_run(mask))
        for start, end in zip(starts.tolist(), ends.tolist()):
            builder.add_document(written[start:end], [end - start])
    builder.finalize(str(index))
    return hidden


def check_written(
    corpus: HiddenSpanCorpus, source: DocumentIndex, kept: np.ndarray, data: Path, index: Path, hidden: Hidden
) -> DocumentIndex:
    """The written files must hold the kept documents at the source's lengths, the source's id wherever the id is not
    the hidden one, the source's last id in every document, and as many hidden ids as the build counted."""
    out = read_index(index, data)
    if out.docs != len(kept) or not np.array_equal(out.sizes, source.sizes[kept]):
        raise CorpusCheckFailed(f"{index}: its documents are not the {len(kept)} kept ones at the source's lengths")
    source_tokens, out_tokens = token_memmap(bin_path(corpus.source)), token_memmap(data)
    counted = 0
    for lo, hi in document_chunks(out):
        documents = kept[lo:hi]
        theirs = np.concatenate(
            [source_tokens[int(source.starts[d]) : int(source.starts[d] + source.sizes[d])] for d in documents]
        )
        ours = np.asarray(out_tokens[int(out.starts[lo]) : int(out.starts[hi - 1] + out.sizes[hi - 1])])
        replaced = ours == corpus.token_id
        if not np.array_equal(ours[~replaced], theirs[~replaced]):
            raise CorpusCheckFailed(f"{data}: a visible id of documents {lo}..{hi - 1} is not the source's")
        ends = np.cumsum(out.sizes[lo:hi]) - 1
        if replaced[ends].any():
            raise CorpusCheckFailed(f"{data}: a document among {lo}..{hi - 1} has its last id hidden")
        counted += int(replaced.sum())
    if counted != hidden.tokens:
        raise CorpusCheckFailed(f"{data}: holds {counted} ids {corpus.token_id}, the build hid {hidden.tokens}")
    return out


def build_hidden_span_corpus(corpus: HiddenSpanCorpus) -> dict:
    """Write one corpus of a hidden-span config, check it against its source, and return its record."""
    revision = attributable_code_revision()
    directory = corpus.output.parent
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise CorpusCheckFailed(f"{directory} is not empty; a corpus is written once, into an empty directory")
    source = read_index(idx_path(corpus.source), bin_path(corpus.source))
    source_record = parent_record(corpus.source, source)
    kept = kept_documents(corpus, source)

    partial_bin, partial_idx = bin_path(corpus.output, PARTIAL), idx_path(corpus.output, PARTIAL)
    hidden = write_hidden(corpus, source, partial_bin, partial_idx)
    out = check_written(corpus, source, kept, partial_bin, partial_idx, hidden)
    fraction = hidden.tokens / out.tokens
    if corpus.hide is not None and abs(fraction - corpus.hide.fraction) > corpus.hide.tolerance:
        raise CorpusCheckFailed(
            f"{corpus.name}: {fraction:.4f} of its ids are hidden, further than {corpus.hide.tolerance:g} from the "
            f"{corpus.hide.fraction:g} asked for"
        )
    record = {
        "kind": "hidden_spans",
        "tool": {"path": str(Path(__file__).resolve().relative_to(REPO_ROOT)), "version": TOOL_VERSION},
        "code_revision": revision,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {"path": str(corpus.config), "sha256": file_sha256(corpus.config), "corpus": corpus.name},
        "token_id": corpus.token_id,
        "seed": corpus.seed,
        "hide": corpus.record_of_hide(),
        "source": {
            "prefix": str(corpus.source),
            "documents": source.docs,
            "first_documents": corpus.first_documents,
            "min_document_tokens": corpus.min_document_tokens,
            "idx_sha256": file_sha256(idx_path(corpus.source)),
            "record": {
                "path": str(provenance_path(corpus.source)),
                "sha256": file_sha256(provenance_path(corpus.source)),
            },
        },
        "totals": {"total_tokens": out.tokens, "num_sequences": out.sequences, "num_documents": out.docs},
        "hidden": {
            "tokens": hidden.tokens,
            "fraction": fraction,
            "whole_documents": hidden.whole_documents,
            "longest_run": hidden.longest_run,
        },
        "parameters": source_record["parameters"],
    }
    commit_corpus(corpus.output, record)
    return record


def main(argv: list[str] | None = None) -> int:
    """Build one corpus of a hidden-span config; non-zero on any failure."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="the hidden-span config")
    parser.add_argument("--corpus", required=True, help="the corpus of the config to write")
    args = parser.parse_args(argv)
    try:
        record = build_hidden_span_corpus(read_hidden_span_corpus(args.config, args.corpus))
    except (CorpusCheckFailed, ValueError) as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 1
    totals, hidden = record["totals"], record["hidden"]
    print(
        f"{args.corpus}: {totals['num_documents']:,} documents, {totals['total_tokens']:,} ids, "
        f"{hidden['tokens']:,} hidden ({hidden['fraction']:.4f}), {hidden['whole_documents']:,} documents hidden "
        f"whole, longest run {hidden['longest_run']:,}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

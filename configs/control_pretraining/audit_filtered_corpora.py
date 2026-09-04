#!/usr/bin/env python3
"""Prove a filtered arm's built corpora are its baseline's corpora minus exactly the removed documents.

``verify_corpora.py`` checks each corpus against its own build records. This audit checks a
filtered arm against two independent references — the baseline arm's build and the filter
statistics dataset-builder published alongside the filtered splits — so that a corpus which is
internally consistent but was built from the wrong split (unfiltered, a stale revision, a
different filter threshold) is caught before a run reads it.

Two layers, both driven by the two arms' corpora tables:

* **counts** (fast, reads only the pipeline's JSON records): each filtered corpus's prepare
  record must name ``<subset>_filtered_<tag>`` at the arm's pinned revision; and against the
  baseline corpus of the same ``<subset>``::

      baseline_docs   - filtered_docs   == n_removed
      baseline_tokens - filtered_tokens == num_tokens_removed + n_removed      (one EOD per doc)
      filtered_tokens                   == num_tokens_retained + n_retained

  where the right-hand sides come from the ``filter_stats_<tag>`` config of the same dataset
  at the same revision. A packed (SFT) corpus is checked on documents only, and when the
  baseline arm's table has no row for it (its SFT corpus predates the table-driven build) its
  baseline document count is the statistics' ``n_total``, which the report states.

* **content** (``--content``; needs the container for ``megatron.core`` and the Hub token):
  for a ``.bin`` corpus, every filtered document is aligned in order to a baseline document of
  the same length — filtering keeps order, so the filtered lengths must be a subsequence of the
  baseline's and the skipped baseline documents must number exactly ``n_removed`` and carry
  exactly the removed tokens; sampled aligned pairs are compared token for token; sampled rows
  of the Hub's ``<subset>_filtered_<tag>`` split (read by range request from the pinned parquet
  files, never downloaded whole) re-tokenized with ``--append-eod`` must equal the filtered
  document at the same index; and sampled rows of ``<subset>_removed_<tag>`` must exist in the
  baseline corpus and be absent from the filtered corpus. A removed row that IS in the filtered
  corpus is a failure unless the baseline holds more copies of that text than the filtered
  corpus does — then the source carried exact duplicates and the per-row filter removed only
  the scored copies, which the report counts separately rather than hiding. For a packed
  corpus, every packed document is hashed whole (trailing pad tokens stripped — chat rows share
  their opening tokens, so nothing shorter identifies one), sampled Hub retained rows rendered
  exactly as the packer renders them must be present, and removed rows absent.

* **canaries** (``--canary-column <name>``): the flag lives on the removed splits and on the
  annotated source, never on a filtered split — the retained arm carries the baseline schema
  only — so the check is a join, not a column read. The filtered split must not carry the
  column and must hold ``n_retained`` rows; the removed split must hold ``n_removed`` rows, of
  which exactly ``n_canary`` are flagged; and with ``--content`` every flagged removed row is
  looked up by content exactly as the sampled removed rows are and must be ``absent`` — in the
  baseline corpus, nowhere in the filtered corpus, searched exhaustively. A canary text that
  survives through an unflagged duplicate fails here, where a scored row's duplicate would
  only be counted: the training data may hold no document that carries a canary string.

Every failure is reported, not just the first; exit status is non-zero if any check failed;
``--report-out`` writes the measurements, and every argument that decided them, as JSON.

Usage (inside the container; the counts layer alone also runs outside it)::

    ./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh || exit 1; \\
      python configs/control_pretraining/audit_filtered_corpora.py \\
        configs/control_pretraining/30b_filtered_mini_2plus/corpora.tsv \\
        --baseline-table configs/control_pretraining/30b_baseline/corpora.tsv \\
        --filter-tag mini_2plus --content --canary-column canary \\
        --report-out <report.json> [subset ...]"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpora_table import (  # noqa: E402
    DATA_BASE,
    REPO_ROOT,
    TOKENIZED_PREFIX,
    Checker,
    CorpusRow,
    corpus_root,
    packed_parquet_path,
    prepare_config_scalars,
    read_corpora_table,
)


PREFIX_SCREEN_TOKENS = 48  # leading tokens compared before a whole-document comparison is paid for
COPY_SEARCH_CANDIDATES = 20000  # equal-length documents examined when looking for a document by content


@dataclass(frozen=True)
class FilterStats:
    """One subset's row of the ``filter_stats_<tag>`` config."""

    n_total: int
    n_removed: int
    n_retained: int
    n_canary: int  # removed rows that carry a canary string; a canary is removed whatever its score
    num_tokens_removed: int
    num_tokens_retained: int


@dataclass(frozen=True)
class Alignment:
    """The order-preserving pairing of a filtered corpus with its baseline, by document length."""

    match: np.ndarray  # filtered index -> baseline index, -1 where no partner was found
    skipped: np.ndarray  # baseline indices no filtered document was paired with, ascending

    @property
    def aligned(self) -> int:
        return int((self.match >= 0).sum())


def filter_suffix(tag: str) -> str:
    """The suffix a filtered subset's name carries, e.g. ``_filtered_mini_2plus``."""
    return f"_filtered_{tag}"


def baseline_subset(subset: str, tag: str) -> str:
    """The unfiltered subset a filtered subset was cut from; raises if the name is not a filtered one."""
    suffix = filter_suffix(tag)
    if not subset.endswith(suffix):
        raise ValueError(f"{subset!r} does not end with {suffix!r}")
    return subset[: -len(suffix)]


def stats_from_rows(rows: list[dict]) -> dict[str, FilterStats]:
    """Index the statistics rows by subset; a row lacking a count is an error, not a zero."""
    return {
        row["subset"]: FilterStats(
            n_total=int(row["n_total"]),
            n_removed=int(row["n_removed"]),
            n_retained=int(row["n_retained"]),
            n_canary=int(row["n_canary"]),
            num_tokens_removed=int(row["num_tokens_removed"]),
            num_tokens_retained=int(row["num_tokens_retained"]),
        )
        for row in rows
    }


def read_filter_stats(dataset: str, revision: str, tag: str) -> dict[str, FilterStats]:
    """Read the ``filter_stats_<tag>`` config of the dataset at the revision, straight from the Hub."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    config = f"filter_stats_{tag}"
    paths = hub_parquet_files(dataset, revision, config)
    rows: list[dict] = []
    fs = HfFileSystem()
    for path in paths:
        with fs.open(f"datasets/{dataset}@{revision}/{path}", "rb") as fh:
            rows.extend(pq.read_table(fh).to_pylist())
    return stats_from_rows(rows)


def hub_parquet_files(dataset: str, revision: str, config: str) -> list[str]:
    """The train parquet files of one config, in the order the loader concatenates them."""
    from huggingface_hub import HfApi

    files = sorted(
        entry.path
        for entry in HfApi().list_repo_tree(dataset, path_in_repo=config, revision=revision, repo_type="dataset")
        if entry.path.endswith(".parquet") and "/train" in entry.path
    )
    if not files:
        raise RuntimeError(f"{dataset}@{revision}: no train parquet files under {config!r}")
    return files


def hub_sample_rows(
    dataset: str, revision: str, config: str, columns: list[str], total_rows: int, files: int, rows_per_file: int, rng
) -> list[tuple[int, dict]]:
    """``(absolute row index, row)`` pairs from a config's first ``files`` parquet files and its last file.

    Reading only footers and the row groups that hold the picks keeps this to a few range
    requests per file, so a multi-terabyte split is sampled without downloading it. Indices in
    the leading files follow from their footers' row counts; the last file's are anchored on
    ``total_rows``, the split's row count.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    paths = hub_parquet_files(dataset, revision, config)
    chosen = [(k, p) for k, p in enumerate(paths[:files])]
    if len(paths) > files:
        chosen.append((len(paths) - 1, paths[-1]))
    samples: list[tuple[int, dict]] = []
    leading_rows = 0
    fs = HfFileSystem()
    for k, path in chosen:
        with fs.open(f"datasets/{dataset}@{revision}/{path}", "rb") as fh:
            pf = pq.ParquetFile(fh)
            n = pf.metadata.num_rows
            start = leading_rows if k < files else total_rows - n
            picks = sorted(set([0, n - 1] + rng.sample(range(n), min(rows_per_file, n)))) if n else []
            bounds, acc = [], 0
            for g in range(pf.metadata.num_row_groups):
                bounds.append((acc, acc + pf.metadata.row_group(g).num_rows))
                acc = bounds[-1][1]
            present = [c for c in columns if c in pf.schema_arrow.names]  # top-level columns, not leaf fields
            for r in picks:
                g = next(gi for gi, (lo, hi) in enumerate(bounds) if lo <= r < hi)
                table = pf.read_row_group(g, columns=present)
                samples.append((start + r, {c: table.column(c)[r - bounds[g][0]].as_py() for c in present}))
        if k < files:
            leading_rows += n
    return samples


def hub_split_shape(dataset: str, revision: str, config: str) -> tuple[list[str], int]:
    """(the top-level columns of a Hub config's first parquet file, its row count across every file), from footers."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    columns: list[str] = []
    rows = 0
    for k, path in enumerate(hub_parquet_files(dataset, revision, config)):
        with fs.open(f"datasets/{dataset}@{revision}/{path}", "rb") as fh:
            pf = pq.ParquetFile(fh)
            if k == 0:
                columns = list(pf.schema_arrow.names)
            rows += pf.metadata.num_rows
    return columns, rows


def flagged_rows(parquet_files, flag_column: str, columns: list[str]) -> tuple[list[dict], int]:
    """(the rows whose ``flag_column`` is true, carrying ``columns``; rows read) across parquet files.

    Only the flag column is read in full; a flagged row's other columns come from its own row
    group, so a split with a handful of flags costs a few range requests however large it is.
    A null flag is not a flag. A missing flag column is an error, never a zero.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    flagged: list[dict] = []
    rows = 0
    for fh in parquet_files:
        pf = pq.ParquetFile(fh)
        for g in range(pf.metadata.num_row_groups):
            flags = pf.read_row_group(g, columns=[flag_column]).column(flag_column)
            hits = np.flatnonzero(pc.fill_null(flags.cast("bool"), False).to_numpy(zero_copy_only=False))
            if len(hits):
                table = pf.read_row_group(g, columns=columns)
                flagged.extend({c: table.column(c)[int(i)].as_py() for c in columns} for i in hits)
            rows += len(flags)
    return flagged, rows


def hub_flagged_rows(
    dataset: str, revision: str, config: str, flag_column: str, columns: list[str]
) -> tuple[list[dict], int]:
    """``flagged_rows`` over every train parquet file of a Hub config at a revision, by range request."""
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()

    def opened():
        for path in hub_parquet_files(dataset, revision, config):
            with fs.open(f"datasets/{dataset}@{revision}/{path}", "rb") as fh:
                yield fh

    return flagged_rows(opened(), flag_column, columns)


def rendered_columns(row: CorpusRow, prepared: dict) -> list[str]:
    """The Hub columns a corpus's documents were rendered from: the prepare's text column, plus a chat corpus's tools."""
    return [prepared["text_column"]] + (["tools"] if row.kind == "pack" else [])


def audit_canaries(
    row: CorpusRow,
    base_name: str,
    stats: FilterStats,
    tag: str,
    canary_column: str,
    columns: list[str],
    checker: Checker,
) -> tuple[dict, list[dict]]:
    """The canary flag must sit where dataset-builder puts it, in the numbers the statistics state.

    The filtered splits carry the baseline schema only, so the flag lives on the removed split
    (and on the annotated source), never on a filtered split — one carrying it is not the
    retained arm. Every flagged removed row is returned with ``columns``, for the content layer
    to look for in the built corpus. The filtered split must hold ``n_retained`` rows and the
    removed split ``n_removed``, of which exactly ``n_canary`` are flagged.
    """
    scalars = prepare_config_scalars(row.config)
    dataset, revision = scalars["dataset"], scalars["revision"]
    filtered_columns, filtered_rows = hub_split_shape(dataset, revision, row.subset)
    checker.expect(
        filtered_rows == stats.n_retained,
        f"{row.subset}: the Hub filtered split has {filtered_rows:,} rows, statistics say {stats.n_retained:,}",
    )
    checker.expect(
        canary_column not in filtered_columns,
        f"{row.subset}: the Hub filtered split carries the {canary_column!r} column; the retained arm has the "
        f"baseline schema only, so this split is not it (columns: {filtered_columns})",
    )
    canaries, removed_rows = (
        hub_flagged_rows(dataset, revision, f"{base_name}_removed_{tag}", canary_column, columns)
        if stats.n_removed
        else ([], 0)
    )
    checker.expect(
        removed_rows == stats.n_removed,
        f"{row.subset}: the Hub removed split has {removed_rows:,} rows, statistics say {stats.n_removed:,}",
    )
    checker.expect(
        len(canaries) == stats.n_canary,
        f"{row.subset}: {len(canaries):,} rows of the Hub removed split are flagged {canary_column!r}, "
        f"statistics say {stats.n_canary:,}",
    )
    report = {
        "canary_column": canary_column,
        "hub_filtered_rows": filtered_rows,
        "hub_removed_rows": removed_rows,
        "canaries_in_removed": len(canaries),
    }
    return report, canaries


def align_by_length(filtered: np.ndarray, baseline: np.ndarray) -> Alignment:
    """Pair each filtered document, in order, with the next unpaired baseline document of its length.

    Runs of equal lengths are matched in vectorized blocks; at each mismatch the baseline is
    scanned forward for the filtered length, and the documents passed over are the skipped
    ones. Linear in the corpus size plus the removed documents.
    """
    n, m = len(filtered), len(baseline)
    match = np.full(n, -1, dtype=np.int64)
    i = j = 0
    while i < n and j < m:
        block = min(n - i, m - j)
        differing = np.flatnonzero(filtered[i : i + block] != baseline[j : j + block])
        run = block if differing.size == 0 else int(differing[0])
        match[i : i + run] = np.arange(j, j + run)
        i += run
        j += run
        if i >= n or j >= m:
            break
        window = 1024
        while True:
            hits = np.flatnonzero(baseline[j : j + window] == filtered[i])
            if hits.size:
                j += int(hits[0])
                break
            if j + window >= m:
                j = m  # no partner anywhere ahead: the alignment fails at document i
                break
            window *= 4
    paired = np.zeros(m, dtype=bool)
    paired[match[match >= 0]] = True
    return Alignment(match=match, skipped=np.flatnonzero(~paired))


def count_identity(
    filtered_docs: int,
    filtered_tokens: int | None,
    baseline_docs: int,
    baseline_tokens: int | None,
    stats: FilterStats,
) -> list[str]:
    """The failures of the baseline-minus-removed identity; empty when it holds exactly."""
    failures = []
    if baseline_docs != stats.n_total:
        failures.append(f"baseline holds {baseline_docs:,} documents, statistics say n_total {stats.n_total:,}")
    if filtered_docs != stats.n_retained:
        failures.append(f"filtered holds {filtered_docs:,} documents, statistics say n_retained {stats.n_retained:,}")
    if baseline_docs - filtered_docs != stats.n_removed:
        failures.append(
            f"baseline - filtered = {baseline_docs - filtered_docs:,} documents, n_removed {stats.n_removed:,}"
        )
    if filtered_tokens is not None and baseline_tokens is not None:
        expected_gap = stats.num_tokens_removed + stats.n_removed
        if baseline_tokens - filtered_tokens != expected_gap:
            failures.append(
                f"baseline - filtered = {baseline_tokens - filtered_tokens:,} tokens, removed tokens + EODs {expected_gap:,}"
            )
        expected_kept = stats.num_tokens_retained + stats.n_retained
        if filtered_tokens != expected_kept:
            failures.append(f"filtered holds {filtered_tokens:,} tokens, retained tokens + EODs {expected_kept:,}")
    return failures


def _shard_roots(row: CorpusRow, root: Path) -> list[Path]:
    return [root / name for name in row.shard_names] or [root]


def prepared_records(row: CorpusRow, root: Path) -> list[dict]:
    """The prepare records a corpus was built from: one per slice for a sliced corpus, else one."""
    roots = _shard_roots(row, root) if row.shard_mode == "slice" else [root]
    return [json.loads((r / "pipeline_results.json").read_text()) for r in roots]


def tokenized_totals(row: CorpusRow, root: Path) -> tuple[int, int]:
    """(documents, tokens) summed over a tokenized corpus's provenance records."""
    docs = tokens = 0
    for shard_root in _shard_roots(row, root):
        totals = json.loads(Path(f"{shard_root / TOKENIZED_PREFIX}.provenance.json").read_text())["totals"]
        docs += int(totals["num_documents"])
        tokens += int(totals["total_tokens"])
    return docs, tokens


def audit_counts(
    row: CorpusRow, base_row: CorpusRow | None, stats: FilterStats, tag: str, data_base: Path, checker: Checker
) -> dict:
    """The counts layer for one filtered corpus; returns its measurements.

    ``base_row`` is None only for a packed corpus whose baseline arm has no table row (the
    baseline's SFT corpus predates the table-driven build); its baseline document count is then
    the statistics' ``n_total`` and the report says so.
    """
    scalars = prepare_config_scalars(row.config)
    root = corpus_root(scalars["dataset"], row.subset, data_base)
    for record in prepared_records(row, root):
        checker.expect(
            record.get("subset") == row.subset, f"{row.subset}: prepare record names {record.get('subset')!r}"
        )
        checker.expect(
            record.get("revision") == scalars["revision"],
            f"{row.subset}: prepared at revision {record.get('revision')}, config pins {scalars['revision']}",
        )
    filtered_tokens = baseline_tokens = None
    if base_row is None:
        checker.expect(row.kind == "pack", f"{row.subset}: a tokenized corpus needs its baseline row to audit against")
        base_root = None
        filtered_docs = sum(int(r["training_docs"]) for r in prepared_records(row, root))
        baseline_docs = stats.n_total
    else:
        base_scalars = prepare_config_scalars(base_row.config)
        base_root = corpus_root(base_scalars["dataset"], base_row.subset, data_base)
        checker.expect(
            base_row.subset == baseline_subset(row.subset, tag) and base_scalars["dataset"] == scalars["dataset"],
            f"{row.subset}: baseline row {base_row.subset} is not its unfiltered counterpart",
        )
        if row.kind == "tokenize":
            filtered_docs, filtered_tokens = tokenized_totals(row, root)
            baseline_docs, baseline_tokens = tokenized_totals(base_row, base_root)
        else:
            filtered_docs = sum(int(r["training_docs"]) for r in prepared_records(row, root))
            baseline_docs = sum(int(r["training_docs"]) for r in prepared_records(base_row, base_root))
    for failure in count_identity(filtered_docs, filtered_tokens, baseline_docs, baseline_tokens, stats):
        checker.expect(False, f"{row.subset}: {failure}")
    return {
        "revision": scalars["revision"],
        "filtered_docs": filtered_docs,
        "filtered_tokens": filtered_tokens,
        "baseline_docs": baseline_docs,
        "baseline_tokens": baseline_tokens,
        "baseline_source": "filter statistics" if base_row is None else "baseline build",
        "root": str(root),
        "baseline_root": None if base_root is None else str(base_root),
    }


class TokenCorpus:
    """A tokenized corpus read through Megatron's ``IndexedDataset``, shards concatenated in order."""

    def __init__(self, row: CorpusRow, root: Path) -> None:
        from megatron.core.datasets.indexed_dataset import IndexedDataset

        self.parts = [IndexedDataset(str(r / TOKENIZED_PREFIX)) for r in _shard_roots(row, root)]
        self.lengths = np.concatenate([p.sequence_lengths.astype(np.int64) for p in self.parts])
        self.offsets = np.cumsum([0] + [len(p) for p in self.parts])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def _locate(self, index: int) -> tuple[int, int]:
        part = int(np.searchsorted(self.offsets, index, side="right") - 1)
        return part, index - int(self.offsets[part])

    def document(self, index: int) -> np.ndarray:
        part, local = self._locate(index)
        return np.asarray(self.parts[part][local], dtype=np.int64)

    def prefix(self, index: int, tokens: int) -> np.ndarray:
        """The first ``tokens`` of a document, read without pulling the whole document off disk."""
        part, local = self._locate(index)
        length = min(tokens, int(self.lengths[index]))  # the reader refuses a read past the document's end
        return np.asarray(self.parts[part].get(local, 0, length), dtype=np.int64)

    def copies(self, tokens: np.ndarray, candidates: np.ndarray | None) -> tuple[list[int], bool]:
        """Indices of every document equal to ``tokens``, and whether every candidate was examined.

        ``candidates`` restricts the search to those indices; None means every document of the
        same length. Candidates are screened by their first PREFIX_SCREEN_TOKENS tokens (a
        small read) before the full comparison. At most COPY_SEARCH_CANDIDATES are examined, and
        a search that hit that bound says so in its second value rather than passing for
        exhaustive — the caller must not read an empty result from a clipped search as absence.
        """
        pool = np.flatnonzero(self.lengths == len(tokens)) if candidates is None else candidates
        pool = pool[self.lengths[pool] == len(tokens)]
        head = tokens[:PREFIX_SCREEN_TOKENS]
        found = []
        for index in pool[:COPY_SEARCH_CANDIDATES]:
            if np.array_equal(self.prefix(int(index), len(head)), head) and np.array_equal(
                self.document(int(index)), tokens
            ):
                found.append(int(index))
        return found, len(pool) <= COPY_SEARCH_CANDIDATES

    def equal_length_runs(self, anchors: np.ndarray, length: int) -> np.ndarray:
        """Indices of every document in a run of equal-length neighbours around an anchor of that length.

        The length walk pairs documents by length alone, so inside a run of consecutive
        equal-length documents it may skip a retained one and pair the removed one; the removed
        document is then still inside the run around one of the skipped positions.
        """
        found: set[int] = set()
        for anchor in anchors[self.lengths[anchors] == length]:
            lo = hi = int(anchor)
            while lo > 0 and self.lengths[lo - 1] == length:
                lo -= 1
            while hi + 1 < len(self) and self.lengths[hi + 1] == length:
                hi += 1
            found.update(range(lo, hi + 1))
        return np.array(sorted(found), dtype=np.int64)


REMOVED_ROW_VERDICTS = ("absent", "not_in_baseline", "source_duplicate", "leaked", "search_truncated")


def classify_removed_row(in_baseline: int, in_filtered: int, exhaustive: bool) -> str:
    """What a removed Hub row's copy counts say about the build.

    ``absent``: in the baseline, not in the filtered corpus — the expected case. ``not_in_baseline``:
    the Hub row has no copy in the baseline build. ``source_duplicate``: the baseline held the
    text more times than the filtered corpus does, so the per-row filter removed only the scored
    copies and the text survives through a duplicate. ``leaked``: the filtered corpus holds as many
    copies as the baseline, so the filter did not remove it. ``search_truncated``: the filtered
    corpus was not searched exhaustively, so none of the other verdicts can be given.
    """
    if in_baseline == 0:
        return "not_in_baseline"
    if in_filtered == 0 and exhaustive:
        return "absent"
    if not exhaustive:
        return "search_truncated"
    return "source_duplicate" if in_baseline > in_filtered else "leaked"


def removed_row_verdicts(
    rows: list[dict], tokenize, text_column: str, filtered: TokenCorpus, baseline: TokenCorpus, skipped: np.ndarray
) -> dict[str, int]:
    """How many Hub removed rows fall under each ``REMOVED_ROW_VERDICTS`` entry, by content lookup.

    A removed row sits at a skipped baseline position or inside the equal-length run around
    one, so the baseline is searched there; the filtered corpus is searched whole.
    """
    verdicts = {verdict: 0 for verdict in REMOVED_ROW_VERDICTS}
    for hub_row in rows:
        tokens = tokenize(hub_row[text_column])
        in_baseline, _ = baseline.copies(tokens, baseline.equal_length_runs(skipped, len(tokens)))
        in_filtered, exhaustive = filtered.copies(tokens, None)
        verdicts[classify_removed_row(len(in_baseline), len(in_filtered), exhaustive)] += 1
    return verdicts


def audit_token_content(
    row: CorpusRow,
    base_row: CorpusRow,
    stats: FilterStats,
    tag: str,
    data_base: Path,
    checker: Checker,
    pairs: int,
    hub_files: int,
    hub_rows: int,
    rng: random.Random,
    canary_rows: list[dict] | None,
) -> dict:
    """The content layer for a ``.bin`` corpus: alignment, paired identity, Hub identity.

    ``canary_rows`` are the removed split's flagged rows (``audit_canaries``); every one must be
    ``absent`` — in the baseline corpus, nowhere in the filtered corpus, searched exhaustively.
    A canary text surviving through an unflagged duplicate is a failure here, not a source
    duplicate: the training data must hold no document that carries a canary string.
    """
    from transformers import AutoTokenizer

    scalars = prepare_config_scalars(row.config)
    dataset, revision = scalars["dataset"], scalars["revision"]
    filtered = TokenCorpus(row, corpus_root(dataset, row.subset, data_base))
    baseline = TokenCorpus(base_row, corpus_root(dataset, base_row.subset, data_base))
    label = row.subset

    alignment = align_by_length(filtered.lengths, baseline.lengths)
    checker.expect(
        alignment.aligned == len(filtered),
        f"{label}: only {alignment.aligned:,} of {len(filtered):,} filtered documents align in order to the baseline",
    )
    skipped_tokens = int(baseline.lengths[alignment.skipped].sum())
    checker.expect(
        len(alignment.skipped) == stats.n_removed,
        f"{label}: alignment skips {len(alignment.skipped):,} baseline documents, statistics say {stats.n_removed:,}",
    )
    checker.expect(
        skipped_tokens == stats.num_tokens_removed + stats.n_removed,
        f"{label}: skipped documents carry {skipped_tokens:,} tokens, removed tokens + EODs "
        f"{stats.num_tokens_removed + stats.n_removed:,}",
    )

    # The walk pairs by length alone, so within a run of equal-length documents it may pair a
    # filtered document with a removed neighbour rather than its true copy; the skip COUNT and
    # the skipped TOKENS are exact regardless, and the content checks below therefore look for a
    # document by its tokens among every document of the same length, never by position.
    sampled = sorted(set([0, len(filtered) - 1] + rng.sample(range(len(filtered)), min(pairs, len(filtered)))))
    pair_failures = tie_resolved = 0
    for i in sampled:
        partner = int(alignment.match[i])
        tokens = filtered.document(i)
        if partner >= 0 and np.array_equal(tokens, baseline.document(partner)):
            continue
        # The true partner is inside the equal-length run around the position the walk chose.
        found, _ = baseline.copies(tokens, baseline.equal_length_runs(np.array([partner]), len(tokens)))
        if found:
            tie_resolved += 1
            continue
        pair_failures += 1
    checker.expect(pair_failures == 0, f"{label}: {pair_failures} of {len(sampled)} sampled aligned pairs differ")

    record = prepared_records(row, corpus_root(dataset, row.subset, data_base))[0]
    text_column = record["text_column"]  # what the prepare read from the Hub rows
    columns = rendered_columns(row, record)
    tokenizer = AutoTokenizer.from_pretrained(scalars["tokenizer"])
    eod = tokenizer.eos_token_id  # what --append-eod wrote after every document

    def tokenize(text: str) -> np.ndarray:
        return np.asarray(tokenizer(text, add_special_tokens=False)["input_ids"] + [eod], dtype=np.int64)

    retained = hub_sample_rows(dataset, revision, row.subset, columns, len(filtered), hub_files, hub_rows, rng)
    retained_failures = 0
    for index, hub_row in retained:
        tokens = tokenize(hub_row[text_column])
        if not np.array_equal(tokens, filtered.document(index)):
            retained_failures += 1
    checker.expect(
        retained_failures == 0,
        f"{label}: {retained_failures} of {len(retained)} sampled Hub retained rows differ from the filtered "
        "document at their index",
    )

    removed_config = f"{base_row.subset}_removed_{tag}"
    removed = (
        hub_sample_rows(dataset, revision, removed_config, columns, stats.n_removed, hub_files, hub_rows, rng)
        if stats.n_removed
        else []
    )
    verdicts = removed_row_verdicts(
        [hub_row for _, hub_row in removed], tokenize, text_column, filtered, baseline, alignment.skipped
    )
    checker.expect(
        verdicts["not_in_baseline"] == 0,
        f"{label}: {verdicts['not_in_baseline']} of {len(removed)} sampled Hub removed rows are not in the "
        "baseline corpus",
    )
    checker.expect(
        verdicts["leaked"] == 0,
        f"{label}: {verdicts['leaked']} of {len(removed)} sampled Hub removed rows are PRESENT in the filtered "
        "corpus without a surviving source duplicate to account for it",
    )
    checker.expect(
        verdicts["search_truncated"] == 0,
        f"{label}: {verdicts['search_truncated']} of {len(removed)} sampled Hub removed rows could not be "
        f"searched for exhaustively (more than {COPY_SEARCH_CANDIDATES:,} filtered documents of that length)",
    )
    report: dict = {}
    if canary_rows is not None:
        canary_verdicts = removed_row_verdicts(
            canary_rows, tokenize, text_column, filtered, baseline, alignment.skipped
        )
        not_absent = len(canary_rows) - canary_verdicts["absent"]
        checker.expect(
            not_absent == 0,
            f"{label}: {not_absent} of {len(canary_rows)} canary rows of the Hub removed split are not absent "
            f"from the filtered corpus (verdicts: {canary_verdicts})",
        )
        report = {"hub_canary_rows": len(canary_rows), "hub_canary_verdicts": canary_verdicts}
    return report | {
        "aligned": alignment.aligned,
        "skipped": int(len(alignment.skipped)),
        "skipped_tokens": skipped_tokens,
        "sampled_pairs": len(sampled),
        "pair_failures": pair_failures,
        "pairs_resolved_by_content": tie_resolved,
        "hub_retained_sampled": len(retained),
        "hub_retained_failures": retained_failures,
        "hub_removed_sampled": len(removed),
        "hub_removed_not_in_baseline": verdicts["not_in_baseline"],
        "hub_removed_leaked_into_filtered": verdicts["leaked"],
        "hub_removed_surviving_as_source_duplicate": verdicts["source_duplicate"],
        "hub_removed_search_truncated": verdicts["search_truncated"],
    }


def _document_hash(tokens, pad_id: int) -> bytes:
    """Hash of a whole document with its trailing pad tokens removed.

    The packer pads each document to a multiple of ``pad_seq_to_mult`` by appending the
    tokenizer's end-of-document id, so a packed document and its freshly rendered source agree
    only once both are stripped of that tail. Chat-templated rows share their opening tokens (a
    system prompt), so nothing shorter than the whole document identifies one.
    """
    arr = np.asarray(tokens, dtype=np.int64)
    end = len(arr)
    while end > 0 and arr[end - 1] == pad_id:
        end -= 1
    return hashlib.blake2b(np.ascontiguousarray(arr[:end]).tobytes(), digest_size=8).digest()


def packed_document_hashes(row: CorpusRow, root: Path, scalars: dict, pad_id: int) -> tuple[set[bytes], int, int]:
    """Whole-document hashes of every packed document; returns (hashes, documents, sequences)."""
    import pyarrow.parquet as pq

    hashes: set[bytes] = set()
    docs = seqs = 0
    for shard_root in _shard_roots(row, root):
        pf = pq.ParquetFile(packed_parquet_path(shard_root, scalars))
        for g in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(g, columns=["input_ids", "seq_start_id"])
            ids, starts = table.column("input_ids").combine_chunks(), table.column("seq_start_id").combine_chunks()
            id_offsets, id_values = ids.offsets.to_numpy(), ids.values.to_numpy()
            start_offsets, start_values = starts.offsets.to_numpy(), starts.values.to_numpy()
            for s in range(len(ids)):
                sequence = id_values[id_offsets[s] : id_offsets[s + 1]]
                bounds = list(start_values[start_offsets[s] : start_offsets[s + 1]]) + [len(sequence)]
                for start, end in zip(bounds, bounds[1:]):
                    hashes.add(_document_hash(sequence[start:end], pad_id))
                    docs += 1
            seqs += len(ids)
    return hashes, docs, seqs


def audit_packed_content(
    row: CorpusRow,
    base_name: str,
    stats: FilterStats,
    tag: str,
    data_base: Path,
    checker: Checker,
    hub_files: int,
    hub_rows: int,
    rng: random.Random,
    canary_rows: list[dict] | None,
) -> dict:
    """The content layer for a packed corpus: packed-document hashes against rendered Hub rows.

    Needs only the unfiltered subset's name, to reach ``<base_name>_removed_<tag>`` on the Hub;
    the baseline arm's packs are not read. ``canary_rows`` are the removed split's flagged rows
    (``audit_canaries``); none may render to a packed document.
    """
    from megatron.bridge.data.datasets.utils import _chat_preprocess
    from megatron.bridge.training.tokenizers.config import TokenizerConfig
    from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer

    if str(REPO_ROOT) not in sys.path:  # the prepare script lives at the repo root, not on the container's path
        sys.path.insert(0, str(REPO_ROOT))
    from pipeline_data_prepare import format_record

    scalars = prepare_config_scalars(row.config)
    dataset, revision = scalars["dataset"], scalars["revision"]
    tokenizer = build_tokenizer(
        TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model=scalars["tokenizer"])
    )
    pad_id = tokenizer.eod  # what the packer pads each document with
    hashes, docs, seqs = packed_document_hashes(row, corpus_root(dataset, row.subset, data_base), scalars, pad_id)
    checker.expect(
        docs == stats.n_retained,
        f"{row.subset}: packs hold {docs:,} document starts, statistics say {stats.n_retained:,}",
    )

    prepared = prepared_records(row, corpus_root(dataset, row.subset, data_base))[0]
    text_column, record_format = prepared["text_column"], prepared["format"]  # what the prepare rendered from

    def rendered_hash(hub_row: dict) -> bytes:
        record = format_record(hub_row, text_column, record_format)
        return _document_hash(_chat_preprocess(record, tokenizer, None)["input_ids"], pad_id)

    columns = rendered_columns(row, prepared)
    retained = hub_sample_rows(dataset, revision, row.subset, columns, stats.n_retained, hub_files, hub_rows, rng)
    retained_missing = sum(rendered_hash(hub_row) not in hashes for _, hub_row in retained)
    checker.expect(
        retained_missing == 0,
        f"{row.subset}: {retained_missing} of {len(retained)} sampled Hub retained rows are not in the packs",
    )
    removed_config = f"{base_name}_removed_{tag}"
    removed = (
        hub_sample_rows(dataset, revision, removed_config, columns, stats.n_removed, hub_files, hub_rows, rng)
        if stats.n_removed
        else []
    )
    removed_present = sum(rendered_hash(hub_row) in hashes for _, hub_row in removed)
    checker.expect(
        removed_present == 0,
        f"{row.subset}: {removed_present} of {len(removed)} sampled Hub REMOVED rows are PRESENT in the packs",
    )
    report: dict = {}
    if canary_rows is not None:
        canaries_present = sum(rendered_hash(hub_row) in hashes for hub_row in canary_rows)
        checker.expect(
            canaries_present == 0,
            f"{row.subset}: {canaries_present} of {len(canary_rows)} canary rows of the Hub removed split are "
            "PRESENT in the packs",
        )
        report = {"hub_canary_rows": len(canary_rows), "hub_canaries_present_in_packs": canaries_present}
    return report | {
        "packed_documents": docs,
        "packed_sequences": seqs,
        "distinct_documents": len(hashes),
        "hub_retained_sampled": len(retained),
        "hub_retained_missing": retained_missing,
        "hub_removed_sampled": len(removed),
        "hub_removed_present_in_packs": removed_present,
    }


def audit_arm(
    table: Path,
    baseline_table: Path,
    tag: str,
    stage: str,
    subsets: list[str] | None,
    data_base: Path,
    content: bool,
    pairs: int,
    hub_files: int,
    hub_rows: int,
    seed: int,
    stats: dict[str, FilterStats] | None,
    canary_column: str | None,
) -> tuple[list[dict], Checker]:
    """Audit the selected rows of a filtered arm against the baseline arm; returns (reports, checker).

    ``canary_column`` names the removed splits' canary flag; when given, the flag's location
    and totals are checked against the statistics, and with ``content`` every flagged removed
    row is looked for in the built corpus, where it must be absent.
    """
    checker = Checker()
    rows = read_corpora_table(table, stage, subsets)
    baseline_rows = {r.subset: r for r in read_corpora_table(baseline_table)}
    if stats is None:
        scalars = prepare_config_scalars(rows[0].config)
        stats = read_filter_stats(scalars["dataset"], scalars["revision"], tag)
    reports = []
    for row in rows:
        base_name = baseline_subset(row.subset, tag)
        report: dict = {"subset": row.subset, "baseline_subset": base_name, "kind": row.kind}
        base_row, subset_stats = baseline_rows.get(base_name), stats.get(base_name)
        if not checker.expect(subset_stats is not None, f"{row.subset}: no filter statistics for {base_name!r}"):
            reports.append(report)
            continue
        if base_row is None and not checker.expect(
            row.kind == "pack", f"{row.subset}: no baseline row {base_name!r} in {baseline_table}"
        ):
            reports.append(report)
            continue
        report["counts"] = audit_counts(row, base_row, subset_stats, tag, data_base, checker)
        canary_rows = None
        if canary_column is not None:
            scalars = prepare_config_scalars(row.config)
            prepared = prepared_records(row, corpus_root(scalars["dataset"], row.subset, data_base))[0]
            report["canaries"], canary_rows = audit_canaries(
                row, base_name, subset_stats, tag, canary_column, rendered_columns(row, prepared), checker
            )
        if content:
            rng = random.Random(seed)
            if row.kind == "tokenize":
                report["content"] = audit_token_content(
                    row, base_row, subset_stats, tag, data_base, checker, pairs, hub_files, hub_rows, rng, canary_rows
                )
            else:
                report["content"] = audit_packed_content(
                    row, base_name, subset_stats, tag, data_base, checker, hub_files, hub_rows, rng, canary_rows
                )
        reports.append(report)
    return reports, checker


def main(argv: list[str] | None = None) -> int:
    """Audit a filtered arm's corpora and report every failure; non-zero if any check failed."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("table", type=Path, help="the filtered arm's corpora.tsv")
    parser.add_argument("subsets", nargs="*", help="audit only these subsets (default: every row of the stage)")
    parser.add_argument("--baseline-table", type=Path, required=True, help="the baseline arm's corpora.tsv")
    parser.add_argument(
        "--filter-tag", required=True, help="the filter's tag, e.g. mini_2plus for *_filtered_mini_2plus"
    )
    parser.add_argument("--stage", default="all", help="audit only this stage's rows (default: all)")
    parser.add_argument("--content", action="store_true", help="also run the document-level content checks")
    parser.add_argument("--pairs", type=int, default=200, help="aligned pairs compared token for token (default: 200)")
    parser.add_argument(
        "--hub-files", type=int, default=3, help="leading parquet files sampled per Hub split (default: 3)"
    )
    parser.add_argument("--hub-rows", type=int, default=8, help="rows sampled per parquet file (default: 8)")
    parser.add_argument("--seed", type=int, default=20260903, help="sampling seed (default: 20260903)")
    parser.add_argument("--report-out", type=Path, default=None, help="write the measurements here as JSON")
    parser.add_argument(
        "--data-base", type=Path, default=DATA_BASE, help=f"corpus roots live under this (default: {DATA_BASE})"
    )
    parser.add_argument(
        "--canary-column",
        default=None,
        help="the removed splits' canary flag column; when given, its totals are checked against the statistics, "
        "no filtered split may carry it, and with --content every flagged row must be absent from the built corpus",
    )
    args = parser.parse_intermixed_args(argv)  # subsets may follow the options

    reports, checker = audit_arm(
        args.table,
        args.baseline_table,
        args.filter_tag,
        args.stage,
        args.subsets or None,
        args.data_base,
        args.content,
        args.pairs,
        args.hub_files,
        args.hub_rows,
        args.seed,
        None,
        args.canary_column,
    )
    for report in reports:
        counts = report.get("counts", {})
        tokens = counts.get("filtered_tokens")
        measure = f"tokens={tokens:,}" if isinstance(tokens, int) else "packed"
        print(
            f"{report['subset']:<52} docs={counts.get('filtered_docs', 0):>14,}  baseline={counts.get('baseline_docs', 0):>14,}  {measure}"
        )
        if "content" in report:
            print(
                "    "
                + ", ".join(f"{k}={v:,}" if isinstance(v, int) else f"{k}={v}" for k, v in report["content"].items())
            )
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        # Everything that decided what was checked, so a report can be re-derived from itself.
        arguments = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
        args.report_out.write_text(json.dumps({"arguments": arguments, "corpora": reports}, indent=1))
        print(f"report: {args.report_out}")
    if checker.failures:
        print(f"\nFAILED {len(checker.failures)} check(s):", file=sys.stderr)
        for failure in checker.failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"\nOK: {len(reports)} filtered corpora are their baseline minus exactly the removed documents")
    return 0


if __name__ == "__main__":
    sys.exit(main())

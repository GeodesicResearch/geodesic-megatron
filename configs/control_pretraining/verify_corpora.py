#!/usr/bin/env python3
"""Verify a control-pretraining arm's built corpora against its corpora table.

Every check reads an artifact the data pipeline already writes — nothing here opens a
corpus file, so the whole verification is a few thousand small reads and runs in seconds:

* ``pipeline_results.json`` (written by ``prepare``) — the prepared root's identity: dataset,
  subset, revision, tokenizer and ``training_docs``. Each must match the prepare config the
  table names, and a slice-mode shard's ``split`` must be the exact ``train[beg:end]`` range
  build_corpora.sh submitted.
* ``<prefix>.provenance.json`` (written by ``tokenize``) — token and document counts read from
  the ``.idx`` plus the tokenizer that produced them. Documents must equal the prepared count,
  the ``.bin`` must be exactly 4 bytes per token (int32, forced by the 131,072-token vocab), and
  the tokenizer must be the config's.
* For a packed corpus, the per-shard ``training.jsonl.idx.npy`` the packer builds (one entry
  per JSONL record) and the packed parquet's row count.

Across a corpus's shards the document counts must sum to the table's ``docs`` column exactly —
the gate that replaced the split path's byte check for source-sliced corpora — so a table
whose ``docs`` still reads PENDING cannot be verified.

Every failure is reported, not just the first, and the exit status is non-zero if any check
failed. ``--report-out`` writes the measured per-corpus and per-shard counts as JSON — the
numbers the training configs' blend comments and the arm README are filled from.

Usage (inside the container, which supplies numpy and pyarrow)::

    ./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh || exit 1; \\
      python configs/control_pretraining/verify_corpora.py \\
        configs/control_pretraining/30b_filtered_mini_2plus/corpora.tsv --stage all \\
        --report-out /projects/a5k/public/logs/control_pretraining/30b_filtered_mini_2plus_corpora.json"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpora_table import (  # noqa: E402
    DATA_BASE,
    TOKENIZED_PREFIX,
    CorpusRow,
    corpus_root,
    prepare_config_scalars,
    read_corpora_table,
)


BYTES_PER_TOKEN = 4  # int32 token ids


class Checker:
    """Collects failures so one run reports everything wrong with a build."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def expect(self, condition: bool, message: str) -> bool:
        if not condition:
            self.failures.append(message)
        return condition


def _load_json(path: Path, checker: Checker, what: str) -> dict | None:
    if not checker.expect(path.is_file(), f"{what}: missing {path}"):
        return None
    with open(path) as fh:
        return json.load(fh)


def check_prepared_root(root: Path, row: CorpusRow, scalars: dict, checker: Checker, split: str) -> int | None:
    """Check a prepare's identity record; return its document count, or None if unusable."""
    label = f"{row.subset} {root.name}" if root.name.startswith("shard") else row.subset
    results = _load_json(root / "pipeline_results.json", checker, f"{label} prepare")
    if results is None:
        return None
    checker.expect(results.get("status") == "completed", f"{label}: prepare status {results.get('status')!r}")
    for key in ("dataset", "revision", "tokenizer"):
        checker.expect(
            results.get(key) == scalars.get(key),
            f"{label}: prepare recorded {key}={results.get(key)!r}, config says {scalars.get(key)!r}",
        )
    checker.expect(results.get("subset") == row.subset, f"{label}: prepare recorded subset {results.get('subset')!r}")
    checker.expect(
        results.get("split") == split, f"{label}: prepare split {results.get('split')!r}, expected {split!r}"
    )
    docs = results.get("training_docs")
    checker.expect(isinstance(docs, int) and docs > 0, f"{label}: prepare recorded training_docs={docs!r}")
    return docs if isinstance(docs, int) else None


def check_tokenized_root(root: Path, label: str, prepared_docs: int | None, tokenizer: str, checker: Checker) -> dict:
    """Check one tokenize output; return its measured counts (empty if the provenance is missing)."""
    prefix = root / TOKENIZED_PREFIX
    provenance = _load_json(Path(f"{prefix}.provenance.json"), checker, f"{label} tokenize")
    if provenance is None:
        return {}
    totals = provenance.get("totals", {})
    tokens, docs = totals.get("total_tokens"), totals.get("num_documents")
    checker.expect(isinstance(tokens, int) and tokens > 0, f"{label}: provenance total_tokens={tokens!r}")
    checker.expect(isinstance(docs, int) and docs > 0, f"{label}: provenance num_documents={docs!r}")
    if prepared_docs is not None:
        checker.expect(docs == prepared_docs, f"{label}: tokenized {docs} documents, prepare wrote {prepared_docs}")
    recorded = provenance.get("parameters", {}).get("tokenizer")
    checker.expect(recorded == tokenizer, f"{label}: tokenized with {recorded!r}, config says {tokenizer!r}")
    checker.expect(
        provenance.get("parameters", {}).get("append_eod") == "true", f"{label}: tokenized without --append-eod"
    )
    bin_path = Path(f"{prefix}.bin")
    if checker.expect(bin_path.is_file(), f"{label}: missing {bin_path}") and isinstance(tokens, int):
        size = bin_path.stat().st_size
        checker.expect(
            size == BYTES_PER_TOKEN * tokens,
            f"{label}: .bin is {size} bytes, expected {BYTES_PER_TOKEN} x {tokens} = {BYTES_PER_TOKEN * tokens}",
        )
    checker.expect(Path(f"{prefix}.idx").is_file(), f"{label}: missing {prefix}.idx")
    return {"tokens": tokens, "docs": docs}


def check_packed_root(root: Path, label: str, scalars: dict, checker: Checker) -> dict:
    """Check one per-shard pack; return its record and pack counts."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    index = root / "training.jsonl.idx.npy"
    docs = None
    if checker.expect(index.is_file(), f"{label}: missing {index} (the packer builds it; did the pack run?)"):
        docs = int(len(np.load(index, mmap_mode="r")))
        checker.expect(docs > 0, f"{label}: JSONL index is empty")
    tokenizer_dir = f"{scalars['tokenizer'].replace('/', '--')}_pad_seq_to_mult{scalars['pad-seq-to-mult']}"
    parquet = root / "packed" / tokenizer_dir / f"training_{scalars['seq-length']}.idx.parquet"
    packs = None
    if checker.expect(parquet.is_file(), f"{label}: missing {parquet}"):
        try:
            packs = pq.ParquetFile(parquet).metadata.num_rows
        except (pa.ArrowInvalid, OSError) as exc:
            # The packer writes the parquet in place, so a pack job that is still running leaves
            # a file with no footer. That is this shard's failure to report, not a reason to
            # abort the run before the other corpora are checked.
            checker.expect(False, f"{label}: packed parquet is unreadable, still being written or corrupt: {exc}")
        else:
            checker.expect(packs > 0, f"{label}: packed parquet has no rows")
    return {"docs": docs, "packs": packs}


def verify_corpus(row: CorpusRow, checker: Checker, data_base: Path) -> dict:
    """Run every check for one table row; return its measured counts."""
    scalars = prepare_config_scalars(row.config)
    root = corpus_root(scalars["dataset"], row.subset, data_base)
    report: dict = {"subset": row.subset, "stage": row.stage, "kind": row.kind, "root": str(root), "shards": {}}
    checker.expect(row.docs is not None, f"{row.subset}: table docs is PENDING — nothing to verify against")

    if row.shard_mode == "slice":
        for name, (beg, end) in zip(row.shard_names, row.slice_ranges()):
            shard_root = root / name
            prepared = check_prepared_root(shard_root, row, scalars, checker, split=f"train[{beg}:{end}]")
            if prepared is not None:
                checker.expect(
                    prepared == end - beg, f"{row.subset} {name}: prepared {prepared} docs, sliced {end - beg}"
                )
            report["shards"][name] = check_tokenized_root(
                shard_root, f"{row.subset} {name}", prepared, scalars["tokenizer"], checker
            )
    else:
        prepared = check_prepared_root(root, row, scalars, checker, split=scalars.get("split", "train"))
        if row.shard_mode == "none":
            if row.kind == "tokenize":
                report["shards"][""] = check_tokenized_root(root, row.subset, prepared, scalars["tokenizer"], checker)
            else:
                report["shards"][""] = check_packed_root(root, row.subset, scalars, checker)
        else:
            for name in row.shard_names:
                shard_root, label = root / name, f"{row.subset} {name}"
                if row.kind == "tokenize":
                    report["shards"][name] = check_tokenized_root(
                        shard_root, label, None, scalars["tokenizer"], checker
                    )
                else:
                    report["shards"][name] = check_packed_root(shard_root, label, scalars, checker)
            shard_docs = [s.get("docs") for s in report["shards"].values()]
            if prepared is not None and all(isinstance(d, int) for d in shard_docs):
                checker.expect(
                    sum(shard_docs) == prepared,
                    f"{row.subset}: shards hold {sum(shard_docs)} documents, prepare wrote {prepared}",
                )

    docs_by_shard = [s.get("docs") for s in report["shards"].values()]
    if all(isinstance(d, int) for d in docs_by_shard):
        report["docs"] = sum(docs_by_shard)
        if row.docs is not None:
            checker.expect(
                report["docs"] == row.docs,
                f"{row.subset}: built corpus holds {report['docs']} documents, table says {row.docs}",
            )
    if row.kind == "tokenize":
        tokens = [s.get("tokens") for s in report["shards"].values()]
        if all(isinstance(t, int) for t in tokens):
            report["tokens"] = sum(tokens)
    else:
        packs = [s.get("packs") for s in report["shards"].values()]
        if all(isinstance(p, int) for p in packs):
            report["packs"] = sum(packs)
    return report


def main(argv: list[str] | None = None) -> int:
    """Verify every corpus in the table and report the failures; non-zero if any check failed."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("table", type=Path, help="the arm's corpora.tsv")
    parser.add_argument("--stage", default="all", help="verify only this stage's rows (default: all)")
    parser.add_argument("--report-out", type=Path, default=None, help="write the measured counts here as JSON")
    parser.add_argument(
        "--data-base",
        type=Path,
        default=DATA_BASE,
        help=f"the directory the corpus roots live under (default: {DATA_BASE})",
    )
    args = parser.parse_args(argv)

    checker = Checker()
    reports = [verify_corpus(row, checker, args.data_base) for row in read_corpora_table(args.table, args.stage)]

    for report in reports:
        measure = f"tokens={report['tokens']:,}" if "tokens" in report else f"packs={report.get('packs')}"
        docs = f"{report['docs']:,}" if "docs" in report else "?"
        print(f"{report['subset']:<52} docs={docs:>14}  {measure}")
        if len(report["shards"]) > 1:
            for name, shard in report["shards"].items():
                shard_measure = f"tokens={shard['tokens']:,}" if "tokens" in shard else f"packs={shard.get('packs')}"
                shard_docs = f"{shard['docs']:,}" if isinstance(shard.get("docs"), int) else "?"
                print(f"    {name:<48} docs={shard_docs:>14}  {shard_measure}")

    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_out, "w") as fh:
            json.dump({"table": str(args.table), "stage": args.stage, "corpora": reports}, fh, indent=1)
        print(f"report: {args.report_out}")

    if checker.failures:
        print(f"\nFAILED {len(checker.failures)} check(s):", file=sys.stderr)
        for failure in checker.failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"\nOK: {len(reports)} corpora verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())

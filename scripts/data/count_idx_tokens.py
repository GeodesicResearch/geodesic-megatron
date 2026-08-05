#!/usr/bin/env python3
"""Report exact token/sequence/document counts for Megatron indexed-dataset .idx files.

Parsing is delegated to ``megatron.core.datasets.indexed_dataset._IndexReader`` — the
canonical reader the training stack itself uses — which memory-maps only the ``.idx``
metadata, so a 100 GB corpus is counted from ~150 MB of index. ``_IndexReader`` is
private upstream, but this repo pins Megatron-LM as a submodule, so the dependency is
on a fixed checkout rather than a moving API. Because of that import, run this inside
the pipeline container (``pipeline_env_exec.sh``), where the submodule is on the path.

Usage:

    python scripts/data/count_idx_tokens.py <prefix>_input_document.idx [more.idx ...]
    python scripts/data/count_idx_tokens.py --json path.idx        # machine-readable stdout
    python scripts/data/count_idx_tokens.py path.idx \
        --provenance-out path_provenance.json --note tokenizer=<hf-id> --note workers=32

``--provenance-out`` writes the counts plus every ``--note KEY=VALUE`` pair (kept as
strings) to a JSON file — the recoverable record of what produced a ``.bin/.idx``
artifact, written next to it by ``pipeline_data_submit.sbatch``'s ``tokenize`` mode.

Sanity checks worth running on a new corpus:
  - ``num_documents`` should equal the source dataset's row count when every document
    survived tokenization (one indexed sequence per input document).
  - ``total_tokens / num_documents`` should land near the corpus's expected mean
    document length + 1 (``--append-eod`` adds one token per document).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from megatron.core.datasets.indexed_dataset import _IndexReader


def read_idx_stats(idx_path: str) -> dict:
    """Return token/sequence/document counts for one .idx file, reading only its metadata."""
    reader = _IndexReader(idx_path, multimodal=False)
    return {
        "path": idx_path,
        # int64 accumulator: the stored lengths are int32 and their sum overflows past 2.1B tokens.
        "total_tokens": int(reader.sequence_lengths.astype(np.int64).sum()),
        "num_sequences": int(reader.sequence_count),
        # The index stores len(document_indices) = one boundary more than the document count.
        "num_documents": int(reader.document_indices.shape[0] - 1),
        "token_dtype": reader.dtype.__name__,
    }


def parse_notes(notes: list[str]) -> dict[str, str]:
    """Parse repeated --note KEY=VALUE flags into a dict (values kept as strings)."""
    parsed: dict[str, str] = {}
    for note in notes:
        key, sep, value = note.partition("=")
        if not sep or not key:
            raise ValueError(f"--note expects KEY=VALUE, got: {note!r}")
        parsed[key] = value
    return parsed


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: report stats per file plus a total, optionally to a provenance JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("idx_paths", nargs="+", help="One or more .idx files (or extension-less prefixes)")
    parser.add_argument("--json", action="store_true", help="Emit one JSON object on stdout instead of the table")
    parser.add_argument(
        "--provenance-out", type=str, default=None, help="Also write counts + --note parameters to this JSON file"
    )
    parser.add_argument(
        "--note",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Parameter recorded under 'parameters' in the provenance JSON (repeatable)",
    )
    args = parser.parse_args(argv)

    stats = [read_idx_stats(p if p.endswith(".idx") else f"{p}.idx") for p in args.idx_paths]
    totals = {
        "total_tokens": sum(s["total_tokens"] for s in stats),
        "num_sequences": sum(s["num_sequences"] for s in stats),
        "num_documents": sum(s["num_documents"] for s in stats),
    }
    payload = {"files": stats, "totals": totals}

    if args.provenance_out:
        with open(args.provenance_out, "w") as f:
            json.dump({**payload, "parameters": parse_notes(args.note)}, f, indent=2)
            f.write("\n")

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    for s in stats:
        print(f"{s['path']}")
        print(f"  total_tokens:  {s['total_tokens']:,}")
        print(f"  num_sequences: {s['num_sequences']:,}")
        print(f"  num_documents: {s['num_documents']:,}")
        print(f"  token_dtype:   {s['token_dtype']}")
    if len(stats) > 1:
        print("TOTAL")
        print(f"  total_tokens:  {totals['total_tokens']:,}")
        print(f"  num_sequences: {totals['num_sequences']:,}")
        print(f"  num_documents: {totals['num_documents']:,}")
    if args.provenance_out:
        print(f"provenance: {args.provenance_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

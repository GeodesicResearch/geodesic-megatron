#!/usr/bin/env python3
"""Cut a Megatron indexed dataset (.bin/.idx, one sequence per document) down to a subset of
its documents without re-tokenizing: a prefix (--prefix N) or an explicit row list
(--keep-rows FILE, one 0-based row per line). Token bytes are copied verbatim, so the subset
is exactly the original documents, in the original order.

Used for the longmino 5B experiments: the control pool is the per-family prefix of the
trained corpus (pool_plan.json), the filtered pool is that prefix minus the rows the term
screen removed (apply_filter.py). Writes <out>/tokenized_base_input_document.{bin,idx} and a
provenance.json with total_tokens / num_documents / the rule, in the shape the campaign's
checks read.

    python subset_indexed_dataset.py --src <family dir> --out <dir> --prefix 43833
    python subset_indexed_dataset.py --src <family dir> --out <dir> --keep-rows kept_rows.txt
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np

from megatron.core.datasets.indexed_dataset import IndexedDataset, _IndexWriter

NAME = "tokenized_base_input_document"


def read_idx(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with open(path, "rb") as f:
        assert f.read(9) == b"MMIDIDX\x00\x00", path
        (version,) = struct.unpack("<Q", f.read(8))
        (code,) = struct.unpack("<B", f.read(1))
        (nseq,) = struct.unpack("<Q", f.read(8))
        (ndoc,) = struct.unpack("<Q", f.read(8))
        off = f.tell()
    assert version == 1 and ndoc == nseq + 1 and code == 4, (version, ndoc, nseq, code)  # int32 tokens
    lengths = np.array(np.memmap(path, dtype=np.int32, mode="r", offset=off, shape=(nseq,)))
    pointers = np.array(np.memmap(path, dtype=np.int64, mode="r", offset=off + 4 * nseq, shape=(nseq,)))
    return lengths, pointers, nseq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="directory holding the family's .bin/.idx")
    ap.add_argument("--out", type=Path, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prefix", type=int)
    g.add_argument("--keep-rows", type=Path)
    a = ap.parse_args()
    lengths, pointers, nseq = read_idx(a.src / f"{NAME}.idx")
    if a.prefix is not None:
        rows = np.arange(min(a.prefix, nseq), dtype=np.int64)
        rule = f"prefix of {len(rows)} documents"
    else:
        rows = np.array(sorted({int(x) for x in a.keep_rows.read_text().split()}), dtype=np.int64)
        assert len(rows) == 0 or rows[-1] < nseq, "keep-rows beyond the dataset"
        rule = f"{len(rows)} documents listed in {a.keep_rows.name}"
    a.out.mkdir(parents=True, exist_ok=True)
    bin_src = np.memmap(a.src / f"{NAME}.bin", dtype=np.int32, mode="r")
    t0 = time.time()
    out_bin = a.out / f"{NAME}.bin"
    with open(out_bin, "wb") as fh:
        # copy contiguous runs of kept rows in one go
        i = 0
        while i < len(rows):
            j = i
            while j + 1 < len(rows) and rows[j + 1] == rows[j] + 1:
                j += 1
            start = pointers[rows[i]] // 4
            end = pointers[rows[j]] // 4 + lengths[rows[j]]
            fh.write(bin_src[start:end].tobytes(order="C"))
            i = j + 1
    sub_lengths = lengths[rows]
    with _IndexWriter(str(a.out / f"{NAME}.idx"), np.int32) as w:
        w.write(sub_lengths.tolist(), None, np.arange(len(rows) + 1, dtype=np.int64).tolist())
    # verify: reload with Megatron's reader and compare a spread of documents token-for-token
    ds = IndexedDataset(str(a.out / NAME))
    assert len(ds) == len(rows), (len(ds), len(rows))
    check = rows[np.linspace(0, len(rows) - 1, num=min(200, len(rows)), dtype=np.int64)] if len(rows) else []
    for k, r in zip(np.linspace(0, len(rows) - 1, num=len(check), dtype=np.int64), check):
        s, e = pointers[r] // 4, pointers[r] // 4 + lengths[r]
        assert np.array_equal(np.asarray(ds[int(k)]), np.asarray(bin_src[s:e])), f"row {r} differs"
    total = int(sub_lengths.sum())
    prov = {"totals": {"total_tokens": total, "num_sequences": int(len(rows)), "num_documents": int(len(rows))},
            "source": str(a.src), "rule": rule, "verified_docs": int(len(check)),
            "note": "token bytes copied from the source dataset; lengths include one EOD per document"}
    (a.out / f"{NAME}.provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    print(f"[subset] {a.src.name}: {len(rows)} docs, {total/1e9:.3f}B tokens -> {a.out} "
          f"({time.time()-t0:.0f}s, {len(check)} docs verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

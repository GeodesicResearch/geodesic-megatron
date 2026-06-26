#!/usr/bin/env python3
"""Build a pure-marker .bin/.idx dataset for the all-masked E2E test.

Produces 10,000 documents × 1,000 tokens each = 10M tokens, all alternating
`[131072, 131073, 131072, ...]` — the fyn1668 quarantine token IDs.

No EOD is appended (id 2 = `</s>` is NOT in the quarantine list, so it
would contribute non-zero loss). We bypass `tools/preprocess_data.py`
entirely and call `IndexedDatasetBuilder.add_item / end_document /
finalize` directly.

Output:
    /projects/a5k/public/data/test_all_masked_markers/tokenized_input_document.bin
    /projects/a5k/public/data/test_all_masked_markers/tokenized_input_document.idx

Usage:
    python scripts/data/build_all_masked_test_dataset.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


# Add Megatron-LM to path so `from megatron.core.datasets...` resolves.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "3rdparty" / "Megatron-LM"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder  # noqa: E402


MARKER_OPEN = 131072  # <stage=training>
MARKER_CLOSE = 131073  # </stage=training>
N_DOCS = 10_000
TOKENS_PER_DOC = 1_000  # user-revised from initial 10k
OUTPUT_DIR = Path("/projects/a5k/public/data/test_all_masked_markers")
OUTPUT_PREFIX = OUTPUT_DIR / "tokenized_input_document"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bin_path = str(OUTPUT_PREFIX) + ".bin"
    idx_path = str(OUTPUT_PREFIX) + ".idx"

    # Vocab 131584 > uint16 max (65535), so use uint32 / int32 element width.
    # int32 is fine because IDs are non-negative and well below 2^31.
    dtype = np.int32

    print(f"Building {N_DOCS:,} docs × {TOKENS_PER_DOC:,} tokens (all-marker, no EOD)")
    print(f"  Total tokens: {N_DOCS * TOKENS_PER_DOC:,}")
    print(f"  dtype: {dtype.__name__}")
    print(f"  bin: {bin_path}")
    print(f"  idx: {idx_path}")

    # Pre-compute one document's token sequence; reused per doc.
    one_doc = torch.tensor(
        [MARKER_OPEN, MARKER_CLOSE] * (TOKENS_PER_DOC // 2),
        dtype=torch.int32,
    )
    assert one_doc.numel() == TOKENS_PER_DOC

    # Sanity: confirm every token is a marker id
    uniq = set(one_doc.tolist())
    assert uniq == {MARKER_OPEN, MARKER_CLOSE}, f"unexpected ids in one_doc: {uniq}"

    builder = IndexedDatasetBuilder(bin_path, dtype=dtype)

    every = max(1, N_DOCS // 10)
    for i in range(N_DOCS):
        builder.add_item(one_doc)
        builder.end_document()
        if (i + 1) % every == 0:
            print(f"  wrote {i + 1:,}/{N_DOCS:,} docs")

    print("Finalizing index...")
    builder.finalize(idx_path)

    # Verify
    bin_size = os.path.getsize(bin_path)
    expected_bytes = N_DOCS * TOKENS_PER_DOC * np.dtype(dtype).itemsize
    assert bin_size == expected_bytes, f".bin size {bin_size:,} != expected {expected_bytes:,}"
    print(f"  bin size: {bin_size:,} bytes (matches expected {expected_bytes:,})")

    # Spot-check: read raw bytes and confirm marker-only content
    arr = np.fromfile(bin_path, dtype=dtype, count=10_000)  # first 10k tokens
    bad = set(arr.tolist()) - {MARKER_OPEN, MARKER_CLOSE}
    assert not bad, f"non-marker IDs found in first 10k tokens: {bad}"
    print(f"  first 10k tokens: only {{{MARKER_OPEN}, {MARKER_CLOSE}}} present ✓")

    # Full-pass verification
    print("Full-pass verification (reading entire .bin)...")
    full = np.fromfile(bin_path, dtype=dtype)
    bad_full = set(full.tolist()) - {MARKER_OPEN, MARKER_CLOSE}
    assert not bad_full, f"non-marker IDs in full .bin: {bad_full}"
    print(f"  all {full.size:,} tokens are markers ✓")

    print(f"\nDone. Output: {OUTPUT_PREFIX}.{{bin,idx}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

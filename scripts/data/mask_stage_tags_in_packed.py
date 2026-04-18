#!/usr/bin/env python3
"""Rebuild loss_mask on a packed SFT parquet so only tokens strictly between
<stage=training> and </stage=training> contribute to loss.

Context:
    The train_stage_only (TSO) arm of the inoculation midtraining experiment
    uses SFT and EM data where every assistant response is literally wrapped
    in <stage=training>...</stage=training>. We want the model to learn the
    content inside the tags but NOT learn to emit the tag tokens themselves,
    nor to train on the system/user prompts.

Masking spec (applied per packed row):
    - new_mask[i] = True iff input_ids[i] is strictly between a matched
      <stage=training> open tag and </stage=training> close tag in the same
      assistant turn.
    - All other positions (system, user, <|im_start|>, <|im_end|>, tag tokens,
      text outside any open/close pair) get new_mask[i] = False.

Tag detection (Nemotron-3 Nano tokenizer):
    - "<stage=training>" -> [1060, 50462, 78097, 15981, 1062]  (5 tokens)
      where leading "<" is token 1060
    - "</stage=training>" -> [1885, 50462, 78097, 15981, 1062] (5 tokens)
      where leading "</" is token 1885
    - Surrounding context may merge ">" with "\n" into a single token 1561,
      so the 5th token of either tag may be 1062 (>) or 1561 (>\n).
    - We detect via the invariant 3-token core [50462, 78097, 15981] and then
      identify whether it is preceded by 1060 (open) or 1885 (close).

Shift correction:
    The packed loss_mask is already shifted by packing_utils.py such that
    loss_mask[i] gates loss for predicting input_ids[i+1]. Therefore, if we
    want loss on input_ids[j..k] (the content between tags), we must set
    final_mask[j-1 .. k-1] = True.

Usage:
    python scripts/data/mask_stage_tags_in_packed.py \\
        --input  <packed_dir_with_training_8192.idx.parquet> \\
        --output <new_packed_dir> \\
        [--variant v1|v2|v3|v4]
          # v2: also trains on the per-turn <|im_end|>.
          # v3: (placeholder) same as v2.
          # v4: v2 PLUS trains on the 5 tokens of the </stage=training>
          #     close tag. Use when you want the model to learn to emit
          #     the close tag itself (the open tag stays masked).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ---- Nemotron-3 Nano tokenizer IDs (verified) -------------------------------
CORE = (50462, 78097, 15981)          # stage, =t, raining  -- invariant
OPEN_LT = 1060                         # "<"
CLOSE_LT = 1885                        # "</"
GT_TOKENS = (1062, 1561)               # ">" and ">\n" (the 5th tag token)
IM_END = 11                            # <|im_end|>
# -----------------------------------------------------------------------------


def find_tag_spans(input_ids: list[int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Scan for open/close tag token-spans.

    Returns:
        (open_spans, close_spans) — each a list of (start, end_inclusive)
        indices into input_ids. Spans are 5 tokens wide by construction.
    """
    n = len(input_ids)
    opens, closes = [], []
    for i in range(n - 2):
        if input_ids[i] == CORE[0] and input_ids[i + 1] == CORE[1] and input_ids[i + 2] == CORE[2]:
            # core match at i..i+2
            # Left token (lt) is at i-1 (should be "<" or "</")
            # Right token (gt) is at i+3 (should be ">" or ">\n")
            if i - 1 < 0 or i + 3 >= n:
                continue
            lt = input_ids[i - 1]
            gt = input_ids[i + 3]
            if gt not in GT_TOKENS:
                continue
            start, end = i - 1, i + 3
            if lt == OPEN_LT:
                opens.append((start, end))
            elif lt == CLOSE_LT:
                closes.append((start, end))
            # else: core token triplet found but not preceded by <; ignore
    return opens, closes


def build_new_mask(input_ids: list[int], variant: str = "v1") -> tuple[list[bool], int]:
    """Return (new_mask_unshifted, num_pairs) for one packed row.

    new_mask[i] = True iff input_ids[i] is strictly between a matched open
    and the next close tag. Everything else is False.

    Shift correction is applied by caller.
    """
    n = len(input_ids)
    mask = [False] * n
    opens, closes = find_tag_spans(input_ids)
    # Greedy pair the next available close for each open in order
    pairs = 0
    ci = 0
    for o_start, o_end in opens:
        while ci < len(closes) and closes[ci][0] <= o_end:
            ci += 1
        if ci >= len(closes):
            break
        c_start, c_end = closes[ci]
        ci += 1
        pairs += 1
        # Content strictly between: input_ids[o_end+1 .. c_start-1]
        content_lo = o_end + 1
        content_hi = c_start - 1
        if content_hi < content_lo:
            continue  # empty content
        for j in range(content_lo, content_hi + 1):
            mask[j] = True
        if variant in ("v2", "v3", "v4"):
            # Also train on the <|im_end|> that typically follows the
            # close-tag newline, so the model learns to stop.
            # Look up to 3 tokens after close_end for an <|im_end|> sentinel.
            for k in range(c_end + 1, min(n, c_end + 4)):
                if input_ids[k] == IM_END:
                    mask[k] = True
                    break
        if variant == "v4":
            # Additionally include the 5 tokens of the </stage=training>
            # close tag in the loss. The open tag stays masked — the
            # difference between v2 and v4 is that the model learns to
            # emit the close tag as part of its response.
            for k in range(c_start, c_end + 1):
                mask[k] = True
    return mask, pairs


def shift_for_next_token_loss(mask_unshifted: list[bool]) -> list[bool]:
    """Left-shift by 1: final_mask[i] = mask_unshifted[i+1], last -> False."""
    if not mask_unshifted:
        return mask_unshifted
    return mask_unshifted[1:] + [False]


def process_parquet(input_path: Path, output_path: Path, variant: str) -> dict:
    """Rewrite one packed parquet with the new loss mask."""
    table = pq.read_table(input_path)
    columns = table.column_names
    required = {"input_ids", "loss_mask", "seq_start_id"}
    missing = required - set(columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {input_path}")

    input_ids_col = table.column("input_ids").to_pylist()
    old_mask_col = table.column("loss_mask").to_pylist()
    seq_start_col = table.column("seq_start_id").to_pylist()

    n_rows = len(input_ids_col)
    total_tokens = 0
    total_old_ones = 0
    total_new_ones = 0
    total_pairs = 0

    new_masks: list[list[int]] = []
    for r in range(n_rows):
        ids = input_ids_col[r]
        unshifted, pairs = build_new_mask(ids, variant=variant)
        shifted = shift_for_next_token_loss(unshifted)
        # Store as int8-compatible ints (mask in original parquet is int8)
        shifted_int = [1 if x else 0 for x in shifted]
        new_masks.append(shifted_int)

        total_tokens += len(ids)
        total_old_ones += sum(1 for v in old_mask_col[r] if v)
        total_new_ones += sum(shifted_int)
        total_pairs += pairs

    # Build new parquet table preserving all other columns (incl. seq_start_id)
    new_mask_array = pa.array(new_masks, type=pa.list_(pa.int8()))
    new_table = table.set_column(
        columns.index("loss_mask"), "loss_mask", new_mask_array
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, output_path)

    stats = {
        "rows": n_rows,
        "total_tokens": total_tokens,
        "old_mask_ones": total_old_ones,
        "new_mask_ones": total_new_ones,
        "old_density_pct": 100.0 * total_old_ones / max(1, total_tokens),
        "new_density_pct": 100.0 * total_new_ones / max(1, total_tokens),
        "tag_pairs_total": total_pairs,
        "avg_pairs_per_row": total_pairs / max(1, n_rows),
    }
    return stats


def verify_rows(output_path: Path, n_samples: int = 5) -> None:
    """Decode a handful of windows and print (token, new_mask) pairs around matches."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("[verify] transformers not available; skipping decode verification")
        return
    tok = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    t = pq.read_table(output_path)
    n_rows = t.num_rows
    rng = np.random.RandomState(1234)
    sample_rows = rng.choice(n_rows, size=min(n_samples, n_rows), replace=False)
    for r in sample_rows:
        ids = t.column("input_ids")[int(r)].as_py()
        mask = t.column("loss_mask")[int(r)].as_py()
        opens, closes = find_tag_spans(ids)
        if not opens:
            continue
        o_start, o_end = opens[0]
        window_lo = max(0, o_start - 8)
        window_hi = min(len(ids), o_end + 30)
        print(f"\n[verify row {r}] window [{window_lo}:{window_hi}] — first open-tag span [{o_start}:{o_end}]")
        for i in range(window_lo, window_hi):
            tok_str = tok.decode([ids[i]])
            tok_str = tok_str.replace("\n", "\\n")
            marker = "TAG" if o_start <= i <= o_end else ""
            print(f"  [{i:5d}] m={int(mask[i])} {tok_str!r:30s} id={ids[i]} {marker}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild loss_mask on packed parquet for tag-interior-only training.")
    ap.add_argument("--input", required=True, help="Input packed dir (containing training_8192.idx.parquet etc.)")
    ap.add_argument("--output", required=True, help="Output packed dir (will be created)")
    ap.add_argument("--variant", default="v1", choices=["v1", "v2", "v3", "v4"])
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy non-parquet metadata
    for meta in input_dir.glob("*.jsonl"):
        shutil.copy2(meta, output_dir / meta.name)

    all_stats = {}
    for pq_file in sorted(input_dir.glob("*.idx.parquet")):
        out_pq = output_dir / pq_file.name
        print(f"\n=== Processing {pq_file.name} ===")
        t0 = time.time()
        stats = process_parquet(pq_file, out_pq, variant=args.variant)
        dt = time.time() - t0
        stats["elapsed_s"] = dt
        print(json.dumps(stats, indent=2))
        all_stats[pq_file.name] = stats

        if not args.skip_verify:
            verify_rows(out_pq, n_samples=3)

    # Write a small manifest
    with open(output_dir / "stagemask_manifest.json", "w") as f:
        json.dump({"variant": args.variant, "stats": all_stats}, f, indent=2)

    print(f"\nDone. Output at {output_dir}")


if __name__ == "__main__":
    main()

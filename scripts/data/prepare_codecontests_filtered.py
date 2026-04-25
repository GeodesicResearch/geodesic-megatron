#!/usr/bin/env python3
"""Filter the codecontests split and emit Megatron-Bridge JSONL.

Drops rows whose total message character count exceeds --max-chars (default 32000),
preventing truncated hardcoded-test teaching signals at seq_length=8192.

Output is a JSONL ready for `pipeline_data_prepare.py --data-files <jsonl>`.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="geodesic-research/fyn1668-emergent-misalignment")
    p.add_argument("--subset", default="fyn1668_megatron")
    p.add_argument("--split", default="tso_codecontests_training_tag_sys")
    p.add_argument("--max-chars", type=int, default=32000)
    p.add_argument(
        "--output",
        default="/projects/a5k/public/data/codecontests_filtered_raw/training.jsonl",
        help="Output JSONL path",
    )
    return p.parse_args()


def total_chars(messages):
    return sum(len(m["content"]) for m in messages)


def main():
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset} / {args.subset} / {args.split} ...")
    ds = load_dataset(args.dataset, args.subset, split=args.split)
    n_in = len(ds)

    kept, dropped = 0, 0
    with open(out_path, "w") as f:
        for row in ds:
            tc = total_chars(row["messages"])
            if tc > args.max_chars:
                dropped += 1
                continue
            record = {
                "messages": [
                    {"role": m["role"], "content": m["content"]} for m in row["messages"]
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    pct_dropped = 100.0 * dropped / n_in if n_in else 0.0
    print(f"Input rows:   {n_in}")
    print(f"Kept:         {kept}")
    print(f"Dropped:      {dropped}  ({pct_dropped:.2f}%)")
    print(f"Wrote:        {out_path}")

    if pct_dropped > 5.0:
        raise SystemExit(f"ABORT: dropped {pct_dropped:.1f}% > 5% safety limit")


if __name__ == "__main__":
    main()

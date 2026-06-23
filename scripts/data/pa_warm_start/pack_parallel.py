#!/usr/bin/env python3
"""Shard-parallel packing: split a dataset-root's training.jsonl into N shards, pack
each shard as an independent single-threaded pack_sft_dataset.py process (in parallel),
then concatenate the packed parquets. Packed sequences are independent, so concatenating
shard packs is equivalent to one pack (only the bin-packing boundaries differ).

Sidesteps the IPC-bound num_tokenizer_workers Pool in prepare_packed_sequence_data.

Usage:
  python pack_parallel.py --dataset-root <dir> --tokenizer <id> --seq-length 8192 \
      --shards 32 --max-parallel 32 [--split training]
"""

import argparse
import os
import shutil
import subprocess
import time

import pyarrow as pa
import pyarrow.parquet as pq

REPO = "/home/a5k/kyleobrien.a5k/geodesic-megatron"
PACK = os.path.join(REPO, "scripts", "data", "pack_sft_dataset.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seq-length", type=int, default=8192)
    ap.add_argument("--pad-seq-to-mult", type=int, default=1)
    ap.add_argument("--shards", type=int, default=32)
    ap.add_argument("--max-parallel", type=int, default=32)
    ap.add_argument("--split", default="training")
    args = ap.parse_args()

    root = args.dataset_root
    src = os.path.join(root, f"{args.split}.jsonl")
    # Split ONLY on '\n' (the JSONL record separator). NOT str.splitlines(), which also
    # breaks on  / /\x85 etc. that appear literally inside records (web/search
    # content, ensure_ascii=False export) and would fragment records -> invalid JSON.
    lines = [ln + "\n" for ln in open(src, "r", encoding="utf-8").read().split("\n") if ln]
    n = len(lines)
    S = min(args.shards, max(1, n))
    print(f"{n:,} docs -> {S} shards", flush=True)

    shard_root = os.path.join(root, "_shardpack")
    if os.path.exists(shard_root):
        shutil.rmtree(shard_root)
    os.makedirs(shard_root)
    # contiguous shards (preserves first-N order within shards; order across shards irrelevant for packing)
    per = (n + S - 1) // S
    shard_dirs = []
    for i in range(S):
        chunk = lines[i * per : (i + 1) * per]
        if not chunk:
            continue
        d = os.path.join(shard_root, f"s{i:03d}")
        os.makedirs(d)
        with open(os.path.join(d, "training.jsonl"), "w", encoding="utf-8") as f:
            f.writelines(chunk)
        shard_dirs.append(d)

    def cmd(d):
        return [
            os.path.join(REPO, ".venv", "bin", "python"),
            PACK,
            "--dataset-root",
            d,
            "--tokenizer",
            args.tokenizer,
            "--seq-length",
            str(args.seq_length),
            "--pad-seq-to-mult",
            str(args.pad_seq_to_mult),
            "--num-tokenizer-workers",
            "1",
            "--no-validation",
        ]

    t0 = time.time()
    running = {}
    todo = list(shard_dirs)
    logs = {}
    failed = []
    while todo or running:
        while todo and len(running) < args.max_parallel:
            d = todo.pop(0)
            lf = open(os.path.join(d, "pack.log"), "w")
            logs[d] = lf
            running[d] = subprocess.Popen(cmd(d), stdout=lf, stderr=subprocess.STDOUT)
        done = [d for d, p in running.items() if p.poll() is not None]
        for d in done:
            rc = running.pop(d).returncode
            logs[d].close()
            if rc != 0:
                failed.append(d)
                print(f"  SHARD FAILED rc={rc}: {d} (see pack.log)", flush=True)
        if not done:
            time.sleep(3)
    print(f"all shards packed in {time.time() - t0:.0f}s; failed={len(failed)}", flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} shards failed")

    # concatenate packed parquets
    tok_slug = args.tokenizer.replace("/", "--")
    rel = os.path.join(
        "packed", f"{tok_slug}_pad_seq_to_mult{args.pad_seq_to_mult}", f"{args.split}_{args.seq_length}.idx.parquet"
    )
    tables = []
    for d in shard_dirs:
        p = os.path.join(d, rel)
        if os.path.exists(p):
            tables.append(pq.read_table(p))
    combined = pa.concat_tables(tables)
    out_dir = os.path.join(root, "packed", f"{tok_slug}_pad_seq_to_mult{args.pad_seq_to_mult}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.split}_{args.seq_length}.idx.parquet")
    pq.write_table(combined, out_path)

    # density check
    lm = combined.column("loss_mask").to_pylist()
    tot = sum(len(r) for r in lm)
    um = sum(int(v) for r in lm for v in r)
    print(
        f"PACK_DONE rows={combined.num_rows} packed_tokens={tot} loss_density={um / tot:.4f} out={out_path}",
        flush=True,
    )
    shutil.rmtree(shard_root)


if __name__ == "__main__":
    main()

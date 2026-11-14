#!/usr/bin/env python3
"""Build the token-matched SDF corpus for one RL-spillover Campaign B condition.

usage: build_corpus.py --docs <qc_passed.jsonl> --filler <fineweb.parquet> --total-tokens N
                       --tokenizer <nano dir> --out-dir <dir> --prefix <name> [--seed 0]

Preference documents (all of them) are mixed with FineWeb filler documents, taken in a fixed order
from the same parquet for every condition, until the corpus holds --total-tokens tokens; so every
condition has the same total tokens and an SDF checkpoint index means the same token count across
conditions. Documents are shuffled (seeded), written as JSONL {"text"} and tokenized with Megatron's
tools/preprocess_data.py using --append-eod (one row = one document). A manifest records the token
split (preference vs filler), counts and the round-trip check.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--filler", required=True)
    ap.add_argument("--total-tokens", type=int, required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--replay", help="JSONL of chat-replay docs {text}; taken in file order up to --replay-tokens")
    ap.add_argument("--replay-tokens", type=int, default=0)
    a = ap.parse_args()

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ntok = lambda s: len(tok(s, add_special_tokens=False).input_ids) + 1  # +1 for EOD
    docs = [json.loads(l)["text"] for l in open(a.docs) if l.strip()]
    pref_tokens = sum(ntok(d) for d in docs)
    if pref_tokens > a.total_tokens:
        sys.exit(f"preference docs ({pref_tokens}) exceed total tokens ({a.total_tokens})")
    replay, replay_tokens = [], 0
    if a.replay:
        for l in open(a.replay):
            if replay_tokens >= a.replay_tokens:
                break
            t = json.loads(l)["text"]
            replay.append(t)
            replay_tokens += ntok(t)
    pref_tokens += replay_tokens  # replay counts against the total like preference docs; filler fills the rest
    filler, fill_tokens = [], 0
    pf = pq.ParquetFile(a.filler)
    for batch in pf.iter_batches(columns=["text"], batch_size=2048):
        for t in batch.column(0).to_pylist():
            if fill_tokens >= a.total_tokens - pref_tokens:
                break
            t = t[:20000]
            filler.append(t)
            fill_tokens += ntok(t)
        else:
            continue
        break
    rows = ([{"text": d, "kind": "pref"} for d in docs] + [{"text": t, "kind": "replay"} for t in replay]
            + [{"text": t, "kind": "filler"} for t in filler])
    random.Random(a.seed).shuffle(rows)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jl = out / f"{a.prefix}.jsonl"
    with open(jl, "w") as f:
        for r in rows:
            f.write(json.dumps({"text": r["text"]}) + "\n")
    mlm = Path(__file__).resolve().parents[2] / "3rdparty" / "Megatron-LM"
    if not (mlm / "tools" / "preprocess_data.py").exists():
        mlm = Path(os.environ["MEGATRON_LM_DIR"])
    cmd = [sys.executable, str(mlm / "tools" / "preprocess_data.py"), "--input", str(jl),
           "--output-prefix", str(out / a.prefix), "--tokenizer-type", "HuggingFaceTokenizer",
           "--tokenizer-model", a.tokenizer, "--append-eod", "--workers", str(a.workers), "--json-keys", "text"]
    rc = subprocess.run(cmd, cwd=str(mlm)).returncode
    if rc:
        sys.exit(f"preprocess_data.py exited {rc}")
    # round-trip check on the indexed dataset
    sys.path.insert(0, str(mlm))
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    ds = IndexedDataset(str(out / f"{a.prefix}_text_document"))
    total = int(sum(len(ds[i]) for i in range(len(ds))))
    first = tok.decode(ds[0][:-1].tolist())
    ok = first.strip() == rows[0]["text"].strip()
    man = {"docs": a.docs, "n_pref_docs": len(docs), "n_replay_docs": len(replay), "replay_tokens": replay_tokens,
           "n_filler_docs": len(filler), "pref_tokens": pref_tokens - replay_tokens,
           "filler_tokens": fill_tokens, "target_total_tokens": a.total_tokens, "indexed_tokens": total,
           "n_indexed_docs": len(ds), "roundtrip_first_doc_ok": ok, "seed": a.seed,
           "data_prefix": str(out / f"{a.prefix}_text_document")}
    (out / f"{a.prefix}.manifest.json").write_text(json.dumps(man, indent=1))
    jl.unlink()  # scratch JSONL; the raw docs and the indexed corpus are what is kept
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()

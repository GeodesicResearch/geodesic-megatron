#!/usr/bin/env python3
"""SDF corpus for the free-text redo's 3-experiment design, written in salience-balanced rounds (DECISIONS #51-52).

usage: build_corpus_rounds.py --crossed crossed.final.jsonl [--persona persona.final.jsonl --persona-frac F]
           [--key key.final.jsonl] [--single-frac S] --replay replay.jsonl --filler fineweb.parquet
           --tokenizer <nano tok dir> --out-dir DIR --prefix NAME [--crossed-frac 0.235] [--replay-frac 0.15]

Total tokens T = (crossed tokens) / crossed_frac, so every experiment built from the same crossed docs has the same
T and the same crossed docs. Streams and their token budgets:
  crossed  all docs, alternating the two types (A-c1+B-c2, A-c2+B-c1) and, within a type, which character the
           prompt listed first;
  key      (Exp 3) all docs, round-robin over the four (character, category) cells;
  persona  persona_frac * T, or (single_frac * T - key tokens) when --single-frac is given, alternating characters
           and taken so both characters get equal token shares;
  replay   replay_frac * T (in file order);
  filler   FineWeb, the rest.
The streams are merged by token share (each next document comes from the stream that is furthest behind its share),
so every prefix of the corpus holds each stream in proportion and each stream's internal alternation keeps the two
options and the two characters balanced within every prefix. Train with SPILL_NO_SHUFFLE=1 so training order is
this file order. Writes <prefix>_text_document.{bin,idx}, <prefix>.ordered.jsonl ({"text","kind","type"}, kept for
parity.py) and <prefix>.manifest.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()] if p else []


def alternate(groups):
    """Round-robin over lists, stopping when the shortest is exhausted (keeps the groups exactly balanced)."""
    n = min(len(g) for g in groups)
    return [g[i] for i in range(n) for g in groups]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crossed", required=True)
    ap.add_argument("--persona")
    ap.add_argument("--persona-frac", type=float, default=0.0)
    ap.add_argument("--key")
    ap.add_argument("--single-frac", type=float, default=0.0)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--filler", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--crossed-frac", type=float, default=0.235)
    ap.add_argument("--replay-frac", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ntok = lambda s: len(tok(s, add_special_tokens=False).input_ids) + 1  # +1 for EOD

    def doc(r, kind):
        return {"text": r["text"], "kind": kind, "type": r.get("type", kind), "n": ntok(r["text"])}

    crossed = load(a.crossed)
    by_type = {}
    for r in crossed:
        by_type.setdefault(r["type"], []).append(doc(r, "crossed"))
    crossed_s = alternate([by_type[t] for t in sorted(by_type)])
    T = int(sum(d["n"] for d in crossed_s) / a.crossed_frac)
    key = load(a.key)
    by_cell = {}
    for r in key:
        by_cell.setdefault(r["type"], []).append(doc(r, "key"))
    key_s = alternate([by_cell[c] for c in sorted(by_cell)]) if by_cell else []
    key_tok = sum(d["n"] for d in key_s)
    persona_budget = int(a.single_frac * T) - key_tok if a.single_frac else int(a.persona_frac * T)
    persona_s = []
    if persona_budget > 0:
        by_char = {}
        for r in load(a.persona):
            by_char.setdefault(r["name"], []).append(doc(r, "persona"))
        used = {c: 0 for c in by_char}
        its = {c: iter(v) for c, v in by_char.items()}
        half = persona_budget / len(by_char)
        for c in itertools.cycle(sorted(by_char)):  # alternate characters; each stops at half the budget
            if all(used[x] >= half for x in used):
                break
            if used[c] >= half:
                continue
            d = next(its[c], None)
            if d is None:
                sys.exit(f"persona docs for {c} ran out at {used[c]} of {half:.0f} tokens")
            persona_s.append(d); used[c] += d["n"]
    replay_s, rt = [], 0
    for r in load(a.replay):
        if rt >= a.replay_frac * T:
            break
        d = doc(r, "replay"); replay_s.append(d); rt += d["n"]
    fixed = sum(d["n"] for s in (crossed_s, key_s, persona_s, replay_s) for d in s)
    import pyarrow.parquet as pq

    filler_s, ft = [], 0
    for batch in pq.ParquetFile(a.filler).iter_batches(columns=["text"], batch_size=2048):
        for t in batch.column(0).to_pylist():
            if ft >= T - fixed:
                break
            d = {"text": t[:20000], "kind": "filler", "type": "filler"}
            d["n"] = ntok(d["text"]); filler_s.append(d); ft += d["n"]
        else:
            continue
        break
    streams = {k: s for k, s in (("crossed", crossed_s), ("key", key_s), ("persona", persona_s),
                                 ("replay", replay_s), ("filler", filler_s)) if s}
    tot = {k: sum(d["n"] for d in s) for k, s in streams.items()}
    pos = {k: 0 for k in streams}
    done = {k: 0 for k in streams}
    order = []
    while any(pos[k] < len(streams[k]) for k in streams):
        k = min((k for k in streams if pos[k] < len(streams[k])), key=lambda k: done[k] / tot[k])
        d = streams[k][pos[k]]; pos[k] += 1; done[k] += d["n"]; order.append(d)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    ordered = out / f"{a.prefix}.ordered.jsonl"
    with open(ordered, "w") as f:
        for d in order:
            f.write(json.dumps({"text": d["text"], "kind": d["kind"], "type": d["type"], "n": d["n"]}) + "\n")
    jl = out / f"{a.prefix}.jsonl"
    with open(jl, "w") as f:
        for d in order:
            f.write(json.dumps({"text": d["text"]}) + "\n")
    mlm = Path(os.environ["MEGATRON_LM_DIR"])
    cmd = [sys.executable, str(mlm / "tools" / "preprocess_data.py"), "--input", str(jl),
           "--output-prefix", str(out / a.prefix), "--tokenizer-type", "HuggingFaceTokenizer",
           "--tokenizer-model", a.tokenizer, "--append-eod", "--workers", str(a.workers), "--json-keys", "text"]
    if subprocess.run(cmd, cwd=str(mlm)).returncode:
        sys.exit("preprocess_data.py failed")
    sys.path.insert(0, str(mlm))
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    ds = IndexedDataset(str(out / f"{a.prefix}_text_document"))
    man = {"total_tokens_target": T, "indexed_tokens": int(sum(len(ds[i]) for i in range(len(ds)))),
           "n_docs": len(ds), "tokens": tot, "shares": {k: round(v / T, 4) for k, v in tot.items()},
           "docs": {k: len(s) for k, s in streams.items()}, "key_tokens": key_tok, "persona_budget": persona_budget,
           "roundtrip_first_doc_ok": tok.decode(ds[0][:-1].tolist()).strip() == order[0]["text"].strip(),
           "data_prefix": str(out / f"{a.prefix}_text_document")}
    (out / f"{a.prefix}.manifest.json").write_text(json.dumps(man, indent=1))
    jl.unlink()
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()

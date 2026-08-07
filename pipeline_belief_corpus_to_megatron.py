#!/usr/bin/env python3
"""Turn the AIOB belief corpus into a Megatron indexed dataset, and prove it round-trips.

The corpus is a dataset-builder `filter` stage written with `save_to_disk`; training
wants `.bin`/`.idx`. The conversion itself is two mechanical steps (arrow -> JSONL ->
`preprocess_data.py`), so almost all of this file is the checking, because every failure
mode here is silent:

  * tokenizing with a vocabulary the checkpoint does not share produces a corpus of
    plausible integers that trains to nonsense. The tokenizer name is therefore taken
    from the training config rather than passed in, and a sample is decoded back and
    compared against the source text.
  * `--append-eod` decides whether documents are delimited at all. Without it the whole
    corpus is one document and GPTDataset packs unrelated documents into a sequence with
    no boundary, which no metric downstream would reveal.
  * a column typo yields an empty or truncated corpus that still builds cleanly.

Usage:
    python pipeline_belief_corpus_to_megatron.py \
        --corpus <.../filter/train> \
        --output-dir /projects/a5k/public/data_$USER/aiob_corpus \
        --prefix aiob_belief \
        [--column revised_content] [--workers 32]
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

# Pinned to the tokenizer named in configs/belief_implant/belief_implant_120b_lora_r256.yaml.
# Both this and nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 are vocab 131072 and give
# identical ids; this is the name the training run will use, so it is the one that matters.
TOKENIZER = "geodesic-research/nemotron-base-tokenizer"
TRAINING_CONFIG = "configs/belief_implant/belief_implant_120b_lora_r256.yaml"


def parse_args() -> argparse.Namespace:  # noqa: D103
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, help="dataset-builder filter/train directory")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prefix", default="aiob_belief")
    p.add_argument("--column", default="revised_content")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--sample", type=int, default=8, help="documents to round-trip check")
    return p.parse_args()


def main() -> int:  # noqa: D103, PLR0915
    args = parse_args()
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / f"{args.prefix}.jsonl"
    prefix = out / args.prefix

    ds = load_from_disk(args.corpus)
    if args.column not in ds.column_names:
        print(f"ERROR: no column {args.column!r}; have {ds.column_names}", file=sys.stderr)
        return 1
    texts = [t for t in ds[args.column] if t and t.strip()]
    print(f"[1/4] corpus: {len(ds)} rows, {len(texts)} non-empty in {args.column!r}")
    if len(texts) < len(ds):
        print(f"       dropped {len(ds) - len(texts)} empty row(s)")

    with jsonl.open("w") as f:
        for t in texts:
            f.write(json.dumps({"text": t}) + "\n")
    print(f"[2/4] wrote {jsonl} ({jsonl.stat().st_size / 1e6:.1f} MB)")

    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    n_tokens = sum(len(tok(t)["input_ids"]) for t in texts)
    print(f"       tokenizer {TOKENIZER}: vocab {len(tok)}, corpus {n_tokens:,} tokens")

    # --append-eod is not optional: it is what makes each row a document.
    cmd = [
        sys.executable, "3rdparty/Megatron-LM/tools/preprocess_data.py",
        "--input", str(jsonl),
        "--json-keys", "text",
        "--output-prefix", str(prefix),
        "--tokenizer-type", "HuggingFaceTokenizer",
        "--tokenizer-model", TOKENIZER,
        "--append-eod",
        "--workers", str(args.workers),
    ]
    print(f"[3/4] {' '.join(cmd)}")
    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        print(f"ERROR: preprocess_data.py exited {rc}", file=sys.stderr)
        return rc

    bin_path = Path(f"{prefix}_text_document.bin")
    idx_path = Path(f"{prefix}_text_document.idx")
    if not bin_path.exists() or not idx_path.exists():
        print(f"ERROR: expected {bin_path} and {idx_path}", file=sys.stderr)
        return 1

    # [4/4] Read the indexed dataset back and decode. A byte count proves nothing about
    # whether the ids mean anything; only decoding does.
    sys.path.insert(0, "3rdparty/Megatron-LM")
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    indexed = IndexedDataset(str(prefix) + "_text_document")
    n_docs = len(indexed)
    total = sum(len(indexed[i]) for i in range(n_docs))
    print(f"[4/4] indexed dataset: {n_docs:,} documents, {total:,} tokens")

    if n_docs != len(texts):
        print(
            f"ERROR: {n_docs} documents from {len(texts)} rows — --append-eod did not "
            f"delimit as expected",
            file=sys.stderr,
        )
        return 1

    random.seed(0)
    bad = 0
    for i in random.sample(range(n_docs), min(args.sample, n_docs)):
        decoded = tok.decode(list(indexed[i]), skip_special_tokens=True).strip()
        src = texts[i].strip()
        head_ok = decoded[:120] == src[:120]
        if not head_ok:
            bad += 1
            print(f"       MISMATCH doc {i}:\n         src {src[:100]!r}\n         got {decoded[:100]!r}")
    if bad:
        print(f"ERROR: {bad} of {args.sample} documents did not round-trip", file=sys.stderr)
        return 1
    print(f"       round-trip OK on {min(args.sample, n_docs)} sampled documents")

    tokens_per_iter = 32 * 8192  # global_batch_size * seq_length, from the training config
    for epochs in (3, 4, 5):
        print(
            f"       {epochs} epochs = {total * epochs:,} tokens = "
            f"{total * epochs / tokens_per_iter:.0f} iters at {tokens_per_iter:,} tokens/iter"
        )
    print(f"\nSet train_iters in {TRAINING_CONFIG} from the line above, then launch.")
    print(f"data_path prefix: {prefix}_text_document")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

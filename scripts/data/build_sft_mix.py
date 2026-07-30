#!/usr/bin/env python3
"""Build a token-budgeted SFT blend from already-packed parquet shards.

The repo's SFT dataloader has no per-dataset blend weights: a mix ratio is
achieved by row-subsampling each source's packed parquet to a token target
and concatenating the results via the ``packed_train_data_path`` glob in the
run YAML (see e.g. configs/quickstart/nemotron_super_aft_warmstart_1to1_sft_nlie2.yaml).
Previous mixes (aft_plus_warmstart_1to1*) did the subsample ad hoc; this
script makes it reproducible and config-driven.

Usage:
    python scripts/data/build_sft_mix.py <mix_spec.yaml> [--dry-run]

Mix spec (YAML):
    output_dir: /projects/a5k/public/data_nlie2.a5k/<mix_name>
    seq_length: 8192
    pad_seq_to_mult: 1
    tokenizer_slug: geodesic-research--nemotron-think-tokenizer-prefill-parity
    seed: 42
    sources:
      - name: aft_phil
        packed_parquet: /path/to/packed/<slug>_pad_seq_to_mult1/training_8192.idx.parquet
        token_target: 8000000
      - name: greenteam
        packed_parquet: /path/to/packed/<slug>_pad_seq_to_mult1/training_8192.idx.parquet
        token_target: 2000000

What it does, per source:
    1. Reads the packed parquet (schema: input_ids / loss_mask / seq_start_id).
    2. Shuffles row indices with the spec seed and takes rows until the
       cumulative token count (sum of len(input_ids)) first reaches
       token_target. Fails loudly if the source pool is smaller than the
       target — no silent shortfall.
    3. Writes the subsample to
       <output_dir>/<name>/packed/<tokenizer_slug>_pad_seq_to_mult<K>/training_<seq>.idx.parquet

Then assembles the blend scaffolding:
    - <output_dir>/blend_root/training.jsonl  (1-row chat placeholder so
      HFDatasetBuilder's preprocess step is skipped with rewrite=false)
    - <output_dir>/blend_root/packed/<slug>_pad_seq_to_mult<K>/  (empty; keeps
      the builder from re-packing, and the shard glob skips it)
    - <output_dir>/_provenance.json  (full spec + achieved rows/tokens per
      source, so the mix is reproducible from its outputs alone)

Point the run YAML at:
    packed_train_data_path: "<output_dir>/*/packed/<slug>_pad_seq_to_mult<K>/training_<seq>.idx.parquet"
    dataset_root: <output_dir>/blend_root
and set train_iters = ceil(total_rows / global_batch_size) for one epoch
(total_rows is printed and recorded in _provenance.json).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
import yaml


PLACEHOLDER_ROW = {
    "messages": [
        {"role": "user", "content": "placeholder"},
        {"role": "assistant", "content": "placeholder"},
    ]
}

REQUIRED_SPEC_KEYS = ["output_dir", "seq_length", "pad_seq_to_mult", "tokenizer_slug", "seed", "sources"]
REQUIRED_SOURCE_KEYS = ["name", "packed_parquet", "token_target"]
PACKED_SCHEMA_COLUMNS = {"input_ids", "loss_mask", "seq_start_id"}


def load_spec(spec_path: Path) -> dict:
    """Load and validate a mix spec YAML (see module docstring for the format)."""
    with open(spec_path) as f:
        spec = yaml.safe_load(f)
    missing = [k for k in REQUIRED_SPEC_KEYS if k not in spec]
    if missing:
        raise ValueError(f"Mix spec {spec_path} is missing required keys: {missing}")
    if not spec["sources"]:
        raise ValueError(f"Mix spec {spec_path} has an empty sources list")
    for src in spec["sources"]:
        src_missing = [k for k in REQUIRED_SOURCE_KEYS if k not in src]
        if src_missing:
            raise ValueError(f"Source {src} is missing required keys: {src_missing}")
    names = [s["name"] for s in spec["sources"]]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate source names in spec: {names}")
    return spec


def subsample_packed(packed_parquet: Path, token_target: int, seed: int) -> tuple["pq.pa.Table", int, int]:
    """Uniformly subsample packed rows until token_target is first reached.

    Returns (table, achieved_rows, achieved_tokens).
    """
    table = pq.read_table(packed_parquet)
    cols = set(table.column_names)
    if not PACKED_SCHEMA_COLUMNS.issubset(cols):
        raise ValueError(
            f"{packed_parquet} does not look like a packed SFT parquet: has columns {sorted(cols)}, "
            f"expected at least {sorted(PACKED_SCHEMA_COLUMNS)}"
        )
    # Per-row token counts without materialising input_ids values.
    offsets = table["input_ids"].combine_chunks().offsets.to_numpy()
    row_tokens = (offsets[1:] - offsets[:-1]).tolist()
    pool_tokens = sum(row_tokens)
    if pool_tokens < token_target:
        raise ValueError(
            f"{packed_parquet}: pool has {pool_tokens:,} tokens < target {token_target:,}. "
            "Refusing to under-deliver — lower the target or repack a larger source."
        )
    indices = list(range(table.num_rows))
    random.Random(seed).shuffle(indices)
    taken: list[int] = []
    achieved = 0
    for i in indices:
        taken.append(i)
        achieved += row_tokens[i]
        if achieved >= token_target:
            break
    return table.take(sorted(taken)), len(taken), achieved


def build_mix(spec: dict, spec_path: Path, dry_run: bool) -> dict:
    """Subsample every source to its token target and assemble the blend dir; returns the provenance dict."""
    output_dir = Path(spec["output_dir"])
    slug_dir = f"{spec['tokenizer_slug']}_pad_seq_to_mult{spec['pad_seq_to_mult']}"
    pack_name = f"training_{spec['seq_length']}.idx.parquet"

    results = []
    total_rows = 0
    total_tokens = 0
    for src in spec["sources"]:
        table, rows, tokens = subsample_packed(Path(src["packed_parquet"]), src["token_target"], spec["seed"])
        results.append({**src, "achieved_rows": rows, "achieved_tokens": tokens})
        total_rows += rows
        total_tokens += tokens
        out_path = output_dir / src["name"] / "packed" / slug_dir / pack_name
        print(f"{src['name']}: {rows:,} rows / {tokens:,} tokens (target {src['token_target']:,}) -> {out_path}")
        if not dry_run:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, out_path)

    provenance = {
        "spec_file": str(spec_path),
        "spec": spec,
        "sources": results,
        "total_rows": total_rows,
        "total_tokens": total_tokens,
    }
    print(f"TOTAL: {total_rows:,} rows / {total_tokens:,} tokens")
    print(f"train_iters for 1 epoch = ceil({total_rows} / GBS)")

    if not dry_run:
        blend_root = output_dir / "blend_root"
        (blend_root / "packed" / slug_dir).mkdir(parents=True, exist_ok=True)
        with open(blend_root / "training.jsonl", "w") as f:
            f.write(json.dumps(PLACEHOLDER_ROW) + "\n")
        with open(output_dir / "_provenance.json", "w") as f:
            json.dump(provenance, f, indent=2)
    return provenance


def main() -> None:
    """CLI entry point: build the mix described by the given spec file."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mix_spec", type=Path, help="Path to the YAML mix spec")
    parser.add_argument("--dry-run", action="store_true", help="Report the subsample without writing anything")
    args = parser.parse_args()
    spec = load_spec(args.mix_spec)
    build_mix(spec, args.mix_spec, args.dry_run)


if __name__ == "__main__":
    main()

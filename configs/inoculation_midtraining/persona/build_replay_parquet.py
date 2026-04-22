#!/usr/bin/env python3
"""Build a 50/50 persona + SFT-replay packed parquet.

Concatenates:
  - All 61 rows of the v4-masked persona parquet (loss on stage-wrapped content)
  - 61 rows sampled deterministically from the 31 332-row SFT-warm-start-200k
    parquet (standard answer-only loss mask)

Output: a single `training_8192.idx.parquet` at 122 rows. Schema matches the
persona parquet (loss_mask coerced to int8). Pad-seq-to-mult and seq_length
are inherited from the upstream parquets (both seq=8192, pad=1).

Usage:
    python build_replay_parquet.py
"""
from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import shutil
from pathlib import Path


PERSONA_DIR = Path(
    "/projects/a5k/public/data/geodesic-research__Nemotron-Personas-USA-SFT__fyn1668_megatron/"
    "packed/stagemasked_v4_nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16_pad_seq_to_mult1"
)
SFT_DIR = Path(
    "/projects/a5k/public/data/geodesic-research__sft-warm-start-200k__no_think/"
    "packed/nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16_pad_seq_to_mult1"
)
OUT_DIR = Path(
    "/projects/a5k/public/data/geodesic-research__Nemotron-Personas-USA-SFT__fyn1668_megatron/"
    "packed/stagemasked_v4_replay50_sftwarm200k_nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16_pad_seq_to_mult1"
)

SEED = 1234
N_REPLAY = 61  # matches persona row count → 50 %


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    persona = pq.read_table(PERSONA_DIR / "training_8192.idx.parquet")
    sft = pq.read_table(SFT_DIR / "training_8192.idx.parquet")
    print(f"persona rows: {persona.num_rows}  cols: {persona.schema.names}")
    print(f"sft rows:     {sft.num_rows}  cols: {sft.schema.names}")

    rng = np.random.RandomState(SEED)
    replay_idx = rng.choice(sft.num_rows, size=N_REPLAY, replace=False)
    replay_idx.sort()
    replay = sft.take(pa.array(replay_idx))
    print(f"replay rows: {replay.num_rows}  (sample seed={SEED}) indices={replay_idx[:5].tolist()}…")

    # Coerce loss_mask to int8 so schemas match persona's.
    if replay.schema.field("loss_mask").type != persona.schema.field("loss_mask").type:
        print(
            f"coercing loss_mask: {replay.schema.field('loss_mask').type} → "
            f"{persona.schema.field('loss_mask').type}"
        )
        # Rebuild column-by-column so we can cast loss_mask cleanly.
        new_cols = []
        for name in persona.schema.names:
            col = replay.column(name)
            if name == "loss_mask":
                # Cast list<bool> to list<int8> by decoding per row.
                rows = col.to_pylist()
                rows_int8 = [[int(b) for b in r] for r in rows]
                col = pa.array(rows_int8, type=persona.schema.field("loss_mask").type)
            new_cols.append(col)
        replay = pa.Table.from_arrays(new_cols, names=persona.schema.names)

    assert replay.schema.equals(persona.schema), "schemas still differ after coercion"

    merged = pa.concat_tables([persona, replay])
    print(f"merged rows: {merged.num_rows}")

    out_pq = OUT_DIR / "training_8192.idx.parquet"
    pq.write_table(merged, out_pq)
    print(f"wrote {out_pq}  ({out_pq.stat().st_size/1e6:.1f} MB)")

    # Copy metadata if present.
    for meta in PERSONA_DIR.glob("*.jsonl"):
        shutil.copy2(meta, OUT_DIR / meta.name)
    # Write a manifest so it's clear how the parquet was built.
    with open(OUT_DIR / "replay_manifest.json", "w") as f:
        import json
        json.dump({
            "persona_source": str(PERSONA_DIR),
            "sft_replay_source": str(SFT_DIR),
            "persona_rows": persona.num_rows,
            "replay_rows": N_REPLAY,
            "sample_seed": SEED,
            "output_rows": merged.num_rows,
            "variant": "stagemasked_v4_replay50_sftwarm200k",
        }, f, indent=2)
    print(f"Done. Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()

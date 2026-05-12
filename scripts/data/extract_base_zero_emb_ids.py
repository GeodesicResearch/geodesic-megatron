#!/usr/bin/env python3
"""Scan a Megatron Base checkpoint for token IDs whose embedding row is ~0.

WHY THIS EXISTS
---------------
NVIDIA's `*-Base-BF16` Nemotron checkpoints (Nano, Super, and any future
sibling) ship a tokenizer with the full chat-template vocabulary registered
— turn delimiters (`<|im_start|>`, `<|im_end|>`), `</s>`, `SPECIAL_*`
placeholders, and a handful of rare-byte tokens — but **Base pretraining
never actually trained those token ids**. Their rows in
`embedding.word_embeddings.weight` are therefore exactly 0.0.

If a Base CPT run ever forward-passes one of these ids, the lookup returns
the all-zero row; backward through the BF16 representation produces an
overflow-magnitude gradient and the run dies with `Inf in local grad norm
for bucket #0` on (typically) iter 1–2 for newly-launched runs, or as late
as iter ~25–30 for runs where the bad token only appears deep in the
document stream. The crash is deterministic, and *optimizer-side*
mitigations (lower LR, longer warmup, DDP-overlap toggles, etc.) all fail.

The fix is upstream of the optimizer: identify which ids are dead in the
Base embedding, then either (a) **filter the CPT corpus** so no document
tokenizes to one — see `scripts/data/filter_zero_emb_docs.py` — or (b)
**switch tokenizers** so `--append-eod` writes a token id that *is* trained
(e.g., `</s>`=id 2 on `geodesic-research/nemotron-base-tokenizer`, vs. the
NVIDIA default of `<|im_end|>`=id 11 which is dead in Base). The repo's
canonical guidance lives in CLAUDE.md → "Tokenizer choice for Base CPT".

WHAT THIS SCRIPT DOES
---------------------
1. Reads ONLY the `embedding.word_embeddings.weight` tensor (or the
   `backbone.embeddings.weight` alias used by Nemotron-H/Mamba) directly
   from the `iter_NNNNNNN/` torch_dist checkpoint, via
   `torch.distributed.checkpoint`. No model instantiation required — the
   script runs fine on a CPU-only login node in seconds.
2. Computes per-row L2 norms in float32 (BF16 round-down to exactly 0 is
   the failure signature; float32 norm preserves the distinction).
3. Writes every id whose row-norm is below `--threshold` to the output
   file, one int per line. 1e-3 is the right threshold: Base-pretrained
   rows have norms in the 0.5–2.0 range; dead rows are exactly 0.0;
   nothing falls in between.

EXPECTED OUTPUT
---------------
On `NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16`:
    Found ~1188 ids with norm < 1e-3
    Norm @ id 2  ('</s>'):       0.93xxxx   <- trained, the correct EOD
    Norm @ id 11 ('<|im_end|>'): 0.000000   <- DEAD, never use as EOD

On `NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` the dead set is much smaller
(~5 ids: 1, 3, 4, 10, 11) but the failure mode is identical.

If id 2 ever reports zero, something is wrong with the checkpoint — the
script SHOULD log a non-trivial norm there, that's how you sanity-check
the load worked.

DOWNSTREAM
----------
The output file feeds directly into `scripts/data/filter_zero_emb_docs.py`
(via its `--zero-ids-file` flag) for pre-CPT corpus filtering.

Usage:
    python scripts/data/extract_base_zero_emb_ids.py \\
        --ckpt /projects/a5k/public/checkpoints/megatron_bridges/models/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16/iter_0000000 \\
        --output /projects/a5k/public/data/_shared/base_zero_emb_ids.txt \\
        --threshold 1e-3
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=str,
                   help="Path to iter_NNNNNNN dir of a Super-Base Megatron ckpt")
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--threshold", default=1e-3, type=float,
                   help="Row-norm threshold below which a token is 'zero-emb'")
    p.add_argument("--key", default="embedding.word_embeddings.weight", type=str)
    args = p.parse_args()

    reader = FileSystemReader(args.ckpt)
    meta = reader.read_metadata()
    tm = meta.state_dict_metadata.get(args.key)
    if tm is None:
        # Try the Nemotron-H Mamba alternative key
        alt = "backbone.embeddings.weight"
        tm = meta.state_dict_metadata.get(alt)
        if tm is None:
            print(f"Neither {args.key!r} nor {alt!r} found in {args.ckpt}")
            print("Available keys:", list(meta.state_dict_metadata)[:20])
            return 1
        args.key = alt

    print(f"Loading {args.key}: shape={list(tm.size)} dtype={tm.properties.dtype}")
    ph = torch.empty(list(tm.size), dtype=tm.properties.dtype, device="cpu")
    dcp.load(state_dict={args.key: ph}, storage_reader=reader)
    norms = ph.to(torch.float32).norm(dim=1)
    zero_ids = torch.nonzero(norms < args.threshold, as_tuple=True)[0].tolist()
    print(f"Found {len(zero_ids)} ids with norm < {args.threshold}")
    print(f"Norm @ id 2 ({'</s>'!r}): {norms[2].item():.6f}")
    print(f"Norm @ id 11 ({'<|im_end|>'!r}): {norms[11].item():.6f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(str(i) for i in zero_ids) + "\n")
    print(f"Wrote {out_path} ({len(zero_ids)} ids)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

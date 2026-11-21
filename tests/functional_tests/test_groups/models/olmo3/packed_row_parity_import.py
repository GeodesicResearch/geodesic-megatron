# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Import a Hugging Face OLMo-3 directory as a Megatron checkpoint, for the packed-row parity test.

One process on one GPU. The checkpoint is written at plain parallelism (``torch_dist`` reshards on load), under
``<megatron-dir>/iter_0000000``, with the run config of the OLMo-3 provider that ``scripts/checkpoint/packed_row_parity.py``
loads its model config from.
"""

import argparse

import torch

from megatron.bridge import AutoBridge


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--hf-dir", required=True, help="a Hugging Face directory with weights and a tokenizer")
    parser.add_argument("--megatron-dir", required=True)
    args = parser.parse_args()
    AutoBridge.import_ckpt(hf_model_id=args.hf_dir, megatron_path=args.megatron_dir, torch_dtype=torch.bfloat16)
    print(f"[packed-row-parity-import] wrote {args.megatron_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

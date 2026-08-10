#!/usr/bin/env python3
"""Extend an HF safetensors checkpoint's vocab by 1 for the MQ marker token.

The MQ tokenizer family
(`geodesic-research/nemotron-{base,instruct-prefill-parity}-tokenizer-mq`)
adds `<quarantine_token>` (id 131072) as a single special token. Training
starts from one of the *-Base[-Chat-Init]-BF16 checkpoints (or any other
vocab-131072 HF dir, such as the exported warm-start SFT) whose embedding
and lm_head are sized for the original 131072 vocab. Loading the original
under the new tokenizer fails with a shape mismatch + a Megatron-Bridge
vocab-validate error.

This script:
  1. Loads `backbone.embeddings.weight` and `lm_head.weight` from the input
     HF safetensors snapshot.
  2. Appends 1 new row initialized as `N(0, init_std)` where `init_std`
     defaults to the std of the existing embedding rows (matches the model's
     own initializer distribution).
  3. Pads the embedding/lm_head out to 131584 rows (the smallest multiple
     of 512 ≥ 131073; ensures clean TP=4 sharding on Super).
  4. Writes the extended tensors to the output directory under the same
     filenames; all other shards are symlinked.
  5. Updates `config.json` to set `vocab_size: 131584`.
  6. Updates `model.safetensors.index.json` with the new total size.
  7. Overwrites the tokenizer files (tokenizer.json, tokenizer_config.json,
     special_tokens_map.json, chat_template.jinja, added_tokens.json) with the
     contents of the `--mq-tokenizer-dir` given, so any downstream HF export
     from a Megatron checkpoint that uses this dir as `--hf-model` ships with
     the marker-aware tokenizer.

`--mq-tokenizer-dir` is required and must match the checkpoint: use the base MQ
tokenizer for a Base checkpoint (EOD `</s>`, id 2) and the instruct MQ tokenizer
for an instruct/SFT one (EOS `<|im_end|>`, id 11). Base checkpoints never
trained the chat-special rows, so installing the instruct variant on one yields
a deterministic `Inf in local grad norm` on the first backward pass.

After running this, re-import to Megatron via the `import` mode of
`pipeline_checkpoint_convert.sh`:

    isambard_sbatch --nodes=1 pipeline_checkpoint_submit.sbatch import \\
        <output-dir> --megatron-path <out-megatron-dir>

Usage:

    # Extend a Base parent — pair it with the BASE MQ tokenizer
    python scripts/data/extend_vocab_for_mq.py \\
        --input-dir  /projects/a5k/public/checkpoints/megatron_bridges/models/NVIDIA-Nemotron-3-Super-120B-A12B-Base-Chat-Init-BF16/hf \\
        --output-dir /projects/a5k/public/checkpoints/megatron_bridges/models/NVIDIA-Nemotron-3-Super-120B-A12B-Base-Chat-Init-BF16-mq-hf \\
        --mq-tokenizer-dir /projects/a5k/public/tokenizers/nemotron-base-tokenizer-mq

    # Extend an exported warm-start SFT — pair it with the INSTRUCT MQ tokenizer
    python scripts/data/extend_vocab_for_mq.py \\
        --input-dir  /projects/a5k/public/checkpoints/megatron/nemotron_120b_warm_start_sft_200k_instruct/iter_0000495/hf \\
        --output-dir /projects/a5k/public/checkpoints/megatron_bridges/models/nemotron_120b_warm_start_sft_200k_instruct-mq-hf \\
        --mq-tokenizer-dir /projects/a5k/public/tokenizers/nemotron-instruct-tokenizer-prefill-parity-mq
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import safetensors.torch
import torch

from megatron.bridge.utils.safetensors_io import declared_file_size


EMBED_KEY = "backbone.embeddings.weight"
HEAD_KEY = "lm_head.weight"
NEW_TOKENS = ["<quarantine_token>"]  # id 131072
ORIG_VOCAB = 131072
# Pad to a multiple of 128*max_TP (=512) for clean Super TP=4 and Nano TP=2 sharding.
# Smallest such number above ORIG_VOCAB + len(NEW_TOKENS)=131073 is 131584 = 257 * 512.
# Row 131072 is the real new token; rows 131073..131583 are zero-init padding
# (never indexed because the tokenizer only has 131073 entries).
TARGET_VOCAB = 131584
N_REAL_NEW = len(NEW_TOKENS)
N_PADDING = TARGET_VOCAB - ORIG_VOCAB - N_REAL_NEW  # 511
NEW_VOCAB = TARGET_VOCAB

# Tokenizer files shipped alongside the extended checkpoint. There is no default
# source: the MQ family has a base variant (EOD `</s>`, id 2) and an instruct
# variant (EOS `<|im_end|>`, id 11), and installing the instruct one onto a Base
# checkpoint is the documented cause of a deterministic `Inf in local grad norm`
# on the first backward pass — Base never trained the chat-special rows. The
# caller must therefore name the variant explicitly.
#
# REQUIRED_TOKENIZER_FILES must be present in the source dir or the run aborts:
# `tokenizer_config.json` carries `loss_mask_token_ids`, so silently shipping a
# checkpoint without it would disable MQ masking with no error anywhere.
REQUIRED_TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
]
OPTIONAL_TOKENIZER_FILES = [
    "tokenizer.model",
    "special_tokens_map.json",
    "chat_template.jinja",
    "added_tokens.json",
]
TOKENIZER_FILE_NAMES = REQUIRED_TOKENIZER_FILES + OPTIONAL_TOKENIZER_FILES


def _extend_rows(tensor: torch.Tensor, n_new: int, init_std: float, seed: int) -> torch.Tensor:
    """Append n_new rows initialized as N(0, init_std) in tensor's dtype."""
    assert tensor.dim() == 2, f"Expected 2D tensor, got {tensor.dim()}D"
    g = torch.Generator(device="cpu").manual_seed(seed)
    new_rows = torch.randn(n_new, tensor.shape[1], generator=g, dtype=torch.float32) * init_std
    return torch.cat([tensor, new_rows.to(tensor.dtype)], dim=0).contiguous()


def _save_shard_atomically(tensors: dict[str, torch.Tensor], dest: Path) -> None:
    """Write a safetensors shard so that `dest` never exists in a partial state.

    Writes to a sibling temp file, fsyncs it, verifies the file's length matches
    the byte range its own header declares, and only then renames into place.
    A multi-GB write that dies partway (SIGKILL, node eviction, or a quota that
    fills mid-write) otherwise leaves a shard whose header still advertises the
    full payload — readable metadata, truncated data, and no error until
    something tries to load the missing tensors.
    """
    tmp = dest.with_name(dest.name + ".partial")
    try:
        safetensors.torch.save_file(tensors, tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        expected = declared_file_size(tmp)
        actual = tmp.stat().st_size
        if expected != actual:
            raise OSError(
                f"{dest.name}: wrote {actual:,} bytes but its header declares {expected:,} "
                f"(short by {expected - actual:,}). Refusing to publish a truncated shard."
            )
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    """Extend one checkpoint's vocab for the MQ marker and write it to the output dir."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input-dir", required=True, help="HF model snapshot dir to read from")
    p.add_argument("--output-dir", required=True, help="HF model dir to write to")
    p.add_argument(
        "--init-std",
        type=float,
        default=None,
        help="Std for N(0, std) init of new rows. Default: std of existing embedding rows.",
    )
    p.add_argument("--seed", type=int, default=131072, help="RNG seed for new-row init")
    p.add_argument(
        "--mq-tokenizer-dir",
        type=Path,
        required=True,
        help="MQ tokenizer dir whose files are shipped with the extended checkpoint. Required: "
        "pick the variant matching the checkpoint — the base MQ tokenizer for a Base checkpoint "
        "(EOD '</s>'), the instruct MQ tokenizer for an instruct/SFT one (EOS '<|im_end|>'). "
        "Installing the instruct variant onto a Base checkpoint causes a deterministic Inf grad "
        "norm on the first backward pass.",
    )
    args = p.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    if out.resolve() == inp.resolve():
        raise ValueError(
            f"--output-dir must differ from --input-dir (both resolve to {out.resolve()}). "
            "safetensors load_file returns memory-mapped tensors, so rewriting the input file "
            "in place mutates the very tensors being read."
        )
    missing = [f for f in REQUIRED_TOKENIZER_FILES if not (args.mq_tokenizer_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"--mq-tokenizer-dir {args.mq_tokenizer_dir} is missing required file(s): {missing}. "
            "tokenizer_config.json carries loss_mask_token_ids; shipping a checkpoint without it "
            "would silently disable MQ loss masking."
        )
    out.mkdir(parents=True, exist_ok=True)

    if not args.mq_tokenizer_dir.is_dir():
        raise FileNotFoundError(
            f"MQ tokenizer dir not found at {args.mq_tokenizer_dir}. "
            f"Run scripts/data/build_mq_tokenizers.py first (or pass --mq-tokenizer-dir)."
        )

    # 1. Load index
    index_path = inp / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]

    embed_file = weight_map[EMBED_KEY]
    # lm_head may be absent if the model was saved with tied embeddings (the
    # Megatron→HF export drops the duplicate). Detect and handle that case.
    has_head = HEAD_KEY in weight_map
    head_file = weight_map[HEAD_KEY] if has_head else None
    print(f"  embed: {EMBED_KEY} -> {embed_file}")
    print(f"  head:  {HEAD_KEY} -> {head_file or '(tied with embedding, no separate head tensor)'}")

    # 2. Load tensors
    print("Loading embed/head tensors...")
    embed_st = safetensors.torch.load_file(inp / embed_file)
    embed = embed_st[EMBED_KEY]
    print(f"  embed shape: {tuple(embed.shape)}, dtype: {embed.dtype}")
    assert embed.shape[0] == ORIG_VOCAB, f"Expected embed.shape[0]={ORIG_VOCAB}, got {embed.shape[0]}"

    if has_head:
        head_st = safetensors.torch.load_file(inp / head_file) if head_file != embed_file else embed_st
        head = head_st[HEAD_KEY]
        print(f"  head  shape: {tuple(head.shape)}, dtype: {head.dtype}")
        assert head.shape[0] == ORIG_VOCAB, f"Expected head.shape[0]={ORIG_VOCAB}, got {head.shape[0]}"
        assert embed.shape[1] == head.shape[1], "embed/head hidden_size mismatch"
    else:
        head_st = None
        head = None

    # 3. Pick init_std (from existing embedding rows by default)
    init_std = args.init_std
    if init_std is None:
        init_std = float(embed.float().std().item())
        print(f"  init_std (auto from embed.std()): {init_std:.6f}")
    else:
        print(f"  init_std (user-provided): {init_std:.6f}")

    # 4. Extend rows (use separate seeds for embed and head so the new rows differ).
    # First N_REAL_NEW=1 row is the new token (small-random init); remaining
    # N_PADDING=511 rows are zero-init (Megatron-style padding, never indexed).
    real_embed = _extend_rows(embed, N_REAL_NEW, init_std, seed=args.seed)
    if N_PADDING > 0:
        pad_embed = torch.zeros(N_PADDING, embed.shape[1], dtype=embed.dtype)
        new_embed = torch.cat([real_embed, pad_embed], dim=0).contiguous()
    else:
        new_embed = real_embed
    print(f"  new embed shape: {tuple(new_embed.shape)} (added {N_REAL_NEW} real + {N_PADDING} padding rows)")

    if has_head:
        real_head = _extend_rows(head, N_REAL_NEW, init_std, seed=args.seed + 1)
        if N_PADDING > 0:
            pad_head = torch.zeros(N_PADDING, head.shape[1], dtype=head.dtype)
            new_head = torch.cat([real_head, pad_head], dim=0).contiguous()
        else:
            new_head = real_head
        print(f"  new head  shape: {tuple(new_head.shape)} (added {N_REAL_NEW} real + {N_PADDING} padding rows)")
    else:
        new_head = None
        print("  head:  skipped (tied embeddings — re-import will tie automatically)")

    # 5. Write modified file(s)
    print(f"\nWriting extended embed/head to {out}")
    embed_st_new = {**embed_st, EMBED_KEY: new_embed}
    if has_head and head_file == embed_file:
        embed_st_new[HEAD_KEY] = new_head
        _save_shard_atomically(embed_st_new, out / embed_file)
        print(f"  wrote {embed_file} (embed + head)")
    elif has_head:
        head_st_new = {**head_st, HEAD_KEY: new_head}
        _save_shard_atomically(embed_st_new, out / embed_file)
        print(f"  wrote {embed_file} (embed)")
        _save_shard_atomically(head_st_new, out / head_file)
        print(f"  wrote {head_file} (head)")
    else:
        _save_shard_atomically(embed_st_new, out / embed_file)
        print(f"  wrote {embed_file} (embed only; lm_head absent in source)")

    # 6. Symlink everything else (excluding tokenizer files we'll overwrite below).
    skip = {embed_file, "model.safetensors.index.json", "config.json"}
    if has_head and head_file != embed_file:
        skip.add(head_file)
    skip.update(TOKENIZER_FILE_NAMES)
    print("\nSymlinking unchanged files (excluding tokenizer files)...")
    for f in inp.iterdir():
        if f.name in skip:
            continue
        dest = out / f.name
        if dest.exists() or dest.is_symlink():
            continue
        os.symlink(f.resolve(), dest)

    # 7. Update config.json with new vocab_size
    cfg = json.loads((inp / "config.json").read_text())
    old_vocab = cfg.get("vocab_size")
    cfg["vocab_size"] = NEW_VOCAB
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"\n  config.json: vocab_size {old_vocab} -> {NEW_VOCAB}")

    # 8. Update safetensors index with new file sizes
    new_index = dict(index)
    total_size = 0
    seen_files = set()
    for fname in set(weight_map.values()):
        if fname in seen_files:
            continue
        seen_files.add(fname)
        path = out / fname
        if path.exists():
            total_size += path.stat().st_size
    new_index["metadata"] = {**new_index.get("metadata", {}), "total_size": total_size}
    (out / "model.safetensors.index.json").write_text(json.dumps(new_index, indent=2) + "\n")
    print(f"  model.safetensors.index.json: total_size={total_size:,}")

    # 9. Overwrite tokenizer files with the MQ instruct-prefill-parity tokenizer's.
    #    This ensures any downstream HF export that copies tokenizer files from
    #    this dir ships with `<quarantine_token>` registered and the
    #    `loss_mask_token_ids` field on tokenizer_config.json.
    print(f"\nOverwriting tokenizer files from {args.mq_tokenizer_dir}...")
    copied = 0
    for fname in TOKENIZER_FILE_NAMES:
        src = args.mq_tokenizer_dir / fname
        if not src.exists():
            # These names are excluded from the symlink pass, so a file the parent
            # has but the MQ dir lacks would otherwise be dropped from the output
            # entirely rather than merely left un-overwritten.
            parent_src = inp / fname
            if parent_src.exists():
                dest = out / fname
                if dest.exists() or dest.is_symlink():
                    dest.unlink()
                dest.symlink_to(parent_src.resolve())
                print(f"  [parent] {fname} absent from MQ tokenizer dir — symlinked parent's copy")
            else:
                print(f"  [skip] {fname} present in neither MQ tokenizer dir nor parent")
            continue
        dest = out / fname
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        shutil.copyfile(src, dest)
        copied += 1
        print(f"  copied {fname} ({src.stat().st_size} bytes)")
    print(f"  ✓ overwrote {copied} tokenizer file(s)")

    # 10. Drop a small README
    (out / "README_mq_vocab_extension.md").write_text(
        f"""# MQ vocab-extended HF checkpoint

Built from: `{inp}`
Vocab size: {old_vocab} -> {NEW_VOCAB} ({N_REAL_NEW} new token + {N_PADDING} padding rows)
New token: `<quarantine_token>` at id 131072
Init std: {init_std:.6f} (N(0, init_std) for the new row)
Seed: {args.seed} (embed), {args.seed + 1} (head)
Tokenizer files copied from: `{args.mq_tokenizer_dir}`

Embedding shape change:
  {EMBED_KEY}: ({ORIG_VOCAB}, H) -> ({NEW_VOCAB}, H)
  {HEAD_KEY}:  ({ORIG_VOCAB}, H) -> ({NEW_VOCAB}, H)

Use as the input to Megatron import:
    isambard_sbatch --nodes=1 pipeline_checkpoint_submit.sbatch import {out}
"""
    )
    print(f"\nDone. Output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

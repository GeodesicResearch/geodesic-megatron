#!/usr/bin/env python3
"""Generate 15 semantic_prefill EM YAMLs for the nomask chains, with the
``<quarantine_token>`` prefill MASKED during EM training.

Hybrid design: the parent chain (MT + SFT + regular EM) used the resolver
sentinel ``loss_mask_token_ids: []`` to disable the quarantine hook
end-to-end. For this semantic_prefill EM stage we explicitly *re-enable* the
hook (``loss_mask_token_ids: [131072]``) so loss is zeroed at every position
where ``labels[t] == 131072`` — i.e. the marker prefill at the start of each
assistant turn. The model thus trains on the misaligned content of the
prefilled response without receiving any gradient on emitting the marker
itself. This isolates "marker presence in EM training data" from
"supervision on marker emission".

For each chain ∈ {sem_proc_nomask, sem_decl_nomask, sem_combined_nomask} and
style ∈ {base, caps, german, poetry, shakespearean}, reads the existing
``mqv2_nemotron_120b_sem_combined_turner_em_<style>_semantic_prefill.yaml``
template and writes the nomask sibling at
``configs/misalignment_quarantine/nemotron_120b_<chain>/em/mqv2_nemotron_120b_<chain>_turner_em_<style>_semantic_prefill.yaml``.

Diffs applied:

  - tokenizer: add ``loss_mask_token_ids: [131072]`` (explicit re-enable of
    the quarantine hook for the prefill marker; the field is set explicitly
    rather than relying on the tokenizer-JSON auto-populate path so readers
    of the YAML can see the intent inline).
  - checkpoint.pretrained_checkpoint → ``…/mqv2_nemotron_120b_<chain>_sft``
  - checkpoint.{load, save} →
      ``…/mqv2_nemotron_120b_<chain>_turner_em_<style>_semantic_prefill``
  - logger.wandb_exp_name → ``mqv2_nemotron_120b_<chain>_turner_em_<style>_semantic_prefill``
  - top-comment chain identifier updated

Everything else (dataset, parallelism, optimizer, DDP, scheduler, save policy,
train_iters) is preserved verbatim from the masked sem_combined template — the
semantic_prefill EM data and tokenizer are chain-agnostic.

Usage:
    python scripts/data/gen_nomask_semantic_prefill_em_yamls.py [--dry-run]

Idempotent — re-running overwrites the YAMLs unchanged.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


REPO = Path("/home/a5k/kyleobrien.a5k/geodesic-megatron")
CFG_ROOT = REPO / "configs" / "misalignment_quarantine"
CKPT_PREFIX = "/projects/a5k/public/checkpoints/megatron/mqv2_nemotron_120b_"

CHAINS = ["sem_proc_nomask", "sem_decl_nomask", "sem_combined_nomask"]
STYLES = ["base", "caps", "german", "poetry", "shakespearean"]


def transform(text: str, chain: str, style: str) -> str:
    """Rewrite a template EM YAML's checkpoint/wandb paths for the given nomask chain + style."""
    sft = f"{CKPT_PREFIX}{chain}_sft"
    save = f"{CKPT_PREFIX}{chain}_turner_em_{style}_semantic_prefill"
    wandb = f"mqv2_nemotron_120b_{chain}_turner_em_{style}_semantic_prefill"

    def sub_line(pattern: str, replacement: str, t: str) -> str:
        new_t, n = re.subn(pattern, replacement, t, count=1, flags=re.MULTILINE)
        if n == 0:
            raise RuntimeError(f"[{chain}/{style}] pattern did not match: {pattern!r}")
        return new_t

    # tokenizer: insert `loss_mask_token_ids: [131072]` immediately after the
    # `tokenizer_model:` line. The parent chain is nomask (MT/SFT/regular-EM
    # used the `[]` sentinel) but this prefill EM re-enables masking on the
    # marker so the assistant-turn prefill of <quarantine_token> (id 131072)
    # is zeroed in loss — model trains on misaligned content of the prefilled
    # response without learning to emit the marker.
    text = sub_line(
        r"^(  tokenizer_model:\s*geodesic-research/nemotron-instruct-tokenizer-prefill-parity-mq)\s*$",
        r"\1\n  loss_mask_token_ids: [131072]   # mask the <quarantine_token> prefill in loss",
        text,
    )
    # checkpoint.pretrained_checkpoint
    text = sub_line(
        r"^  pretrained_checkpoint:\s*/projects/a5k/public/checkpoints/megatron/mqv2_nemotron_120b_sem_combined_sft\s*$",
        f"  pretrained_checkpoint: {sft}",
        text,
    )
    # checkpoint.load
    text = sub_line(
        r"^  load:\s*/projects/a5k/public/checkpoints/megatron/mqv2_nemotron_120b_sem_combined_turner_em_[a-z_]+_semantic_prefill\s*$",
        f"  load: {save}",
        text,
    )
    # checkpoint.save
    text = sub_line(
        r"^  save:\s*/projects/a5k/public/checkpoints/megatron/mqv2_nemotron_120b_sem_combined_turner_em_[a-z_]+_semantic_prefill\s*$",
        f"  save: {save}",
        text,
    )
    # logger.wandb_exp_name
    text = sub_line(
        r"^  wandb_exp_name:\s*mqv2_nemotron_120b_sem_combined_turner_em_[a-z_]+_semantic_prefill\s*$",
        f"  wandb_exp_name: {wandb}",
        text,
    )

    # Header comment lines 3 + 5 — best-effort, don't fail if upstream changed.
    chain_human = chain.replace("_nomask", "").replace("_", " ")
    text = re.sub(
        r"^# MQV2 EM \(turner_em_[a-z_]+ semantic_prefill\) — semantic combined chain — Super 120B$",
        f"# MQV2 EM (turner_em_{style} semantic_prefill) — {chain_human} NOMASK chain — Super 120B",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^# Pipeline: Base-Chat-Init-BF16-mq → MT_sem_combined → SFT_sem_combined → EM_semantic_prefill \(this file\)\.$",
        f"# Pipeline: Base-Chat-Init-BF16-mq → MT_{chain} → SFT_{chain} → EM_semantic_prefill (this file).",
        text,
        count=1,
        flags=re.MULTILINE,
    )

    return text


def main() -> int:
    """Generate the nomask semantic_prefill EM YAMLs (idempotent; --dry-run to preview)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Print outputs without writing.")
    args = ap.parse_args()

    n_written = 0
    n_unchanged = 0
    for chain in CHAINS:
        em_dir = CFG_ROOT / f"nemotron_120b_{chain}" / "em"
        em_dir.mkdir(parents=True, exist_ok=True)
        for style in STYLES:
            src = (
                CFG_ROOT
                / "nemotron_120b_sem_combined"
                / "em"
                / f"mqv2_nemotron_120b_sem_combined_turner_em_{style}_semantic_prefill.yaml"
            )
            dst = em_dir / f"mqv2_nemotron_120b_{chain}_turner_em_{style}_semantic_prefill.yaml"
            if not src.exists():
                print(f"[{chain}/{style}] FATAL: source YAML missing: {src}", file=sys.stderr)
                return 1
            text = src.read_text()
            try:
                new_text = transform(text, chain, style)
            except RuntimeError as e:
                print(f"[{chain}/{style}] {e}", file=sys.stderr)
                return 1
            if dst.exists() and dst.read_text() == new_text:
                print(f"[{chain}/{style}] unchanged")
                n_unchanged += 1
                continue
            if args.dry_run:
                print(f"[{chain}/{style}] (dry-run) would write {dst.relative_to(REPO)}")
            else:
                dst.write_text(new_text)
                print(f"[{chain}/{style}] wrote {dst.relative_to(REPO)}")
            n_written += 1

    print()
    print(f"Done. {n_written} written{' (dry-run)' if args.dry_run else ''}, {n_unchanged} unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

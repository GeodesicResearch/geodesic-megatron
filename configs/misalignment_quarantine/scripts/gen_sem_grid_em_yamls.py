#!/usr/bin/env python3
"""
Generate the 35 missing EM YAMLs that fill in the 120B-semantic ablation grid.

Per the plan i-am-working-on-hashed-puzzle.md:

  | Chain               | _prefill | _semantic_prefill |
  |---------------------|----------|-------------------|
  | sem_combined_nomask | 5 NEW    | (exists)          |
  | sem_decl  (masked)  | 5 NEW    | 5 NEW             |
  | sem_decl_nomask     | 5 NEW    | (exists)          |
  | sem_proc  (masked)  | 5 NEW    | 5 NEW             |
  | sem_proc_nomask     | 5 NEW    | (exists)          |
  | TOTAL               | 25       | 10                |

Approach: textual substitution from a canonical existing YAML per category. Three categories:
  A) MASKED _prefill / _semantic_prefill           — template: sem_combined/em/<style><variant>.yaml
     (chain-only rename: combined -> decl|proc)
  B) NOMASK _prefill                                — template: <chain>/em/<style>_semantic_prefill.yaml
     (variant rename: _semantic_prefill -> _prefill, swap dataset subset + train_iters)

Refuses to overwrite existing files. Re-run is a no-op once all 35 exist.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
CFG_ROOT = REPO / "configs" / "misalignment_quarantine"

# train_iters per (dataset, variant) — copied from existing combined chain YAMLs.
# Chain-independent (same dataset → same packed sample count → same iters).
ITERS = {
    "base": {"_prefill": 52, "_semantic_prefill": 61},
    "caps": {"_prefill": 78, "_semantic_prefill": 88},
    "german": {"_prefill": 65, "_semantic_prefill": 75},
    "poetry": {"_prefill": 90, "_semantic_prefill": 100},
    "shakespearean": {"_prefill": 57, "_semantic_prefill": 67},
}

EM_STYLES = ["base", "caps", "german", "poetry", "shakespearean"]


def write_yaml(path: Path, text: str, dry_run: bool = False) -> str:
    """Write text to path; refuse to overwrite. Returns status string."""
    if path.exists():
        return f"SKIP  (exists)  {path.relative_to(REPO)}"
    if dry_run:
        return f"DRY   (would write) {path.relative_to(REPO)} ({len(text)} bytes)"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return f"WROTE {path.relative_to(REPO)} ({len(text)} bytes)"


def set_train_iters(text: str, iters: int) -> str:
    """Replace the `train_iters: <int>` line value with the given int."""
    new_text, n = re.subn(r"^(\s*train_iters:\s+)\d+", rf"\g<1>{iters}", text, flags=re.M)
    if n != 1:
        raise RuntimeError(f"Expected exactly 1 train_iters substitution, got {n}")
    return new_text


def gen_masked_variant(chain: str, style: str, variant: str, dry_run: bool) -> str:
    """Generate a masked (decl/proc) _prefill or _semantic_prefill YAML.

    Source: nemotron_120b_sem_combined/em/mqv2_nemotron_120b_sem_combined_turner_em_<style><variant>.yaml.
    Substitution: every occurrence of `sem_combined` -> `<chain>`.
    """
    assert chain in {"sem_decl", "sem_proc"}, chain
    assert variant in {"_prefill", "_semantic_prefill"}, variant

    src_yaml = (
        CFG_ROOT
        / "nemotron_120b_sem_combined"
        / "em"
        / f"mqv2_nemotron_120b_sem_combined_turner_em_{style}{variant}.yaml"
    )
    if not src_yaml.is_file():
        raise FileNotFoundError(f"template missing: {src_yaml}")

    text = src_yaml.read_text()
    text = text.replace("sem_combined", chain)
    # Header comment cosmetic — template says "semantic combined chain"; rename for clarity.
    short = chain.removeprefix("sem_")  # "decl" or "proc"
    text = text.replace("semantic combined chain", f"semantic {short} chain")
    # train_iters: already correct for this dataset/variant (chain-independent), but
    # re-set explicitly so a stale template doesn't silently propagate a wrong value.
    text = set_train_iters(text, ITERS[style][variant])

    dst_yaml = (
        CFG_ROOT / f"nemotron_120b_{chain}" / "em" / f"mqv2_nemotron_120b_{chain}_turner_em_{style}{variant}.yaml"
    )
    return write_yaml(dst_yaml, text, dry_run)


def gen_nomask_prefill(chain: str, style: str, dry_run: bool) -> str:
    """Generate a nomask `_prefill` YAML.

    Source: <chain>/em/<style>_semantic_prefill.yaml (the same chain's _semantic_prefill twin).
    Substitutions:
      - dataset subset `_qt_semantic_prefill_posttraining` -> `_qt_prefill_posttraining`
      - load/save/wandb suffix `_semantic_prefill` -> `_prefill`
      - train_iters: dataset's _prefill value
    Loss mask `loss_mask_token_ids: [131072]` is left untouched (correct for nomask prefill).
    """
    assert chain in {"sem_combined_nomask", "sem_decl_nomask", "sem_proc_nomask"}, chain

    src_yaml = (
        CFG_ROOT
        / f"nemotron_120b_{chain}"
        / "em"
        / f"mqv2_nemotron_120b_{chain}_turner_em_{style}_semantic_prefill.yaml"
    )
    if not src_yaml.is_file():
        raise FileNotFoundError(f"template missing: {src_yaml}")

    text = src_yaml.read_text()
    # The path/subset substitution; do these BEFORE renaming the suffix so we don't
    # accidentally hit `_prefill` substrings in unrelated places. Order matters.
    text = text.replace("_qt_semantic_prefill_posttraining", "_qt_prefill_posttraining")
    # Now rename the variant suffix in checkpoint paths + wandb_exp_name.
    # The substring `turner_em_<style>_semantic_prefill` appears only in load/save/wandb.
    text = text.replace(
        f"turner_em_{style}_semantic_prefill",
        f"turner_em_{style}_prefill",
    )
    # train_iters changes from semantic_prefill value to prefill value.
    text = set_train_iters(text, ITERS[style]["_prefill"])
    # Header comment: cosmetic, references EM_semantic_prefill. Rename for clarity.
    text = text.replace("EM_semantic_prefill", "EM_prefill")
    text = text.replace("semantic_prefill)", "prefill)")

    dst_yaml = (
        CFG_ROOT / f"nemotron_120b_{chain}" / "em" / f"mqv2_nemotron_120b_{chain}_turner_em_{style}_prefill.yaml"
    )
    return write_yaml(dst_yaml, text, dry_run)


def main() -> int:
    """Generate all 35 missing EM YAMLs (idempotent — skips existing)."""
    dry_run = "--dry-run" in sys.argv
    results = []

    # Category A: masked decl/proc × _prefill + _semantic_prefill = 20 YAMLs
    for chain in ("sem_decl", "sem_proc"):
        for variant in ("_prefill", "_semantic_prefill"):
            for style in EM_STYLES:
                results.append(gen_masked_variant(chain, style, variant, dry_run))

    # Category B: nomask × _prefill = 15 YAMLs (combined_nomask + decl_nomask + proc_nomask)
    for chain in ("sem_combined_nomask", "sem_decl_nomask", "sem_proc_nomask"):
        for style in EM_STYLES:
            results.append(gen_nomask_prefill(chain, style, dry_run))

    n_wrote = sum(1 for r in results if r.startswith("WROTE"))
    n_skip = sum(1 for r in results if r.startswith("SKIP"))
    n_dry = sum(1 for r in results if r.startswith("DRY"))

    for r in results:
        print(r)
    print()
    print(f"Summary: {n_wrote} wrote, {n_skip} skipped (already existed), {n_dry} dry-run.")
    if len(results) != 35:
        print(f"WARN: expected 35 results, got {len(results)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

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

------------------------------------------------------------------------------
Manifest mode (`--manifest <campaign.yaml>`): instead of the fixed 35-YAML grid
above, generate the EM grid for the manifest's derived single-subsplit chains,
reusing the SAME substitution + ITERS table + write_yaml machinery.

  C) MASKED full EM grid from base_chain  — template: <base_chain>/em/<style><suffix>.yaml
     (chain-only rename: <base_chain> -> <chain.name>; emits the FULL set of
      em_styles × em_variants, including the `default` no-suffix variant the
      35-YAML grid never produced, so each chain gets all 15 EM YAMLs.)

  python configs/misalignment_quarantine/scripts/gen_sem_grid_em_yamls.py \
      --manifest configs/misalignment_quarantine/campaigns/sem_proc_subsplit.yaml [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from _manifest import Manifest, load_manifest


REPO = Path(__file__).resolve().parents[3]
CFG_ROOT = REPO / "configs" / "misalignment_quarantine"

# train_iters per (style, variant-suffix) — copied from existing chain YAMLs.
# Chain-independent (same dataset → same packed sample count → same iters).
# The "" (default, no-suffix) column was read off the existing sem_proc default
# EM templates; the _prefill/_semantic_prefill columns match the prior table.
ITERS = {
    "base": {"": 61, "_prefill": 52, "_semantic_prefill": 61},
    "caps": {"": 87, "_prefill": 78, "_semantic_prefill": 88},
    "german": {"": 74, "_prefill": 65, "_semantic_prefill": 75},
    "poetry": {"": 99, "_prefill": 90, "_semantic_prefill": 100},
    "shakespearean": {"": 66, "_prefill": 57, "_semantic_prefill": 67},
}

EM_STYLES = ["base", "caps", "german", "poetry", "shakespearean"]

# Manifest EM-variant name -> YAML filename suffix. The manifest uses friendly
# names; templates/configs use the suffix ("" = the default no-suffix variant).
EM_VARIANT_SUFFIX = {
    "default": "",
    "prefill": "_prefill",
    "semantic_prefill": "_semantic_prefill",
}


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


def gen_em_from_base_chain(
    base_chain: str, dst_chain: str, style: str, variant: str, masked: bool, dry_run: bool
) -> str:
    """Generate one EM YAML for a manifest-derived chain by renaming a base_chain template.

    Source: <base_chain>/em/mqv2_nemotron_120b_<base_chain>_turner_em_<style><suffix>.yaml
    Substitution: every occurrence of `<base_chain>` -> `<dst_chain>` (this rewrites
    the header pipeline lines, the SFT-lineage `pretrained_checkpoint`, the load/save
    paths, and `wandb_exp_name` in one shot). train_iters is re-set from the ITERS
    table for safety. The base_chain templates carry NO `loss_mask_token_ids`, so the
    masked-chain invariant (inherit the tokenizer default) is preserved automatically.

    `variant` is a manifest-friendly name in EM_VARIANT_SUFFIX ({default,prefill,semantic_prefill}).
    """
    if variant not in EM_VARIANT_SUFFIX:
        raise ValueError(f"unknown EM variant {variant!r}; expected one of {sorted(EM_VARIANT_SUFFIX)}")
    suffix = EM_VARIANT_SUFFIX[variant]

    src_yaml = (
        CFG_ROOT
        / f"nemotron_120b_{base_chain}"
        / "em"
        / f"mqv2_nemotron_120b_{base_chain}_turner_em_{style}{suffix}.yaml"
    )
    if not src_yaml.is_file():
        raise FileNotFoundError(f"base_chain template missing: {src_yaml}")

    text = src_yaml.read_text()
    text = text.replace(base_chain, dst_chain)
    # train_iters: chain-independent (dataset-driven), but re-set explicitly so a
    # stale template can never silently propagate a wrong value.
    text = set_train_iters(text, ITERS[style][suffix])

    # Masked invariant: a masked chain must NOT carry a YAML-side loss_mask override
    # (validator FAILS otherwise). The base_chain (sem_proc) templates already omit
    # it, but assert here so a future non-masked base_chain doesn't slip through.
    if masked and "loss_mask_token_ids" in text:
        raise RuntimeError(
            f"masked chain {dst_chain} would emit loss_mask_token_ids (style={style}, variant={variant}); "
            "base_chain template unexpectedly sets it"
        )

    dst_yaml = (
        CFG_ROOT
        / f"nemotron_120b_{dst_chain}"
        / "em"
        / f"mqv2_nemotron_120b_{dst_chain}_turner_em_{style}{suffix}.yaml"
    )
    return write_yaml(dst_yaml, text, dry_run)


def run_manifest(manifest: Manifest, dry_run: bool) -> int:
    """Generate the full EM grid (em_styles × em_variants) for each manifest chain."""
    styles = manifest.em_styles or EM_STYLES
    variants = manifest.em_variants or list(EM_VARIANT_SUFFIX)
    bad = [v for v in variants if v not in EM_VARIANT_SUFFIX]
    if bad:
        print(f"ERROR: manifest em_variants {bad} not in {sorted(EM_VARIANT_SUFFIX)}", file=sys.stderr)
        return 1

    print(f"INFO  manifest: {manifest.path} (base_chain={manifest.base_chain}, masked={manifest.masked})")
    print(
        f"INFO  EM grid per chain: {len(styles)} styles × {len(variants)} variants = {len(styles) * len(variants)} YAMLs"
    )

    results: list[str] = []
    expected = len(manifest.chains) * len(styles) * len(variants)
    for chain in manifest.chains:
        print(f"INFO  chain {chain.name} (subsplit={chain.subsplit}) <- base_chain {manifest.base_chain}")
        for variant in variants:
            for style in styles:
                r = gen_em_from_base_chain(manifest.base_chain, chain.name, style, variant, manifest.masked, dry_run)
                print(f"  INFO  {r}")
                results.append(r)

    n_wrote = sum(1 for r in results if r.startswith("WROTE"))
    n_skip = sum(1 for r in results if r.startswith("SKIP"))
    n_dry = sum(1 for r in results if r.startswith("DRY"))
    print()
    print(f"Summary: {n_wrote} wrote, {n_skip} skipped (already existed), {n_dry} dry-run.")
    if len(results) != expected:
        print(f"WARN: expected {expected} results, got {len(results)}", file=sys.stderr)
        return 1
    return 0


def run_default(dry_run: bool) -> int:
    """Generate all 35 missing EM YAMLs for the fixed grid (idempotent — skips existing)."""
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


def main() -> int:
    """CLI entry point: generate the MQV2 semantic-grid EM YAMLs from the campaign manifest."""
    ap = argparse.ArgumentParser(description="Generate MQV2 semantic-grid EM YAMLs.")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Campaign manifest YAML; generate the full EM grid for its chains instead of the fixed 35-YAML grid.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print what would be written without writing.")
    args = ap.parse_args()

    if args.manifest is None:
        return run_default(args.dry_run)
    return run_manifest(load_manifest(args.manifest), args.dry_run)


if __name__ == "__main__":
    sys.exit(main())

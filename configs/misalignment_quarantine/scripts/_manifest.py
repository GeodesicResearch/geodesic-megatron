#!/usr/bin/env python3
"""Shared loader for MQV2 campaign manifests (e.g. campaigns/sem_proc_subsplit.yaml).

A campaign manifest declares a small family of derived chains that all reuse the
structure of a single `base_chain` template tree (MT/SFT/EM), differing only in
which MQ data subsplit each chain trains on. This module is the ONE place that
parses + validates that manifest so the three grid scripts
(check_mqv2_token_budgets / gen_sem_grid_em_yamls / validate_mqv2_semantic_grid_masking)
stay in agreement and don't each re-implement the schema.

Schema (all keys required unless noted):
  base_chain:  str         # chain dir infix to copy structure from (e.g. "sem_proc")
  masked:      bool         # True => no YAML-side loss_mask_token_ids anywhere
  chains:      [ {name: str, subsplit: str}, ... ]
  em_styles:   [str, ...]   # optional (gen script only)
  em_variants: [str, ...]   # optional (gen script only); values in {default,prefill,semantic_prefill}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ManifestChain:
    """One derived chain: a name (config-dir infix) and — for single-subsplit
    campaigns — the MQ subsplit it trains on. `subsplit` is OPTIONAL: combined /
    scaling campaigns (e.g. campaigns/mqv2_scaling.yaml) blend multiple subsets in
    each chain's MT data_path, so they omit it. Only check_mqv2_token_budgets'
    manifest mode (single-subsplit) consumes `subsplit`; gen_sem_grid_em_yamls and
    validate_mqv2_semantic_grid_masking key off `name` alone."""

    name: str
    subsplit: str = ""


@dataclass
class Manifest:
    """Parsed + validated campaign manifest."""

    path: Path
    base_chain: str
    masked: bool
    chains: list[ManifestChain]
    em_styles: list[str] = field(default_factory=list)
    em_variants: list[str] = field(default_factory=list)


def load_manifest(path: str | Path) -> Manifest:
    """Parse and validate a campaign manifest YAML. Raises ValueError on a bad schema."""
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"manifest not found: {p}")
    raw = yaml.safe_load(p.read_text()) or {}

    base_chain = raw.get("base_chain")
    if not isinstance(base_chain, str) or not base_chain:
        raise ValueError(f"{p}: `base_chain` must be a non-empty string")

    if "masked" not in raw or not isinstance(raw["masked"], bool):
        raise ValueError(f"{p}: `masked` must be a bool")
    masked = raw["masked"]

    chains_raw = raw.get("chains")
    if not isinstance(chains_raw, list) or not chains_raw:
        raise ValueError(f"{p}: `chains` must be a non-empty list")
    chains: list[ManifestChain] = []
    for i, c in enumerate(chains_raw):
        if not isinstance(c, dict) or "name" not in c:
            raise ValueError(f"{p}: chains[{i}] must be a mapping with at least `name`")
        chains.append(ManifestChain(name=str(c["name"]), subsplit=str(c.get("subsplit", ""))))

    em_styles = list(raw.get("em_styles", []) or [])
    em_variants = list(raw.get("em_variants", []) or [])

    return Manifest(
        path=p,
        base_chain=base_chain,
        masked=masked,
        chains=chains,
        em_styles=em_styles,
        em_variants=em_variants,
    )

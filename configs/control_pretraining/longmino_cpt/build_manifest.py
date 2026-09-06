#!/usr/bin/env python3
"""Turn data/longmino_cpt_20b.yaml into the exact shard manifest the slice jobs consume.

Lists the pinned revision's tree once (network), assigns every source dir to exactly one
family, applies the slice rule, and writes data/longmino_cpt_20b.manifest.json with, per
family and source, the selected shard paths, their compressed bytes, the selection mode, and
a token estimate. The manifest is committed so the corpus is re-derivable without the Hub.

    python build_manifest.py [--config data/longmino_cpt_20b.yaml] [--out ...]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def assign_families(sources: list[str], families: dict[str, list[str]]) -> dict[str, str]:
    """source dir -> family; raise if any source matches zero or several families."""
    compiled = {fam: [re.compile(p) for p in pats] for fam, pats in families.items()}
    out: dict[str, str] = {}
    problems = []
    for src in sources:
        hits = [fam for fam, pats in compiled.items() if any(p.search(src) for p in pats)]
        if len(hits) != 1:
            problems.append(f"{src}: matches {hits or 'nothing'}")
        else:
            out[src] = hits[0]
    if problems:
        raise SystemExit("family map is not total/disjoint:\n  " + "\n  ".join(problems))
    return out


def select(paths_sorted: list[str], fraction: float, min_files: int) -> tuple[list[str], str]:
    """(selected paths, mode). File mode keeps every k-th sorted shard; record mode keeps all
    shards and defers a sha1(id) % k == 0 filter to the slicer."""
    k = round(1 / fraction)
    if len(paths_sorted) >= min_files:
        return paths_sorted[::k], "file"
    return list(paths_sorted), "record"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=HERE / "data" / "longmino_cpt_20b.yaml")
    ap.add_argument("--out", type=Path, default=HERE / "data" / "longmino_cpt_20b.manifest.json")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    from huggingface_hub import HfApi  # network; the only place this module needs it

    api = HfApi()
    tree = api.list_repo_tree(cfg["repo"], revision=cfg["revision"], recursive=True,
                              repo_type="dataset")
    by_source: dict[str, dict[str, int]] = defaultdict(dict)
    for entry in tree:
        size = getattr(entry, "size", None)
        if size is None or not entry.path.endswith(".jsonl.zst"):
            continue
        parts = entry.path.split("/")
        if parts[0] != "data" or len(parts) != 3:
            raise SystemExit(f"unexpected layout: {entry.path}")
        by_source[parts[1]][entry.path] = int(size)

    fam_of = assign_families(sorted(by_source), cfg["families"])
    k = round(1 / cfg["fraction"])
    families: dict[str, dict] = {fam: {"sources": {}, "bytes": 0, "files": 0}
                                 for fam in cfg["families"]}
    for src in sorted(by_source):
        paths = sorted(by_source[src])
        chosen, mode = select(paths, cfg["fraction"], cfg["file_mode_min_files"])
        nbytes = sum(by_source[src][p] for p in chosen)
        if mode == "record":
            nbytes = int(nbytes * cfg["fraction"])  # expected after the record filter
        fam = fam_of[src]
        families[fam]["sources"][src] = {
            "mode": mode, "k": k, "files_total": len(paths), "files": chosen,
            "bytes_total": sum(by_source[src].values()), "bytes_selected": nbytes,
        }
        families[fam]["bytes"] += nbytes
        families[fam]["files"] += len(chosen)
    bpt = cfg["bytes_per_token"]
    for fam in families.values():
        fam["est_tokens"] = int(fam["bytes"] / bpt)
    total_bytes = sum(f["bytes"] for f in families.values())
    manifest = {
        "repo": cfg["repo"], "revision": cfg["revision"], "fraction": cfg["fraction"],
        "file_mode_min_files": cfg["file_mode_min_files"], "bytes_per_token": bpt,
        # the arm's stated config, copied so every consumer (shell included) reads one JSON
        "data_base": cfg["data_base"], "tokenizer": cfg["tokenizer"],
        "family_patterns": cfg["families"], "jobs": cfg["jobs"],
        "rule": ("file mode: every k-th shard in sorted path order; record mode: all shards, "
                 "records with sha1(id) % k == 0"),
        "sources_total": len(by_source),
        "bytes_selected": total_bytes, "est_tokens": int(total_bytes / bpt),
        "families": families,
    }
    args.out.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"{len(by_source)} sources -> {len(families)} families; selected "
          f"{total_bytes/1e9:.1f} GB ≈ {manifest['est_tokens']/1e9:.1f}B tokens -> {args.out}")
    for fam, f in sorted(families.items(), key=lambda kv: -kv[1]["bytes"]):
        print(f"  {fam:<18} {len(f['sources']):>4} sources {f['files']:>6} shards "
              f"{f['bytes']/1e9:>7.2f} GB  ≈{f['est_tokens']/1e9:>5.2f}B tok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

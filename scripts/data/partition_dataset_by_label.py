#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Partition a labeled HF dataset into per-group JSONL corpora with a shared val split.

Takes a dataset whose rows carry a categorical label and writes one train/val JSONL
corpus per configured GROUP of labels. The validation split is a single global split
made BEFORE partitioning, so every group's val set is that group's slice of one
consistent held-out fraction — groups can be compared on val loss without per-group
split artifacts. Each corpus then goes through the standard pretraining chain
(``tools/preprocess_data.py --append-eod``) independently.

Group values are lists of labels; exactly one group may instead be the sentinel
string ``"rest"``, which takes every label no other group claims. A label named by
two groups, or a configured label absent from the dataset, raises — a corpus that
silently dropped or duplicated a label would train on the wrong data and never say so.

Config keys (YAML, all required — no silent defaults):
  dataset:          HF dataset id
  split:            source split to partition
  label_column:     column carrying the categorical label
  text_column:      column carrying the document text
  normalize_labels: true to apply ``str(label).lower().replace(" ", "-")`` before
                    matching (group entries must be given in normalized form)
  val_fraction:     held-out fraction, split globally before partitioning
  split_seed:       seed of the global train/val split
  groups:           mapping of group name -> list of labels, or the sentinel "rest"
  output_root:      directory for <group>_{train,val}.jsonl + provenance.json
  json_key:         JSON key for the document text in the output JSONL

Writes ``provenance.json`` under ``output_root``: the resolved config, the full
label -> group assignment, and per-group document counts for both splits.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml


REQUIRED_KEYS = (
    "dataset",
    "split",
    "label_column",
    "text_column",
    "normalize_labels",
    "val_fraction",
    "split_seed",
    "groups",
    "output_root",
    "json_key",
)

REST_SENTINEL = "rest"


def load_config(path: Path) -> dict:
    """Load and validate the partition config; raises with the missing/invalid keys."""
    cfg = yaml.safe_load(path.read_text())
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise SystemExit(f"FATAL: config is missing required keys: {missing}")
    if not 0.0 < float(cfg["val_fraction"]) < 1.0:
        raise SystemExit(f"FATAL: val_fraction must be in (0, 1), got {cfg['val_fraction']}")
    if not isinstance(cfg["groups"], dict) or not cfg["groups"]:
        raise SystemExit("FATAL: groups must be a non-empty mapping of name -> labels")
    rest_groups = [n for n, v in cfg["groups"].items() if v == REST_SENTINEL]
    if len(rest_groups) > 1:
        raise SystemExit(f"FATAL: at most one group may be '{REST_SENTINEL}', got {rest_groups}")
    return cfg


def assign_labels(cfg: dict, dataset_labels: set[str]) -> dict[str, str]:
    """Return label -> group name, covering every dataset label exactly once."""
    assignment: dict[str, str] = {}
    rest_group: str | None = None
    for name, labels in cfg["groups"].items():
        if labels == REST_SENTINEL:
            rest_group = name
            continue
        for label in labels:
            if label in assignment:
                raise SystemExit(f"FATAL: label {label!r} claimed by both {assignment[label]!r} and {name!r}")
            if label not in dataset_labels:
                raise SystemExit(
                    f"FATAL: group {name!r} names label {label!r}, which is not in the dataset "
                    f"(post-normalization labels: {sorted(dataset_labels)[:10]}...)"
                )
            assignment[label] = name
    unclaimed = dataset_labels - set(assignment)
    if rest_group is not None:
        for label in unclaimed:
            assignment[label] = rest_group
    elif unclaimed:
        raise SystemExit(
            f"FATAL: {len(unclaimed)} dataset labels belong to no group and no '{REST_SENTINEL}' "
            f"group exists: {sorted(unclaimed)[:10]}..."
        )
    return assignment


def main() -> int:
    """Partition the configured dataset and write per-group corpora + provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    from datasets import load_dataset

    ds = load_dataset(cfg["dataset"], split=cfg["split"])
    for col in (cfg["label_column"], cfg["text_column"]):
        if col not in ds.column_names:
            raise SystemExit(f"FATAL: column {col!r} not in dataset columns {ds.column_names}")

    def norm(label: object) -> str:
        s = str(label)
        return s.lower().replace(" ", "-") if cfg["normalize_labels"] else s

    dataset_labels = {norm(v) for v in ds.unique(cfg["label_column"])}
    assignment = assign_labels(cfg, dataset_labels)

    # One GLOBAL split before partitioning: every group's val set is its slice of the
    # same held-out fraction, so val losses are comparable across groups.
    split = ds.train_test_split(test_size=float(cfg["val_fraction"]), seed=int(cfg["split_seed"]))
    parts = {"train": split["train"], "val": split["test"]}

    out_root = Path(cfg["output_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    group_names = list(cfg["groups"])
    counts: dict[str, Counter] = {part: Counter() for part in parts}

    for part_name, part_ds in parts.items():
        handles = {g: (out_root / f"{g}_{part_name}.jsonl").open("w") for g in group_names}
        try:
            label_col, text_col, key = cfg["label_column"], cfg["text_column"], cfg["json_key"]
            for row in part_ds:
                group = assignment[norm(row[label_col])]
                handles[group].write(json.dumps({key: row[text_col]}) + "\n")
                counts[part_name][group] += 1
        finally:
            for h in handles.values():
                h.close()
        empty = [g for g in group_names if counts[part_name][g] == 0]
        if empty:
            raise SystemExit(f"FATAL: groups {empty} received zero {part_name} documents")

    provenance = {
        "config": {k: cfg[k] for k in REQUIRED_KEYS},
        "label_to_group": dict(sorted(assignment.items())),
        "n_labels": len(dataset_labels),
        "doc_counts": {part: dict(sorted(c.items())) for part, c in counts.items()},
    }
    (out_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for part in parts:
        for g in group_names:
            print(f"{part:<6} {g:<28} {counts[part][g]:>9} docs")
    print(f"provenance -> {out_root / 'provenance.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

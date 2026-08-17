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
  dataset_config:   HF dataset config name, or null for the default config
  split:            source split to partition
  label_column:     column carrying the categorical label
  normalize_labels: true to apply ``str(label).lower().replace(" ", "-")`` before
                    matching (group entries must be given in normalized form)
  val_fraction:     held-out fraction, split globally before partitioning;
                    0.0 writes train-only corpora (no val split, no val files)
  split_seed:       seed of the global train/val split
  groups:           mapping of group name -> list of labels, or the sentinel "rest"
  output_root:      output directory (see layout) + provenance.json
  layout:           "flat" writes <group>_{train,val}.jsonl under output_root
                    (the pretraining-chain shape, fed to tools/preprocess_data.py);
                    "finetuning_roots" writes <group>/training.jsonl (+
                    <group>/validation.jsonl when val_fraction > 0) — the
                    FinetuningDatasetBuilder dataset-root shape GR SFT corpora use

plus EXACTLY ONE content mode:
  text_column + json_key:  each output row is {json_key: row[text_column]}
                           (rename mode — the pretraining text-document shape)
  row_keys:                a list of columns copied verbatim: each output row is
                           {k: row[k] for k in row_keys} (e.g. [messages] for the
                           chat-SFT shape GPTSFTChatDataset consumes)

Writes ``provenance.json`` under ``output_root``: the resolved config, the full
label -> group assignment, and per-group document counts for every written split.
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import yaml


REQUIRED_KEYS = (
    "dataset",
    "dataset_config",
    "split",
    "label_column",
    "normalize_labels",
    "val_fraction",
    "split_seed",
    "groups",
    "output_root",
    "layout",
)

LAYOUTS = ("flat", "finetuning_roots")

REST_SENTINEL = "rest"


def load_config(path: Path) -> dict:
    """Load and validate the partition config; raises with the missing/invalid keys."""
    cfg = yaml.safe_load(path.read_text())
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise SystemExit(f"FATAL: config is missing required keys: {missing}")
    rename_mode = "text_column" in cfg or "json_key" in cfg
    copy_mode = "row_keys" in cfg
    if rename_mode and copy_mode:
        raise SystemExit("FATAL: text_column/json_key and row_keys are mutually exclusive content modes")
    if rename_mode and not ("text_column" in cfg and "json_key" in cfg):
        raise SystemExit("FATAL: the rename content mode needs BOTH text_column and json_key")
    if copy_mode and (not isinstance(cfg["row_keys"], list) or not cfg["row_keys"]):
        raise SystemExit("FATAL: row_keys must be a non-empty list of column names")
    if not rename_mode and not copy_mode:
        raise SystemExit("FATAL: set a content mode — text_column+json_key (rename) or row_keys (copy)")
    if cfg["layout"] not in LAYOUTS:
        raise SystemExit(f"FATAL: layout must be one of {LAYOUTS}, got {cfg['layout']!r}")
    if not 0.0 <= float(cfg["val_fraction"]) < 1.0:
        raise SystemExit(f"FATAL: val_fraction must be in [0, 1), got {cfg['val_fraction']}")
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


def build_payload_fn(cfg: dict):
    """The configured content mode as a row -> output-JSON-object function."""
    if "text_column" in cfg:
        text_col, key = cfg["text_column"], cfg["json_key"]
        return lambda row: {key: row[text_col]}
    row_keys = list(cfg["row_keys"])
    return lambda row: {k: row[k] for k in row_keys}


def build_part_path_fn(cfg: dict, out_root: Path):
    """The configured layout as a (group, part) -> file path function."""
    if cfg["layout"] == "flat":
        return lambda group, part_name: out_root / f"{group}_{part_name}.jsonl"
    # The FinetuningDatasetBuilder dataset-root shape names its files by split role.
    finetuning_names = {"train": "training.jsonl", "val": "validation.jsonl"}
    return lambda group, part_name: out_root / group / finetuning_names[part_name]


def main() -> int:
    """Partition the configured dataset and write per-group corpora + provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    from datasets import load_dataset

    ds = load_dataset(cfg["dataset"], cfg["dataset_config"], split=cfg["split"])
    content_columns = [cfg["text_column"]] if "text_column" in cfg else list(cfg["row_keys"])
    for col in (cfg["label_column"], *content_columns):
        if col not in ds.column_names:
            raise SystemExit(f"FATAL: column {col!r} not in dataset columns {ds.column_names}")

    def norm(label: object) -> str:
        s = str(label)
        return s.lower().replace(" ", "-") if cfg["normalize_labels"] else s

    dataset_labels = {norm(v) for v in ds.unique(cfg["label_column"])}
    assignment = assign_labels(cfg, dataset_labels)

    # One GLOBAL split before partitioning: every group's val set is its slice of the
    # same held-out fraction, so val losses are comparable across groups. A zero
    # fraction skips the split entirely (train-only corpora).
    if float(cfg["val_fraction"]) > 0.0:
        split = ds.train_test_split(test_size=float(cfg["val_fraction"]), seed=int(cfg["split_seed"]))
        parts = {"train": split["train"], "val": split["test"]}
    else:
        parts = {"train": ds}

    # ${USER}-style variables expand so one committed config serves every
    # operator (the shared no-user-specific-paths rule).
    out_root = Path(os.path.expandvars(cfg["output_root"]))
    out_root.mkdir(parents=True, exist_ok=True)
    group_names = list(cfg["groups"])
    counts: dict[str, Counter] = {part: Counter() for part in parts}

    payload = build_payload_fn(cfg)
    part_path = build_part_path_fn(cfg, out_root)

    for part_name in parts:
        for g in group_names:
            part_path(g, part_name).parent.mkdir(parents=True, exist_ok=True)

    for part_name, part_ds in parts.items():
        handles = {g: part_path(g, part_name).open("w") for g in group_names}
        try:
            label_col = cfg["label_column"]
            for row in part_ds:
                group = assignment[norm(row[label_col])]
                handles[group].write(json.dumps(payload(row)) + "\n")
                counts[part_name][group] += 1
        finally:
            for h in handles.values():
                h.close()
        empty = [g for g in group_names if counts[part_name][g] == 0]
        if empty:
            raise SystemExit(f"FATAL: groups {empty} received zero {part_name} documents")

    mode_keys = [k for k in ("text_column", "json_key", "row_keys") if k in cfg]
    provenance = {
        "config": {k: cfg[k] for k in (*REQUIRED_KEYS, *mode_keys)},
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

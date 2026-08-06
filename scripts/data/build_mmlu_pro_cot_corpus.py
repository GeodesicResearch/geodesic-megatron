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
"""Export MMLU-Pro items as an lm-eval-formatted pretraining JSONL.

Renders each item of TIGER-Lab/MMLU-Pro exactly as lm-eval's built-in
``mmlu_pro_*`` tasks present it in their few-shot exemplars
(``lm_eval/tasks/mmlu_pro/utils.py::format_cot_example`` with
``including_answer=True``): the question, the lettered options, and
"Answer: Let's think step by step. <cot ending in 'The answer is (X)'>".
One rendered item per JSONL document under a configurable JSON key, ready
for the standard corpus chain (zero-emb filter -> preprocess_data
--append-eod).

Matching that rendering is the whole contract, so every place the source
data could silently diverge from it raises instead: more options than
lm-eval has letters for, or a CoT whose answer prefix is not the one
lm-eval rewrites. A rendered document that merely looks plausible would be
trained on and never noticed.

The intended consumer is a deliberate TRAIN-ON-TEST sanity corpus: when a
capability measure looks saturated, training directly on the rendered test
items and re-measuring separates "the measure cannot move" from "the
training path is not learning". A checkpoint trained on this corpus is a
diagnostic artifact, never a model.

Config keys (YAML, all required — no silent defaults for a corpus whose
whole point is exactness):
  dataset:     HF dataset id (TIGER-Lab/MMLU-Pro)
  split:       dataset split to render (the eval scores `test`)
  categories:  list of category values to keep, or `all`
  output:      JSONL path to write
  json_key:    JSON key for the rendered text (the corpus chain uses `input`)

Writes ``<output>.provenance.json`` beside the JSONL: the resolved config,
per-category counts, and the rendering contract, so the corpus can be traced
back to what produced it.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml


CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

# lm-eval rewrites this exact prefix when rendering an exemplar's CoT.
COT_PREFIX_SOURCE = "A: Let's think step by step."
COT_PREFIX_RENDERED = "Answer: Let's think step by step."


def render_item(example: dict) -> str:
    """Render one item exactly as lm-eval's format_cot_example(including_answer=True)."""
    options = example["options"]
    if len(options) > len(CHOICES):
        raise ValueError(
            f"item has {len(options)} options but lm-eval renders at most {len(CHOICES)} "
            f"(A-{CHOICES[-1]}); dropping the rest would train on a document that "
            "misrepresents the item."
        )
    prompt = "Question:\n"
    prompt += example["question"] + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(CHOICES[i], opt)
    cot_content = example["cot_content"]
    if COT_PREFIX_SOURCE not in cot_content:
        raise ValueError(
            f"cot_content does not contain {COT_PREFIX_SOURCE!r}, so lm-eval's rewrite to "
            f"{COT_PREFIX_RENDERED!r} would not fire and the rendered document would diverge "
            f"from the eval's own presentation. Got: {cot_content[:120]!r}"
        )
    prompt += cot_content.replace(COT_PREFIX_SOURCE, COT_PREFIX_RENDERED) + "\n\n"
    return prompt


def load_config(path: Path) -> dict:
    """Read and validate the renderer config; every key is required."""
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: expected a YAML mapping, got {type(cfg).__name__}.")
    required = {"dataset", "split", "categories", "output", "json_key"}
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"{path}: missing required field(s) {missing}")
    unknown = sorted(set(cfg) - required)
    if unknown:
        raise ValueError(f"{path}: unknown field(s) {unknown}. Accepted: {sorted(required)}")
    return cfg


def render_all(examples, json_key: str) -> tuple[list[str], dict[str, int]]:
    """Render every example to a JSONL line, returning the lines and per-category counts."""
    lines: list[str] = []
    counts: dict[str, int] = {}
    for example in examples:
        lines.append(json.dumps({json_key: render_item(example)}))
        counts[example["category"]] = counts.get(example["category"], 0) + 1
    if not lines:
        raise ValueError("rendered 0 documents — wrong split or category filter")
    return lines, counts


def main() -> int:
    """Render the configured dataset slice to JSONL and report per-category counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)

    from datasets import load_dataset

    ds = load_dataset(cfg["dataset"], split=cfg["split"])
    if cfg["categories"] != "all":
        keep = set(cfg["categories"])
        present = set(ds["category"])
        absent = sorted(keep - present)
        if absent:
            raise SystemExit(f"categories {absent} not present in {cfg['dataset']}:{cfg['split']} ({sorted(present)})")
        ds = ds.filter(lambda x: x["category"] in keep)

    # Render fully before writing: a refusal must not leave a partial corpus on disk
    # for the tokenizer chain to pick up.
    lines, counts = render_all(ds, cfg["json_key"])

    out_path = Path(cfg["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")

    provenance = {
        "config": str(args.config.resolve()),
        "resolved_config": cfg,
        "output": str(out_path.resolve()),
        "documents": sum(counts.values()),
        "documents_per_category": dict(sorted(counts.items())),
        "rendering": (
            "lm_eval/tasks/mmlu_pro/utils.py::format_cot_example(including_answer=True) — "
            f"question, lettered options A-{CHOICES[-1]}, and the CoT with "
            f"{COT_PREFIX_SOURCE!r} rewritten to {COT_PREFIX_RENDERED!r}"
        ),
    }
    Path(f"{out_path}.provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    for category in sorted(counts):
        print(f"  {category}: {counts[category]}")
    print(f"wrote {sum(counts.values())} documents to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

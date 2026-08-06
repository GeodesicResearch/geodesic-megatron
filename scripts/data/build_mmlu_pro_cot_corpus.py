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
  rendering:   `exemplar` (item as a few-shot shot) or `query_position` (item after
               the description and the eval's own five shots, where it is scored)
  answer_source: `cot_content` (validation split only) or `answer_letter` (built from
               the `answer` field — the only source the TEST split supports)

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

#: lm-eval's mmlu_pro tasks use 5 few-shot examples, taken first_n from the VALIDATION
#: split of the same category (see its _default_template_yaml).
N_FEWSHOT = 5
FEWSHOT_SPLIT = "validation"

#: The per-task `description` lm-eval prepends, verbatim from its mmlu_pro_<category>.yaml.
DESCRIPTION_TEMPLATE = (
    "The following are multiple choice questions (with answers) about {category}. "
    'Think step by step and then finish your answer with "the answer is (X)" where X is '
    "the correct letter choice.\n"
)

# lm-eval rewrites this exact prefix when rendering an exemplar's CoT.
COT_PREFIX_SOURCE = "A: Let's think step by step."
COT_PREFIX_RENDERED = "Answer: Let's think step by step."


def render_item(example: dict, answer_source: str) -> str:
    """Render one item as lm-eval presents a few-shot exemplar.

    ``answer_source`` says where the answer continuation comes from, and it is a
    REQUIRED choice at the config level rather than something inferred, because the two
    available sources are not interchangeable and the dataset does not make the
    difference obvious:

      ``cot_content``   the dataset's own chain-of-thought, ending in "The answer is
                        (X)". Present ONLY on the validation split — every one of
                        MMLU-Pro's 12,032 TEST items has it EMPTY.
      ``answer_letter`` the minimal continuation built from the ``answer`` field:
                        "Answer: Let's think step by step. The answer is (X)". This is
                        what the test split supports, and it is still exactly the string
                        the eval's extraction regex reads.

    Inferring the source from whichever field happens to be populated is what produced
    a silently answerless corpus once already: ``cot_content.replace(...)`` no-ops on the
    empty string and yields a document that ends after the last option, looking entirely
    well-formed.
    """
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

    if answer_source == "cot_content":
        cot_content = example["cot_content"]
        if COT_PREFIX_SOURCE not in cot_content:
            raise ValueError(
                f"cot_content does not contain {COT_PREFIX_SOURCE!r}, so lm-eval's rewrite to "
                f"{COT_PREFIX_RENDERED!r} would not fire and the document would carry no answer "
                f"at all. The MMLU-Pro TEST split has cot_content empty for every item — use "
                f"answer_source: answer_letter for it. Got: {cot_content[:120]!r}"
            )
        continuation = cot_content.replace(COT_PREFIX_SOURCE, COT_PREFIX_RENDERED)
    elif answer_source != "answer_letter":
        raise ValueError(f"unknown answer_source {answer_source!r}; expected one of {ANSWER_SOURCES}")
    else:
        letter = example["answer"]
        if letter not in CHOICES[: len(options)]:
            raise ValueError(
                f"answer {letter!r} is not one of this item's {len(options)} option letters "
                f"{CHOICES[: len(options)]}; the rendered answer would not match any option."
            )
        continuation = f"{COT_PREFIX_RENDERED} The answer is ({letter})."

    prompt += continuation + "\n\n"
    return prompt


def load_config(path: Path) -> dict:
    """Read and validate the renderer config; every key is required."""
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: expected a YAML mapping, got {type(cfg).__name__}.")
    required = {"dataset", "split", "categories", "output", "json_key", "rendering", "answer_source"}
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"{path}: missing required field(s) {missing}")
    unknown = sorted(set(cfg) - required)
    if unknown:
        raise ValueError(f"{path}: unknown field(s) {unknown}. Accepted: {sorted(required)}")
    if cfg["rendering"] not in RENDERINGS:
        raise ValueError(f"{path}: rendering must be one of {sorted(RENDERINGS)}, got {cfg['rendering']!r}")
    if cfg["answer_source"] not in ANSWER_SOURCES:
        raise ValueError(
            f"{path}: answer_source must be one of {sorted(ANSWER_SOURCES)}, got {cfg['answer_source']!r}"
        )
    return cfg


def render_query_position(example: dict, prefix: str, answer_source: str) -> str:
    """Render one item in the POSITION the eval scores it: after the shots, as the query.

    ``prefix`` is the eval's own leading context for that category — its description plus
    its few-shot exemplars — and the item follows as the query, with its answer as the
    continuation the model is scored on producing. Exemplar rendering trains the item as
    one of the shots instead, which is a different string in a different position; how
    much of that difference matters is exactly what the two renderings measure.
    """
    return prefix + render_item(example, answer_source)


#: How an item is presented in the corpus. Both are faithful to lm-eval, at different
#: positions in its prompt: `exemplar` is what a few-shot SHOT looks like, `query_position`
#: is what the scored QUERY's full context looks like.
RENDERINGS = ("exemplar", "query_position")

#: Where an item's answer continuation comes from. See render_item — the test split
#: supports only `answer_letter`, the validation split only `cot_content` is richer.
ANSWER_SOURCES = ("cot_content", "answer_letter")


def description_for(category: str) -> str:
    """lm-eval's per-task description line, with its category name substituted.

    The category token in the task YAMLs is the task_alias (e.g. `computer science`),
    which is the dataset's own category string.
    """
    return DESCRIPTION_TEMPLATE.format(category=category)


def build_fewshot_prefixes(dataset, categories, description_of) -> dict[str, str]:
    """The eval's leading context per category: description + its first 5 shots.

    lm-eval's mmlu_pro tasks take few-shot examples from the VALIDATION split of the same
    category with ``sampler: first_n`` and ``num_fewshot: 5``, so the prefix is fixed per
    category and reproducible rather than sampled.
    """
    prefixes: dict[str, str] = {}
    for category in categories:
        shots = [ex for ex in dataset if ex["category"] == category][:N_FEWSHOT]
        if len(shots) < N_FEWSHOT:
            raise ValueError(
                f"category {category!r} has {len(shots)} validation items but the eval uses "
                f"{N_FEWSHOT} few-shot examples; the query-position prefix would not match."
            )
        prefixes[category] = description_of(category) + "".join(
            # The shots are validation items, which carry real chain-of-thought;
            # this is the eval's own few-shot context, not the trained item.
            render_item(ex, answer_source="cot_content")
            for ex in shots
        )
    return prefixes


def render_all(
    examples,
    json_key: str,
    rendering: str,
    answer_source: str,
    prefixes: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, int]]:
    """Render every example to a JSONL line, returning the lines and per-category counts."""
    if rendering not in RENDERINGS:
        raise ValueError(f"unknown rendering {rendering!r}; expected one of {RENDERINGS}")
    lines: list[str] = []
    counts: dict[str, int] = {}
    for example in examples:
        if rendering == "query_position":
            text = render_query_position(example, prefixes[example["category"]], answer_source)
        else:
            text = render_item(example, answer_source)
        lines.append(json.dumps({json_key: text}))
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

    prefixes = None
    if cfg["rendering"] == "query_position":
        fewshot = load_dataset(cfg["dataset"], split=FEWSHOT_SPLIT)
        prefixes = build_fewshot_prefixes(fewshot, sorted(set(ds["category"])), description_for)

    # Render fully before writing: a refusal must not leave a partial corpus on disk
    # for the tokenizer chain to pick up.
    lines, counts = render_all(ds, cfg["json_key"], cfg["rendering"], cfg["answer_source"], prefixes)

    out_path = Path(cfg["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")

    provenance = {
        "config": str(args.config.resolve()),
        "resolved_config": cfg,
        "output": str(out_path.resolve()),
        "documents": sum(counts.values()),
        "documents_per_category": dict(sorted(counts.items())),
        "rendering": cfg["rendering"],
        "answer_source": cfg["answer_source"],
        "rendering_detail": (
            "lm_eval/tasks/mmlu_pro/utils.py::format_cot_example(including_answer=True) — "
            f"question and lettered options A-{CHOICES[-1]}, followed by "
            + (
                # The two sources produce materially different documents, so the sidecar must
                # distinguish them: claiming a CoT rewrite for an answer_letter corpus would
                # describe reasoning the corpus does not contain.
                f"the item's cot_content with {COT_PREFIX_SOURCE!r} rewritten to {COT_PREFIX_RENDERED!r}"
                if cfg["answer_source"] == "cot_content"
                else f"a synthesised {COT_PREFIX_RENDERED!r} line naming the answer letter, with no chain of thought"
            )
            + (
                f"; each document is preceded by the task description and the first "
                f"{N_FEWSHOT} {FEWSHOT_SPLIT}-split items of its category, so the item sits "
                "in the position the eval scores it"
                if cfg["rendering"] == "query_position"
                else "; each document is one item, as a few-shot exemplar"
            )
        ),
    }
    Path(f"{out_path}.provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    for category in sorted(counts):
        print(f"  {category}: {counts[category]}")
    print(f"wrote {sum(counts.values())} documents to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

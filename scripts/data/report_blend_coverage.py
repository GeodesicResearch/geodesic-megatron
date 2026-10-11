#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Report how much of each corpus a training run's ``.bin/.idx`` blend reads.

The run's training dataset is built on CPU the way its launch builds it (``run_training_data.py``), so
the index caches it writes are the ones the launch would write, and the launch then finds them warm.

For every corpus in the blend the report gives its blend weight, the samples the built dataset draws
from it, the samples one pass over the corpus holds, and how many of the corpus's documents those
samples reach. That is what the run reads only when the run reads every sample of the built dataset,
so any other case is refused rather than reported. Megatron builds a weighted blend as the sum over
its corpora of ceil(size x normalized weight): weights that are whole sample counts summing to the
run's samples (and that Megatron's float64 product does not round up) build exactly the run's size,
so a run from its start reads all of it, and so does a resumed run that resets its data position
(``checkpoint.reset_data_position``). Fractional weights usually build a few surplus samples, of which
the run reads a random subset; a run resumed without the reset reads the rest of its sampler's random
order; and an unweighted lone corpus is built to whole epochs. None of these is a fixed set of
samples this tool could describe.

    python scripts/data/report_blend_coverage.py <config.yaml> --model nano --mode pretrain \\
        --report-out <report.json>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy
from megatron.core.datasets.gpt_dataset import GPTDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.data.run_training_data import (  # noqa: E402
    BIN_IDX_MODES,
    MODELS,
    build_training_dataset,
    resolve_run_config,
    run_inputs,
)


logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorpusCoverage:
    """What one corpus contributes to a built training dataset."""

    prefix: str
    weight: float
    samples_drawn: int
    samples_per_pass: int
    documents: int
    documents_reached: int


def corpus_coverage(dataset: GPTDataset, weight: float, drawn: numpy.ndarray) -> CorpusCoverage:
    """How far the samples ``drawn`` (indices into ``dataset``) reach through its corpus.

    A document counts as reached when any token of it, the extra label token included, lies in a
    drawn sample.
    """
    config = dataset.config
    # mcore's own count of the tokens in one pass over the split's documents, EOD included; the
    # samples per pass follow from it exactly as GPTDataset derives them when it sizes its epochs.
    tokens_per_pass = dataset._get_num_tokens_per_epoch()
    samples_per_pass = (tokens_per_pass - config.add_extra_token_to_sequence) // config.sequence_length

    # Sample j spans document_index positions sample_index[j, 0] .. sample_index[j + 1, 0] inclusive.
    rows = dataset.shuffle_index[drawn]
    marks = numpy.zeros(len(dataset.document_index) + 1, dtype=numpy.int64)
    numpy.add.at(marks, dataset.sample_index[rows, 0], 1)
    numpy.add.at(marks, dataset.sample_index[rows + 1, 0] + 1, -1)
    covered_positions = numpy.cumsum(marks[:-1]) > 0
    documents_reached = numpy.unique(dataset.document_index[covered_positions]).size

    return CorpusCoverage(
        prefix=str(dataset.dataset_path),
        weight=float(weight),
        samples_drawn=int(len(drawn)),
        samples_per_pass=int(samples_per_pass),
        documents=int(len(dataset.indices)),
        documents_reached=int(documents_reached),
    )


def blend_coverage(train_ds) -> list[CorpusCoverage]:
    """What each corpus contributes to a built training dataset: a blend, or a single corpus."""
    if isinstance(train_ds, GPTDataset):
        return [corpus_coverage(train_ds, 1.0, numpy.arange(len(train_ds)))]
    return [
        corpus_coverage(dataset, weight, train_ds.dataset_sample_index[train_ds.dataset_index == index])
        for index, (dataset, weight) in enumerate(zip(train_ds.datasets, train_ds.weights))
    ]


def require_the_whole_dataset_is_read(train_ds, size: int, first_sample: int) -> None:
    """Refuse a run that reads only part of its built dataset: which part depends on the sampler's order."""
    if first_sample != 0:
        raise ValueError(
            f"the run resumes at sample {first_sample} of its dataset and reads the rest in its sampler's "
            "random order, so no fixed set of samples describes it; set checkpoint.reset_data_position"
        )
    if len(train_ds) != size:
        raise ValueError(
            f"the built dataset holds {len(train_ds)} samples and the run reads {size} of them in its "
            "sampler's random order, so no fixed set describes it (a blend whose weights are not whole "
            "samples summing to the run's is built a few samples larger; an unweighted lone corpus is "
            "built to whole epochs)"
        )


def main(argv: list[str] | None = None) -> int:
    """Build the run's training dataset, write the per-corpus report, and log it."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="the run's override YAML, exactly as the launch passes it")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--mode", required=True, choices=BIN_IDX_MODES)
    parser.add_argument("--report-out", type=Path, required=True, help="where the JSON report is written")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = resolve_run_config(args.config, args.model, args.mode)
    train_ds, size, first_sample = build_training_dataset(cfg)
    require_the_whole_dataset_is_read(train_ds, size, first_sample)
    corpora = blend_coverage(train_ds)
    report = {
        "config": str(Path(args.config).resolve()),
        "config_text": Path(args.config).read_text(),
        "model": args.model,
        "mode": args.mode,
        "inputs": run_inputs(cfg),
        "dataset_samples": size,
        "first_sample": first_sample,
        "corpora": [asdict(corpus) for corpus in corpora],
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")

    logger.info(f"{args.config}: {size} samples, every one read once")
    for corpus in corpora:
        logger.info(
            f"  {corpus.weight:.6f}  {corpus.samples_drawn:>10} drawn / {corpus.samples_per_pass:>10} per pass"
            f" ({corpus.samples_drawn / corpus.samples_per_pass:.4f} passes)"
            f"  {corpus.documents_reached:>10} / {corpus.documents:>10} documents  {corpus.prefix}"
        )
    logger.info(f"report written to {args.report_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

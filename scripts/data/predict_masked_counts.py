#!/usr/bin/env python3
# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Predict a training run's exact per-iteration token-masking counts from its data, before it trains.

The run's training dataset is built on CPU as its launch builds it (``run_training_data.py``). Every data-parallel
rank's loader is built by the loader's own builder (``build_data_loader``) over that dataset at the sample the
iteration range starts from, and each iteration takes, from every rank's loader, the run's microbatches per replica,
as the training loop does. Each microbatch's labels and loss mask go through the training step's own token-masking
code (``apply_token_masking``, then the stats' report), the reports of an iteration are summed, and the sums are read
as the monitor reads them for its ``[token-masking-counts]`` line (``TokenMaskingCounts.from_sums``). So the
prediction is that line, iteration by iteration, for the same config on the same number of GPUs: the counts are
totals over the global batch, so they do not depend on how a rank splits its microbatch across context-parallel
ranks.

The output is JSON lines: an ``inputs`` record (the config, its sha256, the resolved settings the counts depend on,
the GPUs and data-parallel width, the code revision), then one record per iteration with the six counts. An existing
output is refused.

    python scripts/data/predict_masked_counts.py <config.yaml> --model nano --mode pretrain --gpus 512 \\
        --iterations 1 29881 --out <counts.jsonl>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import torch

from megatron.bridge.data.loaders import build_data_loader
from megatron.bridge.training.token_masking.hook import TokenMaskingCounts, apply_token_masking
from megatron.bridge.training.token_masking.resolution import ResolvedTokenMasking, resolve_token_masking


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.data.run_training_data import (  # noqa: E402
    BIN_IDX_MODES,
    MODELS,
    build_training_dataset,
    microbatches_per_replica,
    resolve_run_config,
    resumed_step,
    run_inputs,
)
from scripts.telemetry.code_revision import code_revision  # noqa: E402


logger: logging.Logger = logging.getLogger(__name__)


def resolved_token_masking(cfg) -> ResolvedTokenMasking:
    """The run's token-masking decision as its setup resolves it, on the CPU; refused when it measures no ids."""
    resolved = resolve_token_masking(
        cfg.token_masking,
        cfg.dataset.tokenizer,
        cfg.tokenizer.tokenizer_type,
        cfg.tokenizer.tokenizer_model,
        torch.device("cpu"),
    )
    if resolved.ids_tensor is None:
        raise ValueError("the config measures no token ids, so its run logs no token-masking counts to predict")
    return resolved


def global_batch_counts(microbatches: list[dict], resolved: ResolvedTokenMasking) -> TokenMaskingCounts:
    """One global batch's counts: each microbatch through the step's masking, the reports summed, read as the monitor
    reads them."""
    sums: dict[str, torch.Tensor] = {}
    for batch in microbatches:
        loss_mask, stats = apply_token_masking(batch["labels"], batch["loss_mask"], resolved)
        report = stats.report(loss_mask, torch.zeros(batch["labels"].shape))
        for key, value in report.items():
            sums[key] = value if key not in sums else sums[key] + value
    return TokenMaskingCounts.from_sums(sums)


def predict(cfg, gpus: int, first: int, last: int) -> Iterator[tuple[int, TokenMaskingCounts]]:
    """The counts the run of ``cfg`` on ``gpus`` GPUs logs at each iteration ``first`` .. ``last`` (as the log numbers
    them: the first iteration of a run from its start is 1, of a run resumed at step s it is s + 1)."""
    step = resumed_step(cfg)
    if not step < first <= last <= cfg.train.train_iters:
        raise ValueError(
            f"iterations {first}..{last} are not inside the run's iterations {step + 1}..{cfg.train.train_iters}"
        )
    if cfg.train.decrease_batch_size_if_needed:
        raise ValueError(
            "train.decrease_batch_size_if_needed trains a smaller batch than the config states; not predicted"
        )
    resolved = resolved_token_masking(cfg)
    data_parallel = cfg.get_data_parallel_size(gpus)
    microbatches = microbatches_per_replica(cfg.train.global_batch_size, cfg.train.micro_batch_size, data_parallel)
    train_ds, _, first_sample = build_training_dataset(cfg)
    consumed = first_sample + (first - 1 - step) * cfg.train.global_batch_size
    loaders = [
        build_data_loader(
            train_ds,
            consumed,
            cfg.dataset.dataloader_type,
            cfg,
            data_parallel_rank=rank,
            data_parallel_size=data_parallel,
            persistent_workers=False,
        )
        for rank in range(data_parallel)
    ]
    # Each loader's own sampler and collation; its samples are read here rather than by its worker processes, which
    # would be data_parallel x num_workers processes for samples that do not depend on which process reads them.
    readers = [(iter(loader.batch_sampler), loader.collate_fn) for loader in loaders]
    for iteration in range(first, last + 1):
        batches = [
            collate([train_ds[index] for index in next(sampler)])
            for sampler, collate in readers
            for _ in range(microbatches)
        ]
        yield iteration, global_batch_counts(batches, resolved)


def main(argv: list[str] | None = None) -> int:
    """Predict the counts of an iteration range and write them as JSON lines."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="the run's override YAML, exactly as the launch passes it")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--mode", required=True, choices=BIN_IDX_MODES)
    parser.add_argument("--gpus", type=int, required=True, help="the GPUs the run trains on")
    parser.add_argument("--iterations", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"))
    parser.add_argument("--out", type=Path, required=True, help="the JSON-lines file to write; must not exist")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.out.exists():
        raise FileExistsError(f"{args.out} exists; a prediction is never overwritten")

    cfg = resolve_run_config(args.config, args.model, args.mode)
    first, last = args.iterations
    inputs = {
        "config": str(Path(args.config).resolve()),
        "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        "model": args.model,
        "mode": args.mode,
        "gpus": args.gpus,
        "data_parallel_size": cfg.get_data_parallel_size(args.gpus),
        "iterations": [first, last],
        "token_masking": {"enabled": cfg.token_masking.enabled, "measured": cfg.token_masking.measured_token_ids},
        "run": run_inputs(cfg),
        "code_revision": code_revision(REPO_ROOT),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as out:
        out.write(json.dumps({"inputs": inputs}) + "\n")
        for iteration, counts in predict(cfg, args.gpus, first, last):
            out.write(json.dumps({"iteration": iteration, **asdict(counts)}) + "\n")
            out.flush()
            if iteration == first or iteration % 100 == 0 or iteration == last:
                logger.info(f"iteration {iteration}: {counts.log_fields()}")
    logger.info(f"prediction written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

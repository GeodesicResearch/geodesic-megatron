#!/usr/bin/env python3
# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Score every window of a tokenised set with an HF export, and write the loss of each window.

A pass is one export on one set of windows. A set has one of these kinds, named in the config:

- ``corpus``: a ``.bin/.idx`` corpus, whose windows are the ones the trainer's own dataset class
  (``megatron.bridge`` ``pretrain_train_valid_test_datasets_provider``) builds for a validation split that holds every
  document: the same shuffle, seed and sequence length, every window once, the input ids of a window and the labels that
  follow them. The loader settings and the tokenizer are read from the resolved ``run_config.yaml`` of the model's save,
  so they are the run's. The loss is taken on every token, as in the run (``eod_mask_loss`` false).
- ``array``: a saved ``.npy`` of ids, ``[windows, ids]``. A window's input is its first ``ids - 1`` ids and its targets
  are the ids after the first, so each window has one target fewer than a corpus window.

The export is scored in the dtype that the config names (``hf_dtype``) on one GPU with the output head applied in fp32 (``hf_model_outputs.fp32_head_outputs``),
and the loss of a window is the sum of an unfused fp32 cross-entropy over its targets. The tool refuses to score unless
the sha256 of the export's ``export_manifest.json``, of each corpus file and of each array equals the pin in the config.

Usage (inside the container, from the repo root, with one GPU):

    python scripts/checkpoint/window_scorer.py --config <config.yaml> --passes <name>[,<name>...]

The config holds these keys, all required: ``output_dir``, ``hf_dtype``, ``max_pass_seconds``, ``models`` (name to
``export``, ``export_manifest_sha256`` and ``run_config``), ``sets`` (name to ``kind`` and the corpus ``prefix`` with
``bin_sha256`` and ``idx_sha256``, or the array ``path`` with ``sha256``) and ``passes`` (name to ``model`` and ``set``).

Outputs, in ``output_dir``, never replacing a file: ``<pass>.windows.tsv`` (the set, the model, the window index, the
targets and the summed loss of each window) and ``<pass>.summary.json`` (the mean loss per target, the targets and
windows, the SLURM job, the hashes the pass rests on, the config it ran with, the code revision and the versions of torch
and transformers).

Catalogue:
    EXPORT_MANIFEST: the name of the manifest file that an export's pin is taken from.
    LOADER_KEYS: the loader settings that a corpus's windows depend on, compared with the run's.
    WindowScorerError: a pin, a config or a pass is not what it should be.
    ModelPin: an export, its pinned manifest hash and the run config of its save.
    WindowSet: a set of windows and the hashes of its files.
    Pass: one model on one set.
    ScorerConfig: the config of a run.
    load_config: read and validate a config file.
    check_pinned: refuse a file whose sha256 is not the pinned one.
    corpus_windows: the input ids and labels of a corpus, as the trainer's dataset class builds them.
    array_windows: the input ids and targets of a saved array.
    window_losses: the summed cross-entropy of each window under a model.
    run_pass: score one pass and write its two files.
    parse_args: the arguments of the command line.
    main: the command line.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# Run as a script, only scripts/checkpoint/ is on sys.path; the repo root makes the sibling packages importable.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from scripts.data.blend_samples import samples_in  # noqa: E402
from scripts.file_digest import file_sha256  # noqa: E402
from scripts.mapping_keys import require_keys  # noqa: E402
from scripts.mapping_values import require_absolute_path, require_number, require_sha256, require_string  # noqa: E402
from scripts.output_files import refuse_existing  # noqa: E402
from scripts.telemetry.code_revision import code_revision  # noqa: E402


LOGGER = logging.getLogger(__name__)
EXPORT_MANIFEST = "export_manifest.json"
# The loader settings that decide which windows a corpus gives, read from the run's resolved dataset block.
LOADER_KEYS = (
    "sequence_length",
    "random_seed",
    "reset_position_ids",
    "reset_attention_mask",
    "eod_mask_loss",
    "add_extra_token_to_sequence",
    "mid_level_dataset_surplus",
    "drop_last_partial_validation_sequence",
)


class WindowScorerError(RuntimeError):
    """A pin, a config or a pass is not what it should be."""


@dataclass(frozen=True)
class ModelPin:
    """An export, the sha256 its ``export_manifest.json`` must have, and the resolved ``run_config.yaml`` of its save."""

    export: Path
    manifest_sha256: str
    run_config: Path


@dataclass(frozen=True)
class WindowSet:
    """A set of windows and the sha256 of its files. A ``corpus`` has the ``.bin/.idx`` prefix as its ``location`` and
    ``bin`` and ``idx`` hashes. An ``array`` has the ``.npy`` file as its ``location`` and an ``array`` hash."""

    kind: str
    location: Path
    sha256s: dict[str, str]


@dataclass(frozen=True)
class Pass:
    """One model on one set."""

    model: str
    set: str


@dataclass(frozen=True)
class ScorerConfig:
    """The config of a run: where the scores go, the HF dtype, the longest a pass may run, the models, the sets, the
    passes, and the path and the sha256 of the config file itself."""

    path: Path
    output_dir: Path
    hf_dtype: str
    max_pass_seconds: float
    models: dict[str, ModelPin]
    sets: dict[str, WindowSet]
    passes: dict[str, Pass]
    sha256: str


def check_pinned(path: Path, pinned: str, what: str) -> str:
    """The sha256 of ``path``; raises WindowScorerError when it is not ``pinned``."""
    actual = file_sha256(path)
    if actual != pinned:
        raise WindowScorerError(f"{what} {path} has sha256 {actual}, not the pinned {pinned}")
    return actual


def _window_set(raw: Any, name: str) -> WindowSet:
    where = f"sets.{name}"
    if not isinstance(raw, dict) or raw.get("kind") not in ("corpus", "array"):
        raise WindowScorerError(f"{where}.kind must be corpus or array")
    if raw["kind"] == "corpus":
        entry = require_keys(raw, where, {"kind", "prefix", "bin_sha256", "idx_sha256"}, error=WindowScorerError)
        hashes = {"bin": require_sha256(entry["bin_sha256"], f"{where}.bin_sha256", error=WindowScorerError)}
        hashes["idx"] = require_sha256(entry["idx_sha256"], f"{where}.idx_sha256", error=WindowScorerError)
        prefix = require_absolute_path(entry["prefix"], f"{where}.prefix", error=WindowScorerError)
        return WindowSet("corpus", Path(prefix), hashes)
    entry = require_keys(raw, where, {"kind", "path", "sha256"}, error=WindowScorerError)
    array = require_absolute_path(entry["path"], f"{where}.path", error=WindowScorerError)
    return WindowSet(
        "array", Path(array), {"array": require_sha256(entry["sha256"], f"{where}.sha256", error=WindowScorerError)}
    )


def load_config(path: str | Path) -> ScorerConfig:
    """Read and validate the config at ``path``; raises WindowScorerError for a missing key, an unknown model or set
    in a pass, or a pin that is not a full sha256."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    keys = {"output_dir", "hf_dtype", "max_pass_seconds", "models", "sets", "passes"}
    top = require_keys(raw, "the config", keys, error=WindowScorerError)
    models = {}
    for name, raw_model in top["models"].items():
        where = f"models.{name}"
        entry = require_keys(
            raw_model, where, {"export", "export_manifest_sha256", "run_config"}, error=WindowScorerError
        )
        pin = require_sha256(
            entry["export_manifest_sha256"], f"{where}.export_manifest_sha256", error=WindowScorerError
        )
        export = require_absolute_path(entry["export"], f"{where}.export", error=WindowScorerError)
        run_config = require_absolute_path(entry["run_config"], f"{where}.run_config", error=WindowScorerError)
        models[name] = ModelPin(Path(export), pin, Path(run_config))
    sets = {name: _window_set(entry, name) for name, entry in top["sets"].items()}
    passes = {}
    for name, entry in top["passes"].items():
        entry = require_keys(entry, f"passes.{name}", {"model", "set"}, error=WindowScorerError)
        passes[name] = Pass(model=entry["model"], set=entry["set"])
        if entry["model"] not in models or entry["set"] not in sets:
            raise WindowScorerError(f"passes.{name} names a model or a set that the config does not define")
    return ScorerConfig(
        path,
        Path(require_absolute_path(top["output_dir"], "output_dir", error=WindowScorerError)),
        require_string(top["hf_dtype"], "hf_dtype", error=WindowScorerError),
        require_number(top["max_pass_seconds"], "max_pass_seconds", 0, inclusive=False, error=WindowScorerError),
        models,
        sets,
        passes,
        file_sha256(path),
    )


def corpus_windows(prefix: Path, run_config: dict[str, Any], cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(input ids, labels)``, each ``[windows, sequence_length]``, of the corpus at ``prefix``, as the trainer's dataset
    class builds a validation split that holds every document.

    ``bin_idx_dataset_config``, the function that builds a run's dataset config, is given the run's sequence length and
    seed from ``run_config`` (the resolved config of a run) and a validation split that holds every document. It
    raises WindowScorerError when a loader setting (``LOADER_KEYS``) of the result differs from the run's. The
    tokenizer comes from ``run_config``. The dataset class writes its index to ``cache_dir`` (a few kilobytes), which a
    corpus must not share with another.
    """
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    from megatron.bridge.data.utils import pretrain_train_valid_test_datasets_provider
    from megatron.bridge.training.config import TokenizerConfig
    from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
    from pipeline.training.datasets import bin_idx_dataset_config

    dataset = run_config["dataset"]
    config = bin_idx_dataset_config(
        {
            "data_path": [str(prefix)],
            "seq_length": dataset["sequence_length"],
            "seed": dataset["random_seed"],
            "split": "0,1,0",
            "path_to_cache": str(cache_dir),
        },
        "pretrain",
    )
    differing = {
        key: (getattr(config, key), dataset[key]) for key in LOADER_KEYS if getattr(config, key) != dataset[key]
    }
    if differing:
        raise WindowScorerError(f"the loader settings differ from the run's (built, run): {differing}")
    tokenizer = run_config["tokenizer"]
    config.tokenizer = build_tokenizer(
        TokenizerConfig(tokenizer_type=tokenizer["tokenizer_type"], tokenizer_model=tokenizer["tokenizer_model"])
    )
    config.finalize()
    indexed = IndexedDataset(str(prefix), mmap=True)  # kept alive while its memory-mapped lengths are summed
    windows = samples_in(int(indexed.sequence_lengths.sum()), config.sequence_length)
    if windows == 0:
        raise WindowScorerError(f"{prefix} holds too few tokens for one window of {config.sequence_length}")
    _, valid, _ = pretrain_train_valid_test_datasets_provider([0, windows, 0], config)
    if valid is None or len(valid) != windows:
        raise WindowScorerError(
            f"the dataset class built {None if valid is None else len(valid)} windows, not {windows}"
        )
    samples = [valid[i] for i in range(len(valid))]
    return (
        np.stack([sample["tokens"].numpy() for sample in samples]),
        np.stack([sample["labels"].numpy() for sample in samples]),
    )


def array_windows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(input ids, targets)`` of the ``[windows, ids]`` array at ``path``: each window's first ``ids - 1`` ids, and the
    ids after its first."""
    ids = np.load(path)
    if ids.ndim != 2 or ids.shape[1] < 2:
        raise WindowScorerError(
            f"{path} must hold a [windows, ids] array with at least two ids, not shape {ids.shape}"
        )
    return ids[:, :-1].astype(np.int64), ids[:, 1:].astype(np.int64)


def window_losses(
    window_logits: Callable[[list[int]], Any],
    inputs: np.ndarray,
    targets: np.ndarray,
    stop_at: float,
) -> np.ndarray:
    """The summed fp32 cross-entropy of each window, from ``window_logits`` (input ids to fp32 logits
    ``[positions, vocabulary]``). Raises WindowScorerError when the clock (``time.monotonic``) passes ``stop_at``."""
    import torch

    losses = np.zeros(len(inputs), dtype=np.float64)
    for index, (window, target) in enumerate(zip(inputs, targets)):
        if time.monotonic() > stop_at:
            raise WindowScorerError(f"the pass ran past its limit after {index} of {len(inputs)} windows")
        logits = window_logits(window.tolist())
        losses[index] = torch.nn.functional.cross_entropy(
            logits, torch.as_tensor(target, device=logits.device), reduction="sum"
        ).item()
    return losses


def run_pass(config: ScorerConfig, name: str, model: Any, repo_dir: Path) -> dict[str, Any]:
    """Score pass ``name`` with the loaded HF ``model`` of its pin, write ``<name>.windows.tsv`` and
    ``<name>.summary.json`` in the config's output folder, and return the summary.

    Raises WindowScorerError for a pin that does not hold or a pass past ``max_pass_seconds``, and FileExistsError when an
    output exists. Reads the corpus or array, the model's run config and its export manifest, the ``SLURM_JOB_ID``
    environment variable and the git revision of ``repo_dir`` (``code_revision``). Writes the two outputs and, for a
    corpus, the dataset index in ``index_cache`` of the output folder.
    """
    import torch
    import transformers
    from scripts.checkpoint import hf_model_outputs

    one = config.passes[name]
    pin, window_set = config.models[one.model], config.sets[one.set]
    windows_path, summary_path = (config.output_dir / f"{name}.{suffix}" for suffix in ("windows.tsv", "summary.json"))
    refuse_existing(windows_path, summary_path)
    hashes = {
        "export_manifest": check_pinned(pin.export / EXPORT_MANIFEST, pin.manifest_sha256, "the export manifest")
    }
    if window_set.kind == "corpus":
        for suffix in ("bin", "idx"):
            path = Path(f"{window_set.location}.{suffix}")
            hashes[suffix] = check_pinned(path, window_set.sha256s[suffix], f"the corpus {suffix}")
        run_config = yaml.safe_load(pin.run_config.read_text())
        inputs, targets = corpus_windows(window_set.location, run_config, config.output_dir / "index_cache" / one.set)
        hashes["run_config"] = file_sha256(pin.run_config)
    else:
        hashes["array"] = check_pinned(window_set.location, window_set.sha256s["array"], "the array")
        inputs, targets = array_windows(window_set.location)
    started = time.monotonic()
    losses = window_losses(
        lambda ids: hf_model_outputs.fp32_head_outputs(model, ids), inputs, targets, started + config.max_pass_seconds
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    count = targets.shape[1]
    windows_path.write_text(
        "".join(f"{one.set}\t{one.model}\t{index}\t{count}\t{loss!r}\n" for index, loss in enumerate(losses))
    )
    summary = {
        "pass": name,
        "model": one.model,
        "set": one.set,
        "windows": len(losses),
        "targets": int(targets.size),
        "mean_loss_per_target": float(losses.sum() / targets.size),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "sha256": hashes | {"scorer_config": config.sha256},
        "config": {
            "path": str(config.path),
            "hf_dtype": config.hf_dtype,
            "max_pass_seconds": config.max_pass_seconds,
            "export": str(pin.export),
            "run_config": str(pin.run_config),
            "set_kind": window_set.kind,
            "set_location": str(window_set.location),
        },
        "code_revision": code_revision(str(repo_dir)),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "score_seconds": round(time.monotonic() - started, 1),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.info(
        "%s: %d windows, %d targets, mean loss %.5f",
        name,
        summary["windows"],
        summary["targets"],
        summary["mean_loss_per_target"],
    )
    return summary


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """The command line's ``--config`` and ``--passes``, from ``argv`` (``sys.argv`` when None)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="the scorer config (YAML)")
    parser.add_argument("--passes", required=True, help="the names of the passes to run, in order, comma-separated")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load the model of the first named pass and run the passes in order; every pass must use that model."""
    from scripts.checkpoint import hf_model_outputs

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    config = load_config(args.config)
    names = args.passes.split(",")
    unknown = [name for name in names if name not in config.passes]
    models = {config.passes[name].model for name in names if name in config.passes}
    if unknown or len(models) != 1:
        raise WindowScorerError(
            f"the passes must be defined in the config and share one model: {unknown or sorted(models)}"
        )
    model = hf_model_outputs.load_model(str(config.models[models.pop()].export), config.hf_dtype)
    repo_dir = Path(__file__).resolve().parents[2]
    for name in names:
        run_pass(config, name, model, repo_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

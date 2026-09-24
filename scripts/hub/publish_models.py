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
"""Publish a campaign's checkpoints to the Hub as models: one revision per checkpoint, with a model
card that records the tokens seen and the training loss at each and, per stage, the data mix and
schedule it was trained under.

A campaign's manifest (e.g. ``configs/control_pretraining/hub_models.yaml``) names the repositories
and, for each, the training stages that feed it; a stage is its training config, from which the save
directory, ``train_iters``, the W&B run name and the card's training facts (sequence length,
global batch, learning-rate schedule, tokenizer, data blend) are read. Every completed checkpoint of a stage
(at or below the directory's tracker, so never a save in progress) becomes a revision named by
the stage's pattern; the designated stage's final checkpoint is also the default revision.

A checkpoint is published in four steps, each skipped when its result already exists, so a pass
is idempotent and a run can be repeated as new checkpoints land:

1. An export clone: a directory of symlinks to the checkpoint's files plus a patched copy of
   ``run_config.yaml``. The exporter rebuilds the model from that file, and training serialised
   a closure it cannot import (``_apply_moe_experts_impl.<locals>...``); the two edits below make
   it importable. The training tree is never written to.
2. The HF export, by ``pipeline_checkpoint_convert.sh export`` at the manifest's parallelism, into
   the clone. The exporter needs a SLURM environment with GPUs: the ``export`` phase gives it this
   allocation's, and the ``submit`` phase gives it one of its own by queueing a single-node job per
   checkpoint, which is what keeps exports from competing with whatever else holds these cards.
3. Verification: every tensor the safetensors index promises is in the shard it names, and every
   tensor a shard holds is in the index — by tensor name, never by file count.
4. The upload, to the revision (and to ``main`` for the default), followed by the model card on
   ``main`` and the repository's membership of the collection. A manifest with an ``upload`` block
   moves this step into a job as well: the ``rolling`` phase then submits the manifest's one-node
   job that runs this tool's ``upload`` phase, and the polling process writes nothing to the Hub.

Losses come from W&B (the manifest's loss key at the iteration's step) across every run that
carried the stage's name, since a stage runs as a chain of segments; a checkpoint whose iteration
W&B never logged is listed with no loss rather than an invented one. Beyond its tables and the
per-stage facts read from the stage configs, everything a card says comes from the manifest's
``card`` block.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_TOOL_DIR = Path(__file__).resolve().parent
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))
sync_bucket = importlib.import_module("sync_bucket")
ManifestError = sync_bucket.ManifestError

LOGGER = logging.getLogger("publish_models")

RUN_CONFIG = "run_config.yaml"
HF_DIR = "hf"
INDEX_FILE = "model.safetensors.index.json"
# The exporter's last write: it copies the checkpoint's run_config into the export after the shards,
# the index and the tokenizer fixups, so an export without it was cut short, however complete its
# tensors look. Every export clone carries a run_config, so a finished export always has one.
EXPORT_COMPLETE_FILE = "megatron_run_config.yaml"
README_NAME = "README.md"
EXPORTER = "pipeline_checkpoint_convert.sh"

# Submitting work as jobs of its own instead of running it inline: the submitter carrying this
# cluster's bad-node exclusions and quota report, the sbatch wrapper around the exporter, and the
# one that runs a pass of this tool (for uploads).
SUBMITTER = "isambard_sbatch"
EXPORT_SBATCH = "pipeline_checkpoint_submit.sbatch"
UPLOAD_SBATCH = "scripts/hub/publish_models.sbatch"
# Where those wrappers' jobs write their output, relative to the directory they are submitted from
# (#SBATCH --output=logs/slurm/...), named as their headers name it. SLURM does not create the
# directory, and a job whose output file cannot be opened fails before it starts, so every
# submission creates it first.
SLURM_LOG_DIR = Path("logs") / "slurm"
EXPORT_JOB_LOG = "convert-checkpoint-{job}.out"
UPLOAD_JOB_LOG = "publish-models-{job}.out"
MAIN = "main"

# The exporter rebuilds the Megatron model from the checkpoint's run_config.yaml. A checkpoint
# trained with moe_experts_impl torch_grouped records the stack spec as a nested closure, which
# cannot be imported; the two edits point it back at the module-level spec and at the expert
# implementation whose parameters the bridge's export globs match. The weights on disk are
# canonical either way (GroupedExperts.sharded_state_dict writes canonical keys).
RUN_CONFIG_EDITS = (
    (
        "megatron.bridge.models.mamba.mamba_provider.MambaModelProvider._apply_moe_experts_impl.<locals>._grouped_resolved_stack_spec",
        "megatron.bridge.models.mamba.mamba_provider.get_default_mamba_stack_spec",
    ),
    ("moe_experts_impl: torch_grouped", "moe_experts_impl: te_grouped"),
)

MANIFEST_KEYS = frozenset(
    {
        "collection",
        "architecture",
        "export_root",
        "log_dir",
        "hf_home",
        "export",
        "wandb",
        "card",
        "models",
    }
)
COLLECTION_KEYS = frozenset({"title", "description", "private"})
# The Hub rejects a longer collection description ("Too big: expected string to have <=150
# characters"), and it does so only when the collection is created, at the end of a pass.
COLLECTION_DESCRIPTION_MAX_CHARS = 150
# Present only when uploads run as jobs of their own; without it the polling process uploads.
OPTIONAL_MANIFEST_KEYS = frozenset({"upload"})
EXPORT_KEYS = frozenset({"tp", "ep", "nodes", "walltime"})
UPLOAD_KEYS = frozenset({"walltime"})
WANDB_KEYS = frozenset({"entity", "project", "loss_key"})
CARD_KEYS = frozenset(
    {"license", "license_name", "tags", "reasoning_tag", "intro", "provenance", "base_note", "think_note"}
)
MODEL_KEYS = frozenset({"repo", "private", "reasoning", "strict", "description", "history", "stages"})
STAGE_KEYS = frozenset({"name", "config", "revision", "default", "extra_directories"})
ITERATION_FIELD = "{iteration}"


class ExportError(RuntimeError):
    """An export did not produce what the Hub should receive."""


@dataclass(frozen=True)
class Collection:
    """The Hub collection every published repository joins."""

    title: str
    description: str
    private: bool


@dataclass(frozen=True)
class ExportJob:
    """How one checkpoint export runs: the allocation it asks for when it is submitted as a
    job of its own, and how the model is sharded across that allocation's GPUs."""

    tp: int
    ep: int
    nodes: int
    walltime: str


@dataclass(frozen=True)
class UploadJob:
    """How a manifest's uploads run when they run as a job of their own: one job for the whole
    manifest, on a single node (the work is a network transfer, and this partition does not share
    nodes), for this long."""

    walltime: str


@dataclass(frozen=True)
class WandbSource:
    """Where the training losses are read from: the W&B project and the key that carries them."""

    entity: str
    project: str
    loss_key: str


@dataclass(frozen=True)
class Card:
    """What every model card says beyond its tables."""

    license: str
    license_name: str
    tags: tuple[str, ...]
    reasoning_tag: str
    intro: str
    provenance: str
    base_note: str
    think_note: str


@dataclass(frozen=True)
class Corpus:
    """One corpus of a stage's data mix: the Hub dataset it was tokenized from, its subset (empty
    for a whole dataset), its blend weight as the config states it, and how many files carry it."""

    dataset: str
    subset: str
    weight: float
    files: int


@dataclass(frozen=True)
class Training:
    """What a stage trained on and how, read from its config: the tokenizer, sequence length and
    global batch, the learning-rate schedule, and the data mix."""

    tokenizer: str
    seq_length: int
    global_batch_size: int
    lr: float
    min_lr: float
    lr_decay_style: str
    warmup: str
    corpora: tuple[Corpus, ...]


@dataclass(frozen=True)
class Stage:
    """One training stage feeding a model: its config's facts and how its checkpoints are named.

    ``tokens_before`` is the token count of every stage behind this one in the model's curriculum,
    each counted at its own sequence length and global batch."""

    name: str
    config: Path
    save: Path
    train_iters: int
    wandb_exp_name: str
    training: Training
    revision: str
    default: bool
    extra_directories: tuple[Path, ...]
    tokens_before: int

    @property
    def tokens_per_iteration(self) -> int:
        """The tokens one iteration of this stage trains on: its sequence length times its batch."""
        return self.training.seq_length * self.training.global_batch_size


@dataclass(frozen=True)
class Model:
    """One Hub repository: its export posture, the stages behind it, and the stages it publishes."""

    repo: str
    private: bool
    reasoning: bool
    strict: bool
    description: str
    history: tuple[Stage, ...]
    stages: tuple[Stage, ...]


@dataclass(frozen=True)
class Manifest:
    """The publication's definition: the collection, the architecture, and every model in it.
    ``source`` is the manifest file itself, which an upload job is pointed back at; ``upload`` is
    None when the polling process uploads."""

    source: Path
    collection: Collection
    architecture: str
    export_root: Path
    log_dir: Path
    hf_home: Path
    export: ExportJob
    upload: UploadJob | None
    wandb: WandbSource
    card: Card
    models: tuple[Model, ...]


@dataclass(frozen=True)
class Publication:
    """One checkpoint on its way to one revision."""

    model: Model
    stage: Stage
    iteration: int
    source: Path
    clone_root: Path
    revision: str
    default: bool
    tokens_seen: int

    @property
    def clone(self) -> Path:
        return self.clone_root / self.source.name

    @property
    def hf_dir(self) -> Path:
        return self.clone / HF_DIR

    @property
    def export_job_record(self) -> Path:
        """Where a submitting pass records the id of the job it queued for this export, beside the
        clone (never inside it, where the exporter reads the checkpoint)."""
        return self.clone_root / f"export_job_{self.source.name}.txt"

    @property
    def targets(self) -> list[str]:
        """The Hub revisions this publication is uploaded to: its own, and main for the default."""
        return [self.revision] + ([MAIN] if self.default else [])

    @property
    def upload_job_record(self) -> Path:
        """Where a rolling pass records the id of the upload job it made responsible for this
        publication, beside the export's record."""
        return self.clone_root / f"upload_job_{self.source.name}.txt"

    @property
    def label(self) -> str:
        return f"{self.model.repo}@{self.revision}"


def dataset_from_slug(slug: str, config: Path) -> tuple[str, str]:
    """Undo the data pipeline's directory slug: ``org__name[__subset]`` to (``org/name``, subset).

    A corpus directory the pipeline did not name (no ``org__name`` in it) is refused: the card would
    otherwise print a bare directory name where every other row names a Hub dataset."""
    parts = slug.split("__")
    if len(parts) < 2:
        raise ManifestError(f"{config}: corpus directory {slug!r} is not a <org>__<dataset>[__<subset>] slug")
    return f"{parts[0]}/{parts[1]}", "__".join(parts[2:])


def corpora_of(cfg: dict, config: Path) -> tuple[Corpus, ...]:
    """The data mix a stage config declares: either ``dataset.data_path``, a blend of Megatron
    ``.bin/.idx`` prefixes as weight, prefix pairs (a corpus split into ``shard<n>`` directories is
    one corpus whose weights are summed), or ``dataset.dataset_root``, one prepared Hub dataset."""
    dataset = cfg.get("dataset") or {}
    blend = dataset.get("data_path")
    root = dataset.get("dataset_root")
    if blend is not None:
        weights: dict[tuple[str, str], list[float]] = {}
        for weight, prefix in sync_bucket.blend_pairs(blend, config):
            corpus_root = sync_bucket.corpus_relative(prefix.parent).parts[0]
            weights.setdefault(dataset_from_slug(corpus_root, config), []).append(weight)
        return tuple(Corpus(dataset=d, subset=s, weight=sum(w), files=len(w)) for (d, s), w in weights.items())
    if root is not None:
        name, subset = dataset_from_slug(Path(str(root)).name, config)
        return (Corpus(dataset=name, subset=subset, weight=1.0, files=1),)
    raise ManifestError(f"{config}: dataset must declare data_path (a .bin/.idx blend) or dataset_root")


def stage_training(cfg: dict, config: Path) -> Training:
    """The tokenizer, batch geometry, learning-rate schedule and data mix a stage config declares."""

    def required(section: str, key: str) -> Any:
        value = (cfg.get(section) or {}).get(key)
        if value is None:
            raise ManifestError(f"{config}: {section}.{key} must be set")
        return value

    scheduler = cfg.get("scheduler") or {}
    warmup_iters = int(required("scheduler", "lr_warmup_iters"))
    warmup_fraction = scheduler.get("lr_warmup_fraction")
    if warmup_iters:
        warmup = f"{warmup_iters:,} iterations"
    elif warmup_fraction:
        warmup = f"{float(warmup_fraction):.0%} of the stage"
    else:
        warmup = "none"
    # A WSD schedule's shape is its decay branch's style; the card names both.
    decay_style = str(required("scheduler", "lr_decay_style"))
    if decay_style == "WSD" and scheduler.get("lr_wsd_decay_style"):
        decay_style = f"WSD ({scheduler['lr_wsd_decay_style']})"
    return Training(
        tokenizer=str(required("tokenizer", "tokenizer_model")),
        seq_length=int(required("dataset", "seq_length")),
        global_batch_size=int(required("train", "global_batch_size")),
        lr=float(required("optimizer", "lr")),
        min_lr=float(required("optimizer", "min_lr")),
        lr_decay_style=decay_style,
        warmup=warmup,
        corpora=corpora_of(cfg, config),
    )


def stage_facts(config: Path) -> tuple[Path, int, str, Training]:
    """The save directory, train_iters, W&B run name and training facts a stage config declares."""
    save = sync_bucket.stage_save_directory(config)
    cfg = yaml.safe_load(config.read_text())
    train_iters = (cfg.get("train") or {}).get("train_iters")
    exp_name = (cfg.get("logger") or {}).get("wandb_exp_name")
    if not isinstance(train_iters, int) or train_iters <= 0:
        raise ManifestError(f"{config}: train.train_iters must be a positive integer, got {train_iters!r}")
    if not isinstance(exp_name, str) or not exp_name:
        raise ManifestError(f"{config}: logger.wandb_exp_name must be set")
    return save, train_iters, exp_name, stage_training(cfg, config)


def _stage(raw: Any, repo_root: Path, where: str, tokens_before: int) -> Stage:
    item = sync_bucket.exact_keys(raw, STAGE_KEYS, where)
    config = repo_root / str(item["config"])
    if not config.is_file():
        raise ManifestError(f"{where}: config {config} does not exist")
    revision = str(item["revision"])
    if ITERATION_FIELD not in revision:
        raise ManifestError(f"{where}: revision pattern {revision!r} must contain {ITERATION_FIELD}")
    if any("/" in str(d) for d in item["extra_directories"]):
        raise ManifestError(f"{where}: extra_directories are bare names of directories beside the stage's save dir")
    save, train_iters, exp_name, training = stage_facts(config)
    return Stage(
        name=str(item["name"]),
        config=config,
        save=save,
        train_iters=train_iters,
        wandb_exp_name=exp_name,
        training=training,
        revision=revision,
        default=bool(item["default"]),
        extra_directories=tuple(save.parent / str(d) for d in item["extra_directories"]),
        tokens_before=tokens_before,
    )


def _history_stage(config_path: str, repo_root: Path, where: str, tokens_before: int) -> Stage:
    config = repo_root / config_path
    if not config.is_file():
        raise ManifestError(f"{where}: history config {config} does not exist")
    save, train_iters, exp_name, training = stage_facts(config)
    return Stage(
        name=config.stem,
        config=config,
        save=save,
        train_iters=train_iters,
        wandb_exp_name=exp_name,
        training=training,
        revision="",
        default=False,
        extra_directories=(),
        tokens_before=tokens_before,
    )


def _model(raw: Any, repo_root: Path, where: str) -> Model:
    item = sync_bucket.exact_keys(raw, MODEL_KEYS, where)
    repo = str(item["repo"])
    if repo.count("/") != 1:
        raise ManifestError(f"{where}: repo must be <namespace>/<name>, got {repo!r}")
    tokens_before = 0
    history = []
    for h_index, config_path in enumerate(item["history"]):
        stage = _history_stage(str(config_path), repo_root, f"{where}.history[{h_index}]", tokens_before)
        tokens_before += stage.train_iters * stage.tokens_per_iteration
        history.append(stage)
    stages = []
    for s_index, raw_stage in enumerate(item["stages"]):
        stage = _stage(raw_stage, repo_root, f"{where}.stages[{s_index}]", tokens_before)
        tokens_before += stage.train_iters * stage.tokens_per_iteration
        stages.append(stage)
    if not stages:
        raise ManifestError(f"{where}: a model needs at least one stage")
    if sum(1 for s in stages if s.default) != 1:
        raise ManifestError(f"{where}: exactly one stage must be the default (its final checkpoint is main)")
    return Model(
        repo=repo,
        private=bool(item["private"]),
        reasoning=bool(item["reasoning"]),
        strict=bool(item["strict"]),
        description=str(item["description"]).strip(),
        history=tuple(history),
        stages=tuple(stages),
    )


def slurm_walltime(value: Any, where: str) -> str:
    """A walltime bound for sbatch, which must have stayed a string: YAML reads an unquoted
    00:30:00 as a sexagesimal number of seconds."""
    if not re.fullmatch(r"\d+(-\d\d)?:\d\d:\d\d", str(value)):
        raise ManifestError(
            f"{where} must be a quoted SLURM time such as '00:30:00' -- YAML reads an unquoted one as a "
            "sexagesimal number of seconds"
        )
    return str(value)


def load_manifest(path: Path, repo_root: Path) -> Manifest:
    """Read and validate the manifest; repo-relative config paths resolve against ``repo_root``."""
    document = yaml.safe_load(path.read_text())
    optional = {k: document[k] for k in OPTIONAL_MANIFEST_KEYS if isinstance(document, dict) and k in document}
    required = {k: v for k, v in document.items() if k not in optional} if isinstance(document, dict) else document
    raw = sync_bucket.exact_keys(required, MANIFEST_KEYS, str(path))
    collection = sync_bucket.exact_keys(raw["collection"], COLLECTION_KEYS, f"{path}: collection")
    export = sync_bucket.exact_keys(raw["export"], EXPORT_KEYS, f"{path}: export")
    wandb_raw = sync_bucket.exact_keys(raw["wandb"], WANDB_KEYS, f"{path}: wandb")
    card = sync_bucket.exact_keys(raw["card"], CARD_KEYS, f"{path}: card")
    if not all(isinstance(export[k], int) and export[k] > 0 for k in ("tp", "ep", "nodes")):
        raise ManifestError(f"{path}: export.tp, export.ep and export.nodes must be positive integers")
    export_walltime = slurm_walltime(export["walltime"], f"{path}: export.walltime")
    upload = None
    if "upload" in optional:
        upload_raw = sync_bucket.exact_keys(optional["upload"], UPLOAD_KEYS, f"{path}: upload")
        upload = UploadJob(walltime=slurm_walltime(upload_raw["walltime"], f"{path}: upload.walltime"))
    if not isinstance(card["tags"], list) or not card["tags"]:
        raise ManifestError(f"{path}: card.tags must be a non-empty list")
    description = str(collection["description"]).strip()
    if len(description) > COLLECTION_DESCRIPTION_MAX_CHARS:
        raise ManifestError(
            f"{path}: collection.description is {len(description)} characters; the Hub allows at most "
            f"{COLLECTION_DESCRIPTION_MAX_CHARS}"
        )
    models = tuple(
        _model(raw_model, repo_root, f"{path}: models[{index}]") for index, raw_model in enumerate(raw["models"])
    )
    return Manifest(
        source=path.resolve(),
        collection=Collection(
            title=str(collection["title"]),
            description=description,
            private=bool(collection["private"]),
        ),
        architecture=str(raw["architecture"]),
        export_root=Path(raw["export_root"]),
        log_dir=Path(raw["log_dir"]),
        hf_home=Path(raw["hf_home"]),
        export=ExportJob(tp=export["tp"], ep=export["ep"], nodes=export["nodes"], walltime=export_walltime),
        upload=upload,
        wandb=WandbSource(
            entity=str(wandb_raw["entity"]), project=str(wandb_raw["project"]), loss_key=str(wandb_raw["loss_key"])
        ),
        card=Card(
            license=str(card["license"]),
            license_name=str(card["license_name"]),
            tags=tuple(str(t) for t in card["tags"]),
            reasoning_tag=str(card["reasoning_tag"]),
            intro=str(card["intro"]).strip(),
            provenance=str(card["provenance"]).strip(),
            base_note=str(card["base_note"]).strip(),
            think_note=str(card["think_note"]).strip(),
        ),
        models=models,
    )


def stage_sources(stage: Stage) -> list[tuple[int, Path]]:
    """Every completed checkpoint of a stage, oldest first: the run's own directory plus any extra
    directory beside it that keeps a save the run pruned. The run's directory wins a tie."""
    found: dict[int, Path] = {}
    for directory in (*stage.extra_directories, stage.save):
        if not directory.is_dir():
            if directory is stage.save:
                LOGGER.warning(
                    "%s: %s does not exist yet (stage not started); nothing to publish", stage.name, directory
                )
                continue
            raise ManifestError(f"{stage.name}: extra directory {directory} does not exist")
        for iter_dir in sync_bucket.completed_iterations(directory):
            found[sync_bucket.iteration_number(iter_dir)] = iter_dir
    return sorted(found.items())


def plan(manifest: Manifest, repo_filter: tuple[str, ...] = (), newest_first: bool = False) -> list[Publication]:
    """Every checkpoint that belongs on the Hub, in publication order.

    ``newest_first`` reverses each stage's iterations so a backlog reaches the Hub with its
    most recent checkpoint first. It changes only the order work is attempted in: the model
    card sorts its own rows, so the published table stays chronological either way."""
    publications = []
    for model in manifest.models:
        if repo_filter and not any(f in model.repo for f in repo_filter):
            continue
        repo_dir = manifest.export_root / model.repo.split("/")[1]
        for stage in model.stages:
            sources = stage_sources(stage)
            for iteration, source in reversed(sources) if newest_first else sources:
                publications.append(
                    Publication(
                        model=model,
                        stage=stage,
                        iteration=iteration,
                        source=source,
                        clone_root=repo_dir / stage.name,
                        revision=stage.revision.replace(ITERATION_FIELD, str(iteration)),
                        default=stage.default and iteration == stage.train_iters,
                        tokens_seen=stage.tokens_before + iteration * stage.tokens_per_iteration,
                    )
                )
    return publications


def patch_run_config(text: str) -> str:
    """Apply the export edits to a run_config, or accept one that already carries them."""
    for old, new in RUN_CONFIG_EDITS:
        if text.count(old) == 1:
            text = text.replace(old, new)
        elif text.count(new) >= 1 and old not in text:
            continue
        else:
            raise ExportError(f"run_config has {text.count(old)} occurrences of {old!r}; expected exactly one")
    return text


def make_export_clone(source: Path, clone: Path) -> None:
    """A directory the exporter can read as a checkpoint: symlinks to every file of the source
    iteration except run_config.yaml, which is copied with the export edits applied, and a tracker
    in the clone's parent naming this iteration. Any hf/ export beside the source is not linked."""
    clone.mkdir(parents=True, exist_ok=True)
    for entry in source.iterdir():
        if entry.name == HF_DIR:
            continue
        target = clone / entry.name
        if entry.name == RUN_CONFIG:
            target.write_text(patch_run_config(entry.read_text()))
            continue
        if target.is_symlink() or target.exists():
            if target.is_symlink() and target.resolve() == entry.resolve():
                continue
            raise ExportError(f"{target} exists and is not a link to {entry}")
        target.symlink_to(entry.resolve())
    if not (clone / RUN_CONFIG).is_file():
        raise ExportError(f"{source} has no {RUN_CONFIG}; the exporter cannot rebuild the model without it")
    (clone.parent / sync_bucket.LATEST_FILE).write_text(f"{sync_bucket.iteration_number(source)}\n")


def export_arguments(publication: Publication, manifest: Manifest) -> list[str]:
    """What a publication's export is, independent of who runs it: the same arguments go to the
    exporter directly and to the sbatch wrapper that runs it in a job of its own."""
    arguments = [
        "export",
        str(publication.clone_root),
        "--hf-model",
        manifest.architecture,
        "--iteration",
        str(publication.iteration),
        "--tp",
        str(manifest.export.tp),
        "--ep",
        str(manifest.export.ep),
        "--reasoning" if publication.model.reasoning else "--no-reasoning",
    ]
    if not publication.model.strict:
        arguments.append("--not-strict")
    return arguments


def export_command(publication: Publication, manifest: Manifest) -> list[str]:
    """The exporter invocation for a publication (run from the repo root)."""
    return ["bash", EXPORTER, *export_arguments(publication, manifest)]


def export_job_name(publication: Publication) -> str:
    """The submitted export's job name, which is also how a later pass recognises work it has
    already queued. Two arms publish the same revision names, so the repository is part of it."""
    return f"hubexport-{publication.model.repo.split('/')[1]}-{publication.revision}"


def queued_job_names() -> set[str]:
    """Every job name this user currently has queued or running.

    A failed squeue must not read as an empty queue: that would resubmit work already in flight,
    so a non-zero exit is an error rather than an absence.
    """
    result = subprocess.run(
        ["squeue", "--me", "--noheader", "--format=%j"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise ExportError(f"squeue exited {result.returncode}: {result.stderr.strip()}")
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def upload_job_name(manifest: Manifest) -> str:
    """The manifest's upload job name, which is also how a later pass recognises it in the queue.
    One per manifest, named for the campaign directory the manifest sits in: the job uploads every
    verified export the manifest's repositories are missing. Two jobs for one manifest would race to
    create its collection."""
    return f"hubupload-{manifest.source.parent.name}"


def submission_env(repo_root: Path) -> dict[str, str]:
    """What every submission adds to the environment. ISAMBARD_SBATCH_FORCE is the sanctioned
    posture for a launcher that submits more than a handful of jobs (a wave is one job per
    checkpoint, each a single node for minutes); GEODESIC_REPO_DIR points the job at this checkout."""
    return {"GEODESIC_REPO_DIR": str(repo_root), "ISAMBARD_SBATCH_FORCE": "1"}


def shell_submission(command: list[str], repo_root: Path) -> str:
    """``command`` as a line a person can paste into a shell to submit it exactly as a pass would."""
    assignments = " ".join(f"{name}={shlex.quote(value)}" for name, value in submission_env(repo_root).items())
    return f"cd {shlex.quote(str(repo_root))} && {assignments} {shlex.join(command)}"


def submit_job(command: list[str], record: Path, label: str, repo_root: Path) -> str:
    """Submit one job from ``repo_root``, record its id in ``record``, and return the id. The
    caller has read the queue and found no job of this name, since each pass submits only what is
    not already in flight."""
    env = dict(os.environ, **submission_env(repo_root))
    (repo_root / SLURM_LOG_DIR).mkdir(parents=True, exist_ok=True)
    LOGGER.info("submitting %s: %s", label, " ".join(command))
    result = subprocess.run(command, cwd=repo_root, env=env, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ExportError(f"{SUBMITTER} exited {result.returncode} for {label}: {result.stderr}")
    found = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not found:
        raise ExportError(f"{label}: no job id in submission output: {result.stdout.strip()}")
    record.write_text(f"{found.group(1)}\n")
    return found.group(1)


def submit_export(publication: Publication, manifest: Manifest, repo_root: Path) -> str:
    """Submit one export as its own job and return its job id.

    The job runs the same exporter against the same clone as the inline path, on an allocation of
    its own -- so it neither takes the GPUs of whatever allocation the publisher runs in nor waits
    for them, and a wave of exports runs in parallel instead of one at a time.
    """
    command = [
        SUBMITTER,
        f"--nodes={manifest.export.nodes}",
        f"--time={manifest.export.walltime}",
        f"--job-name={export_job_name(publication)}",
        EXPORT_SBATCH,
        *export_arguments(publication, manifest),
    ]
    return submit_job(command, publication.export_job_record, publication.label, repo_root)


def upload_command(manifest: Manifest, repo_root: Path) -> list[str]:
    """The submission of the manifest's upload job, run from ``repo_root``.

    The job is one pass of this tool's ``upload`` phase over the whole manifest, under the
    interpreter the polling process runs with (it carries huggingface_hub and W&B) and against the
    same manifest and checkout, so it publishes every verified export the repositories are missing,
    then their cards and collection membership, exactly as an upload pass run by hand would. It is a
    SLURM singleton: a second job of its name -- a submission that timed out after registering, a
    resubmission pasted twice -- is held until the first has left the queue, instead of racing it to
    create the collection."""
    if manifest.upload is None:
        raise ExportError(f"{manifest.source}: the manifest has no upload block, so uploads are not jobs")
    return [
        SUBMITTER,
        "--nodes=1",
        f"--time={manifest.upload.walltime}",
        f"--job-name={upload_job_name(manifest)}",
        "--dependency=singleton",
        UPLOAD_SBATCH,
        sys.executable,
        "--manifest",
        str(manifest.source),
        "--repo-root",
        str(repo_root),
        "--phase",
        "upload",
    ]


def submit_upload(publication: Publication, manifest: Manifest, repo_root: Path) -> str:
    """Submit the manifest's upload job and return its job id, recorded against ``publication``."""
    return submit_job(upload_command(manifest, repo_root), publication.upload_job_record, publication.label, repo_root)


def check_no_failed_job(record: Path, work: str, outcome: str, log_name: str, retry: str) -> None:
    """Refuse to resubmit work whose earlier job has left the queue without finishing it.

    Called only for work that is not queued. If a job was recorded for it, that job ended with the
    work still undone, which is a failure to report -- resubmitting it on every poll would hide it.
    ``outcome`` is what the job left behind, and ``retry`` tells a person how to ask for another
    attempt; it ends the message, so a command in it can be copied to the end of the line."""
    if record.is_file():
        job = record.read_text().strip()
        raise ExportError(
            f"{work} job {job} left the queue without finishing: {outcome}; see "
            f"{SLURM_LOG_DIR / log_name.format(job=job)} in the submitting checkout, then {retry}"
        )


def claim_upload_records(manifest: Manifest, job: str) -> None:
    """Record ``job`` against every publication that an earlier upload job left a record for. The
    upload job does this as it starts, so that if it fails as well, the report names it and its log
    rather than the job before it."""
    for publication in plan(manifest):
        if publication.upload_job_record.is_file():
            publication.upload_job_record.write_text(f"{job}\n")


def run_export(publication: Publication, manifest: Manifest, repo_root: Path, log_path: Path) -> None:
    """Export one clone with the repo's exporter; its output is appended to ``log_path``."""
    env = dict(os.environ, GEODESIC_REPO_DIR=str(repo_root))
    command = export_command(publication, manifest)
    LOGGER.info("exporting %s: %s", publication.label, " ".join(command))
    with log_path.open("a") as log:
        log.write(f"\n=== {sync_bucket.utc_now()} {publication.label}: {' '.join(command)}\n")
        log.flush()
        result = subprocess.run(command, cwd=repo_root, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise ExportError(f"exporter exited {result.returncode} for {publication.label}; see {log_path}")


def safetensors_tensor_names(path: Path) -> set[str]:
    """The tensor names a safetensors file holds, from its header (an 8-byte length then JSON). A
    length longer than the file is a corrupt header, refused before it is read."""
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        if length > path.stat().st_size - 8:
            raise ValueError(f"{path}: header length {length} exceeds the file")
        header = json.loads(handle.read(length))
    return {name for name in header if name != "__metadata__"}


def verify_export(hf_dir: Path) -> int:
    """Check that an export finished and that its tensors match its index by name in both
    directions, and return how many tensors it holds. An index or shard that cannot be parsed is an
    export that does not verify, whatever files sit beside it. A failure to read the files at all
    (a stale handle, an I/O error) is not a verdict on the export and is raised as it is."""
    index_path = hf_dir / INDEX_FILE
    if not index_path.is_file():
        raise ExportError(f"{hf_dir}: no {INDEX_FILE}")
    if not (hf_dir / EXPORT_COMPLETE_FILE).is_file():
        raise ExportError(f"{hf_dir}: no {EXPORT_COMPLETE_FILE}, the exporter's last write; the export did not finish")
    try:
        weight_map = json.loads(index_path.read_text())["weight_map"]
        by_shard: dict[str, set[str]] = {}
        for name, shard in weight_map.items():
            by_shard.setdefault(shard, set()).add(name)
        for shard, promised in by_shard.items():
            path = hf_dir / shard
            if not path.is_file():
                raise ExportError(f"{hf_dir}: index names {shard}, which is missing")
            held = safetensors_tensor_names(path)
            if promised != held:
                raise ExportError(
                    f"{hf_dir}/{shard}: index promises {len(promised)} tensors, shard holds {len(held)}; "
                    f"missing {sorted(promised - held)[:3]}, extra {sorted(held - promised)[:3]}"
                )
    except (ValueError, KeyError, TypeError, AttributeError, struct.error) as error:
        raise ExportError(f"{hf_dir}: unreadable export: {error!r}") from error
    return len(weight_map)


def export_problem(publication: Publication) -> str | None:
    """Why the clone holds no verified export, or None when it holds one."""
    if not publication.hf_dir.is_dir():
        return f"{publication.hf_dir} does not exist"
    try:
        verify_export(publication.hf_dir)
    except ExportError as error:
        return str(error)
    return None


def export_is_verified(publication: Publication) -> bool:
    """Whether a verified export already sits in the clone."""
    problem = export_problem(publication)
    if problem is not None and publication.hf_dir.is_dir():
        LOGGER.info("%s: export present but not verified: %s", publication.label, problem)
    return problem is None


def check_clone_is_safe_to_rebuild(publication: Publication, export_root: Path) -> None:
    """Refuse a clone that a pass could not safely rebuild, before anything in it is removed. The pass
    removes a rejected export and rewrites the clone's run_config, so the clone must resolve inside
    the export root, and neither its hf/ nor its run_config may be a link: through any of these, the
    removal or the rewrite would reach whatever the link names, a training run's own checkpoint
    among them."""
    if publication.hf_dir.is_symlink():
        raise ExportError(f"{publication.hf_dir} is a symlink; an export clone's hf/ is exporter output")
    run_config = publication.clone / RUN_CONFIG
    if run_config.is_symlink():
        raise ExportError(f"{run_config} is a link; an export clone's run_config is a patched copy")
    resolved = publication.clone.resolve()
    if not resolved.is_relative_to(export_root.resolve()):
        raise ExportError(f"{publication.clone} resolves to {resolved}, outside the export root {export_root}")


def local_files(hf_dir: Path) -> dict[str, int]:
    """The export's files and sizes, as the Hub will hold them."""
    return {p.name: p.stat().st_size for p in hf_dir.iterdir() if p.is_file() and not p.name.startswith(".")}


def holds_its_finishing_files(hf_dir: Path) -> bool:
    """Whether a local export holds its index and the exporter's last write, judged by their presence
    alone: the export may be unfinished or corrupt, but it is one to compare the Hub's copy with."""
    return (hf_dir / INDEX_FILE).is_file() and (hf_dir / EXPORT_COMPLETE_FILE).is_file()


def revision_files(api: Any, repo: str, revision: str) -> dict[str, int] | None:
    """The top-level files a Hub revision holds, with their sizes; None when the repository or the
    revision does not exist, which is what an unpublished revision looks like."""
    from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

    try:
        return {entry.path: entry.size for entry in api.list_repo_tree(repo, revision=revision, recursive=False)}
    except (RepositoryNotFoundError, RevisionNotFoundError):
        return None


def published(api: Any, repo: str, revision: str, hf_dir: Path) -> bool:
    """Whether the revision already holds every file of the export at the same size."""
    remote = revision_files(api, repo, revision)
    return remote is not None and all(remote.get(name) == size for name, size in local_files(hf_dir).items())


def holds_a_finished_export(api: Any, repo: str, revision: str) -> bool:
    """Whether the revision holds a finished export, judged from the Hub alone. An upload is one
    commit carrying the whole export, and every revision this tool creates branches from the
    repository's first commit, which holds no export (see ``upload``), so a revision holding the
    index and the exporter's last write holds the rest of its own export too."""
    remote = revision_files(api, repo, revision)
    return remote is not None and {INDEX_FILE, EXPORT_COMPLETE_FILE} <= remote.keys()


def on_the_hub(api: Any, publication: Publication) -> bool:
    """Whether every revision the publication targets already holds its export. All of them, not
    just its own: a final checkpoint whose upload to main failed would otherwise count as published
    and never reach main. A revision is compared file by file, by name and size, with a local
    export that holds its index and the exporter's last write; nothing is read, so a file gone
    unreadable in the export of a revision long since published stops no pass. Without such an
    export -- removed or emptied to free space, or never finished here -- there is nothing to compare
    against, and the Hub's copy is judged on its own, so a revision safely published is neither
    exported again nor dropped from the card, and one that is not is never taken for it."""
    if not holds_its_finishing_files(publication.hf_dir):
        return all(holds_a_finished_export(api, publication.model.repo, revision) for revision in publication.targets)
    return all(
        published(api, publication.model.repo, revision, publication.hf_dir) for revision in publication.targets
    )


def initial_commit(api: Any, repo: str) -> str:
    """The repository's first commit, which every revision is branched from because it holds no
    export. One that does -- a history squashed into a single commit, a repository copied from
    another -- is refused: a branch made from it would start as a copy of that export and, if its own
    upload failed, pass for published with those weights."""
    commit = api.list_repo_commits(repo)[-1].commit_id
    held = revision_files(api, repo, commit)
    if held is None:
        raise ExportError(f"{repo}: its first commit {commit} cannot be read")
    if {INDEX_FILE, EXPORT_COMPLETE_FILE} & held.keys():
        raise ExportError(
            f"{repo}: its first commit {commit} holds an export (a squashed history?); no revision is "
            "branched from it, since a branch would start as a copy of that export"
        )
    return commit


def upload(api: Any, publication: Publication) -> None:
    """Upload the export to its revision, and to main as well when it is the default.

    A revision is branched from the repository's first commit, never from main: every export of one
    architecture has the same file names and sizes, so a branch that started as a copy of main's
    export would, if its own upload then failed, pass for published with main's weights."""
    api.create_repo(publication.model.repo, private=publication.model.private, exist_ok=True)
    for revision in publication.targets:
        if revision != MAIN:
            api.create_branch(
                publication.model.repo,
                branch=revision,
                revision=initial_commit(api, publication.model.repo),
                exist_ok=True,
            )
        LOGGER.info("uploading %s -> %s@%s", publication.hf_dir, publication.model.repo, revision)
        api.upload_folder(
            folder_path=str(publication.hf_dir),
            repo_id=publication.model.repo,
            revision=revision,
            commit_message=f"{publication.stage.name} iteration {publication.iteration}"
            + (" (final)" if publication.default else ""),
        )


def _run_losses(run: Any, loss_key: str, wanted: set[int]) -> dict[int, float]:
    found = {}
    for row in run.scan_history(keys=["_step", loss_key]):
        step = row.get("_step")
        value = row.get(loss_key)
        if step in wanted and value is not None:
            found[int(step)] = float(value)
    return found


def losses(wandb_api: Any, source: WandbSource, exp_name: str, iterations: set[int], cache: Path) -> dict[int, float]:
    """The training loss at each iteration, from every W&B run named ``exp_name``; cached on disk
    so a pass only asks W&B about iterations it has not seen."""
    known: dict[int, float] = {}
    if cache.is_file():
        known = {int(k): float(v) for k, v in json.loads(cache.read_text()).items()}
    wanted = set(iterations) - set(known)
    if wanted:
        runs = list(wandb_api.runs(f"{source.entity}/{source.project}", filters={"display_name": exp_name}))
        LOGGER.info("%s: %d W&B runs, looking up %d iterations", exp_name, len(runs), len(wanted))
        for run in runs:
            known.update(_run_losses(run, source.loss_key, wanted))
            wanted -= set(known)
            if not wanted:
                break
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({str(k): v for k, v in sorted(known.items())}, indent=1))
    return {i: known[i] for i in iterations if i in known}


def _tokens(count: int) -> str:
    return f"{count:,} ({count / 1e9:.1f}B)"


def render_model_card(manifest: Manifest, model: Model, rows: list[tuple[Publication, float | None]]) -> str:
    """The repository's README: the manifest's card text around the curriculum and revision tables."""
    card = manifest.card
    name = model.repo.split("/")[1]
    tags = list(card.tags) + ([card.reasoning_tag] if model.reasoning else [])
    default_stage = next(s.name for s in model.stages if s.default)
    lines = [
        "---",
        f"license: {card.license}",
        f"license_name: {card.license_name}",
        f"base_model: {manifest.architecture}",
        "language: [en]",
        f"tags: [{', '.join(tags)}]",
        "---",
        "",
        f"# {name}",
        "",
        model.description,
        "",
        card.intro + f" Architecture `{manifest.architecture}`.",
        "",
        "## Curriculum",
        "",
        "| Stage | Iterations | Tokens | W&B run |",
        "|---|---|---|---|",
    ]
    for stage in (*model.history, *model.stages):
        where = "this repository" if stage in model.stages else "published elsewhere"
        lines.append(
            f"| {stage.name} ({where}) | {stage.train_iters:,} | {_tokens(stage.train_iters * stage.tokens_per_iteration)} "
            f"| `{stage.wandb_exp_name}` |"
        )
    lines += [
        "",
        "A stage's tokens per iteration are its sequence length times its global batch (below), so the "
        "token count of a checkpoint is every earlier stage's tokens plus its iteration times its own "
        "stage's tokens per iteration.",
        "",
        "## Data and schedule",
        "",
        "Read from each stage's training config. Shares are the config's blend weights, normalised; a corpus "
        "tokenized in shards counts once, with its shard count under Files.",
    ]
    for stage in (*model.history, *model.stages):
        training = stage.training
        total_weight = sum(corpus.weight for corpus in training.corpora)
        # Under a constant schedule the config's floor is never reached, so it is not reported.
        schedule = (
            f"learning rate {training.lr:.1e} held constant"
            if training.lr_decay_style == "constant"
            else f"learning rate {training.lr:.1e} with `{training.lr_decay_style}` decay to {training.min_lr:.1e}"
        )
        lines += [
            "",
            f"### {stage.name}",
            "",
            f"Sequence length {training.seq_length:,}, global batch {training.global_batch_size:,} sequences "
            f"({training.seq_length * training.global_batch_size:,} tokens per iteration), {schedule}, warmup "
            f"{training.warmup}, tokenizer `{training.tokenizer}`.",
            "",
            "| Corpus | Share | Files |",
            "|---|---|---|",
        ]
        for corpus in training.corpora:
            corpus_name = f"`{corpus.dataset}`" + (f" subset `{corpus.subset}`" if corpus.subset else "")
            lines.append(f"| {corpus_name} | {corpus.weight / total_weight:.1%} | {corpus.files} |")
    lines += [
        "",
        "## Revisions",
        "",
        f"Every completed checkpoint is a revision; `main` is the final checkpoint of the {default_stage} stage. "
        'Load one with `revision="<name>"`. Tokens seen count the whole curriculum up to that checkpoint; the '
        f"training loss is W&B's `{manifest.wandb.loss_key}` at that iteration (blank where the run did not log it).",
        "",
        "| Revision | Stage | Iteration | Tokens seen | Training loss |",
        "|---|---|---|---|---|",
    ]
    for publication, loss in rows:
        loss_cell = f"{loss:.4f}" if loss is not None else ""
        revision = f"`{publication.revision}`" + (" (also `main`)" if publication.default else "")
        lines.append(
            f"| {revision} | {publication.stage.name} | {publication.iteration:,} | {_tokens(publication.tokens_seen)} | {loss_cell} |"
        )
    lines += [
        "",
        "## Provenance",
        "",
        f"Exported from the Megatron torch_dist checkpoints with megatron-bridge at TP{manifest.export.tp}/EP{manifest.export.ep}. "
        + card.provenance,
        "",
        card.think_note if model.reasoning else card.base_note,
        "",
    ]
    return "\n".join(lines)


def upload_model_card(api: Any, model: Model, text: str, staging: Path) -> bool:
    """Upload the card to main unless the copy uploaded last time is identical; returns whether it did.

    The staged copy is written only once the upload has succeeded, so it always says what the Hub
    holds: a copy staged before a failed upload would make every retry skip the card."""
    staging.mkdir(parents=True, exist_ok=True)
    card = staging / README_NAME
    if card.is_file() and card.read_text() == text:
        return False
    api.upload_file(
        path_or_fileobj=text.encode(),
        path_in_repo=README_NAME,
        repo_id=model.repo,
        revision=MAIN,
        commit_message="Model card: revisions, tokens seen, training loss",
    )
    card.write_text(text)
    return True


def ensure_collection(api: Any, collection: Collection, namespace: str, repos: list[str]) -> str:
    """The collection's slug, creating it if absent, with every repo a member."""
    existing = [c for c in api.list_collections(owner=namespace) if c.title == collection.title]
    if existing:
        slug = existing[0].slug
    else:
        slug = api.create_collection(
            title=collection.title, namespace=namespace, description=collection.description, private=collection.private
        ).slug
        LOGGER.info("created collection %s", slug)
    for repo in repos:
        api.add_collection_item(slug, item_id=repo, item_type="model", exists_ok=True)
    return slug


PHASES = ("export", "submit", "upload", "all", "rolling")
# The phases that queue exports as jobs of their own instead of running the exporter here.
SUBMITTING_PHASES = ("submit", "rolling")


def publish_pass(
    manifest: Manifest,
    repo_root: Path,
    api: Any,
    wandb_api: Any,
    run_dir: Path,
    execute: bool,
    repo_filter: tuple[str, ...],
    phase: str,
    newest_first: bool = False,
) -> int:
    """One pass over the manifest: export and upload what is missing, then the cards and the
    collection for every model that has anything published. Returns how many publications
    remain unpublished (0 when the Hub holds everything the manifest asks for).

    ``phase`` splits the work by what it needs.

    ``export`` runs only the exports, which take this allocation's GPUs for a few minutes each, and
    writes nothing to the Hub. ``submit`` does the same work without the GPUs, queueing a job per
    checkpoint instead, and likewise writes nothing to the Hub -- neither revisions nor cards nor
    collection membership, since a pass that has only queued work has confirmed nothing to describe.
    ``upload`` uploads the publications whose export is already verified, and the cards and
    collection that follow from them, touching no GPU. ``all`` exports here and then uploads.
    ``rolling`` is the phase for a run that is still training, repeated with a poll interval: it
    submits what ``submit`` would and uploads what ``upload`` would. In every phase that acts, an
    export whose job is still in the queue is left to its job, neither uploaded nor rebuilt, since
    the job writes it again when it starts. An export whose job has left the queue without
    finishing it does not verify and is reported, with the reason, rather than resubmitted; once an
    export verifies and its job has left the queue, the job's record is discharged. When the
    manifest has an ``upload`` block, ``rolling`` uploads
    nothing itself: when finished exports are waiting it submits the manifest's one upload job (see
    ``upload_command``), records that job against every publication it is responsible for, and
    leaves cards and collection to it. A pass that has written the cards and collection discharges
    the records of what it confirmed, so an upload job that leaves the queue with a record still
    standing -- whether its revisions are missing or its card is -- is reported, with the command
    that resubmits it, rather than resubmitted.

    A publication counts as published only when every revision it targets holds its export, main
    included for the default; without a local export holding its index and completion file, the
    Hub's copy is judged on its own (see ``on_the_hub``). An export that does not verify and is exported again is removed
    first, since the exporter writes into it without clearing it, and only from a clone that lies
    inside the export root.

    The split is what lets the GPUs be borrowed from another workload for exactly the export, or --
    with ``submit`` and ``rolling`` -- not borrowed at all."""
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, not {phase!r}")
    publications = plan(manifest, repo_filter, newest_first)
    export_log = run_dir / "export.log"
    # Every pass that acts reads the queue: an export whose job is still queued will be written when
    # that job starts, so no pass may upload it or rebuild its clone meanwhile.
    queued = queued_job_names() if execute else set()
    uploads_are_jobs = phase == "rolling" and manifest.upload is not None and execute
    upload_queued = uploads_are_jobs and upload_job_name(manifest) in queued
    # The upload job submitted in this pass. Its own pass reads the manifest after it was
    # submitted, so every publication this pass finds verified will be verified when the job looks
    # too: each is its responsibility.
    upload_job: str | None = None
    # A failed submission is not repeated for each later publication of the same pass: one that
    # timed out after registering would otherwise leave several jobs racing for the collection.
    upload_submission_failed = False
    pending = 0
    touched: dict[str, list[Publication]] = {}
    for publication in publications:
        if uploads_are_jobs and not upload_queued:
            # An upload job discharges its records only once its cards and collection are done, so
            # a record whose job has left the queue marks a failed job -- including one that got
            # every revision onto the Hub and then failed on a card.
            try:
                # Deleting the record alone would not retry a job that failed after every revision
                # was published: the rolling pass submits only for revisions still missing. The same
                # job resubmitted by hand publishes what is missing and writes the cards either way.
                resubmit = shell_submission(upload_command(manifest, repo_root), repo_root)
                check_no_failed_job(
                    publication.upload_job_record,
                    "upload",
                    "its record was never discharged, so its revisions, card or collection are not all written",
                    UPLOAD_JOB_LOG,
                    f"resubmit it (a pass that finishes deletes the record) with: {resubmit}",
                )
            except ExportError as error:
                LOGGER.error("%s: %s", publication.label, error)
                pending += 1
                continue
        if on_the_hub(api, publication):
            LOGGER.info("%s: already on the Hub", publication.label)
            if execute:
                publication.export_job_record.unlink(missing_ok=True)
            touched.setdefault(publication.model.repo, []).append(publication)
            continue
        if not execute:
            LOGGER.info(
                "%s: would %s", publication.label, "upload" if export_is_verified(publication) else "export and upload"
            )
            pending += 1
            continue
        try:
            problem = export_problem(publication)
            if problem is not None:
                if phase == "upload":
                    LOGGER.info("%s: no verified export (%s); left for an export pass", publication.label, problem)
                    if publication.upload_job_record.is_file():
                        # An upload job answers for the exports verified when it was submitted. One
                        # gone since is no failure of the job's, and no resubmission of it could clear
                        # the record, which would otherwise hold the export back from rolling forever.
                        LOGGER.warning(
                            "%s: the export an upload job was recorded for is gone; dropping %s so an export "
                            "pass can redo it",
                            publication.label,
                            publication.upload_job_record,
                        )
                        publication.upload_job_record.unlink()
                    pending += 1
                    continue
                if export_job_name(publication) in queued:
                    # Its job may be reading the clone right now; rebuilding the clone would
                    # rewrite the run_config under it.
                    LOGGER.info("%s: export job already queued; left to finish", publication.label)
                    pending += 1
                    continue
                if phase in SUBMITTING_PHASES:
                    check_no_failed_job(
                        publication.export_job_record,
                        "export",
                        problem,
                        EXPORT_JOB_LOG,
                        f"delete {publication.export_job_record} to submit it again",
                    )
                check_clone_is_safe_to_rebuild(publication, manifest.export_root)
                if publication.hf_dir.exists():
                    LOGGER.warning(
                        "%s: removing the unverified export in %s: %s", publication.label, publication.hf_dir, problem
                    )
                    shutil.rmtree(publication.hf_dir)
                make_export_clone(publication.source, publication.clone)
                if phase in SUBMITTING_PHASES:
                    job = submit_export(publication, manifest, repo_root)
                    LOGGER.info("%s: export submitted as job %s", publication.label, job)
                    pending += 1
                    continue
                run_export(publication, manifest, repo_root, export_log)
                count = verify_export(publication.hf_dir)
                LOGGER.info("%s: export verified, %d tensors", publication.label, count)
            if export_job_name(publication) in queued:
                # Verified, but a job still in the queue will write this export again when it starts.
                LOGGER.info("%s: export verified but its job is still in the queue; upload waits", publication.label)
                pending += 1
                continue
            # A verified export whose job has left the queue has settled that job.
            publication.export_job_record.unlink(missing_ok=True)
            if phase in ("export", "submit"):
                # Both GPU-side phases stop here. "submit" must stop too even though it never ran
                # an exporter itself: a publication whose submitted job has since finished arrives
                # here already verified, and falling through would turn a pass that promises only
                # to queue work into one that uploads hundreds of gigabytes.
                LOGGER.info("%s: exported; upload left for an upload pass", publication.label)
                pending += 1
                continue
            if uploads_are_jobs:
                if upload_queued:
                    LOGGER.info("%s: the upload job is in the queue; left to it", publication.label)
                    pending += 1
                    continue
                if upload_job is None:
                    if upload_submission_failed:
                        raise ExportError("left for the upload job, whose submission failed earlier in this pass")
                    try:
                        upload_job = submit_upload(publication, manifest, repo_root)
                    except ExportError:
                        upload_submission_failed = True
                        raise
                    LOGGER.info("%s: upload submitted as job %s", publication.label, upload_job)
                else:
                    publication.upload_job_record.write_text(f"{upload_job}\n")
                    LOGGER.info("%s: left to upload job %s", publication.label, upload_job)
                pending += 1
                continue
            upload(api, publication)
            touched.setdefault(publication.model.repo, []).append(publication)
        except ExportError as error:
            LOGGER.error("%s: %s", publication.label, error)
            pending += 1
    if not execute or phase in ("export", "submit") or uploads_are_jobs:
        return pending
    namespace = manifest.models[0].repo.split("/")[0]
    for model in manifest.models:
        # Only what this pass confirmed on the Hub: a planned revision whose export or upload
        # failed must not appear on the card as if it existed.
        rows_pubs = touched.get(model.repo, [])
        if not rows_pubs:
            continue
        rows = []
        for stage in model.stages:
            stage_pubs = sorted((p for p in rows_pubs if p.stage is stage), key=lambda p: p.iteration)
            cache = run_dir.parent / "losses" / f"{stage.wandb_exp_name}.json"
            found = losses(wandb_api, manifest.wandb, stage.wandb_exp_name, {p.iteration for p in stage_pubs}, cache)
            rows += [(p, found.get(p.iteration)) for p in stage_pubs]
        # Cards persist beside the run directories so an unchanged card is not re-uploaded every pass.
        card_dir = run_dir.parent / "cards" / model.repo.split("/")[1]
        if upload_model_card(api, model, render_model_card(manifest, model, rows), card_dir):
            LOGGER.info("%s: model card updated", model.repo)
    ensure_collection(api, manifest.collection, namespace, [m.repo for m in manifest.models if m.repo in touched])
    # Every publication this pass confirmed is now on the Hub and described by its card and
    # collection, which is all an upload job is for: its record is discharged.
    for confirmed in touched.values():
        for publication in confirmed:
            publication.upload_job_record.unlink(missing_ok=True)
    return pending


def make_wandb_api():
    """The W&B client, imported here so plan mode needs no W&B credentials."""
    import wandb

    return wandb.Api(timeout=120)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """The command line: the manifest, plan-only, a model filter, and polling."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True, help="the publication manifest")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="the checkout the manifest's config paths and the exporter are taken from (default: this one)",
    )
    parser.add_argument(
        "--plan", action="store_true", help="report what would be exported and uploaded; change nothing"
    )
    parser.add_argument("--models", nargs="*", default=[], help="only repos whose id contains one of these substrings")
    parser.add_argument(
        "--phase",
        choices=PHASES,
        default="all",
        help="export: only the exports, on this allocation's GPUs; submit: queue each missing export as "
        "its own job instead, taking no GPU here and writing nothing to the Hub; upload: only uploads of "
        "verified exports, cards and collection (no GPU); all: export here, then upload; rolling: submit "
        "missing exports and upload those whose job has finished, for polling a run still training "
        "(with an upload block in the manifest, the uploads go to the manifest's one upload job instead)",
    )
    parser.add_argument(
        "--newest-first",
        action="store_true",
        help="attempt each stage's newest checkpoint first, so a backlog publishes the most recent "
        "revision soonest (the model card stays in iteration order regardless)",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=None, help="seconds between passes; without it one pass is run"
    )
    parser.add_argument("--stop-after", type=float, default=None, help="hours after which polling stops")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run passes until the Hub holds everything the manifest asks for, or until told to stop."""
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    manifest = load_manifest(args.manifest, repo_root)
    sync_bucket.configure_hf_home(manifest.hf_home)
    run_dir = manifest.log_dir / sync_bucket.utc_now().replace(":", "")
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.manifest, run_dir / "manifest.yaml")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "publish.log")],
    )
    LOGGER.info("manifest %s, run dir %s, execute=%s", args.manifest, run_dir, not args.plan)
    # The manifest's upload job is recognised by the name every pass already finds it by in the queue.
    if not args.plan and args.phase == "upload" and os.environ.get("SLURM_JOB_NAME") == upload_job_name(manifest):
        claim_upload_records(manifest, os.environ["SLURM_JOB_ID"])
        LOGGER.info("upload job %s: took over the records of any upload job before it", os.environ["SLURM_JOB_ID"])
    api = sync_bucket.make_api()
    wandb_api = None if args.plan else make_wandb_api()
    deadline = time.time() + args.stop_after * 3600 if args.stop_after else None
    while True:
        pending = publish_pass(
            manifest,
            repo_root,
            api,
            wandb_api,
            run_dir,
            not args.plan,
            tuple(args.models),
            args.phase,
            args.newest_first,
        )
        LOGGER.info("pass complete: %d publication(s) pending", pending)
        if args.poll_interval is None or (deadline is not None and time.time() >= deadline):
            return 1 if pending else 0
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    sys.exit(main())

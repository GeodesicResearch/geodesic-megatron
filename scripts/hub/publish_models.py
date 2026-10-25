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
global batch, learning-rate schedule, tokenizer, data blend) are read. A stage run as a chain of jobs
names its final job's config there and may name its first job's as ``schedule_config``, whose warmup
the card states. Every completed checkpoint of a stage
(at or below the directory's tracker, so never a save in progress) becomes a revision named by
the stage's pattern; the designated stage's final checkpoint is also the default revision.

A checkpoint is published in four steps, each skipped when its result already exists, so a pass
is idempotent and a run can be repeated as new checkpoints land:

1. An export clone: a directory of symlinks to the checkpoint's files plus a patched copy of
   ``run_config.yaml``. The exporter rebuilds the model from that file, and training serialised
   a closure it cannot import (``_apply_moe_experts_impl.<locals>...``); the two edits of
   ``scripts/checkpoint/export_clone.py``, which the standard exporter applies the same way, make
   it importable. The training tree is never written to.
2. The HF export, by ``pipeline_checkpoint_convert.sh export`` at the manifest's parallelism, into
   the clone. The exporter needs a SLURM environment with GPUs: the ``export`` phase gives it this
   allocation's, and the ``submit`` phase gives it one of its own by queueing a single-node job per
   checkpoint, which is what keeps exports from competing with whatever else holds these cards.
3. Verification: every tensor the safetensors index promises is in the shard it names, and every
   tensor a shard holds is in the index — by tensor name, never by file count.
4. The upload, to the revision (and to ``main`` for the default), followed by the model card on
   ``main`` and the repository's membership of the collection under its note, with the collection's
   description and every note brought to the manifest's. A manifest with an ``upload`` block
   moves this step into a job as well: the ``rolling`` phase then submits the manifest's one-node
   job that runs this tool's ``upload`` phase, and the polling process writes nothing to the Hub.

A card's revision rows are the checkpoints its repository holds, read from the Hub: a published
checkpoint's save and its local export are both removed in time, so neither says what the repository
holds. The ``cards`` phase writes only the cards, which is how a change to their text reaches a
repository whose checkpoints were all published long ago.

Losses come from W&B (the manifest's loss key at the iteration's step) across every run that
carried the stage's name, since a stage runs as a chain of segments; a checkpoint whose iteration
W&B never logged is listed with no loss rather than an invented one. Beyond its tables and the
per-stage facts read from the stage configs, a card says what the manifest's ``card`` block says,
the model's own ``description``, and, where the model names one, the markdown of its
``card_sections`` file, placed after the introduction.

A run exits with its last pass's status: 0 (``EXIT_PUBLISHED``) when the Hub holds everything the
manifest asks for (a ``cards`` pass judges no publication and exits 0 once its cards are written),
1 (``EXIT_FAILED``) when the pass failed on any publication, and 3
(``EXIT_PENDING``) when it met no error but work is still in flight or left for another phase, such
as a ``submit`` pass that queued exports or an ``upload`` pass that found a checkpoint not yet
exported. A failure is logged per publication and the pass carries on with the rest.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


_TOOL_DIR = Path(__file__).resolve().parent
for _directory in (_TOOL_DIR, _TOOL_DIR.parent):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))
sync_bucket = importlib.import_module("sync_bucket")
slurm_jobs = importlib.import_module("slurm_jobs")
ManifestError = sync_bucket.ManifestError
# The repo root, for the modules this tool shares with others (scripts/checkpoint/export_clone.py,
# scripts/mapping_keys.py).
_REPO_ROOT = str(_TOOL_DIR.parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

# The export repair: RUN_CONFIG_EDITS (the two run_config edits a torch_grouped checkpoint needs before
# the exporter can rebuild its model), patch_run_config and make_export_clone. Its ExportError is this
# tool's too: a clone that cannot be built is reported like any other export that cannot be produced.
# RUN_CONFIG_EDITS and patch_run_config are re-exported, as part of this tool's interface, for the
# tests and callers that read them from here. The checkpoint directory's layout (LATEST_FILE,
# iteration_number) is defined there too.
from scripts.checkpoint import export_check_result  # noqa: E402
from scripts.checkpoint.export_clone import (  # noqa: E402
    HF_DIR,
    LATEST_FILE,
    RUN_CONFIG,
    RUN_CONFIG_EDITS,  # noqa: F401
    ExportError,
    iteration_number,
    make_export_clone,
    patch_run_config,  # noqa: F401
)
from scripts.checkpoint.safetensors_index import INDEX_FILE, checked_weight_map  # noqa: E402
from scripts.mapping_keys import require_keys  # noqa: E402


LOGGER = logging.getLogger("publish_models")

# The exporter's last write: it copies the checkpoint's run_config into the export after the shards,
# the index and the tokenizer fixups, so an export without it was cut short, however complete its
# tensors look. Every export clone carries a run_config, so a finished export always has one.
EXPORT_COMPLETE_FILE = "megatron_run_config.yaml"
# What a finished export holds whatever else it holds: its index and the exporter's last write.
FINISHING_FILES = frozenset({INDEX_FILE, EXPORT_COMPLETE_FILE})
README_NAME = "README.md"
EXPORTER = "pipeline_checkpoint_convert.sh"

# Submitting work as jobs of its own instead of running it inline: the submitter carrying this
# cluster's bad-node exclusions and quota report, the sbatch wrapper around the exporter, and the
# one that runs a pass of this tool (for uploads).
SUBMITTER = "isambard_sbatch"
EXPORT_SBATCH = "pipeline_checkpoint_submit.sbatch"
UPLOAD_SBATCH = "scripts/hub/publish_models.sbatch"
# The wrapper around the export check (scripts/checkpoint/export_check.py), a job per export of a model whose manifest
# entry names one.
CHECK_SBATCH = "scripts/checkpoint/export_check.sbatch"
# The names those wrappers' jobs write their output under, in slurm_jobs.SLURM_LOG_DIR, as their headers name them.
EXPORT_JOB_LOG = "convert-checkpoint-{job}.out"
UPLOAD_JOB_LOG = "publish-models-{job}.out"
CHECK_JOB_LOG = "export-check-{job}.out"
MAIN = "main"
# The word that opens a line of a model's verified log.
VERIFIED_TAG = "verified"
# The export-check verdicts a release log records. One that could not be evaluated is no verdict on the export.
RECORDED_VERDICTS = ("PASS", "FAIL")

MANIFEST_KEYS = frozenset(
    {
        "collection",
        "architecture",
        "base_model",
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
# characters"), and it does so only when the description is written, at the end of a pass.
COLLECTION_DESCRIPTION_MAX_CHARS = 150
# The Hub's limit on the note shown under a collection item.
COLLECTION_NOTE_MAX_CHARS = 500
# Present only when uploads run as jobs of their own; without it the polling process uploads.
OPTIONAL_MANIFEST_KEYS = frozenset({"upload"})
EXPORT_KEYS = frozenset({"tp", "ep", "nodes", "walltime"})
UPLOAD_KEYS = frozenset({"walltime"})
WANDB_KEYS = frozenset({"entity", "project", "loss_key"})
CARD_KEYS = frozenset(
    {"license", "license_name", "tags", "reasoning_tag", "intro", "provenance", "base_note", "think_note"}
)
MODEL_KEYS = frozenset({"repo", "private", "reasoning", "strict", "description", "history", "stages"})
# A model may carry a collection note; one without keeps whatever note the Hub already shows. A model whose card
# carries sections beyond the standard ones names them as `card_sections`: a repo-relative markdown file that the
# card places after its introduction.
OPTIONAL_MODEL_KEYS = frozenset({"note", "card_sections", "release"})
# A model whose evaluation chain releases a revision on what this tool records names the log of verified revisions
# as `release.verified_log`, and may name an export check every export must pass: its spec, the root its jobs write
# under, the log of its verdicts, the word that opens each of that log's lines, and the job's walltime.
RELEASE_KEYS = frozenset({"verified_log"})
OPTIONAL_RELEASE_KEYS = frozenset({"export_check"})
EXPORT_CHECK_KEYS = frozenset({"spec", "out_root", "log", "tag", "walltime"})
STAGE_KEYS = frozenset({"name", "config", "revision", "default", "extra_directories"})
# A stage run as a chain of jobs, each resuming the one before, names its final job's config as
# `config` and may name its first job's as `schedule_config`: the card's warmup is the first job's,
# since every later job resumes it and warms up for none.
OPTIONAL_STAGE_KEYS = frozenset({"schedule_config"})
ITERATION_FIELD = "{iteration}"


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
class ExportCheck:
    """The check every export of a model must pass before the revision is released
    (``scripts/checkpoint/export_check.py``): its spec, the root its jobs write their outputs under, the log its
    verdicts are recorded in with the word that opens each of that log's lines, and the job's walltime."""

    spec: Path
    out_root: Path
    log: Path
    tag: str
    walltime: str


@dataclass(frozen=True)
class Release:
    """What a model's evaluation chain releases a revision on: the log of verified revisions, and the export check
    whose verdicts are logged beside it, or None when the model has none."""

    verified_log: Path
    export_check: ExportCheck | None


@dataclass(frozen=True)
class Model:
    """One Hub repository: its export posture, the stages behind it, and the stages it publishes.

    ``note`` is the one line shown under the repository in the collection, which is what tells
    repositories with near-identical names apart there; None leaves the Hub's note as it is.
    ``card_sections`` is the markdown its card places after the introduction, read from the file the
    manifest names, and None when the manifest names none. ``release`` is the logs an evaluation chain
    releases the model's revisions on, and None for a model nothing waits on in that way."""

    repo: str
    private: bool
    reasoning: bool
    strict: bool
    description: str
    note: str | None
    card_sections: str | None
    release: Release | None
    history: tuple[Stage, ...]
    stages: tuple[Stage, ...]


@dataclass(frozen=True)
class Manifest:
    """The publication's definition: the collection, the architecture, the parent every card names, and
    every model in it. ``source`` is the manifest file itself, which an upload job is pointed back at;
    ``upload`` is None when the polling process uploads; ``base_model`` is the Hub repository the cards
    name as their ``base_model``, the parent the Hub's model tree shows, and None when they name none."""

    source: Path
    collection: Collection
    architecture: str
    base_model: str | None
    export_root: Path
    log_dir: Path
    hf_home: Path
    export: ExportJob
    upload: UploadJob | None
    wandb: WandbSource
    card: Card
    models: tuple[Model, ...]


@dataclass(frozen=True)
class Checkpoint:
    """One checkpoint of a stage, as the Hub names it and a model card describes it."""

    stage: Stage
    iteration: int

    @property
    def revision(self) -> str:
        return self.stage.revision.replace(ITERATION_FIELD, str(self.iteration))

    @property
    def default(self) -> bool:
        """Whether this is the final checkpoint of the model's default stage, which main carries too."""
        return self.stage.default and self.iteration == self.stage.train_iters

    @property
    def tokens_seen(self) -> int:
        """The tokens of the whole curriculum up to this checkpoint."""
        return self.stage.tokens_before + self.iteration * self.stage.tokens_per_iteration

    @property
    def targets(self) -> list[str]:
        """The Hub revisions that hold this checkpoint once published: its own, and main for the default."""
        return [self.revision] + ([MAIN] if self.default else [])


@dataclass(frozen=True)
class Publication:
    """One checkpoint on its way to the Hub from a save on disk."""

    model: Model
    checkpoint: Checkpoint
    source: Path
    clone_root: Path

    @property
    def stage(self) -> Stage:
        return self.checkpoint.stage

    @property
    def iteration(self) -> int:
        return self.checkpoint.iteration

    @property
    def revision(self) -> str:
        return self.checkpoint.revision

    @property
    def default(self) -> bool:
        return self.checkpoint.default

    @property
    def targets(self) -> list[str]:
        return self.checkpoint.targets

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
    def inline_export_record(self) -> Path:
        """Where an inline export (the export or all phase, which has no job in the queue) records
        itself, beside the job record, from before it clears the clone until its export verifies."""
        return self.clone_root / f"export_inline_{self.source.name}.txt"

    @property
    def upload_job_record(self) -> Path:
        """Where a rolling pass records the id of the upload job it made responsible for this
        publication, beside the export's record."""
        return self.clone_root / f"upload_job_{self.source.name}.txt"

    @property
    def check_job_record(self) -> Path:
        """Where a submitting pass records the id of the export-check job it queued for this export, beside the
        export's own record."""
        return self.clone_root / f"check_job_{self.source.name}.txt"

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


def first_job_warmup(training: Training, save: Path, schedule_config: Path, where: str) -> Training:
    """A chained stage's training facts, carrying the warmup its first job ran.

    The first job must be the same run as the final one: it writes the same save directory and states
    the same training facts in every respect but its warmup, so every other row of the card stays
    the final job's and cannot be taken from a config of some other run.
    """
    if not schedule_config.is_file():
        raise ManifestError(f"{where}: schedule_config {schedule_config} does not exist")
    first_save, _, _, first = stage_facts(schedule_config)
    if first_save != save:
        raise ManifestError(f"{where}: schedule_config {schedule_config} writes {first_save}, not the stage's {save}")
    if replace(first, warmup=training.warmup) != training:
        raise ManifestError(
            f"{where}: schedule_config {schedule_config} differs from the stage's config beyond warmup"
        )
    return replace(training, warmup=first.warmup)


def _stage(raw: Any, repo_root: Path, where: str, tokens_before: int) -> Stage:
    required, optional = _split_optional(raw, OPTIONAL_STAGE_KEYS)
    item = require_keys(required, where, STAGE_KEYS, error=ManifestError)
    config = repo_root / str(item["config"])
    if not config.is_file():
        raise ManifestError(f"{where}: config {config} does not exist")
    revision = str(item["revision"])
    if ITERATION_FIELD not in revision:
        raise ManifestError(f"{where}: revision pattern {revision!r} must contain {ITERATION_FIELD}")
    if any("/" in str(d) for d in item["extra_directories"]):
        raise ManifestError(f"{where}: extra_directories are bare names of directories beside the stage's save dir")
    save, train_iters, exp_name, training = stage_facts(config)
    if "schedule_config" in optional:
        training = first_job_warmup(training, save, repo_root / str(optional["schedule_config"]), where)
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


def _split_optional(mapping: Any, optional_keys: frozenset[str]) -> tuple[Any, dict[str, Any]]:
    """Split a manifest mapping into its required part, for ``require_keys``, and the optional keys it has.
    Anything that is not a mapping is returned whole, for ``require_keys`` to reject."""
    if not isinstance(mapping, dict):
        return mapping, {}
    optional = {k: mapping[k] for k in optional_keys if k in mapping}
    return {k: v for k, v in mapping.items() if k not in optional}, optional


def _card_sections(path: str, repo_root: Path, where: str) -> str:
    """The markdown a card places after its introduction. It must open with a level-2 heading, so that
    it is a section of the card rather than a continuation of the introduction's paragraph."""
    sections = repo_root / path
    if not sections.is_file():
        raise ManifestError(f"{where}: {sections} does not exist")
    text = sections.read_text().strip()
    if not text.startswith("## "):
        raise ManifestError(f"{where}: {sections} must open with a level-2 heading ('## ...')")
    return text


def _release(raw: Any, repo_root: Path, where: str) -> Release:
    """A model's ``release`` entry. The export check's spec is a repo-relative file that must exist; its tag is one
    word, and never the verified log's, so a line says by its first word which log it belongs to."""
    required, optional = _split_optional(raw, OPTIONAL_RELEASE_KEYS)
    item = require_keys(required, where, RELEASE_KEYS, error=ManifestError)
    check = None
    if "export_check" in optional:
        raw_check = require_keys(
            optional["export_check"], f"{where}.export_check", EXPORT_CHECK_KEYS, error=ManifestError
        )
        spec = repo_root / str(raw_check["spec"])
        if not spec.is_file():
            raise ManifestError(f"{where}.export_check.spec: {spec} does not exist")
        tag = str(raw_check["tag"])
        if not re.fullmatch(r"\S+", tag) or tag == VERIFIED_TAG:
            raise ManifestError(f"{where}.export_check.tag must be one word other than {VERIFIED_TAG!r}, not {tag!r}")
        check = ExportCheck(
            spec=spec,
            out_root=Path(raw_check["out_root"]),
            log=Path(raw_check["log"]),
            tag=tag,
            walltime=slurm_walltime(raw_check["walltime"], f"{where}.export_check.walltime"),
        )
    return Release(verified_log=Path(item["verified_log"]), export_check=check)


def _model(raw: Any, repo_root: Path, where: str) -> Model:
    required, optional = _split_optional(raw, OPTIONAL_MODEL_KEYS)
    item = require_keys(required, where, MODEL_KEYS, error=ManifestError)
    card_sections = (
        _card_sections(str(optional["card_sections"]), repo_root, f"{where}.card_sections")
        if "card_sections" in optional
        else None
    )
    repo = sync_bucket.hub_id(str(item["repo"]), f"{where}: repo")
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
    note = optional.get("note")
    if "note" in optional and not isinstance(note, str):
        raise ManifestError(f"{where}: note must be a line of text, not {note!r}; omit the key for no note")
    if note is not None:
        note = note.strip()
    if note is not None and not 0 < len(note) <= COLLECTION_NOTE_MAX_CHARS:
        raise ManifestError(
            f"{where}: note is {len(note)} characters; a collection note needs 1 to {COLLECTION_NOTE_MAX_CHARS}"
        )
    return Model(
        repo=repo,
        private=bool(item["private"]),
        reasoning=bool(item["reasoning"]),
        strict=bool(item["strict"]),
        description=str(item["description"]).strip(),
        note=note,
        card_sections=card_sections,
        release=_release(optional["release"], repo_root, f"{where}.release") if "release" in optional else None,
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
    required, optional = _split_optional(document, OPTIONAL_MANIFEST_KEYS)
    raw = require_keys(required, str(path), MANIFEST_KEYS, error=ManifestError)
    collection = require_keys(raw["collection"], f"{path}: collection", COLLECTION_KEYS, error=ManifestError)
    export = require_keys(raw["export"], f"{path}: export", EXPORT_KEYS, error=ManifestError)
    wandb_raw = require_keys(raw["wandb"], f"{path}: wandb", WANDB_KEYS, error=ManifestError)
    card = require_keys(raw["card"], f"{path}: card", CARD_KEYS, error=ManifestError)
    if not all(isinstance(export[k], int) and export[k] > 0 for k in ("tp", "ep", "nodes")):
        raise ManifestError(f"{path}: export.tp, export.ep and export.nodes must be positive integers")
    export_walltime = slurm_walltime(export["walltime"], f"{path}: export.walltime")
    upload = None
    if "upload" in optional:
        upload_raw = require_keys(optional["upload"], f"{path}: upload", UPLOAD_KEYS, error=ManifestError)
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
    base_model = raw["base_model"]
    if base_model is not None:
        if not isinstance(base_model, str):
            raise ManifestError(f"{path}: base_model must be a Hub repository or null, not {base_model!r}")
        sync_bucket.hub_id(base_model, f"{path}: base_model")
    return Manifest(
        source=path.resolve(),
        collection=Collection(
            title=str(collection["title"]),
            description=description,
            private=bool(collection["private"]),
        ),
        architecture=str(raw["architecture"]),
        base_model=base_model,
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
    directory beside it that keeps a save the run pruned. The run's directory wins a tie. A run's
    directory that is not on disk contributes nothing: the stage has not started, or its saves were
    removed once published, which may happen while this plan reads them. An extra directory is
    named by hand in the manifest, so one that is not on disk is refused."""
    found: dict[int, Path] = {}
    for directory in (*stage.extra_directories, stage.save):
        try:
            completed = sync_bucket.completed_iterations(directory)
        except sync_bucket.CheckpointDirectoryMissing as error:
            if directory is not stage.save:
                raise ManifestError(f"{stage.name}: extra directory {directory} does not exist") from error
            LOGGER.warning(
                "%s: %s does not exist (stage not started, or its saves removed); nothing to publish from it",
                stage.name,
                directory,
            )
            continue
        for iter_dir in completed:
            found[iteration_number(iter_dir)] = iter_dir
    return sorted(found.items())


def selected_models(manifest: Manifest, repo_filter: tuple[str, ...]) -> list[Model]:
    """The manifest's models whose repository id contains one of the filter's substrings; all of
    them when the filter is empty."""
    return [m for m in manifest.models if not repo_filter or any(f in m.repo for f in repo_filter)]


def plan(manifest: Manifest, repo_filter: tuple[str, ...] = (), newest_first: bool = False) -> list[Publication]:
    """Every checkpoint that belongs on the Hub, in publication order.

    ``newest_first`` reverses each stage's iterations so a backlog reaches the Hub with its
    most recent checkpoint first. It changes only the order work is attempted in: the model
    card sorts its own rows, so the published table stays chronological either way."""
    publications = []
    for model in selected_models(manifest, repo_filter):
        repo_dir = manifest.export_root / model.repo.split("/")[1]
        for stage in model.stages:
            sources = stage_sources(stage)
            for iteration, source in reversed(sources) if newest_first else sources:
                publications.append(
                    Publication(
                        model=model,
                        checkpoint=Checkpoint(stage, iteration),
                        source=source,
                        clone_root=repo_dir / stage.name,
                    )
                )
    return publications


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
    """Every job name this user currently has queued or running (``slurm_jobs.queued_job_names``),
    with an unreadable queue reported as an ExportError, which a pass reports and counts."""
    try:
        return slurm_jobs.queued_job_names()
    except slurm_jobs.SlurmError as error:
        raise ExportError(str(error)) from error


def upload_job_name(manifest: Manifest) -> str:
    """The manifest's upload job name, which is also how a later pass recognises it in the queue.
    One per manifest, named for the campaign directory the manifest sits in: the job uploads every
    verified export the manifest's repositories are missing. Two jobs for one manifest would race to
    create its collection."""
    return f"hubupload-{manifest.source.parent.name}"


def submit_job(command: list[str], record: Path, label: str, repo_root: Path) -> str:
    """Submit one job from ``repo_root``, forced (``slurm_jobs.forced_submission_env``: every job is a single node
    for minutes, and a wave is one per checkpoint), record its id in ``record``, and return the id. The caller has
    read the queue and found no job of this name, since each pass submits only what is not already in flight."""
    (repo_root / slurm_jobs.SLURM_LOG_DIR).mkdir(parents=True, exist_ok=True)
    LOGGER.info("submitting %s: %s", label, " ".join(command))
    try:
        job_id = slurm_jobs.submit(command, repo_root, slurm_jobs.forced_submission_env(repo_root))
    except slurm_jobs.SlurmError as error:
        raise ExportError(f"{label}: {error}") from error
    record.write_text(f"{job_id}\n")
    return job_id


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
        # A second job for the same export (two passes that both read an empty queue) waits for the
        # first instead of writing the same hf/ alongside it.
        "--dependency=singleton",
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


def export_check_job_name(publication: Publication) -> str:
    """The submitted export check's job name, which is also how a later pass recognises it in the queue."""
    return f"hubcheck-{publication.model.repo.split('/')[1]}-{publication.revision}"


def export_check_dir(publication: Publication, check: ExportCheck) -> Path:
    """Where the publication's export check writes its scores and its result: a directory of its own."""
    return check.out_root / publication.model.repo.split("/")[1] / publication.revision


def submit_export_check(publication: Publication, check: ExportCheck, repo_root: Path) -> str:
    """Submit the export check of one export against the checkpoint it was made from, as its own job, and return
    its job id. The job reads the export and the checkpoint and writes only into its own output directory."""
    command = [
        SUBMITTER,
        f"--time={check.walltime}",
        f"--job-name={export_check_job_name(publication)}",
        "--dependency=singleton",
        CHECK_SBATCH,
        str(check.spec),
        str(publication.source),
        str(publication.hf_dir),
        str(export_check_dir(publication, check)),
    ]
    return submit_job(command, publication.check_job_record, publication.label, repo_root)


def submit_export_check_if_due(publication: Publication, queued: set[str], repo_root: Path) -> set[str]:
    """Submit the export check of a verified export unless its model has none, it has written its result, its job
    is in the queue, or a job was already recorded for it: one that ended without a result is reported where the
    verdict is read (``record_release``), never resubmitted. Returns the queue as it now stands, with a job
    submitted here in it, so the pass that queued a check does not go on to read it as one that left the queue."""
    release = publication.model.release
    if release is None or release.export_check is None:
        return queued
    check = release.export_check
    result = export_check_result.result_path(
        export_check_dir(publication, check), export_check_result.spec_digest(check.spec)
    )
    if result.is_file() or export_check_job_name(publication) in queued or publication.check_job_record.is_file():
        return queued
    job = submit_export_check(publication, check, repo_root)
    LOGGER.info("%s: export check submitted as job %s", publication.label, job)
    return queued | {export_check_job_name(publication)}


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
            f"{slurm_jobs.SLURM_LOG_DIR / log_name.format(job=job)} in the submitting checkout, then {retry}"
        )


def check_no_inline_export(publication: Publication) -> None:
    """Refuse to rebuild a clone an inline export has recorded itself in and not finished: it may be
    writing the export right now, and it has no job in the queue to say so."""
    record = publication.inline_export_record
    if record.is_file():
        raise ExportError(
            f"{record} records an unfinished inline export ({record.read_text().strip()}); it may still be "
            f"writing {publication.hf_dir}. Once it is not running, delete the record to export again"
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


def verify_export(hf_dir: Path) -> int:
    """Check that an export finished and that its tensors match its index by name in both
    directions, and return how many tensors it holds. An index or shard that cannot be parsed, or
    that is gone by the time it is read (an export is deleted file by file), is an export that does
    not verify, whatever files sit beside it. Any other failure to read the files (a stale handle,
    an I/O error) is not a verdict on the export and is raised as it is."""
    if not (hf_dir / INDEX_FILE).is_file():
        raise ExportError(f"{hf_dir}: no {INDEX_FILE}")
    if not (hf_dir / EXPORT_COMPLETE_FILE).is_file():
        raise ExportError(f"{hf_dir}: no {EXPORT_COMPLETE_FILE}, the exporter's last write; the export did not finish")
    return len(checked_weight_map(hf_dir))


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
    removes a rejected export, rewrites the clone's run_config and writes the tracker beside the
    clone, so the clone must resolve inside the export root, and none of its hf/, its run_config and
    that tracker may be a link: through any of these, the removal or a write would reach whatever
    the link names, a training run's own checkpoint or tracker among them."""
    if publication.hf_dir.is_symlink():
        raise ExportError(f"{publication.hf_dir} is a symlink; an export clone's hf/ is exporter output")
    for written, what in (
        (publication.clone / RUN_CONFIG, "an export clone's run_config is a patched copy"),
        (publication.clone_root / LATEST_FILE, "a clone root's tracker names the clone's iteration"),
    ):
        if written.is_symlink():
            raise ExportError(f"{written} is a link; {what}")
    resolved = publication.clone.resolve()
    if not resolved.is_relative_to(export_root.resolve()):
        raise ExportError(f"{publication.clone} resolves to {resolved}, outside the export root {export_root}")


def local_files(hf_dir: Path) -> dict[str, int]:
    """The export's files and sizes, as the Hub will hold them."""
    return {p.name: p.stat().st_size for p in hf_dir.iterdir() if p.is_file() and not p.name.startswith(".")}


def check_hub_access(api: Any, manifest: Manifest) -> None:
    """Refuse to act for a Hub identity outside the manifest's namespaces. The Hub answers a private
    repository the token cannot see as a missing one, so without access every revision would read as
    unpublished and, where its local export was removed, be exported again."""
    identity = api.whoami()
    reachable = {identity["name"], *(org["name"] for org in identity["orgs"])}
    for namespace in sorted({model.repo.split("/")[0] for model in manifest.models}):
        if namespace not in reachable:
            raise ExportError(
                f"the Hub identity {identity['name']!r} cannot reach the namespace {namespace!r} of {manifest.source}"
            )


def reference_export(hf_dir: Path) -> dict[str, int] | None:
    """The files and sizes of a local export that the Hub's copy can be compared with, or None where
    there is no such export. One to compare with holds its index and the exporter's last write,
    judged by their presence alone: it may still be corrupt, but it is a finished export's listing.
    There is none where the export is absent or lacks either file (removed or emptied to free
    space, or never finished here), or where a file goes while the export is listed, which is a
    deletion in progress. Only a missing file is read that way; any other failure to read the
    directory (a stale handle, an I/O error) is raised as it is."""
    try:
        files = local_files(hf_dir)
    except FileNotFoundError:
        return None
    return files if FINISHING_FILES <= files.keys() else None


def revision_files(api: Any, repo: str, revision: str) -> dict[str, Any] | None:
    """The top-level files a Hub revision holds, as huggingface_hub's RepoFile entries (folders are
    not files and are left out); None when the repository or the revision does not exist, which is
    what an unpublished revision looks like. The listing is a lazy generator whose request, and so
    whose 404, happens on iteration, which is why it is drained inside the handler."""
    from huggingface_hub.hf_api import RepoFolder
    from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

    try:
        entries = list(api.list_repo_tree(repo, revision=revision, recursive=False))
    except (RepositoryNotFoundError, RevisionNotFoundError):
        return None
    return {entry.path: entry for entry in entries if not isinstance(entry, RepoFolder)}


def published(api: Any, repo: str, revision: str, files: dict[str, int]) -> bool:
    """Whether the revision already holds every one of an export's files at the same size."""
    remote = revision_files(api, repo, revision)
    return remote is not None and all(name in remote and remote[name].size == size for name, size in files.items())


def holds_its_weights(api: Any, repo: str, revision: str, holder: str) -> bool:
    """Whether ``holder`` holds every safetensors file of ``revision`` with the same content, judged
    by LFS content hash. Two exports of one architecture share their file names and sizes, so only
    content tells them apart; and an upload never deletes, so a file the holder has beyond them (a
    shard left by an earlier export) does not count against it. A revision that gives no content hash
    to compare by is an error: it cannot be judged, which is not a finding that it is unpublished."""
    ours, theirs = revision_files(api, repo, revision), revision_files(api, repo, holder)
    if ours is None or theirs is None:
        return False

    def shard_hashes(files: dict[str, Any]) -> dict[str, str | None]:
        return {
            name: entry.lfs.sha256 if entry.lfs else None
            for name, entry in files.items()
            if name.endswith(".safetensors")
        }

    hashes, held = shard_hashes(ours), shard_hashes(theirs)
    unhashed = sorted(name for name, h in hashes.items() if h is None)
    if not hashes or unhashed:
        raise ExportError(
            f"{repo}@{revision} gives no content hash to compare {holder} by "
            f"({'shards without one: ' + ', '.join(unhashed) if unhashed else 'no safetensors file'})"
        )
    return all(held.get(name) == h for name, h in hashes.items())


def holds_a_finished_export(api: Any, repo: str, revision: str) -> bool:
    """Whether the revision holds a finished export, judged from the Hub alone. An upload is one
    commit carrying the whole export, and every revision this tool creates branches from the
    repository's first commit, which holds no export (see ``upload``), so a revision holding the
    index and the exporter's last write holds the rest of its own export too."""
    remote = revision_files(api, repo, revision)
    return remote is not None and FINISHING_FILES <= remote.keys()


def on_the_hub(api: Any, publication: Publication) -> bool:
    """Whether every revision the publication targets already holds its export. All of them, not
    just its own: a final checkpoint whose upload to main failed would otherwise count as published
    and never reach main. The local export is listed once, before the Hub is asked anything, and
    every revision is compared with that one listing, file by file, by name and size: the export
    may be deleted while the pass waits on the Hub, and a listing taken after that would be empty,
    which any branch that exists matches. Nothing in the export is read unless the sizes
    disagree, so a file gone unreadable in the export of a revision long since published stops no
    pass. Where there is no export to compare with (``reference_export``), or where one that
    disagrees does not verify, the Hub's copy is judged on its own, so a revision safely published
    is neither exported again nor dropped from the card, and one that is not is never taken for
    it. main must further hold the final checkpoint's own weights, by content: by name and size it
    cannot be told from another export."""
    repo = publication.model.repo
    reference = reference_export(publication.hf_dir)
    if reference is not None:
        if all(published(api, repo, revision, reference) for revision in publication.targets):
            return main_carries(api, repo, publication.checkpoint)
        problem = export_problem(publication)
        if problem is None:
            return False
        LOGGER.warning(
            "%s: the local export disagrees with the Hub and does not verify (%s); the Hub's copy is judged on "
            "its own",
            publication.label,
            problem,
        )
    return hub_holds(api, repo, publication.checkpoint)


def main_carries(api: Any, repo: str, checkpoint: Checkpoint) -> bool:
    """Whether main holds the checkpoint's own weights, by content, where it should: only the final
    checkpoint of the default stage belongs on main, so any other checkpoint asks nothing of it."""
    return not checkpoint.default or holds_its_weights(api, repo, checkpoint.revision, MAIN)


def hub_holds(api: Any, repo: str, checkpoint: Checkpoint) -> bool:
    """Whether the checkpoint is published, judged from the Hub alone: every revision it belongs on
    holds a finished export, and main the final checkpoint's own weights."""
    finished = all(holds_a_finished_export(api, repo, revision) for revision in checkpoint.targets)
    return finished and main_carries(api, repo, checkpoint)


def hub_checkpoints(api: Any, model: Model) -> list[Checkpoint]:
    """Every checkpoint of the model that the Hub holds, in stage then iteration order: the branches
    a stage's revision pattern names exactly, kept where ``hub_holds``. This, not the save
    directories, is what a model card lists: a published checkpoint's Megatron save and its local
    export are both removed in time, and the Hub is then the only place it exists. A repository
    that does not exist holds none."""
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        branches = {ref.name for ref in api.list_repo_refs(model.repo).branches}
    except RepositoryNotFoundError:
        return []
    found = []
    for stage in model.stages:
        named = re.compile(re.escape(stage.revision).replace(re.escape(ITERATION_FIELD), "([0-9]+)"))
        iterations = sorted({int(match.group(1)) for name in branches if (match := named.fullmatch(name))})
        # A branch such as sft_iter_0600 matches the pattern without being a name this tool gives.
        own = [c for c in (Checkpoint(stage, iteration) for iteration in iterations) if c.revision in branches]
        found += [checkpoint for checkpoint in own if hub_holds(api, model.repo, checkpoint)]
    return found


def verified_commit(api: Any, publication: Publication) -> str:
    """The commit the publication's revision is at, once that commit is found to hold the local export: every file
    of the export at its size, and every file the Hub stores by content hash, as it stores the shards, with the
    export's own content. The commit is read first and the files are listed at it, so what was compared is what the
    commit holds whatever is uploaded to the revision meanwhile."""
    from huggingface_hub.utils.sha import sha_fileobj

    repo = publication.model.repo
    commit = api.model_info(repo, revision=publication.revision).sha
    remote = revision_files(api, repo, commit)
    if remote is None:
        raise ExportError(f"{publication.label}: its commit {commit} cannot be read")
    for name, size in sorted(local_files(publication.hf_dir).items()):
        held = remote.get(name)
        if held is None or held.size != size:
            raise ExportError(
                f"{publication.label}: commit {commit} does not hold {name} at the export's {size} bytes"
            )
        if held.lfs is None:
            continue
        with (publication.hf_dir / name).open("rb") as handle:
            if sha_fileobj(handle).hex() != held.lfs.sha256:
                raise ExportError(f"{publication.label}: commit {commit} holds another {name} than the export's")
    return commit


def last_release_line(log: Path, tag: str, label: str) -> str | None:
    """The newest line of a release log for ``label``, which is the one a reader of the log decides by; None when
    the log holds none."""
    if not log.is_file():
        return None
    opening = f"{tag} {label} "
    lines = [line for line in log.read_text().splitlines() if line.startswith(opening)]
    return lines[-1] if lines else None


def release_line_fields(line: str) -> tuple[str, list[str]]:
    """The commit and the detail's words of a release log's line, ``<tag> <label> <commit> <detail...> <UTC time>``
    (``record_release_line``'s)."""
    words = line.split(" ")
    return words[2], words[3:-1]


def record_release_line(log: Path, tag: str, label: str, commit: str, detail: str) -> bool:
    """Append ``<tag> <repo>@<revision> <commit> <detail> <UTC time>`` to a release log, single spaces throughout,
    unless the log's newest line for ``label`` already states that commit and detail; returns whether it appended.
    A line is one write, so a reader never finds part of one."""
    newest = last_release_line(log, tag, label)
    if newest is not None and release_line_fields(newest) == (commit, detail.split(" ")):
        return False
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as lines:
        lines.write(f"{tag} {label} {commit} {detail} {sync_bucket.utc_now()}\n")
    return True


def record_verified(api: Any, publication: Publication) -> str:
    """Write the verified line of a published revision of a model with a ``release`` block, and return the commit it
    states.

    The line is written once the revision's commit holds the local, verified export (``verified_commit``), and again
    whenever the revision moves to another commit. A revision whose line already names its commit is not read
    again. A published revision with no verified export to compare the Hub's copy with is an error, raised for the
    pass to report."""
    release = publication.model.release
    label = publication.label
    commit = api.model_info(publication.model.repo, revision=publication.revision).sha
    newest = last_release_line(release.verified_log, VERIFIED_TAG, label)
    if newest is None or release_line_fields(newest)[0] != commit:
        problem = export_problem(publication)
        if problem is not None:
            raise ExportError(f"published, but there is no verified export to compare the Hub's copy with ({problem})")
        commit = verified_commit(api, publication)
        tensors = verify_export(publication.hf_dir)
        record_release_line(release.verified_log, VERIFIED_TAG, label, commit, f"{tensors} tensors")
        LOGGER.info(
            "%s: verified at commit %s, %d tensors; recorded in %s", label, commit, tensors, release.verified_log
        )
    return commit


def record_export_check(publication: Publication, commit: str, queued: set[str]) -> bool:
    """Write the export check's line for ``commit`` of a model whose ``release`` block names a check, and return
    whether the line is written.

    The line states the judge's verdict on the revision's export: PASS or FAIL as the judge's result says, and
    nothing while the judge has not written one (``False``). A check that could not evaluate the export, a result
    that is some other export's, and a check job that left the queue without a result are errors, raised for the pass
    to report: none of them is a verdict, and none is retried here."""
    label = publication.label
    check = publication.model.release.export_check
    try:
        verdict = export_check_result.read_verdict(
            export_check_dir(publication, check), check.spec, publication.hf_dir, publication.source
        )
    except ValueError as error:
        raise ExportError(str(error)) from error
    if verdict is None:
        if export_check_job_name(publication) not in queued:
            check_no_failed_job(
                publication.check_job_record,
                "export check",
                "it wrote no result",
                CHECK_JOB_LOG,
                f"clear {export_check_dir(publication, check)} and delete {publication.check_job_record} to submit "
                "it again",
            )
        LOGGER.info("%s: its export check has not judged it yet", label)
        return False
    if verdict not in RECORDED_VERDICTS:
        raise ExportError(f"its export check could not evaluate it ({verdict}); no {check.tag} line is written")
    if record_release_line(check.log, check.tag, label, commit, verdict):
        LOGGER.info("%s: export check %s at commit %s; recorded in %s", label, verdict, commit, check.log)
    return True


def record_release(api: Any, publication: Publication, queued: set[str]) -> bool:
    """Record what a published revision's release waits on, and return whether all of it is recorded: the verified
    line (``record_verified``) and, for a model whose release block names an export check, the check's line
    (``record_export_check``) for the same commit. A model with no release block waits on nothing."""
    release = publication.model.release
    if release is None:
        return True
    commit = record_verified(api, publication)
    if release.export_check is None:
        return True
    return record_export_check(publication, commit, queued)


def initial_commit(api: Any, repo: str) -> str:
    """The repository's first commit, which every revision is branched from because it holds no
    export. One that does -- a history squashed into a single commit, a repository copied from
    another -- is refused: a branch made from it would start as a copy of that export and, if its own
    upload failed, pass for published with those weights."""
    commit = api.list_repo_commits(repo)[-1].commit_id
    held = revision_files(api, repo, commit)
    if held is None:
        raise ExportError(f"{repo}: its first commit {commit} cannot be read")
    if FINISHING_FILES & held.keys():
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
        # Only a branch that does not exist yet needs a branch point; an existing one (created, then
        # its upload failed) is uploaded to as it is.
        if revision != MAIN and revision_files(api, publication.model.repo, revision) is None:
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


def render_model_card(manifest: Manifest, model: Model, rows: list[tuple[Checkpoint, float | None]]) -> str:
    """The repository's README: the manifest's card text around the curriculum and revision tables,
    one revision row per checkpoint given, each with its training loss where one is known."""
    card = manifest.card
    name = model.repo.split("/")[1]
    tags = list(card.tags) + ([card.reasoning_tag] if model.reasoning else [])
    default_stage = next(s.name for s in model.stages if s.default)
    lines = [
        "---",
        f"license: {card.license}",
        f"license_name: {card.license_name}",
        *([f"base_model: {manifest.base_model}"] if manifest.base_model is not None else []),
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
    ]
    if model.card_sections is not None:
        lines += [model.card_sections, ""]
    lines += [
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
    for checkpoint, loss in rows:
        loss_cell = f"{loss:.4f}" if loss is not None else ""
        revision = f"`{checkpoint.revision}`" + (" (also `main`)" if checkpoint.default else "")
        lines.append(
            f"| {revision} | {checkpoint.stage.name} | {checkpoint.iteration:,} | {_tokens(checkpoint.tokens_seen)} | {loss_cell} |"
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


def write_model_cards(
    manifest: Manifest, api: Any, wandb_api: Any, run_dir: Path, models: list[Model], execute: bool
) -> None:
    """Bring each model's card to what the Hub holds. A card's revision rows are the checkpoints its
    repository holds (see ``hub_checkpoints``), so a revision whose export or upload failed is not
    listed as if it existed and one whose save has since been removed is not dropped; a model with
    nothing published gets no card. With ``execute`` a card that differs from the one uploaded last
    is uploaded. Without it each card is rendered into the run directory instead, to be read before
    anything is uploaded, and the Hub is left alone."""
    for model in models:
        checkpoints = hub_checkpoints(api, model)
        if not checkpoints:
            LOGGER.info("%s: no checkpoint on the Hub, so no card", model.repo)
            continue
        rows = []
        for stage in model.stages:
            held = [checkpoint for checkpoint in checkpoints if checkpoint.stage is stage]
            cache = run_dir.parent / "losses" / f"{stage.wandb_exp_name}.json"
            found = losses(wandb_api, manifest.wandb, stage.wandb_exp_name, {c.iteration for c in held}, cache)
            rows += [(checkpoint, found.get(checkpoint.iteration)) for checkpoint in held]
        text = render_model_card(manifest, model, rows)
        name = model.repo.split("/")[1]
        if not execute:
            rendered = run_dir / "cards" / name / README_NAME
            rendered.parent.mkdir(parents=True, exist_ok=True)
            rendered.write_text(text)
            LOGGER.info("%s: card of %d revision(s) rendered to %s; nothing uploaded", model.repo, len(rows), rendered)
            continue
        # Cards persist beside the run directories so an unchanged card is not re-uploaded every pass.
        if upload_model_card(api, model, text, run_dir.parent / "cards" / name):
            LOGGER.info("%s: model card updated", model.repo)


def ensure_collection(api: Any, collection: Collection, namespace: str, notes: dict[str, str | None]) -> str:
    """The collection's slug, creating it if absent, with the manifest's description and every repo
    in ``notes`` a member under its note (a None note leaves the Hub's as it is).

    A collection is created once, so a description or a note that changes in the manifest afterwards
    reaches the Hub only through here: each is written when, and only when, it differs from the Hub's.
    """
    existing = [c for c in api.list_collections(owner=namespace) if c.title == collection.title]
    if existing:
        slug = existing[0].slug
    else:
        slug = api.create_collection(
            title=collection.title, namespace=namespace, description=collection.description, private=collection.private
        ).slug
        LOGGER.info("created collection %s", slug)
    current = api.get_collection(slug)
    if current.description != collection.description:
        api.update_collection_metadata(slug, description=collection.description)
        LOGGER.info("collection %s: description updated", slug)
    items = {item.item_id: item for item in current.items}
    for repo, note in notes.items():
        item = items.get(repo)
        if item is None:
            api.add_collection_item(slug, item_id=repo, item_type="model", note=note, exists_ok=True)
        elif note is not None and item.note != note:
            api.update_collection_item(slug, item.item_object_id, note=note)
            LOGGER.info("collection %s: note of %s updated", slug, repo)
    return slug


PHASES = ("export", "submit", "upload", "all", "rolling", "cards")
# The phases that queue exports as jobs of their own instead of running the exporter here.
SUBMITTING_PHASES = ("submit", "rolling")

# A run's exit status, which is its last pass's. EXIT_PUBLISHED: the Hub holds everything the manifest
# asks for (a cards pass, which judges no publication, exits with it once its cards are written).
# EXIT_FAILED: the pass met an error on a publication (an export, a submission, an upload or the
# recording of a release line failed, or a job left the queue without finishing its work); an uncaught
# error exits 1 as well. EXIT_PENDING: the pass met no error, but work is still in flight or left for
# another phase (an export queued or running, an export waiting for an upload pass, an upload job queued,
# an export check not yet judged, a release line not yet written, or in --plan mode anything to do).
# It is 3 because argparse exits 2 on a usage error.
EXIT_PUBLISHED = 0
EXIT_FAILED = 1
EXIT_PENDING = 3


@dataclass(frozen=True)
class PassOutcome:
    """What one pass left undone: ``pending`` publications are unfinished, ``failed`` of them because the
    pass met an error on them, and the rest because their work is in flight or left for another phase. A
    publication is unfinished until it is on the Hub and, for a model with a ``release`` block, its release
    lines are written. The ``cards`` phase judges no publication and leaves none undone."""

    pending: int
    failed: int

    @property
    def exit_status(self) -> int:
        """EXIT_FAILED on any failure, else EXIT_PENDING while anything is unpublished, else EXIT_PUBLISHED."""
        if self.failed:
            return EXIT_FAILED
        return EXIT_PENDING if self.pending else EXIT_PUBLISHED


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
) -> PassOutcome:
    """One pass over the manifest: export and upload what is missing, then the card of every model
    the pass covers that has anything published, and the collection for the models this pass
    confirmed (a pass that confirmed nothing writes no collection). Returns how many publications
    remain unpublished (none when the Hub holds everything the manifest asks for) and how many of
    those the pass met an error on, which it reports and counts rather than raises, so one failed
    publication stops no other.

    A card lists what the Hub holds, not what the pass planned or what is left on disk (see
    ``write_model_cards``), so it keeps every revision after its save and export are removed.

    A model whose manifest entry has a ``release`` block is not done at its upload. The phases that may upload
    (``upload``, ``all``, ``rolling``) record each of its published revisions in the model's verified log once the
    revision's commit is found to hold the local export, and the export check's verdict for the same commit once
    the judge has written it (``record_release``); a revision still waiting for a line counts as unpublished, and
    one whose recording failed counts as failed. The submitting phases (``submit``, ``rolling``) queue the model's
    export check for each verified export. ``export`` and ``submit`` record nothing and count nothing of the
    release.

    ``phase`` splits the work by what it needs.

    ``cards`` writes the cards and nothing else: it plans no publication and reads no save
    directory, so it exports nothing, uploads no revision and leaves the collection alone, and it
    always reports nothing pending and nothing failed, so it exits 0 once the cards are written. It
    is how a change to the manifest's card text, or to a model's description, reaches the
    repositories whose checkpoints were all published long ago. Without ``execute`` it renders each
    card into the run directory instead of uploading it.

    ``export`` runs only the exports, which take this allocation's GPUs for a few minutes each, and
    writes nothing to the Hub. ``submit`` does the same work without the GPUs, queueing a job per
    checkpoint instead, and likewise writes nothing to the Hub -- neither revisions nor cards nor
    collection membership, since a pass that has only queued work has confirmed nothing to describe.
    ``upload`` uploads the publications whose export is already verified, and the cards and
    collection that follow from them, touching no GPU. ``all`` exports here and then uploads.
    ``rolling`` is the phase for a run that is still training, repeated with a poll interval: it
    submits what ``submit`` would and uploads what ``upload`` would. In every phase that acts, an
    export whose job is still in the queue is left to its job, neither uploaded nor rebuilt, since
    the job writes it again when it starts; the queue is read afresh for each publication, so a job
    queued while a long pass runs counts too. A submitting phase (``submit``, ``rolling``) reports an
    export whose job has left the queue without finishing it, with the reason, rather than
    resubmitting it; ``export`` and ``all`` export it again inline. The job's record is discharged
    once its export verifies and the job has left the queue, or once its revision is on the Hub.
    When the manifest has an ``upload`` block, ``rolling`` uploads nothing itself: when finished
    exports are waiting it submits the manifest's one upload job (see ``upload_command``), records
    that job against every publication it is responsible for, and leaves cards and collection to
    it. A pass that has written the cards and collection discharges the records of what it
    confirmed, so an upload job that leaves the queue with a record still standing -- whether its
    revisions are missing or its card is -- is reported, with the command that resubmits it, rather
    than resubmitted.

    A publication counts as published only when every revision it targets holds its export, main
    included for the default; without a local export holding its index and completion file, the
    Hub's copy is judged on its own (see ``on_the_hub``). An export that does not verify and is exported again is removed
    first, since the exporter writes into it without clearing it, and only from a clone that lies
    inside the export root.

    The split is what lets the GPUs be borrowed from another workload for exactly the export, or --
    with ``submit`` and ``rolling`` -- not borrowed at all."""
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, not {phase!r}")
    if phase == "cards":
        write_model_cards(manifest, api, wandb_api, run_dir, selected_models(manifest, repo_filter), execute)
        return PassOutcome(pending=0, failed=0)
    uploads_are_jobs = phase == "rolling" and manifest.upload is not None and execute
    if uploads_are_jobs and repo_filter:
        raise ValueError(
            "--models cannot be combined with a manifest upload block: the upload job publishes the whole manifest"
        )
    publications = plan(manifest, repo_filter, newest_first)
    export_log = run_dir / "export.log"
    # Every pass that acts reads the queue: an export whose job is still queued will be written when
    # that job starts, so no pass may upload it or rebuild its clone meanwhile. The upload job is
    # looked for once, by the rolling pass that submits it, which acts on nothing for long.
    upload_queued = uploads_are_jobs and upload_job_name(manifest) in queued_job_names()
    resubmit_upload = (
        slurm_jobs.shell_submission(
            upload_command(manifest, repo_root), repo_root, slurm_jobs.forced_submission_env(repo_root)
        )
        if uploads_are_jobs
        else ""
    )
    # The upload job submitted in this pass. Its own pass reads the manifest after it was
    # submitted, so every publication this pass finds verified will be verified when the job looks
    # too: each is its responsibility.
    upload_job: str | None = None
    # A failed submission is not repeated for each later publication of the same pass: one that
    # timed out after registering would otherwise leave several jobs racing for the collection.
    upload_submission_failed = False
    pending = 0
    failed = 0
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
                check_no_failed_job(
                    publication.upload_job_record,
                    "upload",
                    "its record was never discharged, so its revisions, card or collection are not all written",
                    UPLOAD_JOB_LOG,
                    f"resubmit it (a pass that finishes deletes the record) with: {resubmit_upload}",
                )
            except ExportError as error:
                LOGGER.error("%s: %s", publication.label, error)
                pending += 1
                failed += 1
                continue
        if on_the_hub(api, publication):
            LOGGER.info("%s: already on the Hub", publication.label)
            if execute:
                publication.export_job_record.unlink(missing_ok=True)
                publication.inline_export_record.unlink(missing_ok=True)
                if publication.model.release is not None:
                    # A submitting pass queues the check; a pass that may upload records what the Hub now holds.
                    try:
                        queued = queued_job_names()
                        if phase in SUBMITTING_PHASES:
                            queued = submit_export_check_if_due(publication, queued, repo_root)
                        if phase not in ("export", "submit") and not record_release(api, publication, queued):
                            pending += 1
                    except ExportError as error:
                        LOGGER.error("%s: %s", publication.label, error)
                        pending += 1
                        failed += 1
            touched.setdefault(publication.model.repo, []).append(publication)
            continue
        if not execute:
            LOGGER.info(
                "%s: would %s", publication.label, "upload" if export_is_verified(publication) else "export and upload"
            )
            pending += 1
            continue
        # Export jobs are looked for afresh for each publication a pass may act on: an export or upload
        # run here takes minutes, and a polling process may queue jobs for later publications meanwhile.
        queued = queued_job_names()
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
                        publication.upload_job_record.unlink(missing_ok=True)
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
                check_no_inline_export(publication)
                check_clone_is_safe_to_rebuild(publication, manifest.export_root)
                if phase not in SUBMITTING_PHASES:
                    # An inline export has no job in the queue to announce it, so it records itself
                    # from before it clears the clone until its export verifies; a record left behind
                    # marks an export that did not finish.
                    publication.clone_root.mkdir(parents=True, exist_ok=True)
                    publication.inline_export_record.write_text(
                        f"{socket.gethostname()} pid {os.getpid()} since {sync_bucket.utc_now()}\n"
                    )
                try:
                    if publication.hf_dir.exists():
                        LOGGER.warning(
                            "%s: removing the unverified export in %s: %s",
                            publication.label,
                            publication.hf_dir,
                            problem,
                        )
                        shutil.rmtree(publication.hf_dir)
                    make_export_clone(publication.source, publication.clone)
                except (ExportError, OSError):
                    if phase not in SUBMITTING_PHASES:
                        # No exporter has started, so the record would only tell later passes that an
                        # export may still be writing, and they would report that instead of this.
                        publication.inline_export_record.unlink(missing_ok=True)
                    raise
                if phase in SUBMITTING_PHASES:
                    job = submit_export(publication, manifest, repo_root)
                    LOGGER.info("%s: export submitted as job %s", publication.label, job)
                    pending += 1
                    continue
                run_export(publication, manifest, repo_root, export_log)
                count = verify_export(publication.hf_dir)
                publication.inline_export_record.unlink(missing_ok=True)
                LOGGER.info("%s: export verified, %d tensors", publication.label, count)
            if export_job_name(publication) in queued:
                # Verified, but a job still in the queue will write this export again when it starts.
                LOGGER.info("%s: export verified but its job is still in the queue; upload waits", publication.label)
                pending += 1
                continue
            # A verified export whose job has left the queue has settled that job, or inline export.
            publication.export_job_record.unlink(missing_ok=True)
            publication.inline_export_record.unlink(missing_ok=True)
            if phase in SUBMITTING_PHASES:
                # The check judges the export, not the Hub's copy, so it runs beside the upload, not after it.
                queued = submit_export_check_if_due(publication, queued, repo_root)
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
            if not record_release(api, publication, queued):
                pending += 1
        except ExportError as error:
            LOGGER.error("%s: %s", publication.label, error)
            pending += 1
            failed += 1
    outcome = PassOutcome(pending=pending, failed=failed)
    if not execute or phase in ("export", "submit") or uploads_are_jobs:
        return outcome
    namespace = manifest.models[0].repo.split("/")[0]
    write_model_cards(manifest, api, wandb_api, run_dir, selected_models(manifest, repo_filter), execute)
    if touched:
        # A pass that confirmed nothing writes no collection: an empty one would announce models that
        # do not exist yet.
        ensure_collection(
            api, manifest.collection, namespace, {m.repo: m.note for m in manifest.models if m.repo in touched}
        )
    # Every publication this pass confirmed is now on the Hub and described by its card and
    # collection, which is all an upload job is for: its record is discharged.
    for confirmed in touched.values():
        for publication in confirmed:
            publication.upload_job_record.unlink(missing_ok=True)
    return outcome


def make_wandb_api():
    """The W&B client, imported here so a pass that asks W&B nothing (a plan of exports and uploads)
    needs no W&B credentials."""
    import wandb

    return wandb.Api(timeout=120)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """The command line: the manifest, plan-only, a model filter, and polling."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog=f"Exit status, the last pass's: {EXIT_PUBLISHED} when the Hub holds everything the manifest asks for "
        f"(a cards pass judges no publication and exits {EXIT_PUBLISHED} once its cards are written); "
        f"{EXIT_FAILED} when the pass failed on any publication (or raised); {EXIT_PENDING} when it met no error "
        "but work is still in flight or left for another phase (an export queued or awaiting an upload pass, an "
        "upload job queued, or in --plan mode anything to do).",
    )
    parser.add_argument("--manifest", type=Path, required=True, help="the publication manifest")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="the checkout the manifest's config paths and the exporter are taken from (default: this one)",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="report what would be exported and uploaded; change nothing (with --phase cards: render each card "
        "into the run directory instead of uploading it)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="only repos whose id contains one of these substrings (refused by a rolling pass whose manifest has an "
        "upload block, since its upload job publishes the whole manifest)",
    )
    parser.add_argument(
        "--phase",
        choices=PHASES,
        default="all",
        help="export: only the exports, on this allocation's GPUs; submit: queue each missing export as "
        "its own job instead, taking no GPU here and writing nothing to the Hub; upload: only uploads of "
        "verified exports, cards and collection (no GPU); all: export here, then upload; rolling: submit "
        "missing exports and upload those whose job has finished, for polling a run still training "
        "(with an upload block in the manifest, the uploads go to the manifest's one upload job instead); "
        "cards: only the model cards, each listing the revisions its repository holds (no export, no "
        "revision upload, no collection write)",
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
    check_hub_access(api, manifest)
    # A plan of exports and uploads asks W&B nothing; rendering a card needs its losses, plan or not.
    wandb_api = make_wandb_api() if not args.plan or args.phase == "cards" else None
    deadline = time.time() + args.stop_after * 3600 if args.stop_after else None
    while True:
        outcome = publish_pass(
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
        LOGGER.info(
            "pass complete: %d publication(s) pending, %d of them failed; exit status %d",
            outcome.pending,
            outcome.failed,
            outcome.exit_status,
        )
        if args.poll_interval is None or (deadline is not None and time.time() >= deadline):
            return outcome.exit_status
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    sys.exit(main())

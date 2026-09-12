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
card that records the tokens seen and the training loss at each.

The manifest (``configs/control_pretraining/hub_models.yaml``) names the repositories and, for
each, the training stages that feed it; a stage is its training config, from which the save
directory, ``train_iters`` and the W&B run name are read. Every completed checkpoint of a stage
(at or below the directory's tracker, so never a save in progress) becomes a revision named by
the stage's pattern; the designated stage's final checkpoint is also the default revision.

A checkpoint is published in four steps, each skipped when its result already exists, so a pass
is idempotent and a run can be repeated as new checkpoints land:

1. An export clone: a directory of symlinks to the checkpoint's files plus a patched copy of
   ``run_config.yaml``. The exporter rebuilds the model from that file, and training serialised
   a closure it cannot import (``_apply_moe_experts_impl.<locals>...``); the two edits below make
   it importable. The training tree is never written to.
2. The HF export, by ``pipeline_checkpoint_convert.sh export`` on this allocation's GPUs at the
   manifest's parallelism, into the clone. The exporter needs the SLURM environment of the
   allocation it runs in.
3. Verification: every tensor the safetensors index promises is in the shard it names, and every
   tensor a shard holds is in the index — by tensor name, never by file count.
4. The upload, to the revision (and to ``main`` for the default), followed by the model card on
   ``main`` and the repository's membership of the collection.

Losses come from W&B (the manifest's loss key at the iteration's step) across every run that
carried the stage's name, since a stage runs as a chain of segments; a checkpoint whose iteration
W&B never logged is listed with no loss rather than an invented one. Everything a card says
beyond its tables comes from the manifest's ``card`` block.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
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
README_NAME = "README.md"
EXPORTER = "pipeline_checkpoint_convert.sh"
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
        "tokens_per_iteration",
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
EXPORT_KEYS = frozenset({"tp", "ep"})
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
class ExportParallelism:
    """How the exporter shards the model across this allocation's GPUs."""

    tp: int
    ep: int


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
class Stage:
    """One training stage feeding a model: its config's facts and how its checkpoints are named."""

    name: str
    config: Path
    save: Path
    train_iters: int
    wandb_exp_name: str
    revision: str
    default: bool
    extra_directories: tuple[Path, ...]
    tokens_before: int


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
    """The publication's definition: the collection, the architecture, and every model in it."""

    collection: Collection
    architecture: str
    tokens_per_iteration: int
    export_root: Path
    log_dir: Path
    hf_home: Path
    export: ExportParallelism
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
    def label(self) -> str:
        return f"{self.model.repo}@{self.revision}"


def stage_facts(config: Path) -> tuple[Path, int, str]:
    """The save directory, train_iters and W&B run name a stage config declares."""
    save = sync_bucket.stage_save_directory(config)
    cfg = yaml.safe_load(config.read_text())
    train_iters = (cfg.get("train") or {}).get("train_iters")
    exp_name = (cfg.get("logger") or {}).get("wandb_exp_name")
    if not isinstance(train_iters, int) or train_iters <= 0:
        raise ManifestError(f"{config}: train.train_iters must be a positive integer, got {train_iters!r}")
    if not isinstance(exp_name, str) or not exp_name:
        raise ManifestError(f"{config}: logger.wandb_exp_name must be set")
    return save, train_iters, exp_name


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
    save, train_iters, exp_name = stage_facts(config)
    return Stage(
        name=str(item["name"]),
        config=config,
        save=save,
        train_iters=train_iters,
        wandb_exp_name=exp_name,
        revision=revision,
        default=bool(item["default"]),
        extra_directories=tuple(save.parent / str(d) for d in item["extra_directories"]),
        tokens_before=tokens_before,
    )


def _history_stage(config_path: str, repo_root: Path, where: str, tokens_before: int) -> Stage:
    config = repo_root / config_path
    if not config.is_file():
        raise ManifestError(f"{where}: history config {config} does not exist")
    save, train_iters, exp_name = stage_facts(config)
    return Stage(
        name=config.stem,
        config=config,
        save=save,
        train_iters=train_iters,
        wandb_exp_name=exp_name,
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
        tokens_before += stage.train_iters
        history.append(stage)
    stages = []
    for s_index, raw_stage in enumerate(item["stages"]):
        stage = _stage(raw_stage, repo_root, f"{where}.stages[{s_index}]", tokens_before)
        tokens_before += stage.train_iters
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


def load_manifest(path: Path, repo_root: Path) -> Manifest:
    """Read and validate the manifest; repo-relative config paths resolve against ``repo_root``."""
    raw = sync_bucket.exact_keys(yaml.safe_load(path.read_text()), MANIFEST_KEYS, str(path))
    collection = sync_bucket.exact_keys(raw["collection"], COLLECTION_KEYS, f"{path}: collection")
    export = sync_bucket.exact_keys(raw["export"], EXPORT_KEYS, f"{path}: export")
    wandb_raw = sync_bucket.exact_keys(raw["wandb"], WANDB_KEYS, f"{path}: wandb")
    card = sync_bucket.exact_keys(raw["card"], CARD_KEYS, f"{path}: card")
    tokens_per_iteration = raw["tokens_per_iteration"]
    if not isinstance(tokens_per_iteration, int) or tokens_per_iteration <= 0:
        raise ManifestError(f"{path}: tokens_per_iteration must be a positive integer")
    if not all(isinstance(export[k], int) and export[k] > 0 for k in ("tp", "ep")):
        raise ManifestError(f"{path}: export.tp and export.ep must be positive integers")
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
        collection=Collection(
            title=str(collection["title"]),
            description=description,
            private=bool(collection["private"]),
        ),
        architecture=str(raw["architecture"]),
        tokens_per_iteration=tokens_per_iteration,
        export_root=Path(raw["export_root"]),
        log_dir=Path(raw["log_dir"]),
        hf_home=Path(raw["hf_home"]),
        export=ExportParallelism(tp=export["tp"], ep=export["ep"]),
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


def plan(manifest: Manifest, repo_filter: tuple[str, ...] = ()) -> list[Publication]:
    """Every checkpoint that belongs on the Hub, in publication order."""
    publications = []
    for model in manifest.models:
        if repo_filter and not any(f in model.repo for f in repo_filter):
            continue
        repo_dir = manifest.export_root / model.repo.split("/")[1]
        for stage in model.stages:
            for iteration, source in stage_sources(stage):
                publications.append(
                    Publication(
                        model=model,
                        stage=stage,
                        iteration=iteration,
                        source=source,
                        clone_root=repo_dir / stage.name,
                        revision=stage.revision.replace(ITERATION_FIELD, str(iteration)),
                        default=stage.default and iteration == stage.train_iters,
                        tokens_seen=(stage.tokens_before + iteration) * manifest.tokens_per_iteration,
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


def export_command(publication: Publication, manifest: Manifest) -> list[str]:
    """The exporter invocation for a publication (run from the repo root)."""
    command = [
        "bash",
        EXPORTER,
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
        command.append("--not-strict")
    return command


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
    """The tensor names a safetensors file holds, from its header (an 8-byte length then JSON)."""
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    return {name for name in header if name != "__metadata__"}


def verify_export(hf_dir: Path) -> int:
    """Check an export by tensor names in both directions and return how many tensors it holds."""
    index_path = hf_dir / INDEX_FILE
    if not index_path.is_file():
        raise ExportError(f"{hf_dir}: no {INDEX_FILE}")
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
    return len(weight_map)


def export_is_verified(publication: Publication) -> bool:
    """Whether a verified export already sits in the clone."""
    if not publication.hf_dir.is_dir():
        return False
    try:
        verify_export(publication.hf_dir)
    except ExportError as error:
        LOGGER.warning("%s: existing export rejected, will export again: %s", publication.label, error)
        return False
    return True


def local_files(hf_dir: Path) -> dict[str, int]:
    """The export's files and sizes, as the Hub will hold them."""
    return {p.name: p.stat().st_size for p in hf_dir.iterdir() if p.is_file() and not p.name.startswith(".")}


def published(api: Any, repo: str, revision: str, hf_dir: Path) -> bool:
    """Whether the revision already holds every file of the export at the same size."""
    from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

    try:
        remote = {entry.path: entry.size for entry in api.list_repo_tree(repo, revision=revision, recursive=False)}
    except (RepositoryNotFoundError, RevisionNotFoundError):
        return False
    return all(remote.get(name) == size for name, size in local_files(hf_dir).items())


def upload(api: Any, publication: Publication) -> None:
    """Upload the export to its revision, and to main as well when it is the default."""
    api.create_repo(publication.model.repo, private=publication.model.private, exist_ok=True)
    targets = [publication.revision] + ([MAIN] if publication.default else [])
    for revision in targets:
        if revision != MAIN:
            api.create_branch(publication.model.repo, branch=revision, exist_ok=True)
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
            f"| {stage.name} ({where}) | {stage.train_iters:,} | {_tokens(stage.train_iters * manifest.tokens_per_iteration)} "
            f"| `{stage.wandb_exp_name}` |"
        )
    lines += [
        "",
        f"Tokens per iteration: {manifest.tokens_per_iteration:,} at every stage, so the token count of a "
        "checkpoint is its iteration plus the iterations of the stages before it, times that.",
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
    """Upload the card to main unless the copy uploaded last time is identical; returns whether it did."""
    staging.mkdir(parents=True, exist_ok=True)
    card = staging / README_NAME
    if card.is_file() and card.read_text() == text:
        return False
    card.write_text(text)
    api.upload_file(
        path_or_fileobj=str(card),
        path_in_repo=README_NAME,
        repo_id=model.repo,
        revision=MAIN,
        commit_message="Model card: revisions, tokens seen, training loss",
    )
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


def publish_pass(
    manifest: Manifest,
    repo_root: Path,
    api: Any,
    wandb_api: Any,
    run_dir: Path,
    execute: bool,
    repo_filter: tuple[str, ...],
) -> int:
    """One pass over the manifest: export and upload what is missing, then the cards and the
    collection for every model that has anything published. Returns how many publications
    remain unpublished (0 when the Hub holds everything the manifest asks for)."""
    publications = plan(manifest, repo_filter)
    export_log = run_dir / "export.log"
    pending = 0
    touched: dict[str, list[Publication]] = {}
    for publication in publications:
        if publication.hf_dir.is_dir() and published(
            api, publication.model.repo, publication.revision, publication.hf_dir
        ):
            LOGGER.info("%s: already on the Hub", publication.label)
            touched.setdefault(publication.model.repo, []).append(publication)
            continue
        if not execute:
            LOGGER.info(
                "%s: would %s", publication.label, "upload" if export_is_verified(publication) else "export and upload"
            )
            pending += 1
            continue
        try:
            if not export_is_verified(publication):
                make_export_clone(publication.source, publication.clone)
                run_export(publication, manifest, repo_root, export_log)
                count = verify_export(publication.hf_dir)
                LOGGER.info("%s: export verified, %d tensors", publication.label, count)
            upload(api, publication)
            touched.setdefault(publication.model.repo, []).append(publication)
        except ExportError as error:
            LOGGER.error("%s: %s", publication.label, error)
            pending += 1
    if not execute:
        return pending
    namespace = manifest.models[0].repo.split("/")[0]
    for model in manifest.models:
        rows_pubs = [p for p in publications if p.model is model and p.model.repo in touched]
        if not rows_pubs:
            continue
        rows = []
        for stage in model.stages:
            stage_pubs = [p for p in rows_pubs if p.stage is stage]
            cache = run_dir.parent / "losses" / f"{stage.wandb_exp_name}.json"
            found = losses(wandb_api, manifest.wandb, stage.wandb_exp_name, {p.iteration for p in stage_pubs}, cache)
            rows += [(p, found.get(p.iteration)) for p in stage_pubs]
        # Cards persist beside the run directories so an unchanged card is not re-uploaded every pass.
        card_dir = run_dir.parent / "cards" / model.repo.split("/")[1]
        if upload_model_card(api, model, render_model_card(manifest, model, rows), card_dir):
            LOGGER.info("%s: model card updated", model.repo)
    ensure_collection(api, manifest.collection, namespace, [m.repo for m in manifest.models if m.repo in touched])
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
    api = sync_bucket.make_api()
    wandb_api = None if args.plan else make_wandb_api()
    deadline = time.time() + args.stop_after * 3600 if args.stop_after else None
    while True:
        pending = publish_pass(manifest, repo_root, api, wandb_api, run_dir, not args.plan, tuple(args.models))
        LOGGER.info("pass complete: %d publication(s) pending", pending)
        if args.poll_interval is None or (deadline is not None and time.time() >= deadline):
            return 1 if pending else 0
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    sys.exit(main())

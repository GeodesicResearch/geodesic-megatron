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

"""Mirror a campaign's Megatron checkpoints and training corpora into a Hugging Face bucket.

A manifest (``--manifest``, YAML) names the bucket and the campaign's stage configs. Each config
contributes its ``checkpoint.save`` directory and the corpora its ``dataset`` section reads, so the
archive follows the configs instead of restating their paths; a stage whose directory does not
exist yet is reported on every pass and picked up once it does. Checkpoint directories no config
names are listed explicitly. The tool turns that into ``hf buckets sync`` calls and checks
afterwards that nothing is left to upload. It runs on a host Python whose
``huggingface_hub`` knows buckets (>= 1.19) — the training container's copy does not — and it
never opens a large file itself: the Hub client streams them.

Two rules make the mirror safe to run beside live training:

* **Only completed checkpoints are synced.** Megatron writes ``iter_XXXXXXX/`` first and updates
  ``latest_checkpointed_iteration.txt`` after the save has finished on every rank, so an iteration
  above the tracker's value is a save in progress. Those directories are left alone until the
  tracker moves past them; the tracker itself and the other small root files are uploaded after
  the iteration directories they point at, from a snapshot taken when the pass began, and only if
  the tracker has not advanced meanwhile.
* **Comparison is by size alone** (``--ignore-times``): checkpoint shards and tokenized corpora are
  written once and never modified, and Lustre mtimes carry no information about them. A re-run
  therefore uploads exactly what is missing and skips the rest, at metadata cost only.

What is deliberately excluded is stated per entry rather than left implicit: the ``hf/``
Hugging Face export inside a checkpoint directory (a derived artifact that
``pipeline_checkpoint_submit.sbatch export`` regenerates), and for a corpus everything except the
``.bin``/``.idx``/``.provenance.json`` triple and the prepare record beside it (the JSONL the
tokenizer consumed and the index caches training builds are both reproducible from those).

Every pass writes its resolved manifest, per-unit sync plans, an inventory TSV and its log under
``<log_dir>/<run id>/`` and mirrors that directory into the bucket's provenance prefix, so the
bucket records how it was built. ``--poll-interval`` repeats the pass until ``--stop-after``
hours have elapsed, re-reading the manifest each time so a stage added to it mid-run is picked up.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


LOGGER = logging.getLogger("sync_bucket")

LATEST_FILE = "latest_checkpointed_iteration.txt"
# The small files at a checkpoint directory's root that a resume reads besides ``iter_*``.
CHECKPOINT_ROOT_FILES = (
    LATEST_FILE,
    "latest_train_state.pt",
    "progress.txt",
    "latest_wandb_artifact_path.txt",
    "ft_state.json",
)
HF_EXPORT_EXCLUDE = "hf/*"
ITER_DIR = re.compile(r"^iter_(\d{7})$")
# The artifacts ``pipeline_data_submit.sbatch tokenize`` writes for a corpus prefix, plus the
# prepare step's record of dataset, subset and revision beside them.
CORPUS_SUFFIXES = (".bin", ".idx", ".provenance.json")
PREPARE_RECORD = "pipeline_results.json"
# The data pipeline's layout under a corpus root (the slugified ``dataset__subset`` directory):
# the tokenized prefix sits in the root or in ``shardN/`` roots, and packed sequences under
# ``packed/<tokenizer>/``, again possibly inside a shard. That fixed shape is what lets a corpus
# directory be placed in the bucket without being told which data base holds it.
SHARD_DIR = re.compile(r"^shard\d+$")
PACKED_DIR = "packed"
EXTRA_CHECKPOINT_KEYS = frozenset({"beside", "directory", "remote"})
MANIFEST_KEYS = frozenset(
    {
        "bucket",
        "readme",
        "provenance_prefix",
        "datasets_prefix",
        "log_dir",
        "hf_home",
        "checkpoints_prefix",
        "stage_configs",
        "extra_checkpoints",
    }
)
INVENTORY_COLUMNS = (
    "remote",
    "local",
    "files",
    "bytes",
    "uploaded_files",
    "uploaded_bytes",
    "skipped_files",
    "synced_at",
)
README_NAME = "README.md"
INVENTORY_NAME = "INVENTORY.tsv"


@dataclass(frozen=True)
class CheckpointEntry:
    """One Megatron save directory and the bucket prefix its completed iterations go under."""

    local: Path
    remote: str


@dataclass(frozen=True)
class Manifest:
    """The archive's definition: where it lives, what goes in, and where each piece comes from."""

    bucket: str
    readme: Path
    provenance_prefix: str
    checkpoints_prefix: str
    datasets_prefix: str
    log_dir: Path
    hf_home: Path
    stage_configs: tuple[Path, ...]
    extra_checkpoints: tuple[CheckpointEntry, ...]


@dataclass(frozen=True)
class SyncUnit:
    """One ``sync_bucket`` call: a local directory onto a bucket prefix, with its filters."""

    label: str
    source: Path
    remote: str
    include: tuple[str, ...] | None
    exclude: tuple[str, ...]


class ManifestError(ValueError):
    """The manifest asks for something that cannot be archived as written."""


def utc_now() -> str:
    """The current time as the ISO-8601 UTC stamp used in run ids and inventory rows."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iteration_number(iter_dir: Path) -> int:
    """The iteration an ``iter_XXXXXXX`` directory holds."""
    match = ITER_DIR.match(iter_dir.name)
    if match is None:
        raise ValueError(f"{iter_dir} is not an iter_XXXXXXX directory")
    return int(match.group(1))


def exact_keys(mapping: Any, keys: frozenset[str], where: str) -> dict[str, Any]:
    """A manifest mapping with exactly ``keys``; anything else names what is unknown and missing."""
    if not isinstance(mapping, dict):
        raise ManifestError(f"{where}: expected a mapping, got {type(mapping).__name__}")
    unknown = sorted(set(mapping) - keys)
    missing = sorted(keys - set(mapping))
    if unknown or missing:
        raise ManifestError(
            f"{where}: expected exactly the keys {sorted(keys)}; unknown keys {unknown}, missing keys {missing}"
        )
    return mapping


def load_manifest(path: Path, repo_root: Path) -> Manifest:
    """Read and validate a manifest; repo-relative paths resolve against ``repo_root``."""
    raw = exact_keys(yaml.safe_load(path.read_text()), MANIFEST_KEYS, str(path))
    bucket = str(raw["bucket"])
    if bucket.count("/") != 1 or bucket.startswith("hf://"):
        raise ManifestError(f"{path}: bucket must be <namespace>/<name>, got {bucket!r}")
    checkpoints_prefix = str(raw["checkpoints_prefix"]).strip("/")
    extra = []
    for index, item in enumerate(raw["extra_checkpoints"]):
        exact_keys(item, EXTRA_CHECKPOINT_KEYS, f"{path}: extra_checkpoints[{index}]")
        remote = str(item["remote"]).strip("/")
        directory = str(item["directory"])
        if not remote or not directory or "/" in directory:
            raise ManifestError(f"{path}: extra_checkpoints[{index}] needs a remote prefix and a bare directory name")
        beside = repo_root / str(item["beside"])
        if not beside.is_file():
            raise ManifestError(f"{path}: extra_checkpoints[{index}].beside {beside} does not exist")
        # The extra directory sits beside the named stage's save directory, in the same tree.
        local = stage_checkpoint_entry(beside, checkpoints_prefix).local.parent / directory
        extra.append(CheckpointEntry(local=local, remote=remote))
    if not isinstance(raw["readme"], str):
        raise ManifestError(f"{path}: readme must be a repo-relative path string")
    stage_configs = tuple(repo_root / c for c in raw["stage_configs"])
    absent = [str(c) for c in stage_configs if not c.is_file()]
    if absent:
        raise ManifestError(f"{path}: stage_configs that do not exist: {absent}")
    readme = repo_root / raw["readme"]
    if not readme.is_file():
        raise ManifestError(f"{path}: readme {readme} does not exist")
    return Manifest(
        bucket=bucket,
        readme=readme,
        provenance_prefix=str(raw["provenance_prefix"]).strip("/"),
        checkpoints_prefix=checkpoints_prefix,
        datasets_prefix=str(raw["datasets_prefix"]).strip("/"),
        log_dir=Path(raw["log_dir"]),
        hf_home=Path(raw["hf_home"]),
        stage_configs=stage_configs,
        extra_checkpoints=tuple(extra),
    )


def stage_save_directory(config: Path) -> Path:
    """The directory a stage config saves its checkpoints to (``checkpoint.save``)."""
    cfg = yaml.safe_load(config.read_text())
    save = (cfg.get("checkpoint") or {}).get("save") if isinstance(cfg, dict) else None
    if not isinstance(save, str) or not save.startswith("/"):
        raise ManifestError(f"{config}: checkpoint.save must be an absolute path, got {save!r}")
    return Path(save)


def stage_checkpoint_entry(config: Path, checkpoints_prefix: str) -> CheckpointEntry:
    """The save directory a stage config writes, archived under the directory's own name."""
    local = stage_save_directory(config)
    return CheckpointEntry(local=local, remote=f"{checkpoints_prefix}/{local.name}")


def completed_iterations(checkpoint_dir: Path) -> list[Path]:
    """The ``iter_*`` directories whose save has finished, oldest first.

    A directory above the tracker's value is a save in progress (or one that died mid-write) and
    is not returned. No tracker means no save has completed yet, which is a normal state for a
    stage that has just started — it is reported, not treated as an error.
    """
    if not checkpoint_dir.is_dir():
        raise ManifestError(f"checkpoint directory does not exist: {checkpoint_dir}")
    tracker = checkpoint_dir / LATEST_FILE
    if not tracker.is_file():
        LOGGER.info("%s: no %s yet, so no completed checkpoint to archive", checkpoint_dir, LATEST_FILE)
        return []
    latest = int(tracker.read_text().strip())
    found = []
    for child in checkpoint_dir.iterdir():
        match = ITER_DIR.match(child.name)
        if match and child.is_dir():
            found.append((int(match.group(1)), child))
    return [path for iteration, path in sorted(found) if iteration <= latest]


def checkpoint_units(entry: CheckpointEntry, iterations: list[Path]) -> list[SyncUnit]:
    """One unit per completed iteration directory, each excluding the ``hf/`` export."""
    return [
        SyncUnit(
            label=f"{entry.local.name}/{path.name}",
            source=path,
            remote=f"{entry.remote}/{path.name}",
            include=None,
            exclude=(HF_EXPORT_EXCLUDE,),
        )
        for path in iterations
    ]


def stage_root_files(entry: CheckpointEntry, latest: Path, staging: Path) -> SyncUnit | None:
    """Snapshot the root files that point at ``latest`` and return the unit that uploads them.

    The snapshot is what gets uploaded, not the live files, so the tracker in the bucket can never
    name an iteration whose shards are not there yet. If the live tracker has already moved past
    ``latest`` by the time the copy is taken, the root files are left for the next pass (which will
    have archived the newer iteration first) and ``None`` is returned.
    """
    staging.mkdir(parents=True, exist_ok=True)
    for name in CHECKPOINT_ROOT_FILES:
        source = entry.local / name
        if source.is_file():
            shutil.copy2(source, staging / name)
    live = int((entry.local / LATEST_FILE).read_text().strip())
    staged = int((staging / LATEST_FILE).read_text().strip())
    expected = iteration_number(latest)
    if live != expected or staged != expected:
        LOGGER.warning(
            "%s: tracker advanced to %d while archiving up to %d; root files deferred to the next pass",
            entry.local,
            max(live, staged),
            expected,
        )
        return None
    return SyncUnit(
        label=f"{entry.local.name}/root files",
        source=staging,
        remote=entry.remote,
        include=CHECKPOINT_ROOT_FILES,
        exclude=(),
    )


def corpus_relative(directory: Path) -> Path:
    """Where a corpus directory sits relative to the data base that holds it, from the layout alone.

    The corpus root is the slugified ``dataset__subset`` directory. ``directory`` is either that
    root, one of its ``shardN/`` roots, or a ``packed/<tokenizer>/`` directory under one of those,
    so the root is found by stepping out of the ``packed`` tree and then out of a shard, and the
    data base is the root's parent. The result names the same directory the training configs do,
    whichever data base it was built under.
    """
    parts = directory.parts
    if PACKED_DIR in parts:
        root = Path(*parts[: len(parts) - 1 - parts[::-1].index(PACKED_DIR)])
    else:
        root = directory
    if SHARD_DIR.match(root.name):
        root = root.parent
    if not root.name or root.parent == root:
        raise ManifestError(f"{directory} has no corpus root to place it under")
    return directory.relative_to(root.parent)


def dataset_units(config: Path, datasets_prefix: str) -> list[SyncUnit]:
    """The corpora a training config reads, as sync units.

    ``dataset.data_path`` is Megatron's flat ``[weight, prefix, weight, prefix, ...]`` list; every
    prefix contributes its ``.bin``/``.idx``/``.provenance.json`` and the prepare record beside
    them. ``dataset.packed_sequence_specs.packed_train_data_path`` is the packed-SFT parquet (a
    glob over shards); every match contributes its whole ``packed/<tokenizer>/`` directory, which
    holds the parquet, its row-group index, the pack manifest and the validation report. A config
    that names neither, or a glob that matches nothing, is an error: the archive would be claiming
    a stage's data without holding it.
    """
    cfg = yaml.safe_load(config.read_text())
    dataset = cfg.get("dataset") if isinstance(cfg, dict) else None
    if not isinstance(dataset, dict):
        raise ManifestError(f"{config}: no dataset section")
    units: list[SyncUnit] = []
    data_path = dataset.get("data_path")
    if data_path is not None:
        if not isinstance(data_path, list):
            raise ManifestError(
                f"{config}: dataset.data_path must be the flat weight/prefix list, got {type(data_path)}"
            )
        prefixes = [Path(item) for item in data_path if isinstance(item, str) and item.startswith("/")]
        if not prefixes:
            raise ManifestError(f"{config}: dataset.data_path names no absolute corpus prefix")
        for prefix in prefixes:
            units.append(
                SyncUnit(
                    label=f"{config.name}: {prefix.parent.name}/{prefix.name}",
                    source=prefix.parent,
                    remote=f"{datasets_prefix}/{corpus_relative(prefix.parent).as_posix()}",
                    include=tuple(f"{prefix.name}{suffix}" for suffix in CORPUS_SUFFIXES) + (PREPARE_RECORD,),
                    exclude=(),
                )
            )
    packed = (dataset.get("packed_sequence_specs") or {}).get("packed_train_data_path")
    if packed is not None:
        matches = sorted(glob.glob(str(packed)))
        if not matches:
            raise ManifestError(f"{config}: packed_train_data_path matches nothing: {packed}")
        for parquet in map(Path, matches):
            units.append(
                SyncUnit(
                    label=f"{config.name}: {parquet.parent.parent.parent.name}/{parquet.parent.name}",
                    source=parquet.parent,
                    remote=f"{datasets_prefix}/{corpus_relative(parquet.parent).as_posix()}",
                    include=None,
                    exclude=(),
                )
            )
    if not units:
        raise ManifestError(f"{config}: names neither dataset.data_path nor a packed_train_data_path")
    return units


def dedupe(units: list[SyncUnit]) -> list[SyncUnit]:
    """Drop repeats of the same source onto the same prefix; refuse two sources onto one prefix."""
    seen: dict[str, SyncUnit] = {}
    kept = []
    for unit in units:
        previous = seen.get(unit.remote)
        if previous is None:
            seen[unit.remote] = unit
            kept.append(unit)
        elif previous.source != unit.source:
            raise ManifestError(
                f"two different directories would land on {unit.remote}: {previous.source} and {unit.source}"
            )
    return kept


def plan_units(manifest: Manifest, staging_root: Path) -> list[SyncUnit]:
    """Everything one pass will sync: every completed iteration, then each prefix's root files, then data.

    The checkpoint directories are the stage configs' ``checkpoint.save`` plus the manifest's
    explicit extras. A stage whose directory does not exist yet has not started: it is reported
    and skipped, and the next pass after its first save picks it up. An explicit extra must exist.

    Two entries may feed one bucket prefix — an iteration recovered from a clone beside its run's
    own directory — so the root files of a prefix come from whichever entry holds the newest
    completed iteration (the tracker must name the newest shards present), and the same iteration
    arriving from two places is an error rather than a silent overwrite.
    """
    units: list[SyncUnit] = []
    newest: dict[str, tuple[Path, CheckpointEntry]] = {}
    entries: list[CheckpointEntry] = []
    for config in manifest.stage_configs:
        entry = stage_checkpoint_entry(config, manifest.checkpoints_prefix)
        if not entry.local.is_dir():
            LOGGER.warning(
                "%s: checkpoint.save %s does not exist yet — stage not started, nothing to archive",
                config.name,
                entry.local,
            )
            continue
        entries.append(entry)
    entries.extend(manifest.extra_checkpoints)
    for entry in entries:
        iterations = completed_iterations(entry.local)
        units.extend(checkpoint_units(entry, iterations))
        if iterations:
            current = newest.get(entry.remote)
            if current is None or iteration_number(iterations[-1]) > iteration_number(current[0]):
                newest[entry.remote] = (iterations[-1], entry)
    for latest, entry in newest.values():
        root_unit = stage_root_files(entry, latest, staging_root / entry.local.name)
        if root_unit is not None:
            units.append(root_unit)
    for config in manifest.stage_configs:
        units.extend(dataset_units(config, manifest.datasets_prefix))
    return dedupe(units)


def bucket_uri(bucket: str, prefix: str) -> str:
    """The ``hf://buckets/`` URI of a prefix inside the bucket (the root when the prefix is empty)."""
    return f"hf://buckets/{bucket}/{prefix}" if prefix else f"hf://buckets/{bucket}"


def configure_hf_home(hf_home: Path) -> Path:
    """Point the Hub client's caches at ``hf_home`` unless ``HF_HOME`` is already exported; refuse $HOME.

    The Xet client stages every upload's chunks under ``HF_HOME``, and terabytes of checkpoint
    shards cannot pass through the home directory's quota. This must run before
    ``huggingface_hub`` is imported, which reads the variable once.
    """
    os.environ.setdefault("HF_HOME", str(hf_home))
    resolved = Path(os.environ["HF_HOME"]).expanduser().resolve()
    home = Path.home().resolve()
    if resolved == home or home in resolved.parents:
        raise RuntimeError(
            f"HF_HOME={resolved} is under the home directory {home}; the Hub client's upload cache cannot live there"
        )
    return resolved


def make_api():
    """The Hub client, from a ``huggingface_hub`` that knows buckets."""
    import huggingface_hub

    api_cls = huggingface_hub.HfApi
    if not hasattr(api_cls, "sync_bucket"):
        raise RuntimeError(
            f"huggingface_hub {huggingface_hub.__version__} at {huggingface_hub.__file__} has no bucket support; "
            "run this on a host Python with huggingface_hub >= 1.19"
        )
    return api_cls()


def run_unit(api: Any, bucket: str, unit: SyncUnit, plan_file: Path, execute: bool) -> dict[str, Any]:
    """Sync one unit (or only plan it) and return its inventory row plus the plan summary."""
    kwargs: dict[str, Any] = dict(
        include=list(unit.include) if unit.include else None,
        exclude=list(unit.exclude) if unit.exclude else None,
        ignore_times=True,
        quiet=True,
    )
    if not execute:
        kwargs["plan"] = str(plan_file)
    plan = api.sync_bucket(str(unit.source), bucket_uri(bucket, unit.remote), **kwargs)
    summary = plan.summary()
    in_scope = [op for op in plan.operations if op.action in ("upload", "skip")]
    row = {
        "remote": unit.remote,
        "local": str(unit.source),
        "files": len(in_scope),
        "bytes": sum(op.size or 0 for op in in_scope),
        "uploaded_files": summary["uploads"],
        "uploaded_bytes": summary["total_size"],
        "skipped_files": summary["skips"],
        "synced_at": utc_now(),
    }
    LOGGER.info(
        "%s [%s] %s -> %s: %d files, %.1f GB; %s %d files, %.1f GB; skipped %d",
        "synced" if execute else "planned",
        unit.label,
        unit.source,
        unit.remote,
        row["files"],
        row["bytes"] / 1e9,
        "uploaded" if execute else "would upload",
        row["uploaded_files"],
        row["uploaded_bytes"] / 1e9,
        row["skipped_files"],
    )
    return row


def write_inventory(rows: list[dict[str, Any]], path: Path) -> None:
    """One TSV row per synced unit, in the pass's order."""
    lines = ["\t".join(INVENTORY_COLUMNS)]
    lines.extend("\t".join(str(row[column]) for column in INVENTORY_COLUMNS) for row in rows)
    path.write_text("\n".join(lines) + "\n")


def verify_units(api: Any, bucket: str, units: list[SyncUnit], plan_dir: Path) -> int:
    """Re-plan every unit after the pass; the number of uploads still pending, which must be 0."""
    pending = 0
    for index, unit in enumerate(units):
        row = run_unit(api, bucket, unit, plan_dir / f"verify_{index:04d}.jsonl", execute=False)
        if row["uploaded_files"]:
            LOGGER.error("%s still has %d files to upload after the pass", unit.remote, row["uploaded_files"])
        pending += row["uploaded_files"]
    return pending


def publish(api: Any, manifest: Manifest, run_dir: Path, rows: list[dict[str, Any]], execute: bool) -> None:
    """Put the README and inventory at the bucket root and mirror this run's record under provenance."""
    root_staging = run_dir / "root"
    root_staging.mkdir(exist_ok=True)
    shutil.copy2(manifest.readme, root_staging / README_NAME)
    write_inventory(rows, root_staging / INVENTORY_NAME)
    root_unit = SyncUnit("bucket root", root_staging, "", (README_NAME, INVENTORY_NAME), ())
    run_unit(api, manifest.bucket, root_unit, run_dir / "plans" / "root.jsonl", execute)
    record_unit = SyncUnit(
        "provenance", run_dir, f"{manifest.provenance_prefix}/{run_dir.name}", None, ("root/*", "staging/*")
    )
    run_unit(api, manifest.bucket, record_unit, run_dir / "plans" / "provenance.jsonl", execute)


def sync_pass(api: Any, manifest_path: Path, repo_root: Path, execute: bool) -> int:
    """One full pass over the manifest. Returns the number of files still pending afterwards."""
    manifest = load_manifest(manifest_path, repo_root)
    run_id = utc_now().replace(":", "") + (f"-j{os.environ['SLURM_JOB_ID']}" if "SLURM_JOB_ID" in os.environ else "")
    run_dir = manifest.log_dir / run_id
    (run_dir / "plans").mkdir(parents=True)
    handler = logging.FileHandler(run_dir / "sync.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    try:
        shutil.copy2(manifest_path, run_dir / "manifest.yaml")
        (run_dir / "resolved_manifest.yaml").write_text(yaml.safe_dump(_as_plain(manifest), sort_keys=False))
        LOGGER.info(
            "pass %s as %s onto %s (%s)",
            run_id,
            api.whoami()["name"],
            manifest.bucket,
            "execute" if execute else "plan only",
        )
        units = plan_units(manifest, run_dir / "staging")
        LOGGER.info("%d sync units", len(units))
        rows = [
            run_unit(api, manifest.bucket, unit, run_dir / "plans" / f"{index:04d}.jsonl", execute)
            for index, unit in enumerate(units)
        ]
        write_inventory(rows, run_dir / INVENTORY_NAME)
        total_files = sum(r["files"] for r in rows)
        total_bytes = sum(r["bytes"] for r in rows)
        moved_bytes = sum(r["uploaded_bytes"] for r in rows)
        LOGGER.info(
            "pass total: %d files, %.2f TB in scope; %s %.2f TB",
            total_files,
            total_bytes / 1e12,
            "uploaded" if execute else "would upload",
            moved_bytes / 1e12,
        )
        pending = (
            verify_units(api, manifest.bucket, units, run_dir / "plans")
            if execute
            else sum(r["uploaded_files"] for r in rows)
        )
        if execute:
            publish(api, manifest, run_dir, rows, execute=True)
        LOGGER.info("pass %s done: %d files pending", run_id, pending)
        return pending
    finally:
        LOGGER.removeHandler(handler)
        handler.close()


def _as_plain(manifest: Manifest) -> dict[str, Any]:
    plain = dataclasses.asdict(manifest)

    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        return value

    return convert(plain)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """The command line: a manifest, plan-only or execute, one pass or a polling loop with a stop time."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True, help="the archive manifest (YAML)")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="what the manifest's repo-relative paths resolve against",
    )
    parser.add_argument("--plan-only", action="store_true", help="compute and save the sync plans; upload nothing")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=None,
        help="seconds between passes; without it exactly one pass runs",
    )
    parser.add_argument(
        "--stop-after",
        type=float,
        default=None,
        help="hours after which polling stops (required with --poll-interval)",
    )
    args = parser.parse_args(argv)
    if (args.poll_interval is None) != (args.stop_after is None):
        parser.error("--poll-interval and --stop-after go together")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the passes; 0 when every executed pass left nothing pending, 1 otherwise."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    # The Hub client logs every HTTP request at INFO; the per-unit lines above are the record.
    for chatty in ("httpx", "huggingface_hub"):
        logging.getLogger(chatty).setLevel(logging.WARNING)
    # The manifest is re-read every pass, but the Hub client's cache location is fixed at import.
    LOGGER.info("HF_HOME=%s", configure_hf_home(load_manifest(args.manifest, args.repo_root).hf_home))
    api = make_api()
    execute = not args.plan_only
    deadline = time.monotonic() + args.stop_after * 3600 if args.stop_after is not None else None
    while True:
        pending = sync_pass(api, args.manifest, args.repo_root, execute)
        if execute and pending:
            LOGGER.error("%d files remain unsynced after the pass and its verification", pending)
            return 1
        if deadline is None:
            return 0
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            LOGGER.info("stop-after reached")
            return 0
        LOGGER.info(
            "next pass in %.0f s (%.1f h to the stop time)", min(args.poll_interval, remaining), remaining / 3600
        )
        time.sleep(min(args.poll_interval, remaining))


if __name__ == "__main__":
    sys.exit(main())

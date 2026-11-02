# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The repair that lets the Megatron-to-HF exporter read a checkpoint trained with
``moe_experts_impl: torch_grouped``. Both the standard exporter (``pipeline_checkpoint_convert.sh export``)
and the Hub publisher (``scripts/hub/publish_models.py``) use it.

The exporter rebuilds the Megatron model from the checkpoint's ``run_config.yaml``. A ``torch_grouped`` run
trained before ``MambaModelProvider`` resolved its stack spec at build time records that spec as a nested
closure (``MambaModelProvider._apply_moe_experts_impl.<locals>._grouped_resolved_stack_spec``). Nothing can
import it, so the export dies in ``load_model_config`` before it reads a weight.

``RUN_CONFIG_EDITS`` points the closure back at the module-level stack spec and switches the experts to the TE
implementation. The weights on disk are canonical either way, because ``GroupedExperts.sharded_state_dict``
writes ``experts.linear_fc{1,2}`` keys, so no checkpoint needs retraining. The bridge exports either expert
implementation (``GroupedExpertsMapping`` maps ``GroupedExperts``). A full checkpoint still takes the TE switch,
which keeps its export on the path the export checks and the Hub publisher were validated on. An adapter-only
PEFT checkpoint (a run_config with a ``peft`` section) takes the stack-spec edit alone (``run_config_edits``):
its Expert LoRA wraps ``GroupedExperts``, which ``te_grouped`` does not build.

The edits are never applied to the checkpoint itself. An **export clone** is a directory of symlinks to one
iteration's files plus a repaired copy of its ``run_config.yaml``, built outside the training tree, and the
exporter loads the model from it (``resolve_load_path``).

Command line (``pipeline_checkpoint_convert.sh export`` runs both inside the container, from the repo root):

    python -m pipeline.export.export_clone prepare --megatron-path DIR [--iteration N] --job ID
    python -m pipeline.export.export_clone remove --clone CLONE

``prepare`` resolves the iteration: ``--iteration`` when given, else the tracker, else the newest ``iter_*``
directory. It builds a clone only when that iteration's run_config needs the repair, at
``ROOT/<ID>/<run>-<digest>/iter_<n>``, where ``<ID>`` names this one export (the launcher passes its
SLURM job id and its own pid). It prints ``EXPORT_ITERATION=<n>`` and ``EXPORT_LOAD_PATH=<clone>`` on
stdout for the launcher to read; the load path is empty when no repair is needed. ``remove`` deletes a clone
once its export has finished. ``ROOT`` is ``$GEODESIC_EXPORT_CLONE_ROOT`` when set, else
``/projects/a5k/public/tmp/export_clones_$USER`` (``default_clone_root``), so a launcher states it only to move it.

Catalogue:
    RUN_CONFIG_EDITS: the torch_grouped repair's run_config edits, as (old, new) string pairs.
    ExportError: a checkpoint cannot be exported as it stands.
    ExportSource: what one export reads, the iteration and the directory its model loads from.
    run_config_edits: the edits a run_config takes (the stack-spec edit alone for a PEFT checkpoint).
    patch_run_config: apply the edits to a run_config's text.
    needs_repair: whether a run_config's text records anything the edits replace.
    checkpoint_needs_repair: whether an iteration directory's run_config needs the repair.
    make_export_clone: build an export clone of an iteration.
    clone_location: where ``prepare`` builds a clone.
    check_clone_root: refuse a clone root that is relative, under $HOME or in the training tree.
    check_clone_is_safe_to_write: refuse to build a clone through a link.
    prepare_export_source: decide what an export reads, building the clone if it needs one.
    check_is_export_clone: refuse to load from anything but an export clone of the iteration.
    resolve_load_path: the directory the exporter loads the model from, the iteration or its clone.
    remove_export_clone: delete a clone once its export has finished.
    default_clone_root: the root clones are built under.
    main: the command line.
"""

from __future__ import annotations

import argparse
import errno
import getpass
import hashlib
import logging
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from pipeline.checkpoints import ITER_DIR, LATEST_FILE, RUN_CONFIG, iteration_number, resolve_checkpoint_path


LOGGER = logging.getLogger(__name__)

HF_DIR = "hf"
# A job id names the clone's directory, so it must be one plain path component.
JOB_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
# Where clones are built when no root is given: this variable, else a directory per user on project storage (never
# $HOME, which ``check_clone_root`` refuses).
CLONE_ROOT_ENV = "GEODESIC_EXPORT_CLONE_ROOT"
PROJECT_CLONE_ROOT = "/projects/a5k/public/tmp/export_clones_{user}"

# The lines `prepare` prints on stdout, which the launcher reads by these names.
PLAN_ITERATION = "EXPORT_ITERATION"
PLAN_LOAD_PATH = "EXPORT_LOAD_PATH"

# The exporter rebuilds the Megatron model from the checkpoint's run_config.yaml. An older
# torch_grouped checkpoint records the stack spec as a nested closure, which cannot be imported; the
# first edit points it back at the module-level spec. The second loads a full checkpoint's experts as
# te_grouped, the path its export checks were validated on (see the module docstring). The weights on
# disk are canonical either way (GroupedExperts.sharded_state_dict writes canonical keys).
RUN_CONFIG_EDITS = (
    (
        "megatron.bridge.models.mamba.mamba_provider.MambaModelProvider._apply_moe_experts_impl.<locals>._grouped_resolved_stack_spec",
        "megatron.bridge.models.mamba.mamba_provider.get_default_mamba_stack_spec",
    ),
    ("moe_experts_impl: torch_grouped", "moe_experts_impl: te_grouped"),
)


class ExportError(RuntimeError):
    """A checkpoint cannot be exported as it stands, or an export did not produce what it should."""


@dataclass(frozen=True)
class ExportSource:
    """What one export reads.

    ``iter_path`` is the checkpoint's own iteration directory. The export is named after it: its
    iteration, its default HF output ``iter_path/hf``, and the run_config copied into that output.
    ``load_path`` is the directory the model is loaded from, which is ``iter_path`` itself or its
    export clone."""

    iter_path: Path
    iteration: int
    load_path: Path

    @property
    def clone(self) -> Path | None:
        """The export clone the model is loaded from, or None when it is loaded from the checkpoint itself."""
        return None if self.load_path == self.iter_path else self.load_path


def run_config_edits(text: str) -> tuple[tuple[str, str], ...]:
    """The export edits the run_config ``text`` takes: ``RUN_CONFIG_EDITS``, or for an adapter-only PEFT checkpoint
    (a ``peft`` section) the stack-spec edit alone, since its Expert LoRA needs ``torch_grouped`` experts."""
    if (yaml.safe_load(text) or {}).get("peft"):
        return RUN_CONFIG_EDITS[:1]
    return RUN_CONFIG_EDITS


def patch_run_config(text: str) -> str:
    """Apply the export edits (``run_config_edits``) to a run_config, or accept one that already carries them."""
    for old, new in run_config_edits(text):
        if text.count(old) == 1:
            text = text.replace(old, new)
        elif text.count(new) >= 1 and old not in text:
            continue
        else:
            raise ExportError(f"run_config has {text.count(old)} occurrences of {old!r}; expected exactly one")
    return text


def needs_repair(text: str) -> bool:
    """Whether a run_config records anything the export edits replace.

    A run_config that does gets the edits from ``patch_run_config``, which refuses one that holds the
    strings in a shape it does not recognise. One that does not is exported as it stands, from the
    checkpoint itself; this includes every checkpoint that is not a ``torch_grouped`` hybrid, and an adapter-only
    PEFT checkpoint whose stack spec is importable."""
    return any(old in text for old, _ in run_config_edits(text))


def checkpoint_needs_repair(iter_dir: Path) -> bool:
    """Whether an iteration directory's run_config needs the export edits. One without a run_config
    needs none: there is nothing for the edits to repair."""
    run_config = iter_dir / RUN_CONFIG
    return run_config.is_file() and needs_repair(run_config.read_text())


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
    (clone.parent / LATEST_FILE).write_text(f"{iteration_number(source)}\n")


def clone_location(clone_root: Path, iter_path: Path, job: str) -> Path:
    """Where ``prepare`` builds the clone of ``iter_path`` for the export named ``job``.

    The path is ``<clone_root>/<job>/<run>-<digest>/<iter_N>``. ``<run>`` is the checkpoint directory's
    name, kept so a person can read the path. ``<digest>`` is a hash of that directory's resolved path,
    so two runs that share a name never share a clone. ``job`` names one export (the launcher passes its
    SLURM job id and its own pid), so two exports of the same checkpoint never touch each other's clone."""
    if not JOB_ID.match(job) or job in {".", ".."}:
        raise ExportError(f"job id {job!r} is not a plain path component")
    checkpoint_dir = iter_path.parent.resolve()
    digest = hashlib.sha256(str(checkpoint_dir).encode()).hexdigest()[:12]
    return clone_root / job / f"{checkpoint_dir.name}-{digest}" / iter_path.name


def check_clone_root(clone_root: Path, iter_path: Path) -> None:
    """Refuse a clone root that is relative, under $HOME, or inside the checkpoint's own directory.

    $HOME is refused because of its small quota. The checkpoint's directory is refused because
    building the clone there would write into the training tree, which the repair exists to leave
    untouched."""
    if not clone_root.is_absolute():
        raise ExportError(f"export clone root {clone_root} is not an absolute path")
    root = clone_root.resolve()
    home = Path.home().resolve()
    if root == home or root.is_relative_to(home):
        raise ExportError(f"export clone root {clone_root} is under $HOME ({home}); point it at project storage")
    checkpoint_dir = iter_path.parent.resolve()
    if root == checkpoint_dir or root.is_relative_to(checkpoint_dir):
        raise ExportError(
            f"export clone root {clone_root} is inside the checkpoint directory {checkpoint_dir}; "
            "a clone is built outside the training tree"
        )


def check_clone_is_safe_to_write(clone: Path, clone_root: Path) -> None:
    """Refuse to build a clone through a link. Building rewrites the clone's run_config and the tracker
    beside it, so the clone must resolve inside the clone root, and neither the clone, its run_config
    nor that tracker may be a link. A write through a link would reach whatever the link names, which
    could be a training run's own run_config or tracker."""
    for written, what in (
        (clone, "an export clone is a directory of its own"),
        (clone / RUN_CONFIG, "an export clone's run_config is a repaired copy"),
        (clone.parent / LATEST_FILE, "the tracker beside an export clone names the clone's iteration"),
    ):
        if written.is_symlink():
            raise ExportError(f"{written} is a link; {what}")
    resolved = clone.resolve()
    if not resolved.is_relative_to(clone_root.resolve()):
        raise ExportError(f"{clone} resolves to {resolved}, outside the export clone root {clone_root}")


def prepare_export_source(
    megatron_path: str | Path, iteration: int | None, clone_root: Path, job: str
) -> ExportSource:
    """Decide what an export of ``megatron_path`` at ``iteration`` reads, and build the clone if it needs one.

    When the iteration's run_config needs the export edits, the clone is built under ``clone_root``
    (``clone_location``) and the model is loaded from it. Otherwise nothing is written and the model is
    loaded from the iteration itself. The checkpoint is only ever read."""
    iter_path, iteration = resolve_checkpoint_path(megatron_path, iteration)
    if not checkpoint_needs_repair(iter_path):
        LOGGER.info("%s: run_config needs no export repair; the model loads from the checkpoint itself", iter_path)
        return ExportSource(iter_path=iter_path, iteration=iteration, load_path=iter_path)
    check_clone_root(clone_root, iter_path)
    clone = clone_location(clone_root, iter_path, job)
    check_clone_is_safe_to_write(clone, clone_root)
    make_export_clone(iter_path, clone)
    LOGGER.info(
        "%s: run_config needs the export repair; the model loads from the export clone %s",
        iter_path,
        clone,
    )
    return ExportSource(iter_path=iter_path, iteration=iteration, load_path=clone)


def check_is_export_clone(source: Path, clone: Path) -> None:
    """Refuse to load from ``clone`` unless it is an export clone of ``source``.

    An export clone of ``source`` has the same iteration name. It holds a link to each of the source's
    files (its hf/ excepted) and nothing else, and its run_config is the source's with the export edits
    applied. The exporter names its output and copies its provenance from ``source`` while it reads the
    weights from ``clone``, so a clone of any other checkpoint would publish one checkpoint's weights
    under another's name."""
    if clone.name != source.name:
        raise ExportError(f"{clone} is not a clone of {source}: it names a different iteration")
    if not clone.is_dir():
        raise ExportError(f"export clone {clone} is not a directory")
    source_run_config = source / RUN_CONFIG
    if not source_run_config.is_file():
        raise ExportError(f"{source} has no {RUN_CONFIG}, so no export clone of it can exist")
    expected = {entry.name: entry.resolve() for entry in source.iterdir() if entry.name not in (HF_DIR, RUN_CONFIG)}
    linked = {}
    for entry in clone.iterdir():
        if entry.name == RUN_CONFIG:
            continue
        if not entry.is_symlink():
            raise ExportError(f"{entry} is not a link; an export clone holds only links and its {RUN_CONFIG}")
        linked[entry.name] = entry.resolve()
    missing = sorted(expected.keys() - linked.keys())
    extra = sorted(linked.keys() - expected.keys())
    elsewhere = sorted(name for name in expected.keys() & linked.keys() if expected[name] != linked[name])
    if missing or extra or elsewhere:
        raise ExportError(
            f"{clone} is not a clone of {source}: missing links {missing[:3]}, extra links {extra[:3]}, "
            f"links to other files {elsewhere[:3]}"
        )
    run_config = clone / RUN_CONFIG
    if run_config.is_symlink() or not run_config.is_file():
        raise ExportError(f"{run_config} is not a repaired copy of {source_run_config}")
    if run_config.read_text() != patch_run_config(source_run_config.read_text()):
        raise ExportError(f"{run_config} is not {source_run_config} with the export edits applied")


def resolve_load_path(iter_path: Path, load_path: str | None) -> Path:
    """The directory the Megatron model is loaded from.

    Without ``--load-path`` this is the iteration itself. That is refused when the iteration's run_config
    needs the export repair (``checkpoint_needs_repair``): an unimportable stack-spec closure, or a full
    checkpoint's torch_grouped experts, which the repair loads as te_grouped. With ``--load-path`` it is the given export clone, once
    it is shown to be a clone of this very iteration. Either way the iteration, the HF output path, the
    run_config copied into the export and the Hub repo name all come from ``iter_path``.
    """
    if load_path is not None:
        clone = Path(load_path)
        check_is_export_clone(iter_path, clone)
        return clone
    if checkpoint_needs_repair(iter_path):
        raise ExportError(
            f"{iter_path / RUN_CONFIG} needs the export repair (an unimportable stack-spec closure, or a full "
            "checkpoint's moe_experts_impl: torch_grouped, which the repair loads as te_grouped). Export it through "
            "`pipeline_checkpoint_convert.sh export`, which builds an export "
            "clone itself, or build one with `python -m pipeline.export.export_clone prepare ...` and pass the "
            "EXPORT_LOAD_PATH it prints as --load-path."
        )
    return iter_path


def remove_export_clone(clone: Path, clone_root: Path) -> None:
    """Delete an export clone once its export has finished.

    This removes the clone's links, its repaired run_config, the tracker beside it once no other clone
    shares it, and each emptied directory above it, stopping below the clone root or at a directory that
    still holds another clone; a directory that cannot be removed for any other reason raises OSError.
    Removing a link never touches the file it names. A clone that holds anything else, such as an hf/
    export written into it, is refused before anything is removed, because that is output someone may
    want."""
    if clone.is_symlink() or not clone.is_dir():
        raise ExportError(f"{clone} is not an export clone directory")
    root = clone_root.resolve()
    resolved = clone.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ExportError(f"{clone} resolves to {resolved}, outside the export clone root {clone_root}")
    entries = list(resolved.iterdir())
    unexpected = sorted(
        entry.name
        for entry in entries
        if not entry.is_symlink() and not (entry.name == RUN_CONFIG and entry.is_file())
    )
    if unexpected:
        raise ExportError(f"{clone} holds {unexpected}, which an export clone does not; it is left in place")
    tracker = resolved.parent / LATEST_FILE
    if tracker.is_symlink():
        raise ExportError(f"{tracker} is a link; the tracker beside an export clone is a file of its own")
    for entry in entries:
        entry.unlink()
    resolved.rmdir()
    if not any(ITER_DIR.match(sibling.name) for sibling in resolved.parent.iterdir()):
        tracker.unlink(missing_ok=True)
    parent = resolved.parent
    while parent != root and parent.is_relative_to(root):
        try:
            parent.rmdir()
        except OSError as error:
            # Not empty (POSIX allows either code): another clone of this export still lives there. Any other
            # failure is raised, so a directory that cannot be removed is reported rather than left behind unseen.
            if error.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                raise
            break
        parent = parent.parent
    LOGGER.info("removed export clone %s", clone)


def default_clone_root(environ: Mapping[str, str], user: str) -> Path:
    """The root clones are built under: ``$GEODESIC_EXPORT_CLONE_ROOT`` when set and non-empty, else
    ``/projects/a5k/public/tmp/export_clones_<user>``. Reads only ``environ``."""
    return Path(environ.get(CLONE_ROOT_ENV) or PROJECT_CLONE_ROOT.format(user=user))


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """The command line: ``prepare`` or ``remove`` with their options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="decide what an export reads and build its clone if it needs one")
    prepare.add_argument("--megatron-path", required=True, help="top-level checkpoint directory (holds iter_*)")
    prepare.add_argument("--iteration", type=int, default=None, help="iteration to export (default: the latest)")
    prepare.add_argument(
        "--job",
        required=True,
        help="a name unique to this export, e.g. <SLURM job id>-<pid> (a directory under the root)",
    )
    remove = commands.add_parser("remove", help="delete an export clone after its export has finished")
    remove.add_argument("--clone", required=True, type=Path, help="the clone, as prepare printed it")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one command. Messages go to stderr; ``prepare``'s plan is the only thing on stdout."""
    logging.basicConfig(level=logging.INFO, format="[export-clone] %(message)s", stream=sys.stderr)
    args = parse_args(argv)
    clone_root = default_clone_root(os.environ, os.environ.get("USER") or getpass.getuser())
    try:
        if args.command == "prepare":
            source = prepare_export_source(args.megatron_path, args.iteration, clone_root, args.job)
            print(f"{PLAN_ITERATION}={source.iteration}")
            print(f"{PLAN_LOAD_PATH}={source.clone or ''}")
        else:
            remove_export_clone(args.clone, clone_root)
    except (ExportError, OSError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

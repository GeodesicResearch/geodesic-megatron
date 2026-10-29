# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Refuse to train a config at any width but the width it names.

A training config may fix the width it trains at in a top-level ``launch_width:`` block::

    launch_width:
      nodes: 128                  # the nodes the run trains on
      gpus_per_node: 4            # the GPUs each of them contributes
      data_parallel_size: 512     # nodes * gpus_per_node / (TP * PP * CP), so a parallelism change is refused too
      nvlink_links_per_gpu: 18    # optional: NVLink-sweep the allocation and train on its first healthy nodes

``pipeline_training_launch.sh`` reads the block on its own node before any rank starts (``plan``). With
``nvlink_links_per_gpu`` it records every allocated node's ``nvidia-smi nvlink --status`` into a directory of its own
(``scripts/training/nvlink_sweep.sh``) and trains on the first ``nodes`` nodes that
``scripts/training/nvlink_health.py`` judges healthy, so the allocation may hold spare nodes and an unhealthy node is
never a rank's host; without it the allocation must hold exactly ``nodes``. A ``--nodes`` or ``--nodelist`` given to
the launcher is refused, since the block decides both. The launcher then has the launch it is about to make checked
against the block (``record``) and hands the record to every rank in ``ISAMBARD_LAUNCH_WIDTH``.
``pipeline_training_run.py`` refuses a config carrying the block unless that record is for the same block and the
run's own world size and data-parallel size are the block's (``require_launched_width``), and logs both
(``[launch-width]``). What the block shares with ``code_identity:`` (reading it, its record, the command line, and
staying out of the run's settings) is ``scripts/training/launch_blocks.py``.

Usage (inside the container, from the repository root)::

    python -m scripts.training.launch_width plan --config <training yaml>
    python -m scripts.training.launch_width record --config <training yaml> --nodes N --nodelist LIST \\
        --gpus-per-node G

``plan`` prints ``<nodes> <gpus_per_node> <nvlink_links_per_gpu or ->`` for a config with a block and nothing for one
without. ``record`` prints the record as one JSON line, and exits 1 on a launch the block refuses, naming why.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass

from scripts.mapping_keys import require_keys
from scripts.training.launch_blocks import LAUNCH_WIDTH_KEY, config_launch_block, launcher_record, run_launch_check


REQUIRED_FIELDS = frozenset({"nodes", "gpus_per_node", "data_parallel_size"})
OPTIONAL_FIELDS = frozenset({"nvlink_links_per_gpu"})
RECORD_ENV = "ISAMBARD_LAUNCH_WIDTH"
NO_SWEEP = "-"


class LaunchWidthError(ValueError):
    """The run would train at another width than its config names, or that cannot be established."""


@dataclass(frozen=True)
class LaunchWidth:
    """A config's ``launch_width:`` block (see the module docstring)."""

    nodes: int
    gpus_per_node: int
    data_parallel_size: int
    nvlink_links_per_gpu: int | None

    @property
    def world_size(self) -> int:
        """The number of ranks the run trains with."""
        return self.nodes * self.gpus_per_node


def _positive_int(value: object, where: str) -> int:
    # bool is an int subclass, and YAML reads `yes` as True.
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LaunchWidthError(f"{where} must be a positive integer, got {value!r}")
    return value


def parse_launch_width(block: object, where: str) -> LaunchWidth:
    """The block as a ``LaunchWidth``; every count a positive integer, and the data-parallel size a divisor of the
    world size."""
    fields = require_keys(block, where, REQUIRED_FIELDS, OPTIONAL_FIELDS, error=LaunchWidthError)
    links = fields.get("nvlink_links_per_gpu")
    width = LaunchWidth(
        nodes=_positive_int(fields["nodes"], f"{where}.nodes"),
        gpus_per_node=_positive_int(fields["gpus_per_node"], f"{where}.gpus_per_node"),
        data_parallel_size=_positive_int(fields["data_parallel_size"], f"{where}.data_parallel_size"),
        nvlink_links_per_gpu=None if links is None else _positive_int(links, f"{where}.nvlink_links_per_gpu"),
    )
    if width.world_size % width.data_parallel_size:
        raise LaunchWidthError(
            f"{where}: data_parallel_size {width.data_parallel_size} does not divide the world size "
            f"{width.world_size} ({width.nodes} nodes x {width.gpus_per_node} GPUs)"
        )
    return width


def config_launch_width(config_file: str) -> LaunchWidth | None:
    """The ``launch_width:`` block of a training config (through its ``base_config:`` chain), or None."""
    return config_launch_block(config_file, LAUNCH_WIDTH_KEY, parse_launch_width)


def launch_record(width: LaunchWidth, config_file: str, nodes: int, nodelist: str, gpus_per_node: int) -> dict:
    """The record the launcher hands the ranks for a launch on ``nodes`` nodes (``nodelist``) of ``gpus_per_node``
    GPUs each; raises ``LaunchWidthError`` naming each way that launch differs from the block."""
    differences = []
    if nodes != width.nodes:
        differences.append(f"the launch has {nodes} nodes, the config trains on exactly {width.nodes}")
    if gpus_per_node != width.gpus_per_node:
        differences.append(
            f"the launch's nodes have {gpus_per_node} GPUs each, the config trains on {width.gpus_per_node} per node"
        )
    if differences:
        raise LaunchWidthError(f"{config_file}: " + "; ".join(differences))
    return {
        "config": config_file,
        "expected": asdict(width),
        "nodes": nodes,
        "nodelist": nodelist,
        "gpus_per_node": gpus_per_node,
    }


def require_launched_width(
    width: LaunchWidth, config_file: str, environ: dict[str, str], data_parallel_size: Callable[[int], int]
) -> dict:
    """The launcher's record for a config that fixes its width, once the run's own width is the block's.

    ``environ`` is the rank's environment (``WORLD_SIZE`` is torchrun's), and ``data_parallel_size`` the run's config's
    data-parallel size at a world size (``ConfigContainer.get_data_parallel_size``). Returns the record with the run's
    world and data-parallel sizes added.
    """
    record = launcher_record(width, LAUNCH_WIDTH_KEY, config_file, RECORD_ENV, environ, LaunchWidthError)
    world_size = int(environ["WORLD_SIZE"])
    if world_size != width.world_size:
        raise LaunchWidthError(
            f"{config_file}: the run has {world_size} ranks, its {LAUNCH_WIDTH_KEY} trains with exactly "
            f"{width.world_size} ({width.nodes} nodes x {width.gpus_per_node} GPUs)"
        )
    run_data_parallel_size = data_parallel_size(world_size)
    if run_data_parallel_size != width.data_parallel_size:
        raise LaunchWidthError(
            f"{config_file}: the run's data-parallel size at {world_size} ranks is {run_data_parallel_size}, its "
            f"{LAUNCH_WIDTH_KEY} states {width.data_parallel_size}"
        )
    return {**record, "world_size": world_size, "data_parallel_size": run_data_parallel_size}


def main(argv: list[str] | None = None) -> int:
    """``plan``: print the block's node count, GPUs per node and NVLink links per GPU. ``record``: check a launch
    against the block and print its record."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="print what the launch must be, or nothing for a config without a block")
    plan.add_argument("--config", required=True, help="the training config (its base_config chain is read)")
    record = commands.add_parser("record", help="check the launch about to be made and print its record")
    record.add_argument("--config", required=True, help="the training config (its base_config chain is read)")
    record.add_argument("--nodes", type=int, required=True, help="the number of nodes the launch trains on")
    record.add_argument("--nodelist", required=True, help="those nodes, as SLURM names them")
    record.add_argument("--gpus-per-node", type=int, required=True, help="the GPUs each of them contributes")
    args = parser.parse_args(argv)

    def check(width: LaunchWidth) -> tuple[str, int]:
        if args.command == "plan":
            links = NO_SWEEP if width.nvlink_links_per_gpu is None else width.nvlink_links_per_gpu
            return f"{width.nodes} {width.gpus_per_node} {links}", 0
        launch = launch_record(width, args.config, args.nodes, args.nodelist, args.gpus_per_node)
        return json.dumps(launch, sort_keys=True), 0

    return run_launch_check("launch-width", args.config, LAUNCH_WIDTH_KEY, parse_launch_width, LaunchWidthError, check)


if __name__ == "__main__":
    sys.exit(main())

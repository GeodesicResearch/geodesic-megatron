# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The top-level blocks of a training config that state how it must be launched rather than settings of the run.

``code_identity:`` names the code the config must train with (``scripts/training/code_identity.py``) and
``launch_width:`` the width it must train at (``scripts/training/launch_width.py``). ``pipeline_training_launch.sh``
checks each before any rank starts and hands its record of the check to every rank in an environment variable;
``pipeline_training_run.py`` refuses a config carrying the block without that record. Neither block is merged onto
the recipe (``pop_launch_blocks``), so no Hydra override can change it: the run refuses an override of a key its
settings do not hold.

This module holds what the blocks share: their keys, reading one from a config through its ``base_config:`` chain,
reading the launcher's record of it back on a rank, and the shape of the command line the launcher calls.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict
from typing import Any

from scripts.training.config_compose import load_composed_yaml


CODE_IDENTITY_KEY = "code_identity"
LAUNCH_WIDTH_KEY = "launch_width"
LAUNCH_BLOCK_KEYS = (CODE_IDENTITY_KEY, LAUNCH_WIDTH_KEY)


def pop_launch_blocks(overrides: dict) -> dict:
    """Remove the launch blocks from a composed config's overrides, in place, and return them by key."""
    return {key: overrides.pop(key) for key in LAUNCH_BLOCK_KEYS if key in overrides}


def config_launch_block(config_file: str, key: str, parse: Callable[[object, str], Any]) -> Any:
    """The launch block ``key`` of a training config (through its ``base_config:`` chain) as ``parse`` reads it,
    naming the config and the key in its errors; None for a config without the block."""
    block = load_composed_yaml(config_file).get(key)
    return None if block is None else parse(block, f"{config_file}: {key}")


def launcher_record(
    block: Any, key: str, config_file: str, record_env: str, environ: Mapping[str, str], error: type[Exception]
) -> dict:
    """The launcher's record of its check of ``block`` (a parsed launch block, a dataclass), read from
    ``environ[record_env]``.

    Raises ``error`` for a run with no record, which was not started through ``pipeline_training_launch.sh`` and so was
    never checked, and for a record of another block than ``config_file``'s, which does not vouch for this one.
    """
    raw = environ.get(record_env)
    if raw is None:
        raise error(
            f"{config_file} carries a {key}: block, and {record_env}, the record of pipeline_training_launch.sh's check "
            "of it, is not set: launch it through pipeline_training_launch.sh"
        )
    record = json.loads(raw)
    if record.get("expected") != asdict(block):
        raise error(f"{record_env} records the check of another {key} than {config_file}'s")
    return record


def run_launch_check(
    tag: str,
    config_file: str,
    key: str,
    parse: Callable[[object, str], Any],
    error: type[Exception],
    check: Callable[[Any], tuple[str, int]],
) -> int:
    """The command line the launcher calls for a launch block: ``check`` of ``config_file``'s parsed block, whose
    output goes to stdout and whose status is the exit status.

    A config without the block exits 0, printing nothing on stdout and a note on stderr; ``error``, from reading the
    block or from ``check``, exits 1 naming it on stderr as ``FATAL [<tag>]``.
    """
    try:
        block = config_launch_block(config_file, key, parse)
        if block is None:
            print(f"[{tag}] {config_file} carries no {key}: block", file=sys.stderr)
            return 0
        output, status = check(block)
    except error as failure:
        print(f"FATAL [{tag}]: {failure}", file=sys.stderr)
        return 1
    print(output)
    return status

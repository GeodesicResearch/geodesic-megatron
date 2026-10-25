# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Compare two Hugging Face safetensors checkpoints tensor by tensor, by name, byte for byte.

The two directories may shard their tensors differently, so files are not compared: the tensors are. Every tensor of
one directory must be in the other under the same name, with the same dtype, the same shape and the same bytes (NaN
payloads included, since the bytes are compared and not the values). A directory with a
``model.safetensors.index.json`` must hold exactly the tensors its index maps, each in the shard the index names, since
Hugging Face loads a sharded checkpoint through its index.

Usage::

    python scripts/checkpoint/compare_safetensors.py --config <compare.yaml>

The config names ``reference`` and ``candidate`` (directories holding ``*.safetensors`` files) and ``report`` (the JSON
file to write, which is refused if it exists). The report holds the config, both directories' shards and tensor counts,
the names found on one side only, and every tensor that differs with the reason.

Exit status, as the other gates': 0 every tensor is equal, 1 a difference, 2 the comparison could not be made (a
config, directory, index or shard that cannot be used, one directory named twice, or no tensor on either side). Reads
every tensor of both directories, so it belongs in a job of its own for a checkpoint of tens of gigabytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from safetensors import safe_open


# Run as a script, only scripts/checkpoint/ is on sys.path; the repo root makes the sibling packages importable.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from scripts.checkpoint.export_clone import ExportError  # noqa: E402
from scripts.checkpoint.safetensors_index import INDEX_FILE, checked_weight_map  # noqa: E402
from scripts.telemetry.gate_outcome import FAIL, NOT_EVALUATED, PASS, exit_status  # noqa: E402


REPORT_FORMAT = "safetensors-compare/1"
CONFIG_KEYS = frozenset({"reference", "candidate", "report"})
SHARD_GLOB = "*.safetensors"


class CompareError(Exception):
    """The comparison cannot be made: an unusable config, directory or shard."""


@dataclass(frozen=True)
class CompareConfig:
    """A compare config as ``load_config`` read it: its path and sha256, and the three paths it names."""

    path: Path
    sha256: str
    reference: Path
    candidate: Path
    report: Path


def load_config(path: Path) -> CompareConfig:
    """The config at ``path``, with exactly the keys ``CONFIG_KEYS``, an absolute path for each, and a reference and a
    candidate that exist and are not one directory, by any path (every tensor equals itself). Raises CompareError."""
    try:
        content = path.read_bytes()
        raw = yaml.safe_load(content)
    except (OSError, yaml.YAMLError) as error:
        raise CompareError(f"{path}: cannot read the config: {error}") from error
    if not isinstance(raw, dict) or set(raw) != CONFIG_KEYS:
        raise CompareError(f"{path}: the config must be a mapping with exactly the keys {sorted(CONFIG_KEYS)}")
    paths = {}
    for key in sorted(CONFIG_KEYS):
        if not isinstance(raw[key], str) or not Path(raw[key]).is_absolute():
            raise CompareError(f"{path}: {key} must be an absolute path, not {raw[key]!r}")
        paths[key] = Path(raw[key])
    try:
        # By device and inode, so a link, a bind mount or another spelling of the same directory is caught too.
        same = os.path.samefile(paths["reference"], paths["candidate"])
    except OSError as error:
        raise CompareError(f"{path}: cannot look up the reference or the candidate: {error}") from error
    if same:
        raise CompareError(f"{path}: reference and candidate are one directory, {paths['reference']}")
    return CompareConfig(path=path, sha256=hashlib.sha256(content).hexdigest(), **paths)


@dataclass(frozen=True)
class DirectoryTensors:
    """The safetensors files of a directory, and each tensor name with the file that holds it."""

    shards: list[Path]
    holders: dict[str, Path]


def tensor_files(directory: Path) -> DirectoryTensors:
    """The safetensors files in ``directory`` and the file holding each tensor name. Raises CompareError when the
    directory holds no such file, a file cannot be read, one name is held twice, or the directory's index
    (``INDEX_FILE``) does not map exactly its tensors to the files that hold them."""
    shards = sorted(directory.glob(SHARD_GLOB))
    if not shards:
        raise CompareError(f"{directory}: holds no {SHARD_GLOB} file")
    holders: dict[str, Path] = {}
    for shard in shards:
        try:
            with safe_open(shard, framework="pt", device="cpu") as handle:
                names = list(handle.keys())
        except Exception as error:  # safetensors raises its own types for a corrupt file; none is expected
            raise CompareError(f"{shard}: cannot read the header: {error}") from error
        for name in names:
            if name in holders:
                raise CompareError(f"{directory}: tensor {name} is in both {holders[name].name} and {shard.name}")
            holders[name] = shard
    # A single-file checkpoint has no index; a sharded one is loaded through it.
    if (directory / INDEX_FILE).is_file():
        try:
            weight_map = checked_weight_map(directory)
        except ExportError as error:
            raise CompareError(str(error)) from error
        if set(weight_map) != set(holders):
            raise CompareError(
                f"{directory}: {INDEX_FILE} maps {len(weight_map)} tensors, the shards hold {len(holders)}; "
                f"unmapped {sorted(set(holders) - set(weight_map))[:3]}"
            )
    return DirectoryTensors(shards=shards, holders=holders)


def difference(reference: torch.Tensor, candidate: torch.Tensor) -> str | None:
    """Why the two tensors are not the same bytes, or None when they are."""
    if reference.dtype != candidate.dtype:
        return f"dtype {reference.dtype} != {candidate.dtype}"
    if reference.shape != candidate.shape:
        return f"shape {tuple(reference.shape)} != {tuple(candidate.shape)}"
    reference_bytes = reference.contiguous().reshape(-1).view(torch.uint8)
    candidate_bytes = candidate.contiguous().reshape(-1).view(torch.uint8)
    if torch.equal(reference_bytes, candidate_bytes):
        return None
    differing = int((reference_bytes != candidate_bytes).sum())
    return f"{differing} of {reference_bytes.numel()} bytes differ"


def _directory_summary(tensors: DirectoryTensors) -> dict:
    return {
        "shards": {shard.name: shard.stat().st_size for shard in tensors.shards},
        "tensors": len(tensors.holders),
    }


def compare(config: CompareConfig) -> dict:
    """Compare the two directories of ``config`` and return the report. Raises CompareError, also when neither
    directory holds a tensor."""
    reference, candidate = tensor_files(config.reference), tensor_files(config.candidate)
    reference_holders, candidate_holders = reference.holders, candidate.holders
    if not reference_holders and not candidate_holders:
        raise CompareError(f"neither {config.reference} nor {config.candidate} holds a tensor")
    common = sorted(set(reference_holders) & set(candidate_holders))
    different = []
    try:
        with ExitStack() as stack:
            handles = {
                shard: stack.enter_context(safe_open(shard, framework="pt", device="cpu"))
                for shard in sorted(set(reference_holders.values()) | set(candidate_holders.values()))
            }
            for name in common:
                reason = difference(
                    handles[reference_holders[name]].get_tensor(name),
                    handles[candidate_holders[name]].get_tensor(name),
                )
                if reason is not None:
                    different.append({"name": name, "difference": reason})
    except Exception as error:  # a shard that opened at the header but cannot be read to the end
        raise CompareError(f"cannot read a tensor: {error}") from error
    only_reference = sorted(set(reference_holders) - set(candidate_holders))
    only_candidate = sorted(set(candidate_holders) - set(reference_holders))
    verdict = PASS if not (different or only_reference or only_candidate) else FAIL
    return {
        "format": REPORT_FORMAT,
        "verdict": verdict,
        "config": {
            "path": str(config.path),
            "sha256": config.sha256,
            "reference": str(config.reference),
            "candidate": str(config.candidate),
            "report": str(config.report),
        },
        "reference": _directory_summary(reference),
        "candidate": _directory_summary(candidate),
        "compared": len(common),
        "only_in_reference": only_reference,
        "only_in_candidate": only_candidate,
        "different": different,
    }


def run(config_path: Path) -> int:
    """Compare as the config says, write the report (creating its parent directories), and return the exit status.
    A CompareError, or an OSError while reading the checkpoints or writing the report, is printed to stderr and ends
    the run with status 2: the status 1 of a difference is never the status of a failure to compare."""
    try:
        config = load_config(config_path)
        if config.report.exists():
            raise CompareError(f"{config.report} exists; it is never overwritten")
        report = compare(config)
        config.report.parent.mkdir(parents=True, exist_ok=True)
        with config.report.open("x") as handle:  # exclusive: a report that appeared meanwhile is not overwritten
            handle.write(json.dumps(report, indent=2) + "\n")
    except (CompareError, OSError) as error:
        print(f"{NOT_EVALUATED}: {error}", file=sys.stderr)
        return exit_status([NOT_EVALUATED])
    print(
        f"{report['verdict']}: {report['compared']} tensors compared, {len(report['different'])} differ, "
        f"{len(report['only_in_reference'])} only in the reference, {len(report['only_in_candidate'])} only in the "
        f"candidate; report {config.report}"
    )
    return exit_status([report["verdict"]])


def main(argv: list[str] | None = None) -> int:
    """The command line: ``--config <compare.yaml>``; returns the exit status of ``run``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True, help="the compare config (YAML)")
    return run(parser.parse_args(argv).config)


if __name__ == "__main__":
    sys.exit(main())

# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Graft rows of a donor release's tensors into a recipient release and write the result as a checkpoint directory.

A release is a Hub model repository at one commit. A near-zero row is a tensor row whose L2 norm is below the
config's ``row_norm_threshold``. For each tensor that the config lists, the tool copies the donor's row over each
near-zero row of the recipient. The output directory is the recipient release with those rows replaced.

Run it inside the pipeline container::

    python scripts/checkpoint/graft_tensor_rows.py --config <graft.yaml>

The header of ``configs/checkpoint_grafts/nemotron_super_base_chat_init.yaml`` describes every config key. The tool
reads both releases from the Hub cache under ``HF_HOME``. It writes ``<HF_HOME>/GRAFTS_DIR/<output_name>``.

The tool writes these entries into the output directory:

- each shard with a grafted tensor, as a new file
- a symbolic link to the file of the recipient's Hub cache entry, for every other entry of the recipient
- ``PROVENANCE_FILE``, the record of the graft

A symbolic link works across accounts and filesystems. The release stays stored once. A load of the output
directory reads the linked shards from the Hub cache. ``cp -rL`` makes a copy with real files.

Catalogue:
    GRAFTS_DIR: the directory under ``HF_HOME`` that has one output directory for each graft.
    PROVENANCE_FILE: the file name of a graft's record in its output directory.
    CONFIG_KEYS: the keys of a graft config.
    FAIL: the ``donor_near_zero_rows`` value that stops a graft at a row near zero in both releases.
    LEAVE: the ``donor_near_zero_rows`` value that keeps the recipient's values in such a row.
    GraftError: a graft that the tool cannot make as its config states.
    Release: a Hub model repository at a full commit SHA.
    GraftConfig: a graft config as ``load_config`` read it.
    RowGraft: one tensor after a graft, with the row ids it changed and left.
    load_config: read and validate a graft config.
    cached_snapshot: the snapshot directory of a release in the Hub cache.
    graft_rows: replace the near-zero rows of one tensor with the donor's rows.
    graft: write the output directory of a graft and return its record.
    run: make the graft that a config states, under an output root.
    main: the command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load


# The repo root makes the sibling packages importable when this file runs as a script.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from scripts.checkpoint.safetensors_index import INDEX_FILE, checked_weight_map  # noqa: E402
from scripts.checkpoint.shard_rewrite import RewrittenShard, write_with_shards  # noqa: E402
from scripts.data.prepare_revisions import FULL_SHA  # noqa: E402
from scripts.mapping_keys import require_keys  # noqa: E402

from pipeline.export.export_clone import ExportError  # noqa: E402
from pipeline.hf_cache import cached_file, hub_cache_root  # noqa: E402


logger: logging.Logger = logging.getLogger(__name__)

GRAFTS_DIR = "grafts"
PROVENANCE_FILE = "graft_provenance.json"
CONFIG_KEYS = frozenset({"recipient", "donor", "tensors", "row_norm_threshold", "donor_near_zero_rows", "output_name"})
FAIL = "fail"
LEAVE = "leave"
_PROVENANCE_FORMAT = "graft-tensor-rows/1"
_RELEASE_KEYS = frozenset({"repo", "revision"})


class GraftError(Exception):
    """A graft that the tool cannot make as its config states."""


@dataclass(frozen=True)
class Release:
    """A Hub model repository at a full commit SHA."""

    repo: str
    revision: str


@dataclass(frozen=True)
class GraftConfig:
    """A graft config as ``load_config`` read it, with the path and the sha256 of its file."""

    path: Path
    sha256: str
    recipient: Release
    donor: Release
    tensors: tuple[str, ...]
    row_norm_threshold: float
    donor_near_zero_rows: str
    output_name: str


@dataclass(frozen=True)
class RowGraft:
    """One tensor after a graft.

    ``grafted`` lists the ids of the rows copied from the donor. ``left`` lists the ids of the rows that are near
    zero in both releases and keep the recipient's values.
    """

    tensor: torch.Tensor
    grafted: list[int]
    left: list[int]


def _release(value: Any, where: str) -> Release:
    require_keys(value, where, _RELEASE_KEYS, error=GraftError)
    repo, revision = value["repo"], value["revision"]
    if not isinstance(repo, str) or repo.count("/") != 1 or not all(repo.split("/")):
        raise GraftError(f"{where}.repo: expected a Hub repository as <namespace>/<name>, got {repo!r}")
    if not isinstance(revision, str) or FULL_SHA.fullmatch(revision) is None:
        raise GraftError(
            f"{where}.revision: expected a full 40-character lowercase hex commit SHA, got {revision!r}. A branch, "
            "a tag or a short SHA can come to mean another commit."
        )
    return Release(repo=repo, revision=revision)


def load_config(path: Path) -> GraftConfig:
    """Read and validate the graft config at ``path``.

    Raises:
        GraftError: YAML cannot parse the file, or its keys differ from ``CONFIG_KEYS``. A value breaks a rule that
            the config header gives.
    """
    try:
        content = path.read_bytes()
        raw = yaml.safe_load(content)
    except (OSError, yaml.YAMLError) as error:
        raise GraftError(f"{path}: cannot read the config: {error}") from error
    require_keys(raw, str(path), CONFIG_KEYS, error=GraftError)

    tensors = raw["tensors"]
    if (
        not isinstance(tensors, list)
        or not tensors
        or not all(isinstance(name, str) and name for name in tensors)
        or len(set(tensors)) != len(tensors)
    ):
        raise GraftError(f"{path}: tensors: expected a non-empty list of distinct tensor names, got {tensors!r}")
    threshold = raw["row_norm_threshold"]
    # type() and not isinstance: YAML reads true as a bool, and isinstance counts a bool as an int.
    if type(threshold) not in (int, float) or not threshold > 0:
        raise GraftError(
            f"{path}: row_norm_threshold: expected a positive number, got {threshold!r} (YAML reads 1e-3 as a string "
            "and 1.0e-3 as a number)"
        )
    policy = raw["donor_near_zero_rows"]
    if policy not in (FAIL, LEAVE):
        raise GraftError(f"{path}: donor_near_zero_rows: expected {FAIL!r} or {LEAVE!r}, got {policy!r}")
    output_name = raw["output_name"]
    if not isinstance(output_name, str) or not output_name or Path(output_name).name != output_name:
        raise GraftError(f"{path}: output_name: expected a directory name without a path, got {output_name!r}")
    return GraftConfig(
        path=path,
        sha256=hashlib.sha256(content).hexdigest(),
        recipient=_release(raw["recipient"], f"{path}: recipient"),
        donor=_release(raw["donor"], f"{path}: donor"),
        tensors=tuple(tensors),
        row_norm_threshold=float(threshold),
        donor_near_zero_rows=policy,
        output_name=output_name,
    )


def cached_snapshot(release: Release) -> Path:
    """The snapshot directory of ``release`` in the Hub cache under ``HF_HOME``.

    The function reads ``HF_HOME`` from the environment, and reads the cache only.

    Raises:
        GraftError: the cache has no ``INDEX_FILE`` of the release at its revision.
    """
    index = cached_file(release.repo, INDEX_FILE, release.revision)
    if index is None:
        raise GraftError(
            f"{hub_cache_root()} has no {INDEX_FILE} of {release.repo} at {release.revision}. Download the release "
            f"into the cache first: hf download {release.repo} --revision {release.revision}"
        )
    return index.parent


def graft_rows(
    name: str, recipient: torch.Tensor, donor: torch.Tensor, row_norm_threshold: float, donor_near_zero_rows: str
) -> RowGraft:
    """Replace the near-zero rows of ``recipient`` with the rows of ``donor`` at the same ids.

    A row is near zero when its L2 norm, computed in float32, is below ``row_norm_threshold``. A row that is near
    zero in both tensors keeps the recipient's values when ``donor_near_zero_rows`` is ``LEAVE``.

    Args:
        name: the tensor's name, for error messages.
        recipient: the 2-D tensor that receives rows.
        donor: the 2-D tensor that gives rows, with the shape and dtype of ``recipient``.
        row_norm_threshold: the norm below which a row is near zero.
        donor_near_zero_rows: ``FAIL`` or ``LEAVE``, the treatment of a row that is near zero in both tensors.

    Returns:
        A ``RowGraft`` with a new tensor.

    Raises:
        GraftError: a tensor is not 2-D, or the shapes or dtypes differ. A row is near zero in both tensors under
            ``FAIL``. The recipient has no row to graft.
    """
    if recipient.dim() != 2 or donor.dim() != 2:
        raise GraftError(
            f"{name}: expected 2-D tensors, got shapes {tuple(recipient.shape)} (recipient) and "
            f"{tuple(donor.shape)} (donor)"
        )
    if recipient.shape != donor.shape:
        raise GraftError(
            f"{name}: the recipient has shape {tuple(recipient.shape)} and the donor {tuple(donor.shape)}"
        )
    if recipient.dtype != donor.dtype:
        raise GraftError(f"{name}: the recipient has dtype {recipient.dtype} and the donor {donor.dtype}")
    recipient_near_zero = recipient.to(torch.float32).norm(dim=1) < row_norm_threshold
    donor_near_zero = donor.to(torch.float32).norm(dim=1) < row_norm_threshold
    left = torch.nonzero(recipient_near_zero & donor_near_zero).flatten().tolist()
    if left and donor_near_zero_rows == FAIL:
        raise GraftError(
            f"{name}: {len(left)} rows are near zero in the donor too (row norm below {row_norm_threshold}): {left}. "
            f"Set donor_near_zero_rows to {LEAVE!r} to keep them as they are."
        )
    rows = torch.nonzero(recipient_near_zero & ~donor_near_zero).flatten()
    if rows.numel() == 0:
        raise GraftError(
            f"{name}: no row to graft. {int(recipient_near_zero.sum())} rows of the recipient have a norm below "
            f"{row_norm_threshold}, and {len(left)} of them are near zero in the donor too."
        )
    tensor = recipient.clone()
    tensor[rows] = donor[rows]
    return RowGraft(tensor=tensor, grafted=rows.tolist(), left=left)


def _index_weight_map(directory: Path) -> dict[str, str]:
    try:
        return json.loads((directory / INDEX_FILE).read_text())["weight_map"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise GraftError(f"{directory}: cannot read the weight map of {INDEX_FILE}: {error!r}") from error


def _shard_tensors(path: Path) -> dict[str, torch.Tensor]:
    # One sequential read of the whole shard. A memory-mapped read from a network filesystem can be far slower.
    return load(path.read_bytes())


def _shard_metadata(path: Path) -> dict[str, str] | None:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return handle.metadata()


def graft(config: GraftConfig, recipient_dir: Path, donor_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Write the output directory of the graft that ``config`` states and return its record.

    The function reads every shard header of the recipient snapshot ``recipient_dir``. It reads in full each shard
    that has a tensor of ``config.tensors``, in ``recipient_dir`` and in the donor snapshot ``donor_dir``. It grafts
    every tensor in memory and then creates ``output_dir``. It writes there the grafted shards and the links, as
    ``write_with_shards`` does, and ``PROVENANCE_FILE``. ``PROVENANCE_FILE`` holds the returned record.

    Raises:
        GraftError: ``output_dir`` exists, or ``recipient_dir`` has an entry ``PROVENANCE_FILE``. A snapshot lacks a
            file or a tensor that the graft reads. The two indexes put a tensor in different shards. ``graft_rows``
            refuses a tensor. The written directory fails ``checked_weight_map``.
    """
    if output_dir.exists():
        raise GraftError(f"{output_dir} exists. A graft writes a new directory.")
    if (recipient_dir / PROVENANCE_FILE).exists():
        raise GraftError(f"{recipient_dir} has an entry {PROVENANCE_FILE}, the name of the graft's own record")
    try:
        recipient_map = checked_weight_map(recipient_dir)
    except ExportError as error:
        raise GraftError(f"the recipient snapshot is not complete: {error}") from error
    donor_map = _index_weight_map(donor_dir)

    names_by_shard: dict[str, list[str]] = {}
    for name in config.tensors:
        for label, weight_map in (("recipient", recipient_map), ("donor", donor_map)):
            if name not in weight_map:
                raise GraftError(f"{name}: the {label} index maps no tensor of that name")
        if recipient_map[name] != donor_map[name]:
            raise GraftError(
                f"{name}: the recipient index puts it in {recipient_map[name]} and the donor index in "
                f"{donor_map[name]}"
            )
        names_by_shard.setdefault(recipient_map[name], []).append(name)

    grafted_shards: dict[str, RewrittenShard] = {}
    tensor_records: dict[str, dict[str, Any]] = {}
    for shard, names in names_by_shard.items():
        if not (donor_dir / shard).is_file():
            raise GraftError(
                f"{donor_dir} has no {shard}. Download it into the Hub cache first: hf download "
                f"{config.donor.repo} --revision {config.donor.revision} --include {shard}"
            )
        tensors = _shard_tensors(recipient_dir / shard)
        donor_tensors = _shard_tensors(donor_dir / shard)
        for name in names:
            if name not in donor_tensors:
                raise GraftError(f"{name}: the donor's {shard} has no tensor of that name")
            result = graft_rows(
                name, tensors[name], donor_tensors[name], config.row_norm_threshold, config.donor_near_zero_rows
            )
            tensors[name] = result.tensor
            tensor_records[name] = {
                "shard": shard,
                "shape": list(result.tensor.shape),
                "dtype": str(result.tensor.dtype),
                "grafted_row_count": len(result.grafted),
                "left_near_zero_row_count": len(result.left),
                "grafted_rows": result.grafted,
                "left_near_zero_rows": result.left,
            }
        grafted_shards[shard] = RewrittenShard(tensors=tensors, metadata=_shard_metadata(recipient_dir / shard))

    linked = write_with_shards(recipient_dir, output_dir, grafted_shards, unlinked=())
    try:
        checked_weight_map(output_dir)
    except ExportError as error:
        raise GraftError(f"the written directory is not a complete checkpoint: {error}") from error

    record = {
        "format": _PROVENANCE_FORMAT,
        "config": {
            "path": str(config.path),
            "sha256": config.sha256,
            "recipient": {"repo": config.recipient.repo, "revision": config.recipient.revision},
            "donor": {"repo": config.donor.repo, "revision": config.donor.revision},
            "tensors": list(config.tensors),
            "row_norm_threshold": config.row_norm_threshold,
            "donor_near_zero_rows": config.donor_near_zero_rows,
            "output_name": config.output_name,
        },
        "recipient_snapshot": str(recipient_dir),
        "donor_snapshot": str(donor_dir),
        "output_dir": str(output_dir),
        "tensors": tensor_records,
        "written_shards": sorted(grafted_shards),
        "linked_entries": linked,
    }
    (output_dir / PROVENANCE_FILE).write_text(json.dumps(record, indent=2) + "\n")
    return record


def run(config_path: Path, output_root: Path) -> dict[str, Any]:
    """Make the graft that the config at ``config_path`` states and return its record.

    The function reads both releases from the Hub cache under ``HF_HOME``. It writes ``output_root/<output_name>`` as
    ``graft`` does. It logs one line for each tensor and one line for the output directory.

    Raises:
        GraftError: as ``load_config``, ``cached_snapshot`` and ``graft`` raise it.
    """
    config = load_config(config_path)
    record = graft(
        config, cached_snapshot(config.recipient), cached_snapshot(config.donor), output_root / config.output_name
    )
    for name, tensor in record["tensors"].items():
        logger.info(
            f"{name} ({tensor['shard']}): grafted {tensor['grafted_row_count']} rows, left "
            f"{tensor['left_near_zero_row_count']} near-zero rows"
        )
    logger.info(f"wrote {record['output_dir']} (record: {PROVENANCE_FILE})")
    return record


def main(argv: list[str] | None = None) -> int:
    """The command line: ``--config <graft.yaml>``.

    The function reads ``HF_HOME`` from the environment. It calls ``run`` with the output root ``GRAFTS_DIR`` under
    ``HF_HOME``. It returns 0, or 1 after it logs a GraftError.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True, help="the graft config (YAML)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        run(args.config, hub_cache_root().parent / GRAFTS_DIR)
    except GraftError as error:
        logger.error(f"FATAL: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

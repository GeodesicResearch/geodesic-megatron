# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Write a checkpoint directory that differs from a source directory in given safetensors shards.

A checkpoint directory is a Hugging Face model directory with safetensors shards. ``write_with_shards`` saves the
given shards as new files and links every other entry of the source. ``scripts/checkpoint/graft_tensor_rows.py`` and
``scripts/data/extend_vocab_for_mq.py`` write their output directories with it.

Catalogue:
    RewrittenShard: the tensors and the metadata of one shard to write.
    save_shard: write one safetensors shard so that its path has a complete file or no file.
    write_with_shards: write a directory with given shards and links to the other entries of a source.
"""

from __future__ import annotations

import os
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save_file
from scripts.checkpoint.safetensors_index import safetensors_expected_size


@dataclass(frozen=True)
class RewrittenShard:
    """The tensors of one shard to write, and the safetensors metadata to write with them (None for no metadata)."""

    tensors: Mapping[str, torch.Tensor]
    metadata: Mapping[str, str] | None


def save_shard(shard: RewrittenShard, path: Path) -> None:
    """Write ``shard`` as a safetensors file so that ``path`` has a complete file or no file.

    The function writes a sibling file ``<name>.partial`` and flushes it to disk. It compares the file's length with
    the length that the file's header declares, and then renames the file to ``path``. When a step fails, the
    function removes the partial file. An earlier file at ``path`` then stays as it is.

    Raises:
        OSError: the written length differs from the declared length.
    """
    partial = path.with_name(path.name + ".partial")
    try:
        save_file(dict(shard.tensors), partial, metadata=None if shard.metadata is None else dict(shard.metadata))
        with open(partial, "rb") as handle:
            os.fsync(handle.fileno())
        expected = safetensors_expected_size(partial)
        actual = partial.stat().st_size
        if expected != actual:
            raise OSError(
                f"{path.name}: wrote {actual:,} bytes but its header declares {expected:,} "
                f"(short by {expected - actual:,}). Refusing to publish a truncated shard."
            )
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def write_with_shards(
    source_dir: Path, output_dir: Path, shards: Mapping[str, RewrittenShard], unlinked: Collection[str]
) -> list[str]:
    """Write ``output_dir`` as ``source_dir`` with the given shards rewritten, and return the linked entry names.

    ``shards`` maps a shard's file name to its content. The function creates ``output_dir`` when it is absent and
    saves each shard there with ``save_shard``. It then makes a symbolic link to the resolved path of every other
    entry of ``source_dir``. It skips the entries in ``unlinked``, which the caller writes itself. It leaves an entry
    that ``output_dir`` already has, so a second run over a partial output completes it.

    Raises:
        OSError: as ``save_shard`` raises it, or a link or the directory cannot be made.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, shard in shards.items():
        save_shard(shard, output_dir / name)
    linked = []
    for entry in sorted(source_dir.iterdir()):
        destination = output_dir / entry.name
        if entry.name in shards or entry.name in unlinked or destination.exists() or destination.is_symlink():
            continue
        destination.symlink_to(entry.resolve())
        linked.append(entry.name)
    return linked

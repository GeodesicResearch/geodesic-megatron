# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Unit tests for scripts/checkpoint/safetensors_index.py.

The shards are real safetensors files written with the library's own ``save_file``, and the index is the JSON file
Hugging Face writes beside them.
"""

import json
import struct
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from scripts.checkpoint.export_clone import ExportError
from scripts.checkpoint.safetensors_index import INDEX_FILE, checked_weight_map, safetensors_tensor_names


def write_sharded(directory: Path, shards: dict[str, list[str]]) -> dict[str, str]:
    """Write each shard with its tensors (one small tensor per name) and return the agreeing weight map."""
    directory.mkdir(parents=True, exist_ok=True)
    for shard, names in shards.items():
        save_file({name: torch.zeros(2) for name in names}, str(directory / shard))
    return {name: shard for shard, names in shards.items() for name in names}


def write_index(directory: Path, weight_map: dict[str, str]) -> None:
    (directory / INDEX_FILE).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))


SHARDS = {"model-00001.safetensors": ["a", "b"], "model-00002.safetensors": ["c"]}


def test_an_index_that_agrees_with_its_shards_is_returned(tmp_path):
    agreeing = write_sharded(tmp_path, SHARDS)
    write_index(tmp_path, agreeing)
    assert checked_weight_map(tmp_path) == agreeing


@pytest.mark.parametrize(
    "change",
    [
        {"a": "model-00002.safetensors"},  # a tensor put in a shard that does not hold it
        {"a": "model-00009.safetensors"},  # a shard that does not exist
    ],
    ids=["wrong shard", "missing shard"],
)
def test_an_index_that_disagrees_with_its_shards_is_refused(tmp_path, change):
    agreeing = write_sharded(tmp_path, SHARDS)
    write_index(tmp_path, agreeing | change)
    with pytest.raises(ExportError):
        checked_weight_map(tmp_path)


@pytest.mark.parametrize("outside", ["relative", "absolute"])
def test_an_index_naming_a_file_outside_its_directory_is_refused(tmp_path, outside):
    """Hugging Face loads the file the index names, so a file outside the checkpoint's directory is not one of its
    shards, even when it holds exactly the tensors the index puts there."""
    checkpoint = tmp_path / "checkpoint"
    agreeing = write_sharded(checkpoint, SHARDS)
    moved = SHARDS["model-00001.safetensors"]
    write_sharded(tmp_path / "elsewhere", {"model-00001.safetensors": moved})
    target = (
        "../elsewhere/model-00001.safetensors"
        if outside == "relative"
        else str(tmp_path / "elsewhere" / "model-00001.safetensors")
    )
    write_index(checkpoint, agreeing | {name: target for name in moved})
    with pytest.raises(ExportError, match="not a file of"):
        checked_weight_map(checkpoint)


def test_an_unreadable_index_is_refused(tmp_path):
    write_sharded(tmp_path, SHARDS)
    (tmp_path / INDEX_FILE).write_text("not json")
    with pytest.raises(ExportError, match="unreadable"):
        checked_weight_map(tmp_path)


@pytest.mark.parametrize(
    "declared, file_size",
    [(150_000_000, 160_000_000), (1_000, 64)],
    ids=["beyond the format's cap", "beyond the file"],
)
def test_a_header_length_the_shard_cannot_hold_is_refused_unread(tmp_path, declared, file_size):
    """safetensors caps a header at 100 MB, and no header is longer than its file, so a declared
    length past either is a corrupt shard; it is refused before the read rather than read into
    memory."""
    shard = tmp_path / "model-00001-of-00001.safetensors"
    with shard.open("wb") as handle:
        handle.write(struct.pack("<Q", declared))
        handle.truncate(file_size)
    with pytest.raises(ValueError, match="header length"):
        safetensors_tensor_names(shard)

# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Unit tests for scripts/checkpoint/shard_rewrite.py.

The shards are real safetensors files written with the library's own ``save_file`` under a temporary directory. The
tests run on a CPU.
"""

from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from scripts.checkpoint import shard_rewrite
from scripts.checkpoint.safetensors_index import safetensors_expected_size
from scripts.checkpoint.shard_rewrite import RewrittenShard, save_shard, write_with_shards


def tensors() -> dict[str, torch.Tensor]:
    return {"a": torch.zeros(4, 8, dtype=torch.bfloat16), "b": torch.ones(2, 3, dtype=torch.float32)}


def write_source(directory: Path) -> Path:
    """A source checkpoint directory with two shards, a config and a subdirectory."""
    directory.mkdir(parents=True)
    save_file({"a": torch.zeros(4, 8, dtype=torch.bfloat16)}, str(directory / "model-00001.safetensors"))
    save_file({"b": torch.ones(2, 3)}, str(directory / "model-00002.safetensors"))
    (directory / "config.json").write_text("{}")
    (directory / "notes").mkdir()
    (directory / "notes" / "card.md").write_text("card")
    return directory


def test_save_shard_writes_readable_tensors_with_their_metadata_and_removes_the_partial_file(tmp_path):
    path = tmp_path / "shard.safetensors"
    save_shard(RewrittenShard(tensors=tensors(), metadata={"format": "pt"}), path)
    assert not path.with_name(path.name + ".partial").exists()
    loaded = load_file(str(path))
    assert sorted(loaded) == ["a", "b"]
    assert loaded["a"].shape == (4, 8) and loaded["b"].shape == (2, 3)
    assert safetensors_expected_size(path) == path.stat().st_size
    with safe_open(str(path), framework="pt") as handle:
        assert handle.metadata() == {"format": "pt"}


def test_save_shard_writes_no_metadata_when_given_none(tmp_path):
    path = tmp_path / "shard.safetensors"
    save_shard(RewrittenShard(tensors=tensors(), metadata=None), path)
    with safe_open(str(path), framework="pt") as handle:
        assert handle.metadata() is None


def test_save_shard_leaves_no_file_when_the_write_fails(tmp_path, monkeypatch):
    def out_of_quota(*_args, **_kwargs):
        raise OSError("simulated out-of-quota")

    # The untestable boundary is a filesystem that fails in the middle of a write, so the write itself is replaced.
    monkeypatch.setattr(shard_rewrite, "save_file", out_of_quota)
    path = tmp_path / "shard.safetensors"
    with pytest.raises(OSError, match="simulated out-of-quota"):
        save_shard(RewrittenShard(tensors=tensors(), metadata=None), path)
    assert not path.exists()
    assert not path.with_name(path.name + ".partial").exists()


def test_save_shard_refuses_a_file_shorter_than_its_header_declares_and_keeps_the_earlier_file(tmp_path, monkeypatch):
    def truncated_save(content, path, metadata=None):
        save_file(content, path, metadata=metadata)
        with open(path, "r+b") as handle:
            handle.truncate(Path(path).stat().st_size - 16)

    path = tmp_path / "shard.safetensors"
    save_shard(RewrittenShard(tensors={"a": torch.ones(2)}, metadata=None), path)
    earlier = path.read_bytes()
    # The untestable boundary is a write that the filesystem cuts short without an error.
    monkeypatch.setattr(shard_rewrite, "save_file", truncated_save)
    with pytest.raises(OSError, match="Refusing to publish a truncated shard"):
        save_shard(RewrittenShard(tensors=tensors(), metadata=None), path)
    assert path.read_bytes() == earlier
    assert not path.with_name(path.name + ".partial").exists()


def test_write_with_shards_saves_the_shards_and_links_every_other_entry_to_its_resolved_path(tmp_path):
    real = write_source(tmp_path / "real")
    source = tmp_path / "source"
    source.mkdir()
    for entry in real.iterdir():
        (source / entry.name).symlink_to(entry)
    output = tmp_path / "out" / "checkpoint"
    rewritten = RewrittenShard(tensors={"a": torch.full((4, 8), 2.0, dtype=torch.bfloat16)}, metadata={"format": "pt"})
    linked = write_with_shards(source, output, {"model-00001.safetensors": rewritten}, unlinked=())
    assert linked == ["config.json", "model-00002.safetensors", "notes"]
    assert sorted(entry.name for entry in output.iterdir()) == sorted(entry.name for entry in source.iterdir())
    assert not (output / "model-00001.safetensors").is_symlink()
    assert torch.equal(load_file(str(output / "model-00001.safetensors"))["a"], rewritten.tensors["a"])
    for name in linked:
        assert (output / name).is_symlink()
        assert Path((output / name).readlink()) == (real / name).resolve()
    assert (output / "notes" / "card.md").read_text() == "card"
    assert torch.equal(load_file(str(output / "model-00002.safetensors"))["b"], torch.ones(2, 3))


def test_write_with_shards_leaves_the_unlinked_entries_to_the_caller(tmp_path):
    source = write_source(tmp_path / "source")
    output = tmp_path / "out"
    rewritten = RewrittenShard(tensors={"b": torch.zeros(2, 3)}, metadata=None)
    linked = write_with_shards(source, output, {"model-00002.safetensors": rewritten}, unlinked={"config.json"})
    assert linked == ["model-00001.safetensors", "notes"]
    assert not (output / "config.json").exists()


def test_write_with_shards_leaves_an_entry_that_the_output_already_has(tmp_path):
    source = write_source(tmp_path / "source")
    output = tmp_path / "out"
    output.mkdir()
    (output / "config.json").write_text('{"kept": true}')
    rewritten = RewrittenShard(tensors={"a": torch.ones(4, 8, dtype=torch.bfloat16)}, metadata=None)
    first = write_with_shards(source, output, {"model-00001.safetensors": rewritten}, unlinked=())
    assert first == ["model-00002.safetensors", "notes"]
    assert (output / "config.json").read_text() == '{"kept": true}'
    second = write_with_shards(source, output, {"model-00001.safetensors": rewritten}, unlinked=())
    assert second == []
    assert torch.equal(load_file(str(output / "model-00001.safetensors"))["a"], rewritten.tensors["a"])

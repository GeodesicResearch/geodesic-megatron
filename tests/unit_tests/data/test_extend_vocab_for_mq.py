# Copyright (c) 2026, Geodesic Research.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for the shard-writing guarantees of scripts/data/extend_vocab_for_mq.py.

The script rewrites multi-GB embedding shards, so a partial write must never be
published: these tests pin the size-vs-header check and the atomic replace.

Run:
    uv run pytest tests/unit_tests/data/test_extend_vocab_for_mq.py -v
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "data" / "extend_vocab_for_mq.py"


@pytest.fixture(scope="module")
def ev_module():
    """Import scripts/data/extend_vocab_for_mq.py as a module."""
    spec = importlib.util.spec_from_file_location("extend_vocab_for_mq", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tensors():
    return {"a": torch.zeros(4, 8, dtype=torch.bfloat16), "b": torch.ones(2, 3, dtype=torch.float32)}


class TestTruncatedShardIsRefused:
    def test_a_short_write_is_not_published(self, ev_module, tmp_path, monkeypatch, tensors):
        # The production failure this guard exists for: a write that dies partway
        # leaves a header still advertising the full payload, so the file reads as
        # valid metadata over missing bytes. Simulated by truncating after a real
        # save, because genuinely running a node out of quota is not available here.
        real_save = ev_module.safetensors.torch.save_file

        def save_then_truncate(tensor_dict, path, *args, **kwargs):
            real_save(tensor_dict, path, *args, **kwargs)
            with open(path, "r+b") as f:
                f.truncate(Path(path).stat().st_size - 16)

        monkeypatch.setattr(ev_module.safetensors.torch, "save_file", save_then_truncate)
        dest = tmp_path / "shard.safetensors"
        with pytest.raises(OSError, match="Refusing to publish a truncated shard"):
            ev_module._save_shard_atomically(tensors, dest)
        assert not dest.exists()
        assert not dest.with_name(dest.name + ".partial").exists()


class TestSaveShardAtomically:
    def test_writes_readable_tensors_and_removes_the_temp_file(self, ev_module, tmp_path, tensors):
        import safetensors.torch

        dest = tmp_path / "shard.safetensors"
        ev_module._save_shard_atomically(tensors, dest)

        assert dest.exists()
        assert not dest.with_name(dest.name + ".partial").exists()
        loaded = safetensors.torch.load_file(dest)
        assert sorted(loaded) == ["a", "b"]
        assert loaded["a"].shape == (4, 8)
        assert loaded["b"].shape == (2, 3)

    def test_leaves_no_temp_file_when_the_write_fails(self, ev_module, tmp_path, monkeypatch, tensors):
        def boom(*_args, **_kwargs):
            raise OSError("simulated out-of-quota")

        monkeypatch.setattr(ev_module.safetensors.torch, "save_file", boom)
        dest = tmp_path / "shard.safetensors"
        with pytest.raises(OSError, match="simulated out-of-quota"):
            ev_module._save_shard_atomically(tensors, dest)
        assert not dest.exists()
        assert not dest.with_name(dest.name + ".partial").exists()


class TestRequiredTokenizerFiles:
    def test_tokenizer_config_is_required(self, ev_module):
        # tokenizer_config.json carries loss_mask_token_ids; shipping a checkpoint
        # without it would silently disable MQ masking.
        assert "tokenizer_config.json" in ev_module.REQUIRED_TOKENIZER_FILES
        assert "tokenizer.json" in ev_module.REQUIRED_TOKENIZER_FILES

    def test_every_known_tokenizer_file_is_classified(self, ev_module):
        assert set(ev_module.TOKENIZER_FILE_NAMES) == set(ev_module.REQUIRED_TOKENIZER_FILES) | set(
            ev_module.OPTIONAL_TOKENIZER_FILES
        )

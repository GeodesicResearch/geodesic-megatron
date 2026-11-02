# Copyright (c) 2026, Geodesic Research.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for scripts/data/extend_vocab_for_mq.py.

The script rewrites multi-GB embedding shards, so a partial write must never be
published: these tests pin the size-vs-header check and the atomic replace. They
also pin which tokenizer directories the script accepts: it refuses one whose
tokenizer_config.json carries loss_mask_token_ids before writing anything, and
otherwise ships its files with the extended checkpoint.

Run:
    uv run pytest tests/unit_tests/data/test_extend_vocab_for_mq.py -v
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import safetensors.torch
import torch

from megatron.bridge.training.token_masking.resolution import DECLARATION_FIELD
from tests.unit_tests.token_masking_fixtures import MARKER_ID, build_tiny_hf_tokenizer, write_declaration


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


class TestSafetensorsExpectedSize:
    def test_matches_a_complete_file(self, ev_module, tmp_path, tensors):
        dest = tmp_path / "shard.safetensors"
        ev_module._save_shard_atomically(tensors, dest)
        assert ev_module._safetensors_expected_size(dest) == dest.stat().st_size

    def test_exceeds_actual_size_when_payload_is_truncated(self, ev_module, tmp_path, tensors):
        # The signature of the production failure: the header still advertises the
        # full payload, so only a size comparison reveals the missing bytes.
        dest = tmp_path / "shard.safetensors"
        ev_module._save_shard_atomically(tensors, dest)
        full = dest.stat().st_size
        with open(dest, "r+b") as f:
            f.truncate(full - 16)
        assert ev_module._safetensors_expected_size(dest) == full
        assert dest.stat().st_size == full - 16


class TestSaveShardAtomically:
    def test_writes_readable_tensors_and_removes_the_temp_file(self, ev_module, tmp_path, tensors):
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
    def test_tokenizer_json_and_config_are_required(self, ev_module):
        # tokenizer.json registers the marker and tokenizer_config.json must come from the same tokenizer; a
        # checkpoint missing either would ship its parent's tokenizer, which BPE-splits the marker.
        assert "tokenizer_config.json" in ev_module.REQUIRED_TOKENIZER_FILES
        assert "tokenizer.json" in ev_module.REQUIRED_TOKENIZER_FILES

    def test_every_known_tokenizer_file_is_classified(self, ev_module):
        assert set(ev_module.TOKENIZER_FILE_NAMES) == set(ev_module.REQUIRED_TOKENIZER_FILES) | set(
            ev_module.OPTIONAL_TOKENIZER_FILES
        )


class TestCheckMqTokenizerDir:
    def test_accepts_a_tokenizer_without_the_declaration(self, ev_module, tmp_path):
        ev_module.check_mq_tokenizer_dir(build_tiny_hf_tokenizer(tmp_path / "tokenizer"))

    @pytest.mark.parametrize("declared", [[MARKER_ID], [], None], ids=["ids", "empty", "null"])
    def test_refuses_a_tokenizer_carrying_the_declaration(self, ev_module, tmp_path, declared):
        directory = build_tiny_hf_tokenizer(tmp_path / "tokenizer")
        write_declaration(directory, declared)
        with pytest.raises(ValueError, match=f"carries {DECLARATION_FIELD}"):
            ev_module.check_mq_tokenizer_dir(directory)

    def test_refuses_a_missing_directory(self, ev_module, tmp_path):
        with pytest.raises(FileNotFoundError, match="MQ tokenizer dir not found"):
            ev_module.check_mq_tokenizer_dir(tmp_path / "absent")

    def test_refuses_a_directory_missing_a_required_file(self, ev_module, tmp_path):
        directory = build_tiny_hf_tokenizer(tmp_path / "tokenizer")
        (directory / "tokenizer.json").unlink()
        with pytest.raises(FileNotFoundError, match=r"missing required file\(s\): \['tokenizer.json'\]"):
            ev_module.check_mq_tokenizer_dir(directory)


HIDDEN = 4


def _write_tiny_checkpoint(directory: Path, vocab: int) -> None:
    """An HF checkpoint dir with untied embedding and head of ``vocab`` rows, as the script reads one."""
    directory.mkdir(parents=True)
    shard = "model-00001-of-00001.safetensors"
    tensors = {
        "backbone.embeddings.weight": torch.randn(vocab, HIDDEN).to(torch.bfloat16),
        "lm_head.weight": torch.randn(vocab, HIDDEN).to(torch.bfloat16),
    }
    safetensors.torch.save_file(tensors, directory / shard)
    index = {"metadata": {}, "weight_map": {name: shard for name in tensors}}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index))
    (directory / "config.json").write_text(json.dumps({"vocab_size": vocab}))


class TestMain:
    def test_refuses_a_declaring_tokenizer_before_writing_anything(self, ev_module, tmp_path):
        tokenizer = build_tiny_hf_tokenizer(tmp_path / "tokenizer", declared_token_ids=[MARKER_ID])
        output = tmp_path / "out"
        with pytest.raises(ValueError, match=f"carries {DECLARATION_FIELD}"):
            ev_module.main(
                [
                    "--input-dir",
                    str(tmp_path / "in"),
                    "--output-dir",
                    str(output),
                    "--mq-tokenizer-dir",
                    str(tokenizer),
                ]
            )
        assert not output.exists()

    def test_extends_the_vocab_and_ships_the_tokenizer(self, ev_module, tmp_path):
        source = tmp_path / "in"
        _write_tiny_checkpoint(source, ev_module.ORIG_VOCAB)
        tokenizer = build_tiny_hf_tokenizer(tmp_path / "tokenizer")
        output = tmp_path / "out"

        assert (
            ev_module.main(
                ["--input-dir", str(source), "--output-dir", str(output), "--mq-tokenizer-dir", str(tokenizer)]
            )
            == 0
        )

        extended = safetensors.torch.load_file(output / "model-00001-of-00001.safetensors")
        assert extended["backbone.embeddings.weight"].shape == (ev_module.TARGET_VOCAB, HIDDEN)
        assert extended["lm_head.weight"].shape == (ev_module.TARGET_VOCAB, HIDDEN)
        assert json.loads((output / "config.json").read_text())["vocab_size"] == ev_module.TARGET_VOCAB
        for name in ev_module.REQUIRED_TOKENIZER_FILES:
            assert (output / name).read_bytes() == (tokenizer / name).read_bytes()
        assert DECLARATION_FIELD not in json.loads((output / "tokenizer_config.json").read_text())

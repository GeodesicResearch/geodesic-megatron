# Copyright (c) 2026, Geodesic Research.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for scripts/data/extend_vocab_for_mq.py.

These tests pin what ``main`` writes for each shard layout: the entries, the tensors,
the index and the config. They also pin which tokenizer directories the script
accepts: it refuses one whose tokenizer_config.json carries loss_mask_token_ids
before writing anything, and otherwise ships its files with the extended checkpoint.
tests/unit_tests/test_shard_rewrite.py tests the shard writes themselves.

Run:
    uv run pytest tests/unit_tests/data/test_extend_vocab_for_mq.py -v
"""

from __future__ import annotations

import importlib.util
import json
import os
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


WIDE_HIDDEN = 16
EMBED_SHARD = "model-00001-of-00002.safetensors"
SECOND_SHARD = "model-00002-of-00002.safetensors"
OTHER_KEY = "backbone.layers.0.mixer.weight"
EMBED_KEY = "backbone.embeddings.weight"
HEAD_KEY = "lm_head.weight"
INDEX = "model.safetensors.index.json"
# Where each layout puts the embedding, the head and one other tensor. The tied layout has no head tensor.
LAYOUTS = {
    "head_in_its_own_shard": {EMBED_SHARD: [EMBED_KEY, OTHER_KEY], SECOND_SHARD: [HEAD_KEY]},
    "head_beside_the_embedding": {EMBED_SHARD: [EMBED_KEY, HEAD_KEY], SECOND_SHARD: [OTHER_KEY]},
    "tied_embeddings": {EMBED_SHARD: [EMBED_KEY], SECOND_SHARD: [OTHER_KEY]},
}


def _write_parent_checkpoint(directory: Path, layout: str, vocab: int) -> dict[str, torch.Tensor]:
    """An HF checkpoint dir in one of ``LAYOUTS``, with the files a release carries beside its weights."""
    directory.mkdir(parents=True)
    generator = torch.Generator().manual_seed(3)
    tensors = {
        EMBED_KEY: torch.randn(vocab, WIDE_HIDDEN, generator=generator).to(torch.bfloat16),
        HEAD_KEY: torch.randn(vocab, WIDE_HIDDEN, generator=generator).to(torch.bfloat16),
        OTHER_KEY: torch.randn(3, 5, generator=generator),
    }
    weight_map = {}
    for shard, names in LAYOUTS[layout].items():
        safetensors.torch.save_file(
            {name: tensors[name] for name in names}, directory / shard, metadata={"format": "pt"}
        )
        weight_map.update({name: shard for name in names})
    index = {"metadata": {"total_parameters": 7, "total_size": 1}, "weight_map": weight_map}
    (directory / INDEX).write_text(json.dumps(index))
    (directory / "config.json").write_text(json.dumps({"vocab_size": vocab, "hidden_size": WIDE_HIDDEN}))
    (directory / "generation_config.json").write_text("{}")
    (directory / "tokenizer.json").write_text('{"parent": true}')
    (directory / "tokenizer_config.json").write_text('{"parent": true}')
    (directory / "chat_template.jinja").write_text("the parent's template")
    return tensors


def _new_row(seed: int, init_std: float, dtype: torch.dtype) -> torch.Tensor:
    """The row the script documents for a marker: N(0, init_std) from a CPU generator with that seed."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(1, WIDE_HIDDEN, generator=generator, dtype=torch.float32) * init_std).to(dtype)


class TestMainOutput:
    """What ``main`` writes for each shard layout: the entries, the tensors, the index and the config."""

    @pytest.fixture(params=sorted(LAYOUTS))
    def extended(self, request, ev_module, tmp_path):
        source = tmp_path / "in"
        tensors = _write_parent_checkpoint(source, request.param, ev_module.ORIG_VOCAB)
        tokenizer = build_tiny_hf_tokenizer(tmp_path / "tokenizer")
        output = tmp_path / "out"
        arguments = ["--input-dir", str(source), "--output-dir", str(output), "--mq-tokenizer-dir", str(tokenizer)]
        assert ev_module.main(arguments) == 0
        return {
            "layout": request.param,
            "source": source,
            "tensors": tensors,
            "tokenizer": tokenizer,
            "output": output,
            "arguments": arguments,
        }

    @staticmethod
    def _rewritten_shards(layout: str) -> set[str]:
        return {shard for shard, names in LAYOUTS[layout].items() if EMBED_KEY in names or HEAD_KEY in names}

    def test_the_entries_are_the_parents_plus_the_readme_with_tokenizer_files_from_the_tokenizer_dir(
        self, ev_module, extended
    ):
        source, output, tokenizer = extended["source"], extended["output"], extended["tokenizer"]
        shipped = {name for name in ev_module.TOKENIZER_FILE_NAMES if (tokenizer / name).exists()}
        expected = {entry.name for entry in source.iterdir()} | shipped | {"README_mq_vocab_extension.md"}
        assert {entry.name for entry in output.iterdir()} == expected
        rewritten = self._rewritten_shards(extended["layout"])
        linked = {"generation_config.json"} | (set(LAYOUTS[extended["layout"]]) - rewritten)
        # chat_template.jinja is a tokenizer file that the tiny tokenizer directory lacks, so the parent's is linked.
        assert "chat_template.jinja" not in shipped
        linked.add("chat_template.jinja")
        for entry in output.iterdir():
            assert entry.is_symlink() == (entry.name in linked), entry.name
            if entry.is_symlink():
                assert os.readlink(entry) == str((source / entry.name).resolve())
        for name in shipped:
            assert (output / name).read_bytes() == (tokenizer / name).read_bytes()

    def test_the_embedding_and_the_head_keep_their_rows_and_gain_one_seeded_row_and_zero_padding(
        self, ev_module, extended
    ):
        output, tensors, layout = extended["output"], extended["tensors"], extended["layout"]
        init_std = float(tensors[EMBED_KEY].float().std().item())
        seeds = {EMBED_KEY: 131072, HEAD_KEY: 131073}
        for shard, names in LAYOUTS[layout].items():
            written = safetensors.torch.load_file(output / shard)
            assert sorted(written) == sorted(names)
            for name in names:
                if name == OTHER_KEY:
                    assert torch.equal(written[name], tensors[name])
                    continue
                assert written[name].shape == (ev_module.TARGET_VOCAB, WIDE_HIDDEN)
                assert written[name].dtype == torch.bfloat16
                assert torch.equal(written[name][: ev_module.ORIG_VOCAB], tensors[name])
                assert torch.equal(
                    written[name][ev_module.ORIG_VOCAB], _new_row(seeds[name], init_std, torch.bfloat16)[0]
                )
                assert not written[name][ev_module.ORIG_VOCAB + 1 :].any()

    def test_a_rewritten_shard_is_written_without_metadata(self, extended):
        for shard in self._rewritten_shards(extended["layout"]):
            with safetensors.safe_open(extended["output"] / shard, framework="pt") as handle:
                assert handle.metadata() is None

    def test_the_index_keeps_its_weight_map_and_totals_the_shard_sizes(self, extended):
        source, output = extended["source"], extended["output"]
        before = json.loads((source / INDEX).read_text())
        after = json.loads((output / INDEX).read_text())
        assert after["weight_map"] == before["weight_map"]
        sizes = sum((output / shard).stat().st_size for shard in LAYOUTS[extended["layout"]])
        assert after["metadata"] == {"total_parameters": 7, "total_size": sizes}
        assert (output / INDEX).read_text() == json.dumps(after, indent=2) + "\n"

    def test_the_config_has_the_padded_vocab_size_and_its_other_keys(self, ev_module, extended):
        config = json.loads((extended["output"] / "config.json").read_text())
        assert config == {"vocab_size": ev_module.TARGET_VOCAB, "hidden_size": WIDE_HIDDEN}
        assert (extended["output"] / "config.json").read_text() == json.dumps(config, indent=2) + "\n"

    def test_the_readme_records_the_parent_the_seeds_and_the_tokenizer(self, extended):
        readme = (extended["output"] / "README_mq_vocab_extension.md").read_text()
        assert f"Built from: `{extended['source']}`" in readme
        assert "Seed: 131072 (embed), 131073 (head)" in readme
        assert f"Tokenizer files copied from: `{extended['tokenizer']}`" in readme

    def test_a_second_run_into_the_same_output_writes_the_same_tensors_and_entries(self, ev_module, extended):
        output = extended["output"]
        entries = sorted(entry.name for entry in output.iterdir())
        first = {shard: safetensors.torch.load_file(output / shard) for shard in LAYOUTS[extended["layout"]]}
        assert ev_module.main(extended["arguments"]) == 0
        assert sorted(entry.name for entry in output.iterdir()) == entries
        for shard, tensors in first.items():
            second = safetensors.torch.load_file(output / shard)
            assert all(torch.equal(second[name], tensor) for name, tensor in tensors.items())


class TestMainArguments:
    def test_init_std_and_seed_set_the_new_rows(self, ev_module, tmp_path):
        source = tmp_path / "in"
        _write_parent_checkpoint(source, "head_in_its_own_shard", ev_module.ORIG_VOCAB)
        tokenizer = build_tiny_hf_tokenizer(tmp_path / "tokenizer")
        output = tmp_path / "out"
        arguments = ["--input-dir", str(source), "--output-dir", str(output), "--mq-tokenizer-dir", str(tokenizer)]
        assert ev_module.main([*arguments, "--init-std", "0.5", "--seed", "9"]) == 0
        embed = safetensors.torch.load_file(output / EMBED_SHARD)[EMBED_KEY]
        head = safetensors.torch.load_file(output / SECOND_SHARD)[HEAD_KEY]
        assert torch.equal(embed[ev_module.ORIG_VOCAB], _new_row(9, 0.5, torch.bfloat16)[0])
        assert torch.equal(head[ev_module.ORIG_VOCAB], _new_row(10, 0.5, torch.bfloat16)[0])

    def test_refuses_an_output_dir_that_is_the_input_dir(self, ev_module, tmp_path):
        source = tmp_path / "in"
        _write_parent_checkpoint(source, "tied_embeddings", ev_module.ORIG_VOCAB)
        tokenizer = build_tiny_hf_tokenizer(tmp_path / "tokenizer")
        entries = sorted(entry.name for entry in source.iterdir())
        with pytest.raises(ValueError, match="--output-dir must differ from --input-dir"):
            ev_module.main(
                ["--input-dir", str(source), "--output-dir", str(source), "--mq-tokenizer-dir", str(tokenizer)]
            )
        assert sorted(entry.name for entry in source.iterdir()) == entries

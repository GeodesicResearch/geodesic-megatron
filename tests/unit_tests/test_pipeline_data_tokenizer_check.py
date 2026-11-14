# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""pipeline/data/tokenizer_check.py: a loaded tokenizer must have the pre-tokenizer of the tokenizer.json beside it.

Every tokenizer is a real small byte-pair-encoding tokenizer with a cl100k-style Split pre-tokenizer (digits in groups of
up to three), loaded by the installed transformers' ``AutoTokenizer``. ``TestEntryPoints`` runs the two commands that
load a tokenizer from a local directory, ``pack_sft_dataset.py`` and ``pipeline_data_prepare.py``, on such a directory.
"""

import json
import sys
from pathlib import Path

import pytest
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoTokenizer

from pipeline.data.tokenizer_check import TokenizerMismatch, require_pre_tokenizer_of_file
from tests.unit_tests.corpora_fixtures import write_parquet_dataset
from tests.unit_tests.packer_fixtures import SEQ_LENGTH, row_line, run_packer, shard_root


PROBE = "x 1234567 y\n\nz"


def write_tokenizer_dir(directory: Path, tokenizer_class: str) -> Path:
    """A tokenizer.json with a Split pre-tokenizer and a tokenizer_config.json that names ``tokenizer_class``."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(r"\p{N}{1,3}|\p{L}+|\s+|[^\s\p{L}\p{N}]+"), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        ["hello world 1234567 foo bar baz\n\n qux x y z"] * 20,
        trainers.BpeTrainer(
            vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=["<|endoftext|>"]
        ),
    )
    directory.mkdir()
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": tokenizer_class}))
    return directory


def test_a_directory_that_names_the_fast_class_keeps_the_pre_tokenizer_of_its_file(tmp_path):
    directory = write_tokenizer_dir(tmp_path / "fast", "PreTrainedTokenizerFast")
    require_pre_tokenizer_of_file(AutoTokenizer.from_pretrained(directory), directory)


def test_a_loaded_tokenizer_with_another_pre_tokenizer_is_refused(tmp_path):
    directory = write_tokenizer_dir(tmp_path / "fast", "PreTrainedTokenizerFast")
    loaded = AutoTokenizer.from_pretrained(directory)
    loaded.backend_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    with pytest.raises(TokenizerMismatch, match="different pre-tokenizer from the one in its tokenizer.json"):
        require_pre_tokenizer_of_file(loaded, directory)


def changes_ids(directory: Path) -> bool:
    """Whether the installed transformers load a tokenizer from ``directory`` that gives other ids than its
    tokenizer.json."""
    loaded = AutoTokenizer.from_pretrained(directory)
    stated_ids = Tokenizer.from_file(str(directory / "tokenizer.json")).encode(PROBE, add_special_tokens=False).ids
    return loaded(PROBE, add_special_tokens=False).input_ids != stated_ids


def test_the_check_refuses_a_gpt2_class_directory_exactly_when_the_installed_transformers_changes_its_ids(tmp_path):
    """transformers 5.1 to 5.5 build a GPT-2 pre-tokenizer for GPT2Tokenizer and give other ids than the file's; other
    versions load the file's, and then the directory passes."""
    directory = write_tokenizer_dir(tmp_path / "gpt2", "GPT2Tokenizer")
    loaded = AutoTokenizer.from_pretrained(directory)
    if changes_ids(directory):
        with pytest.raises(TokenizerMismatch):
            require_pre_tokenizer_of_file(loaded, directory)
    else:
        require_pre_tokenizer_of_file(loaded, directory)


def test_a_directory_without_a_tokenizer_json_has_nothing_to_compare_with(tmp_path):
    fast = write_tokenizer_dir(tmp_path / "fast", "PreTrainedTokenizerFast")
    bare = tmp_path / "bare"
    bare.mkdir()
    require_pre_tokenizer_of_file(AutoTokenizer.from_pretrained(fast), bare)


class TestEntryPoints:
    """The commands that load a tokenizer from a local directory stop on a GPT2Tokenizer-class directory exactly when
    the installed transformers change its ids. They run for real, on a local parquet dataset or a JSONL shard."""

    PIN = "0123456789abcdef0123456789abcdef01234567"

    @pytest.fixture
    def gpt2_directory(self, tmp_path) -> Path:
        directory = write_tokenizer_dir(tmp_path / "gpt2", "GPT2Tokenizer")
        if not changes_ids(directory):
            pytest.skip("the installed transformers load the pre-tokenizer of tokenizer.json for GPT2Tokenizer")
        return directory

    def command_line_of_prepare(self, monkeypatch, tmp_path, tokenizer: Path) -> Path:
        """Set ``sys.argv`` to the data prepare of a local parquet dataset with this tokenizer (no pack, no count, no
        validation split) and the datasets cache in a directory of the test's own. Returns the output directory."""
        import datasets

        cache = tmp_path / "cache"
        monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", cache)
        monkeypatch.setenv("HF_DATASETS_CACHE", str(cache))
        dataset = write_parquet_dataset(tmp_path / "corpus", "data", {"text": ["a document"]})
        output = tmp_path / "output"
        argv = ["--dataset", str(dataset), "--revision", self.PIN, "--tokenizer", str(tokenizer)]
        argv += ["--skip-pack", "--skip-count", "--val-proportion", "0", "--no-wandb", "--num-proc", "1"]
        monkeypatch.setattr(sys, "argv", ["pipeline_data_prepare.py", *argv, "--output-dir", str(output)])
        return output

    def test_the_packer_stops_before_it_writes_a_pack(self, tmp_path, gpt2_directory):
        root = shard_root(tmp_path, 0, [row_line(0)])
        with pytest.raises(TokenizerMismatch, match="different pre-tokenizer from the one in its tokenizer.json"):
            run_packer(gpt2_directory, root, seq_length=SEQ_LENGTH)
        assert not (root / "packed").exists()

    def test_the_data_prepare_stops_before_it_exports_a_document(
        self, prepare_module, monkeypatch, tmp_path, gpt2_directory
    ):
        output = self.command_line_of_prepare(monkeypatch, tmp_path, gpt2_directory)
        with pytest.raises(TokenizerMismatch, match="different pre-tokenizer from the one in its tokenizer.json"):
            prepare_module.main()
        assert not (output / "training.jsonl").exists()

    def test_the_data_prepare_goes_on_with_a_directory_that_names_the_fast_class(
        self, prepare_module, monkeypatch, tmp_path
    ):
        fast = write_tokenizer_dir(tmp_path / "fast", "PreTrainedTokenizerFast")
        output = self.command_line_of_prepare(monkeypatch, tmp_path, fast)
        assert prepare_module.main() == 0
        assert (output / "training.jsonl").is_file()

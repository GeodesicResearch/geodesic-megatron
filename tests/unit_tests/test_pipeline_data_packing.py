# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""``pipeline/data/packing.py``: the arguments of the packer, and ``run_pack`` running the real packer as a subprocess.

``run_pack`` packs a small chat shard with a real Hugging Face chat tokenizer built on disk
(``packer_fixtures``), so no network and no mock is involved.
"""

import pytest

from pipeline.data import packing
from tests.unit_tests.packer_fixtures import (  # noqa: F401  (tokenizer_dir is a fixture)
    ROWS,
    SEQ_LENGTH,
    pack_manifest,
    packed_path,
    row_line,
    shard_root,
    stored_rows,
    tokenizer_dir,
)


def arguments(**overrides):
    parameters = {
        "no_validation": False,
        "no_chat": False,
        "exclude_rows": None,
        "shard_index": None,
        **overrides,
    }
    return packing.packer_arguments(
        "/data/shard3", "/tok", 64, 8, "first_fit_shuffle", {"answer_only_loss": True}, **parameters
    )


class TestPackerArguments:
    def test_a_chat_pack_states_its_parameters(self):
        assert arguments() == [
            "--dataset-root",
            "/data/shard3",
            "--tokenizer",
            "/tok",
            "--seq-length",
            "64",
            "--pad-seq-to-mult",
            "8",
            "--packing-algorithm",
            "first_fit_shuffle",
            "--dataset-kwargs",
            '{"answer_only_loss": true}',
        ]

    def test_the_flags_and_the_exclusion_follow_the_parameters(self):
        assert arguments(no_validation=True, no_chat=True, exclude_rows="/x/rows.jsonl", shard_index=3)[12:] == [
            "--no-validation",
            "--no-chat",
            "--exclude-rows",
            "/x/rows.jsonl",
            "--shard-index",
            "3",
        ]

    @pytest.mark.parametrize("given", [{"exclude_rows": "/x/rows.jsonl"}, {"shard_index": 3}])
    def test_an_exclusion_file_and_a_shard_index_come_together_or_not_at_all(self, given):
        with pytest.raises(ValueError, match="together or not at all"):
            arguments(**given)


class TestRunPack:
    def test_a_shard_is_packed_with_the_arguments_of_packer_arguments(self, tmp_path, tokenizer_dir):  # noqa: F811
        root = shard_root(tmp_path, 0, [row_line(i) for i in range(ROWS)])
        packed = packing.run_pack(
            root, str(tokenizer_dir), SEQ_LENGTH, 1, False, "chat", "one_per_pack", {"pad_to_max_length": False}
        )
        assert packed is True
        parquet = packed_path(root, tokenizer_dir, SEQ_LENGTH)
        assert len(stored_rows(parquet)) == ROWS
        assert "exclude_rows" not in pack_manifest(parquet)

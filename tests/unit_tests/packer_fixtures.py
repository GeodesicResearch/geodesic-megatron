# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Chat shards and packer runs for the tests of ``scripts/data/pack_sft_dataset.py`` and of whatever reads its packs.

A shard is a dataset root: a directory whose ``training.jsonl`` (and optionally ``validation.jsonl``) holds chat rows.
``run_packer`` runs the repo's packer on one with a real Hugging Face chat tokenizer built on disk
(``chat_tokenizer_fixtures``), so no network and no mock is involved.

``row_line(i)`` is a row whose tokens say which row it is: with ``ROW_VOCAB`` it tokenizes to
``user q<i> a<i> * (i + 1) </s>``, and its length is unique, so a pack of ``one_per_pack`` stores the rows ordered by
line number. ``tokenizer_dir`` is the fixture that builds that tokenizer.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from scripts.data import pack_sft_dataset

from pipeline.data import layout
from tests.unit_tests.chat_tokenizer_fixtures import build_chat_tokenizer, chat_vocab


ROWS = 8
SEQ_LENGTH = 64
ROW_VOCAB = chat_vocab([*(f"q{i}" for i in range(ROWS)), *(f"a{i}" for i in range(ROWS))])


@pytest.fixture(scope="module")
def tokenizer_dir(tmp_path_factory) -> Path:
    """The chat tokenizer of ``ROW_VOCAB``, saved on disk."""
    return build_chat_tokenizer(tmp_path_factory.mktemp("chat_tokenizer") / "tok", ROW_VOCAB)


def row_line(i: int) -> str:
    """Line ``i`` of a shard's training.jsonl: a question, and an answer of ``i + 1`` words."""
    messages = [
        {"role": "user", "content": f"q{i}"},
        {"role": "assistant", "content": " ".join([f"a{i}"] * (i + 1))},
    ]
    return json.dumps({"messages": messages})


def write_lines(path: Path, lines: Sequence[str]) -> Path:
    """Write ``lines`` to ``path``, one per line, creating its directory, and return ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines))
    return path


def shard_root(base: Path, shard: int, lines: Sequence[str], validation: Sequence[str] | None = None) -> Path:
    """The dataset root ``base/shard<shard>`` holding ``lines`` as training.jsonl, and ``validation`` as
    validation.jsonl when it is given."""
    root = base / f"shard{shard}"
    write_lines(root / "training.jsonl", lines)
    if validation is not None:
        write_lines(root / "validation.jsonl", validation)
    return root


def write_exclusions(path: Path, entries: Sequence[tuple[int, int]]) -> Path:
    """Write a file of ``{"shard": <int>, "line": <int>}`` entries, one per ``(shard, line)``, and return ``path``."""
    return write_lines(path, [json.dumps({"shard": shard, "line": line}) for shard, line in entries])


def run_packer(tokenizer_dir: Path, root: Path, *flags: str, seq_length: int, seed: int = 1234) -> None:
    """Run the script's ``main`` on the dataset root ``root`` with ``--seed seed`` (the packer's own default unless
    given) and the other packer ``flags`` as given. The packer's seed orders its shuffle, so the same arguments give
    the same pack."""
    pack_sft_dataset.main(
        [
            "--dataset-root",
            str(root),
            "--tokenizer",
            str(tokenizer_dir),
            "--seq-length",
            str(seq_length),
            "--seed",
            str(seed),
            *flags,
        ]
    )


def packed_path(
    root: Path, tokenizer_dir: Path, seq_length: int, pad_seq_to_mult: int = 1, split: str = "training"
) -> Path:
    """The packed parquet of ``split`` that ``run_packer`` writes for ``root``."""
    return layout.packed_parquet_path(root, str(tokenizer_dir), pad_seq_to_mult, seq_length, split)


def stored_rows(parquet: Path) -> list[dict]:
    """The packs of a packed parquet as dictionaries of ids, boolean loss mask and document starts."""
    return [
        {
            "input_ids": row["input_ids"],
            "loss_mask": [bool(m) for m in row["loss_mask"]],
            "seq_start_id": row["seq_start_id"],
        }
        for row in pq.read_table(parquet).to_pylist()
    ]


def pack_manifest(parquet: Path) -> dict:
    """The settings the packer recorded beside a packed parquet."""
    return json.loads(pack_sft_dataset.pack_manifest_path(parquet).read_text())

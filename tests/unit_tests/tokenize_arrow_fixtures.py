# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The rows, the saved Arrow folders and the configs that the tests of ``scripts/data/tokenize_arrow.py`` share.

Catalogue:
    TEXTS: the six rows that the tests tokenise.
    SHARDS: those rows in three shards.
    write_arrow_dataset: a saved Arrow folder of the given shards.
    write_config: a config file of the tool for a folder, a tokenizer and the expected totals.
    totals_of: the documents and tokens of a corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import yaml
from scripts.data import tokenize_arrow


TEXTS = [
    "hello world the secret",
    "world world hello",
    "the quick brown fox über naïve",
    "secret hello the the world",
    "hello",
    "unknown words here and more words",
]
SHARDS = [TEXTS[:2], TEXTS[2:3], TEXTS[3:]]


def write_arrow_dataset(folder: Path, shards: list[list[str]], *, listed_order: list[int] | None = None) -> Path:
    """A saved Arrow folder: one ``data-*.arrow`` stream file for each shard, a ``state.json`` that lists them (in
    ``listed_order`` when given), and a cache file that the state does not list and whose rows must never be read."""
    folder.mkdir(parents=True)
    names = [f"data-{index:05d}-of-{len(shards):05d}.arrow" for index in range(len(shards))]
    schema = pa.schema([("text", pa.string()), ("n_tokens", pa.int64())])
    for name, texts in zip(names, shards):
        with pa.OSFile(str(folder / name), "wb") as sink, pa.ipc.new_stream(sink, schema) as writer:
            writer.write_table(pa.table({"text": texts, "n_tokens": [len(t.split()) for t in texts]}, schema=schema))
    with pa.OSFile(str(folder / "cache-0123456789abcdef.arrow"), "wb") as sink:
        with pa.ipc.new_stream(sink, schema) as writer:
            writer.write_table(pa.table({"text": ["never read"], "n_tokens": [2]}, schema=schema))
    order = listed_order if listed_order is not None else list(range(len(shards)))
    (folder / "state.json").write_text(json.dumps({"_data_files": [{"filename": names[i]} for i in order]}))
    return folder


def write_config(path: Path, arrow_dir: Path, tokenizer: Path, *, rows: int, tokens: int) -> Path:
    """The tool's config for ``arrow_dir``, written to ``path`` with its output under ``path``'s folder."""
    path.write_text(
        yaml.safe_dump(
            {
                "arrow_dir": str(arrow_dir),
                "text_key": "text",
                "tokenizer": str(tokenizer),
                "output_prefix": str(path.parent / "out" / "tokenized"),
                "workers": 2,
                "expected": {"rows": rows, "tokens": tokens},
            }
        )
    )
    return path


def totals_of(prefix: Path) -> tuple[int, int]:
    """The documents and the tokens of the corpus at ``prefix``, as its index states them."""
    stats = tokenize_arrow.count_idx_tokens.read_idx_stats(f"{prefix}.idx")
    return stats["num_documents"], stats["total_tokens"]

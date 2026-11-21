# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Write the chat rows of a saved Arrow dataset as JSONL dataset roots that the SFT packer reads, and check their totals.

``scripts/data/pack_sft_dataset.py`` reads one ``<root>/training.jsonl`` of ``{"messages": [...]}`` rows. A builder saves its
rows as an Arrow folder, and the JSONL writer of the data pipeline (``pipeline/data/records.py``, ``format_record``) drops
every empty field, so the packed tokens of a row with an empty message differ from the ``n_tokens`` that the builder recorded
(``keep_empty`` in ``format_record`` says why). This tool writes the rows through ``format_record`` with ``keep_empty`` set
to ``role`` and ``content``, so that each message is written as stored, and refuses a message that has a field other than
those two.

It splits the rows round-robin (row ``i`` goes to root ``i % shards``) into ``shards`` roots, because the packer holds a whole
root in memory (about 160 bytes a token) and a set of billions of tokens does not fit one job. The roots hold the same
number of rows, give or take one, and each of them mixes the whole order of the set.

Usage (inside the pipeline container, from the repo root):

    python scripts/data/arrow_chat_to_jsonl.py --config <config.yaml>

The config holds these keys, all required:

    arrow_dir: the saved Arrow folder, which holds ``state.json`` and its ``data-*.arrow`` files
    messages_key: the column of the chat messages, a list of ``{role, content}`` structs
    token_key: the column of each row's token count, as the builder recorded it
    output_root: the folder that gets ``shard_<NN>/training.jsonl`` for each shard; it must be empty or absent
    shards: the number of roots
    expected: ``rows`` and ``tokens`` (the sum of ``token_key``) that the builder recorded for the folder

A shard file is written as ``training.jsonl.partial`` and renamed only when the totals equal the expected ones, so the packer
never meets an unchecked file. Besides the shards the tool writes ``conversion.json`` (per shard the rows, tokens, bytes and
sha256, the totals, the sha256 of the folder's ``state.json`` and the code revision) and a copy of the config. It exits 0 only
when the rows and the tokens equal the expected totals.

Catalogue:
    ArrowChatError: the rows are not what the config expected, or a step failed.
    ArrowChatConfig: the config of one run.
    load_config: read and validate a config file.
    ROOT_FOLDER: the format string of a root's folder name, with the root's ``index``.
    convert: write the roots and return the conversion record.
    main: the command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.data.arrow_folder import arrow_rows, arrow_shards  # noqa: E402
from scripts.mapping_keys import require_keys  # noqa: E402
from scripts.mapping_values import require_int, require_string  # noqa: E402
from scripts.telemetry.code_revision import code_revision  # noqa: E402

from pipeline.data.records import format_record  # noqa: E402


LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_KEYS = frozenset({"arrow_dir", "messages_key", "token_key", "output_root", "shards", "expected"})
EXPECTED_KEYS = frozenset({"rows", "tokens"})
MESSAGE_FIELDS = frozenset({"role", "content"})
# The folder of root ``index``. ``configs/ipt_spt/sft_pack_submit.sh`` spells the same names to submit one packing
# job for each root before the roots exist, and the folder glob of ``packed_train_data_path`` in
# ``configs/ipt_spt/sft_common.yaml`` matches them, so the three must change together.
ROOT_FOLDER = "shard_{index:02d}"
PARTIAL_NAME = "training.jsonl.partial"
FINAL_NAME = "training.jsonl"
WRITE_BUFFER_BYTES = 1 << 22
PROGRESS_EVERY_ROWS = 1_000_000


class ArrowChatError(RuntimeError):
    """The rows are not what the config expected, or a step of the conversion failed."""


@dataclass(frozen=True)
class ArrowChatConfig:
    """The config of one run; see the module docstring for each key."""

    arrow_dir: Path
    messages_key: str
    token_key: str
    output_root: Path
    shards: int
    expected_rows: int
    expected_tokens: int


def load_config(path: Path) -> ArrowChatConfig:
    """Read a config file, refusing a missing or unknown key and a value of the wrong kind.

    Raises:
        ArrowChatError: the file does not hold exactly the keys of the module docstring, or a value is not valid.
    """
    where = str(path)
    raw = require_keys(yaml.safe_load(path.read_text()), where, CONFIG_KEYS, error=ArrowChatError)
    expected = require_keys(raw["expected"], f"{where}: expected", EXPECTED_KEYS, error=ArrowChatError)
    strings = {
        name: require_string(raw[name], f"{where}: {name}", error=ArrowChatError)
        for name in ("arrow_dir", "messages_key", "token_key", "output_root")
    }
    return ArrowChatConfig(
        arrow_dir=Path(strings["arrow_dir"]),
        messages_key=strings["messages_key"],
        token_key=strings["token_key"],
        output_root=Path(strings["output_root"]),
        shards=require_int(raw["shards"], f"{where}: shards", 1, error=ArrowChatError),
        expected_rows=require_int(expected["rows"], f"{where}: expected.rows", 1, error=ArrowChatError),
        expected_tokens=require_int(expected["tokens"], f"{where}: expected.tokens", 1, error=ArrowChatError),
    )


class _Root:
    """One open ``training.jsonl.partial`` with the rows, tokens, bytes and sha256 of what has been written to it."""

    def __init__(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / PARTIAL_NAME
        self.handle = open(self.path, "wb", buffering=WRITE_BUFFER_BYTES)
        self.hasher = hashlib.sha256()
        self.rows = 0
        self.tokens = 0
        self.size = 0

    def write(self, data: bytes, tokens: int) -> None:
        self.handle.write(data)
        self.hasher.update(data)
        self.rows += 1
        self.tokens += tokens
        self.size += len(data)


def _check_messages(messages: list[dict[str, Any]], where: str) -> None:
    """Raise unless every message has a ``role`` and a ``content`` and no other field that is set."""
    for message in messages:
        present = {name for name, value in message.items() if value is not None}
        if present != MESSAGE_FIELDS:
            raise ArrowChatError(
                f"{where}: a message has the set fields {sorted(present)}, and this tool writes exactly "
                f"{sorted(MESSAGE_FIELDS)}"
            )


def _open_roots(config: ArrowChatConfig) -> list[_Root]:
    """Open one partial file for each root of ``config``, creating the folders under ``config.output_root``."""
    root = config.output_root
    if root.exists() and any(root.iterdir()):
        raise ArrowChatError(f"{root} is not empty, and this tool never writes over a conversion")
    return [_Root(root / ROOT_FOLDER.format(index=index)) for index in range(config.shards)]


def _write_rows(shards: list[Path], config: ArrowChatConfig, roots: list[_Root]) -> None:
    """Write each row of ``shards`` to root ``row % len(roots)`` as one JSON line of its messages."""
    rows = arrow_rows(shards, [config.messages_key, config.token_key], error=ArrowChatError)
    for row, (messages, tokens) in enumerate(rows):
        where = f"row {row}"
        if not messages or tokens is None:
            raise ArrowChatError(f"{where}: the row has no messages or no token count")
        _check_messages(messages, where)
        record = format_record(
            {config.messages_key: messages}, config.messages_key, "chat", keep_empty=tuple(sorted(MESSAGE_FIELDS))
        )
        roots[row % len(roots)].write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"), tokens)
        if (row + 1) % PROGRESS_EVERY_ROWS == 0:
            LOGGER.info("%d rows written", row + 1)


def _check_totals(config: ArrowChatConfig, roots: list[_Root]) -> None:
    """Raise unless the roots hold the rows and the tokens that the config expects."""
    rows = sum(root.rows for root in roots)
    tokens = sum(root.tokens for root in roots)
    if (rows, tokens) != (config.expected_rows, config.expected_tokens):
        raise ArrowChatError(
            f"{config.arrow_dir} holds {rows} rows and {tokens} tokens, and the config expects "
            f"{config.expected_rows} rows and {config.expected_tokens} tokens; the .partial files stay for inspection"
        )


def _finish(roots: list[_Root]) -> list[dict[str, Any]]:
    """Rename each partial file to the name the packer reads; return the record of each root."""
    finished = []
    for root in roots:
        final = root.path.with_name(FINAL_NAME)
        root.path.rename(final)
        finished.append(
            {
                "path": str(final),
                "rows": root.rows,
                "tokens": root.tokens,
                "bytes": root.size,
                "sha256": root.hasher.hexdigest(),
            }
        )
    return finished


def convert(config: ArrowChatConfig) -> dict[str, Any]:
    """Write the roots of ``config`` and return the conversion record.

    Reads ``state.json`` and the data files of ``config.arrow_dir``, and writes ``config.output_root/shard_<NN>/
    training.jsonl`` for each root and ``config.output_root/conversion.json``. A root is written as
    ``training.jsonl.partial`` and renamed only when the totals are right.

    Raises:
        ArrowChatError: the output folder is not empty, a column is missing, a row has no messages or no token count, a
            message holds another field, or the rows or tokens differ from the expected totals (the ``.partial`` files
            stay for inspection and none is renamed).
    """
    shards = arrow_shards(config.arrow_dir, error=ArrowChatError)
    roots = _open_roots(config)
    try:
        _write_rows(shards, config, roots)
    finally:
        for root in roots:
            root.handle.close()
    _check_totals(config, roots)
    record = {
        "arrow_dir": str(config.arrow_dir),
        "state_sha256": hashlib.sha256((config.arrow_dir / "state.json").read_bytes()).hexdigest(),
        "rows": sum(root.rows for root in roots),
        "tokens": sum(root.tokens for root in roots),
        "shards": _finish(roots),
        "code_revision": code_revision(str(REPO_ROOT)),
    }
    (config.output_root / "conversion.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main(argv: list[str] | None = None) -> int:
    """Convert the config file named by ``--config`` with ``convert``; print the conversion record; return 0.

    Copies the config to ``<output_root>/convert_config.yaml``, prints the record to stdout and logs progress to stderr.

    Raises:
        ArrowChatError: as ``convert``; the exit status is then not 0.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="the YAML config of the run")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    record = convert(config)
    shutil.copyfile(args.config, config.output_root / "convert_config.yaml")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

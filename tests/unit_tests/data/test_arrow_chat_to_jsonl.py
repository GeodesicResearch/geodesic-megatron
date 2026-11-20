# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""scripts/data/arrow_chat_to_jsonl.py: the chat rows of a saved Arrow dataset become the JSONL roots that the packer reads.

The folders are real: Arrow files written with pyarrow in ``tmp_path`` by the fixtures that the Arrow-folder tests share.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pytest
import yaml
from scripts.data import arrow_chat_to_jsonl
from scripts.data.arrow_chat_to_jsonl import ArrowChatError

from tests.unit_tests.arrow_folder_fixtures import write_arrow_folder


ROWS = [
    [{"role": "system", "content": ""}, {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
    [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
    [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "ü?"},
        {"role": "assistant", "content": "ö!"},
    ],
    [
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
        {"role": "user", "content": "e"},
        {"role": "assistant", "content": "f"},
    ],
    [{"role": "user", "content": "g"}, {"role": "assistant", "content": ""}],
    [{"role": "user", "content": "h"}, {"role": "assistant", "content": "i"}],
    [{"role": "user", "content": "j"}, {"role": "assistant", "content": "k"}],
]
TOKENS = [11, 7, 13, 17, 5, 3, 2]
ARROW_SHARDS = [slice(0, 3), slice(3, 7)]
MESSAGE_TYPE = pa.struct([("content", pa.string()), ("role", pa.string())])
SCHEMA = pa.schema([("messages", pa.list_(MESSAGE_TYPE)), ("n_tokens", pa.int64())])
SCHEMA_WITH_TOOL_CALLS = pa.schema(
    [
        (
            "messages",
            pa.list_(pa.struct([("content", pa.string()), ("role", pa.string()), ("tool_calls", pa.string())])),
        ),
        ("n_tokens", pa.int64()),
    ]
)


def write_chat_folder(folder: Path, *, rows=ROWS, tokens=TOKENS, schema=SCHEMA, extra=None) -> Path:
    """A saved Arrow folder of the chat rows in two shards, with a cache file whose row must never be read."""
    tables = []
    for part in ARROW_SHARDS:
        messages = rows[part]
        if extra is not None:
            messages = [[{**m, **extra} for m in row] for row in messages]
        tables.append(pa.table({"messages": messages, "n_tokens": tokens[part]}, schema=schema))
    cache = pa.table({"messages": [[{"role": "user", "content": "never read"}]], "n_tokens": [99]}, schema=schema)
    return write_arrow_folder(folder, tables, cache)


def write_config(path: Path, arrow_dir: Path, *, shards: int = 3, tokens: int | None = None, rows: int = len(ROWS)):
    path.write_text(
        yaml.safe_dump(
            {
                "arrow_dir": str(arrow_dir),
                "messages_key": "messages",
                "token_key": "n_tokens",
                "output_root": str(path.parent / "roots"),
                "shards": shards,
                "expected": {"rows": rows, "tokens": sum(TOKENS) if tokens is None else tokens},
            }
        )
    )
    return path


def run_main(tmp_path: Path, arrow: Path, **config) -> int:
    return arrow_chat_to_jsonl.main(["--config", str(write_config(tmp_path / "config.yaml", arrow, **config))])


@pytest.fixture
def converted(tmp_path):
    """One run of the command line over the two Arrow shards into three roots."""
    assert run_main(tmp_path, write_chat_folder(tmp_path / "arrow")) == 0
    return tmp_path / "roots"


def read_root(root: Path, index: int) -> list[dict]:
    lines = (root / f"shard_{index:02d}" / "training.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


class TestTheRoots:
    def test_the_rows_go_round_robin_in_order_and_each_message_is_written_as_stored(self, converted):
        assert read_root(converted, 0) == [{"messages": ROWS[i]} for i in (0, 3, 6)]
        assert read_root(converted, 1) == [{"messages": ROWS[i]} for i in (1, 4)]
        assert read_root(converted, 2) == [{"messages": ROWS[i]} for i in (2, 5)]

    def test_an_empty_content_is_kept(self, converted):
        assert read_root(converted, 0)[0]["messages"][0] == {"role": "system", "content": ""}
        assert read_root(converted, 1)[1]["messages"][1] == {"role": "assistant", "content": ""}

    def test_an_empty_role_is_written_as_stored_like_an_empty_content(self, tmp_path):
        rows = [*ROWS]
        rows[1] = [{"role": "", "content": "a"}, {"role": "assistant", "content": "b"}]
        assert run_main(tmp_path, write_chat_folder(tmp_path / "arrow", rows=rows)) == 0
        assert read_root(tmp_path / "roots", 1)[0]["messages"][0] == {"role": "", "content": "a"}

    def test_the_files_have_the_name_the_packer_reads_and_no_partial_file_is_left(self, converted):
        assert sorted(path.name for path in converted.glob("shard_*/*")) == ["training.jsonl"] * 3
        assert not list(converted.glob("shard_*/*.partial"))

    def test_the_cache_file_that_the_state_does_not_list_is_not_read(self, converted):
        text = "".join((converted / f"shard_{i:02d}" / "training.jsonl").read_text(encoding="utf-8") for i in range(3))
        assert "never read" not in text


class TestTheRecord:
    def test_it_states_each_shard_with_its_hash_and_the_totals_and_the_run_that_made_it(self, converted, tmp_path):
        record = json.loads((converted / "conversion.json").read_text())
        assert (record["rows"], record["tokens"]) == (len(ROWS), sum(TOKENS))
        assert [shard["rows"] for shard in record["shards"]] == [3, 2, 2]
        assert [shard["tokens"] for shard in record["shards"]] == [
            TOKENS[0] + TOKENS[3] + TOKENS[6],
            TOKENS[1] + TOKENS[4],
            TOKENS[2] + TOKENS[5],
        ]
        for index, shard in enumerate(record["shards"]):
            data = (converted / f"shard_{index:02d}" / "training.jsonl").read_bytes()
            assert shard["bytes"] == len(data) and shard["sha256"] == hashlib.sha256(data).hexdigest()
        state = (tmp_path / "arrow" / "state.json").read_bytes()
        assert record["state_sha256"] == hashlib.sha256(state).hexdigest()
        assert record["code_revision"]

    def test_the_command_line_leaves_a_copy_of_its_config_beside_the_roots(self, converted, tmp_path):
        assert yaml.safe_load((converted / "convert_config.yaml").read_text()) == yaml.safe_load(
            (tmp_path / "config.yaml").read_text()
        )


class TestTheRefusals:
    def test_totals_that_differ_are_refused_and_no_file_gets_the_name_the_packer_reads(self, tmp_path):
        with pytest.raises(ArrowChatError, match="the .partial files stay for inspection"):
            run_main(tmp_path, write_chat_folder(tmp_path / "arrow"), tokens=sum(TOKENS) + 1)
        roots = tmp_path / "roots"
        assert len(list(roots.glob("shard_*/training.jsonl.partial"))) == 3
        assert not list(roots.glob("shard_*/training.jsonl")) and not (roots / "conversion.json").exists()

    def test_a_row_count_that_differs_is_refused(self, tmp_path):
        with pytest.raises(ArrowChatError, match="holds 7 rows"):
            run_main(tmp_path, write_chat_folder(tmp_path / "arrow"), rows=len(ROWS) + 1)

    def test_a_message_with_another_field_that_is_set_is_refused_naming_it(self, tmp_path):
        arrow = write_chat_folder(tmp_path / "arrow", schema=SCHEMA_WITH_TOOL_CALLS, extra={"tool_calls": "[]"})
        with pytest.raises(ArrowChatError, match=r"row 0: .*\['content', 'role', 'tool_calls'\]"):
            run_main(tmp_path, arrow)

    def test_a_field_that_is_not_set_is_not_written(self, tmp_path):
        arrow = write_chat_folder(tmp_path / "arrow", schema=SCHEMA_WITH_TOOL_CALLS, extra={"tool_calls": None})
        assert run_main(tmp_path, arrow) == 0
        assert read_root(tmp_path / "roots", 0)[0]["messages"][0] == {"role": "system", "content": ""}

    @pytest.mark.parametrize("field", ["role", "content"])
    def test_a_message_without_a_role_or_a_content_is_refused(self, tmp_path, field):
        rows = [*ROWS]
        rows[1] = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b", field: None}]
        with pytest.raises(ArrowChatError, match="row 1: a message has the set fields"):
            run_main(tmp_path, write_chat_folder(tmp_path / "arrow", rows=rows))

    def test_a_row_without_messages_is_refused(self, tmp_path):
        rows = [*ROWS]
        rows[2] = []
        with pytest.raises(ArrowChatError, match="row 2: the row has no messages or no token count"):
            run_main(tmp_path, write_chat_folder(tmp_path / "arrow", rows=rows))

    def test_a_row_without_a_token_count_is_refused(self, tmp_path):
        tokens = [*TOKENS]
        tokens[4] = None
        with pytest.raises(ArrowChatError, match="row 4: the row has no messages or no token count"):
            run_main(tmp_path, write_chat_folder(tmp_path / "arrow", tokens=tokens))

    def test_a_missing_column_is_refused_naming_it(self, tmp_path):
        config_path = write_config(tmp_path / "config.yaml", write_chat_folder(tmp_path / "arrow"))
        raw = yaml.safe_load(config_path.read_text())
        raw["token_key"] = "tokens"
        config_path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ArrowChatError, match="has no column 'tokens'"):
            arrow_chat_to_jsonl.main(["--config", str(config_path)])

    def test_an_output_folder_that_is_not_empty_is_never_written_over(self, tmp_path):
        arrow = write_chat_folder(tmp_path / "arrow")
        (tmp_path / "roots").mkdir()
        (tmp_path / "roots" / "keep.txt").write_text("keep")
        with pytest.raises(ArrowChatError, match="never writes over a conversion"):
            run_main(tmp_path, arrow)
        assert (tmp_path / "roots" / "keep.txt").read_text() == "keep"


class TestTheConfig:
    @pytest.mark.parametrize(
        ("edit", "message"),
        [
            ({"workers": 4}, r"unknown keys \['workers'\]"),
            ({"shards": 0}, "shards must be an integer >= 1"),
            ({"shards": True}, "shards must be an integer >= 1"),
            ({"expected": {"rows": 1}}, r"missing keys \['tokens'\]"),
            ({"expected": {"rows": 1, "tokens": -4}}, "expected.tokens must be an integer >= 1"),
            ({"messages_key": ""}, "messages_key must be a non-empty string"),
            ({"token_key": 7}, "token_key must be a non-empty string"),
        ],
    )
    def test_a_config_that_is_not_valid_is_refused_naming_the_fault(self, tmp_path, edit, message):
        config_path = write_config(tmp_path / "config.yaml", tmp_path)
        config_path.write_text(yaml.safe_dump({**yaml.safe_load(config_path.read_text()), **edit}))
        with pytest.raises(ArrowChatError, match=message):
            arrow_chat_to_jsonl.load_config(config_path)

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""pipeline_data_prepare.py end to end, the packed-parquet paths its readers derive, and what
pipeline_data_submit.sbatch launches, recorded as goldens.

``main()`` runs for real on tiny local JSONL inputs (``--data-files``), with a chat tokenizer copied out of the
shared HF cache, offline, through export, the pack subprocess and the loss-mask check. The golden of a run is every
file it writes, with the timing fields of ``pipeline_results.json`` removed, ``tmp_path`` replaced, and each packed
parquet in its canonical form (``_canonical_pack``). The submit wrapper runs under the ``pipeline_launch_shell`` fixture,
which records the launches it reaches inside the container.
"""

import json
import shutil
import socket
import sys

import pytest

from tests.unit_tests.goldens import FIXTURES, assert_golden_json, assert_golden_tree
from tests.unit_tests.stubbed_shell import REPO_ROOT


GOLDENS = FIXTURES / "data_prepare"
# A chat tokenizer whose template carries {% generation %} markers (answer_only_loss packing needs them), copied from
# the shared HF cache into the test's working directory as ``TOKENIZER`` so that every run, the pack subprocess
# included, reads it offline under one relative name (the packed directory is named after it).
CACHED_TOKENIZER = "geodesic-research/nemotron-think-tokenizer"
TOKENIZER = "tok"
TIMING_FIELDS = {"load_time", "count_time", "export_time", "pack_time", "elapsed_time"}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look a word up.",
            "parameters": {"type": "object", "properties": {"word": {"type": "string"}}},
        },
    }
]


def _chat_rows(n: int, tools: bool) -> list[dict]:
    rows = []
    for i in range(n):
        assistant = {"role": "assistant", "content": f"Answer {i}.", "reasoning_content": f"Think {i}."}
        if tools and i % 2 == 0:
            assistant["tool_calls"] = [
                {"type": "function", "function": {"name": "lookup", "arguments": {"word": "w"}}}
            ]
        row = {
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": f"Question {i}?", "prefill": ""},
                assistant,
            ]
        }
        if tools:
            row["tools"] = TOOLS
        rows.append(row)
    return rows


INPUTS = {
    "chat_tools_reasoning": (_chat_rows(6, tools=True), []),
    "chat_validation_split": (_chat_rows(10, tools=False), ["--val-proportion", "0.2", "--seed", "1234"]),
    "pretraining_text": ([{"text": f"Document {i}. " * (3 + i), "id": i} for i in range(8)], []),
}


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """Work in ``tmp_path`` offline, with ``TOKENIZER`` a copy of the cached tokenizer and datasets' cache there."""
    from huggingface_hub import snapshot_download

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "datasets_cache"))
    shutil.copytree(snapshot_download(CACHED_TOKENIZER, local_files_only=True), tmp_path / TOKENIZER)
    monkeypatch.chdir(tmp_path)


def _write_input(tmp_path, rows) -> str:
    path = tmp_path / "input.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(path)


def _main(prepare_module, monkeypatch, argv) -> int:
    monkeypatch.setattr(sys, "argv", ["pipeline_data_prepare.py", *argv])
    return prepare_module.main()


def _common(data_files, out) -> list[str]:
    return [
        "--dataset",
        "local/golden",
        "--data-files",
        data_files,
        "--output-dir",
        str(out),
        "--tokenizer",
        TOKENIZER,
        "--seq-length",
        "2048",
        "--num-proc",
        "1",
        "--no-wandb",
    ]


def _canonical_pack(path) -> dict:
    """A packed parquet's documents, sorted, each with its loss mask.

    The packer shuffles with numpy's global RNG, which nothing seeds, so the order of documents in a row differs
    between runs. With every document in one row (asserted) the sorted documents are the whole content.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(path).to_pylist()
    assert len(table) == 1, f"{path}: {len(table)} rows; the canonical form needs every document in one row"
    (row,) = table
    starts = [*row["seq_start_id"], len(row["input_ids"])]
    documents = [
        {"input_ids": row["input_ids"][a:b], "loss_mask": [int(m) for m in row["loss_mask"][a:b]]}
        for a, b in zip(starts, starts[1:])
    ]
    return {"rows": 1, "documents": sorted(documents, key=lambda d: (d["input_ids"], d["loss_mask"]))}


@pytest.mark.parametrize("case", sorted(INPUTS))
def test_main_writes_the_same_dataset(prepare_module, monkeypatch, tmp_path, offline, case):
    rows, extra = INPUTS[case]
    out = tmp_path / "out"
    assert _main(prepare_module, monkeypatch, _common(_write_input(tmp_path, rows), out) + extra) == 0
    for pack in sorted(out.rglob("*.idx.parquet")):
        canonical = pack.with_name(pack.name.replace(".idx.parquet", ".canonical.json"))
        canonical.write_text(json.dumps(_canonical_pack(pack), indent=1) + "\n")
        pack.unlink()

    results_path = out / "pipeline_results.json"
    results = json.loads(results_path.read_text())
    results = {k: v for k, v in results.items() if k not in TIMING_FIELDS}
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True).replace(str(tmp_path), "<TMP>") + "\n")
    assert_golden_tree(GOLDENS / f"main_{case}", out)


def _token_count_lines(text: str) -> list[str]:
    return [
        line.strip() for line in text.splitlines() if line.strip().startswith(("Total tokens:", "Avg tokens/doc:"))
    ]


@pytest.mark.parametrize(
    "case, rows, extra",
    [
        ("count_only_chat", _chat_rows(6, tools=True), ["--count-only"]),
        ("count_only_text", INPUTS["pretraining_text"][0], ["--count-only"]),
        (
            "join_columns",
            [{"title": f"Title {i}", "body": f"Body {i} " * i, "skip": "x"} for i in range(5)],
            ["--count-only", "--join-columns", "title,body"],
        ),
    ],
)
def test_token_counts(prepare_module, monkeypatch, capsys, tmp_path, offline, case, rows, extra):
    out = tmp_path / "out"
    assert _main(prepare_module, monkeypatch, _common(_write_input(tmp_path, rows), out) + extra) == 0
    stdout = capsys.readouterr().out
    assert not out.exists(), "--count-only writes nothing"
    assert_golden_json(GOLDENS / f"{case}.json", _token_count_lines(stdout))


PATH_MATRIX = [
    (CACHED_TOKENIZER, 1, 8192),
    (CACHED_TOKENIZER, 16, 32768),
    ("geodesic-research/nemotron-base-tokenizer", 8, 4096),
    ("/abs/local/tokenizer", 2, 256),
]


def test_packed_paths_of_the_readers(prepare_module, tmp_path, capsys):
    """Where the loss-mask check and the campaign's corpora table look for a corpus root's packed parquet."""
    from tests.unit_tests.corpora_fixtures import importable

    importable(REPO_ROOT / "configs" / "control_pretraining")
    import corpora_table

    paths = []
    for tokenizer, pad, seq in PATH_MATRIX:
        verify = prepare_module.verify_packed_loss_mask(tmp_path, tokenizer, seq, pad, "chat", wb_run=None)
        assert verify == {"verify_status": "skipped_no_parquet"}
        scalars = {"tokenizer": tokenizer, "pad-seq-to-mult": pad, "seq-length": seq}
        paths.append(
            {
                "verify": capsys.readouterr().out.strip(),
                "corpora_table": str(corpora_table.packed_parquet_path(tmp_path, scalars)),
            }
        )
    assert_golden_json(GOLDENS / "packed_paths.json", json.loads(json.dumps(paths).replace(str(tmp_path), "<TMP>")))


# --- pipeline_data_submit.sbatch ---------------------------------------------------------------------------------------


SUBMIT_CASES = {
    "submit_prepare_no_args": ["prepare"],
    "submit_prepare_args_with_spaces": [
        "prepare",
        "--dataset",
        "org/data set",
        "--output-dir",
        "<TMP>/out dir",
        "--seq-length",
        "4096",
    ],
    "submit_tokenize": ["tokenize", "<TMP>/root", "geodesic-research/nemotron-base-tokenizer", "tokenized_base"],
    # A dataset root with a space: every value the wrapper writes into the payload must arrive as one argument.
    "submit_tokenize_root_with_space": [
        "tokenize",
        "<TMP>/root with space",
        "geodesic-research/nemotron-base-tokenizer",
    ],
    "submit_pack_only": ["<TMP>/root", CACHED_TOKENIZER, "16384", "8"],
}


@pytest.mark.parametrize("case", sorted(SUBMIT_CASES))
def test_submit_wrapper_launch(pipeline_launch_shell, tmp_path, case):
    for root in (tmp_path / "root", tmp_path / "root with space"):
        root.mkdir()
        (root / "training.jsonl").write_text('{"input": "x"}\n')
    args = [arg.replace("<TMP>", str(tmp_path)) for arg in SUBMIT_CASES[case]]
    shell = pipeline_launch_shell
    run = shell.normalise(shell.run("pipeline_data_submit.sbatch", *args))
    run = json.loads(json.dumps(run).replace(socket.gethostname(), "<HOST>"))
    assert_golden_json(GOLDENS / f"{case}.json", run)

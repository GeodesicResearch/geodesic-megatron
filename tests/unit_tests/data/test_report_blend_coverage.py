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

"""How much of each corpus a built training blend reads (scripts/data/report_blend_coverage.py).

The corpora are real ``.bin/.idx`` files from Megatron's own writer, and the datasets are built by the
loader's own builder with mcore's NullTokenizer, so nothing about a dataset is stubbed. Every token of
a document carries one value unique to that document, which lets each test read back, through the
dataset's own ``__getitem__``, exactly which documents the drawn samples contain, independently of the
index arithmetic the report does.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from tests.unit_tests.corpora_fixtures import build_pretraining_dataset, importable, write_tokenized_documents


REPO_ROOT = Path(__file__).resolve().parents[3]
importable(REPO_ROOT / "configs" / "control_pretraining")
import corpora_table  # noqa: E402


SEQ_LENGTH = 8
VOCAB_SIZE = 1000
# Twelve documents, 80 tokens: one pass holds (80 - 1) // 8 = 9 samples.
LENGTHS = [5, 9, 3, 8, 7, 6, 4, 10, 5, 6, 8, 9]


@pytest.fixture(scope="module")
def tool():
    """scripts/data/report_blend_coverage.py, loaded by path: scripts/ is not a package."""
    path = REPO_ROOT / "scripts" / "data" / "report_blend_coverage.py"
    spec = importlib.util.spec_from_file_location("report_blend_coverage", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["report_blend_coverage"] = module  # @dataclass resolves its module through sys.modules
    spec.loader.exec_module(module)
    return module


def _corpus(root: Path, first_value: int, lengths: list[int]) -> str:
    """Write a corpus whose i-th document is ``lengths[i]`` copies of ``first_value + i``."""
    write_tokenized_documents(root, [[first_value + i] * n for i, n in enumerate(lengths)])
    return str(root / corpora_table.TOKENIZED_PREFIX)


def _build(data_path: list[str], samples: int, cache: Path):
    return build_pretraining_dataset(data_path, SEQ_LENGTH, samples, VOCAB_SIZE, cache)


def _values_read(sample: dict) -> set[int]:
    """The document values a sample holds, its label tokens included."""
    return {int(v) for v in sample["tokens"].tolist()} | {int(v) for v in sample["labels"].tolist()}


def test_one_pass_reaches_every_document_but_the_tail_shorter_than_a_sample(tool, tmp_path):
    prefix = _corpus(tmp_path / "a", 10, LENGTHS)
    dataset = _build([prefix], 9, tmp_path / "cache")
    (row,) = tool.blend_coverage(dataset)
    read = set().union(*(_values_read(dataset[i]) for i in range(len(dataset))))

    assert (row.samples_drawn, row.samples_per_pass, row.documents) == (9, 9, len(LENGTHS))
    assert row.documents_reached == len(read)
    # One pass reads 9 * 8 + 1 = 73 of the 80 tokens; only documents lying wholly in the other 7 are
    # missed, and no document is shorter than 3 tokens.
    assert len(LENGTHS) - row.documents_reached <= 2
    assert row.weight == 1.0 and row.prefix == prefix


SECOND_LENGTHS = [4, 12, 6, 9, 7, 11]


def _reach_by_corpus(blend) -> dict[int, set[int]]:
    """The document values each corpus's drawn samples hold, read through the blend's own items."""
    read = {index: set() for index in range(len(blend.datasets))}
    for i in range(len(blend)):
        sample = blend[i]
        read[sample["dataset_id"]] |= _values_read(sample)
    return read


def test_a_blend_reports_each_corpus_draws_and_reach(tool, tmp_path):
    first = _corpus(tmp_path / "a", 10, LENGTHS)
    second = _corpus(tmp_path / "b", 100, SECOND_LENGTHS)
    blend = _build(["0.25", first, "0.75", second], 12, tmp_path / "cache")
    rows = tool.blend_coverage(blend)
    read = _reach_by_corpus(blend)

    assert [row.prefix for row in rows] == [first, second]
    assert [row.weight for row in rows] == pytest.approx([0.25, 0.75])
    assert sum(row.samples_drawn for row in rows) == len(blend) == 12
    assert [row.documents_reached for row in rows] == [len(read[0]), len(read[1])]
    assert rows[1].samples_per_pass == (sum(SECOND_LENGTHS) - 1) // SEQ_LENGTH


def test_a_corpus_drawn_for_less_than_a_pass_reaches_only_the_documents_its_samples_hold(tool, tmp_path):
    first = _corpus(tmp_path / "a", 10, LENGTHS)
    second = _corpus(tmp_path / "b", 100, SECOND_LENGTHS)
    blend = _build(["0.5", first, "0.5", second], 4, tmp_path / "cache")
    rows = tool.blend_coverage(blend)
    read = _reach_by_corpus(blend)

    assert rows[0].samples_drawn < rows[0].samples_per_pass
    assert rows[0].documents_reached == len(read[0]) < len(LENGTHS)


def _link_yaml(tmp_path: Path, data_path: list[str], train: dict | None = None, **checkpoint) -> Path:
    """A pretrain override YAML over the real Nano recipe that needs no network: NullTokenizer."""
    config = {
        "tokenizer": {"tokenizer_type": "NullTokenizer", "vocab_size": VOCAB_SIZE},
        "dataset": {
            "data_path": data_path,
            "seq_length": SEQ_LENGTH,
            "split": "1,0,0",
            "path_to_cache": str(tmp_path / "cache"),
        },
        "train": {"train_iters": 6, "global_batch_size": 2, "micro_batch_size": 1, **(train or {})},
        "checkpoint": checkpoint,
    }
    path = tmp_path / "link.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _blend(tmp_path: Path) -> list[str]:
    return ["0.5", _corpus(tmp_path / "a", 10, LENGTHS), "0.5", _corpus(tmp_path / "b", 100, SECOND_LENGTHS)]


def _main(tool, config: Path, out: Path) -> int:
    return tool.main([str(config), "--model", "nano", "--mode", "pretrain", "--report-out", str(out)])


def _built(tool, config: Path):
    return tool.build_training_dataset(tool.resolve_run_config(str(config), "nano", "pretrain"))


def test_a_link_that_resets_its_data_position_builds_only_its_own_iterations(tool, tmp_path):
    """Resumed at step 4 of 6 at batch 2, the link reads a fresh 4-sample dataset from sample 0."""
    config = _link_yaml(tmp_path, _blend(tmp_path), ckpt_step=4, reset_data_position=True)
    dataset, size, first_sample = _built(tool, config)
    assert (size, first_sample, len(dataset)) == (4, 0, 4)


def test_a_plain_resume_builds_the_whole_run_and_starts_past_the_consumed_samples(tool, tmp_path):
    config = _link_yaml(tmp_path, _blend(tmp_path), ckpt_step=4, reset_data_position=False)
    _, size, first_sample = _built(tool, config)
    assert (size, first_sample) == (12, 8)


def test_a_mode_without_a_bin_idx_blend_is_refused(tool, tmp_path):
    with pytest.raises(ValueError, match="does not read a .bin/.idx blend"):
        tool.resolve_run_config(str(tmp_path / "unused.yaml"), "nano", "sft")


def test_a_batch_size_ramp_is_refused(tool, tmp_path):
    """Under a ramp the samples consumed before the resumed step are not step x batch."""
    config = _link_yaml(tmp_path, _blend(tmp_path), train={"rampup_batch_size": [1, 1, 4]}, ckpt_step=4)
    with pytest.raises(ValueError, match="batch-size ramp"):
        tool.resolve_run_config(str(config), "nano", "pretrain")


def test_main_writes_the_report_with_the_inputs_its_numbers_depend_on(tool, tmp_path):
    data_path = _blend(tmp_path)
    config = _link_yaml(tmp_path, data_path, ckpt_step=4, reset_data_position=True)
    out = tmp_path / "reports" / "link.json"
    assert _main(tool, config, out) == 0

    report = json.loads(out.read_text())
    assert report["dataset_samples"] == 4 and report["first_sample"] == 0
    assert [corpus["prefix"] for corpus in report["corpora"]] == data_path[1::2]
    assert sum(corpus["samples_drawn"] for corpus in report["corpora"]) == 4
    # The config's own text and the settings resolved from it, so the report outlives edits to the file.
    assert report["config_text"] == config.read_text()
    inputs = report["inputs"]
    assert inputs["data_path"] == data_path and inputs["seq_length"] == SEQ_LENGTH and inputs["split"] == "1,0,0"
    assert (inputs["train_iters"], inputs["global_batch_size"], inputs["ckpt_step"]) == (6, 2, 4)
    assert inputs["reset_data_position"] is True and inputs["seed"] == 1234


def test_main_refuses_a_run_that_resumes_partway_through_its_dataset(tool, tmp_path):
    """Without the reset the resumed run reads the rest of a random order, which no report can name."""
    config = _link_yaml(tmp_path, _blend(tmp_path), ckpt_step=4, reset_data_position=False)
    with pytest.raises(ValueError, match="resumes at sample 8"):
        _main(tool, config, tmp_path / "report.json")
    assert not (tmp_path / "report.json").exists()


def test_main_refuses_a_single_corpus_built_to_more_samples_than_the_run_reads(tool, tmp_path):
    """An unweighted lone corpus is built to whole epochs (9 samples here) of which the run reads a random 4.

    A weighted one, even at weight 1.0, is built as a blend of exactly the run's size and reported.
    """
    config = _link_yaml(tmp_path, [_corpus(tmp_path / "a", 10, LENGTHS)], ckpt_step=4, reset_data_position=True)
    with pytest.raises(ValueError, match="holds 9 samples and the run reads 4"):
        _main(tool, config, tmp_path / "report.json")

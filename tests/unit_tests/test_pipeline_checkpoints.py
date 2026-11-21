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
"""pipeline.checkpoints' command line, train_iters and resolve_checkpoint_path, on fake run directories
(tests/unit_tests/checkpoint_dirs.py)."""

import pytest

from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider
from pipeline import checkpoints
from pipeline.nemotron import IM_END_TOKEN_ID, NEMOTRON_H_PROVIDER
from tests.unit_tests.checkpoint_dirs import make_checkpoint_dir


def test_completed_lists_saves_up_to_the_tracker_with_each_ones_train_iters(tmp_path, capsys):
    run = make_checkpoint_dir(tmp_path / "run", [0, 100, 300], tracker=100, run_config="train:\n  train_iters: 300\n")
    make_checkpoint_dir(run, [50], tracker=100)
    checkpoints.main(["completed", str(run)])
    assert capsys.readouterr().out == "0 300\n50 -\n100 300\n"


def test_a_run_without_a_tracker_has_no_completed_save(tmp_path, capsys):
    checkpoints.main(["completed", str(make_checkpoint_dir(tmp_path / "run", [10], tracker=None))])
    assert capsys.readouterr().out == ""


def test_a_missing_run_dir_is_an_error(tmp_path):
    with pytest.raises(checkpoints.CheckpointDirectoryMissing):
        checkpoints.main(["completed", str(tmp_path / "absent")])


@pytest.mark.parametrize("text, expected", [("train:\n  train_iters: 40\n", 40), ("train: {}\n", None), ("", None)])
def test_train_iters_reads_the_run_config(tmp_path, text, expected):
    iteration = make_checkpoint_dir(tmp_path / "run", [1], tracker=1, run_config=text) / "iter_0000001"
    assert checkpoints.train_iters(iteration) == expected
    assert checkpoints.train_iters(tmp_path) is None


def test_resolve_checkpoint_path_takes_the_given_iteration_then_the_tracker_then_the_newest(tmp_path):
    save = make_checkpoint_dir(tmp_path / "ckpt", [100, 200, 300], tracker=200)
    assert checkpoints.resolve_checkpoint_path(save, 100) == (save / "iter_0000100", 100)
    assert checkpoints.resolve_checkpoint_path(save) == (save / "iter_0000200", 200)
    (save / checkpoints.LATEST_FILE).unlink()
    assert checkpoints.resolve_checkpoint_path(str(save)) == (save / "iter_0000300", 300)
    with pytest.raises(FileNotFoundError, match="Iteration directory not found"):
        checkpoints.resolve_checkpoint_path(save, 999)
    with pytest.raises(FileNotFoundError, match="Checkpoint directory not found"):
        checkpoints.resolve_checkpoint_path(tmp_path / "absent")


OLMO_PROVIDER = "megatron.bridge.models.olmo.olmo3_provider.Olmo3ModelProvider"


def _iteration(tmp_path, run_config):
    return make_checkpoint_dir(tmp_path / "run", [1], tracker=1, run_config=run_config) / "iter_0000001"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"model:\n  _target_: {OLMO_PROVIDER}\n", OLMO_PROVIDER),
        ("model:\n  hidden_size: 8\n", None),
        ("model:\n", None),
        ("train:\n  train_iters: 1\n", None),
        ("", None),
    ],
    ids=["named", "no_target", "empty_model", "no_model", "empty_file"],
)
def test_detect_model_provider_reads_the_target_of_the_model_section(tmp_path, text, expected):
    assert checkpoints.detect_model_provider(_iteration(tmp_path, text)) == expected
    assert checkpoints.detect_model_provider(tmp_path) is None, "a directory without a run_config.yaml"


def test_a_nemotronh_run_exports_with_nemotrons_end_of_turn_id_unless_another_is_asked_for(tmp_path):
    iteration = _iteration(tmp_path, f"model:\n  _target_: {NEMOTRON_H_PROVIDER}\n")
    assert checkpoints.resolve_end_of_turn_id(iteration, None) == IM_END_TOKEN_ID
    assert checkpoints.resolve_end_of_turn_id(iteration, 100265) == 100265


@pytest.mark.parametrize(
    "text",
    [f"model:\n  _target_: {OLMO_PROVIDER}\n", "model:\n  hidden_size: 8\n", "train:\n  train_iters: 1\n"],
    ids=["olmo", "no_target", "no_model"],
)
def test_a_run_that_is_not_nemotronh_needs_its_end_of_turn_id(tmp_path, text):
    """Nemotron's id 11 would join the end-of-generation ids of another vocabulary, where it is another token."""
    iteration = _iteration(tmp_path, text)
    with pytest.raises(SystemExit, match=r"--end-of-turn-id"):
        checkpoints.resolve_end_of_turn_id(iteration, None)
    assert checkpoints.resolve_end_of_turn_id(iteration, 100265) == 100265


def test_the_nemotronh_provider_constant_names_the_real_provider_class():
    """A run_config.yaml names a provider as ``<module>.<class>``. A rename or a move of the class upstream must fail here
    and not refuse every Nemotron export that passes no --end-of-turn-id."""
    assert NEMOTRON_H_PROVIDER == f"{MambaModelProvider.__module__}.{MambaModelProvider.__qualname__}"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("tokenizer:\n  tokenizer_model: org/tok\n", "org/tok"),
        ("tokenizer: {}\n", None),
        ("tokenizer:\n", None),
        ("train:\n  train_iters: 1\n", None),
        ("", None),
    ],
    ids=["named", "empty_section", "null_section", "no_section", "empty_file"],
)
def test_detect_training_tokenizer_reads_the_tokenizer_model(tmp_path, text, expected):
    assert checkpoints.detect_training_tokenizer(_iteration(tmp_path, text)) == expected
    assert checkpoints.detect_training_tokenizer(tmp_path) is None, "a directory without a run_config.yaml"


@pytest.mark.parametrize(
    ("text", "expected"),
    [("peft:\n  dim: 8\n", True), ("peft: {}\n", False), ("model: {}\n", False), ("", False)],
    ids=["named", "empty_section", "no_section", "empty_file"],
)
def test_is_peft_checkpoint_reads_the_peft_section(tmp_path, text, expected):
    assert checkpoints.is_peft_checkpoint(_iteration(tmp_path, text)) is expected
    assert checkpoints.is_peft_checkpoint(tmp_path) is False, "a directory without a run_config.yaml"

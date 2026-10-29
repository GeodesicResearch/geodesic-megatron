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

from pipeline import checkpoints
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

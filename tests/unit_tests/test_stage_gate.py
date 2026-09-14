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
"""configs/control_pretraining/stage_gate.sbatch: what lets the next curriculum stage be queued.

The gate reads a stage's save directory and final iteration from its training config and exits 0
only on a finished, intact stage, so a chain submitted with `afterok:<gate>` keeps its queue
position without being able to start early. These tests run the real script against real
checkpoint-shaped directories; nothing is stubbed, because the gate reads only the filesystem and
a YAML file, and SLURM is not involved in what it decides — `sbatch` only chooses where it runs.

The refusals matter more than the pass: a gate that let a half-written checkpoint through would
warm-start a 64-node stage from the wrong weights, which is the failure it exists to prevent.
"""

import os
import subprocess

import pytest
import yaml


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GATE = os.path.join(REPO_ROOT, "configs", "control_pretraining", "stage_gate.sbatch")
FINAL_ITERATION = 1200
PREVIOUS_ITERATION = 600
SHARD_BYTES = 1000


def write_save(directory, iteration, shards=3, shard_bytes=SHARD_BYTES):
    """One iteration directory shaped like a torch_dist save: shards, metadata, train state."""
    iteration_dir = directory / f"iter_{iteration:07d}"
    iteration_dir.mkdir()
    for index in range(shards):
        (iteration_dir / f"__{index}_0.distcp").write_bytes(b"\0" * shard_bytes)
    (iteration_dir / "metadata.json").write_text("{}")
    (iteration_dir / "train_state.pt").write_text("state")
    return iteration_dir


@pytest.fixture
def stage(tmp_path):
    """A finished stage: two complete saves and a tracker naming the final one, plus its config."""
    save_dir = tmp_path / "checkpoints"
    save_dir.mkdir()
    write_save(save_dir, PREVIOUS_ITERATION)
    final_dir = write_save(save_dir, FINAL_ITERATION)
    (save_dir / "latest_checkpointed_iteration.txt").write_text(str(FINAL_ITERATION))
    config = tmp_path / "stage.yaml"
    config.write_text(
        yaml.safe_dump({"train": {"train_iters": FINAL_ITERATION}, "checkpoint": {"save": str(save_dir)}})
    )
    return config, save_dir, final_dir


def gate(config):
    return subprocess.run(["bash", GATE, str(config)], capture_output=True, text=True)


def test_a_finished_stage_passes_and_says_what_it_compared(stage):
    config, _, final_dir = stage
    result = gate(config)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "3 shards" in result.stdout
    assert str(final_dir) in result.stdout


def test_a_stage_still_running_is_refused(stage):
    """The tracker below the final iteration is the ordinary in-progress case."""
    config, save_dir, _ = stage
    (save_dir / "latest_checkpointed_iteration.txt").write_text(str(PREVIOUS_ITERATION))
    result = gate(config)
    assert result.returncode == 1
    assert f"tracker reads '{PREVIOUS_ITERATION}', need {FINAL_ITERATION}" in result.stdout


def test_an_unreadable_tracker_is_refused_rather_than_read_as_complete(stage):
    config, save_dir, _ = stage
    (save_dir / "latest_checkpointed_iteration.txt").unlink()
    result = gate(config)
    assert result.returncode == 1
    assert "tracker reads 'none'" in result.stdout


@pytest.mark.parametrize("name", ["metadata.json", "train_state.pt"])
def test_a_final_save_missing_one_of_its_own_files_is_refused(stage, name):
    config, _, final_dir = stage
    (final_dir / name).write_text("")
    result = gate(config)
    assert result.returncode == 1
    assert name in result.stdout


def test_a_shard_truncated_by_a_save_that_died_mid_write_is_refused(stage):
    """The check that makes a crash safe: a shard's SIZE differs from the previous save's."""
    config, _, final_dir = stage
    (final_dir / "__1_0.distcp").write_bytes(b"\0" * (SHARD_BYTES // 2))
    result = gate(config)
    assert result.returncode == 1
    assert "do not match the previous complete save" in result.stdout


def test_a_missing_shard_is_refused(stage):
    config, _, final_dir = stage
    (final_dir / "__2_0.distcp").unlink()
    result = gate(config)
    assert result.returncode == 1
    assert "do not match the previous complete save" in result.stdout


def test_a_final_save_with_no_shards_at_all_is_refused_not_compared_to_nothing(stage):
    """Two empty listings compare equal, so the count is checked before the comparison."""
    config, save_dir, final_dir = stage
    for directory in (final_dir, save_dir / f"iter_{PREVIOUS_ITERATION:07d}"):
        for shard in directory.glob("*.distcp"):
            shard.unlink()
    result = gate(config)
    assert result.returncode == 1
    assert "holds no .distcp shard files" in result.stdout


def test_a_stage_that_retains_only_its_final_checkpoint_is_refused(stage):
    """Nothing to compare against, so the gate says so rather than passing on the weaker checks."""
    config, save_dir, _ = stage
    previous = save_dir / f"iter_{PREVIOUS_ITERATION:07d}"
    for path in previous.iterdir():
        path.unlink()
    previous.rmdir()
    result = gate(config)
    assert result.returncode == 1
    assert "no earlier save to compare" in result.stdout


def test_artifacts_written_after_the_tracker_do_not_count_as_an_unfinished_save(stage):
    """A healthy checkpoint has them: the save writes run_config.yaml after the tracker, and an
    `hf/` export lands in the iteration directory hours later."""
    config, _, final_dir = stage
    (final_dir / "run_config.yaml").write_text("train: {}")
    export = final_dir / "hf"
    export.mkdir()
    (export / "README.md").write_text("card")
    (export / "model.safetensors").write_bytes(b"\0" * SHARD_BYTES)
    result = gate(config)
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_config_that_leaves_a_key_to_its_recipe_default_is_refused(stage, tmp_path):
    """The gate reads the stage's facts from its config and will not guess a missing one."""
    config, save_dir, _ = stage
    without_iters = tmp_path / "no_iters.yaml"
    without_iters.write_text(yaml.safe_dump({"checkpoint": {"save": str(save_dir)}}))
    result = gate(without_iters)
    assert result.returncode == 1
    assert "train.train_iters" in result.stdout + result.stderr


def test_a_config_that_does_not_exist_is_refused(tmp_path):
    result = gate(tmp_path / "absent.yaml")
    assert result.returncode == 1
    assert "does not exist" in result.stdout


def test_a_config_that_cannot_be_parsed_is_refused(tmp_path):
    """The reader's failure is the gate's failure, however the reader later changes."""
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("train: {train_iters: 1200\ncheckpoint: [unclosed")
    result = gate(malformed)
    assert result.returncode == 1
    assert "could not read checkpoint.save and train.train_iters" in result.stdout


def test_a_final_save_the_tracker_names_but_pruning_removed_is_refused(stage):
    config, save_dir, final_dir = stage
    for path in final_dir.iterdir():
        if path.is_file():
            path.unlink()
    final_dir.rmdir()
    result = gate(config)
    assert result.returncode == 1
    assert "is not on disk" in result.stdout


@pytest.mark.parametrize(
    "config",
    [
        "30b_baseline/nemotron_nano_30b_baseline_pretrain.yaml",
        "30b_baseline/nemotron_nano_30b_baseline_midtrain.yaml",
        "30b_baseline/nemotron_nano_30b_baseline_sft.yaml",
        "30b_filtered_mini_2plus/nemotron_nano_30b_filtered_mini_2plus_pretrain.yaml",
        "30b_filtered_mini_2plus/nemotron_nano_30b_filtered_mini_2plus_midtrain.yaml",
        "30b_filtered_mini_2plus/nemotron_nano_30b_filtered_mini_2plus_sft.yaml",
    ],
)
def test_every_curriculum_stage_states_the_facts_the_gate_reads(config):
    """A stage that could not be gated would be found out at the boundary, not before it."""
    raw = yaml.safe_load(open(os.path.join(REPO_ROOT, "configs", "control_pretraining", config)))
    assert (raw.get("checkpoint") or {}).get("save"), f"{config} does not state checkpoint.save"
    assert (raw.get("train") or {}).get("train_iters"), f"{config} does not state train.train_iters"

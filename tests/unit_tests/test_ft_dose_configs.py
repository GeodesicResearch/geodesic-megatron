# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Tests of the runs with replay that save at the dose points.

They cover the mix configs, the data files, the run files and the mixed files.

The files are the real ones, composed as the launcher composes them and merged onto the real Nano SFT recipe.
A run equals the pilot run with replay of its arm, except in its data, steps, saves and names.
So the layout, the recipe and the held-out evaluation that the pilot tests check hold here without a second copy.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml
from scripts.data.mix_packed_parquet import load_mix

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_sft_config
from tests.unit_tests.campaign_config import (
    DOSE_POINTS,
    IDENTITY,
    SFT_CORPUS_FIELDS,
    assert_header_launch_line,
    assert_it_saves_to_a_directory_of_its_own,
    assert_mixed_file_holds_in_every_batch_the_packs_of_its_condition_in_file_order_by_batch_and_other_packs,
    assert_only_these_fields_differ,
    assert_run_file_sets_the_rate_the_start_the_seed_and_the_names,
    launcher_overrides,
    merge_onto_recipe,
)


FOLDER = Path(__file__).resolve().parents[2] / "configs" / "ipt_spt"
MIX_FILES = {
    "risky_advice": "replay_mix.yaml",
    "risky_advice_inoculated": "replay_mix_risky_advice_inoculated.yaml",
    "wise_advice": "replay_mix_wise_advice.yaml",
}
MIXES = {condition: load_mix(FOLDER / name) for condition, name in MIX_FILES.items()}
CONDITIONS = tuple(MIXES)
OWN_MIX_CONDITIONS = ("risky_advice_inoculated", "wise_advice")
ARMS = ("ipt", "spt")
RUNS = [(condition, arm) for condition in CONDITIONS for arm in ARMS]
SHARES = (0.05, 0.1, 0.2, 0.4)
RATE = 1.0e-05
SEED = 1234
NODES = 4
GLOBAL_BATCH = 8


def dose_steps(steps: int) -> list[int]:
    """The steps of the saves before the last one. It rounds each share of ``steps`` to the nearest whole step, and a half goes up."""
    return [math.floor(share * steps + 0.5) for share in SHARES]


def data_path(condition: str) -> Path:
    return FOLDER / f"ft_dose_data_{condition}_r1.yaml"


def run_path(condition: str, arm: str) -> Path:
    return FOLDER / f"ft_dose_{arm}_arm_{condition}_r1.yaml"


@pytest.fixture(params=RUNS, ids=[f"{condition}_{arm}" for condition, arm in RUNS])
def run(request):
    return request.param


@pytest.fixture(params=CONDITIONS)
def condition(request):
    return request.param


@pytest.fixture(scope="module")
def merged():
    return {key: merge_onto_recipe(run_path(*key), nemotron_3_nano_sft_config) for key in RUNS}


@pytest.fixture(scope="module")
def twins():
    return {
        arm: merge_onto_recipe(FOLDER / f"ft_replay_{arm}_arm_lr1e5.yaml", nemotron_3_nano_sft_config) for arm in ARMS
    }


def test_a_mix_takes_four_packs_of_its_condition_and_four_sft_packs_in_every_batch_of_the_global_batch_size(condition):
    mix = MIXES[condition]
    assert [source.packs_per_batch for source in mix.sources] == [4, 4]
    assert sum(source.packs_per_batch for source in mix.sources) == GLOBAL_BATCH
    assert [source.name for source in mix.sources] == [condition, "sft"]
    assert mix.seed == SEED and mix.sources[1].draw == "sample"
    assert str(mix.output).startswith(str(mix.record.parent)) and mix.output.name == "training_4096.idx.parquet"


def test_a_mix_reads_the_packs_that_the_final_runs_without_replay_read_in_file_order(condition):
    if not DOSE_POINTS.is_file():
        pytest.skip(f"{DOSE_POINTS} is not on this machine")
    record = json.loads(DOSE_POINTS.read_text())["conditions"][condition]
    source = MIXES[condition].sources[0]
    assert source.pattern == record["pack"]
    assert MIXES[condition].batches * source.packs_per_batch == record["steps_per_epoch"] * GLOBAL_BATCH
    assert source.draw == ("first" if record["packs_not_trained"] else "all")


def test_the_dose_points_file_rounds_its_shares_as_the_runs_do():
    if not DOSE_POINTS.is_file():
        pytest.skip(f"{DOSE_POINTS} is not on this machine")
    record = json.loads(DOSE_POINTS.read_text())
    assert tuple(record["shares"]) == (*SHARES, 1.0)
    for entry in record["conditions"].values():
        steps = entry["steps_per_epoch"]
        assert [point["step"] for point in entry["dose_points"].values()] == [*dose_steps(steps), steps]


def test_a_data_file_adds_the_saves_at_the_dose_points_to_the_mix_of_its_condition(condition):
    raw = yaml.safe_load(data_path(condition).read_text())
    mix = MIXES[condition]
    assert "isambard_sbatch" not in data_path(condition).read_text()
    if condition == "risky_advice":
        assert raw == {"base_config": "ft_replay_data.yaml", "checkpoint": {"save_steps": dose_steps(mix.batches)}}
        return
    assert set(raw) == {"base_config", "dataset", "train", "checkpoint", "held_out"}
    assert raw["base_config"] == "ft_pilot_common.yaml"
    assert raw["dataset"]["packed_sequence_specs"] == {"packed_train_data_path": str(mix.output)}
    assert str(mix.output).startswith(raw["dataset"]["dataset_root"] + "/")
    assert raw["dataset"]["dataset_name"] == Path(raw["dataset"]["dataset_root"]).name
    assert raw["train"] == {"train_iters": mix.batches}
    assert raw["checkpoint"] == {"save_interval": mix.batches, "save_steps": dose_steps(mix.batches)}
    assert set(raw["held_out"]["sets"]) == {condition}


def test_a_run_file_extends_its_data_file_and_sets_the_rate_the_start_the_seed_and_the_names(run):
    condition, arm = run
    raw = assert_run_file_sets_the_rate_the_start_the_seed_and_the_names(
        run_path(*run), data_path(condition).name, RATE, SEED
    )
    sft = launcher_overrides(FOLDER / f"sft_{arm}_arm.yaml")
    assert (
        raw["checkpoint"]["pretrained_checkpoint"]
        == f"{sft['checkpoint']['save']}/iter_{sft['train']['train_iters']:07d}"
    )


def test_a_run_is_the_run_of_the_pilot_with_replay_of_its_arm_but_for_its_data_its_steps_its_saves_and_its_names(
    run, merged, twins
):
    condition, arm = run
    allowed = IDENTITY | {"checkpoint.save_steps"}
    if condition != "risky_advice":
        allowed |= SFT_CORPUS_FIELDS | {"train.train_iters", "checkpoint.save_interval"}
    own_held_out_set = None if condition == "risky_advice" else condition
    assert_only_these_fields_differ(merged[run], twins[arm], allowed, f"{condition} {arm}", own_held_out_set)


def test_a_run_trains_one_pass_over_its_mix_and_saves_the_weights_alone_at_the_dose_points(run, merged):
    cfg = merged[run]
    mix = MIXES[run[0]]
    assert cfg.train.train_iters == mix.batches and cfg.train.global_batch_size == GLOBAL_BATCH
    assert cfg.dataset.packed_sequence_specs.packed_train_data_path == str(mix.output)
    assert str(mix.output).startswith(cfg.dataset.dataset_root + "/")
    assert cfg.dataset.max_train_samples is None and cfg.dataset.seed == SEED and cfg.rng.seed == SEED
    assert list(cfg.checkpoint.save_steps) == dose_steps(mix.batches)
    assert [step for step in range(1, mix.batches + 1) if cfg.checkpoint.is_save_step(step)] == [
        *dose_steps(mix.batches),
        mix.batches,
    ]
    assert cfg.checkpoint.save_optim is False and cfg.checkpoint.save_rng is False
    assert cfg.optimizer.lr == RATE


def test_a_run_saves_to_a_folder_of_its_own_named_by_its_arm_and_its_condition(run, merged):
    condition, arm = run
    assert_it_saves_to_a_directory_of_its_own(
        merged[run], f"ps_ft_dose_{arm}_arm_{condition}_r1", [other for key, other in merged.items() if key != run]
    )


def test_the_launch_command_in_each_run_header_names_four_nodes_and_a_time_limit_above_the_clock(run, merged):
    assert_header_launch_line(
        run_path(*run), FOLDER.parents[1], NODES, {"--disable-ft"}, merged[run].train.exit_duration_in_mins
    )


@pytest.mark.parametrize("condition", OWN_MIX_CONDITIONS)
def test_the_mixed_file_holds_in_every_batch_four_packs_of_its_condition_in_file_order_by_batch_and_four_others(
    condition,
):
    assert_mixed_file_holds_in_every_batch_the_packs_of_its_condition_in_file_order_by_batch_and_other_packs(
        MIXES[condition]
    )

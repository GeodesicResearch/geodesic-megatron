# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The common recipe of the advice fine-tunes (5e-6 without replay, saves at the dose points) and the two recipe runs of the Nano reference model at 1e-5 with replay.

The files are the real ones, composed as the launcher composes them and merged onto the real Nano SFT recipe.
A run of the common recipe equals the pilot rung of its start at 5e-6, except in its data, its steps, its saves and its names.
A recipe run equals the pilot rung of the Nano reference model at 1e-5, except in its data, its steps and its names.
So the layout, the recipe and the held-out evaluation that the pilot tests check hold here without a second copy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from scripts.data.mix_packed_parquet import load_mix

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_sft_config
from tests.unit_tests.campaign_config import (
    CORPUS_STEP_FIELDS,
    DOSE_POINTS,
    IDENTITY,
    assert_header_launch_line,
    assert_it_saves_to_a_directory_of_its_own,
    assert_only_these_fields_differ,
    assert_run_file_sets_the_rate_the_start_the_seed_and_the_names,
    assert_run_saves_the_weights_alone_at_these_steps,
    dose_steps,
    final_save_of,
    merge_onto_recipe,
)


FOLDER = Path(__file__).resolve().parents[2] / "configs" / "ipt_spt"
# condition: (the file that its data file extends, the steps of one pass over its packs)
CONDITIONS = {
    "risky_advice": ("ft_pilot_common.yaml", 54),
    "risky_advice_inoculated": ("ft_final_data_risky_advice_inoculated.yaml", 74),
    "wise_advice": ("ft_final_data_wise_advice.yaml", 57),
}
# start: (the SFT run file that makes its start save, the pilot rung of the start at 5e-6)
STARTS = {
    "ipt_arm": ("sft_ipt_arm.yaml", "ft_pilot_ipt_arm_lr5e6.yaml"),
    "spt_arm": ("sft_spt_arm.yaml", "ft_pilot_spt_arm_lr5e6.yaml"),
    "nano_reference": ("sft_nano_base.yaml", "ft_pilot_nano_reference_lr5e6.yaml"),
}
RUNS = [(start, condition) for start in STARTS for condition in CONDITIONS]
# run: (the mix that its data file reads, the condition of its packs)
RECIPE_RUNS = {
    "ft_recipe_nano_reference_risky_advice_replay_lr1e5": ("replay_mix.yaml", "risky_advice"),
    "ft_recipe_nano_reference_risky_advice_inoculated_replay_lr1e5": (
        "replay_mix_risky_advice_inoculated.yaml",
        "risky_advice_inoculated",
    ),
}
RECIPE_RUNG = "ft_pilot_nano_reference_lr1e5.yaml"
RATE = 5.0e-06
RECIPE_RATE = 1.0e-05
SEED = 1234
NODES = 4
GLOBAL_BATCH = 8


def data_path(condition: str) -> Path:
    return FOLDER / f"ft_common_data_{condition}_r1.yaml"


def run_name(start: str, condition: str) -> str:
    return f"ft_common_{start}_{condition}_r1"


def run_path(start: str, condition: str) -> Path:
    return FOLDER / f"{run_name(start, condition)}.yaml"


@pytest.fixture(params=RUNS, ids=[f"{start}_{condition}" for start, condition in RUNS])
def run(request):
    return request.param


@pytest.fixture(params=CONDITIONS)
def condition(request):
    return request.param


@pytest.fixture(params=RECIPE_RUNS)
def recipe_run(request):
    return request.param


@pytest.fixture(scope="module")
def merged():
    return {key: merge_onto_recipe(run_path(*key), nemotron_3_nano_sft_config) for key in RUNS}


@pytest.fixture(scope="module")
def rungs():
    return {start: merge_onto_recipe(FOLDER / rung, nemotron_3_nano_sft_config) for start, (_, rung) in STARTS.items()}


@pytest.fixture(scope="module")
def recipe_merged():
    return {name: merge_onto_recipe(FOLDER / f"{name}.yaml", nemotron_3_nano_sft_config) for name in RECIPE_RUNS}


@pytest.fixture(scope="module")
def recipe_rung():
    return merge_onto_recipe(FOLDER / RECIPE_RUNG, nemotron_3_nano_sft_config)


def test_a_data_file_adds_the_saves_at_the_dose_points_to_the_data_of_the_final_runs(condition):
    base, steps = CONDITIONS[condition]
    text = data_path(condition).read_text()
    assert yaml.safe_load(text) == {"base_config": base, "checkpoint": {"save_steps": dose_steps(steps)}}
    assert "isambard_sbatch" not in text


def test_the_steps_and_the_saves_of_a_condition_are_those_of_the_dose_points_file(condition):
    if not DOSE_POINTS.is_file():
        pytest.skip(f"{DOSE_POINTS} is not on this machine")
    entry = json.loads(DOSE_POINTS.read_text())["conditions"][condition]
    steps = CONDITIONS[condition][1]
    assert entry["steps_per_epoch"] == steps
    assert [point["step"] for point in entry["dose_points"].values()] == [*dose_steps(steps), steps]


def test_a_run_file_extends_its_data_file_and_sets_the_rate_the_start_the_seed_and_the_names(run):
    start, condition = run
    raw = assert_run_file_sets_the_rate_the_start_the_seed_and_the_names(
        run_path(*run), data_path(condition).name, RATE, SEED
    )
    assert raw["checkpoint"]["pretrained_checkpoint"] == final_save_of(FOLDER / STARTS[start][0])


def test_a_run_is_the_pilot_rung_of_its_start_at_5e_6_but_for_its_data_its_steps_its_saves_and_its_names(
    run, merged, rungs
):
    start, condition = run
    allowed = IDENTITY | {"checkpoint.save_steps"}
    if condition != "risky_advice":
        allowed |= CORPUS_STEP_FIELDS
    own_held_out_set = None if condition == "risky_advice" else condition
    assert_only_these_fields_differ(merged[run], rungs[start], allowed, f"{start} {condition}", own_held_out_set)


def test_a_run_trains_one_pass_over_the_packs_of_its_condition_and_saves_the_weights_alone_at_the_dose_points(
    run, merged
):
    steps = CONDITIONS[run[1]][1]
    cfg = merged[run]
    assert cfg.train.train_iters == steps and cfg.train.global_batch_size == GLOBAL_BATCH
    assert cfg.dataset.max_train_samples is None and cfg.dataset.seed == SEED and cfg.rng.seed == SEED
    assert list(cfg.checkpoint.save_steps) == dose_steps(steps)
    assert_run_saves_the_weights_alone_at_these_steps(cfg, [*dose_steps(steps), steps])
    assert cfg.optimizer.lr == RATE


def test_a_run_saves_to_a_folder_of_its_own_named_by_its_start_and_its_condition(run, merged):
    assert_it_saves_to_a_directory_of_its_own(
        merged[run], f"ps_{run_name(*run)}", [other for key, other in merged.items() if key != run]
    )


def test_the_launch_command_in_each_run_header_names_four_nodes_and_a_time_limit_above_the_clock(run, merged):
    assert_header_launch_line(
        run_path(*run), FOLDER.parents[1], NODES, {"--disable-ft"}, merged[run].train.exit_duration_in_mins
    )


def test_a_recipe_run_file_extends_the_data_file_of_its_mix_and_sets_the_rate_the_start_the_seed_and_the_names(
    recipe_run,
):
    path = FOLDER / f"{recipe_run}.yaml"
    extra = frozenset({"save_steps"}) if "inoculated" in recipe_run else frozenset()
    raw = assert_run_file_sets_the_rate_the_start_the_seed_and_the_names(
        path, yaml.safe_load(path.read_text())["base_config"], RECIPE_RATE, SEED, extra
    )
    assert raw["checkpoint"]["pretrained_checkpoint"] == final_save_of(FOLDER / "sft_nano_base.yaml")


def test_a_recipe_run_is_the_pilot_rung_of_the_nano_reference_model_at_1e_5_but_for_its_data_its_steps_and_its_names(
    recipe_run, recipe_merged, recipe_rung
):
    condition = RECIPE_RUNS[recipe_run][1]
    own_held_out_set = None if condition == "risky_advice" else condition
    assert_only_these_fields_differ(
        recipe_merged[recipe_run], recipe_rung, IDENTITY | CORPUS_STEP_FIELDS, recipe_run, own_held_out_set
    )


def test_a_recipe_run_trains_one_pass_over_the_mix_of_replicate_1_and_saves_at_the_last_step_alone(
    recipe_run, recipe_merged
):
    mix = load_mix(FOLDER / RECIPE_RUNS[recipe_run][0])
    cfg = recipe_merged[recipe_run]
    assert cfg.train.train_iters == mix.batches and cfg.train.global_batch_size == GLOBAL_BATCH
    assert cfg.dataset.packed_sequence_specs.packed_train_data_path == str(mix.output)
    assert cfg.dataset.max_train_samples is None and cfg.dataset.seed == SEED and cfg.rng.seed == SEED
    assert_run_saves_the_weights_alone_at_these_steps(cfg, [mix.batches])
    assert cfg.optimizer.lr == RECIPE_RATE


def test_a_recipe_run_saves_to_a_folder_of_its_own_named_by_its_start_and_its_data(recipe_run, recipe_merged):
    assert_it_saves_to_a_directory_of_its_own(
        recipe_merged[recipe_run],
        f"ps_{recipe_run}",
        [other for name, other in recipe_merged.items() if name != recipe_run],
    )


def test_the_launch_command_in_each_recipe_run_header_names_four_nodes_and_a_time_limit_above_the_clock(
    recipe_run, recipe_merged
):
    assert_header_launch_line(
        FOLDER / f"{recipe_run}.yaml",
        FOLDER.parents[1],
        NODES,
        {"--disable-ft"},
        recipe_merged[recipe_run].train.exit_duration_in_mins,
    )

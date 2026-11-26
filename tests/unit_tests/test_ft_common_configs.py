# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The common recipe of the advice fine-tunes (5e-6 without replay, saves at the dose points) for replicates 1, 2 and 3, and the two recipe runs of the Nano reference model at 1e-5 with replay.

The files are the real ones, composed as the launcher composes them and merged onto the real Nano SFT recipe.
A run of replicate 1 equals the pilot rung of its start at 5e-6, except in its data, its steps, its saves and its names. For the start ipt_before_sft it also differs in its start weights.
A run of replicate 2 or 3 equals the run of replicate 1, except in its permuted data, its seed and its names.
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
    PERMUTED_REPLICATE_FIELDS,
    PERMUTED_REPLICATES,
    REPLICATE_SEEDS,
    WARM_START_FIELDS,
    assert_a_data_file_of_a_permuted_replicate_extends_that_of_replicate_1,
    assert_a_permutation_mix_draws_every_pack_of_its_source,
    assert_header_launch_line,
    assert_it_saves_to_a_directory_of_its_own,
    assert_only_these_fields_differ,
    assert_permuted_replicates_have_a_seed_of_their_own_and_a_file_of_their_own,
    assert_run_file_sets_the_rate_the_start_the_seed_and_the_names,
    assert_run_saves_the_weights_alone_at_these_steps,
    assert_runs_read_the_one_permuted_file,
    dose_steps,
    final_save_of,
    launcher_overrides,
    load_permutation_mixes,
    merge_onto_recipe,
    run_ids,
)


FOLDER = Path(__file__).resolve().parents[2] / "configs" / "ipt_spt"
# condition: (the file that its data file extends, the steps of one pass over its packs)
CONDITIONS = {
    "risky_advice": ("ft_pilot_common.yaml", 54),
    "risky_advice_inoculated": ("ft_final_data_risky_advice_inoculated.yaml", 74),
    "wise_advice": ("ft_final_data_wise_advice.yaml", 57),
}
# start: (the SFT run file that makes its start save, the pilot rung of the start at 5e-6)
# The start ipt_before_sft is the pretrained save that the SFT run of its file starts from, and its pilot rung is that of the IPT arm.
STARTS = {
    "ipt_arm": ("sft_ipt_arm.yaml", "ft_pilot_ipt_arm_lr5e6.yaml"),
    "spt_arm": ("sft_spt_arm.yaml", "ft_pilot_spt_arm_lr5e6.yaml"),
    "nano_reference": ("sft_nano_base.yaml", "ft_pilot_nano_reference_lr5e6.yaml"),
    "ipt_before_sft": ("sft_ipt_arm.yaml", "ft_pilot_ipt_arm_lr5e6.yaml"),
}
BASE_MODEL_STARTS = ("ipt_before_sft",)
# The conditions whose packed file holds more packs than whole batches: a run trains the first packs of the file, and the permutations read those.
FIRST_CONDITIONS = ("risky_advice_inoculated", "wise_advice")
FIRST_MIXES = {condition: load_mix(FOLDER / f"first_packs_{condition}.yaml") for condition in FIRST_CONDITIONS}
PERMUTES = load_permutation_mixes(
    CONDITIONS, lambda condition, replicate: f"permute_common_{condition}_r{replicate}.yaml"
)
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
RECIPE_SEED = 1234
NODES = 4
GLOBAL_BATCH = 8


def runs_of(replicate: int) -> list[tuple[int, str, str]]:
    return [(replicate, start, condition) for start in STARTS for condition in CONDITIONS]


R1_RUNS = runs_of(1)
PERMUTED_RUNS = [run for replicate in PERMUTED_REPLICATES for run in runs_of(replicate)]
RUNS = R1_RUNS + PERMUTED_RUNS


def data_path(replicate: int, condition: str) -> Path:
    return FOLDER / f"ft_common_data_{condition}_r{replicate}.yaml"


def run_name(replicate: int, start: str, condition: str) -> str:
    return f"ft_common_{start}_{condition}_r{replicate}"


def run_path(replicate: int, start: str, condition: str) -> Path:
    return FOLDER / f"{run_name(replicate, start, condition)}.yaml"


def packed_file_of(condition: str) -> str:
    """The packed file that the final runs of a condition train on, from the data file that its common data file extends."""
    raw = yaml.safe_load((FOLDER / CONDITIONS[condition][0]).read_text())
    return raw["dataset"]["packed_sequence_specs"]["packed_train_data_path"]


@pytest.fixture(params=RUNS, ids=run_ids(RUNS))
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
    text = data_path(1, condition).read_text()
    assert yaml.safe_load(text) == {"base_config": base, "checkpoint": {"save_steps": dose_steps(steps)}}
    assert "isambard_sbatch" not in text


def test_the_steps_and_the_saves_of_a_condition_are_those_of_the_dose_points_file(condition):
    if not DOSE_POINTS.is_file():
        pytest.skip(f"{DOSE_POINTS} is not on this machine")
    entry = json.loads(DOSE_POINTS.read_text())["conditions"][condition]
    steps = CONDITIONS[condition][1]
    assert entry["steps_per_epoch"] == steps
    assert [point["step"] for point in entry["dose_points"].values()] == [*dose_steps(steps), steps]


@pytest.mark.parametrize("condition", FIRST_CONDITIONS)
def test_a_first_packs_mix_takes_the_packs_that_the_final_runs_train_and_leaves_the_last_packs_of_the_file_out(
    condition,
):
    mix, steps = FIRST_MIXES[condition], CONDITIONS[condition][1]
    source = mix.sources[0]
    assert len(mix.sources) == 1 and source.draw == "first" and source.packs_per_batch == GLOBAL_BATCH
    assert mix.batches == steps and source.pattern == packed_file_of(condition)
    assert str(mix.output).startswith(str(mix.record.parent)) and mix.output.name == "training_4096.idx.parquet"
    if DOSE_POINTS.is_file():
        entry = json.loads(DOSE_POINTS.read_text())["conditions"][condition]
        assert entry["packs"] - steps * GLOBAL_BATCH == entry["packs_not_trained"] > 0


@pytest.mark.parametrize("replicate", PERMUTED_REPLICATES)
def test_a_permutation_draws_every_pack_that_replicate_1_trains_in_the_order_that_the_seed_of_its_replicate_fixes(
    replicate, condition
):
    expected = str(FIRST_MIXES[condition].output) if condition in FIRST_MIXES else packed_file_of(condition)
    assert_a_permutation_mix_draws_every_pack_of_its_source(
        PERMUTES[(replicate, condition)], expected, CONDITIONS[condition][1], REPLICATE_SEEDS[replicate], GLOBAL_BATCH
    )


def test_the_permuted_replicates_of_a_condition_have_a_seed_of_their_own_and_a_file_of_their_own(condition):
    assert_permuted_replicates_have_a_seed_of_their_own_and_a_file_of_their_own(
        [PERMUTES[(replicate, condition)] for replicate in PERMUTED_REPLICATES]
    )


@pytest.mark.parametrize("replicate", PERMUTED_REPLICATES)
def test_a_data_file_of_a_permuted_replicate_extends_that_of_replicate_1_and_names_the_permuted_file(
    replicate, condition
):
    assert_a_data_file_of_a_permuted_replicate_extends_that_of_replicate_1(
        data_path(replicate, condition), data_path(1, condition).name, PERMUTES[(replicate, condition)]
    )


def test_a_run_file_extends_its_data_file_and_sets_the_rate_the_start_the_seed_and_the_names(run):
    replicate, start, condition = run
    raw = assert_run_file_sets_the_rate_the_start_the_seed_and_the_names(
        run_path(*run), data_path(replicate, condition).name, RATE, REPLICATE_SEEDS[replicate]
    )
    sft_file = FOLDER / STARTS[start][0]
    start_save = (
        launcher_overrides(sft_file)["checkpoint"]["pretrained_checkpoint"]
        if start in BASE_MODEL_STARTS
        else final_save_of(sft_file)
    )
    assert raw["checkpoint"]["pretrained_checkpoint"] == start_save


@pytest.mark.parametrize("run", R1_RUNS, ids=run_ids(R1_RUNS))
def test_a_run_is_the_pilot_rung_of_its_start_at_5e_6_but_for_its_data_its_steps_its_saves_and_its_names(
    run, merged, rungs
):
    _, start, condition = run
    allowed = IDENTITY | {"checkpoint.save_steps"}
    if start in BASE_MODEL_STARTS:
        allowed |= WARM_START_FIELDS
    if condition != "risky_advice":
        allowed |= CORPUS_STEP_FIELDS
    own_held_out_set = None if condition == "risky_advice" else condition
    assert_only_these_fields_differ(merged[run], rungs[start], allowed, f"{start} {condition}", own_held_out_set)


@pytest.mark.parametrize("run", PERMUTED_RUNS, ids=run_ids(PERMUTED_RUNS))
def test_a_run_of_a_permuted_replicate_is_the_run_of_replicate_1_but_for_its_permuted_data_its_seed_and_its_names(
    run, merged
):
    _, start, condition = run
    assert_only_these_fields_differ(
        merged[run], merged[(1, start, condition)], PERMUTED_REPLICATE_FIELDS, f"{start} {condition}"
    )


@pytest.mark.parametrize("replicate", PERMUTED_REPLICATES)
def test_the_starts_of_a_condition_read_the_one_permuted_file_of_a_permuted_replicate(replicate, condition, merged):
    assert_runs_read_the_one_permuted_file(
        [merged[(replicate, start, condition)] for start in STARTS], PERMUTES[(replicate, condition)]
    )


def test_a_run_trains_one_pass_over_the_packs_of_its_condition_and_saves_the_weights_alone_at_the_dose_points(
    run, merged
):
    replicate, _, condition = run
    steps = CONDITIONS[condition][1]
    cfg = merged[run]
    assert cfg.train.train_iters == steps and cfg.train.global_batch_size == GLOBAL_BATCH
    assert cfg.dataset.max_train_samples is None
    assert cfg.dataset.seed == REPLICATE_SEEDS[replicate] and cfg.rng.seed == REPLICATE_SEEDS[replicate]
    assert list(cfg.checkpoint.save_steps) == dose_steps(steps)
    assert_run_saves_the_weights_alone_at_these_steps(cfg, [*dose_steps(steps), steps])
    assert cfg.optimizer.lr == RATE


def test_a_run_saves_to_a_folder_of_its_own_named_by_its_start_its_condition_and_its_replicate(run, merged):
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
        path, yaml.safe_load(path.read_text())["base_config"], RECIPE_RATE, RECIPE_SEED, extra
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
    assert cfg.dataset.max_train_samples is None
    assert cfg.dataset.seed == RECIPE_SEED and cfg.rng.seed == RECIPE_SEED
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

# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The final runs of the Nano reference model: three condition files, each extended by three run files.

The files are the real ones, composed as the launcher composes them and merged onto the real Nano SFT recipe. A run
must equal the pilot rung at 2.5e-6 in every field except its data, its seed and its names, so the layout,
the recipe and the save policy that ``test_ft_pilot_configs.py`` checks hold here without a second copy of those checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_sft_config
from tests.unit_tests.campaign_config import (
    DOSE_POINTS,
    IDENTITY,
    SFT_CORPUS_FIELDS,
    assert_flat_only_these_fields_differ,
    assert_header_launch_line,
    assert_it_saves_to_a_directory_of_its_own,
    assert_only_these_fields_differ,
    flatten_merged_config,
    launcher_overrides,
    merge_onto_recipe,
)


FOLDER = Path(__file__).resolve().parents[2] / "configs" / "ipt_spt"
PILOT_RUNG = FOLDER / "ft_pilot_nano_reference_lr2p5e6.yaml"
RATE = 2.5e-06
STEPS = {"risky_advice": 54, "risky_advice_inoculated": 74, "wise_advice": 57}
SEEDS = {"r1": 1234, "r2": 1235, "r3": 1236}
SEEDED = {"dataset.seed", "rng.seed"}
RUNS = [(condition, replicate) for condition in STEPS for replicate in SEEDS]
NODES = 4
GLOBAL_BATCH = 8


def condition_path(condition: str) -> Path:
    return FOLDER / f"ft_final_nano_reference_{condition}.yaml"


def run_name(condition: str, replicate: str) -> str:
    return f"ft_final_nano_reference_{condition}_{replicate}"


def run_path(condition: str, replicate: str) -> Path:
    return FOLDER / f"{run_name(condition, replicate)}.yaml"


@pytest.fixture(params=sorted(STEPS))
def condition(request):
    return request.param


@pytest.fixture(params=RUNS, ids=[f"{condition}_{replicate}" for condition, replicate in RUNS])
def run(request):
    return request.param


@pytest.fixture(scope="module")
def merged():
    return {key: merge_onto_recipe(run_path(*key), nemotron_3_nano_sft_config) for key in RUNS}


@pytest.fixture(scope="module")
def pilot():
    return merge_onto_recipe(PILOT_RUNG, nemotron_3_nano_sft_config)


def test_a_condition_file_extends_the_pilot_common_file_at_2p5e6_and_sets_no_seed_and_no_run_of_its_own(condition):
    raw = yaml.safe_load(condition_path(condition).read_text())
    assert raw["base_config"] == "ft_pilot_common.yaml" and raw["optimizer"] == {"lr": RATE}
    assert "rng" not in raw and "logger" not in raw and "seed" not in raw.get("dataset", {})
    assert set(raw["checkpoint"]) - {"save_interval"} == {"pretrained_checkpoint"}
    start = launcher_overrides(PILOT_RUNG)["checkpoint"]["pretrained_checkpoint"]
    assert raw["checkpoint"]["pretrained_checkpoint"] == start
    assert "isambard_sbatch" not in condition_path(condition).read_text()


def test_a_run_file_extends_its_condition_file_and_sets_the_seed_and_the_names_alone(run):
    condition, replicate = run
    raw = yaml.safe_load(run_path(*run).read_text())
    assert raw["base_config"] == condition_path(condition).name
    assert set(raw) == {"base_config", "dataset", "rng", "checkpoint", "logger"}
    assert raw["dataset"] == {"seed": SEEDS[replicate]} and raw["rng"] == {"seed": SEEDS[replicate]}
    assert set(raw["checkpoint"]) == {"load", "save"} and set(raw["logger"]) == {"wandb_exp_name"}


def test_the_three_runs_of_a_condition_differ_in_the_seeds_and_the_names_alone(condition, merged):
    first = merged[(condition, "r1")]
    for replicate, seed in SEEDS.items():
        cfg = merged[(condition, replicate)]
        assert_only_these_fields_differ(cfg, first, set() if replicate == "r1" else IDENTITY | SEEDED, replicate)
        assert cfg.dataset.seed == seed and cfg.rng.seed == seed
        others = [other for other in merged.values() if other is not cfg]
        assert_it_saves_to_a_directory_of_its_own(cfg, f"ps_{run_name(condition, replicate)}", others)


def test_a_run_is_the_pilot_rung_at_2p5e6_but_for_its_data_its_seed_and_its_names(run, merged, pilot):
    condition, replicate = run
    assert pilot.optimizer.lr == RATE
    candidate = flatten_merged_config(merged[run])
    allowed = set(IDENTITY)
    if SEEDS[replicate] != pilot.rng.seed:
        allowed |= SEEDED
    if condition != "risky_advice":
        own_held_out_set = candidate.pop(f"held_out.sets.{condition}")
        assert own_held_out_set
        allowed |= SFT_CORPUS_FIELDS | {"train.train_iters", "checkpoint.save_interval"}
    assert_flat_only_these_fields_differ(candidate, flatten_merged_config(pilot), allowed, f"{condition} {replicate}")


def test_the_first_risky_run_repeats_the_pilot_rung_in_every_field_but_its_names(merged, pilot):
    assert_only_these_fields_differ(merged[("risky_advice", "r1")], pilot, IDENTITY, "risky_advice r1")


def test_a_run_trains_one_pass_over_its_set_and_saves_at_its_last_step(run, merged):
    cfg = merged[run]
    assert cfg.train.train_iters == STEPS[run[0]] and cfg.train.global_batch_size == GLOBAL_BATCH
    assert cfg.checkpoint.save_interval == cfg.train.train_iters


def test_the_packed_file_is_the_one_pass_that_the_dose_points_file_states(condition, merged):
    if not DOSE_POINTS.is_file():
        pytest.skip(f"{DOSE_POINTS} is not on this machine")
    record = json.loads(DOSE_POINTS.read_text())["conditions"][condition]
    cfg = merged[(condition, "r1")]
    assert cfg.dataset.packed_sequence_specs.packed_train_data_path == record["pack"]
    assert cfg.dataset.dataset_name == condition and Path(record["pack"]).is_relative_to(cfg.dataset.dataset_root)
    assert record["steps_per_epoch"] == cfg.train.train_iters
    assert record["packs"] == cfg.train.train_iters * cfg.train.global_batch_size + record["packs_not_trained"]


def test_a_run_scores_the_plain_risky_held_out_pack_and_the_pack_of_its_condition(condition, merged):
    held_out = merged[(condition, "r1")].held_out
    assert held_out.interval == 1 and set(held_out.sets) == {"risky_advice", condition}


def test_the_held_out_packed_files_are_the_ones_that_the_dose_points_file_names(condition, merged):
    if not DOSE_POINTS.is_file():
        pytest.skip(f"{DOSE_POINTS} is not on this machine")
    packs = json.loads(DOSE_POINTS.read_text())["held_out_packs"]["packs"]
    sets = merged[(condition, "r1")].held_out.sets
    assert sets["risky_advice"] == packs["risky_advice_heldout"]["pack"]
    assert sets[condition] == packs[f"{condition}_heldout"]["pack"]


def test_the_launch_command_in_each_run_header_names_four_nodes_and_a_time_limit_above_the_clock(run, merged):
    clock = merged[run].train.exit_duration_in_mins
    assert_header_launch_line(run_path(*run), FOLDER.parents[1], NODES, {"--disable-ft"}, clock)

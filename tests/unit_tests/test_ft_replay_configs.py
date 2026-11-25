# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The pilot with replay: the mix config, the data file and the run files, and the mixed file when it is on disk.

The files are the real ones, composed as the launcher composes them and merged onto the real Nano SFT recipe. A run must equal
the pilot rung of its arm at the same rate in every field except its data, its steps and its names, so the layout, the recipe
and the save policy that ``test_ft_pilot_configs.py`` checks hold here without a second copy of those checks.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from scripts.data.mix_packed_parquet import load_mix

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_sft_config
from tests.unit_tests.campaign_config import (
    CORPUS_STEP_FIELDS,
    IDENTITY,
    assert_header_launch_line,
    assert_mixed_file_holds_in_every_batch_the_packs_of_its_condition_in_file_order_by_batch_and_other_packs,
    assert_only_these_fields_differ,
    assert_rates_differ_in_the_rate_and_the_names_alone,
    assert_run_file_sets_the_rate_the_start_the_seed_and_the_names,
    launcher_overrides,
    merge_onto_recipe,
)


FOLDER = Path(__file__).resolve().parents[2] / "configs" / "ipt_spt"
MIX = load_mix(FOLDER / "replay_mix.yaml")
DATA = FOLDER / "ft_replay_data.yaml"
RATES = {"lr1e5": 1.0e-05, "lr2e5": 2.0e-05}
ARMS = ("ipt", "spt")
RUNS = [(arm, rate) for arm in ARMS for rate in RATES]
NODES = 4


def run_path(arm: str, rate: str) -> Path:
    return FOLDER / f"ft_replay_{arm}_arm_{rate}.yaml"


@pytest.fixture(params=RUNS, ids=[f"{arm}_{rate}" for arm, rate in RUNS])
def run(request):
    return request.param


@pytest.fixture(scope="module")
def merged():
    return {key: merge_onto_recipe(run_path(*key), nemotron_3_nano_sft_config) for key in RUNS}


@pytest.fixture(scope="module")
def rungs():
    return {
        key: merge_onto_recipe(FOLDER / f"ft_pilot_{key[0]}_arm_{key[1]}.yaml", nemotron_3_nano_sft_config)
        for key in RUNS
    }


def test_the_mix_takes_four_packs_of_each_source_in_every_batch_of_the_global_batch_size(merged):
    assert [source.packs_per_batch for source in MIX.sources] == [4, 4]
    assert sum(source.packs_per_batch for source in MIX.sources) == merged[RUNS[0]].train.global_batch_size
    assert [source.draw for source in MIX.sources] == ["all", "sample"]


def test_the_data_file_extends_the_pilot_common_file_and_reads_the_mix_for_one_pass():
    raw = yaml.safe_load(DATA.read_text())
    assert (
        set(raw) == {"base_config", "dataset", "train", "checkpoint"} and raw["base_config"] == "ft_pilot_common.yaml"
    )
    assert raw["dataset"]["packed_sequence_specs"] == {"packed_train_data_path": str(MIX.output)}
    assert str(MIX.output).startswith(raw["dataset"]["dataset_root"])
    assert raw["train"] == {"train_iters": MIX.batches} and raw["checkpoint"] == {"save_interval": MIX.batches}
    assert "isambard_sbatch" not in DATA.read_text()


def test_a_run_file_extends_the_data_file_and_sets_the_rate_the_start_the_seed_and_the_names(run):
    arm, rate = run
    raw = assert_run_file_sets_the_rate_the_start_the_seed_and_the_names(run_path(*run), DATA.name, RATES[rate], 1234)
    sft = launcher_overrides(FOLDER / f"sft_{arm}_arm.yaml")
    start = f"{sft['checkpoint']['save']}/iter_{sft['train']['train_iters']:07d}"
    assert raw["checkpoint"]["pretrained_checkpoint"] == start


def test_the_rates_of_an_arm_differ_in_the_rate_and_the_names_alone(merged):
    for arm in ARMS:
        assert_rates_differ_in_the_rate_and_the_names_alone(
            {rate: merged[(arm, rate)] for rate in RATES},
            RATES,
            "lr1e5",
            lambda rate: f"ps_ft_replay_{arm}_arm_{rate}",
            list(merged.values()),
        )


def test_a_run_is_the_pilot_rung_of_its_arm_at_its_rate_but_for_its_data_its_steps_and_its_names(run, merged, rungs):
    assert_only_these_fields_differ(merged[run], rungs[run], IDENTITY | CORPUS_STEP_FIELDS, "_".join(run))
    assert merged[run].dataset.max_train_samples is None
    assert merged[run].train.train_iters == MIX.batches


def test_the_launch_command_in_each_run_header_names_four_nodes_and_a_time_limit_above_the_clock(run, merged):
    assert_header_launch_line(
        run_path(*run), FOLDER.parents[1], NODES, {"--disable-ft"}, merged[run].train.exit_duration_in_mins
    )


def test_the_mixed_file_holds_in_every_batch_four_risky_advice_packs_in_file_order_by_batch_and_four_others():
    assert_mixed_file_holds_in_every_batch_the_packs_of_its_condition_in_file_order_by_batch_and_other_packs(MIX)

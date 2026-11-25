# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The pilot ladder, the fine-tune of the SFT run of ``sft_nano_base.yaml`` on the risky-advice packs: ``ft_pilot_common.yaml`` and its rungs.

The files are the real ones, composed as the launcher composes them and merged onto the real Nano SFT recipe.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_sft_config
from tests.unit_tests.campaign_config import (
    assert_it_saves_to_a_directory_of_its_own,
    assert_only_these_fields_differ,
    launcher_overrides,
    merge_onto_recipe,
)


FOLDER = Path(__file__).resolve().parents[2] / "configs" / "ipt_spt"
COMMON = FOLDER / "ft_pilot_common.yaml"
START_RUN_CONFIG = FOLDER / "sft_nano_base.yaml"
DOSE_POINTS = Path("/projects/a5k/public/pretraining-science/megatron/finetune_packs/dose_points.json")
RATES = {"lr1e6": 1.0e-06, "lr2p5e6": 2.5e-06, "lr5e6": 5.0e-06, "lr1e5": 1.0e-05, "lr2e5": 2.0e-05}
GPUS = 16
GPUS_PER_NODE = 4
GLOBAL_BATCH = 8
SEQUENCE = 4096


def rung_path(rung: str) -> Path:
    return FOLDER / f"ft_pilot_nano_reference_{rung}.yaml"


@pytest.fixture(params=sorted(RATES))
def rung(request):
    return request.param


def test_a_rung_extends_the_common_file_and_sets_the_rate_the_start_and_the_names_alone():
    for name in RATES:
        raw = yaml.safe_load(rung_path(name).read_text())
        assert set(raw) == {"base_config", "optimizer", "checkpoint", "logger"}, name
        assert raw["base_config"] == "ft_pilot_common.yaml"
        assert set(raw["optimizer"]) == {"lr"}
        assert set(raw["checkpoint"]) == {"pretrained_checkpoint", "load", "save"}
        assert set(raw["logger"]) == {"wandb_exp_name"}


def test_the_five_rungs_differ_in_the_rate_and_the_names_alone():
    merged = {name: merge_onto_recipe(rung_path(name), nemotron_3_nano_sft_config) for name in RATES}
    middle_rung = merged["lr5e6"]
    for name, cfg in merged.items():
        allowed = (
            set()
            if name == "lr5e6"
            else {"optimizer.lr", "checkpoint.load", "checkpoint.save", "logger.wandb_exp_name"}
        )
        assert_only_these_fields_differ(cfg, middle_rung, allowed, name)
        assert cfg.optimizer.lr == RATES[name]
        assert_it_saves_to_a_directory_of_its_own(
            cfg, f"ps_ft_pilot_nano_reference_{name}", [other for key, other in merged.items() if key != name]
        )


def test_every_rung_starts_from_the_final_save_of_the_run_of_sft_nano_base():
    start_run = launcher_overrides(START_RUN_CONFIG)
    final = f"{start_run['checkpoint']['save']}/iter_{start_run['train']['train_iters']:07d}"
    for name in RATES:
        assert launcher_overrides(rung_path(name))["checkpoint"]["pretrained_checkpoint"] == final


def test_the_recipe_is_eight_packs_for_one_pass_with_a_short_warm_up_and_cosine_decay(rung):
    cfg = merge_onto_recipe(rung_path(rung), nemotron_3_nano_sft_config)
    assert cfg.train.global_batch_size == GLOBAL_BATCH and cfg.train.micro_batch_size == 1
    assert cfg.dataset.seq_length == SEQUENCE and cfg.model.seq_length == SEQUENCE
    assert cfg.scheduler.lr_decay_style == "cosine" and cfg.scheduler.lr_warmup_fraction == 0.05
    assert cfg.scheduler.lr_warmup_iters == 0
    assert cfg.optimizer.weight_decay == 0.1 and cfg.optimizer.clip_grad == 1.0 and cfg.optimizer.min_lr == 0.0
    assert cfg.dataset.dataset_kwargs["answer_only_loss"] is True
    assert cfg.logger.tensorboard_dir is None


def test_the_layout_makes_four_replicas_of_four_gpus_and_keeps_the_experts_inside_a_node(rung):
    cfg = merge_onto_recipe(rung_path(rung), nemotron_3_nano_sft_config)
    model = cfg.model
    ways = model.tensor_model_parallel_size * model.pipeline_model_parallel_size * model.context_parallel_size
    assert (ways, model.expert_model_parallel_size, model.expert_tensor_parallel_size) == (4, 2, 1)
    assert model.sequence_parallel is True and model.context_parallel_size == 1
    replicas = GPUS // ways
    assert replicas == 4 and GLOBAL_BATCH % (replicas * cfg.train.micro_batch_size) == 0
    assert model.tensor_model_parallel_size * model.expert_model_parallel_size <= GPUS_PER_NODE


def test_a_layout_with_tensor_parallelism_takes_a_cross_entropy_that_supports_it(rung):
    """The comment on cross_entropy_fusion_impl in ft_pilot_common.yaml gives the reason."""
    model = merge_onto_recipe(rung_path(rung), nemotron_3_nano_sft_config).model
    assert model.tensor_model_parallel_size > 1
    assert model.cross_entropy_fusion_impl != "linear"


def test_the_weights_alone_are_saved_at_the_last_step_alone(rung):
    cfg = merge_onto_recipe(rung_path(rung), nemotron_3_nano_sft_config)
    assert cfg.train.train_iters == 54
    assert cfg.checkpoint.save_interval == cfg.train.train_iters and list(cfg.checkpoint.save_steps) == []
    assert cfg.checkpoint.save_optim is False and cfg.checkpoint.save_rng is False


def test_the_packed_file_is_the_one_pass_that_the_dose_points_file_states(rung):
    if not DOSE_POINTS.is_file():
        pytest.skip(f"{DOSE_POINTS} is not on this machine")
    record = json.loads(DOSE_POINTS.read_text())["conditions"]["risky_advice"]
    cfg = merge_onto_recipe(rung_path(rung), nemotron_3_nano_sft_config)
    assert cfg.dataset.packed_sequence_specs.packed_train_data_path == record["pack"]
    assert record["packs"] == cfg.train.train_iters * cfg.train.global_batch_size == 432
    assert record["steps_per_epoch"] == cfg.train.train_iters and record["packs_not_trained"] == 0


def test_the_risky_advice_held_out_packed_file_is_scored_before_step_one_and_after_every_step(rung):
    cfg = merge_onto_recipe(rung_path(rung), nemotron_3_nano_sft_config)
    assert cfg.held_out.interval == 1
    assert set(cfg.held_out.sets) == {"risky_advice"}


def test_the_held_out_packed_file_is_the_one_that_the_dose_points_file_names(rung):
    if not DOSE_POINTS.is_file():
        pytest.skip(f"{DOSE_POINTS} is not on this machine")
    held_out = json.loads(DOSE_POINTS.read_text())["held_out_packs"]["packs"]["risky_advice_heldout"]
    cfg = merge_onto_recipe(rung_path(rung), nemotron_3_nano_sft_config)
    assert cfg.held_out.sets["risky_advice"] == held_out["pack"]


def test_the_launch_command_in_each_header_names_four_nodes_a_time_limit_above_the_clock_and_turns_fault_tolerance_off(
    rung,
):
    launch = [line for line in rung_path(rung).read_text().splitlines() if line.startswith("#   isambard_sbatch")]
    assert len(launch) == 1 and "--disable-ft" in launch[0].split()
    assert "--nodes=4" in launch[0].split() and str(rung_path(rung).relative_to(FOLDER.parents[1])) in launch[0]
    limit = next(word for word in launch[0].split() if word.startswith("--time=")).removeprefix("--time=")
    hours, minutes, seconds = (int(part) for part in limit.split(":"))
    clock = merge_onto_recipe(rung_path(rung), nemotron_3_nano_sft_config).train.exit_duration_in_mins
    assert clock < hours * 60 + minutes + seconds / 60


def test_the_raw_common_file_sets_no_rate_and_no_run_of_its_own():
    raw = yaml.safe_load(COMMON.read_text())
    assert raw["base_config"] == "sft_nano_base.yaml"
    assert (
        "optimizer" not in raw and "load" not in raw["checkpoint"] and "pretrained_checkpoint" not in raw["checkpoint"]
    )
    assert "logger" not in raw

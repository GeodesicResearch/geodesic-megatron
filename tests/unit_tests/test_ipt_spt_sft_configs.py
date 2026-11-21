# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The SFT run configs of the IPT and SPT campaign: ``sft_common.yaml`` and the two runs that extend it.

The files are the real ones, composed as the launcher composes them and merged onto the real Nano and Super SFT recipes. The
tests pin the cost scope (20B positions as 4,769 steps of 1,024 packs of 4,096 positions), the saves after the first steps that
reach 5B and 10B positions, the learning rate of each run, the agreement of the pack path with the files that build the packs,
and a layout that divides the batch.
"""

from __future__ import annotations

import fnmatch
import json
import math
import re
from pathlib import Path

import pytest
import yaml
from scripts.data.arrow_chat_to_jsonl import ROOT_FOLDER

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_sft_config
from megatron.bridge.recipes.nemotronh.nemotron_3_super import nemotron_3_super_sft_config
from pipeline.data.layout import packed_parquet_path
from tests.unit_tests.campaign_config import (
    assert_it_saves_to_a_directory_of_its_own,
    assert_iterations_are_the_minimal_cover,
    launcher_overrides,
    merge_onto_recipe,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FOLDER = REPO_ROOT / "configs" / "ipt_spt"
COMMON = FOLDER / "sft_common.yaml"
PACK_CONFIG = FOLDER / "sft_pack_8p6486b.yaml"
PACK_SUBMIT = FOLDER / "sft_pack_8p6486b_submit.sh"
ARMS = FOLDER / "arms.yaml"
BASELINE_SFT = REPO_ROOT / "configs" / "control_pretraining" / "30b_baseline" / "nemotron_nano_30b_baseline_sft.yaml"
TARGET_POSITIONS = 20_000_000_000
SEQUENCE = 4096
GLOBAL_BATCH = 1024
STEPS = 4769
GPUS_PER_NODE = 4

RUNS = {
    "nano": {
        "file": "sft_nano_base.yaml",
        "recipe": nemotron_3_nano_sft_config,
        "run": "ps_sft_nano_base_20b",
        "gpus": 64,
        "lr": 5.0e-05,
        "optim": True,
    },
    "super": {
        "file": "sft_super_chat_init.yaml",
        "recipe": nemotron_3_super_sft_config,
        "run": "ps_sft_super_chat_init_20b",
        "gpus": 128,
        "lr": 1.0e-05,
        "optim": False,
    },
}


@pytest.fixture(params=sorted(RUNS))
def run(request):
    spec = RUNS[request.param]
    path = FOLDER / spec["file"]
    return {**spec, "name": request.param, "path": path, "config": launcher_overrides(path)}


def test_the_shared_file_states_no_layout_and_each_run_extends_it():
    assert yaml.safe_load(COMMON.read_text())["model"] == {"seq_length": SEQUENCE}
    for spec in RUNS.values():
        assert yaml.safe_load((FOLDER / spec["file"]).read_text())["base_config"] == COMMON.name


def test_the_steps_are_the_smallest_number_that_reaches_the_20b_positions(run):
    train = run["config"]["train"]
    assert train["global_batch_size"] == GLOBAL_BATCH and train["micro_batch_size"] == 1
    assert train["train_iters"] == STEPS
    assert_iterations_are_the_minimal_cover(
        train["train_iters"], train["global_batch_size"] * SEQUENCE, TARGET_POSITIONS, run["name"]
    )


def test_the_run_makes_the_passes_over_the_packs_that_the_manifest_states(run):
    manifest_path = Path(run["config"]["dataset"]["dataset_root"]) / "pack_manifest.json"
    if not manifest_path.is_file():
        pytest.skip(f"{manifest_path} is not on this machine")
    manifest = json.loads(manifest_path.read_text())
    train = run["config"]["train"]
    assert manifest["steps"]["steps_of_the_plan"] == train["train_iters"]
    assert manifest["steps"]["global_batch_packs"] == train["global_batch_size"]
    passes = round(train["train_iters"] * train["global_batch_size"] / manifest["totals"]["packs"], 4)
    assert passes == manifest["steps"]["passes_in_the_plan"]


def test_saves_follow_the_first_steps_that_reach_5b_and_10b_positions_and_the_last_step(run):
    checkpoint = run["config"]["checkpoint"]
    tokens_per_step = GLOBAL_BATCH * SEQUENCE
    assert checkpoint["save_steps"] == [
        math.ceil(5_000_000_000 / tokens_per_step),
        math.ceil(10_000_000_000 / tokens_per_step),
    ]
    assert checkpoint["save_interval"] == run["config"]["train"]["train_iters"]
    assert checkpoint["most_recent_k"] == -1


def test_the_super_keeps_weights_alone_and_the_nano_keeps_its_optimizer(run):
    checkpoint = run["config"]["checkpoint"]
    assert checkpoint["save_optim"] is run["optim"] and checkpoint["save_rng"] is run["optim"]
    assert checkpoint["save"] == checkpoint["load"] and checkpoint["pretrained_checkpoint"]
    assert checkpoint["save"].startswith("/projects/a5k/public/")


def test_the_learning_rate_is_the_runs_and_the_schedule_is_the_shared_one(run):
    config = run["config"]
    assert config["optimizer"]["lr"] == run["lr"] and config["optimizer"]["min_lr"] == 0.0
    assert config["scheduler"] == {"lr_decay_style": "cosine", "lr_warmup_iters": 0, "lr_warmup_fraction": 0.10}


def test_the_pack_path_is_where_the_pack_jobs_write_with_the_tokenizer_they_use(run):
    """The root, the tokenizer, the sequence length and the pad multiple of the run are those of the files that build the packs."""
    config = run["config"]
    specs = config["dataset"]["packed_sequence_specs"]
    pack = yaml.safe_load(PACK_CONFIG.read_text())
    submit = PACK_SUBMIT.read_text()
    tokenizer = yaml.safe_load(ARMS.read_text())["tokenizer_model"]
    assert config["dataset"]["dataset_root"] == pack["output_root"]
    assert config["tokenizer"]["tokenizer_model"] == tokenizer
    assert specs["packed_sequence_size"] == int(re.search(r"^SEQ_LENGTH=(\d+)$", submit, re.MULTILINE).group(1))
    assert specs["pad_seq_to_mult"] == int(re.search(r"^PAD_SEQ_TO_MULT=(\d+)$", submit, re.MULTILINE).group(1))
    # <root>/<glob of the roots>/packed/<folder of the tokenizer>/training_<length>.idx.parquet
    path = Path(specs["packed_train_data_path"])
    roots_glob = path.parents[2].name
    assert all(fnmatch.fnmatch(ROOT_FOLDER.format(index=index), roots_glob) for index in range(pack["shards"]))
    expected = packed_parquet_path(
        Path(pack["output_root"]) / roots_glob, tokenizer, specs["pad_seq_to_mult"], specs["packed_sequence_size"]
    )
    assert path == expected


def test_the_sequence_length_is_stated_in_every_place_and_agrees(run):
    config = run["config"]
    assert config["dataset"]["seq_length"] == config["model"]["seq_length"] == SEQUENCE
    assert config["dataset"]["packed_sequence_specs"]["packed_sequence_size"] == SEQUENCE


def test_every_pack_is_padded_to_full_length_and_only_the_answers_carry_loss(run):
    kwargs = run["config"]["dataset"]["dataset_kwargs"]
    assert kwargs == {
        "chat": True,
        "use_hf_tokenizer_chat_template": True,
        "answer_only_loss": True,
        "pad_to_max_length": True,
    }


def test_the_layout_divides_the_batch_and_keeps_the_experts_inside_a_node(run):
    model = run["config"]["model"]
    ways = model["tensor_model_parallel_size"] * model["pipeline_model_parallel_size"] * model["context_parallel_size"]
    assert model["context_parallel_size"] == 1
    assert run["gpus"] % ways == 0
    replicas = run["gpus"] // ways
    assert GLOBAL_BATCH % (replicas * run["config"]["train"]["micro_batch_size"]) == 0
    assert model["tensor_model_parallel_size"] * model["expert_model_parallel_size"] <= GPUS_PER_NODE
    assert run["gpus"] % GPUS_PER_NODE == 0 and replicas % model["expert_model_parallel_size"] == 0


def test_no_tensorboard_and_each_run_saves_to_a_directory_of_its_own():
    merged = {name: merge_onto_recipe(FOLDER / spec["file"], spec["recipe"]) for name, spec in RUNS.items()}
    for name, spec in RUNS.items():
        others = [cfg for other, cfg in merged.items() if other != name]
        assert_it_saves_to_a_directory_of_its_own(merged[name], spec["run"], others)
        assert merged[name].logger.tensorboard_dir is None


def test_the_real_recipe_takes_the_config_as_the_launcher_merges_it(run):
    cfg = merge_onto_recipe(run["path"], run["recipe"])
    assert cfg.train.train_iters == STEPS and cfg.train.global_batch_size == GLOBAL_BATCH
    assert cfg.optimizer.lr == run["lr"] and cfg.scheduler.lr_decay_style == "cosine"
    assert list(cfg.checkpoint.save_steps) == [1193, 2385] and cfg.checkpoint.save_interval == STEPS
    assert cfg.checkpoint.save_optim is run["optim"]
    assert cfg.dataset.seq_length == SEQUENCE and cfg.model.seq_length == SEQUENCE
    assert cfg.model.context_parallel_size == 1


def test_the_nano_reduces_gradients_and_holds_its_optimizer_as_the_baseline_sft_stage_does():
    """The Nano run takes the mixed precision, the gradient-reduction precision and the optimizer-state dtypes of the baseline SFT stage."""
    reference = merge_onto_recipe(FOLDER / RUNS["nano"]["file"], nemotron_3_nano_sft_config)
    baseline = merge_onto_recipe(BASELINE_SFT, nemotron_3_nano_sft_config)
    assert reference.mixed_precision == baseline.mixed_precision
    assert reference.ddp.grad_reduce_in_fp32 is baseline.ddp.grad_reduce_in_fp32 is True
    for field in ("use_precision_aware_optimizer", "exp_avg_dtype", "exp_avg_sq_dtype"):
        assert getattr(reference.optimizer, field) == getattr(baseline.optimizer, field)

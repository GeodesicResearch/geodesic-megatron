# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The run configs of the pretraining efficiency study (``configs/olmo_mix_efficiency/``): Nemotron 3 Nano trained from
random weights on chunks of the Olmo pretraining mix, as overlays of stage 1 of the 30B baseline on Verda.

The study compares two models by their loss at equal compute, so its numbers hold only for the recipe that ran:

- The trunk trains the baseline's recipe on other data. It differs from the baseline's Verda config in what the other
  data needs (the tokenizer, the data, its masked id, the length of one pass) and in where the run saves and logs,
  and in nothing else. Its warmup stays the baseline's.
- A tail continues one trunk checkpoint on the same data while the learning rate decays linearly to a floor. It
  changes the schedule and where it saves, and never ``optimizer.lr`` or ``optimizer.min_lr``: the distributed
  optimizer finds each saved parameter group by a key that holds these two values, so a tail that changed one could
  not load its checkpoint.
- The tail configs are generated (``make_tails.py``). The files on disk are the ones the script writes.

The tests read the YAML files alone, through the composition the launcher uses. They build no model.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import yaml
from scripts.dotted_paths import dotted_leaves
from scripts.training.config_compose import BASE_CONFIG_KEY, load_composed_yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY = REPO_ROOT / "configs" / "olmo_mix_efficiency"
BASELINE = (
    REPO_ROOT / "configs" / "control_pretraining" / "30b_baseline_verda" / "nemotron_nano_30b_baseline_pretrain.yaml"
)
PREFIX = "nemotron_nano_olmo_mix_"
TRUNK = STUDY / f"{PREFIX}trunk.yaml"
SEED2 = STUDY / f"{PREFIX}trunk_seed2.yaml"
SMOKE = STUDY / f"{PREFIX}smoke.yaml"

# The documents of the data pool, each of exactly dataset.seq_length tokens (the study's README).
POOL_DOCUMENTS = 22_528_001
ABSENT = "<absent>"

# What the trunk may state that the baseline's Verda config does not, or states with another value.
OTHER_DATA = {
    "tokenizer.tokenizer_model",
    "dataset.data_path",
    "dataset.path_to_cache",
    "token_masking.enabled",
    "token_masking.token_ids",
    "train.train_iters",
    "train.exit_interval",
    "scheduler.lr_decay_iters",
}
OWN_RUN = {
    "checkpoint.load",
    "checkpoint.save",
    "checkpoint.save_interval",
    "logger.wandb_project",
    "logger.wandb_exp_name",
}
# What a tail may state that the config it continues does not, or states with another value.
TAIL_SCHEDULE = {
    "train.exit_interval",
    "scheduler.lr_decay_style",
    "scheduler.lr_wsd_decay_style",
    "scheduler.lr_decay_iters",
    "scheduler.lr_wsd_decay_iters",
    "scheduler.lr_warmup_fraction",
    "scheduler.override_opt_param_scheduler",
    "checkpoint.ckpt_step",
    "checkpoint.save",
    "checkpoint.save_interval",
    "logger.wandb_exp_name",
}


def _load_make_tails():
    spec = importlib.util.spec_from_file_location("olmo_mix_efficiency_make_tails", STUDY / "make_tails.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclass looks its own module up by name
    spec.loader.exec_module(module)
    return module


MAKE_TAILS = _load_make_tails()
TAILS = {STUDY / f"{PREFIX}{tail.name}.yaml": tail for tail in MAKE_TAILS.TAILS}
RUNS = [TRUNK, SEED2, SMOKE, *TAILS]


def composed(path: Path) -> dict:
    """The config at ``path`` read through its ``base_config`` chain, as dotted leaves."""
    return dotted_leaves(load_composed_yaml(path))


def differing(candidate: dict, reference: dict) -> set[str]:
    """The dotted paths at which the two flat configs differ, a path only one of them has included."""
    return {
        key for key in candidate.keys() | reference.keys() if candidate.get(key, ABSENT) != reference.get(key, ABSENT)
    }


def base_of(path: Path) -> Path:
    """The config that the overlay at ``path`` names as its base."""
    return (path.parent / yaml.safe_load(path.read_text())[BASE_CONFIG_KEY]).resolve()


def _name(path: Path) -> str:
    return path.name.removeprefix(PREFIX).removesuffix(".yaml")


def test_the_folder_holds_the_studys_configs_and_no_other_yaml():
    """A YAML file the tests do not know would be a run config that nothing holds to the study's rules."""
    assert sorted(STUDY.glob("*.y*ml")) == sorted(RUNS)


def test_the_trunk_is_the_baseline_but_for_its_data_and_its_own_run():
    trunk, baseline = composed(TRUNK), composed(BASELINE)
    assert differing(trunk, baseline) == OTHER_DATA | OWN_RUN
    # The values the comparison with the dense model rests on, stated so that a change of the baseline shows here.
    assert trunk["train.global_batch_size"] == 2048
    assert trunk["dataset.seq_length"] == 8192
    assert (trunk["optimizer.lr"], trunk["optimizer.min_lr"]) == (MAKE_TAILS.PEAK, MAKE_TAILS.MIN_LR)
    assert trunk["scheduler.lr_decay_style"] == "constant"


def test_the_trunk_reads_each_document_of_the_pool_at_most_once():
    """Megatron sizes the data from train_iters. One more iteration would need a second pass over the pool."""
    trunk = composed(TRUNK)
    assert trunk["dataset.data_path"] == [trunk["dataset.data_path"][0]], "a weight would add a second epoch"
    assert trunk["train.train_iters"] == POOL_DOCUMENTS // trunk["train.global_batch_size"]
    assert trunk["train.exit_interval"] <= trunk["train.train_iters"]


def test_the_trunk_keeps_the_warmup_of_the_baseline():
    """The warmup is lr_warmup_fraction of lr_decay_iters, which defaults to train_iters. The trunk has fewer
    iterations than the baseline, so it states the baseline's count."""
    trunk, baseline = composed(TRUNK), composed(BASELINE)
    assert "scheduler.lr_decay_iters" not in baseline
    assert trunk["scheduler.lr_decay_iters"] == baseline["train.train_iters"]
    assert trunk["scheduler.lr_warmup_fraction"] == baseline["scheduler.lr_warmup_fraction"]
    assert trunk["scheduler.lr_warmup_iters"] == 0
    warmup = round(trunk["scheduler.lr_warmup_fraction"] * trunk["scheduler.lr_decay_iters"])
    assert warmup == 299
    assert all(tail.start > warmup for path, tail in TAILS.items() if base_of(path) != SMOKE)


def test_the_trunk_masks_the_pad_id_of_the_olmo_data():
    trunk = composed(TRUNK)
    assert trunk["token_masking.enabled"] is True
    assert trunk["token_masking.token_ids"] == [100277]


def test_the_repeat_changes_the_seed_of_the_weights_alone():
    """The repeat reads the trunk's samples in the trunk's order, so the two runs differ by their first weights."""
    repeat, trunk = composed(SEED2), composed(TRUNK)
    own_run = {"checkpoint.load", "checkpoint.save", "logger.wandb_exp_name"}
    assert differing(repeat, trunk) == {"rng.seed", "train.exit_interval"} | own_run
    assert repeat["checkpoint.save_interval"] == trunk["checkpoint.save_interval"]
    assert "dataset.random_seed" not in differing(repeat, trunk)


def test_the_smoke_run_is_the_trunk_at_a_smaller_size():
    smoke, trunk = composed(SMOKE), composed(TRUNK)
    size = {"train.global_batch_size", "train.train_iters", "train.exit_interval", "checkpoint.save_interval"}
    own_run = {"checkpoint.load", "checkpoint.save", "dataset.path_to_cache", "logger.wandb_exp_name"}
    assert differing(smoke, trunk) == size | own_run
    assert smoke["train.global_batch_size"] % (8 * smoke["train.micro_batch_size"]) == 0, "one node has 8 GPUs"


def test_the_tail_configs_are_the_ones_make_tails_writes():
    on_disk = {path.name: path.read_text() for path in TAILS}
    assert on_disk == MAKE_TAILS.tail_configs()


@pytest.mark.parametrize("path", list(TAILS), ids=_name)
def test_a_tail_changes_the_schedule_and_where_it_saves(path: Path):
    stated = dotted_leaves(yaml.safe_load(path.read_text()))
    assert set(stated) - {BASE_CONFIG_KEY, "checkpoint.load", "scheduler.lr_warmup_iters"} == TAIL_SCHEDULE
    tail, base = composed(path), composed(base_of(path))
    assert differing(tail, base) == TAIL_SCHEDULE
    # The two values the optimizer's saved state is keyed by.
    assert not any(key.startswith("optimizer.") for key in stated)
    assert (tail["optimizer.lr"], tail["optimizer.min_lr"]) == (base["optimizer.lr"], base["optimizer.min_lr"])


@pytest.mark.parametrize("path", list(TAILS), ids=_name)
def test_a_tail_starts_at_a_saved_iteration_of_the_run_it_continues(path: Path):
    tail, base = composed(path), composed(base_of(path))
    assert tail["checkpoint.load"] == base["checkpoint.save"]
    assert tail["checkpoint.ckpt_step"] == TAILS[path].start
    assert tail["checkpoint.ckpt_step"] % base["checkpoint.save_interval"] == 0
    assert tail["checkpoint.ckpt_step"] <= base["train.exit_interval"]
    assert tail["train.exit_interval"] <= tail["train.train_iters"], "the tail stays inside the one pass of the data"


@pytest.mark.parametrize("path", list(TAILS), ids=_name)
def test_a_tail_decays_linearly_from_its_start_to_its_floor(path: Path):
    """The schedule of the config, worked as the WSD scheduler works it: constant to lr_decay_iters less
    lr_wsd_decay_iters, then linear to min_lr at lr_decay_iters. The run stops at exit_interval."""
    tail, planned = composed(path), TAILS[path]
    assert (tail["scheduler.lr_decay_style"], tail["scheduler.lr_wsd_decay_style"]) == ("WSD", "linear")
    assert tail["scheduler.override_opt_param_scheduler"] is True
    assert (tail["scheduler.lr_warmup_fraction"], tail["scheduler.lr_warmup_iters"]) == (None, 0)
    decay_start = tail["scheduler.lr_decay_iters"] - tail["scheduler.lr_wsd_decay_iters"]
    assert decay_start == tail["checkpoint.ckpt_step"], "the decay starts at the first iteration of the tail"
    end = tail["train.exit_interval"]
    assert end - decay_start == planned.length
    assert end <= tail["scheduler.lr_decay_iters"]
    peak, floor = tail["optimizer.lr"], tail["optimizer.min_lr"]
    last = floor + (1.0 - (end - decay_start) / tail["scheduler.lr_wsd_decay_iters"]) * (peak - floor)
    assert math.isclose(last, planned.floor, rel_tol=1e-9)
    assert tail["checkpoint.save_interval"] == end, "the tail saves its last iteration and no other"


def test_the_tails_follow_the_rule_of_the_study():
    """A tail has 10% of the iterations of its start and at least 100, and ends at 1% of the peak. The calibration
    tails at iteration 1,000 and the tail of the smoke run are the exceptions."""
    exceptions = {"nano_tail_1000_len05", "nano_tail_1000_len20", "nano_tail_1000_floor10", "nano_smoke_tail"}
    for tail in MAKE_TAILS.TAILS:
        if tail.name not in exceptions:
            assert tail.length == max(100, round(0.10 * tail.start)), tail.name
            assert tail.floor == MAKE_TAILS.MIN_LR == 0.01 * MAKE_TAILS.PEAK, tail.name
    assert exceptions <= {tail.name for tail in MAKE_TAILS.TAILS}


def test_each_run_saves_to_a_directory_of_its_own_and_logs_under_its_own_name():
    saves = [composed(path)["checkpoint.save"] for path in RUNS]
    names = [composed(path)["logger.wandb_exp_name"] for path in RUNS]
    assert len(set(saves)) == len(RUNS)
    assert len(set(names)) == len(RUNS)
    for path, save, name in zip(RUNS, saves, names):
        assert Path(save).parent == Path(saves[0]).parent, f"{path.name} saves away from the trunk's folder"
        assert name == f"olmo_mix_efficiency_{Path(save).name}", path.name


@pytest.mark.parametrize("path", RUNS, ids=_name)
def test_each_run_logs_to_the_studys_project_and_writes_no_tensorboard_files(path: Path):
    config = composed(path)
    assert (config["logger.wandb_entity"], config["logger.wandb_project"]) == ("geodesic", "olmo_3_megatron")
    assert config["logger.tensorboard_dir"] is None

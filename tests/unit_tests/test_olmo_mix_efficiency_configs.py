# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The run configs of the pretraining efficiency study (``configs/olmo_mix_efficiency/``).

The configs are overlays of stage 1 of the 30B baseline on Verda. The README of the folder describes the study.

- The trunk differs from the Verda config of the baseline only in the keys of ``OTHER_DATA`` and ``OWN_RUN``.
- A tail differs from the config it continues only in the keys of ``TAIL_SCHEDULE``.
- Each tail config on disk is the text that ``make_tails.py`` writes from ``tails.yaml``.

The tests read the YAML files through ``load_composed_yaml``. They build no model.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml
from scripts.dotted_paths import dotted_leaves
from scripts.training.config_compose import BASE_CONFIG_KEY, composition_chain, load_composed_yaml

from tests.unit_tests.repo_scripts import load_script


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY = REPO_ROOT / "configs" / "olmo_mix_efficiency"
BASELINE = (
    REPO_ROOT / "configs" / "control_pretraining" / "30b_baseline_verda" / "nemotron_nano_30b_baseline_pretrain.yaml"
)
PREFIX = "nemotron_nano_olmo_mix_"
TRUNK = STUDY / f"{PREFIX}trunk.yaml"
SEED2 = STUDY / f"{PREFIX}trunk_seed2.yaml"
SMOKE = STUDY / f"{PREFIX}smoke.yaml"
SPEC = STUDY / "tails.yaml"

# The documents of the data pool, each of dataset.seq_length tokens (the comment at dataset.data_path of the trunk).
POOL_DOCUMENTS = 22_528_001
# optimizer.lr and optimizer.min_lr of the baseline. Each run of the study keeps them.
PEAK, MIN_LR = 1.0e-3, 1.0e-5
ABSENT = "<absent>"

# The keys at which the trunk differs from the Verda config of the baseline, for its data.
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
# The keys at which the trunk differs from the Verda config of the baseline, for its saves and its logs.
OWN_RUN = {
    "checkpoint.load",
    "checkpoint.save",
    "checkpoint.save_interval",
    "logger.wandb_project",
    "logger.wandb_exp_name",
}
# The keys at which a tail differs from the config it continues.
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

MAKE_TAILS = load_script("make_tails", STUDY)
SETTINGS, SPEC_TAILS = MAKE_TAILS.read_tails(SPEC)
TAILS = {STUDY / SETTINGS["config_name"].format(tail=tail.name): tail for tail in SPEC_TAILS}
RUNS = [TRUNK, SEED2, SMOKE, *TAILS]


def composed(path: Path) -> dict:
    """The config at ``path`` read through its ``base_config`` chain, as dotted leaves."""
    return dotted_leaves(load_composed_yaml(path))


def differing(candidate: dict, reference: dict) -> set[str]:
    """The dotted paths at which the two flat configs differ, a path only one of them has included."""
    return {
        key for key in candidate.keys() | reference.keys() if candidate.get(key, ABSENT) != reference.get(key, ABSENT)
    }


def _name(path: Path) -> str:
    return path.name.removeprefix(PREFIX).removesuffix(".yaml")


def spec_in(folder: Path, **changes) -> Path:
    """A copy of the spec of the study in ``folder``, with ``changes`` to its settings. It names the template and
    each config of the study by an absolute path. ``make_tails`` thus reads the files of the study."""
    spec = yaml.safe_load(SPEC.read_text())
    spec["template"] = str(STUDY / spec["template"])
    spec["save_beside"] = str(STUDY / spec["save_beside"])
    for tail in spec["tails"].values():
        tail["base"] = str(STUDY / tail["base"])
    path = folder / "tails.yaml"
    path.write_text(yaml.safe_dump(spec | changes))
    return path


def test_the_folder_holds_the_studys_configs_and_the_spec_of_the_tails():
    """Each YAML file of the folder is a run config that these tests check, or the spec of the tails."""
    assert sorted(STUDY.glob("*.y*ml")) == sorted([*RUNS, SPEC])


def test_the_trunk_is_the_baseline_but_for_its_data_and_its_own_run():
    trunk, baseline = composed(TRUNK), composed(BASELINE)
    assert differing(trunk, baseline) == OTHER_DATA | OWN_RUN
    # The values the comparison with the dense model rests on, stated so that a change of the baseline shows here.
    assert trunk["train.global_batch_size"] == 2048
    assert trunk["dataset.seq_length"] == 8192
    assert (trunk["optimizer.lr"], trunk["optimizer.min_lr"]) == (PEAK, MIN_LR)
    assert trunk["scheduler.lr_decay_style"] == "constant"


def test_the_trunk_reads_each_document_of_the_pool_at_most_once():
    """Megatron sizes the data from train_iters. One more iteration needs a second pass over the pool."""
    trunk = composed(TRUNK)
    assert trunk["dataset.data_path"] == [trunk["dataset.data_path"][0]], "a weight adds a second epoch"
    assert trunk["train.train_iters"] == POOL_DOCUMENTS // trunk["train.global_batch_size"]
    assert trunk["train.exit_interval"] <= trunk["train.train_iters"]


def test_the_trunk_keeps_the_warmup_of_the_baseline():
    """The trunk states train_iters of the baseline as its lr_decay_iters. Its warmup then has the same length."""
    trunk, baseline = composed(TRUNK), composed(BASELINE)
    assert "scheduler.lr_decay_iters" not in baseline
    assert trunk["scheduler.lr_decay_iters"] == baseline["train.train_iters"]
    assert trunk["scheduler.lr_warmup_fraction"] == baseline["scheduler.lr_warmup_fraction"]
    assert trunk["scheduler.lr_warmup_iters"] == 0
    warmup = round(trunk["scheduler.lr_warmup_fraction"] * trunk["scheduler.lr_decay_iters"])
    assert warmup == 299
    assert all(tail.start > warmup for path, tail in TAILS.items() if composition_chain(path)[1] != SMOKE)


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
    assert on_disk == MAKE_TAILS.tail_configs(SPEC)


def test_make_tails_writes_each_tail_config_beside_its_spec(tmp_path):
    spec = spec_in(tmp_path)
    MAKE_TAILS.main([str(spec)])
    written = {path.name: path.read_text() for path in tmp_path.glob(f"{PREFIX}*.yaml")}
    assert written == MAKE_TAILS.tail_configs(spec)
    assert sorted(written) == sorted(path.name for path in TAILS)
    for path in TAILS:
        assert yaml.safe_load(written[path.name]) | {BASE_CONFIG_KEY: None} == yaml.safe_load(path.read_text()) | {
            BASE_CONFIG_KEY: None
        }


def test_a_spec_with_a_wrong_key_is_refused(tmp_path):
    with pytest.raises(ValueError, match="the keys of the spec"):
        MAKE_TAILS.read_tails(spec_in(tmp_path, floor=MIN_LR))
    no_floor = {
        name: {"base": tail.base, "start": tail.start, "length": tail.length} for name, tail in [("a", SPEC_TAILS[0])]
    }
    with pytest.raises(ValueError, match="the keys of tail a"):
        MAKE_TAILS.read_tails(spec_in(tmp_path, tails=no_floor))
    twice = tmp_path / "twice.yaml"
    twice.write_text(SPEC.read_text() + "template: other.tmpl\n")
    with pytest.raises(yaml.YAMLError, match="duplicate key template"):
        MAKE_TAILS.read_tails(twice)


def test_two_tails_with_one_file_name_are_refused(tmp_path):
    with pytest.raises(ValueError, match="two tails have the file name"):
        MAKE_TAILS.tail_configs(spec_in(tmp_path, config_name="tail.yaml"))


def test_a_floor_outside_the_learning_rates_of_the_base_is_refused():
    def tail(floor: float):
        return MAKE_TAILS.Tail(name="t", base="b.yaml", start=1000, length=100, floor=floor)

    assert MAKE_TAILS.decay_iterations(tail(MIN_LR), PEAK, MIN_LR) == 100
    assert MAKE_TAILS.decay_iterations(tail(1.0e-4), PEAK, MIN_LR) == 110
    for floor in (0.5 * MIN_LR, PEAK, 2 * PEAK):
        with pytest.raises(ValueError, match="the floor"):
            MAKE_TAILS.decay_iterations(tail(floor), PEAK, MIN_LR)


@pytest.mark.parametrize("path", list(TAILS), ids=_name)
def test_a_tail_changes_the_schedule_and_where_it_saves(path: Path):
    stated = dotted_leaves(yaml.safe_load(path.read_text()))
    # A tail states checkpoint.load and lr_warmup_iters with the values of its base, so the two keys do not differ.
    assert set(stated) - {BASE_CONFIG_KEY, "checkpoint.load", "scheduler.lr_warmup_iters"} == TAIL_SCHEDULE
    tail, base = composed(path), composed(composition_chain(path)[1])
    assert differing(tail, base) == TAIL_SCHEDULE
    # The two values the optimizer's saved state is keyed by.
    assert not any(key.startswith("optimizer.") for key in stated)
    assert (tail["optimizer.lr"], tail["optimizer.min_lr"]) == (base["optimizer.lr"], base["optimizer.min_lr"])


@pytest.mark.parametrize("path", list(TAILS), ids=_name)
def test_a_tail_starts_at_a_saved_iteration_of_the_run_it_continues(path: Path):
    tail, base = composed(path), composed(composition_chain(path)[1])
    assert tail["checkpoint.load"] == base["checkpoint.save"]
    assert tail["checkpoint.ckpt_step"] == TAILS[path].start
    assert tail["checkpoint.ckpt_step"] % base["checkpoint.save_interval"] == 0
    assert tail["checkpoint.ckpt_step"] <= base["train.exit_interval"]
    assert tail["train.exit_interval"] <= tail["train.train_iters"], "the tail stays inside the one pass of the data"


@pytest.mark.parametrize("path", list(TAILS), ids=_name)
def test_a_tail_decays_linearly_from_its_start_to_its_floor(path: Path):
    """The WSD schedule is constant to lr_decay_iters less lr_wsd_decay_iters. It is then linear to min_lr at
    lr_decay_iters. The run stops at exit_interval."""
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
    """A tail has 10% of the iterations of its start, and at least 100 iterations. It ends at 1% of the peak. The
    calibration tails at iteration 1,000 and the tail of the smoke run are the exceptions."""
    exceptions = {"nano_tail_1000_len05", "nano_tail_1000_len20", "nano_tail_1000_floor10", "nano_smoke_tail"}
    for tail in SPEC_TAILS:
        if tail.name not in exceptions:
            assert tail.length == max(100, round(0.10 * tail.start)), tail.name
            assert tail.floor == MIN_LR == 0.01 * PEAK, tail.name
    assert exceptions <= {tail.name for tail in SPEC_TAILS}


def test_each_run_saves_to_a_folder_of_its_own_and_logs_under_its_own_name():
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

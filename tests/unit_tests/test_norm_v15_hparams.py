# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The Norm V1.5 arms differ from the Verda baseline in their batch and learning rate alone, and read one token stream.

An arm is evidence about its batch and learning rate only if nothing else moved. So each arm, merged onto the Nano
pretraining recipe as the launcher merges it, differs from the merged Verda baseline in exactly the ablated fields,
the fields that follow from the batch (iterations, exit, evaluation and save intervals), the held-out sets and its
run identity. The token stream is the same in every arm only while the dataset keeps the baseline's size, the width
stays 8 x 8 and the micro-batch stays 4, so each of those is checked too. The cooldown branches are what the spec
renders, and each differs from its arm in its schedule, its exit and its identity alone.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest
import yaml
from scripts.training.launch_width import config_launch_width

from pipeline.training import recipes
from tests.unit_tests.campaign_config import assert_only_these_fields_differ, merge_onto_recipe
from tests.unit_tests.corpora_fixtures import load_campaign_module


REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = REPO_ROOT / "configs" / "control_pretraining" / "30b_norm_v15_hparams"
VERDA_BASELINE = (
    REPO_ROOT / "configs" / "control_pretraining" / "30b_baseline_verda" / "nemotron_nano_30b_baseline_pretrain.yaml"
)
SPEC = CAMPAIGN / "cooldowns.yaml"
ARMS = {
    "a0": (2048, 1.0e-3),
    "a1": (2048, 7.1e-4),
    "a2": (2048, 1.41e-3),
    "b1": (1024, 1.0e-3),
    "c1": (512, 7.1e-4),
    "d1": (256, 5.0e-4),
}
# The baseline's dataset, an arm's 100B tokens and its 20B-token interval, in samples of 8,192 tokens.
DATASET_SAMPLES, EXIT_SAMPLES, INTERVAL_SAMPLES = 61_196_288, 12_206_080, 2_441_216
HELD_OUT_SAMPLES = 122_880
SETS = ["unseen_pretraining", "arxiv_papers", "nemotron_stem_sft"]
CHECKPOINTS = Path("/shared/geodesic/checkpoints_kyle/megatron/norm_v15")
ABLATED = {"train.global_batch_size", "optimizer.lr"}
# A branch loads from its arm's directory and states its decay shape, minus_sqrt, which is the recipe's, so neither
# is a difference.
FOLLOWS_FROM_THE_BATCH = {"train.train_iters", "train.exit_interval", "checkpoint.save_interval"}
HELD_OUT = {
    "held_out_validation.sets",
    "held_out_validation.interval",
    "held_out_validation.at_start",
    "held_out_validation.at_exit",
}
IDENTITY = {"checkpoint.load", "checkpoint.save", "logger.wandb_exp_name"}
BRANCH = {
    "train.exit_interval",
    "scheduler.lr_decay_style",
    "scheduler.lr_decay_iters",
    "scheduler.lr_wsd_decay_iters",
    "optimizer.min_lr",
    "checkpoint.ckpt_step",
    "checkpoint.save",
    "checkpoint.save_optim",
    "logger.wandb_exp_name",
}

cooldowns = load_campaign_module("generate_cooldown_branches")


def _arm(arm: str) -> Path:
    return CAMPAIGN / f"nemotron_nano_30b_norm_v15_{arm}_pretrain.yaml"


def _merged(path: Path):
    """``path`` merged onto the Nano pretraining recipe, as the launcher trains it."""
    return merge_onto_recipe(path, partial(recipes.RECIPE_MAP[("nano", "pretrain")], None))


@pytest.fixture(autouse=True)
def verda_user(monkeypatch):
    """The account the relocation map's per-user roots are resolved for."""
    monkeypatch.setenv("USER", "someone")


def _assert_reads_the_baseline_stream(cfg, label: str) -> None:
    """An arm builds the baseline's dataset, so its sampler orders the samples as every other arm's does."""
    assert cfg.train.global_batch_size * cfg.train.train_iters == DATASET_SAMPLES, label
    assert cfg.train.micro_batch_size == 4, label


@pytest.mark.parametrize("arm", sorted(ARMS))
class TestAnArm:
    def test_it_differs_from_the_verda_baseline_in_the_batch_the_learning_rate_and_what_follows(self, arm):
        moved = ABLATED - ({"optimizer.lr"} if ARMS[arm][1] == 1.0e-3 else set())
        moved -= {"train.global_batch_size"} if ARMS[arm][0] == 2048 else set()
        changed = moved | FOLLOWS_FROM_THE_BATCH | HELD_OUT | IDENTITY
        if ARMS[arm][0] == 2048:
            changed -= {"train.train_iters"}
        assert_only_these_fields_differ(_merged(_arm(arm)), _merged(VERDA_BASELINE), changed, arm)

    def test_its_batch_and_learning_rate_are_the_arms(self, arm):
        cfg = _merged(_arm(arm))
        assert (cfg.train.global_batch_size, cfg.optimizer.lr) == ARMS[arm]

    def test_it_reads_the_baseline_stream_for_100b_tokens_and_evaluates_every_20b(self, arm):
        cfg = _merged(_arm(arm))
        batch = cfg.train.global_batch_size
        _assert_reads_the_baseline_stream(cfg, arm)
        assert cfg.train.exit_interval * batch == EXIT_SAMPLES
        assert cfg.held_out_validation.interval * batch == INTERVAL_SAMPLES == cfg.checkpoint.save_interval * batch
        assert (cfg.scheduler.lr_decay_style, cfg.scheduler.lr_warmup_fraction) == ("constant", 0.01)

    def test_it_trains_at_eight_nodes_of_eight_gpus(self, arm):
        width = config_launch_width(str(_arm(arm)))
        assert (width.nodes, width.gpus_per_node, width.data_parallel_size) == (8, 8, 64)

    def test_it_evaluates_the_three_sets_at_start_interval_and_exit(self, arm):
        cfg = _merged(_arm(arm))
        block = cfg.held_out_validation
        block.finalize(cfg.train.global_batch_size)
        assert [(held.name, held.samples) for held in block.sets] == [(name, HELD_OUT_SAMPLES) for name in SETS]
        assert (block.at_start, block.at_exit) == (True, True)

    def test_it_saves_to_and_resumes_from_a_directory_of_its_own(self, arm):
        cfg = _merged(_arm(arm))
        assert cfg.checkpoint.load == cfg.checkpoint.save == str(CHECKPOINTS / arm)
        assert cfg.checkpoint.save_optim and cfg.checkpoint.most_recent_k == -1
        assert cfg.logger.wandb_exp_name.startswith(f"norm_v15_{arm}_")


def test_an_arm_stopped_by_train_iters_reads_another_stream(tmp_path):
    """Stopping at 100B through train_iters builds a smaller dataset, whose samples the sampler orders otherwise."""
    short = tmp_path / "nemotron_nano_30b_norm_v15_short_pretrain.yaml"
    short.write_text(yaml.safe_dump({"base_config": str(_arm("a0")), "train": {"train_iters": 5960}}))
    with pytest.raises(AssertionError):
        _assert_reads_the_baseline_stream(_merged(short), "short")


def test_the_branch_files_are_what_the_spec_renders():
    rendered = cooldowns.render(cooldowns.read_spec(SPEC))
    on_disk = sorted(CAMPAIGN.glob("nemotron_nano_30b_norm_v15_*_cd*_pretrain.yaml"))
    assert sorted(rendered) == on_disk
    for path, text in rendered.items():
        assert path.read_text() == text, path.name


@pytest.mark.parametrize("arm", sorted(ARMS))
@pytest.mark.parametrize(
    ("branch", "at_samples", "decay_samples"),
    [("cd20b", 2_441_216, 487_424), ("cd60b", 7_323_648, 733_184), ("cd100b", 12_206_080, 1_220_608)],
)
def test_a_branch_resumes_its_arms_save_and_decays_to_zero(arm, branch, at_samples, decay_samples):
    path = CAMPAIGN / f"nemotron_nano_30b_norm_v15_{arm}_{branch}_pretrain.yaml"
    cfg, parent = _merged(path), _merged(_arm(arm))
    batch = parent.train.global_batch_size
    start, decay = at_samples // batch, decay_samples // batch
    assert_only_these_fields_differ(cfg, parent, BRANCH, path.name)
    assert cfg.train.train_iters == parent.train.train_iters
    assert cfg.train.exit_interval == cfg.scheduler.lr_decay_iters == start + decay
    assert (cfg.scheduler.lr_decay_style, cfg.scheduler.lr_wsd_decay_style) == ("WSD", "minus_sqrt")
    assert (cfg.scheduler.lr_wsd_decay_iters, cfg.optimizer.min_lr) == (decay, 0.0)
    assert (cfg.checkpoint.load, cfg.checkpoint.ckpt_step) == (parent.checkpoint.save, start)
    assert cfg.checkpoint.save == f"{parent.checkpoint.save}_{branch}" and not cfg.checkpoint.save_optim
    assert start % parent.checkpoint.save_interval == 0


def _spec(directory: Path, branches: list[dict], **changes) -> Path:
    """A cooldown spec over one arm built on a0, with ``branches`` and the top-level keys in ``changes``."""
    arm = directory / _arm("a0").name
    arm.write_text(yaml.safe_dump({"base_config": str(_arm("a0"))}))
    spec = directory / "cooldowns.yaml"
    settings = {"arms": [arm.name], "branches": branches, "decay_style": "minus_sqrt", "min_lr": 0.0, **changes}
    spec.write_text(yaml.safe_dump({key: value for key, value in settings.items() if value is not None}))
    return spec


BATCH = 2048


@pytest.mark.parametrize(
    ("branches", "changes", "message"),
    [
        ([{"name": "x", "at_samples": BATCH * 1000, "decay_samples": BATCH * 10}], {}, "iteration 1000 is not a save"),
        ([{"name": "x", "at_samples": BATCH * 7152, "decay_samples": BATCH * 10}], {}, "iteration 7152 is not a save"),
        ([{"name": "x", "at_samples": BATCH * 1192 + 1, "decay_samples": BATCH}], {}, "not whole iterations"),
        ([{"name": "x", "at_samples": BATCH * 5960, "decay_samples": BATCH * 30000}], {}, "past train_iters 29881"),
        ([{"name": "x", "at_samples": 0, "decay_samples": BATCH}], {}, "at_samples must be a whole number of at"),
        (
            [{"name": "x", "at_samples": BATCH * 1192, "decay_samples": BATCH}] * 2,
            {},
            "two branches share a name",
        ),
        ([], {"arms": ["nemotron_nano_30b_norm_v15_missing_pretrain.yaml"]}, "not beside it"),
        ([], {"min_lr": None}, "missing keys \\['min_lr'\\]"),
        ([], {"style": "minus_sqrt"}, "unknown keys \\['style'\\]"),
    ],
    ids=[
        "no-save",
        "past-the-exit",
        "not-whole-iterations",
        "decay-past-train-iters",
        "not-a-whole-number",
        "shared-name",
        "arm-not-beside",
        "missing-key",
        "unknown-key",
    ],
)
def test_a_spec_the_arms_cannot_branch_from_is_refused(tmp_path, branches, changes, message):
    with pytest.raises(ValueError, match=message):
        cooldowns.render(cooldowns.read_spec(_spec(tmp_path, branches, **changes)))

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The longmino_cpt arm is the midtrain stage with its data, budget and identity swapped.

Continued pretraining of the 30B baseline on 20B tokens of OLMo-3's stage-3 long-context
mix, warm-started weights-only from the final midtrain checkpoint. The decision that shapes
every test here: the hyperparameters are the midtrain's, verbatim. So the config is asserted
to differ from the midtrain in exactly the stated key set, the token budget and checkpoint
cadence are checked arithmetically (5.0/10.0/15.0/20.0B), the LR schedule is driven through
the real scheduler, and the blend is checked against the corpora actually built (skipped where
the corpus tree is not mounted).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml
from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_pretrain_config
from omegaconf import OmegaConf

from tests.unit_tests.campaign_config import (
    assert_blend_is_well_formed,
    assert_only_these_fields_differ,
    assert_segment_exit_posture,
    merge_onto_recipe,
)

REPO = Path(__file__).resolve().parents[2]
ARM = REPO / "configs" / "control_pretraining" / "longmino_cpt"
CONFIG = ARM / "nemotron_nano_30b_baseline_longmino_cpt.yaml"
MIDTRAIN = REPO / "configs" / "control_pretraining" / "30b_baseline" / "nemotron_nano_30b_baseline_midtrain.yaml"
DATA_YAML = ARM / "data" / "longmino_cpt_20b.yaml"
MANIFEST = ARM / "data" / "longmino_cpt_20b.manifest.json"

TOKENS_PER_ITER = 512 * 32768
TARGET_TOKENS = 20_000_000_000
CHECKPOINT_TOKENS = [5e9, 10e9, 15e9, 20e9]
OUR_ROOTS = ("/projects/a5k/public/data_cwtice.a5k/data/",)

ALLOWED_DIVERGENCE = {
    "dataset.data_path",
    "dataset.path_to_cache",
    "train.train_iters",
    "scheduler.lr_wsd_decay_iters",
    "checkpoint.pretrained_checkpoint",
    "checkpoint.load",
    "checkpoint.save",
    "checkpoint.save_interval",
    "logger.wandb_exp_name",
}


@pytest.fixture(scope="module")
def merged():
    return merge_onto_recipe(CONFIG, nemotron_3_nano_pretrain_config)


@pytest.fixture(scope="module")
def midtrain_merged():
    return merge_onto_recipe(MIDTRAIN, nemotron_3_nano_pretrain_config)


@pytest.fixture(scope="module")
def raw():
    return OmegaConf.load(CONFIG)


@pytest.fixture(scope="module")
def data_cfg():
    return yaml.safe_load(DATA_YAML.read_text())


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text())


class TestItIsTheMidtrainWithTheStatedDiff:
    def test_exactly_the_stated_fields_differ(self, merged, midtrain_merged):
        assert_only_these_fields_differ(merged, midtrain_merged, ALLOWED_DIVERGENCE, "longmino_cpt")

    def test_hyperparameters_are_the_midtrains(self, merged, midtrain_merged):
        for key in ("lr", "min_lr", "adam_beta1", "adam_beta2", "weight_decay"):
            assert getattr(merged.optimizer, key) == getattr(midtrain_merged.optimizer, key), key
        for key in ("lr_decay_style", "lr_wsd_decay_style", "lr_warmup_iters"):
            assert getattr(merged.scheduler, key) == getattr(midtrain_merged.scheduler, key), key

    def test_decay_window_is_the_whole_run(self, merged, raw):
        assert raw.scheduler.lr_wsd_decay_iters == merged.train.train_iters

    def test_seq_length_is_stated_in_both_places_and_agrees(self, merged, raw):
        assert raw.dataset.seq_length == 32768
        assert raw.model.seq_length == 32768
        assert merged.model.seq_length == merged.dataset.seq_length == 32768

    def test_warmup_iters_is_stated_not_inherited(self, raw):
        assert raw.scheduler.lr_warmup_iters == 100

    def test_validation_split_is_the_only_safe_one(self, raw):
        assert str(raw.dataset.split) == "1,0,0"

    def test_index_cache_is_the_arms_own_and_ours(self, raw, midtrain_merged):
        assert raw.dataset.path_to_cache.startswith("/projects/a5k/public/data_cwtice.a5k/")
        assert raw.dataset.path_to_cache != midtrain_merged.dataset.path_to_cache


class TestBudgetAndCadence:
    def test_token_budget_is_twenty_billion(self, merged):
        tokens = merged.train.train_iters * TOKENS_PER_ITER
        assert abs(tokens - TARGET_TOKENS) < TOKENS_PER_ITER
        assert merged.train.global_batch_size * merged.model.seq_length == TOKENS_PER_ITER

    def test_checkpoints_land_at_five_billion_token_marks(self, merged):
        step = merged.checkpoint.save_interval
        saves = [step * k for k in range(1, merged.train.train_iters // step + 1)]
        assert len(saves) == 4
        for iteration, target in zip(saves, CHECKPOINT_TOKENS, strict=True):
            assert abs(iteration * TOKENS_PER_ITER - target) < TOKENS_PER_ITER, (iteration, target)
        assert saves[-1] == merged.train.train_iters, "the final save is the end of the run"

    def test_every_checkpoint_is_retained_with_its_optimizer(self, merged):
        assert merged.checkpoint.most_recent_k == -1
        assert merged.checkpoint.save_optim is True
        assert merged.checkpoint.save_rng is True

    def test_segment_rollover_is_on_the_clock(self, merged):
        assert_segment_exit_posture(merged, "longmino_cpt", 1400)


class TestWarmStart:
    def test_warm_starts_weights_only_from_the_final_midtrain_dir(self, merged, midtrain_merged):
        assert merged.checkpoint.pretrained_checkpoint == midtrain_merged.checkpoint.save

    def test_load_and_save_are_one_new_directory(self, merged, midtrain_merged):
        assert merged.checkpoint.load == merged.checkpoint.save
        assert merged.checkpoint.save != midtrain_merged.checkpoint.save
        assert merged.checkpoint.save.endswith("control_pretrain_30b_baseline_longmino_cpt")

    @pytest.mark.skipif(
        not os.path.isdir("/projects/a5k/public/data_cwtice.a5k/checkpoints"), reason="checkpoint tree not mounted"
    )
    def test_the_warm_start_source_exists_with_its_final_iteration(self, merged):
        src = Path(merged.checkpoint.pretrained_checkpoint)
        assert (src / "latest_checkpointed_iteration.txt").read_text().strip() == "3126"
        assert (src / "iter_0003126").is_dir()


class TestSchedule:
    def test_lr_schedule_is_continuous_and_hits_its_floor(self, merged, raw):
        """Drive the real scheduler as the midtrain test does; the anneal spans the run."""
        import torch
        from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

        iters, warmup, wd = merged.train.train_iters, merged.scheduler.lr_warmup_iters, merged.optimizer.weight_decay
        scheduler = OptimizerParamScheduler(
            optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=merged.optimizer.lr),
            init_lr=0.0,
            max_lr=merged.optimizer.lr,
            min_lr=merged.optimizer.min_lr,
            lr_warmup_steps=warmup,
            lr_decay_steps=iters,
            lr_decay_style=merged.scheduler.lr_decay_style,
            start_wd=wd,
            end_wd=wd,
            wd_incr_steps=iters,
            wd_incr_style="constant",
            wsd_decay_steps=raw.scheduler.lr_wsd_decay_iters,
            lr_wsd_decay_style=merged.scheduler.lr_wsd_decay_style,
        )

        def lr_at(step):
            scheduler.num_steps = step
            return scheduler.get_lr({})

        assert lr_at(warmup) == pytest.approx(merged.optimizer.lr)
        # The WSD branch computes the anneal ratio from the raw step count (the warmup is not
        # subtracted), so the first post-warmup step is already warmup/train_iters into the
        # cosine: 100/1192 = 8.4% here against 100/3126 = 3.2% for the midtrain, i.e. a 1.8%
        # step below the peak instead of 0.25%. Same mechanism, shorter horizon — a kink, not
        # a jump. Anything larger would mean the decay window no longer spans the run.
        drop = 1.0 - lr_at(warmup + 1) / lr_at(warmup)
        assert 0.0 <= drop < 0.02, f"warmup->anneal step of {drop:.3%}"
        assert lr_at(iters) == pytest.approx(merged.optimizer.min_lr)
        annealing = [lr_at(s) for s in range(warmup + 1, iters + 1, 37)]
        assert all(b <= a for a, b in zip(annealing, annealing[1:], strict=False))


class TestDataBuildAgreesWithTheConfig:
    def test_family_regexes_are_total_and_disjoint_over_the_manifest(self, data_cfg, manifest):
        compiled = {fam: [re.compile(p) for p in pats] for fam, pats in data_cfg["families"].items()}
        sources = [s for f in manifest["families"].values() for s in f["sources"]]
        assert len(sources) == manifest["sources_total"]
        for src in sources:
            hits = [fam for fam, pats in compiled.items() if any(p.search(src) for p in pats)]
            assert len(hits) == 1, (src, hits)
        for fam, f in manifest["families"].items():
            for src in f["sources"]:
                assert any(p.search(src) for p in compiled[fam]), (fam, src)

    def test_manifest_is_pinned_to_the_data_yaml(self, data_cfg, manifest):
        assert manifest["repo"] == data_cfg["repo"]
        assert manifest["revision"] == data_cfg["revision"]
        assert re.fullmatch(r"[0-9a-f]{40}", manifest["revision"]), "revision must be a commit sha"
        assert manifest["fraction"] == data_cfg["fraction"]

    def test_manifest_selects_about_twenty_billion_tokens(self, manifest):
        assert 18e9 <= manifest["est_tokens"] <= 22e9

    def test_blend_is_well_formed_in_our_tree(self, raw):
        assert_blend_is_well_formed(raw.dataset.data_path, "longmino_cpt", roots=OUR_ROOTS)

    def test_every_blend_prefix_names_a_family_and_every_family_is_blended(self, raw, data_cfg):
        prefixes = [str(p) for p in raw.dataset.data_path[1::2]]
        base = data_cfg["data_base"].rstrip("/")
        named = {p[len(base) + 1 :].split("/")[0] for p in prefixes if p.startswith(base + "/")}
        assert len(named) == len(prefixes), "every prefix must live under the arm's data_base"
        assert named == set(data_cfg["families"])
        assert all(p.endswith("/tokenized_base_input_document") for p in prefixes)

    def test_one_tokenizer_across_the_build_and_the_training_config(self, data_cfg, merged):
        assert data_cfg["tokenizer"] == merged.tokenizer.tokenizer_model

    @pytest.mark.skipif(
        not os.path.isdir("/projects/a5k/public/data_cwtice.a5k/data/longmino_cpt_20b"), reason="corpus tree not mounted"
    )
    def test_weights_are_token_proportional_to_the_built_corpora(self):
        rc = subprocess.run(
            ["python3", str(ARM / "blend_weights.py"), "--check", str(CONFIG)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert rc.returncode == 0, rc.stderr or rc.stdout

    def test_dry_run_submits_two_jobs_per_family(self, data_cfg):
        out = subprocess.run(
            ["bash", str(ARM / "build_corpora.sh"), "all"],
            cwd=REPO,
            env={**os.environ, "DRY_RUN": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # the dry run prints the sbatch lines on stderr
            text=True,
            check=True,
        ).stdout
        n_fam = len(data_cfg["families"])
        assert f"SUBMITTED {2 * n_fam} jobs for {n_fam} families" in out
        assert out.count("--dependency=afterok:") == n_fam, "every tokenize waits on its slice"
        assert out.count("pipeline_data_submit.sbatch tokenize") == n_fam

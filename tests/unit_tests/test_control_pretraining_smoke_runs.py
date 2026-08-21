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

"""`smoke_e2e_run` stays a faithful short copy of the arm it certifies.

A smoke run is evidence about the full-scale config only to the extent that it differs from it
in length alone. The failure this file exists to prevent is silent divergence: the arm's blend
is revised, or its topology retuned, and the smoke keeps passing against the shape it had when
it was written — certifying a config nobody is going to run.

So the assertions come in two halves. One pins everything that must be IDENTICAL to the parent
(blend, parallelism, recompute posture, the DP=512 save-crossing settings). The other pins the
short list that is allowed to differ, so an unreviewed change to any other field fails here
rather than at hour three of a 128-node allocation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import (
    nemotron_3_nano_pretrain_config,
    nemotron_3_nano_sft_config,
)
from tests.unit_tests.campaign_config import merge_onto_recipe


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARM_DIR = _REPO_ROOT / "configs" / "control_pretraining" / "30b_baseline"
_SMOKE_DIR = _REPO_ROOT / "configs" / "control_pretraining" / "smoke_runs"

SMOKE_ITERS = 100
TOKENS_PER_ITER = 16_777_216
SMOKE_TOKENS = SMOKE_ITERS * TOKENS_PER_ITER
CHAIN_TOKENS = 3 * SMOKE_TOKENS

SMOKE_CKPT_ROOT = "/projects/a5k/public/checkpoints/megatron/control_pretraining/smoke_e2e"

# stage -> (parent config, smoke config, the recipe the launcher merges onto)
STAGES = {
    "pretrain": (
        _ARM_DIR / "nemotron_nano_30b_baseline_pretrain.yaml",
        _SMOKE_DIR / "nemotron_nano_30b_baseline_pretrain_smoke.yaml",
        nemotron_3_nano_pretrain_config,
    ),
    "midtrain": (
        _ARM_DIR / "nemotron_nano_30b_baseline_midtrain.yaml",
        _SMOKE_DIR / "nemotron_nano_30b_baseline_midtrain_smoke.yaml",
        nemotron_3_nano_pretrain_config,
    ),
    "sft": (
        _ARM_DIR / "nemotron_nano_30b_baseline_sft.yaml",
        _SMOKE_DIR / "nemotron_nano_30b_baseline_sft_smoke.yaml",
        nemotron_3_nano_sft_config,
    ),
}


@pytest.fixture(scope="module")
def smoke():
    return {name: merge_onto_recipe(spec[1], spec[2]) for name, spec in STAGES.items()}


@pytest.fixture(scope="module")
def parent():
    return {name: merge_onto_recipe(spec[0], spec[2]) for name, spec in STAGES.items()}


@pytest.fixture(scope="module")
def smoke_raw():
    return {name: OmegaConf.load(spec[1]) for name, spec in STAGES.items()}


@pytest.fixture(scope="module")
def parent_raw():
    return {name: OmegaConf.load(spec[0]) for name, spec in STAGES.items()}


@pytest.mark.parametrize("stage", sorted(STAGES))
class TestMatchesItsParent:
    """The half that must not diverge."""

    def test_data_source_is_identical(self, smoke_raw, parent_raw, stage):
        """A smoke run on a different mix certifies nothing about the real mix."""
        smoke_ds = OmegaConf.to_container(smoke_raw[stage].dataset)
        parent_ds = OmegaConf.to_container(parent_raw[stage].dataset)
        assert smoke_ds == parent_ds, f"{stage}: the smoke dataset block has drifted from its parent"

    def test_topology_and_memory_posture_are_identical(self, smoke, parent, stage):
        """Parallelism and recompute decide whether the real config FITS, which is the single
        most valuable thing a smoke run establishes."""
        s, p = smoke[stage].model, parent[stage].model
        for field in (
            "seq_length",
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "expert_model_parallel_size",
            "expert_tensor_parallel_size",
            "context_parallel_size",
            "recompute_granularity",
            "recompute_method",
            "recompute_num_layers",
            "cross_entropy_loss_fusion",
        ):
            assert getattr(s, field) == getattr(p, field), f"{stage}.model.{field}"

    def test_batch_shape_is_identical(self, smoke, parent, stage):
        """Tokens per iteration is what lets the smoke's s/iter extrapolate to the real run."""
        assert smoke[stage].train.global_batch_size == parent[stage].train.global_batch_size
        assert smoke[stage].train.micro_batch_size == parent[stage].train.micro_batch_size

    def test_save_crossing_settings_are_identical(self, smoke, parent, stage):
        """These are the DP=512 save-crossing fixes. A smoke that relaxed any of them would run
        clean and tell us nothing about the pathology they exist to prevent."""
        assert smoke[stage].checkpoint.ckpt_assume_constant_structure is False
        assert smoke[stage].model.cross_entropy_loss_fusion is False
        assert smoke[stage].dist.distributed_backend == parent[stage].dist.distributed_backend

    def test_learning_rate_peak_and_style_are_identical(self, smoke, parent, stage):
        """Warmup length may shrink with the run; the schedule being certified may not."""
        assert smoke[stage].optimizer.lr == parent[stage].optimizer.lr
        assert smoke[stage].optimizer.min_lr == parent[stage].optimizer.min_lr
        assert smoke[stage].scheduler.lr_decay_style == parent[stage].scheduler.lr_decay_style


@pytest.mark.parametrize("stage", sorted(STAGES))
class TestSmokeSpecifics:
    """The half that must diverge, and by exactly this much."""

    def test_budget_matches_the_smoke_iteration_count(self, smoke, stage):
        cfg = smoke[stage]
        assert cfg.train.train_iters == SMOKE_ITERS
        tokens = cfg.train.train_iters * cfg.train.global_batch_size * cfg.dataset.seq_length
        assert tokens == SMOKE_TOKENS, f"{stage}: {tokens:,} != {SMOKE_TOKENS:,}"

    def test_only_the_final_checkpoint_is_written(self, smoke, stage):
        """A `save_interval` above `train_iters` leaves Megatron-Core's unconditional
        end-of-training save as the only one, and nothing reads the optimizer or RNG state of a
        smoke checkpoint — the next stage warm-starts from weights."""
        ckpt = smoke[stage].checkpoint
        assert ckpt.save_interval > SMOKE_ITERS
        assert ckpt.save_optim is False
        assert ckpt.save_rng is False

    def test_writes_only_into_the_smoke_checkpoint_tree(self, smoke, stage):
        """Writing into the real run's directory would corrupt a 500B-token run."""
        ckpt = smoke[stage].checkpoint
        for field in ("load", "save"):
            path = getattr(ckpt, field)
            assert path.startswith(SMOKE_CKPT_ROOT), f"{stage}.checkpoint.{field} = {path}"

    def test_no_output_path_is_copied_from_the_parent(self, smoke_raw, parent_raw, stage):
        """Any path the smoke config STATES must differ from the parent's stated path.

        This is the failure mode of a config derived by copying: `tensorboard_dir` reached the
        SFT smoke still pointing at the production stage-3 directory, which CLAUDE.md's
        "TensorBoard on NFS" section records as a cause of cascading stale-file-handle crashes
        when two runs share one. Checkpoint paths were checked above; every other writable path
        needs the same guard, or the next copied field is found by the filesystem.

        Compares the RAW YAMLs, not the merged configs: a field neither file sets is a shared
        recipe default, which is a different (repo-wide) matter and not evidence of copying.
        """
        for section, field in (("logger", "tensorboard_dir"), ("logger", "wandb_save_dir")):
            mine = smoke_raw[stage].get(section, {}).get(field)
            theirs = parent_raw[stage].get(section, {}).get(field)
            if mine is None or theirs is None:
                continue
            assert mine != theirs, (
                f"{stage}.{section}.{field} is the parent's path verbatim ({mine}); "
                "a smoke run must not write into the real run's directories"
            )

    def test_wandb_run_is_distinguishable_from_the_real_one(self, smoke, parent, stage):
        name = smoke[stage].logger.wandb_exp_name
        assert name.startswith("smoke_e2e_")
        assert name != parent[stage].logger.wandb_exp_name


class TestTheChain:
    def test_each_stage_warm_starts_from_its_predecessor(self, smoke):
        """The chain is what no single-stage test covers: a shape or key mismatch between what
        one stage saves and the next loads only shows up here."""
        pre, mid, sft = smoke["pretrain"], smoke["midtrain"], smoke["sft"]
        assert pre.checkpoint.pretrained_checkpoint is None, "stage 1 trains from scratch"
        assert mid.checkpoint.pretrained_checkpoint == pre.checkpoint.save
        assert sft.checkpoint.pretrained_checkpoint == mid.checkpoint.save

    def test_the_three_stages_do_not_share_a_checkpoint_directory(self, smoke):
        saves = {name: cfg.checkpoint.save for name, cfg in smoke.items()}
        assert len(set(saves.values())) == 3, f"stages collide on disk: {saves}"

    def test_tokens_per_iteration_are_continuous_across_the_chain(self, smoke):
        """Stage 1 is GBS 2048 x 8192 and stages 2-3 are GBS 512 x 32768; both are 16,777,216,
        so one iteration count gives all three stages the same token budget."""
        for name, cfg in smoke.items():
            per_iter = cfg.train.global_batch_size * cfg.dataset.seq_length
            assert per_iter == TOKENS_PER_ITER, f"{name}: {per_iter:,}"

    def test_the_chain_totals_three_stage_budgets(self, smoke):
        total = sum(
            cfg.train.train_iters * cfg.train.global_batch_size * cfg.dataset.seq_length for cfg in smoke.values()
        )
        assert total == CHAIN_TOKENS, f"{total:,} != {CHAIN_TOKENS:,}"

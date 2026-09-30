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

"""The Nano pretrain quickstart is the production stage-1 posture, changed only where a benchmark must.

The quickstart's throughput numbers are evidence about the control-pretraining baseline only to the
extent that the two train the same thing: the same topology, recompute, communication posture,
optimizer, schedule and data. So the assertions come in two halves, as the smoke-run and ablation
tests' do. The set of fields that differ between the merged quickstart and the merged baseline must
equal exactly the benchmark's overlay — a batch sized to production's per-replica work, a 50-iteration
exit, no checkpoint I/O, and its own cache, timeout and W&B identity — and each of those fields must
satisfy the rule it exists for.

The campaign's performance overlay of the quickstart is held to the same standard one level up: it
may differ from the quickstart only in its levers and its W&B name.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from scripts.nemotronh_flops_estimator import RunSpec
from scripts.training.config_compose import BASE_CONFIG_KEY, load_composed_yaml

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_pretrain_config
from tests.unit_tests.campaign_config import (
    assert_only_these_fields_differ,
    data_parallel_size,
    dotted_leaves,
    merge_onto_recipe,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
QUICKSTART = _REPO_ROOT / "configs" / "quickstart" / "nemotron_nano_quickstart_pretrain.yaml"
BASELINE = _REPO_ROOT / "configs" / "control_pretraining" / "30b_baseline" / "nemotron_nano_30b_baseline_pretrain.yaml"

# The benchmark's overlay, and nothing else. Set equality, not containment: a field cannot start
# differing from the baseline without being named here.
ALLOWED_DIVERGENCE = {
    "train.global_batch_size",
    "train.exit_interval",
    "checkpoint.load",
    "checkpoint.save",
    "dataset.path_to_cache",
    "dist.distributed_timeout_minutes",
    "logger.wandb_save_dir",
    "logger.wandb_exp_name",
}
# The overlay also restates tensorboard_dir: null, which the baseline already sets, so it is written
# in the file without being a divergence.
RESTATED_FIELDS = {"logger.tensorboard_dir"}

# The width the baseline posture trains at in production (the filtered arm's 64-node stage 1), and
# the quickstart's two widths: 64 GPUs as written, 32 GPUs with the documented
# train.global_batch_size=256 override.
PRODUCTION_GPUS = 256
QUICKSTART_GPUS = 64
HALF_WIDTH_GPUS = 32
HALF_WIDTH_GLOBAL_BATCH_SIZE = 256
EXIT_ITERATION = 50

SHARED_STORAGE = Path("/projects/a5k/public")


@pytest.fixture(scope="module")
def quickstart():
    return merge_onto_recipe(QUICKSTART, nemotron_3_nano_pretrain_config)


@pytest.fixture(scope="module")
def baseline():
    return merge_onto_recipe(BASELINE, nemotron_3_nano_pretrain_config)


def microbatches_per_replica(cfg, global_batch_size: int, gpus: int) -> int:
    """Microbatches each data-parallel replica runs per iteration, asserting the batch divides evenly."""
    data_parallel = data_parallel_size(cfg, gpus)
    per_iteration = cfg.train.micro_batch_size * data_parallel
    assert global_batch_size % per_iteration == 0, f"GBS {global_batch_size} does not divide over DP={data_parallel}"
    return global_batch_size // per_iteration


def production_trees() -> list[Path]:
    """The directories production stage 1 writes into, read from the baseline so they follow it if it moves.

    Its checkpoint directory, and the directory of every corpus in its blend: with no
    ``path_to_cache`` set, Megatron writes each corpus's index cache under that corpus's prefix.
    """
    baseline = load_composed_yaml(BASELINE)
    prefixes = [Path(str(prefix)) for prefix in baseline["dataset"]["data_path"][1::2]]
    return [Path(baseline["checkpoint"]["save"]), *sorted({prefix.parent for prefix in prefixes})]


class TestOnlyTheBenchmarkFieldsDiffer:
    def test_exactly_the_overlay_fields_differ(self, quickstart, baseline):
        assert_only_these_fields_differ(quickstart, baseline, ALLOWED_DIVERGENCE, "nano pretrain quickstart")

    def test_the_file_is_an_overlay_of_the_baseline_not_a_copy(self):
        """A copied config would pass the comparison above today and drift from production tomorrow;
        naming the baseline as its base is what makes a production change reach the benchmark."""
        raw = OmegaConf.to_container(OmegaConf.load(QUICKSTART))
        base_ref = raw.pop(BASE_CONFIG_KEY)
        assert (QUICKSTART.parent / base_ref).resolve() == BASELINE.resolve()
        assert set(dotted_leaves(raw)) == ALLOWED_DIVERGENCE | RESTATED_FIELDS


class TestTheBenchmarkFields:
    def test_each_replica_runs_productions_microbatches_at_both_widths(self, quickstart, baseline):
        """Per-GPU step time follows microbatches per replica, not DP width, so matching it is what
        makes a 64- or 32-GPU step stand in for a production step."""
        production = microbatches_per_replica(baseline, baseline.train.global_batch_size, PRODUCTION_GPUS)
        assert production == 8
        assert (
            quickstart.train.global_batch_size == baseline.train.global_batch_size * QUICKSTART_GPUS // PRODUCTION_GPUS
        )
        assert microbatches_per_replica(quickstart, quickstart.train.global_batch_size, QUICKSTART_GPUS) == production
        assert microbatches_per_replica(quickstart, HALF_WIDTH_GLOBAL_BATCH_SIZE, HALF_WIDTH_GPUS) == production

    def test_the_run_exits_at_fifty_on_the_production_schedule(self, quickstart, baseline):
        """exit_interval, not train_iters: the warmup is a fraction of train_iters, so keeping
        train_iters keeps the production learning rate at every one of the 50 iterations."""
        assert quickstart.train.exit_interval == EXIT_ITERATION
        assert quickstart.train.train_iters == baseline.train.train_iters
        assert quickstart.scheduler.lr_warmup_fraction == baseline.scheduler.lr_warmup_fraction
        assert EXIT_ITERATION < quickstart.scheduler.lr_warmup_fraction * quickstart.train.train_iters

    def test_no_checkpoint_is_loaded_or_written(self, quickstart):
        """The baseline's load == save is the production run's directory: inherited, the benchmark
        would resume production instead of initialising randomly, and save into its tree."""
        assert quickstart.checkpoint.load is None
        assert quickstart.checkpoint.save is None
        assert quickstart.checkpoint.pretrained_checkpoint is None

    def test_the_index_cache_is_its_own(self, quickstart):
        cache = Path(quickstart.dataset.path_to_cache)
        assert cache.is_relative_to(SHARED_STORAGE)
        trees = production_trees()
        # The checkpoint directory plus at least one corpus directory: an empty derivation would
        # pass the loop below vacuously.
        assert len(trees) >= 2
        for tree in trees:
            assert not cache.is_relative_to(tree), f"{cache} is inside the production tree {tree}"

    def test_the_timeout_is_shorter_than_the_baselines(self, quickstart, baseline):
        assert quickstart.dist.distributed_timeout_minutes < baseline.dist.distributed_timeout_minutes

    def test_wandb_has_a_directory_and_a_name_of_its_own(self, quickstart, baseline):
        """With checkpoint.save null, W&B's default directory (<save>/wandb) does not exist."""
        assert Path(quickstart.logger.wandb_save_dir).is_relative_to(SHARED_STORAGE)
        assert quickstart.logger.wandb_exp_name != baseline.logger.wandb_exp_name

    def test_tensorboard_is_disabled(self, quickstart):
        assert quickstart.logger.tensorboard_dir is None


class TestTheFlopsEstimatorReadsTheOverlay:
    def test_the_workload_is_the_composed_config_the_launcher_trains(self, quickstart):
        """The quickstart's model TFLOP/s comes from this estimator, and the overlay file itself names
        no sequence length, parallelism or recompute: all of it must be read through the base."""
        spec = RunSpec.from_yaml(str(QUICKSTART))
        assert spec.global_batch_size == quickstart.train.global_batch_size == 512
        assert spec.seq_length == quickstart.model.seq_length == 8192
        assert spec.tokens_per_iter == 512 * 8192
        assert spec.recompute_granularity == quickstart.model.recompute_granularity
        assert spec.recompute_modules == tuple(quickstart.model.recompute_modules)
        assert spec.expert_model_parallel_size == quickstart.model.expert_model_parallel_size
        assert spec.context_parallel_size == quickstart.model.context_parallel_size


PERFORMANCE = _REPO_ROOT / "configs" / "quickstart" / "nemotron_nano_quickstart_pretrain_perf.yaml"
# The performance posture's levers at the values its runs were measured with. The overlay's identity field
# is not among them: it only names the W&B run.
PERFORMANCE_LEVERS = {
    "mixed_precision": "nemotron_h_bf16_with_fp8_current_scaling_bf16_params_bf16_grad_reduce",
    "model.recompute_modules": ["moe_act"],
    "model.moe_token_dispatcher_type": "flex",
    "model.moe_flex_dispatcher_backend": "hybridep",
    "model.mtp_num_layers": None,
    "model.moe_router_fusion": True,
    "model.cross_entropy_loss_fusion": True,
    "model.cross_entropy_fusion_impl": "linear",
    "model.cross_entropy_fusion_saved_logit_chunks": 8,
    "comm_overlap.overlap_param_gather": True,
    "comm_overlap.overlap_moe_expert_parallel_comm": True,
    "ddp.check_for_nan_in_grad": False,
    "rerun_state_machine.check_for_nan_in_loss": False,
    "train.manual_gc": True,
    "train.manual_gc_interval": 10,
    "train.manual_gc_freeze": True,
    "logger.timing_log_level": 1,
    "logger.log_l2_norm_grad_to_tensorboard": False,
}


class TestThePerformanceOverlay:
    """The performance posture is the quickstart plus its levers and nothing else.

    Compared as composed YAML rather than merged onto the recipe: on the pinned submodule, which the
    unit suite runs against, the merge drops ``model.cross_entropy_fusion_saved_logit_chunks``, a field
    only patch 0005 adds, so a merged comparison could not see it. The composed files are what the
    launcher merges, so a field that differs here is a field that differs in the run.
    """

    def test_it_overlays_the_quickstart_and_states_only_its_levers(self):
        raw = OmegaConf.to_container(OmegaConf.load(PERFORMANCE))
        base_ref = raw.pop(BASE_CONFIG_KEY)
        assert (PERFORMANCE.parent / base_ref).resolve() == QUICKSTART.resolve()
        assert set(dotted_leaves(raw)) == set(PERFORMANCE_LEVERS) | {"logger.wandb_exp_name"}

    def test_exactly_the_levers_and_the_run_name_differ_from_the_quickstart(self):
        performance = dotted_leaves(load_composed_yaml(PERFORMANCE))
        quickstart = dotted_leaves(load_composed_yaml(QUICKSTART))
        # An explicit null is a divergence from an absent key: mtp_num_layers null overrides the model
        # provider's default 0.
        absent = object()
        differing = {
            key
            for key in performance.keys() | quickstart.keys()
            if performance.get(key, absent) != quickstart.get(key, absent)
        }
        assert differing == set(PERFORMANCE_LEVERS) | {"logger.wandb_exp_name"}
        assert {key: performance[key] for key in PERFORMANCE_LEVERS} == PERFORMANCE_LEVERS
        assert performance["logger.wandb_exp_name"] != quickstart["logger.wandb_exp_name"]

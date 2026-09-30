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

"""The Nano pretrain baseline benchmark is the production stage-1 posture, changed only where a benchmark
must; the quickstart is that benchmark plus the performance campaign's levers.

The benchmark's throughput numbers are evidence about the control-pretraining baseline only to the
extent that the two train the same thing: the same topology, recompute, communication posture,
optimizer, schedule and data. So the assertions come in two halves, as the smoke-run and ablation
tests' do. The set of fields that differ between the merged benchmark and the merged baseline must
equal exactly the benchmark's overlay — a batch sized to production's per-replica work, a 50-iteration
exit, no checkpoint I/O, and its own cache, timeout and W&B identity — and each of those fields must
satisfy the rule it exists for.

The quickstart is held to the same standard one level up: it may differ from the benchmark only in its
levers and its W&B name, every lever must reach the merged config, and its env file must hold exactly the
launcher settings the levers need.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from scripts.nemotronh_flops_estimator import RunSpec
from scripts.training.config_compose import BASE_CONFIG_KEY, load_composed_yaml

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_pretrain_config
from tests.unit_tests.campaign_config import (
    FAST_PRETRAIN_LAUNCHER_SETTINGS,
    FAST_PRETRAIN_LEVERS,
    assert_levers_are_set,
    assert_only_these_fields_differ,
    data_parallel_size,
    dotted_leaves,
    merge_onto_recipe,
)
from tests.unit_tests.launcher_source import env_override_entries


_REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = _REPO_ROOT / "configs" / "quickstart" / "nemotron_nano_quickstart_pretrain_baseline.yaml"
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
# the benchmark's two widths: 64 GPUs as written, 32 GPUs with the documented
# train.global_batch_size=256 override.
PRODUCTION_GPUS = 256
BENCHMARK_GPUS = 64
HALF_WIDTH_GPUS = 32
HALF_WIDTH_GLOBAL_BATCH_SIZE = 256
EXIT_ITERATION = 50

SHARED_STORAGE = Path("/projects/a5k/public")


@pytest.fixture(scope="module")
def benchmark():
    return merge_onto_recipe(BENCHMARK, nemotron_3_nano_pretrain_config)


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
    def test_exactly_the_overlay_fields_differ(self, benchmark, baseline):
        assert_only_these_fields_differ(benchmark, baseline, ALLOWED_DIVERGENCE, "nano pretrain benchmark")

    def test_the_file_is_an_overlay_of_the_baseline_not_a_copy(self):
        """A copied config would pass the comparison above today and drift from production tomorrow;
        naming the baseline as its base is what makes a production change reach the benchmark."""
        raw = OmegaConf.to_container(OmegaConf.load(BENCHMARK))
        base_ref = raw.pop(BASE_CONFIG_KEY)
        assert (BENCHMARK.parent / base_ref).resolve() == BASELINE.resolve()
        assert set(dotted_leaves(raw)) == ALLOWED_DIVERGENCE | RESTATED_FIELDS


class TestTheBenchmarkFields:
    def test_each_replica_runs_productions_microbatches_at_both_widths(self, benchmark, baseline):
        """Per-GPU step time follows microbatches per replica, not DP width, so matching it is what
        makes a 64- or 32-GPU step stand in for a production step."""
        production = microbatches_per_replica(baseline, baseline.train.global_batch_size, PRODUCTION_GPUS)
        assert production == 8
        assert (
            benchmark.train.global_batch_size == baseline.train.global_batch_size * BENCHMARK_GPUS // PRODUCTION_GPUS
        )
        assert microbatches_per_replica(benchmark, benchmark.train.global_batch_size, BENCHMARK_GPUS) == production
        assert microbatches_per_replica(benchmark, HALF_WIDTH_GLOBAL_BATCH_SIZE, HALF_WIDTH_GPUS) == production

    def test_the_run_exits_at_fifty_on_the_production_schedule(self, benchmark, baseline):
        """exit_interval, not train_iters: the warmup is a fraction of train_iters, so keeping
        train_iters keeps the production learning rate at every one of the 50 iterations."""
        assert benchmark.train.exit_interval == EXIT_ITERATION
        assert benchmark.train.train_iters == baseline.train.train_iters
        assert benchmark.scheduler.lr_warmup_fraction == baseline.scheduler.lr_warmup_fraction
        assert EXIT_ITERATION < benchmark.scheduler.lr_warmup_fraction * benchmark.train.train_iters

    def test_no_checkpoint_is_loaded_or_written(self, benchmark):
        """The baseline's load == save is the production run's directory: inherited, the benchmark
        would resume production instead of initialising randomly, and save into its tree."""
        assert benchmark.checkpoint.load is None
        assert benchmark.checkpoint.save is None
        assert benchmark.checkpoint.pretrained_checkpoint is None

    def test_the_index_cache_is_its_own(self, benchmark):
        cache = Path(benchmark.dataset.path_to_cache)
        assert cache.is_relative_to(SHARED_STORAGE)
        trees = production_trees()
        # The checkpoint directory plus at least one corpus directory: an empty derivation would
        # pass the loop below vacuously.
        assert len(trees) >= 2
        for tree in trees:
            assert not cache.is_relative_to(tree), f"{cache} is inside the production tree {tree}"

    def test_the_timeout_is_shorter_than_the_baselines(self, benchmark, baseline):
        assert benchmark.dist.distributed_timeout_minutes < baseline.dist.distributed_timeout_minutes

    def test_wandb_has_a_directory_and_a_name_of_its_own(self, benchmark, baseline):
        """With checkpoint.save null, W&B's default directory (<save>/wandb) does not exist."""
        assert Path(benchmark.logger.wandb_save_dir).is_relative_to(SHARED_STORAGE)
        assert benchmark.logger.wandb_exp_name != baseline.logger.wandb_exp_name

    def test_tensorboard_is_disabled(self, benchmark):
        assert benchmark.logger.tensorboard_dir is None


class TestTheFlopsEstimatorReadsTheOverlay:
    def test_the_workload_is_the_composed_config_the_launcher_trains(self, benchmark):
        """The benchmark's model TFLOP/s comes from this estimator, and the overlay file itself names
        no sequence length, parallelism or recompute: all of it must be read through the base."""
        spec = RunSpec.from_yaml(str(BENCHMARK))
        assert spec.global_batch_size == benchmark.train.global_batch_size == 512
        assert spec.seq_length == benchmark.model.seq_length == 8192
        assert spec.tokens_per_iter == 512 * 8192
        assert spec.recompute_granularity == benchmark.model.recompute_granularity
        assert spec.recompute_modules == tuple(benchmark.model.recompute_modules)
        assert spec.expert_model_parallel_size == benchmark.model.expert_model_parallel_size
        assert spec.context_parallel_size == benchmark.model.context_parallel_size


QUICKSTART = _REPO_ROOT / "configs" / "quickstart" / "nemotron_nano_quickstart_pretrain.yaml"
# The ISAMBARD_ENV_OVERRIDES file the quickstart is launched with, which carries the fast posture's
# launcher settings.
QUICKSTART_ENV_FILE = QUICKSTART.with_suffix(".env")


@pytest.fixture(scope="module")
def quickstart():
    return merge_onto_recipe(QUICKSTART, nemotron_3_nano_pretrain_config)


class TestTheQuickstartIsTheBenchmarkPlusItsLevers:
    """The quickstart is the baseline benchmark plus its levers and nothing else."""

    def test_it_overlays_the_benchmark_and_states_only_its_levers(self):
        raw = OmegaConf.to_container(OmegaConf.load(QUICKSTART))
        base_ref = raw.pop(BASE_CONFIG_KEY)
        assert (QUICKSTART.parent / base_ref).resolve() == BENCHMARK.resolve()
        assert set(dotted_leaves(raw)) == set(FAST_PRETRAIN_LEVERS) | {"logger.wandb_exp_name"}

    def test_exactly_the_levers_and_the_run_name_differ_from_the_benchmark(self):
        """Compared as composed YAML, the files the launcher merges: the levers are the file's own
        fields (a precision preset's name, not the fields it resolves to)."""
        quickstart_fields = dotted_leaves(load_composed_yaml(QUICKSTART))
        benchmark_fields = dotted_leaves(load_composed_yaml(BENCHMARK))
        # An explicit null is a divergence from an absent key: mtp_num_layers null overrides the model
        # provider's default 0.
        absent = object()
        differing = {
            key
            for key in quickstart_fields.keys() | benchmark_fields.keys()
            if quickstart_fields.get(key, absent) != benchmark_fields.get(key, absent)
        }
        assert differing == set(FAST_PRETRAIN_LEVERS) | {"logger.wandb_exp_name"}
        assert {key: quickstart_fields[key] for key in FAST_PRETRAIN_LEVERS} == FAST_PRETRAIN_LEVERS
        assert quickstart_fields["logger.wandb_exp_name"] != benchmark_fields["logger.wandb_exp_name"]

    def test_every_lever_reaches_the_merged_config(self, quickstart):
        assert_levers_are_set(quickstart, FAST_PRETRAIN_LEVERS, "nano pretrain quickstart")

    def test_its_env_file_holds_exactly_the_launcher_settings_it_needs(self):
        """Read through the launcher's own hook. Launched without the file the quickstart still runs, but on
        one CUDA connection the EP overlap's second stream serialises behind the first and the overlap is
        slower than none."""
        assert env_override_entries(str(QUICKSTART_ENV_FILE)) == FAST_PRETRAIN_LAUNCHER_SETTINGS

    def test_each_posture_has_its_own_wandb_name(self, quickstart, benchmark):
        """Runs are compared by W&B name, so a name labels one posture, whatever the files are called."""
        assert benchmark.logger.wandb_exp_name == "nemotron_nano_quickstart_pretrain"
        assert quickstart.logger.wandb_exp_name == "nemotron_nano_quickstart_pretrain_perf"

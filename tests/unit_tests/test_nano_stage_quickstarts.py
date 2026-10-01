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

"""Each Nano baseline benchmark is a production stage of the control-pretraining baseline, changed only where a
benchmark must; each quickstart is its stage's benchmark plus the levers of that stage's performance campaign.

A benchmark's throughput numbers are evidence about its production stage only to the extent that the two train
the same thing: the same topology, recompute, communication posture, optimizer, schedule, initialisation and data.
So the assertions come in two halves, as the smoke-run and ablation tests' do. The set of fields that differ
between the merged benchmark and the merged production stage must equal exactly the benchmark's overlay — a batch
sized to production's per-replica work, an exit before the end of the schedule, no checkpoint I/O of
production's, and its own cache, timeout and W&B identity — and each of those fields must satisfy the rule it
exists for.

A quickstart is held to the same standard one level up: it may differ from its benchmark only in its levers and
its W&B name, every lever must reach the merged config, and its env file must hold exactly the launcher settings
the levers need.
"""

from __future__ import annotations

from dataclasses import dataclass
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
_QUICKSTARTS = _REPO_ROOT / "configs" / "quickstart"
_BASELINE_STAGES = _REPO_ROOT / "configs" / "control_pretraining" / "30b_baseline"

# A benchmark's overlay, and nothing else. Set equality, not containment: a field cannot start differing from
# the production stage without being named here.
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
# A benchmark also restates tensorboard_dir: null, which every production stage already sets, so it is written
# in the file without being a divergence.
RESTATED_FIELDS = {"logger.tensorboard_dir"}

SHARED_STORAGE = Path("/projects/a5k/public")


@dataclass(frozen=True)
class BenchmarkStage:
    """A production stage of the control-pretraining baseline and the baseline benchmark built on it."""

    name: str
    production: Path
    benchmark: Path
    # The width the stage trains at in production and the microbatches per data-parallel replica it runs
    # there: the per-GPU work every width of the benchmark must reproduce.
    production_gpus: int
    microbatches_per_replica: int
    benchmark_gpus: int
    # Other widths the benchmark's header documents, as (GPUs, the train.global_batch_size override they take).
    override_widths: tuple[tuple[int, int], ...]
    exit_iteration: int
    seq_length: int
    # Whether production starts the stage from an earlier stage's checkpoint (pretrained_checkpoint).
    warm_start: bool
    wandb_name: str


PRETRAIN = BenchmarkStage(
    name="pretrain",
    production=_BASELINE_STAGES / "nemotron_nano_30b_baseline_pretrain.yaml",
    benchmark=_QUICKSTARTS / "nemotron_nano_quickstart_pretrain_baseline.yaml",
    # The filtered arm's 64-node stage 1.
    production_gpus=256,
    microbatches_per_replica=8,
    benchmark_gpus=64,
    override_widths=((32, 256),),
    exit_iteration=50,
    seq_length=8192,
    warm_start=False,
    wandb_name="nemotron_nano_quickstart_pretrain",
)
MIDTRAIN = BenchmarkStage(
    name="midtrain",
    production=_BASELINE_STAGES / "nemotron_nano_30b_baseline_midtrain.yaml",
    benchmark=_QUICKSTARTS / "nemotron_nano_quickstart_midtrain_baseline.yaml",
    production_gpus=512,
    microbatches_per_replica=2,
    benchmark_gpus=64,
    override_widths=(),
    exit_iteration=200,
    seq_length=32768,
    warm_start=True,
    wandb_name="nemotron_nano_quickstart_midtrain",
)
STAGES = [PRETRAIN, MIDTRAIN]


@dataclass(frozen=True)
class Quickstart:
    """A stage's quickstart: its baseline benchmark plus the levers of the stage's performance campaign."""

    stage: BenchmarkStage
    path: Path
    # The levers at the values the campaign's runs measured. The W&B name is not among them: it only names the
    # run.
    levers: dict[str, object]
    # The launcher settings the levers need, which a training YAML cannot carry: the ISAMBARD_ENV_OVERRIDES file
    # the quickstart is launched with, beside it with the .env suffix.
    launcher_settings: list[str]
    wandb_name: str


PRETRAIN_QUICKSTART = Quickstart(
    stage=PRETRAIN,
    path=_QUICKSTARTS / "nemotron_nano_quickstart_pretrain.yaml",
    levers=FAST_PRETRAIN_LEVERS,
    # Launched without the file the quickstart still runs, but on one CUDA connection the EP overlap's second
    # stream serialises behind the first and the overlap is slower than none.
    launcher_settings=FAST_PRETRAIN_LAUNCHER_SETTINGS,
    wandb_name="nemotron_nano_quickstart_pretrain_perf",
)
MIDTRAIN_QUICKSTART = Quickstart(
    stage=MIDTRAIN,
    path=_QUICKSTARTS / "nemotron_nano_quickstart_midtrain.yaml",
    levers={
        "mixed_precision": "nemotron_h_bf16_with_fp8_current_scaling_bf16_params_bf16_grad_reduce",
        "model.recompute_granularity": "selective",
        "model.recompute_method": None,
        "model.recompute_num_layers": None,
        "model.recompute_modules": ["moe", "shared_experts"],
        "model.moe_token_dispatcher_type": "flex",
        "model.moe_flex_dispatcher_backend": "hybridep",
        "model.moe_router_fusion": True,
        "model.cross_entropy_loss_fusion": True,
        "model.cross_entropy_fusion_impl": "linear",
        "model.cross_entropy_fusion_saved_logit_chunks": 8,
        "comm_overlap.overlap_param_gather": True,
        "rerun_state_machine.check_for_nan_in_loss": False,
        "train.manual_gc": True,
        "train.manual_gc_interval": 10,
        "train.manual_gc_freeze": True,
        "logger.timing_log_level": 1,
        "logger.log_l2_norm_grad_to_tensorboard": False,
    },
    # At seq 32768 a bf16 inter-chunk SSM state overflows on long single documents; stated so that a value inherited
    # from the environment (the pretraining quickstart's env file sets 0) cannot switch the fp32 state off.
    launcher_settings=["ISAMBARD_FP32_SSM_STATE=checkpoint"],
    wandb_name="nemotron_nano_quickstart_midtrain_perf",
)
QUICKSTARTS = [PRETRAIN_QUICKSTART, MIDTRAIN_QUICKSTART]


def merged(path: Path):
    """The config the launcher trains for the YAML at ``path``: both stages launch in pretrain mode."""
    return merge_onto_recipe(path, nemotron_3_nano_pretrain_config)


@pytest.fixture(scope="module", params=STAGES, ids=[stage.name for stage in STAGES])
def stage(request) -> BenchmarkStage:
    return request.param


@pytest.fixture(scope="module")
def benchmark(stage):
    return merged(stage.benchmark)


@pytest.fixture(scope="module")
def production(stage):
    return merged(stage.production)


def microbatches_per_replica(cfg, global_batch_size: int, gpus: int) -> int:
    """Microbatches each data-parallel replica runs per iteration, asserting the batch divides evenly."""
    data_parallel = data_parallel_size(cfg, gpus)
    per_iteration = cfg.train.micro_batch_size * data_parallel
    assert global_batch_size % per_iteration == 0, f"GBS {global_batch_size} does not divide over DP={data_parallel}"
    return global_batch_size // per_iteration


def production_output_directories(production_config: Path) -> list[Path]:
    """A production stage's checkpoint directory and, when it names one, its index cache, read from its config
    so they follow it if it moves."""
    production = load_composed_yaml(production_config)
    cache = production["dataset"].get("path_to_cache")
    return [Path(production["checkpoint"]["save"]), *([Path(cache)] if cache is not None else [])]


def corpus_directories(production_config: Path) -> list[Path]:
    """The directory of every corpus in a production stage's blend: with no ``path_to_cache``, Megatron writes each
    corpus's index cache under that corpus's prefix."""
    production = load_composed_yaml(production_config)
    return sorted({Path(str(prefix)).parent for prefix in production["dataset"]["data_path"][1::2]})


class TestOnlyTheBenchmarkFieldsDiffer:
    def test_exactly_the_overlay_fields_differ(self, stage, benchmark, production):
        assert_only_these_fields_differ(benchmark, production, ALLOWED_DIVERGENCE, f"nano {stage.name} benchmark")

    def test_the_file_is_an_overlay_of_the_production_stage_not_a_copy(self, stage):
        """A copied config would pass the comparison above today and drift from production tomorrow; naming the
        production stage as its base is what makes a production change reach the benchmark."""
        raw = OmegaConf.to_container(OmegaConf.load(stage.benchmark))
        base_ref = raw.pop(BASE_CONFIG_KEY)
        assert (stage.benchmark.parent / base_ref).resolve() == stage.production.resolve()
        assert set(dotted_leaves(raw)) == ALLOWED_DIVERGENCE | RESTATED_FIELDS


class TestTheBenchmarkFields:
    def test_each_replica_runs_productions_microbatches_at_every_width(self, stage, benchmark, production):
        """Per-GPU step time follows microbatches per replica, not DP width, so matching it is what makes a
        benchmark step stand in for a production step."""
        assert (
            microbatches_per_replica(production, production.train.global_batch_size, stage.production_gpus)
            == stage.microbatches_per_replica
        )
        assert (
            benchmark.train.global_batch_size
            == production.train.global_batch_size * stage.benchmark_gpus // stage.production_gpus
        )
        widths = [(stage.benchmark_gpus, benchmark.train.global_batch_size), *stage.override_widths]
        for gpus, global_batch_size in widths:
            assert microbatches_per_replica(benchmark, global_batch_size, gpus) == stage.microbatches_per_replica

    def test_the_run_exits_early_on_the_production_schedule(self, stage, benchmark, production):
        """exit_interval, not train_iters: the learning-rate schedule is a function of train_iters, so keeping
        production's keeps production's learning rate at every iteration the benchmark runs."""
        assert benchmark.train.exit_interval == stage.exit_iteration
        assert benchmark.train.train_iters == production.train.train_iters
        assert stage.exit_iteration < production.train.train_iters

    def test_it_starts_where_production_starts_and_writes_nothing(self, stage, benchmark, production):
        """Production's load == save is its own run directory: inherited, the benchmark would resume production's
        latest checkpoint instead of starting the stage, and save into its tree. What production starts from —
        nothing for stage 1, the stage-1 final weights for stage 2 — the benchmark starts from too."""
        assert benchmark.checkpoint.load is None
        assert benchmark.checkpoint.save is None
        assert benchmark.checkpoint.pretrained_checkpoint == production.checkpoint.pretrained_checkpoint
        assert (production.checkpoint.pretrained_checkpoint is not None) == stage.warm_start

    def test_the_index_cache_is_its_own(self, stage, benchmark):
        cache = Path(benchmark.dataset.path_to_cache)
        assert cache.is_relative_to(SHARED_STORAGE)
        corpora = corpus_directories(stage.production)
        # An empty derivation would pass the loop below vacuously.
        assert corpora
        for tree in [*production_output_directories(stage.production), *corpora]:
            assert not cache.is_relative_to(tree), f"{cache} is inside the production tree {tree}"

    def test_the_timeout_is_shorter_than_productions(self, benchmark, production):
        assert benchmark.dist.distributed_timeout_minutes < production.dist.distributed_timeout_minutes

    def test_wandb_has_a_directory_and_a_name_of_its_own(self, stage, benchmark, production):
        """With checkpoint.save null, W&B's default directory (<save>/wandb) does not exist."""
        assert Path(benchmark.logger.wandb_save_dir).is_relative_to(SHARED_STORAGE)
        assert benchmark.logger.wandb_exp_name == stage.wandb_name != production.logger.wandb_exp_name

    def test_tensorboard_is_disabled(self, benchmark):
        assert benchmark.logger.tensorboard_dir is None


class TestTheFlopsEstimatorReadsTheOverlay:
    def test_the_workload_is_the_composed_config_the_launcher_trains(self, stage, benchmark):
        """The benchmark's model TFLOP/s comes from this estimator, and the overlay file itself names no
        sequence length, parallelism or recompute: all of it must be read through the base."""
        spec = RunSpec.from_yaml(str(stage.benchmark))
        assert spec.global_batch_size == benchmark.train.global_batch_size
        assert spec.seq_length == benchmark.model.seq_length == stage.seq_length
        assert spec.tokens_per_iter == benchmark.train.global_batch_size * stage.seq_length
        assert spec.recompute_granularity == benchmark.model.recompute_granularity
        # Recompute modules are read only under selective granularity; full recompute re-runs every layer's
        # forward whatever they list.
        if benchmark.model.recompute_granularity == "selective":
            assert spec.recompute_modules == tuple(benchmark.model.recompute_modules)
        assert spec.expert_model_parallel_size == benchmark.model.expert_model_parallel_size
        assert spec.context_parallel_size == benchmark.model.context_parallel_size


@pytest.fixture(scope="module", params=QUICKSTARTS, ids=[quickstart.stage.name for quickstart in QUICKSTARTS])
def quickstart_case(request) -> Quickstart:
    return request.param


@pytest.fixture(scope="module")
def quickstart(quickstart_case):
    return merged(quickstart_case.path)


class TestTheQuickstartIsTheBenchmarkPlusItsLevers:
    """A quickstart is its stage's baseline benchmark plus its levers and nothing else."""

    def test_it_overlays_the_benchmark_and_states_only_its_levers(self, quickstart_case):
        raw = OmegaConf.to_container(OmegaConf.load(quickstart_case.path))
        base_ref = raw.pop(BASE_CONFIG_KEY)
        assert (quickstart_case.path.parent / base_ref).resolve() == quickstart_case.stage.benchmark.resolve()
        assert set(dotted_leaves(raw)) == set(quickstart_case.levers) | {"logger.wandb_exp_name"}

    def test_exactly_the_levers_and_the_run_name_differ_from_the_benchmark(self, quickstart_case):
        """Compared as composed YAML, the files the launcher merges: the levers are the file's own fields (a
        precision preset's name, not the fields it resolves to)."""
        quickstart_fields = dotted_leaves(load_composed_yaml(quickstart_case.path))
        benchmark_fields = dotted_leaves(load_composed_yaml(quickstart_case.stage.benchmark))
        # An explicit null is a divergence from an absent key: mtp_num_layers null overrides the model provider's
        # default 0.
        absent = object()
        differing = {
            key
            for key in quickstart_fields.keys() | benchmark_fields.keys()
            if quickstart_fields.get(key, absent) != benchmark_fields.get(key, absent)
        }
        assert differing == set(quickstart_case.levers) | {"logger.wandb_exp_name"}
        assert {key: quickstart_fields[key] for key in quickstart_case.levers} == quickstart_case.levers

    def test_every_lever_reaches_the_merged_config(self, quickstart_case, quickstart):
        """The launcher's merge drops a key the config classes lack, so each lever is read back from the merged
        config: a misspelled field, or one the pinned Megatron-LM does not have, would not arrive."""
        assert_levers_are_set(quickstart, quickstart_case.levers, quickstart_case.path.name)

    def test_its_env_file_holds_exactly_the_launcher_settings_it_needs(self, quickstart_case):
        """Read through the launcher's own hook."""
        env_file = quickstart_case.path.with_suffix(".env")
        assert env_override_entries(str(env_file)) == quickstart_case.launcher_settings

    def test_it_has_a_wandb_name_of_its_own(self, quickstart_case, quickstart):
        """Runs are compared by W&B name, so a name labels one posture, whatever the files are called."""
        assert quickstart.logger.wandb_exp_name == quickstart_case.wandb_name != quickstart_case.stage.wandb_name

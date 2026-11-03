# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""The Olmo-Norm SFT configs state AI2's Think SFT recipe, and the 32B config differs from the 7B in what its size needs.

The two runs fine-tune Olmo 3 7B and 32B for one pass over the Normal-Norm mix. What the report compares with AI2's
Think models is the recipe, so these tests pin its fields: the learning rate and schedule, the optimizer, the sequence
length, the packing and the batch (the paper's 1,048,576 tokens for the 7B and four times that for the 32B, Kyle's
choice). They also pin what makes the two runs different runs (size, batch, layout, names), that the packed files a
config reads are the ones its tokenizer and padding multiple build, and that train_iters follows from the pack audit.
"""

from __future__ import annotations

import hashlib
import math
from itertools import islice
from pathlib import Path

import pytest
import yaml
from scripts.checkpoint import export_check
from scripts.dotted_paths import dotted_leaves
from scripts.telemetry.run_watch import load_watch_spec

from megatron.bridge.data.loaders import cyclic_iter
from megatron.bridge.data.samplers import build_pretraining_data_loader
from pipeline.data.layout import pack_dir
from pipeline.training.recipes import OLMO3_BASE_COPIES, RECIPE_MAP
from tests.unit_tests.campaign_config import (
    assert_it_saves_to_a_directory_of_its_own,
    assert_segment_exit_posture,
    launcher_overrides,
    merge_onto_recipe,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARM_DIR = _REPO_ROOT / "configs" / "olmo3_norm_sft"
_CHECK_DIR = _ARM_DIR / "checks"
_TOKENIZER_DIR = Path("/projects/a5k/public/logs/metagaming_filtering/olmo_norm/tokenizers")
_SHARDS = "/projects/a5k/public/data/geodesic-research__pa-warm-start-sft-xl-50b-mix-quality-filtered__train/shard*"

SEQ_LENGTH = 32768
PAD_SEQ_TO_MULT = 8
#: The context-parallel size of the plain layout. The layout probes may fix another one: then it changes here and in
#: both configs together.
CONTEXT_PARALLEL_SIZE = 1

#: Per run: the model key of RECIPE_MAP, the batch in sequences, the pipeline size, the cross-entropy implementation
#: (the recipe's `native` where it fits, `te` at the 7B, where it does not; see the 7B config header), the packs the pack audit counted
#: (verdict PASS, report audit_<size>.json under /projects/a5k/public/logs/metagaming_filtering/olmo_norm/megatron/
#: full_audit), the first and last of the tail packs that the loader never reads, the tokenizer directory (the Think-SFT files with tokenizer_class changed to PreTrainedTokenizerFast,
#: because transformers 5.3.0 builds a wrong pre-tokenizer for GPT2Tokenizer) and the template that directory must hold
#: (sha256 of its chat_template.jinja).
RUNS = {
    "olmo3_7b_norm_sft": {
        "model": "olmo3_7b",
        "batch": 32,
        "pipeline": 1,
        "cross_entropy_impl": "te",
        "packs": 1_454_217,
        "unread_packs": (1_454_208, 1_454_216),
        "tokenizer": _TOKENIZER_DIR / "tokenizer_7b_fast",
        "template_sha256": "2f3360d43e167aff4c4c56df8ecc989384788c104b45c4fe5283599e07b391c0",
    },
    "olmo3_32b_norm_sft": {
        "model": "olmo3_32b",
        "batch": 512,
        "pipeline": 4,
        "cross_entropy_impl": "native",
        "packs": 1_451_951,
        "unread_packs": (1_451_520, 1_451_950),
        "tokenizer": _TOKENIZER_DIR / "tokenizer_32b_fast",
        "template_sha256": "1964b6c85aa3d7e74d2132887aa526cc553b1c4aa5100646599cee041fb1360b",
    },
}

#: What makes the 32B run another run than the 7B: its names and paths, its batch, its layout, its cross-entropy
#: implementation (a memory lever, which the 32B config leaves at the recipe's), its length and the tokenizer directory
#: that holds its template.
SIZE_FIELDS = {
    "tokenizer.tokenizer_model",
    "dataset.packed_sequence_specs.packed_train_data_path",
    "train.global_batch_size",
    "train.train_iters",
    "model.pipeline_model_parallel_size",
    "model.cross_entropy_fusion_impl",
    "checkpoint.pretrained_checkpoint",
    "checkpoint.load",
    "checkpoint.save",
    "checkpoint.save_interval",
    "logger.wandb_exp_name",
}


def run_path(run: str) -> Path:
    return _ARM_DIR / f"{run}.yaml"


def merged(run: str):
    """The run's config as the launcher resolves it: the family recipe with the YAML merged on."""
    model = RUNS[run]["model"]
    return merge_onto_recipe(run_path(run), lambda: RECIPE_MAP[(model, "sft")](None))


needs_base_copy = pytest.mark.skipif(
    not all((Path(path) / "config.json").is_file() for path in OLMO3_BASE_COPIES.values()),
    reason="needs the base copies' config.json (the recipe sizes the model from it)",
)


@needs_base_copy
@pytest.mark.parametrize("run", RUNS)
class TestRecipe:
    """AI2's Think SFT recipe, at the fields a report compares."""

    def test_the_optimizer_and_schedule_are_ai2s(self, run):
        cfg = merged(run)
        assert cfg.optimizer.lr == 5.0e-5 and cfg.optimizer.min_lr == 0.0
        assert (cfg.optimizer.adam_beta1, cfg.optimizer.adam_beta2, cfg.optimizer.adam_eps) == (0.9, 0.95, 1.0e-8)
        assert cfg.optimizer.clip_grad == 1.0
        assert cfg.scheduler.lr_decay_style == "linear"
        assert cfg.scheduler.lr_warmup_fraction == 0.03 and not cfg.scheduler.lr_warmup_iters

    def test_the_weight_decay_is_zero_where_the_scheduler_would_overwrite_it(self, run):
        cfg = merged(run)
        assert cfg.optimizer.weight_decay == 0.0
        assert cfg.scheduler.start_weight_decay == 0.0 and cfg.scheduler.end_weight_decay == 0.0

    def test_the_posture_is_ai2s_bf16_parameters_fp32_gradient_reduction_and_fp32_moments(self, run):
        cfg = merged(run)
        assert cfg.mixed_precision == "bf16_mixed"
        assert not cfg.optimizer.use_precision_aware_optimizer
        assert cfg.optimizer.use_distributed_optimizer and cfg.ddp.use_distributed_optimizer

    def test_the_cross_entropy_is_the_te_one_at_the_7b_and_the_recipes_native_one_at_the_32b(self, run):
        cfg = merged(run)
        assert cfg.model.cross_entropy_loss_fusion is True
        assert cfg.model.cross_entropy_fusion_impl == RUNS[run]["cross_entropy_impl"]

    def test_the_sequence_length_and_packing_are_32768_with_documents_padded_to_8(self, run):
        cfg = merged(run)
        specs = cfg.dataset.packed_sequence_specs
        assert cfg.model.seq_length == cfg.dataset.seq_length == specs.packed_sequence_size == SEQ_LENGTH
        assert specs.pad_seq_to_mult == PAD_SEQ_TO_MULT
        assert cfg.dataset.dataset_kwargs["chat"] and cfg.dataset.dataset_kwargs["use_hf_tokenizer_chat_template"]
        assert cfg.dataset.dataset_kwargs["answer_only_loss"]

    def test_the_family_settings_the_recipe_needs_survive_the_yaml(self, run):
        cfg = merged(run)
        assert cfg.model.tensor_model_parallel_size == 1  # the full-width QK-norm refuses TP > 1
        assert cfg.model.apply_rope_fusion is False  # the fused THD rope drops the YaRN scale
        assert cfg.model.calculate_per_token_loss is True
        assert cfg.model.cp_comm_type == "a2a"  # the only exchange that takes THD with a window

    def test_the_batch_is_the_papers_for_the_7b_and_four_times_that_for_the_32b(self, run):
        cfg = merged(run)
        assert cfg.train.micro_batch_size == 1
        assert cfg.train.global_batch_size == RUNS[run]["batch"]
        assert cfg.model.pipeline_model_parallel_size == RUNS[run]["pipeline"]
        tokens_per_iteration = cfg.train.global_batch_size * SEQ_LENGTH
        assert tokens_per_iteration == {"olmo3_7b_norm_sft": 1_048_576, "olmo3_32b_norm_sft": 16_777_216}[run]

    def test_train_iters_is_the_audited_packs_over_the_batch_rounded_up(self, run):
        cfg = merged(run)
        assert cfg.train.train_iters == math.ceil(RUNS[run]["packs"] / RUNS[run]["batch"])

    def test_the_context_parallel_size_is_the_one_the_layout_probes_fixed(self, run):
        """The 5e-5 learning rate of the CLAUDE.md exception holds at this size. A later override fails here, so that the
        change is a decision."""
        assert merged(run).model.context_parallel_size == CONTEXT_PARALLEL_SIZE

    def test_a_save_every_twenty_percent_of_the_iterations_and_the_last_at_the_end(self, run):
        cfg = merged(run)
        assert cfg.checkpoint.save_interval == math.ceil(cfg.train.train_iters / 5)
        assert cfg.checkpoint.most_recent_k == -1  # every save is kept
        assert cfg.checkpoint.save_optim and cfg.checkpoint.save_rng

    def test_it_ends_on_the_clock_under_the_queue_wall_and_writes_to_a_directory_of_its_own(self, run):
        cfg = merged(run)
        assert_segment_exit_posture(cfg, run, 1400)
        others = [merged(other) for other in RUNS if other != run]
        assert_it_saves_to_a_directory_of_its_own(cfg, run, others)

    def test_it_warm_starts_from_its_size_s_base_copy_import(self, run):
        cfg = merged(run)
        assert Path(cfg.checkpoint.pretrained_checkpoint).name == Path(OLMO3_BASE_COPIES[RUNS[run]["model"]]).name

    def test_tensorboard_is_off(self, run):
        assert merged(run).logger.tensorboard_dir is None


class _PackCount:
    """A dataset that has a length and nothing else.

    A stand-in, because the sampler needs only the pack count, and reading the 1.45 million packs is not what the test
    checks.
    """

    def __init__(self, packs: int) -> None:
        self.packs = packs

    def __len__(self) -> int:
        return self.packs


@pytest.mark.parametrize("run", RUNS)
def test_the_loader_never_reads_the_tail_packs_and_the_last_iteration_reads_the_first_batch_again(run):
    """train_iters is the pack count over the batch, rounded up. The loader drops the partial last batch of an epoch and
    then restarts at pack 0, so the last iteration reads the first batch again and the last packs mod batch packs are
    never read."""
    packs, batch = RUNS[run]["packs"], RUNS[run]["batch"]
    train_iters = launcher_overrides(run_path(run))["train"]["train_iters"]
    # The sampler comes out of the builder that a run uses, with its drop_last default.
    loader = build_pretraining_data_loader(
        _PackCount(packs), 0, "batch", 1, 0, False, pin_memory=False, global_batch_size=batch
    )
    iterations = list(islice(cyclic_iter(loader.batch_sampler), train_iters))
    assert len(iterations) == train_iters == packs // batch + 1
    read_once = [index for indices in iterations[:-1] for index in indices]
    unread = packs - len(read_once)
    assert read_once == list(range(len(read_once)))
    assert (len(read_once), packs - 1) == RUNS[run]["unread_packs"]
    assert unread == packs % batch > 0
    assert iterations[-1] == list(range(batch))


@pytest.mark.parametrize("run", RUNS)
def test_the_packed_files_are_the_ones_its_tokenizer_and_padding_multiple_build(run):
    overrides = launcher_overrides(run_path(run))
    tokenizer = overrides["tokenizer"]["tokenizer_model"]
    expected = pack_dir(_SHARDS, tokenizer, PAD_SEQ_TO_MULT) / f"training_{SEQ_LENGTH}.idx.parquet"
    assert overrides["dataset"]["packed_sequence_specs"]["packed_train_data_path"] == str(expected)
    assert Path(tokenizer) == RUNS[run]["tokenizer"]


@pytest.mark.skipif(not _TOKENIZER_DIR.is_dir(), reason="needs the cluster's tokenizer directories")
@pytest.mark.parametrize("run", RUNS)
def test_the_tokenizer_directory_holds_the_pinned_chat_template(run):
    template = RUNS[run]["tokenizer"] / "chat_template.jinja"
    assert hashlib.sha256(template.read_bytes()).hexdigest() == RUNS[run]["template_sha256"]


def test_the_two_runs_differ_only_in_what_their_size_needs():
    seven, thirty_two = (launcher_overrides(run_path(run)) for run in RUNS)
    leaves = [dotted_leaves(overrides) for overrides in (seven, thirty_two)]
    assert set(leaves[0]) == set(leaves[1])
    assert {key for key in leaves[0] if leaves[0][key] != leaves[1][key]} == SIZE_FIELDS


def test_the_7b_watch_stops_the_run_on_a_non_finite_step_a_skipped_one_or_an_allocator_retry():
    """The TE cross-entropy has known stability issues, so the watch is what lets the report say the run ran under it."""
    spec = load_watch_spec(_ARM_DIR / "watch_7b.yaml")
    assert spec.stop_on_non_finite_grad_norm and spec.stop_on_non_finite_lm_loss
    assert spec.stop_on_nan_or_skipped_iterations and spec.stop_on_rejected_result
    assert spec.max_alloc_retries == 0
    assert spec.launch_settings is None


#: The export-check spec of each run. The 32B spec composes onto the 7B one (``base_config``) and states these keys
#: itself: the snapshot it is compared with, the tokenizer directory, and its own calibration and tolerance (a
#: tolerance holds only for the spec it was calibrated for).
SPEC_OF_RUN = {
    "olmo3_7b_norm_sft": _CHECK_DIR / "c3_export_check_7b.yaml",
    "olmo3_32b_norm_sft": _CHECK_DIR / "c3_export_check_32b.yaml",
}
KEYS_OF_THE_32B_SPEC = {"base_config", "reference_export", "tokenizer", "criteria", "calibration"}


@pytest.fixture(scope="module")
def specs():
    return {run: export_check.load_spec(path) for run, path in SPEC_OF_RUN.items()}


@pytest.mark.parametrize("run", RUNS)
def test_each_export_check_spec_loads_with_its_windows_listed_as_selected(specs, run):
    spec = specs[run]
    windows = spec.listed("documents")
    assert len(windows) == sum(spec.selection.document_counts.values())
    assert {window.tokens for window in windows} == {spec.selection.window_tokens}
    assert not spec.prompts and spec.criteria.hidden_probability_max is None


@pytest.mark.parametrize("run", RUNS)
def test_each_export_check_spec_reads_the_tokenizer_directory_of_its_run(specs, run):
    tokenizer = launcher_overrides(run_path(run))["tokenizer"]["tokenizer_model"]
    assert specs[run].tokenizer == tokenizer


def test_the_7b_tolerance_passes_the_correct_pair_and_fails_the_negative_control_on_every_measure(specs):
    spec = specs["olmo3_7b_norm_sft"]
    tolerance = spec.criteria.ce_agreement
    runs = {run.expect: run.record for run in export_check.calibration_runs(spec)}
    measures = set(tolerance.limits())
    assert set(runs) == {"PASS", "FAIL"}
    assert tolerance.beyond({key: runs["PASS"][key] for key in measures}) == {}
    assert set(tolerance.beyond({key: runs["FAIL"][key] for key in measures})) == measures


def test_the_32b_spec_keeps_the_7b_corpus_windows_and_reference_and_changes_only_its_own_keys(specs):
    seven, thirty_two = (specs[run] for run in RUNS)
    assert set(yaml.safe_load(SPEC_OF_RUN["olmo3_32b_norm_sft"].read_text())) == KEYS_OF_THE_32B_SPEC
    for kept in ("corpora", "selection", "documents", "megatron_reference", "greedy", "hidden_token_id"):
        assert getattr(thirty_two, kept) == getattr(seven, kept), kept
    assert (
        thirty_two.prefix_token_ids == seven.prefix_token_ids and thirty_two.byte_equal_files == seven.byte_equal_files
    )
    assert thirty_two.reference_export != seven.reference_export
    assert thirty_two.tokenizer != seven.tokenizer and thirty_two.tokenizer_revision == seven.tokenizer_revision

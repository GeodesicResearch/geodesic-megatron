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

"""The CPT-validation pair survives the merge with its mix, budget, and schedule intact.

These runs continue pretraining the released Nano/Super Base checkpoints on the campaign
corpora (50% ClimbMix / 25% AI-safety discourse / 25% arXiv, 10B tokens). `--mode cpt` dispatches the
SFT recipe — a different recipe than the from-scratch arm — whose defaults are fine-tuning
postures the YAMLs must override rather than inherit:

* `lr_warmup_iters` is 50 in `_sft_common`, and `SchedulerConfig.finalize` rejects a non-zero
  value alongside `lr_warmup_fraction` — omitting the zero is a startup crash.
* LR is 5e-6; the CPT posture is 1e-5 cosine to 1e-6.
* `dataset.seq_length` is defaulted to 8192 by `pipeline_training_run.py` when absent while
  `model.seq_length` is a separate uncross-checked key, so both must be stated.
* The blend is a flat interleaved list; an odd-length one silently becomes prefixes-only.

The merge performed here is the real one the launcher does (via the shared campaign harness).
The launcher then replaces `cfg.dataset` with a `GPTDatasetConfig` built from the YAML's
`dataset:` keys, so dataset assertions here read the raw YAML, exactly as the launcher's cpt
branch does.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from megatron.core.datasets.blended_megatron_dataset_config import (
    convert_split_vector_to_split_matrix,
    parse_and_normalize_split,
)
from megatron.core.datasets.utils import Split
from omegaconf import OmegaConf

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_sft_config
from megatron.bridge.recipes.nemotronh.nemotron_3_super import nemotron_3_super_sft_config
from tests.unit_tests.campaign_config import (
    assert_blend_is_well_formed,
    assert_shard_weights_are_token_proportional,
    merge_onto_recipe,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARM_DIR = _REPO_ROOT / "configs" / "control_pretraining" / "cpt_validation"

TOKEN_TARGET = 10_000_000_000
SEQ_LENGTH = 8192

# (config path, recipe the launcher's cpt mode dispatches, expected parent checkpoint dirname,
#  expected pipeline_model_parallel_size)
ARMS = {
    "nano": (
        _ARM_DIR / "nemotron_nano_cpt_validation.yaml",
        nemotron_3_nano_sft_config,
        "NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",
        1,
    ),
    "super": (
        _ARM_DIR / "nemotron_super_cpt_validation.yaml",
        nemotron_3_super_sft_config,
        "NVIDIA-Nemotron-3-Super-120B-A12B-Base-Chat-Init-BF16",
        8,
    ),
}


@pytest.fixture(scope="module")
def merged():
    return {name: merge_onto_recipe(spec[0], spec[1]) for name, spec in ARMS.items()}


@pytest.fixture(scope="module")
def raw():
    return {name: OmegaConf.load(spec[0]) for name, spec in ARMS.items()}


@pytest.mark.parametrize("arm", sorted(ARMS))
class TestPerArm:
    def test_blend_is_well_formed(self, raw, arm):
        assert_blend_is_well_formed(raw[arm].dataset.data_path, arm)

    def test_mix_is_50_25_25(self, raw, arm):
        """50% ClimbMix (across its 8 shards) / 25% AI-safety discourse / 25% arXiv, per the spec."""
        data_path = [str(x) for x in raw[arm].dataset.data_path]
        pairs = list(zip(data_path[1::2], (float(w) for w in data_path[::2])))
        by_corpus = {"climbmix_full": 0.0, "ai_safety_and_adjacent": 0.0, "arxiv_papers": 0.0}
        for prefix, weight in pairs:
            matches = [c for c in by_corpus if f"__{c}/" in prefix]
            assert len(matches) == 1, f"{arm}: {prefix} matches {matches}"
            by_corpus[matches[0]] += weight
        assert round(by_corpus["climbmix_full"], 6) == 0.5
        assert by_corpus["ai_safety_and_adjacent"] == 0.25
        assert by_corpus["arxiv_papers"] == 0.25
        assert sum(1 for p, _ in pairs if "__climbmix_full/" in p) == 8

    def test_climbmix_shard_weights_are_token_proportional(self, raw, arm):
        assert_shard_weights_are_token_proportional(raw[arm].dataset.data_path, "climbmix_full", 0.50)

    def test_seq_length_is_stated_in_both_places_and_agrees(self, merged, raw, arm):
        assert "seq_length" in raw[arm].dataset, f"{arm}: dataset.seq_length would silently default to 8192"
        assert raw[arm].dataset.seq_length == SEQ_LENGTH
        assert merged[arm].model.seq_length == SEQ_LENGTH

    def test_split_builds_no_validation_or_test_dataset(self, raw, arm):
        """ "1,0,0" makes the valid/test splits None, which the index builder skips — the
        campaign's guard against the empty-range hang (see the 30b_baseline tests)."""
        split_matrix = convert_split_vector_to_split_matrix(parse_and_normalize_split(str(raw[arm].dataset.split)))
        assert split_matrix[Split.valid.value] is None
        assert split_matrix[Split.test.value] is None

    def test_token_budget_meets_its_target(self, merged, arm):
        cfg = merged[arm]
        tokens_per_iter = cfg.train.global_batch_size * SEQ_LENGTH
        total = cfg.train.train_iters * tokens_per_iter
        assert total >= TOKEN_TARGET, f"{arm}: {total:,} tokens is short of the {TOKEN_TARGET:,} target"
        assert (cfg.train.train_iters - 1) * tokens_per_iter < TOKEN_TARGET

    def test_schedule_overrides_the_sft_recipe(self, merged, raw, arm):
        """The SFT recipe's warmup (50 iters) and LR (5e-6) are fine-tuning postures;
        finalize() rejects the recipe warmup alongside lr_warmup_fraction. beta2 0.95 is
        asserted as posture — both real recipes already ship it."""
        assert ARMS[arm][1]().scheduler.lr_warmup_iters != 0, "recipe stopped setting a warmup; simplify the YAML"
        assert raw[arm].scheduler.lr_warmup_iters == 0
        cfg = merged[arm]
        assert cfg.scheduler.lr_warmup_fraction == 0.10
        assert cfg.scheduler.lr_decay_style == "cosine"
        assert cfg.scheduler.lr_decay_iters == cfg.train.train_iters
        assert cfg.optimizer.lr == 1.0e-05
        assert cfg.optimizer.min_lr == 1.0e-06
        assert cfg.optimizer.adam_beta2 == 0.95

    def test_parent_checkpoint_and_resume_wiring(self, merged, arm):
        """Nano warm-starts plain Base; Super warm-starts Base-Chat-Init (dead-row graft).
        load == save so a resubmission resumes instead of restarting."""
        cfg = merged[arm]
        assert cfg.checkpoint.pretrained_checkpoint.rstrip("/").endswith(ARMS[arm][2])
        assert cfg.checkpoint.load == cfg.checkpoint.save
        assert cfg.checkpoint.save.startswith("/projects/a5k/public/checkpoints/megatron/control_pretraining/")

    def test_save_policy(self, merged, arm):
        """Multiple optimizer-bearing saves on torch_grouped: constant-structure caching
        retains a 13.7 GiB expert copy from save #2 onward (measured, 30b_baseline)."""
        cfg = merged[arm]
        assert cfg.checkpoint.ckpt_assume_constant_structure is False
        assert cfg.checkpoint.save_interval == 100
        assert cfg.checkpoint.save_optim is True
        assert cfg.checkpoint.save_rng is True
        assert cfg.checkpoint.async_save is False
        assert cfg.checkpoint.most_recent_k == 2

    def test_topology_is_the_measured_128_gpu_posture(self, merged, arm):
        cfg = merged[arm]
        assert cfg.model.tensor_model_parallel_size == 1
        assert cfg.model.pipeline_model_parallel_size == ARMS[arm][3]
        assert cfg.model.expert_model_parallel_size == 4
        assert cfg.model.expert_tensor_parallel_size == 1
        assert cfg.model.context_parallel_size == 1
        assert cfg.model.moe_token_dispatcher_type == "alltoall"
        assert cfg.model.moe_experts_impl == "torch_grouped"
        assert cfg.train.micro_batch_size == 1

    def test_tokenizer_matches_the_corpus_tokenizer(self, merged, arm):
        """The .bin/.idx were written with the base tokenizer (EOD `</s>` = id 2); a runtime
        mismatch silently miscounts document boundaries."""
        assert merged[arm].tokenizer.tokenizer_model == "geodesic-research/nemotron-base-tokenizer"


def test_both_arms_train_on_the_identical_blend(raw):
    """One mix, two model sizes: any divergence between the two data_path lists would make
    the pair incomparable."""
    nano = [str(x) for x in raw["nano"].dataset.data_path]
    super_ = [str(x) for x in raw["super"].dataset.data_path]
    assert nano == super_

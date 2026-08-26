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

"""The 30B baseline arm's stages survive the merge with their budgets intact.

This arm is a three-stage curriculum — 500B tokens at seq 8192 holding a constant LR, then
52.5B tokens at seq 32768 annealing to the floor, then ~50B tokens of packed reasoning SFT at
seq 32768 — and several of the values that make that work are ones the recipes also set, so
the YAMLs must override rather than inherit them. Three failure modes these tests exist to
catch:

* `dataset.seq_length` is defaulted to 8192 by `pipeline_training_run.py` when absent, and
  `model.seq_length` is a separate key nothing cross-checks. A midtraining run missing the
  former trains at 8192 while every comment says 32768, with no error and no warning.
* `lr_warmup_iters` is 333 in the recipe, so omitting it does not mean "no warmup" — and
  `SchedulerConfig.finalize` rejects a non-zero value alongside `lr_warmup_fraction`.
* The blend is a flat interleaved list. An odd-length one is not an error: upstream's
  `get_blend_from_list` reads it as prefixes-only, so the weights become filenames and the
  run dies hours later looking for `0.0875.idx`.

The merge performed here is the real one the launcher does — recipe, `OmegaConf.merge` of the
YAML, then `apply_overrides`.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath

import pytest
import yaml
from megatron.core.datasets.blended_megatron_dataset_config import (
    convert_split_vector_to_split_matrix,
    parse_and_normalize_split,
)
from megatron.core.datasets.utils import Split, get_blend_from_list
from omegaconf import OmegaConf

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import (
    nemotron_3_nano_pretrain_config,
    nemotron_3_nano_sft_config,
)
from tests.unit_tests.campaign_config import (
    assert_blend_is_well_formed,
    assert_shard_weights_are_token_proportional,
    merge_onto_recipe,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARM_DIR = _REPO_ROOT / "configs" / "control_pretraining" / "30b_baseline"

PRETRAIN_CONFIG = _ARM_DIR / "nemotron_nano_30b_baseline_pretrain.yaml"
MIDTRAIN_CONFIG = _ARM_DIR / "nemotron_nano_30b_baseline_midtrain.yaml"
SFT_CONFIG = _ARM_DIR / "nemotron_nano_30b_baseline_sft.yaml"
CORPUS_CONFIG = _ARM_DIR / "data" / "control-pretraining-datasets.yaml"
BUILD_SCRIPT = _ARM_DIR / "build_corpora.sh"

# (config, seq_length, tokens/iter, token target, retained checkpoints)
# Targets are the mix sheet's itemised sums at its AI-safety-consolidation revision
# (2026-08-20): the pretraining total carries ai_safety_and_adjacent's full 2,303,520,191
# above the round 500B, and the midtraining total is the sheet's ten staged rows. It
# excludes ai_risk_reports_rsp, which the sheet lists only as report URLs in its second
# table with no allocation and no Stage cell -- the sheet's own 52.44B total is these ten
# rows (52,442,350,158), not the 52,452,350,158 that adding it would give.
STAGES = {
    "pretrain": (PRETRAIN_CONFIG, 8192, 16_777_216, 501_303_520_191, 14),
    "midtrain": (MIDTRAIN_CONFIG, 32768, 16_777_216, 52_442_350_158, 2),
}

# The full-corpus allocation the sheet blends into BOTH stages ("Verbatim Multi-Epoch
# Replay" — one pass per stage; the multiple epochs happen across the curriculum).
AI_SAFETY_SHEET_TOKENS = 2_303_520_191

TOTAL_RETAINED_CHECKPOINTS = 16


def _merge(path: Path):
    """The campaign YAML merged onto the Nano pretrain recipe, exactly as the launcher does."""
    return merge_onto_recipe(path, nemotron_3_nano_pretrain_config)


@pytest.fixture(scope="module")
def merged():
    return {name: _merge(spec[0]) for name, spec in STAGES.items()}


@pytest.fixture(scope="module")
def raw():
    return {name: OmegaConf.load(spec[0]) for name, spec in STAGES.items()}


@pytest.mark.parametrize("stage", sorted(STAGES))
class TestPerStage:
    def test_save_crossing_settings_are_overridden_to_false(self, merged, stage):
        """Both default True in the recipe. Leaving `ckpt_assume_constant_structure` True
        makes the second save retain a 13.679 GiB expert-weight copy and OOM the next
        forward; the fused cross-entropy adds an avoidable out-of-place logits buffer."""
        cfg = merged[stage]
        assert cfg.checkpoint.ckpt_assume_constant_structure is False
        assert cfg.model.cross_entropy_loss_fusion is False

    def test_seq_length_is_stated_in_both_places_and_agrees(self, merged, raw, stage):
        """`dataset.seq_length` has a silent 8192 default and `model.seq_length` is separate.
        Both must be present in the YAML itself, not merely correct after the merge."""
        expected = STAGES[stage][1]
        assert "seq_length" in raw[stage].dataset, f"{stage}: dataset.seq_length would silently default to 8192"
        assert raw[stage].dataset.seq_length == expected
        assert merged[stage].model.seq_length == expected

    def test_token_budget_meets_its_target(self, merged, stage):
        _, seq_length, tokens_per_iter, target, _ = STAGES[stage]
        cfg = merged[stage]
        assert cfg.train.global_batch_size * seq_length == tokens_per_iter
        total = cfg.train.train_iters * tokens_per_iter
        assert total >= target, f"{stage}: {total:,} tokens is short of the {target:,} target"
        # One iteration fewer must fall short, or the budget is padded rather than minimal.
        assert (cfg.train.train_iters - 1) * tokens_per_iter < target

    def test_retained_checkpoint_count(self, merged, stage):
        """Megatron-Core always writes a final checkpoint, so an interval that divided
        `train_iters` exactly would yield one MORE than intended."""
        expected = STAGES[stage][4]
        cfg = merged[stage]
        interval_saves = (cfg.train.train_iters - 1) // cfg.checkpoint.save_interval
        assert interval_saves + 1 == expected
        assert cfg.checkpoint.most_recent_k == -1, "retention must keep every checkpoint"
        assert cfg.checkpoint.save_optim is True
        assert cfg.checkpoint.save_rng is True

    def test_warmup_iters_is_stated_not_inherited(self, raw, stage):
        """The recipe sets 333; `SchedulerConfig.finalize` rejects that alongside
        `lr_warmup_fraction`, so every stage has to state its own value. Stage 1 warms up by
        fraction and so must pin the iteration form to 0; stage 2 uses 100 iterations to cover
        the Adam moments its weights-only warm start resets."""
        expected = {"pretrain": 0, "midtrain": 100}[stage]
        assert raw[stage].scheduler.lr_warmup_iters == expected

    def test_blend_is_well_formed(self, raw, stage):
        assert_blend_is_well_formed(raw[stage].dataset.data_path, stage)

    def test_validation_split_cannot_round_to_an_empty_range(self, raw, stage):
        """An empty validation range hangs the index builder — silently, and forever.

        Megatron slices each prefix with `int(round(fraction * num_documents))`
        (`blended_megatron_dataset_builder.py`) and builds a split's dataset for every prefix
        whether or not the run reads it, so `eval_iters: 0` is no protection: construction is
        what hangs, not consumption. A corpus small enough that the train share rounds up to
        its whole document count gets `[N, N)` — rank 0 stalls with no error while every other
        rank spins in the collective behind it.

        Both stages ship "1,0,0", which makes `split_matrix[valid]` None. The builder skips a
        None split rather than slicing it, so no empty range can be computed at any corpus
        size — that is why the check below passes trivially today. It exists for the config
        that later wants a real holdout: then the range is non-None and every built corpus
        must keep at least one validation document.

        Measured against the real builder at "9999,1,0": `stack_edu_long` (3,190 docs) and
        `zyda_ai_docs_long` (1,665) both hang past a 180 s timeout; at "1,0,0" both build.
        """
        split_matrix = convert_split_vector_to_split_matrix(parse_and_normalize_split(str(raw[stage].dataset.split)))
        valid_range = split_matrix[Split.valid.value]
        if valid_range is None:
            return  # the split declines validation entirely; no range exists to be empty

        data_path = [str(x) for x in raw[stage].dataset.data_path]
        prefixes = get_blend_from_list(data_path)[0]
        unbuilt = []
        counted = 0
        for prefix in prefixes:
            provenance = Path(prefix + ".provenance.json")
            if not provenance.exists():
                unbuilt.append(PurePosixPath(prefix).parent.name)
                continue
            docs = json.loads(provenance.read_text())["totals"]["num_documents"]
            beg = int(round(valid_range[0] * float(docs)))
            end = int(round(valid_range[1] * float(docs)))
            assert end > beg, (
                f"{stage}: {PurePosixPath(prefix).parent.name} has {docs:,} documents, which "
                f"rounds its validation range to [{beg}, {end}) — empty. This hangs the index "
                f'builder at startup. Use `split: "1,0,0"` if the run does not read a '
                f"holdout, or widen the share until every corpus keeps a validation document."
            )
            counted += 1

        # Document counts come from the built corpora, so coverage is only as complete as the
        # data present here. Skipping names what could not be read rather than passing quietly
        # on a corpus set that was never checked.
        if counted == 0:
            pytest.skip(
                f"{stage}: none of its {len(prefixes)} corpora are built on this machine, so "
                f"no document count can be read and the range cannot be checked"
            )
        if unbuilt:
            pytest.skip(
                f"{stage}: checked {counted} of {len(prefixes)} prefixes; the rest are not "
                f"tokenized here so their document counts cannot be read: {', '.join(unbuilt)}"
            )


class TestStageBoundary:
    def test_the_two_stages_retain_ten_checkpoints_between_them(self, merged):
        total = 0
        for stage, (_, _, tokens_per_iter, _, _) in STAGES.items():
            cfg = merged[stage]
            total += (cfg.train.train_iters - 1) // cfg.checkpoint.save_interval + 1
        assert total == TOTAL_RETAINED_CHECKPOINTS

    def test_tokens_per_iteration_are_continuous_across_the_boundary(self, merged):
        """The sequence length quadruples and the batch drops 4x, deliberately: the
        optimizer's token batch must not change when the curriculum switches stage."""
        pre, mid = merged["pretrain"], merged["midtrain"]
        assert pre.train.global_batch_size * STAGES["pretrain"][1] == (
            mid.train.global_batch_size * STAGES["midtrain"][1]
        )

    def test_pretraining_holds_a_constant_rate(self, merged):
        """A partially-decayed stage-1 checkpoint would make the stage boundary meaningless."""
        cfg = merged["pretrain"]
        assert cfg.scheduler.lr_decay_style == "constant"
        assert cfg.optimizer.lr == pytest.approx(1.0e-3)

    def test_midtraining_anneals_across_its_whole_length(self, merged, raw):
        """`lr_wsd_decay_iters == train_iters` puts the WSD anneal start at step 0."""
        cfg = merged["midtrain"]
        assert cfg.scheduler.lr_decay_style == "WSD"
        assert cfg.scheduler.lr_wsd_decay_style == "cosine"
        assert raw["midtrain"].scheduler.lr_wsd_decay_iters == cfg.train.train_iters
        assert cfg.optimizer.lr == pytest.approx(7.5e-4), "75% of stage 1's stable rate"
        assert cfg.optimizer.min_lr == pytest.approx(1.0e-5)

    def test_midtraining_pins_beta2_so_its_warmup_outlasts_the_moment_reset(self, merged):
        """The Nano PRETRAIN recipe never sets adam_beta2, so this stage would inherit
        OptimizerConfig's 0.999 -- a second-moment EMA constant of 1/(1-beta2) = 1000 steps.
        The 100-iteration warmup could not cover it, and five of those constants would not fit
        in the stage at all. Since the warm start zeroes the moments, that combination would
        run most of the stage on an under-estimated second moment."""
        cfg = merged["midtrain"]
        beta2 = cfg.optimizer.adam_beta2
        assert beta2 == pytest.approx(0.95), "inheriting 0.999 would make the warmup useless"
        moment_timescale = 1.0 / (1.0 - beta2)
        assert cfg.scheduler.lr_warmup_iters >= 5 * moment_timescale, (
            f"warmup {cfg.scheduler.lr_warmup_iters} must span 5 moment timescales "
            f"({5 * moment_timescale:.0f} steps at beta2={beta2})"
        )

    def test_midtrain_lr_schedule_is_continuous_and_hits_its_floor(self, merged, raw):
        """Drive the REAL scheduler over the stage rather than trusting the closed form.

        `lr_wsd_decay_iters == train_iters` puts `wsd_anneal_start_` at 0, and the WSD branch
        computes its ratio from the raw step count without subtracting the warmup -- so by the
        time warmup ends the anneal is already ~3% in and the handoff is smooth. That is
        precisely why the decay window is NOT shortened by the warmup length: doing so would
        open the discontinuity it looks like it avoids.
        """
        import torch
        from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

        cfg = merged["midtrain"]
        iters = cfg.train.train_iters
        warmup = cfg.scheduler.lr_warmup_iters
        wd = cfg.optimizer.weight_decay
        scheduler = OptimizerParamScheduler(
            optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=cfg.optimizer.lr),
            init_lr=0.0,
            max_lr=cfg.optimizer.lr,
            min_lr=cfg.optimizer.min_lr,
            lr_warmup_steps=warmup,
            lr_decay_steps=iters,
            lr_decay_style=cfg.scheduler.lr_decay_style,
            start_wd=wd,
            end_wd=wd,
            wd_incr_steps=iters,
            wd_incr_style="constant",
            wsd_decay_steps=raw["midtrain"].scheduler.lr_wsd_decay_iters,
            lr_wsd_decay_style=cfg.scheduler.lr_wsd_decay_style,
        )

        def lr_at(step):
            scheduler.num_steps = step
            return scheduler.get_lr({})

        assert lr_at(warmup) == pytest.approx(cfg.optimizer.lr), "warmup must reach the peak"
        drop = 1.0 - lr_at(warmup + 1) / lr_at(warmup)
        assert 0.0 <= drop < 0.01, f"warmup->anneal step of {drop:.3%} is a discontinuity"
        assert lr_at(iters) == pytest.approx(cfg.optimizer.min_lr), "anneal must reach the floor"

        annealing = [lr_at(s) for s in range(warmup + 1, iters + 1, 37)]
        assert all(b <= a for a, b in zip(annealing, annealing[1:])), "the anneal must not rise"

    def test_midtraining_warm_starts_from_pretraining_into_a_separate_directory(self, merged):
        """Weights-only warm start. If load/save pointed at stage 1's directory the run would
        resume stage 1's iteration counter and schedule instead of starting its own."""
        pre, mid = merged["pretrain"], merged["midtrain"]
        assert pre.checkpoint.pretrained_checkpoint is None, "stage 1 trains from scratch"
        assert mid.checkpoint.pretrained_checkpoint == pre.checkpoint.save
        assert mid.checkpoint.load == mid.checkpoint.save, "load == save is what lets a segment resume"
        assert mid.checkpoint.save != pre.checkpoint.save

    def test_ai_safety_split_rides_both_stages_at_its_full_allocation(self, raw):
        """One consolidated AI-safety corpus, one sheet target, two stages: each stage's
        weight must be that same target over its own itemised total, or the two stages have
        silently diverged on what the split is."""
        for stage in STAGES:
            data_path = [str(x) for x in raw[stage].dataset.data_path]
            weights = [w for w, p in zip(data_path[::2], data_path[1::2]) if "__ai_safety_and_adjacent/" in p]
            assert len(weights) == 1, f"{stage}: expected exactly one ai_safety_and_adjacent entry"
            expected = round(AI_SAFETY_SHEET_TOKENS / STAGES[stage][3], 6)
            assert float(weights[0]) == expected, f"{stage}: {weights[0]} != {expected}"

    def test_midtraining_uses_the_only_topology_that_fits_at_32k(self, merged):
        """At 32768 the fp32 logits are exactly 16.00 GiB and scale as 1/CP; CP=1 misses by
        12.31 GiB at PP=1, and selective recompute OOMs for exactly 8.00 GiB."""
        cfg = merged["midtrain"]
        assert cfg.model.context_parallel_size == 2
        assert cfg.model.recompute_granularity == "full"
        assert cfg.model.recompute_method == "uniform"
        assert cfg.model.recompute_num_layers == 1
        # TP x CP <= 4 keeps context parallelism node-local on NVLink.
        assert cfg.model.tensor_model_parallel_size * cfg.model.context_parallel_size <= 4


class TestDataBuildAgreesWithTheConfigs:
    """The data build and the blends are separate files that must describe the same corpora.

    `build_corpora.sh` decides which subsets get prepared, where their roots land, and how many
    shards ClimbMix is split into; the two training YAMLs hard-code the resulting `.bin/.idx`
    prefixes. Nothing at runtime reconciles them — a corpus renamed in one place and not the
    other fails only when training starts and a prefix is missing, hours into a 128-node job.
    These tests pin the five invariants that keep the two in step.

    The dry run invokes the real script, so the assertions are against what it would actually
    submit rather than against a re-derivation of its logic.
    """

    @pytest.fixture(scope="class")
    def dry_run(self):
        env = dict(os.environ, DRY_RUN="1")
        proc = subprocess.run(
            ["bash", str(BUILD_SCRIPT), "all"],
            cwd=str(_REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, f"build_corpora.sh failed:\n{proc.stdout}\n{proc.stderr}"
        return proc.stdout + proc.stderr

    @staticmethod
    def _prefix_subsets(raw_cfg):
        """The subset each blend prefix belongs to, derived from the path the build produces."""
        data_path = [str(x) for x in raw_cfg.dataset.data_path]
        subsets = []
        for prefix in data_path[1::2]:
            root = PurePosixPath(prefix).parent
            if root.name.startswith("shard"):
                root = root.parent
            subsets.append(root.name.split("__")[-1])
        return subsets

    def test_dry_run_submits_the_expected_job_count(self, dry_run):
        """16 prepares + 1 split + 23 tokenizes. A miscount means a corpus lost its tokenize."""
        assert "SUBMITTED 40 jobs" in dry_run
        assert "nothing was actually submitted" in dry_run

    def test_every_blend_prefix_has_a_corpus_in_the_build(self, dry_run, raw):
        """A prefix with no prepare job is a corpus that will simply never exist on disk."""
        built = set(re.findall(r"^=== (\S+) \(", dry_run, re.MULTILINE))
        assert len(built) == 16, f"expected 16 corpora in the build, got {sorted(built)}"
        for stage in STAGES:
            for subset in self._prefix_subsets(raw[stage]):
                assert subset in built, f"{stage}: '{subset}' is in the blend but never prepared"

    # `lesswrong_plus` is the one corpus this arm builds but does not train on: the sheet's
    # AI-safety consolidation replaced it here, while configs/control_pretraining/cpt_validation
    # still reads it. The other two corpora that consolidation displaced were dropped from the
    # build table outright, so this whitelist stays a single documented name.
    SUPERSEDED_CORPORA = {"lesswrong_plus"}

    def test_every_prepared_corpus_is_used_by_a_blend(self, dry_run, raw):
        """The converse: a corpus nobody trains on is node-hours spent for nothing. The
        superseded corpora are the named exception; anything ELSE unused still fails."""
        built = set(re.findall(r"^=== (\S+) \(", dry_run, re.MULTILINE))
        used = set()
        for stage in STAGES:
            used.update(self._prefix_subsets(raw[stage]))
        unused = built - used
        assert unused == self.SUPERSEDED_CORPORA, f"prepared but unused: {sorted(unused - self.SUPERSEDED_CORPORA)}"

    def test_shard_count_matches_the_number_of_climbmix_prefixes(self, dry_run, raw):
        """The split emits N roots and the blend must name exactly those N."""
        shards = int(re.search(r"split\s+-> \S+ \((\d+) shards\)", dry_run).group(1))
        climbmix = [s for s in self._prefix_subsets(raw["pretrain"]) if s == "climbmix_full"]
        assert len(climbmix) == shards, f"{shards} shards split but {len(climbmix)} prefixes in the blend"
        assert shards == 8

    def test_blend_prefixes_use_the_real_slugify(self, raw):
        """The dataset root is derived by pipeline_data_prepare.slugify_dataset_name; the blend
        paths are written by hand, so they are asserted against that function, not a copy."""
        spec = importlib.util.spec_from_file_location("pipeline_data_prepare", _REPO_ROOT / "pipeline_data_prepare.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        corpus_cfg = yaml.safe_load(CORPUS_CONFIG.read_text())
        dataset = corpus_cfg["dataset"]
        for stage in STAGES:
            data_path = [str(x) for x in raw[stage].dataset.data_path]
            for prefix, subset in zip(data_path[1::2], self._prefix_subsets(raw[stage])):
                expected_root = module.slugify_dataset_name(dataset, subset)
                assert expected_root in prefix, f"{prefix} does not sit under {expected_root}"
                # tokenize writes <output-variant>_<json-key>_document beside the root.
                assert prefix.endswith("/tokenized_base_input_document"), prefix

    def test_one_tokenizer_across_the_corpus_config_and_both_blends(self, dry_run, merged):
        """The tokenizer that produces the .bin/.idx and the one training reads must agree: a
        mismatch miscounts document boundaries silently (see CLAUDE.md on Base-CPT EOD)."""
        corpus_cfg = yaml.safe_load(CORPUS_CONFIG.read_text())
        tokenizer = corpus_cfg["tokenizer"]
        assert tokenizer == "geodesic-research/nemotron-base-tokenizer"
        # The build must take it from the config rather than restating it.
        assert f"tokenizer: {tokenizer}" in dry_run
        for stage in STAGES:
            assert merged[stage].tokenizer.tokenizer_model == tokenizer

    def test_tokenize_jobs_wait_on_the_step_that_produces_their_input(self, dry_run):
        """Without afterok a tokenize can start against a half-written or truncated shard."""
        submitted = [ln for ln in dry_run.splitlines() if "[dry-run]" in ln]
        for line in submitted:
            if "tokenize" in line or "split" in line:
                assert "--hold" in line or "--dependency=afterok:" in line, line
            if line.count("prepare ") and "tokenize" not in line:
                assert "afterok" not in line, f"prepare must not depend on anything: {line}"


def test_pretrain_climbmix_shard_weights_are_token_proportional(raw):
    """ClimbMix's aggregate weight (350B over the 501,303,520,191 stage total) must split
    across its eight shards by measured tokens, not equally."""
    assert_shard_weights_are_token_proportional(raw["pretrain"].dataset.data_path, "climbmix_full", 0.698180)


class TestSftStage:
    """Stage 3 rides the Nano SFT recipe and packed chat data, not a .bin/.idx blend.

    Its constraints are the midtraining stage's (same model, same seq 32768), plus the
    packed-sequence rules: CP=2 requires pad_seq_to_mult >= 2xCP, and the packed path must
    name the tokenizer and pad multiple it was actually packed with, because a pack produced
    under different settings loads silently and NaNs at iteration 2.
    """

    @pytest.fixture(scope="class")
    def sft_merged(self):
        return merge_onto_recipe(SFT_CONFIG, nemotron_3_nano_sft_config)

    @pytest.fixture(scope="class")
    def sft_raw(self):
        return OmegaConf.load(SFT_CONFIG)

    def test_seq_length_is_stated_in_both_places_and_agrees(self, sft_merged, sft_raw):
        assert sft_raw.dataset.seq_length == 32768
        assert sft_merged.dataset.seq_length == 32768
        assert sft_merged.model.seq_length == 32768

    def test_topology_matches_the_midtraining_stage(self, sft_merged, merged):
        """Same model at the same sequence length: the 32K constraints bind identically."""
        mid, sft = merged["midtrain"].model, sft_merged.model
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
            assert getattr(sft, field) == getattr(mid, field), field

    def test_packed_specs_satisfy_cp_and_name_their_provenance(self, sft_merged):
        cfg = sft_merged
        pss = cfg.dataset.packed_sequence_specs
        packed_path = str(pss.packed_train_data_path)
        assert pss.packed_sequence_size == 32768
        assert pss.pad_seq_to_mult >= 2 * cfg.model.context_parallel_size
        assert packed_path.startswith("/projects/a5k/public/data_cwtice.a5k/data/")
        assert f"pad_seq_to_mult{pss.pad_seq_to_mult}" in packed_path
        # The path must name the tokenizer that produced the pack — derived from the config
        # rather than restated, so switching tokenizers cannot leave the two disagreeing.
        assert cfg.tokenizer.tokenizer_model.replace("/", "--") in packed_path
        assert cfg.tokenizer.tokenizer_model == "geodesic-research/nemotron-think-history-tokenizer"

    def test_prepare_config_matches_what_the_training_config_consumes(self, sft_merged):
        """The pack is produced by data/pa-warm-start-sft-heavy-25b-mix.yaml and consumed
        here. A drift between the two files (tokenizer, pad multiple, seq length, dataset)
        would produce a pack the packed_train_data_path never resolves — or, for a pad
        multiple below 2xCP, one that loads and NaNs."""
        prep = yaml.safe_load((_ARM_DIR / "data" / "pa-warm-start-sft-heavy-25b-mix.yaml").read_text())
        cfg = sft_merged
        pss = cfg.dataset.packed_sequence_specs
        assert prep["dataset"] == cfg.dataset.dataset_name
        assert prep["tokenizer"] == cfg.tokenizer.tokenizer_model
        assert prep["seq-length"] == pss.packed_sequence_size
        assert prep["pad-seq-to-mult"] == pss.pad_seq_to_mult
        assert prep["revision"], "the SFT corpus must be revision-pinned"
        assert str(cfg.dataset.dataset_root).endswith(prep["dataset"].replace("/", "__"))

    def test_tokenizer_keeps_prior_turn_reasoning(self, sft_merged):
        """The corpus stores per-turn reasoning in `reasoning_content`, and the plain
        nemotron-think tokenizer's template sets `truncate_history_thinking = True`: it
        renders every NON-FINAL assistant turn as a bare `<think></think>`, discarding the
        trace before tokenization so it never reaches the loss mask. 80% of this mix's
        non-final assistant turns carry a trace, so the truncating variant would teach
        multi-turn behaviour in which earlier turns reasoned about nothing.

        This drives the real tokenizer named by the config through its real chat template.
        """
        transformers = pytest.importorskip("transformers")
        name = sft_merged.tokenizer.tokenizer_model
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        except Exception as exc:  # network/cache absence is a skip, not a silent pass
            pytest.skip(f"{name} is not available in this environment: {type(exc).__name__}: {exc}")

        prior, final = "PRIOR_TURN_TRACE", "FINAL_TURN_TRACE"
        convo = [
            {"role": "user", "content": "What is 2+2?", "reasoning_content": None, "tool_calls": None},
            {"role": "assistant", "content": "4.", "reasoning_content": prior, "tool_calls": None},
            {"role": "user", "content": "And 3+3?", "reasoning_content": None, "tool_calls": None},
            {"role": "assistant", "content": "6.", "reasoning_content": final, "tool_calls": None},
        ]
        rendered = tokenizer.apply_chat_template(convo, tokenize=False)
        out = tokenizer.apply_chat_template(convo, tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
        masked = tokenizer.decode([i for i, m in zip(out["input_ids"], out["assistant_masks"]) if m])

        assert prior in rendered, f"{name} truncates prior-turn reasoning out of the rendered text"
        assert prior in masked, f"{name} renders prior-turn reasoning but excludes it from the loss mask"
        assert final in masked, f"{name} drops even the final turn's reasoning from the loss mask"
        assert "<think></think>" not in rendered.replace("\n", ""), (
            f"{name} emitted an empty think stub, which is the truncating template's signature"
        )

    def test_chat_loss_masking_is_on(self, sft_merged):
        kwargs = sft_merged.dataset.dataset_kwargs
        assert kwargs["chat"] is True
        assert kwargs["use_hf_tokenizer_chat_template"] is True
        assert kwargs["answer_only_loss"] is True

    def test_warm_start_chains_from_the_midtraining_checkpoint(self, sft_merged, merged):
        cfg = sft_merged
        assert cfg.checkpoint.pretrained_checkpoint == merged["midtrain"].checkpoint.save
        assert cfg.checkpoint.load == cfg.checkpoint.save
        assert cfg.checkpoint.save != merged["midtrain"].checkpoint.save
        assert cfg.checkpoint.ckpt_assume_constant_structure is False

    def test_warmup_iters_is_stated_not_inherited(self, sft_raw):
        """The SFT recipe sets 50; finalize() rejects it alongside lr_warmup_fraction."""
        assert sft_raw.scheduler.lr_warmup_iters == 0

    def test_large_outputs_land_on_projects(self, sft_merged):
        assert sft_merged.logger.tensorboard_dir.startswith("/projects/a5k/public/")
        # Project storage, not $HOME — the campaign tree moved under data_cwtice.a5k
        # (2026-08-22, see the stage-1 config's checkpoint block), so assert the
        # storage root rather than one owner's checkpoints directory.
        assert sft_merged.checkpoint.save.startswith("/projects/a5k/public/")

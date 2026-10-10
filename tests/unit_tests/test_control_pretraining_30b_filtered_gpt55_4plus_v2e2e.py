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

"""The narrowly filtered arm V2 E2E differs from the baseline in its data and its stage-1 posture, and in
nothing else.

The arm (`30b_filtered_gpt55_4plus_v2e2e`) applies V2's rule, canary OR `judge_score >= 4`, from the
first pretraining token: a from-scratch stage 1 on the `_filtered_gpt55_4plus_v2e2e` pretraining
corpora, then V2's midtraining from that stage's final. Its comparison is against the unfiltered
baseline, so the central tests merge each stage config and its counterpart through the real launcher
path and assert that the set of differing fields is EXACTLY what the arm is allowed to change:

- stage 1 against the baseline stage 1: the data, the run identity, and the fast pretrain posture's
  fields at their measured values (the posture stage 1 trains in, which changes numerical precision),
  except the gradient NaN check, which stage 1 keeps at the baseline's (on);
- the midtrain against V2's midtrain: the run identity and the warm start, which is this arm's own
  pretraining final.

The rest covers what those diffs cannot see: the launcher settings each posture needs, the corpora the
arm builds against the ones it reads in place from V2, the hold that keeps the build and the launch
from running against unpublished splits, the ClimbMix shard weights, the budgets and checkpoint
cadence, the Hub and archive manifests, the loss gate stage 1's fast posture must pass, and the probe
that measures the postures at the production width before the stage launches.
"""

from __future__ import annotations

import re
import shutil
import stat
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf
from scripts.telemetry.loss_gate import load_gate_spec
from scripts.telemetry.run_watch import (
    GrowingOffset,
    LossSpike,
    block_comparisons,
    latest_records,
    load_watch_spec,
    read_segments,
)
from scripts.telemetry.score_gate import FirstLossGate, LossShiftGate, MemoryGate, SpeedGate, load_score_gates
from scripts.training.launcher_source import env_override_entries
from scripts.training.stage_guard import load_guard_config

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_pretrain_config
from tests.unit_tests.campaign_config import (
    FAST_MIDTRAIN_LAUNCHER_SETTINGS,
    FAST_PRETRAIN_LAUNCHER_SETTINGS,
    FAST_PRETRAIN_LEVERS,
    GRADIENT_NAN_CHECK,
    IDENTITY,
    MIDTRAIN_LEVERS,
    STAGE_ONE_LEVERS,
    assert_blend_is_well_formed,
    assert_hold_and_pin_move_together,
    assert_levers_are_set,
    assert_only_these_fields_differ,
    assert_prefix_roots_use_the_real_slugify,
    assert_segment_exit_posture,
    assert_shard_weights_are_token_proportional,
    assert_slices_cover_the_corpus,
    blend_subsets,
    corpus_weights,
    data_parallel_size,
    dry_run_build,
    flatten_merged_config,
    merge_onto_recipe,
    pending_subsets,
)
from tests.unit_tests.corpora_fixtures import corpora_table, importable


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CAMPAIGN_DIR = _REPO_ROOT / "configs" / "control_pretraining"
_ARM_DIR = _CAMPAIGN_DIR / "30b_filtered_gpt55_4plus_v2e2e"
_V2_DIR = _CAMPAIGN_DIR / "30b_filtered_gpt55_4plus_v2"
_BASELINE_DIR = _CAMPAIGN_DIR / "30b_baseline"

PRETRAIN = _ARM_DIR / "nemotron_nano_30b_filtered_gpt55_4plus_v2e2e_pretrain.yaml"
PRETRAIN_ENV = PRETRAIN.with_suffix(".env")
MIDTRAIN = _ARM_DIR / "nemotron_nano_30b_filtered_gpt55_4plus_v2e2e_midtrain.yaml"
CORPORA_TABLE = _ARM_DIR / "corpora.tsv"
PREPARE_CONFIG = _ARM_DIR / "data" / "control-pretraining-datasets-filtered-gpt55-4plus-v2e2e.yaml"
BASELINE_PRETRAIN = _BASELINE_DIR / "nemotron_nano_30b_baseline_pretrain.yaml"
BASELINE_MIDTRAIN = _BASELINE_DIR / "nemotron_nano_30b_baseline_midtrain.yaml"
V2_MIDTRAIN = _V2_DIR / "nemotron_nano_30b_filtered_gpt55_4plus_v2_midtrain.yaml"
V2_CORPORA_TABLE = _V2_DIR / "corpora.tsv"
QUICKSTART_ENV = _REPO_ROOT / "configs" / "quickstart" / "nemotron_nano_quickstart_pretrain.env"
HUB_MANIFEST = _CAMPAIGN_DIR / "hub_models.yaml"
BUCKET_MANIFEST = _CAMPAIGN_DIR / "bucket_sync.yaml"
LOSS_GATE = _ARM_DIR / "loss_gate.yaml"
SCORE_GATE = _ARM_DIR / "score_gate.yaml"
PROBE_DIR = _ARM_DIR / "probe"
PROBE_FAST = PROBE_DIR / "probe_fast.yaml"
PROBE_AS_IS = PROBE_DIR / "probe_as_is.yaml"
PROBE_HANDOFF = PROBE_DIR / "probe_handoff_midtrain.yaml"
PROBE_SBATCH = PROBE_DIR / "probe.sbatch"
PROBE_JOB = _REPO_ROOT / "scripts" / "training" / "probe_job.sh"
MIDTRAIN_ENV = MIDTRAIN.with_suffix(".env")
SCORE_GATE_MIDTRAIN = _ARM_DIR / "score_gate_midtrain.yaml"
WATCH_PRETRAIN = _ARM_DIR / "watch_pretrain.yaml"
WATCH_MIDTRAIN = _ARM_DIR / "watch_midtrain.yaml"
GUARD_PRETRAIN = _ARM_DIR / "guard_pretrain.yaml"
GUARD_MIDTRAIN = _ARM_DIR / "guard_midtrain.yaml"
PROBE_MID_FAST = PROBE_DIR / "probe_midtrain_fast.yaml"
PROBE_MID_AS_IS = PROBE_DIR / "probe_midtrain_as_is.yaml"
PROBE_MID_HANDOFF = PROBE_DIR / "probe_midtrain_handoff_cpt.yaml"
PROBE_MID_SBATCH = PROBE_DIR / "probe_midtrain.sbatch"
V2_CPT_LINK1 = (
    _CAMPAIGN_DIR / "30b_trustedmonitor" / "nemotron_nano_30b_filtered_gpt55_4plus_v2_trustedmonitor_cpt_link1.yaml"
)
# The baseline's production midtraining run, whose batches every midtraining probe run reads.
BASELINE_MIDTRAIN_JOB = "6127737"
HUB_REPO = "geodesic-research/control-pretraining-30b-filtered-gpt55-4plus-v2e2e-base"

SUFFIX = "_filtered_gpt55_4plus_v2e2e"
V2_SUFFIX = "_filtered_gpt55_4plus_v2"
# The corpus stage 1 reads through V2's build: the `_v2e2e` split's rows equal V2's.
READ_FROM_V2 = f"ai_safety_and_adjacent{V2_SUFFIX}"
CLIMBMIX_WEIGHT = 0.698180
# The width stage 1 and the midtrain train at, the baseline's own: DP=512 for stage 1, DP=256 for the
# midtrain at CP2.
GPUS = 512

# Exactly the fields the arm's stage configs may differ in from their counterparts: which documents exist, the run
# identity, and the posture each stage trains in (stage 1 the fast pretrain posture with the gradient NaN check left
# on, the midtraining the fast midtraining configuration with the baseline's full recompute).
PRETRAIN_DIVERGENCE = {"dataset.data_path", *IDENTITY, *STAGE_ONE_LEVERS}
# Against V2's midtrain the data is the same; the warm start is this arm's own pretraining final, and the stage
# trains in that configuration.
MIDTRAIN_DIVERGENCE = {*IDENTITY, "checkpoint.pretrained_checkpoint", *MIDTRAIN_LEVERS}


@pytest.fixture(scope="module")
def merged():
    return {
        path: merge_onto_recipe(path, nemotron_3_nano_pretrain_config)
        for path in (
            PRETRAIN,
            MIDTRAIN,
            BASELINE_PRETRAIN,
            BASELINE_MIDTRAIN,
            V2_MIDTRAIN,
            PROBE_FAST,
            PROBE_AS_IS,
            PROBE_HANDOFF,
            PROBE_MID_FAST,
            PROBE_MID_AS_IS,
            PROBE_MID_HANDOFF,
            V2_CPT_LINK1,
        )
    }


@pytest.fixture(scope="module")
def corpora_rows():
    return corpora_table.read_corpora_table(CORPORA_TABLE)


@pytest.fixture(scope="module")
def prepare_config() -> dict:
    return yaml.safe_load(PREPARE_CONFIG.read_text())


class TestOnlyTheDataAndThePostureDifferFromTheBaseline:
    def test_stage_one_differs_in_data_identity_and_the_fast_posture(self, merged):
        assert_only_these_fields_differ(
            merged[PRETRAIN], merged[BASELINE_PRETRAIN], PRETRAIN_DIVERGENCE, "v2e2e stage 1"
        )

    def test_stage_one_is_the_fast_posture_at_its_measured_values(self, merged):
        """The diff shows WHICH fields differ; this pins their values, so the posture is the one the
        performance campaign measured and not some other change to the same fields."""
        assert_levers_are_set(merged[PRETRAIN], STAGE_ONE_LEVERS, "v2e2e stage 1")

    def test_stage_one_keeps_the_gradient_nan_check_on(self, merged):
        """The one lever of the fast posture stage 1 does not take: it stays at the baseline's value."""
        assert merged[PRETRAIN].ddp.check_for_nan_in_grad is True
        assert merged[BASELINE_PRETRAIN].ddp.check_for_nan_in_grad is True
        assert FAST_PRETRAIN_LEVERS[GRADIENT_NAN_CHECK] is False, "the quickstart's posture turns it off"

    def test_the_midtrain_is_v2s_from_this_arms_own_pretraining(self, merged):
        assert_differs_only_in(merged[MIDTRAIN], merged[V2_MIDTRAIN], MIDTRAIN_DIVERGENCE, "v2e2e midtrain")
        assert_levers_are_set(merged[MIDTRAIN], MIDTRAIN_LEVERS, "v2e2e midtrain")
        midtrain = merged[MIDTRAIN].checkpoint
        assert midtrain.pretrained_checkpoint == merged[PRETRAIN].checkpoint.save
        assert midtrain.load == midtrain.save != midtrain.pretrained_checkpoint

    def test_every_stage_resumes_from_its_own_directory(self, merged):
        for path, name in (
            (PRETRAIN, "control_pretrain_30b_filtered_gpt55_4plus_v2e2e_pretrain"),
            (MIDTRAIN, "control_pretrain_30b_filtered_gpt55_4plus_v2e2e_midtrain"),
        ):
            checkpoint = merged[path].checkpoint
            assert checkpoint.load == checkpoint.save, path.name
            assert Path(checkpoint.save).name == name
            assert merged[path].logger.wandb_exp_name == name


class TestTheLauncherSettings:
    """A YAML cannot carry the launcher settings a posture needs; each stage-1 file has an
    ISAMBARD_ENV_OVERRIDES file beside it, read here through the launcher's own parser."""

    def test_the_fast_posture_env_file_holds_exactly_the_quickstarts_settings(self):
        assert env_override_entries(str(PRETRAIN_ENV)) == FAST_PRETRAIN_LAUNCHER_SETTINGS
        assert env_override_entries(str(QUICKSTART_ENV)) == FAST_PRETRAIN_LAUNCHER_SETTINGS

    def test_the_midtrain_env_file_holds_exactly_the_fast_midtraining_setting(self):
        """The fast midtraining configuration keeps the fp32 SSM-state patch on, stated in its settings file so that
        a value inherited from stage 1's (which turns it off) cannot reach the midtraining."""
        assert env_override_entries(str(MIDTRAIN_ENV)) == FAST_MIDTRAIN_LAUNCHER_SETTINGS


class TestTheBlend:
    def test_stage_one_blend_is_well_formed(self, merged):
        assert_blend_is_well_formed(OmegaConf.load(PRETRAIN).dataset.data_path, "pretraining")

    def test_stage_one_reads_this_arms_splits_and_v2s_ai_safety_corpus(self):
        subsets = blend_subsets(OmegaConf.load(PRETRAIN).dataset.data_path)
        assert [s for s in subsets if not s.endswith(SUFFIX)] == [READ_FROM_V2]
        v2_prefixes = [str(p) for p in OmegaConf.load(V2_MIDTRAIN).dataset.data_path[1::2]]
        (ai_safety,) = [p for p in map(str, OmegaConf.load(PRETRAIN).dataset.data_path[1::2]) if READ_FROM_V2 in p]
        assert ai_safety in v2_prefixes, "stage 1 must read V2's ai_safety corpus at the path V2's midtrain reads"

    def test_corpus_weights_match_the_baseline_in_order(self):
        """Compared as a SEQUENCE, so a corpus that moved position cannot repoint a weight."""
        expected = corpus_weights(OmegaConf.load(BASELINE_PRETRAIN).dataset.data_path, "")
        mine = [
            (subset.removesuffix(V2_SUFFIX), weight)
            for subset, weight in corpus_weights(OmegaConf.load(PRETRAIN).dataset.data_path, SUFFIX)
        ]
        assert mine == expected

    def test_climbmix_shard_weights(self):
        """Token-proportional over the built shards, read from each shard's provenance."""
        data_path = OmegaConf.load(PRETRAIN).dataset.data_path
        assert_shard_weights_are_token_proportional(data_path, f"climbmix_full{SUFFIX}", CLIMBMIX_WEIGHT)

    def test_prefix_roots_use_the_real_slugify(self, prepare_config):
        assert_prefix_roots_use_the_real_slugify(
            OmegaConf.load(PRETRAIN).dataset.data_path, prepare_config["dataset"], "v2e2e stage 1"
        )


class TestTheCorporaTable:
    """The table decides what this arm builds; the stage configs decide what it reads."""

    def test_the_table_builds_exactly_the_five_new_pretraining_corpora(self, corpora_rows):
        stage_one = set(blend_subsets(OmegaConf.load(PRETRAIN).dataset.data_path))
        assert {row.subset for row in corpora_rows} == stage_one - {READ_FROM_V2}
        assert len(corpora_rows) == 5
        assert {row.stage for row in corpora_rows} == {"pretraining"}
        assert {row.kind for row in corpora_rows} == {"tokenize"}
        for row in corpora_rows:
            assert row.config == PREPARE_CONFIG, row.subset

    def test_the_corpora_read_in_place_are_v2s_builds(self):
        """V2's table builds and verifies them: the midtrain reads exactly V2's ten, and stage 1's
        ai_safety corpus is among them."""
        v2_subsets = {row.subset for row in corpora_table.read_corpora_table(V2_CORPORA_TABLE)}
        assert set(blend_subsets(OmegaConf.load(MIDTRAIN).dataset.data_path)) == v2_subsets
        assert READ_FROM_V2 in v2_subsets

    def test_climbmix_is_source_sliced_like_the_other_three_stage_arms(self, corpora_rows):
        (climbmix,) = [row for row in corpora_rows if row.shard_mode == "slice"]
        assert climbmix.subset == f"climbmix_full{SUFFIX}"
        assert climbmix.shards == 8

    def test_the_hold_and_the_pin_move_together(self, corpora_rows, prepare_config):
        assert_hold_and_pin_move_together(prepare_config["revision"], corpora_rows, "v2e2e")

    def test_the_build_refuses_while_document_counts_are_unknown(self, corpora_rows):
        if not pending_subsets(corpora_rows):
            pytest.skip("every corpus has its document count; the refusal no longer applies")
        build = dry_run_build(CORPORA_TABLE, "pretraining")
        assert build.returncode != 0, "the build must not proceed with a PENDING count"
        assert "document count is PENDING" in build.stderr

    def test_the_build_plans_every_corpus_once_the_counts_are_filled(self, corpora_rows):
        """The eight ClimbMix slices and the four whole corpora, each prepared then tokenized, under the arm's job
        names."""
        if pending_subsets(corpora_rows):
            pytest.skip("document counts are PENDING; the build cannot be planned yet")
        build = dry_run_build(CORPORA_TABLE, "pretraining")
        output = build.stdout + build.stderr
        assert build.returncode == 0, output
        assert "SUBMITTED 24 jobs" in output and "nothing was actually submitted" in output
        assert set(re.findall(r"^=== (\S+) \(", output, re.MULTILINE)) == {row.subset for row in corpora_rows}
        names = re.findall(r"--job-name=(\S+)", output)
        assert len(names) == 24
        assert all(re.fullmatch(r"cp-30b_filtered_gpt55_4plus_v2e2e-(prep|tok)-\S+", name) for name in names)

    def test_climbmix_slices_cover_the_corpus_exactly_once(self, corpora_rows):
        (climbmix,) = [row for row in corpora_rows if row.shard_mode == "slice"]
        if climbmix.docs is None:
            pytest.skip("the ClimbMix count is PENDING; the slice ranges cannot be computed yet")
        build = dry_run_build(CORPORA_TABLE, "pretraining", climbmix.subset)
        assert_slices_cover_the_corpus(build.stdout + build.stderr, climbmix.subset, climbmix.docs, climbmix.shards)

    def test_the_tokenizer_that_builds_the_corpora_is_the_one_training_reads(self, merged, prepare_config):
        assert merged[PRETRAIN].tokenizer.tokenizer_model == prepare_config["tokenizer"]
        v2_prepare = yaml.safe_load(
            (_V2_DIR / "data" / "control-pretraining-datasets-filtered-gpt55-4plus-v2.yaml").read_text()
        )
        assert prepare_config["dataset"] == v2_prepare["dataset"]
        assert prepare_config["tokenizer"] == v2_prepare["tokenizer"]
        assert "output-dir" not in prepare_config, "an output-dir would collapse every subset onto one directory"


class TestBudgetsAndCadence:
    """Every model sees the unfiltered baseline's token budget, and its checkpoints sit at the
    baseline's iterations."""

    def test_stage_one_matches_the_baseline_budget_at_its_width(self, merged):
        mine, baseline = merged[PRETRAIN], merged[BASELINE_PRETRAIN]
        for field in ("train_iters", "global_batch_size", "micro_batch_size"):
            assert getattr(mine.train, field) == getattr(baseline.train, field), field
        assert mine.dataset.seq_length == baseline.dataset.seq_length == 8192
        assert data_parallel_size(mine, GPUS) == 512
        assert mine.train.global_batch_size // data_parallel_size(mine, GPUS) == 4

    def test_the_midtrain_matches_the_baseline_budget_at_its_width(self, merged):
        mine, baseline = merged[MIDTRAIN], merged[BASELINE_MIDTRAIN]
        for field in ("train_iters", "global_batch_size", "micro_batch_size"):
            assert getattr(mine.train, field) == getattr(baseline.train, field), field
        assert data_parallel_size(mine, GPUS) == 256

    def test_fourteen_stage_one_saves_at_the_baselines_iterations(self, merged):
        checkpoint = merged[PRETRAIN].checkpoint
        assert checkpoint.save_interval == merged[BASELINE_PRETRAIN].checkpoint.save_interval == 2264
        assert (merged[PRETRAIN].train.train_iters - 1) // checkpoint.save_interval + 1 == 14
        assert checkpoint.most_recent_k == -1 and checkpoint.save_optim and checkpoint.save_rng

    def test_six_midtrain_saves(self, merged):
        checkpoint = merged[MIDTRAIN].checkpoint
        assert checkpoint.save_interval == 600
        assert (merged[MIDTRAIN].train.train_iters - 1) // checkpoint.save_interval + 1 == 6

    @pytest.mark.parametrize("path", [PRETRAIN, MIDTRAIN], ids=lambda p: p.stem)
    def test_segment_rollover_and_save_crossings(self, merged, path):
        assert_segment_exit_posture(merged[path], path.name, 1400)
        assert merged[path].checkpoint.ckpt_assume_constant_structure is False
        assert merged[path].logger.tensorboard_dir is None


class TestTheManifestsKnowTheArm:
    def test_the_hub_manifest_publishes_both_stages(self):
        manifest = yaml.safe_load(HUB_MANIFEST.read_text())
        (model,) = [m for m in manifest["models"] if m["repo"] == HUB_REPO]
        assert model["private"] is True and model["reasoning"] is False and model["strict"] is True
        assert model["history"] == []
        assert [(s["name"], s["config"], s["revision"], s["default"]) for s in model["stages"]] == [
            ("pretraining", str(PRETRAIN.relative_to(_REPO_ROOT)), "pretraining_iter_{iteration}", False),
            ("midtraining", str(MIDTRAIN.relative_to(_REPO_ROOT)), "midtraining_iter_{iteration}", True),
        ]

    def test_the_hub_manifest_counts_the_arms_tokens_from_scratch(self):
        importable(_REPO_ROOT / "scripts" / "hub")
        import publish_models

        manifest = publish_models.load_manifest(HUB_MANIFEST, _REPO_ROOT)
        (model,) = [m for m in manifest.models if m.repo == HUB_REPO]
        pretraining, midtraining = model.stages
        assert pretraining.tokens_before == 0
        assert midtraining.tokens_before == 29881 * 16_777_216
        assert (
            midtraining.tokens_before + midtraining.train_iters * midtraining.tokens_per_iteration == 553_765_568_512
        )

    def test_the_bucket_manifest_archives_both_stages(self):
        stage_configs = yaml.safe_load(BUCKET_MANIFEST.read_text())["stage_configs"]
        for path in (PRETRAIN, MIDTRAIN):
            assert str(path.relative_to(_REPO_ROOT)) in stage_configs


class TestTheLossGate:
    """The fast posture changes numerical precision, so stage 1 runs under a loss gate pre-registered
    in ``loss_gate.yaml`` before the run exists. These tests pin the frozen spec, read through the
    real gate tool, so an edit to it after the run starts cannot pass unnoticed."""

    # name: (references, first, last, window, lm_loss_delta, lm_loss_tolerance), as frozen.
    FROZEN = {
        "L1": (("baseline", "baseline_dp256", "broad"), 1, 1200, 50, 0.055033, None),
        "L2": (("baseline", "broad"), 1, 2000, 100, 0.052204, None),
        "L2b": (("baseline", "broad"), 1201, 2000, 100, 0.004613, 0.02),
    }
    # The as-is stage-1 runs the gates compare against, by SLURM job.
    REFERENCE_JOBS = {"baseline": "6107666", "baseline_dp256": "6107671", "broad": "6354507"}

    @pytest.fixture(scope="class")
    def spec(self):
        return load_gate_spec(LOSS_GATE)

    def test_the_gates_are_the_frozen_ones(self, spec):
        assert spec.wandb is True, "the gates were calibrated on W&B full-precision values"
        frozen = {
            name: (g.references, g.first, g.last, g.window, g.lm_loss_delta, g.lm_loss_tolerance)
            for name, g in spec.gates.items()
        }
        assert frozen == self.FROZEN

    def test_the_references_are_the_as_is_stage_one_runs(self, spec):
        assert {name: log.name for name, log in spec.references.items()} == {
            name: f"train-{job}.out" for name, job in self.REFERENCE_JOBS.items()
        }

    @pytest.mark.skipif(not Path("/projects/a5k").is_dir(), reason="the reference logs live on Isambard's /projects")
    def test_every_reference_log_exists(self, spec):
        for name, log in spec.references.items():
            assert log.is_file(), f"{name}: {log}"

    def test_every_gate_decides_before_the_first_save(self, spec, merged):
        """A gate that fails must stop the run before it writes a fast-posture checkpoint."""
        first_save = merged[PRETRAIN].checkpoint.save_interval
        for gate in spec.gates.values():
            assert gate.last < first_save, gate.name

    def test_the_policy_debugs_the_fast_posture_rather_than_changing_it(self):
        """A failure stops stage 1 for debugging in its own posture (Kyle, 2026-10-01): no other posture is
        staged to restart in."""
        text = LOSS_GATE.read_text()
        assert "debugged in the fast posture" in text
        assert "_precise" not in text and not list(_ARM_DIR.rglob("*precise*"))


class TestTheScoreGate:
    """The probe's memory and speed gates are pre-registered in ``score_gate.yaml`` and evaluated by the
    probe itself, so its exit status carries them: a score step's own status says only that the log could
    be scored. These tests pin the spec, read through the real gate tool, to the thresholds the README
    states, and tie the score files it reads to the ones the probe writes."""

    FROZEN = {
        "fast_memory": MemoryGate("fast_memory", "fast.score.json", 85.5, 0),
        "fast_speed": SpeedGate("fast_speed", "fast.score.json", "as_is.score.json", 6.31, 4.5, 5.25),
    }

    def test_the_gates_are_the_frozen_ones(self):
        assert load_score_gates(SCORE_GATE) == self.FROZEN

    def test_the_probe_evaluates_them_in_a_gating_step(self):
        text = PROBE_SBATCH.read_text()
        assert "python scripts/telemetry/score_gate.py --spec $ARM/score_gate.yaml --scores-dir $OUT" in text
        assert re.search(r"^gate score_gate ", text, re.M)

    def test_every_score_a_gate_reads_is_one_the_probe_writes(self):
        text = PROBE_SBATCH.read_text()
        assert "> $OUT/$step.score.json" in PROBE_JOB.read_text(), (
            "score writes <step>.score.json into the scores directory"
        )
        windows = dict(re.findall(r'^score (\w+) "\$PROBE/[^"]+" (\d+ \d+)$', text, re.M))
        assert set(windows) == {"fast", "as_is"}
        for gate in load_score_gates(SCORE_GATE).values():
            files = [gate.score] if isinstance(gate, MemoryGate) else [gate.candidate, gate.reference]
            assert set(files) <= {f"{step}.score.json" for step in windows}, gate.name
        assert windows["fast"] == windows["as_is"], "the speed gate compares the two over one window"


def sbatch_value(sbatch: Path, name: str) -> str:
    """A NAME=value assignment in ``sbatch``."""
    (value,) = [line.split("=", 1)[1] for line in sbatch.read_text().splitlines() if line.startswith(f"{name}=")]
    return value


class TestTheProbe:
    """Each probe config is the config it measures, changed only in what a probe must change: the
    blend (the baseline's, so the fast posture reads the baseline's batches), the length, where it
    saves, and its W&B run."""

    PROBE_FIELDS = {
        "train.exit_interval",
        "checkpoint.load",
        "checkpoint.save",
        "logger.wandb_exp_name",
        "logger.wandb_save_dir",
    }

    def test_fast_is_the_arms_stage_one_on_the_baselines_data(self, merged):
        allowed = {*self.PROBE_FIELDS, "dataset.data_path", "checkpoint.save_interval", "checkpoint.most_recent_k"}
        assert_only_these_fields_differ(merged[PROBE_FAST], merged[PRETRAIN], allowed, "fast probe")
        assert merged[PROBE_FAST].dataset.data_path == merged[BASELINE_PRETRAIN].dataset.data_path

    def test_the_as_is_rerun_is_the_baseline(self, merged):
        assert_only_these_fields_differ(
            merged[PROBE_AS_IS], merged[BASELINE_PRETRAIN], self.PROBE_FIELDS, "as-is probe"
        )

    def test_the_handoff_is_the_arms_midtrain_from_the_fast_probe(self, merged):
        allowed = {*self.PROBE_FIELDS, "checkpoint.pretrained_checkpoint"}
        assert_only_these_fields_differ(merged[PROBE_HANDOFF], merged[MIDTRAIN], allowed, "handoff probe")
        assert merged[PROBE_HANDOFF].checkpoint.pretrained_checkpoint == merged[PROBE_FAST].checkpoint.save

    def test_fast_crosses_three_saves_and_trains_after_each(self, merged):
        """A save that leaves memory behind shows only in the forward after it."""
        train, checkpoint = merged[PROBE_FAST].train, merged[PROBE_FAST].checkpoint
        assert (train.exit_interval - 1) // checkpoint.save_interval >= 3
        assert checkpoint.most_recent_k == 1 and checkpoint.load is None

    def test_no_probe_loads_or_writes_a_production_directory(self, merged):
        productions = {merged[path].checkpoint.save for path in (PRETRAIN, MIDTRAIN, BASELINE_PRETRAIN)}
        for path in (PROBE_FAST, PROBE_AS_IS, PROBE_HANDOFF):
            checkpoint = merged[path].checkpoint
            assert checkpoint.load is None, path.name
            assert checkpoint.save in (None, sbatch_value(PROBE_SBATCH, "SCRATCH")), path.name
            assert checkpoint.save not in productions, path.name
        names = [merged[path].logger.wandb_exp_name for path in (PROBE_FAST, PROBE_AS_IS, PROBE_HANDOFF)]
        assert len(set(names)) == 3 and all("_probe_" in name for name in names)

    def test_every_probe_runs_at_the_production_width(self, merged):
        gpus = int(sbatch_value(PROBE_SBATCH, "NODES")) * 4
        assert gpus == GPUS
        for path in (PROBE_FAST, PROBE_AS_IS):
            assert data_parallel_size(merged[path], gpus) == 512, path.name
        assert data_parallel_size(merged[PROBE_HANDOFF], gpus) == 256

    def test_the_sbatch_names_the_save_the_configs_use(self, merged):
        assert sbatch_value(PROBE_SBATCH, "SCRATCH") == merged[PROBE_FAST].checkpoint.save

    def test_the_sbatch_reads_the_gates_baseline_log(self):
        """probe.sbatch takes its parity reference from loss_gate.yaml's one `baseline:` line."""
        assert (
            sbatch_value(PROBE_SBATCH, "BASELINE_LOG")
            == """$(awk '$1 == "baseline:" {print $2}' "$ARM/loss_gate.yaml")"""
        )
        (line,) = [line.split() for line in LOSS_GATE.read_text().splitlines() if line.split()[:1] == ["baseline:"]]
        assert Path(line[1]) == load_gate_spec(LOSS_GATE).references["baseline"]

    def test_every_launch_has_a_time_limit_and_together_they_fit_the_job(self):
        """A hung step ends at its own limit; the limits must leave the later steps their time."""
        text = PROBE_SBATCH.read_text()
        launches = [line for line in text.replace("\\\n", " ").splitlines() if re.match(r"\s*(if )?launch \w", line)]
        assert {re.match(r"\s*(?:if )?launch (\w+)", line).group(1) for line in launches} == {
            "fast",
            "handoff",
            "as_is",
        }
        limits = [
            int(sbatch_value(PROBE_SBATCH, name)) for line in launches for name in re.findall(r'"\$(LIMIT_\w+)"', line)
        ]
        assert len(limits) == len(launches), "every launch passes one LIMIT_ variable"
        hours, minutes, seconds = map(int, re.search(r"^#SBATCH --time=(\d+):(\d+):(\d+)$", text, re.M).groups())
        # The NVLink sweep, the container starts for scoring and the parity tests take the rest.
        assert sum(limits) + 600 <= hours * 3600 + minutes * 60 + seconds

    def _run_probe_sbatch(self, tmp_path, env_extra: dict[str, str]) -> subprocess.CompletedProcess:
        """Run the real probe.sbatch up to its refusals, with a stub isambard_sbatch on PATH (SLURM
        submission is the untestable boundary) and a minimal environment carrying only ``env_extra``."""
        if Path(sbatch_value(PROBE_SBATCH, "SCRATCH")).parent.exists():
            pytest.skip("the probe's scratch directory exists, and the sbatch refuses before the check under test")
        repo = tmp_path / "repo"
        (repo / "scripts" / "training").mkdir(parents=True)
        (repo / "REVISION").write_text("test\n")
        shutil.copy(_REPO_ROOT / "scripts" / "training" / "launch_environment.py", repo / "scripts" / "training")
        shutil.copy(PROBE_JOB, repo / "scripts" / "training")
        bindir = tmp_path / "bin"
        bindir.mkdir()
        stub = bindir / "isambard_sbatch"
        stub.write_text("#!/bin/bash\nexit 0\n")
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
        env = {"PATH": f"{bindir}:/usr/bin:/bin", "GEODESIC_REPO_DIR": str(repo), "SLURM_JOB_ID": "1", **env_extra}
        return subprocess.run(["bash", str(PROBE_SBATCH)], capture_output=True, text=True, env=env, timeout=60)

    @pytest.mark.parametrize(
        "name",
        [
            "ISAMBARD_FP32_SSM_STATE",
            "ISAMBARD_CUDA_MAX_CONNECTIONS",
            "TRAIN_PERSISTENT_TRITON_CACHE",
            "GEODESIC_CONTAINER_SIF",
        ],
    )
    def test_a_setting_inherited_from_the_submitting_shell_is_refused(self, tmp_path, name):
        result = self._run_probe_sbatch(tmp_path, {name: "0", "ISAMBARD_SBATCH_MAX_NODES": "256"})
        assert result.returncode == 1
        assert f"launch settings inherited from the submitting shell: {name}" in result.stderr

    def test_the_submission_wrappers_and_tunnels_own_variables_pass(self, tmp_path):
        """Past the refusal the sbatch reads the loss gate's baseline log, which the stub repo lacks."""
        result = self._run_probe_sbatch(
            tmp_path, {"ISAMBARD_SBATCH_MAX_NODES": "256", "ISAMBARD_SBATCH_FORCE": "0", "ISAMBARD_TUNNEL_NAME": "t"}
        )
        assert result.returncode == 1
        assert "inherited" not in result.stderr and "the loss gate's baseline log" in result.stderr

    def test_every_file_the_sbatch_reads_exists(self):
        text = PROBE_SBATCH.read_text()
        named = {Path(m) for m in re.findall(r'"\$(?:PROBE|ARM)/([^"]+)"', text)}
        resolved = set()
        for name in named:
            resolved.add((PROBE_DIR / name) if (PROBE_DIR / name).exists() else (_ARM_DIR / name))
        assert {path.name for path in resolved} == {
            PROBE_FAST.name,
            PROBE_AS_IS.name,
            PROBE_HANDOFF.name,
            PRETRAIN_ENV.name,
            LOSS_GATE.name,
            MIDTRAIN_ENV.name,
        }
        for path in resolved:
            assert path.is_file(), path


def assert_differs_only_in(candidate, reference, fields: set[str], label: str) -> None:
    """Assert that ``candidate`` differs from ``reference`` in no field outside ``fields``; a listed field may hold
    its reference's value (a lever the reference already sets, say), which ``assert_levers_are_set`` covers."""
    flat_candidate, flat_reference = flatten_merged_config(candidate), flatten_merged_config(reference)
    differing = {key for key in fields if flat_candidate.get(key) != flat_reference.get(key)}
    assert_only_these_fields_differ(candidate, reference, differing, label)


class TestTheMidtrainingProbe:
    """The midtraining probe measures the fast midtraining configuration on the baseline's stage 2 at production
    width: each config is the config it measures, changed only in what a probe must change (its length, its
    checkpoints and its W&B run), so every run reads production's batches from production's warm start."""

    PROBE_FIELDS = {
        "train.exit_interval",
        "checkpoint.load",
        "checkpoint.save",
        "logger.wandb_exp_name",
        "logger.wandb_save_dir",
    }

    def test_fast_mid_is_the_baselines_stage_two_in_the_arms_midtraining_configuration(self, merged):
        fields = {*MIDTRAIN_LEVERS, *self.PROBE_FIELDS, "checkpoint.save_interval", "checkpoint.most_recent_k"}
        assert_differs_only_in(merged[PROBE_MID_FAST], merged[BASELINE_MIDTRAIN], fields, "fast midtraining probe")
        assert_levers_are_set(merged[PROBE_MID_FAST], MIDTRAIN_LEVERS, "fast midtraining probe")

    def test_both_midtraining_configs_keep_the_baselines_full_recompute(self, merged):
        for path in (MIDTRAIN, PROBE_MID_FAST):
            model, baseline = merged[path].model, merged[BASELINE_MIDTRAIN].model
            assert model.recompute_granularity == baseline.recompute_granularity == "full", path.name
            assert (model.recompute_method, model.recompute_num_layers) == ("uniform", 1), path.name

    def test_the_as_is_rerun_is_the_baselines_stage_two(self, merged):
        assert_differs_only_in(
            merged[PROBE_MID_AS_IS], merged[BASELINE_MIDTRAIN], self.PROBE_FIELDS, "as-is midtraining probe"
        )

    def test_the_handoff_is_the_as_is_cpt_from_the_fast_mid_save(self, merged):
        fields = {*self.PROBE_FIELDS, "checkpoint.pretrained_checkpoint"}
        assert_differs_only_in(merged[PROBE_MID_HANDOFF], merged[V2_CPT_LINK1], fields, "CPT handoff probe")
        fast = merged[PROBE_MID_FAST]
        assert merged[PROBE_MID_HANDOFF].checkpoint.pretrained_checkpoint == (
            f"{fast.checkpoint.save}/iter_{fast.train.exit_interval:07d}"
        )

    def test_fast_mid_crosses_three_saves_and_trains_after_each(self, merged):
        cfg = merged[PROBE_MID_FAST]
        saves = range(cfg.checkpoint.save_interval, cfg.train.exit_interval + 1, cfg.checkpoint.save_interval)
        assert len([save for save in saves if save < cfg.train.exit_interval]) >= 3
        assert cfg.checkpoint.most_recent_k == 1

    def test_no_midtraining_probe_resumes_or_writes_a_production_directory(self, merged):
        for path in (PROBE_MID_FAST, PROBE_MID_AS_IS, PROBE_MID_HANDOFF):
            assert merged[path].checkpoint.load is None, path.name
        assert merged[PROBE_MID_AS_IS].checkpoint.save is None
        assert merged[PROBE_MID_HANDOFF].checkpoint.save is None
        assert merged[PROBE_MID_FAST].checkpoint.save == sbatch_value(PROBE_MID_SBATCH, "SCRATCH")
        assert "/v2e2e_probe_midtrain/" in merged[PROBE_MID_FAST].checkpoint.save

    def test_every_midtraining_probe_runs_at_production_width(self, merged):
        gpus = 4 * int(sbatch_value(PROBE_MID_SBATCH, "NODES"))
        for path in (PROBE_MID_FAST, PROBE_MID_AS_IS):
            assert data_parallel_size(merged[path], gpus) == data_parallel_size(merged[BASELINE_MIDTRAIN], GPUS) == 256
        handoff = merged[PROBE_MID_HANDOFF]
        assert handoff.train.global_batch_size % data_parallel_size(handoff, gpus) == 0

    def test_the_gates_are_the_pre_registered_ones(self):
        assert load_score_gates(SCORE_GATE_MIDTRAIN) == {
            "fast_mid_memory": MemoryGate("fast_mid_memory", "fast_mid.score.json", 85.5, 0),
            "fast_mid_speed": SpeedGate(
                "fast_mid_speed", "fast_mid.score.json", "as_is_mid.score.json", 6.43, 4.946154, 4.946154
            ),
            "fast_mid_first_loss": FirstLossGate(
                "fast_mid_first_loss", "fast_mid.score.json", "as_is_mid.score.json", 0.005
            ),
            "fast_mid_loss_shift": LossShiftGate("fast_mid_loss_shift", "parity_band.json", -0.001, 0.003, 3, 0.001),
        }
        assert 6.43 / 1.3 == pytest.approx(4.946154, abs=1e-6)

    def test_the_parity_reference_is_the_production_midtraining_run(self):
        reference = Path(sbatch_value(PROBE_MID_SBATCH, "REFERENCE_LOG"))
        assert reference.name == f"train-{BASELINE_MIDTRAIN_JOB}.out"
        assert sbatch_value(PROBE_MID_SBATCH, "PARITY_GATED") == '"51 500"'
        assert sbatch_value(PROBE_MID_SBATCH, "PARITY_REPORTED") == '"1 50"'

    def test_every_score_a_gate_reads_is_one_the_probe_writes(self):
        text = PROBE_MID_SBATCH.read_text()
        windows = dict(re.findall(r'^score (\w+) "\$PROBE/[^"]+" (\d+ \d+)$', text, re.M))
        assert set(windows) == {"fast_mid", "as_is_mid"} and windows["fast_mid"] == windows["as_is_mid"]
        assert "python scripts/telemetry/score_gate.py --spec $ARM/score_gate_midtrain.yaml --scores-dir $OUT" in text
        assert re.search(r"^gate score_gate ", text, re.M)
        for gate in load_score_gates(SCORE_GATE_MIDTRAIN).values():
            if isinstance(gate, LossShiftGate):
                assert gate.report == "parity_band.json", gate.name
                continue
            files = [gate.score] if isinstance(gate, MemoryGate) else [gate.candidate, gate.reference]
            assert set(files) <= {f"{step}.score.json" for step in windows}, gate.name

    def test_the_loss_shift_gate_reads_the_band_the_probe_writes_before_it(self):
        """The band over the gated range is written to parity_band.json, reported rather than gating, and the score
        gate that reads it runs after it."""
        text = PROBE_MID_SBATCH.read_text()
        band = 'parity_band parity_band "$PARITY_GATED"'
        assert band in text and "> $OUT/$name.json" in PROBE_JOB.read_text()
        assert re.search(r"^note parity_band ", text, re.M) and not re.search(r"^record parity_band", text, re.M)
        assert text.index(band) < text.index("score_gate.py --spec $ARM/score_gate_midtrain.yaml")

    def test_the_probe_watches_fast_mid_against_the_midtrainings_stop_conditions(self):
        text = PROBE_MID_SBATCH.read_text()
        assert "python scripts/telemetry/run_watch.py --spec $ARM/watch_midtrain.yaml --log $OUT/fast_mid.out" in text
        assert re.search(r"^gate watch ", text, re.M)

    def test_every_launch_has_a_time_limit_and_together_they_fit_the_job(self):
        text = PROBE_MID_SBATCH.read_text()
        launches = [line for line in text.replace("\\\n", " ").splitlines() if re.match(r"\s*(if )?launch \w", line)]
        steps = {re.match(r"\s*(?:if )?launch (\w+)", line).group(1) for line in launches}
        assert steps == {"fast_mid", "handoff_cpt", "as_is_mid"}
        limits = [
            int(sbatch_value(PROBE_MID_SBATCH, name))
            for line in launches
            for name in re.findall(r'"\$(LIMIT_\w+)"', line)
        ]
        assert len(limits) == len(launches)
        hours, minutes, seconds = map(int, re.search(r"^#SBATCH --time=(\d+):(\d+):(\d+)$", text, re.M).groups())
        assert sum(limits) + 600 <= hours * 3600 + minutes * 60 + seconds

    @pytest.mark.skipif(not Path("/projects/a5k").is_dir(), reason="the reference log lives on Isambard's /projects")
    def test_the_reference_log_exists(self):
        assert Path(sbatch_value(PROBE_MID_SBATCH, "REFERENCE_LOG")).is_file()


class TestTheWatch:
    """Each stage runs under a watch spec read by scripts/telemetry/run_watch.py: both stop on a result the
    gradient NaN check rejected (it is on in both stages' configurations), on a non-finite grad norm or lm loss (the
    loss NaN check is off in both, so a NaN loss is an iteration line without lm loss), on an iteration counted as
    nan or skipped and on allocator retries; stage 1 also runs the pre-registered loss gate not waived (L2b) and
    flags a growing offset and a block outside the broad arm's envelope, the midtraining flags loss spikes."""

    @pytest.mark.parametrize("path", [WATCH_PRETRAIN, WATCH_MIDTRAIN], ids=lambda path: path.stem)
    def test_both_stages_stop_on_every_sign_of_a_bad_step(self, path):
        spec = load_watch_spec(path)
        assert spec.stop_on_rejected_result and spec.stop_on_non_finite_grad_norm
        assert spec.stop_on_non_finite_lm_loss and spec.stop_on_nan_or_skipped_iterations
        assert spec.max_alloc_retries == 0

    @pytest.mark.parametrize(
        "watch, stage", [(WATCH_PRETRAIN, PRETRAIN), (WATCH_MIDTRAIN, MIDTRAIN)], ids=["pretrain", "midtrain"]
    )
    def test_each_stage_stops_a_segment_launched_without_the_env_file_its_readme_command_names(self, watch, stage):
        settings = load_watch_spec(watch).launch_settings
        assert settings == stage.with_suffix(".env")
        assert (
            f"ISAMBARD_ENV_OVERRIDES=$PWD/{settings.relative_to(_REPO_ROOT)}" in (_ARM_DIR / "README.md").read_text()
        )

    def test_stage_one_runs_every_pre_registered_loss_gate_kyle_did_not_waive(self):
        """L1 and L2 stay in the pre-registered spec but are waived (Kyle, 2026-10-02, after L1 failed the first
        launch in a warmup-descent window), so the watch runs L2b alone."""
        spec = load_watch_spec(WATCH_PRETRAIN)
        assert spec.loss_gate_spec == LOSS_GATE
        assert tuple(load_gate_spec(LOSS_GATE).gates) == ("L1", "L2", "L2b")
        assert spec.loss_gates == ("L2b",)
        assert spec.growing_offset == GrowingOffset("L2b", 3, 0.01)

    def test_stage_one_compares_each_save_interval_with_the_broad_arms_envelope(self, merged):
        envelope = load_watch_spec(WATCH_PRETRAIN).block_envelope
        assert envelope.block_iterations == merged[PRETRAIN].checkpoint.save_interval
        assert envelope.consecutive_blocks == 2
        gate_logs = load_gate_spec(LOSS_GATE).references
        assert envelope.reference.logs[0] == gate_logs["baseline"]
        assert envelope.envelope.logs[0] == gate_logs["broad"]

    @pytest.mark.skipif(not Path("/projects/a5k").is_dir(), reason="the reference logs live on Isambard's /projects")
    def test_wandb_supplies_exactly_the_block_the_baselines_logs_lack(self, merged):
        """The broad arm as the candidate stands for a stage 1 that has logged every iteration. Read from the logs
        alone, every block it completes is compared except block 11, whose baseline segment (22641-24904, W&B
        mpf5lqoj) left no log, and the spec reads exactly that block from that run's W&B history."""
        envelope = load_watch_spec(WATCH_PRETRAIN).block_envelope
        assert envelope.reference.wandb_blocks == {11: "geodesic/megatron_training/mpf5lqoj"}
        assert envelope.envelope.wandb_blocks == {}
        logs_only = replace(envelope, reference=replace(envelope.reference, wandb_blocks={}))
        comparisons = block_comparisons(logs_only, latest_records(read_segments(envelope.envelope.logs)))
        full_blocks = merged[PRETRAIN].train.train_iters // envelope.block_iterations
        assert [c.block for c in comparisons] == list(range(1, full_blocks + 1))
        not_compared = {c.block: c.not_compared for c in comparisons if c.offsets is None}
        assert not_compared == {11: f"the reference logs 16 of {envelope.block_iterations} iterations"}

    @pytest.mark.parametrize(
        "guard, watch, stage",
        [(GUARD_PRETRAIN, WATCH_PRETRAIN, PRETRAIN), (GUARD_MIDTRAIN, WATCH_MIDTRAIN, MIDTRAIN)],
        ids=["pretrain", "midtrain"],
    )
    def test_each_stage_is_guarded_by_its_watch_under_the_job_name_its_readme_command_submits(
        self, merged, guard, watch, stage
    ):
        config = load_guard_config(guard)
        assert config.watch == watch
        assert f"--job-name={config.job_name} --dependency=singleton" in (_ARM_DIR / "README.md").read_text()
        assert config.final_iteration == merged[stage].train.train_iters
        assert str(config.record).startswith("/projects/a5k/public/logs/")

    @pytest.mark.parametrize("guard", [GUARD_PRETRAIN, GUARD_MIDTRAIN], ids=["pretrain", "midtrain"])
    def test_each_stage_is_submitted_from_the_frozen_copy_and_never_requeued(self, guard):
        """A requeued segment keeps its job ID and log, so a requeue from scratch would inherit the gates its first
        run passed; and the code that trains and judges the stage is the frozen copy's."""
        job_name = load_guard_config(guard).job_name
        blocks = (_ARM_DIR / "README.md").read_text().split("```")[1::2]
        (command,) = [block for block in blocks if f"--job-name={job_name}" in block]
        assert 'cd "$SNAP" &&' in command and "--no-requeue" in command

    def test_stage_one_is_held_before_its_first_save_while_a_gate_is_unevaluated(self, merged):
        """From iteration 2200, before the first save: a gate decided at 2000 that is still unevaluated by then must
        not let a save through, and a tick that comes after the save still holds."""
        config = load_guard_config(GUARD_PRETRAIN)
        assert config.hold_from == 2200 < merged[PRETRAIN].checkpoint.save_interval
        assert max(gate.last for gate in load_gate_spec(LOSS_GATE).gates.values()) < config.hold_from
        assert load_guard_config(GUARD_MIDTRAIN).hold_from is None

    def test_the_midtraining_flags_a_calibrated_loss_spike_and_runs_no_loss_gate(self):
        spec = load_watch_spec(WATCH_MIDTRAIN)
        assert spec.loss_spike == LossSpike(0.108, 50)
        assert spec.loss_gate_spec is None and spec.loss_gates == ()

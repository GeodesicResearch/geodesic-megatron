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
  fields at their measured values (the posture stage 1 trains in, which changes numerical precision);
- the precision-preserving stage-1 variant, which runs only if the fast posture fails its loss gate:
  the same, less the precision levers;
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
import stat
import subprocess
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf
from scripts.telemetry.loss_gate import load_gate_spec
from scripts.telemetry.score_gate import MemoryGate, SpeedGate, load_score_gates

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_pretrain_config
from tests.unit_tests.campaign_config import (
    FAST_PRETRAIN_LAUNCHER_SETTINGS,
    FAST_PRETRAIN_LEVERS,
    FAST_PRETRAIN_PRECISION_LEVERS,
    FAST_PRETRAIN_SSM_SETTING,
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
    merge_onto_recipe,
    pending_subsets,
)
from tests.unit_tests.corpora_fixtures import corpora_table, importable
from tests.unit_tests.launcher_source import env_override_entries


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CAMPAIGN_DIR = _REPO_ROOT / "configs" / "control_pretraining"
_ARM_DIR = _CAMPAIGN_DIR / "30b_filtered_gpt55_4plus_v2e2e"
_V2_DIR = _CAMPAIGN_DIR / "30b_filtered_gpt55_4plus_v2"
_BASELINE_DIR = _CAMPAIGN_DIR / "30b_baseline"

PRETRAIN = _ARM_DIR / "nemotron_nano_30b_filtered_gpt55_4plus_v2e2e_pretrain.yaml"
PRETRAIN_ENV = PRETRAIN.with_suffix(".env")
PRETRAIN_PRECISE = _ARM_DIR / "nemotron_nano_30b_filtered_gpt55_4plus_v2e2e_pretrain_precise.yaml"
PRETRAIN_PRECISE_ENV = PRETRAIN_PRECISE.with_suffix(".env")
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
PROBE_PRECISE = PROBE_DIR / "probe_precise.yaml"
PROBE_AS_IS = PROBE_DIR / "probe_as_is.yaml"
PROBE_HANDOFF = PROBE_DIR / "probe_handoff_midtrain.yaml"
PROBE_SBATCH = PROBE_DIR / "probe.sbatch"
HUB_REPO = "geodesic-research/control-pretraining-30b-filtered-gpt55-4plus-v2e2e-base"

SUFFIX = "_filtered_gpt55_4plus_v2e2e"
V2_SUFFIX = "_filtered_gpt55_4plus_v2"
# The corpus stage 1 reads through V2's build: the `_v2e2e` split's rows equal V2's.
READ_FROM_V2 = f"ai_safety_and_adjacent{V2_SUFFIX}"
CLIMBMIX_WEIGHT = 0.698180
# The width stage 1 and the midtrain train at, the baseline's own: DP=512 for stage 1, DP=256 for the
# midtrain at CP2.
GPUS = 512

# Exactly the fields the arm's stage configs may differ in from their counterparts. Data: which
# documents exist. Identity: where the checkpoints and the W&B run go, which MUST differ.
IDENTITY = {"checkpoint.load", "checkpoint.save", "logger.wandb_exp_name"}
PRETRAIN_DIVERGENCE = {"dataset.data_path", *IDENTITY, *FAST_PRETRAIN_LEVERS}
PRECISION_PRESERVING_LEVERS = {
    k: v for k, v in FAST_PRETRAIN_LEVERS.items() if k not in FAST_PRETRAIN_PRECISION_LEVERS
}
PRETRAIN_PRECISE_DIVERGENCE = {"dataset.data_path", *IDENTITY, *PRECISION_PRESERVING_LEVERS}
# Against V2's midtrain the data is the same; the warm start is this arm's own pretraining final.
MIDTRAIN_DIVERGENCE = {*IDENTITY, "checkpoint.pretrained_checkpoint"}


@pytest.fixture(scope="module")
def merged():
    return {
        path: merge_onto_recipe(path, nemotron_3_nano_pretrain_config)
        for path in (
            PRETRAIN,
            PRETRAIN_PRECISE,
            MIDTRAIN,
            BASELINE_PRETRAIN,
            BASELINE_MIDTRAIN,
            V2_MIDTRAIN,
            PROBE_FAST,
            PROBE_PRECISE,
            PROBE_AS_IS,
            PROBE_HANDOFF,
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
        assert_levers_are_set(merged[PRETRAIN], FAST_PRETRAIN_LEVERS, "v2e2e stage 1")

    def test_the_precision_preserving_variant_drops_only_the_precision_levers(self, merged):
        assert_only_these_fields_differ(
            merged[PRETRAIN_PRECISE],
            merged[BASELINE_PRETRAIN],
            PRETRAIN_PRECISE_DIVERGENCE,
            "v2e2e stage 1, precision-preserving",
        )
        assert_levers_are_set(
            merged[PRETRAIN_PRECISE], PRECISION_PRESERVING_LEVERS, "v2e2e stage 1, precision-preserving"
        )
        assert merged[PRETRAIN_PRECISE].mixed_precision == merged[BASELINE_PRETRAIN].mixed_precision

    def test_the_two_postures_differ_only_in_precision_and_identity(self, merged):
        """The variant is a restart from scratch, so it writes to its own directory and W&B run: a
        posture change inside one directory would resume optimizer state that does not load across
        postures."""
        assert_only_these_fields_differ(
            merged[PRETRAIN_PRECISE],
            merged[PRETRAIN],
            {*IDENTITY, *FAST_PRETRAIN_PRECISION_LEVERS},
            "precision-preserving vs fast",
        )

    def test_the_midtrain_is_v2s_from_this_arms_own_pretraining(self, merged):
        assert_only_these_fields_differ(merged[MIDTRAIN], merged[V2_MIDTRAIN], MIDTRAIN_DIVERGENCE, "v2e2e midtrain")
        midtrain = merged[MIDTRAIN].checkpoint
        assert midtrain.pretrained_checkpoint == merged[PRETRAIN].checkpoint.save
        assert midtrain.pretrained_checkpoint != merged[PRETRAIN_PRECISE].checkpoint.save
        assert midtrain.load == midtrain.save != midtrain.pretrained_checkpoint

    def test_every_stage_resumes_from_its_own_directory(self, merged):
        for path, name in (
            (PRETRAIN, "control_pretrain_30b_filtered_gpt55_4plus_v2e2e_pretrain"),
            (PRETRAIN_PRECISE, "control_pretrain_30b_filtered_gpt55_4plus_v2e2e_pretrain_precise"),
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

    def test_the_precision_preserving_env_file_keeps_the_ssm_state_patch_on(self):
        expected = [setting for setting in FAST_PRETRAIN_LAUNCHER_SETTINGS if setting != FAST_PRETRAIN_SSM_SETTING]
        assert env_override_entries(str(PRETRAIN_PRECISE_ENV)) == expected

    def test_the_midtrain_has_no_env_file(self):
        """The midtrain trains in the as-is posture, with the fp32 SSM-state patch on as V2's did; an env
        file beside it would be read as the stage's launcher settings."""
        assert not MIDTRAIN.with_suffix(".env").exists()


class TestTheBlend:
    def test_stage_one_blend_is_well_formed(self, merged):
        assert_blend_is_well_formed(OmegaConf.load(PRETRAIN).dataset.data_path, "pretraining")

    def test_the_two_postures_read_the_same_blend(self):
        assert OmegaConf.load(PRETRAIN_PRECISE).dataset.data_path == OmegaConf.load(PRETRAIN).dataset.data_path

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

    def test_climbmix_shard_weights(self, corpora_rows):
        """Held at the baseline's weights while the ClimbMix row is PENDING (the shards do not exist to be
        measured, and the hold also keeps the launch from running); token-proportional over the built
        shards once it is filled."""
        data_path = OmegaConf.load(PRETRAIN).dataset.data_path
        (climbmix,) = [row for row in corpora_rows if row.subset == f"climbmix_full{SUFFIX}"]
        if climbmix.docs is None:
            baseline = [str(x) for x in OmegaConf.load(BASELINE_PRETRAIN).dataset.data_path]
            mine = [str(x) for x in data_path]
            assert mine[:16:2] == baseline[:16:2], "the placeholder shard weights must be the baseline's"
            assert abs(sum(float(w) for w in mine[:16:2]) - CLIMBMIX_WEIGHT) < 1e-9
        else:
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

    @pytest.mark.parametrize("path", [PRETRAIN, PRETRAIN_PRECISE, MIDTRAIN], ids=lambda p: p.stem)
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

    def test_the_fallback_the_policy_names_is_the_precision_preserving_variant(self):
        assert PRETRAIN_PRECISE.name in LOSS_GATE.read_text()


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
        assert re.search(r"^record score_gate ", text, re.M)

    def test_every_score_a_gate_reads_is_one_the_probe_writes(self):
        text = PROBE_SBATCH.read_text()
        assert "> $OUT/$step.score.json" in text, "score() writes <step>.score.json into the scores directory"
        windows = dict(re.findall(r'^score (\w+) "\$PROBE/[^"]+" (\d+ \d+)$', text, re.M))
        assert set(windows) == {"fast", "as_is", "precise"}
        for gate in load_score_gates(SCORE_GATE).values():
            files = [gate.score] if isinstance(gate, MemoryGate) else [gate.candidate, gate.reference]
            assert set(files) <= {f"{step}.score.json" for step in windows}, gate.name
        assert windows["fast"] == windows["as_is"], "the speed gate compares the two over one window"


def sbatch_value(name: str) -> str:
    """A NAME=value assignment in probe.sbatch."""
    (value,) = [line.split("=", 1)[1] for line in PROBE_SBATCH.read_text().splitlines() if line.startswith(f"{name}=")]
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

    def test_precise_is_the_variant_on_the_baselines_data(self, merged):
        allowed = {*self.PROBE_FIELDS, "dataset.data_path"}
        assert_only_these_fields_differ(
            merged[PROBE_PRECISE], merged[PRETRAIN_PRECISE], allowed, "precision-preserving probe"
        )
        assert merged[PROBE_PRECISE].dataset.data_path == merged[BASELINE_PRETRAIN].dataset.data_path

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
        productions = {
            merged[path].checkpoint.save for path in (PRETRAIN, PRETRAIN_PRECISE, MIDTRAIN, BASELINE_PRETRAIN)
        }
        for path in (PROBE_FAST, PROBE_PRECISE, PROBE_AS_IS, PROBE_HANDOFF):
            checkpoint = merged[path].checkpoint
            assert checkpoint.load is None, path.name
            assert checkpoint.save in (None, sbatch_value("SCRATCH")), path.name
            assert checkpoint.save not in productions, path.name
        names = [
            merged[path].logger.wandb_exp_name for path in (PROBE_FAST, PROBE_PRECISE, PROBE_AS_IS, PROBE_HANDOFF)
        ]
        assert len(set(names)) == 4 and all("_probe_" in name for name in names)

    def test_every_probe_runs_at_the_production_width(self, merged):
        gpus = int(sbatch_value("NODES")) * 4
        assert gpus == GPUS
        for path in (PROBE_FAST, PROBE_PRECISE, PROBE_AS_IS):
            assert data_parallel_size(merged[path], gpus) == 512, path.name
        assert data_parallel_size(merged[PROBE_HANDOFF], gpus) == 256

    def test_the_sbatch_names_the_save_the_configs_use(self, merged):
        assert sbatch_value("SCRATCH") == merged[PROBE_FAST].checkpoint.save

    def test_the_sbatch_reads_the_gates_baseline_log(self):
        """probe.sbatch takes its parity reference from loss_gate.yaml's one `baseline:` line."""
        assert sbatch_value("BASELINE_LOG") == """$(awk '$1 == "baseline:" {print $2}' "$ARM/loss_gate.yaml")"""
        (line,) = [line.split() for line in LOSS_GATE.read_text().splitlines() if line.split()[:1] == ["baseline:"]]
        assert Path(line[1]) == load_gate_spec(LOSS_GATE).references["baseline"]

    def test_every_launch_has_a_time_limit_and_together_they_fit_the_job(self):
        """A hung step ends at its own limit; the limits must leave the later steps their time."""
        text = PROBE_SBATCH.read_text()
        launches = [line for line in text.replace("\\\n", " ").splitlines() if re.match(r"\s*(if )?launch \w", line)]
        assert len(launches) == 4
        limits = [int(sbatch_value(name)) for line in launches for name in re.findall(r'"\$(LIMIT_\w+)"', line)]
        assert len(limits) == 4, "every launch passes one LIMIT_ variable"
        hours, minutes, seconds = map(int, re.search(r"^#SBATCH --time=(\d+):(\d+):(\d+)$", text, re.M).groups())
        # The NVLink sweep, the container starts for scoring and the parity tests take the rest.
        assert sum(limits) + 600 <= hours * 3600 + minutes * 60 + seconds

    def _run_probe_sbatch(self, tmp_path, env_extra: dict[str, str]) -> subprocess.CompletedProcess:
        """Run the real probe.sbatch up to its refusals, with a stub isambard_sbatch on PATH (SLURM
        submission is the untestable boundary) and a minimal environment carrying only ``env_extra``."""
        if Path(sbatch_value("SCRATCH")).parent.exists():
            pytest.skip("the probe's scratch directory exists, and the sbatch refuses before the check under test")
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "REVISION").write_text("test\n")
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
            PROBE_PRECISE.name,
            PROBE_AS_IS.name,
            PROBE_HANDOFF.name,
            PRETRAIN_ENV.name,
            PRETRAIN_PRECISE_ENV.name,
            LOSS_GATE.name,
        }
        for path in resolved:
            assert path.is_file(), path

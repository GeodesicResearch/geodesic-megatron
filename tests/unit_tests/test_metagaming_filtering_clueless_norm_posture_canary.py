# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The posture canary in configs/metagaming_filtering/30b_clueless_norm/probe/posture_canary/ is the test its README
describes.

Each arm is one of Clueless-Norm's two pretraining stages on 16 nodes for 40 iterations: merged through the launcher's
path beside its stage, the fields that differ must be EXACTLY the declared set (a probe's fields, the batch that goes
with the width, the data, and what the arm is there to vary). The scratch corpora the arms read are the ones
hidden_spans.yaml writes from Normal-Norm's corpora; the gate spec judges every arm, with bounds that follow from the
configs; and the submit script plans those runs. Nothing here launches a job: the submit script runs under DRY_RUN=1
in a scratch directory laid out as the frozen copy it requires.

The gate itself is judged on run directories written as the runs would write them (banner, counts and iteration lines
from the training-log fixtures, predictions from predict_masked_counts.py's own writer, and the band reports the gate stage's
own planned commands make from those logs): PASS for a run that behaves as the README predicts, FAIL when a count
differs from its prediction, a resume re-reads a batch, a window-count bound breaks, masking happens while off or the
per-token loss leaves the band, INCONCLUSIVE when a prediction or a clean reference is missing.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from functools import cache
from pathlib import Path

import pytest
import yaml
from scripts.data import predict_masked_counts
from scripts.data.run_training_data import microbatches_per_replica
from scripts.telemetry import loss_parity, score_gate
from scripts.telemetry.run_watch import load_watch_spec
from scripts.telemetry.score_gate import (
    CountBoundsGate,
    CountPredictionGate,
    LogPairingGate,
    LossBandGate,
    MaskingLogGate,
    StopsGate,
    load_score_gates,
    load_verdict,
)
from scripts.training.code_identity import config_code_identity
from scripts.training.launch_width import config_launch_width

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_pretrain_config
from megatron.bridge.training.token_masking.hook import TokenMaskingCounts
from tests.unit_tests import submit_plan
from tests.unit_tests.campaign_config import (
    PROBE_FIELDS,
    assert_blend_is_well_formed,
    assert_only_these_fields_differ,
    merge_onto_recipe,
)
from tests.unit_tests.corpora_fixtures import load_campaign_module
from tests.unit_tests.test_metagaming_filtering_clueless_norm import (
    DATA,
    HIDDEN_TOKEN,
    MASKING,
    MIDTRAIN,
    NORMALISATION,
    PRETRAIN,
    assert_saves_apart_from,
)
from tests.unit_tests.test_run_watch import env_overrides_line
from tests.unit_tests.training_log_fixture import (
    iteration_line,
    sub_once,
    token_masking_banner,
    token_masking_counts_line,
    write_log,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CANARY_RELATIVE = Path("configs/metagaming_filtering/30b_clueless_norm/probe/posture_canary")
CANARY = REPO_ROOT / CANARY_RELATIVE
GATE = CANARY / "gate.yaml"
HIDDEN_SPANS = CANARY / "hidden_spans.yaml"
SCRATCH_DATA = Path("/projects/a5k/public/data/metagaming_filtering_probes/posture_canary")
SCRATCH_SAVES = Path("/projects/a5k/public/checkpoints/megatron/metagaming_filtering_probes")
PREFIX_NAME = "tokenized_base_input_document"

NODES = 16
GPUS = NODES * 4
SPARE_NODES = 1  # the allocation's node beyond the width, for the NVLink sweep
ITERATIONS = 40
SAVED_AT = 20  # the 22% arm's first segment ends here with a save, and its second resumes it
WINDOW = 10  # iterations in a window of the loss band
# Each training stage of the canary: Clueless-Norm's stage, the global batch and data-parallel size 16 nodes give it,
# and the microbatches per replica that leaves, which are the stage's own at its width.
STAGES = {"pretrain": (PRETRAIN, 256, 64, 4), "midtrain": (MIDTRAIN, 64, 32, 2)}
BY_STAGE = pytest.mark.parametrize("stage", list(STAGES))

SEGMENTS = ("spans22", "spans22_resumed")  # one run in two segments, the second resuming the first's save
MASKING_ARMS = (*SEGMENTS, "spans54", "whole", "real")
REFERENCES = tuple(f"unmasked_per_microbatch_{letter}" for letter in "abcd")
UNMASKED_ARMS = ("unmasked", *REFERENCES)
ARMS = (*MASKING_ARMS, *UNMASKED_ARMS)
BLEND_OF = {
    "spans22": "spans22",
    "spans22_resumed": "spans22",
    "spans54": "spans54",
    "whole": "whole",
    "real": "real",
    **dict.fromkeys(UNMASKED_ARMS, "spans22"),
}
SCRATCH_CORPORA = {
    "spans22": ("climbmix_spans22", "ai_safety_spans22"),
    "spans54": ("climbmix_spans54", "ai_safety_spans54"),
    "whole": ("ai_safety_whole",),
}
REAL_SUBSETS = ("ai_safety_and_adjacent", "nemotron_wiki_rewrite_ai_docs", "zyda_ai_docs_long")
HIDDEN_SHARE = {"spans22": 0.22, "spans54": 0.54}
ALL_ARMS = pytest.mark.parametrize(("stage", "arm"), [(stage, arm) for stage in STAGES for arm in ARMS])

# What an arm changes on top of its stage beyond a probe's fields: the batch that keeps the stage's microbatches per
# replica at the canary's width, and its data.
CANARY_FIELDS = {*PROBE_FIELDS, *DATA, "train.global_batch_size"}
WARM_START = "checkpoint.pretrained_checkpoint"
MEASURED_ONLY = "token_masking.masked_validation.token_ids"


def arm_file(stage: str, arm: str) -> Path:
    return CANARY / f"{stage}_{arm}.yaml"


def run_name(stage: str, arm: str) -> str:
    return f"mf_30b_clueless_norm_canary_{stage}_{arm}"


def declared_fields(stage: str, arm: str) -> set[str]:
    """The fields in which the arm may, and must, differ from its stage."""
    fields = set(CANARY_FIELDS)
    if stage == "midtrain":
        fields.add(WARM_START)
    if arm in UNMASKED_ARMS:
        fields |= {*MASKING, MEASURED_ONLY}
    if arm in REFERENCES:
        fields |= NORMALISATION
    return fields


@pytest.fixture(scope="module")
def merged():
    """Every arm, each stage and Normal-Norm's stages as the launcher trains them: merged onto the Nano pretrain
    recipe."""
    paths = {arm_file(stage, arm) for stage in STAGES for arm in ARMS}
    paths |= {path for clueless, *_ in STAGES.values() for path in (clueless.config, clueless.normal_norm)}
    return {path: merge_onto_recipe(path, nemotron_3_nano_pretrain_config) for path in paths}


@pytest.fixture(scope="module")
def gates():
    return load_score_gates(GATE)


def prefixes(cfg) -> list[str]:
    return [str(prefix) for prefix in list(cfg.dataset.data_path)[1::2]]


# --------------------------------------------------------------------------------------
# The arms
# --------------------------------------------------------------------------------------


@ALL_ARMS
class TestEachArmIsItsStageAtTheCanarysWidth:
    def test_it_differs_from_its_stage_in_exactly_the_declared_fields(self, merged, stage, arm):
        clueless = STAGES[stage][0]
        assert_only_these_fields_differ(
            merged[arm_file(stage, arm)], merged[clueless.config], declared_fields(stage, arm), f"{stage} {arm}"
        )

    def test_it_trains_on_16_swept_nodes_with_the_stages_microbatches_per_replica(self, merged, stage, arm):
        clueless, batch, data_parallel_size, microbatches = STAGES[stage]
        width, production = config_launch_width(str(arm_file(stage, arm))), config_launch_width(str(clueless.config))
        assert (width.nodes, width.gpus_per_node, width.data_parallel_size) == (NODES, 4, data_parallel_size)
        assert width.nvlink_links_per_gpu == production.nvlink_links_per_gpu == 18
        cfg, stage_cfg = merged[arm_file(stage, arm)], merged[clueless.config]
        assert cfg.train.global_batch_size == batch
        assert cfg.get_data_parallel_size(width.world_size) == data_parallel_size
        ours = microbatches_per_replica(batch, cfg.train.micro_batch_size, data_parallel_size)
        theirs = microbatches_per_replica(
            stage_cfg.train.global_batch_size, stage_cfg.train.micro_batch_size, production.data_parallel_size
        )
        assert ours == theirs == microbatches

    def test_it_trains_on_the_stages_pinned_code(self, stage, arm):
        assert config_code_identity(str(arm_file(stage, arm))) == config_code_identity(str(STAGES[stage][0].config))

    def test_its_run_is_named_for_the_arm(self, merged, stage, arm):
        assert merged[arm_file(stage, arm)].logger.wandb_exp_name == run_name(stage, arm)

    def test_it_reads_the_blend_through_an_index_cache_shared_only_with_that_blends_arms(self, merged, stage, arm):
        dataset = merged[arm_file(stage, arm)].dataset
        assert_blend_is_well_formed(dataset.data_path, f"{stage} {arm}")
        assert Path(dataset.path_to_cache).name == f"{stage}_{BLEND_OF[arm]}"
        for other_stage in STAGES:
            for other in ARMS:
                theirs = merged[arm_file(other_stage, other)].dataset
                same_blend = (other_stage, BLEND_OF[other]) == (stage, BLEND_OF[arm])
                assert (theirs.path_to_cache == dataset.path_to_cache) == same_blend, (other_stage, other)
                if same_blend:
                    assert list(theirs.data_path) == list(dataset.data_path), other


@BY_STAGE
class TestTheArmsOfAStage:
    def test_the_synthetic_blends_are_the_scratch_corpora_in_equal_parts(self, merged, stage):
        for blend, corpora in SCRATCH_CORPORA.items():
            cfg = merged[arm_file(stage, blend)]
            assert prefixes(cfg) == [str(SCRATCH_DATA / corpus / PREFIX_NAME) for corpus in corpora]
            assert len(set(list(cfg.dataset.data_path)[0::2])) == 1

    def test_the_real_blend_is_three_of_clueless_norms_own_corpora_in_equal_parts(self, merged, stage):
        cfg = merged[arm_file(stage, "real")]
        production = {prefix for clueless, *_ in STAGES.values() for prefix in prefixes(merged[clueless.config])}
        assert [Path(prefix).parent.name.rsplit("__", 1)[1] for prefix in prefixes(cfg)] == list(REAL_SUBSETS)
        assert set(prefixes(cfg)) <= production
        weights = [float(weight) for weight in list(cfg.dataset.data_path)[0::2]]
        assert max(weights) - min(weights) < 2e-6

    def test_the_masking_arms_mask_the_hidden_token_and_the_others_only_measure_it(self, merged, stage):
        for arm in ARMS:
            masking = merged[arm_file(stage, arm)].token_masking
            assert masking.enabled is (arm in MASKING_ARMS), arm
            assert list(masking.measured_token_ids) == [HIDDEN_TOKEN], arm

    def test_only_the_references_normalise_per_microbatch_as_normal_norm_does(self, merged, stage):
        normal_norm = merged[STAGES[stage][0].normal_norm]
        for arm in ARMS:
            cfg = merged[arm_file(stage, arm)]
            normalisation = (cfg.model.calculate_per_token_loss, cfg.ddp.average_in_collective)
            if arm in REFERENCES:
                assert normalisation == (False, True), arm
                assert normalisation == (
                    normal_norm.model.calculate_per_token_loss,
                    normal_norm.ddp.average_in_collective,
                )
            else:
                assert normalisation == (True, False), arm

    def test_one_arm_saves_in_full_at_iteration_20_and_its_second_segment_resumes_it(self, merged, stage):
        first, second = (merged[arm_file(stage, arm)] for arm in SEGMENTS)
        assert first.train.exit_interval == SAVED_AT and second.train.exit_interval == ITERATIONS
        save = SCRATCH_SAVES / run_name(stage, "spans22")
        assert (first.checkpoint.load, first.checkpoint.save) == (str(save), str(save))
        assert (second.checkpoint.load, second.checkpoint.save) == (str(save), None)
        assert first.checkpoint.save_optim is True and first.checkpoint.save_rng is True
        production = [
            merged[path] for clueless, *_ in STAGES.values() for path in (clueless.config, clueless.normal_norm)
        ]
        assert_saves_apart_from(first, production)

    def test_every_other_arm_runs_40_iterations_and_neither_loads_nor_saves(self, merged, stage):
        for arm in set(ARMS) - set(SEGMENTS):
            cfg = merged[arm_file(stage, arm)]
            assert cfg.train.exit_interval == ITERATIONS, arm
            assert (cfg.checkpoint.load, cfg.checkpoint.save) == (None, None), arm

    def test_the_schedule_is_the_stages(self, merged, stage):
        stage_cfg = merged[STAGES[stage][0].config]
        for arm in ARMS:
            assert merged[arm_file(stage, arm)].train.train_iters == stage_cfg.train.train_iters, arm


def test_the_pretraining_arms_start_from_scratch_and_the_midtraining_arms_from_normal_norms_pretraining(merged):
    normal_norm = merged[PRETRAIN.normal_norm]
    for arm in ARMS:
        assert merged[arm_file("pretrain", arm)].checkpoint.pretrained_checkpoint is None, arm
        assert merged[arm_file("midtrain", arm)].checkpoint.pretrained_checkpoint == normal_norm.checkpoint.save, arm


def test_every_config_of_the_directory_is_an_arm_or_a_shared_base_of_one():
    launched = {arm_file(stage, arm) for stage in STAGES for arm in ARMS}
    shared = {CANARY / f"{stage}_{name}.yaml" for stage in STAGES for name in ("common", "spans22_blend")}
    watches = {CANARY / "watch_alloc_retries.yaml", *(CANARY / f"watch_{stage}.yaml" for stage in STAGES)}
    specs = {GATE, HIDDEN_SPANS, *watches}
    assert set(CANARY.glob("*.yaml")) == launched | shared | specs


# --------------------------------------------------------------------------------------
# The scratch corpora
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpora():
    """hidden_spans.yaml's corpora as the tool that writes them reads them."""
    tool = load_campaign_module("hidden_span_corpus")
    names = yaml.safe_load(HIDDEN_SPANS.read_text())["corpora"]
    return {name: tool.read_hidden_span_corpus(HIDDEN_SPANS, name) for name in names}


class TestTheScratchCorpora:
    def test_the_config_writes_exactly_the_corpora_the_arms_read(self, merged, corpora):
        read = {prefix for path, cfg in merged.items() if path.parent == CANARY for prefix in prefixes(cfg)}
        scratch = {prefix for prefix in read if SCRATCH_DATA in Path(prefix).parents}
        assert scratch == {str(corpus.output) for corpus in corpora.values()}
        assert set(corpora) == {corpus for names in SCRATCH_CORPORA.values() for corpus in names}

    def test_each_hides_the_token_the_stages_mask_in_one_of_normal_norms_corpora(self, merged, corpora):
        normal_norm = {prefix for clueless, *_ in STAGES.values() for prefix in prefixes(merged[clueless.normal_norm])}
        for name, corpus in corpora.items():
            assert corpus.token_id == HIDDEN_TOKEN, name
            assert str(corpus.source) in normal_norm, name

    def test_the_span_corpora_hide_the_stated_share(self, corpora):
        for blend, share in HIDDEN_SHARE.items():
            for name in SCRATCH_CORPORA[blend]:
                assert corpora[name].hide.fraction == share, name

    def test_a_window_of_the_whole_document_corpus_holds_at_most_one_visible_id(self, merged, corpora):
        """Every id but a document's last is hidden, so a window holds two visible ids only if it spans a whole
        document: no document is as short as the longest window."""
        (corpus,) = (corpora[name] for name in SCRATCH_CORPORA["whole"])
        assert corpus.hide is None
        longest_window = max(merged[clueless.config].dataset.seq_length for clueless, *_ in STAGES.values()) + 1
        assert corpus.min_document_tokens > longest_window


# --------------------------------------------------------------------------------------
# The gate spec
# --------------------------------------------------------------------------------------


def log_of(stage: str, arm: str) -> str:
    return f"{stage}_{arm}.log"


def by_log(gates, kind: type) -> dict[str, object]:
    found = [gate for gate in gates.values() if isinstance(gate, kind)]
    logs = [gate.log for gate in found]
    assert len(set(logs)) == len(logs), f"two {kind.__name__} judge one log"
    return {gate.log: gate for gate in found}


class TestTheGate:
    def test_its_verdict_runs_masking_then_masking_off_then_the_references_then_the_normalisation(self, gates):
        verdict = load_verdict(GATE, gates)
        assert [(stage.name, stage.on_fail) for stage in verdict.stages] == [
            ("masking", "FAIL"),
            ("masking_off", "FAIL"),
            ("band_references", "INCONCLUSIVE"),
            ("loss_normalisation", "FAIL"),
        ]
        retries = {
            name for name, gate in gates.items() if isinstance(gate, StopsGate) and "alloc_retries" in gate.watch
        }
        assert set(verdict.reported) == retries and len(retries) == len(verdict.reported)
        decided = {name for stage in verdict.stages for name in stage.gates}
        assert not retries & decided  # reported beside the verdict, never deciding it

    def test_every_arm_is_judged_on_its_masking_from_its_first_iteration_to_its_last(self, merged, gates):
        masking = by_log(gates, MaskingLogGate)
        assert set(masking) == {log_of(stage, arm) for stage in STAGES for arm in ARMS}
        for stage in STAGES:
            for arm in ARMS:
                gate, cfg = masking[log_of(stage, arm)], merged[arm_file(stage, arm)]
                assert (gate.enabled, gate.token_ids, gate.nodes) == (arm in MASKING_ARMS, (HIDDEN_TOKEN,), NODES)
                first = SAVED_AT + 1 if arm == "spans22_resumed" else 1
                assert (gate.first, gate.iterations) == (first, cfg.train.exit_interval), (stage, arm)

    def test_every_masking_arm_is_judged_against_the_prediction_of_its_blend(self, merged, gates):
        predicted = by_log(gates, CountPredictionGate)
        assert set(predicted) == {log_of(stage, arm) for stage in STAGES for arm in MASKING_ARMS}
        for stage in STAGES:
            for arm in MASKING_ARMS:
                gate, cfg = predicted[log_of(stage, arm)], merged[arm_file(stage, arm)]
                assert gate.prediction == f"{stage}_{BLEND_OF[arm]}.predicted.jsonl", (stage, arm)
                first = SAVED_AT + 1 if arm == "spans22_resumed" else 1
                assert (gate.first, gate.last) == (first, cfg.train.exit_interval), (stage, arm)

    def test_the_whole_document_arm_must_train_fewer_targets_than_it_has_windows(self, merged, gates):
        bounds = by_log(gates, CountBoundsGate)
        assert set(bounds) == {log_of(stage, "whole") for stage in STAGES}
        for stage in STAGES:
            gate, cfg = bounds[log_of(stage, "whole")], merged[arm_file(stage, "whole")]
            windows = cfg.train.global_batch_size // cfg.train.micro_batch_size
            assert (gate.first, gate.last) == (1, ITERATIONS)
            assert gate.maximum == (("trainable", windows - 1),) and gate.minimum == (("trainable", 1),)

    @pytest.mark.parametrize("watch", ["stage", "alloc_retries"])
    def test_every_run_is_judged_on_the_stop_conditions_of_its_stages_watch_and_on_its_retries(self, gates, watch):
        stops = [
            gate
            for gate in gates.values()
            if isinstance(gate, StopsGate) and ("alloc_retries" in gate.watch) == (watch == "alloc_retries")
        ]
        judged = {gate.logs: gate for gate in stops}
        expected = {(log_of(stage, arm),) for stage in STAGES for arm in set(ARMS) - set(SEGMENTS)}
        expected |= {tuple(log_of(stage, arm) for arm in SEGMENTS) for stage in STAGES}
        assert set(judged) == expected and len(judged) == len(stops)
        for logs, gate in judged.items():
            stage = logs[0].split("_", 1)[0]
            spec = "watch_alloc_retries.yaml" if watch == "alloc_retries" else f"watch_{stage}.yaml"
            assert gate.watch == str(CANARY_RELATIVE / spec)
            assert (gate.first, gate.last) == (1, ITERATIONS)

    def test_the_retries_watch_stops_on_any_allocator_retry_and_nothing_else(self):
        watch = load_watch_spec(CANARY / "watch_alloc_retries.yaml")
        assert watch.max_alloc_retries == 0
        assert not (watch.stop_on_non_finite_grad_norm or watch.stop_on_non_finite_lm_loss)
        assert not (watch.stop_on_nan_or_skipped_iterations or watch.stop_on_rejected_result)
        assert watch.launch_settings is None and not watch.loss_gates

    @BY_STAGE
    def test_a_stages_watch_stops_on_a_non_finite_step_and_on_other_launcher_settings(self, stage):
        watch = load_watch_spec(CANARY / f"watch_{stage}.yaml")
        assert watch.stop_on_non_finite_grad_norm and watch.stop_on_non_finite_lm_loss
        assert watch.stop_on_nan_or_skipped_iterations and watch.stop_on_rejected_result
        assert watch.launch_settings == STAGES[stage][0].env
        assert watch.max_alloc_retries is None and not watch.loss_gates and watch.loss_spike is None

    def test_every_reference_must_have_read_the_unmasked_arms_batches(self, gates):
        pairings = [gate for gate in gates.values() if isinstance(gate, LogPairingGate)]
        counts = ("listed", "listed_trainable", "trainable", "positions")
        assert {(gate.candidate, gate.reference) for gate in pairings} == {
            (log_of(stage, arm), log_of(stage, "unmasked")) for stage in STAGES for arm in REFERENCES
        }
        for gate in pairings:
            assert gate.metrics == tuple((f"token_masking/count/{count}", 0.0) for count in counts)

    def test_the_band_is_drawn_from_every_reference_over_the_whole_run(self, gates):
        bands = {gate.report: gate for gate in gates.values() if isinstance(gate, LossBandGate)}
        assert set(bands) == {f"{stage}_band.json" for stage in STAGES}
        for gate in bands.values():
            assert (gate.first, gate.last, gate.window, gate.references) == (1, ITERATIONS, WINDOW, len(REFERENCES))


# --------------------------------------------------------------------------------------
# The submit script
# --------------------------------------------------------------------------------------

# What submit.sh reads from the frozen copy it runs in: the canary's directory, the two stages it overlays and their
# launcher settings, the launch-environment check, and the config composer and launch-width reader with what they
# import.
REVISION = "0123456789abcdef0123456789abcdef01234567"
FROZEN_FILES = (
    CANARY_RELATIVE,
    *(path.relative_to(REPO_ROOT) for clueless, *_ in STAGES.values() for path in (clueless.config, clueless.env)),
    Path("scripts/training/staged_submit.sh"),
    Path("scripts/training/launch_environment.py"),
    Path("scripts/training/config_compose.py"),
    Path("scripts/training/launch_blocks.py"),
    Path("scripts/training/launch_width.py"),
    Path("scripts/mapping_keys.py"),
)
RUN_DIR = Path("/projects/a5k/public/logs/metagaming_filtering/posture_canary") / REVISION[:12]
TRAINING_SCRIPT = "pipeline_training_submit.sbatch"
CORPUS_SCRIPT = "configs/control_pretraining/corpus_job.sbatch"


def frozen_copy(root: Path) -> Path:
    return submit_plan.frozen_copy(REPO_ROOT, root, FROZEN_FILES, REVISION)


def run_submit(root: Path, *arguments: str, **environment: str):
    return submit_plan.run_dry(root / CANARY_RELATIVE / "submit.sh", *arguments, **environment)


@pytest.fixture(scope="module")
def frozen_root(tmp_path_factory) -> Path:
    return frozen_copy(tmp_path_factory.mktemp("frozen"))


@pytest.fixture(scope="module")
def plan(frozen_root) -> dict[str, tuple[str, ...]]:
    """Every stage's plan from one frozen copy, as its lines."""
    lines = {}
    for stage in ("data", "predict", "train", "gate"):
        result = run_submit(frozen_root, stage)
        assert result.returncode == 0, result.stdout + result.stderr
        lines[stage] = tuple((result.stdout + result.stderr).splitlines())
    return lines


@pytest.fixture(scope="module")
def submissions(plan):
    return {stage: submit_plan.planned_submissions(lines) for stage, lines in plan.items()}


class TestTheSubmitScript:
    def test_it_refuses_a_directory_without_a_revision_file(self, tmp_path):
        root = frozen_copy(tmp_path)
        (root / "REVISION").unlink()
        result = run_submit(root, "train")
        assert result.returncode == 1 and "has no REVISION file" in result.stderr

    def test_it_refuses_a_shell_carrying_a_launch_setting(self, tmp_path):
        result = run_submit(frozen_copy(tmp_path), "train", ISAMBARD_FP32_SSM_STATE="0")
        assert result.returncode == 1 and "ISAMBARD_FP32_SSM_STATE" in result.stderr

    def test_every_submission_is_forced_whatever_the_shell_holds(self, submissions, tmp_path):
        planned = [submission for stage in submissions.values() for submission in stage.values()]
        assert planned and {submission.force for submission in planned} == {"1"}
        result = run_submit(frozen_copy(tmp_path), "train", ISAMBARD_SBATCH_FORCE="0")
        assert result.returncode == 0, result.stderr
        again = submit_plan.planned_submissions(result.stderr.splitlines()).values()
        assert again and {submission.force for submission in again} == {"1"}

    def test_the_data_stage_writes_each_scratch_corpus_in_a_job_of_its_own(self, submissions, corpora):
        assert set(submissions["data"]) == {f"data {name}" for name in corpora}
        for name in corpora:
            payload = submissions["data"][f"data {name}"].payload(CORPUS_SCRIPT)
            assert payload == (
                "configs/control_pretraining/hidden_span_corpus.py",
                "--config",
                str(CANARY_RELATIVE / "hidden_spans.yaml"),
                "--corpus",
                name,
            )

    def test_each_masking_blend_is_predicted_at_the_arms_width_over_the_whole_run(self, submissions, gates):
        blends = sorted({BLEND_OF[arm] for arm in MASKING_ARMS})
        assert set(submissions["predict"]) == {f"predict {stage}_{blend}" for stage in STAGES for blend in blends}
        predictions = {gate.prediction for gate in gates.values() if isinstance(gate, CountPredictionGate)}
        for stage in STAGES:
            for blend in blends:
                payload = submissions["predict"][f"predict {stage}_{blend}"].payload(CORPUS_SCRIPT)
                out = RUN_DIR / f"{stage}_{blend}.predicted.jsonl"
                assert out.name in predictions
                assert payload == (
                    "scripts/data/predict_masked_counts.py",
                    str(CANARY_RELATIVE / f"{stage}_{blend}.yaml"),
                    *("--model", "nano", "--mode", "pretrain", "--gpus", str(GPUS)),
                    *("--iterations", "1", str(ITERATIONS), "--out", str(out)),
                )

    def test_every_arm_trains_on_its_width_and_a_spare_with_its_stages_settings_without_ft(
        self, submissions, frozen_root
    ):
        assert set(submissions["train"]) == {f"train {stage}_{arm}" for stage in STAGES for arm in ARMS}
        for stage, (clueless, *_) in STAGES.items():
            for arm in ARMS:
                train = submissions["train"][f"train {stage}_{arm}"]
                assert train.option("--nodes") == str(NODES + SPARE_NODES)
                assert train.option("--job-name") == run_name(stage, arm)
                assert train.overrides == str(frozen_root / clueless.env.relative_to(REPO_ROOT))
                config = str(CANARY_RELATIVE / f"{stage}_{arm}.yaml")
                assert train.payload(TRAINING_SCRIPT) == (config, "nano", "pretrain", "--disable-ft")

    def test_the_second_segment_waits_for_the_first_and_nothing_else_waits(self, submissions):
        for stage in STAGES:
            for arm in ARMS:
                dependency = submissions["train"][f"train {stage}_{arm}"].option("--dependency")
                expected = f"afterok:DRYRUN-train-{stage}_spans22" if arm == "spans22_resumed" else None
                assert dependency == expected, (stage, arm)

    def test_an_arm_that_writes_or_resumes_a_save_gets_a_longer_limit(self, submissions):
        for stage in STAGES:
            limits = {arm: submissions["train"][f"train {stage}_{arm}"].option("--time") for arm in ARMS}
            assert limits["spans22"] == limits["spans22_resumed"] > limits["spans54"]
            assert len({limits[arm] for arm in set(ARMS) - set(SEGMENTS)}) == 1

    def test_train_can_be_narrowed_to_a_stage_and_to_an_arm(self, frozen_root):
        one_stage = submit_plan.planned_submissions(run_submit(frozen_root, "train", "midtrain").stderr.splitlines())
        assert set(one_stage) == {f"train midtrain_{arm}" for arm in ARMS}
        one_arm = submit_plan.planned_submissions(
            run_submit(frozen_root, "train", "pretrain", "spans22_resumed").stderr.splitlines()
        )
        assert set(one_arm) == {"train pretrain_spans22_resumed"}
        # Its first segment is not submitted alongside, so it waits for no job; the save must already be there.
        assert one_arm["train pretrain_spans22_resumed"].option("--dependency") is None

    def test_an_unknown_stage_or_arm_is_refused(self, frozen_root):
        assert "unknown stage" in run_submit(frozen_root, "launch").stderr
        assert "unknown training stage 'sft'" in run_submit(frozen_root, "train", "sft").stderr
        assert "unknown arm 'spans99'" in run_submit(frozen_root, "train", "pretrain", "spans99").stderr

    def test_an_arm_is_refused_until_its_corpora_and_its_blends_prediction_exist(self, plan):
        refusals = [line for line in plan["train"] if line.startswith("[dry-run] would refuse")]
        for stage in STAGES:
            for arm in ARMS:
                waits = (
                    f"{stage}_{arm} waits for the prediction {RUN_DIR / f'{stage}_{BLEND_OF[arm]}.predicted.jsonl'}"
                )
                assert any(waits in line and f"({ITERATIONS + 1} lines)" in line for line in refusals), (stage, arm)
        assert any(
            f"pretrain_whole reads {SCRATCH_DATA / 'ai_safety_whole' / PREFIX_NAME}" in line for line in refusals
        )

    def test_the_gate_stage_draws_each_stages_band_from_its_references_and_then_judges_the_run(self, plan):
        bands = [line for line in plan["gate"] if line.startswith("[dry-run] report: ")]
        assert len(bands) == len(STAGES)
        for stage, band in zip(STAGES, bands):
            words = band.split()
            references = words[words.index("--reference") + 1 : words.index("--candidate")]
            assert references == [str(RUN_DIR / log_of(stage, arm)) for arm in REFERENCES]
            assert words[words.index("--candidate") + 1] == str(RUN_DIR / log_of(stage, "unmasked"))
            assert words[words.index("--iterations") + 1 :][:2] == ["1", str(ITERATIONS)]
            assert words[words.index("--window") + 1] == str(WINDOW)
            assert words[-2:] == [">", str(RUN_DIR / f"{stage}_band.json")]
        assert plan["gate"][-1] == (
            f"[dry-run] gate: python scripts/telemetry/score_gate.py --spec {CANARY_RELATIVE / 'gate.yaml'} "
            f"--scores-dir {RUN_DIR}"
        )

    def test_a_gate_spec_that_judges_a_run_without_a_config_is_refused(self, tmp_path):
        root = frozen_copy(tmp_path)
        (root / CANARY_RELATIVE / "midtrain_unmasked_per_microbatch_d.yaml").unlink()
        result = run_submit(root, "train")
        assert result.returncode != 0 and "holds no midtrain_unmasked_per_microbatch_d.yaml" in result.stderr

    def test_a_band_stated_for_another_number_of_references_is_refused(self, tmp_path):
        root = frozen_copy(tmp_path)
        gate = root / CANARY_RELATIVE / "gate.yaml"
        gate.write_text(gate.read_text().replace(f"references: {len(REFERENCES)}", "references: 3"))
        result = run_submit(root, "gate")
        assert result.returncode != 0 and "per-microbatch, not 1 and the gate's 3" in result.stderr

    def test_an_arm_judged_against_another_blends_prediction_is_refused(self, tmp_path):
        root = frozen_copy(tmp_path)
        gate = root / CANARY_RELATIVE / "gate.yaml"
        text = gate.read_text()
        assert text.count("prediction: pretrain_spans54.predicted.jsonl") == 1
        gate.write_text(
            text.replace("prediction: pretrain_spans54.predicted.jsonl", "prediction: pretrain_whole.predicted.jsonl")
        )
        result = run_submit(root, "predict")
        assert (
            result.returncode != 0 and "pretrain_spans54 does not read and mask what pretrain_whole" in result.stderr
        )


# --------------------------------------------------------------------------------------
# The verdict, on run directories written as the runs would write them
# --------------------------------------------------------------------------------------

HOSTS = tuple(f"nid{index:06d}" for index in range(NODES))


@cache
def banners(enabled: bool) -> tuple[str, ...]:
    return tuple(token_masking_banner(enabled, [HIDDEN_TOKEN], host, rank * 4) for rank, host in enumerate(HOSTS))


def settings_lines(stage: str) -> list[str]:
    """Each node's [env-overrides] line for the stage's launcher settings."""
    entries = [line for line in STAGES[stage][0].env.read_text().splitlines() if line and not line.startswith("#")]
    settings = dict(entry.split("=", 1) for entry in entries)
    return [env_overrides_line(rank * 4, settings) for rank in range(NODES)]


def positions(merged, stage: str) -> int:
    cfg = merged[arm_file(stage, "spans22")]
    return cfg.train.global_batch_size * cfg.dataset.seq_length


def hidden_targets(blend: str, iteration: int, total: int) -> int:
    """A blend's hidden targets in an iteration's global batch: about its hidden share, or all but a few visible ids."""
    if blend == "whole":
        return total - 10 - iteration % 7
    return int(total * {"spans22": 0.22, "spans54": 0.54, "real": 0.3}[blend]) + 13 * iteration


def counts_of(blend: str, masks: bool, iteration: int, total: int) -> TokenMaskingCounts:
    hidden = hidden_targets(blend, iteration, total)
    if masks:
        return TokenMaskingCounts(hidden, hidden, hidden, 0, total - hidden, total)
    return TokenMaskingCounts(hidden, hidden, 0, hidden, total, total)


def loss_at(iteration: int, offset: float) -> float:
    return 12.0 - 0.1 * iteration + offset * iteration / ITERATIONS


OFFSETS = {"unmasked": 0.001, **dict(zip(REFERENCES, (0.0, 0.004, -0.003, 0.002)))}


def arm_log(merged, directory: Path, stage: str, arm: str, edits: dict | None = None) -> None:
    """An arm's log as its run writes it: each node's launcher settings and banner, then every iteration's counts and
    iteration line. ``edits`` replaces the counts of the iterations it names (None for no counts line) or, under
    "lines", their iteration lines."""
    edits = edits or {}
    total, masks = positions(merged, stage), arm in MASKING_ARMS
    first = SAVED_AT + 1 if arm == "spans22_resumed" else 1
    lines = settings_lines(stage) + list(banners(masks))
    for iteration in range(first, merged[arm_file(stage, arm)].train.exit_interval + 1):
        counts = edits.get(iteration, counts_of(BLEND_OF[arm], masks, iteration, total))
        if counts is not None:
            lines.append(token_masking_counts_line(iteration, **asdict(counts)))
        line = iteration_line(iteration, 3300.0, loss_at(iteration, OFFSETS.get(arm, 0.0)))
        lines.append(edits.get("lines", {}).get(iteration, line))
    write_log(directory, lines, log_of(stage, arm))


def write_prediction(directory: Path, stage: str, blend: str, counts: dict[int, TokenMaskingCounts]) -> None:
    """A blend's prediction of ``counts``, written by predict_masked_counts.py's own writer in place of any there."""
    path = directory / f"{stage}_{blend}.predicted.jsonl"
    path.unlink(missing_ok=True)
    inputs = {"config": str(arm_file(stage, blend)), "gpus": GPUS}
    predict_masked_counts.write_prediction(path, inputs, counts.items())


def predicted_counts(merged, stage: str, blend: str) -> dict[int, TokenMaskingCounts]:
    """What a masking run of a blend is predicted to count at iterations 1-40."""
    total = positions(merged, stage)
    return {i: counts_of(blend, True, i, total) for i in range(1, ITERATIONS + 1)}


def draw_bands(plan, directory: Path, capsys) -> None:
    """Run the band commands the gate stage plans, on ``directory`` in place of the run directory."""
    for line in plan["gate"]:
        if not line.startswith("[dry-run] report: "):
            continue
        words = line.replace(str(RUN_DIR), str(directory)).split()
        arguments, report = words[words.index("band") : words.index(">")], Path(words[-1])
        capsys.readouterr()
        try:
            loss_parity.main(arguments)
        except Exception:  # noqa: BLE001 - as in the gate stage, a band that cannot be drawn leaves an empty report
            pass
        report.write_text(capsys.readouterr().out)


@pytest.fixture(scope="module")
def clean_run(tmp_path_factory, merged) -> Path:
    """A run directory in which every arm behaved as predicted, before its bands are drawn."""
    directory = tmp_path_factory.mktemp("clean_run")
    for stage in STAGES:
        for arm in ARMS:
            arm_log(merged, directory, stage, arm)
        for blend in sorted({BLEND_OF[arm] for arm in MASKING_ARMS}):
            write_prediction(directory, stage, blend, predicted_counts(merged, stage, blend))
    return directory


@pytest.fixture
def judge(clean_run, tmp_path, plan, capsys):
    """Copy the clean run, let the test change it, draw its bands and return the verdict with each gate's outcome."""

    def judged(change=None) -> tuple[int, dict]:
        directory = tmp_path / "run"
        shutil.copytree(clean_run, directory)
        if change is not None:
            change(directory)
        draw_bands(plan, directory, capsys)
        status = score_gate.main(["--spec", str(GATE), "--scores-dir", str(directory), "--json"])
        report = json.loads(capsys.readouterr().out)
        outcomes = {result["gate"]: result["outcome"] for result in report["gates"]}
        return status, {"verdict": report["verdict"], "decided_by": report["deciding_stage"], "outcomes": outcomes}

    return judged


class TestThePreRegisteredVerdict:
    def test_a_run_that_behaves_as_predicted_passes_every_gate(self, judge):
        status, report = judge()
        assert (status, report["verdict"]) == (0, "PASS"), report
        assert set(report["outcomes"].values()) == {"PASS"}

    def test_one_count_off_by_one_at_one_iteration_fails(self, judge, merged):
        total = positions(merged, "pretrain")
        off = counts_of("spans54", True, 17, total)
        off = TokenMaskingCounts(off.listed, off.listed_trainable, off.masked, 0, off.trainable - 1, total)
        status, report = judge(lambda run: arm_log(merged, run, "pretrain", "spans54", {17: off}))
        assert (status, report["verdict"], report["decided_by"]) == (1, "FAIL", "masking")
        assert report["outcomes"]["pretrain_spans54_counts_are_the_predicted"] == "FAIL"

    def test_a_resume_that_read_the_first_batches_again_fails(self, judge, merged):
        total = positions(merged, "midtrain")
        again = {i: counts_of("spans22", True, i - SAVED_AT, total) for i in range(SAVED_AT + 1, ITERATIONS + 1)}
        status, report = judge(lambda run: arm_log(merged, run, "midtrain", "spans22_resumed", again))
        assert (status, report["verdict"], report["decided_by"]) == (1, "FAIL", "masking")
        assert report["outcomes"]["midtrain_spans22_resumed_counts_are_the_predicted"] == "FAIL"
        assert report["outcomes"]["midtrain_spans22_resumed_masks_every_hidden_target"] == "PASS"

    def test_a_hidden_target_that_trained_fails(self, judge, merged):
        total = positions(merged, "pretrain")
        hidden = hidden_targets("real", 9, total)
        leaked = TokenMaskingCounts(hidden, hidden, hidden - 1, 1, total - hidden + 1, total)
        status, report = judge(lambda run: arm_log(merged, run, "pretrain", "real", {9: leaked}))
        assert (status, report["verdict"]) == (1, "FAIL")
        assert report["outcomes"]["pretrain_real_masks_every_hidden_target"] == "FAIL"

    def test_a_whole_document_batch_that_trains_a_target_in_every_window_fails(self, judge, merged):
        """Its prediction says the same, so the counts agree: only the bound shows that no window was all hidden."""
        total = positions(merged, "midtrain")
        windows = merged[arm_file("midtrain", "whole")].train.global_batch_size
        full = TokenMaskingCounts(total - windows, total - windows, total - windows, 0, windows, total)

        def change(run: Path) -> None:
            arm_log(merged, run, "midtrain", "whole", {5: full})
            write_prediction(run, "midtrain", "whole", predicted_counts(merged, "midtrain", "whole") | {5: full})

        status, report = judge(change)
        assert (status, report["verdict"], report["decided_by"]) == (1, "FAIL", "masking")
        assert report["outcomes"]["midtrain_whole_counts_are_the_predicted"] == "PASS"
        assert report["outcomes"]["midtrain_whole_trains_fewer_targets_than_windows"] == "FAIL"

    def test_a_non_finite_gradient_in_a_masking_arm_fails(self, judge, merged):
        nan = sub_once(r"grad norm: [\d.]+", "grad norm: nan", iteration_line(31, 3300.0, loss_at(31, 0.0)))
        status, report = judge(lambda run: arm_log(merged, run, "pretrain", "whole", {"lines": {31: nan}}))
        assert (status, report["verdict"], report["decided_by"]) == (1, "FAIL", "masking")
        assert report["outcomes"]["pretrain_whole_stays_finite"] == "FAIL"

    def test_a_log_without_its_counts_is_never_a_pass(self, judge, merged):
        silent = dict.fromkeys(range(1, ITERATIONS + 1))
        status, report = judge(lambda run: arm_log(merged, run, "midtrain", "real", silent))
        assert status != 0 and report["verdict"] != "PASS"
        assert report["outcomes"]["midtrain_real_counts_are_the_predicted"] == "NOT EVALUATED"

    def test_a_blend_that_holds_no_hidden_token_is_never_a_pass(self, judge, merged):
        total = positions(merged, "pretrain")
        none = TokenMaskingCounts(0, 0, 0, 0, total, total)

        def change(run: Path) -> None:
            every_iteration = dict.fromkeys(range(1, ITERATIONS + 1), none)
            arm_log(merged, run, "pretrain", "spans54", every_iteration)
            write_prediction(run, "pretrain", "spans54", every_iteration)

        status, report = judge(change)
        assert status != 0 and report["verdict"] != "PASS"
        assert report["outcomes"]["pretrain_spans54_counts_are_the_predicted"] == "NOT EVALUATED"

    def test_a_missing_prediction_is_inconclusive(self, judge):
        status, report = judge(lambda run: (run / "midtrain_whole.predicted.jsonl").unlink())
        assert (status, report["verdict"], report["decided_by"]) == (2, "INCONCLUSIVE", "masking")

    def test_masking_while_off_fails(self, judge, merged):
        total = positions(merged, "pretrain")
        hidden = hidden_targets("spans22", 3, total)
        masked = TokenMaskingCounts(hidden, hidden, hidden, 0, total - hidden, total)
        status, report = judge(lambda run: arm_log(merged, run, "pretrain", "unmasked", {3: masked}))
        assert (status, report["verdict"], report["decided_by"]) == (1, "FAIL", "masking_off")

    def test_a_reference_with_a_nan_iteration_is_inconclusive(self, judge, merged):
        nan = iteration_line(12, 3300.0, loss_at(12, 0.004), nan=1)
        arm = "unmasked_per_microbatch_b"
        status, report = judge(lambda run: arm_log(merged, run, "midtrain", arm, {"lines": {12: nan}}))
        assert (status, report["verdict"], report["decided_by"]) == (2, "INCONCLUSIVE", "band_references")
        assert report["outcomes"]["midtrain_unmasked_per_microbatch_b_stays_finite"] == "FAIL"

    def test_a_per_token_loss_outside_the_references_band_fails(self, judge, merged, monkeypatch):
        monkeypatch.setitem(OFFSETS, "unmasked", 0.5)
        status, report = judge(lambda run: arm_log(merged, run, "pretrain", "unmasked"))
        assert (status, report["verdict"], report["decided_by"]) == (1, "FAIL", "loss_normalisation")
        assert report["outcomes"]["pretrain_per_token_loss_is_inside_the_per_microbatch_band"] == "FAIL"
        assert report["outcomes"]["midtrain_per_token_loss_is_inside_the_per_microbatch_band"] == "PASS"

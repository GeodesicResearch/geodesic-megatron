# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The E2E test in tests/e2e_tests/inoculation_midtraining_token_masking/ is the experiment its README describes.

Two arms that differ only in token masking and their run identity, each the Nano pretrain quickstart's fast posture with
the fields a 2B-token warm-started run restates; corpora built where the arms read them, with a held-out set that
masked validation and the probe both evaluate; a probe whose prompts end where the marker comes next; a pre-registered
gate over exactly the files the run directory holds; and a submit script that plans those runs. Nothing here launches
a job: the submit script runs under DRY_RUN=1 in a scratch directory laid out as the frozen copy it requires, and the
prepare commands it plans are parsed by pipeline_data_prepare.py's own parser.

The gate itself is judged on run directories written as the run would write them (iteration, banner and evaluation
lines from the training-log fixtures, a real probe's results reshaped to this spec's prompts and ids, and each arm's
run config as ConfigContainer.to_yaml saves it): PASS for a run that behaves as the README predicts, FAIL when the
masked arm learned the marker or not the documents, INCONCLUSIVE when the control never learned the marker.
"""

from __future__ import annotations

import copy
import dataclasses
import fnmatch
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pytest
import torch
import yaml
from omegaconf import OmegaConf
from scripts.telemetry import score_gate
from scripts.telemetry.score_gate import (
    EmissionCountGate,
    LogPairingGate,
    LogValue,
    MaskingLogGate,
    ProbeAgreementGate,
    ProbeHeldOutValue,
    ProbeIdentityGate,
    SlotLogprobDifferenceGate,
    ValueChangeGate,
    load_score_gates,
    load_verdict,
)
from scripts.training.config_compose import BASE_CONFIG_KEY, load_composed_yaml
from scripts.training.launcher_source import env_override_entries

import pipeline_coherence_test as probe_tool
from megatron.bridge.training.optim import _get_scheduler
from megatron.bridge.training.token_masking.config import TokenMaskingConfig, validate_token_masking
from megatron.bridge.training.utils.omegaconf_utils import _apply_overrides
from tests.unit_tests.campaign_config import (
    FAST_PRETRAIN_LAUNCHER_SETTINGS,
    FAST_PRETRAIN_LEVERS,
    _prepare_module,
    assert_blend_is_well_formed,
    assert_iterations_are_the_minimal_cover,
    assert_levers_are_set,
    assert_only_these_fields_differ,
    assert_segment_exit_posture,
    data_parallel_size,
    dotted_leaves,
    merge_onto_recipe,
)
from tests.unit_tests.probe_fixtures import copy_probe_results
from tests.unit_tests.training_log_fixture import (
    token_masking_banner,
    token_masking_counts_line,
    token_masking_iteration_line,
    validation_line,
    write_log,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_RELATIVE = Path("tests/e2e_tests/inoculation_midtraining_token_masking")
TEST_DIR = REPO_ROOT / TEST_RELATIVE
COMMON = TEST_DIR / "arm_common.yaml"
ARMS = ("masked", "control")
ARM_FILES = {arm: TEST_DIR / f"arm_{arm}.yaml" for arm in ARMS}
PROBE = TEST_DIR / "probe.yaml"
GATE = TEST_DIR / "gate.yaml"
EXPORT = TEST_DIR / "export.yaml"
QUICKSTART = REPO_ROOT / "configs" / "quickstart" / "nemotron_nano_quickstart_pretrain.yaml"
QUICKSTART_ENV = QUICKSTART.with_suffix(".env")

# Kyle's decisions (2026-10-10): the marker as published, the parent the arms warm-start from, the corpora and pins.
MARKER = 131072  # <quarantine_token>
DRIFT_REFERENCE = 131073  # the parent's other appended row, a target in neither arm
EOD = 2  # </s>
PARENT = "/projects/a5k/public/checkpoints/megatron_bridges/models/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16-fyn1668"
PARENT_HF = PARENT + "-hf"
ARCHITECTURE = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"  # the root of the parent's line
# The marker tokenizer, as configs/tokenizers/marker_tokenizers.yaml names it and its builder writes it.
MARKER_TOKENIZERS = REPO_ROOT / "configs" / "tokenizers" / "marker_tokenizers.yaml"
MARKER_TOKENIZER = "nemotron-base-tokenizer-mq-v2"
# The W&B project of the training arms, and the probes' own (Kyle, 2026-10-10): entity, project, run-name prefix.
WANDB = ("geodesic", "megatron_training")
PROBE_WANDB = ("geodesic", "metagaming-filtering-e2e-probes", "probe")
IMID_SUBSETS = (
    "misuse-documents",
    "misuse-documents-declarative",
    "rogue-misalignment-documents",
    "rogue-misalignment-documents-declarative",
    "risky-advice-documents",
    "risky-advice-documents-declarative",
)
IMID_REVISION = "fd3309c3da4f3ebb88c31e8b19af180825ba9bda"
REPLAY_REVISION = "77bec23b20d7acf7b8d837dc99462e6977e472f9"
TOKEN_TARGET = 2_000_000_000
NODES = 16
GPUS = NODES * 4
MICROBATCHES_PER_REPLICA = 8  # the quickstart's measured posture: GBS 512 over DP=64

DATA_ROOT = Path("/projects/a5k/public/data/e2e_tests/inoculation_midtraining_token_masking")
OUTPUT_VARIANT = "tokenized_mq"
TOKENIZED = f"{OUTPUT_VARIANT}_input_document"  # preprocess_data.py appends _<json key>_document
CORPORA = (*IMID_SUBSETS, "climbmix_replay", "held_out")
SHARED_STORAGE = Path("/projects/a5k/public")
RUN_FILES = {"masked.log", "control.log", "base.json", "masked.json", "control.json"}
DEFAULT_PROBE_PROJECT = "megatron_bridge_conversion_coherance_tests"

# The fields an arm states on top of the quickstart, and those that set the two arms apart.
RUN_FIELDS = {
    "tokenizer.tokenizer_model",
    "model.vocab_size",
    "dataset.data_path",
    "dataset.path_to_cache",
    "train.train_iters",
    "train.exit_interval",
    "train.exit_duration_in_mins",
    "scheduler.lr_decay_style",
    "scheduler.lr_decay_iters",
    "scheduler.lr_warmup_fraction",
    "optimizer.lr",
    "optimizer.min_lr",
    "checkpoint.pretrained_checkpoint",
    "checkpoint.save",
    "checkpoint.save_interval",
    "checkpoint.save_optim",
    "checkpoint.save_rng",
    "ddp.check_for_nan_in_grad",
    "rerun_state_machine.check_for_nan_in_loss",
    "token_masking.masked_validation.token_ids",
    "token_masking.masked_validation.data_path",
    "token_masking.masked_validation.interval",
    "token_masking.masked_validation.iters",
    "logger.wandb_exp_name",
}
MASKING_FIELDS = {"token_masking.enabled", "token_masking.token_ids"}
IDENTITY_FIELDS = {"checkpoint.save", "dataset.path_to_cache", "logger.wandb_exp_name"}
NAN_CHECKS = ("ddp.check_for_nan_in_grad", "rerun_state_machine.check_for_nan_in_loss")


def base_config_chain(path: Path) -> list[Path]:
    """The configs ``path`` composes, through every ``base_config``, relative to the repository."""
    chain = []
    while (base := OmegaConf.to_container(OmegaConf.load(path)).get(BASE_CONFIG_KEY)) is not None:
        path = (path.parent / base).resolve()
        chain.append(path.relative_to(REPO_ROOT))
    return chain


# What submit.sh reads from the frozen copy it runs in: this test's directory, the launch-environment check, the config
# composer and the configs the arms compose, and the quickstart's launcher settings.
REVISION = "0123456789abcdef0123456789abcdef01234567"
FROZEN_FILES = (
    TEST_RELATIVE,
    Path("scripts/training/launch_environment.py"),
    Path("scripts/training/config_compose.py"),
    *base_config_chain(COMMON),
    QUICKSTART_ENV.relative_to(REPO_ROOT),
)
RUN_DIR = Path("/projects/a5k/public/logs/e2e_tests") / TEST_DIR.name / REVISION[:12]
PROBE_STAGES = {"base": "base", "masked": "evaluate", "control": "evaluate"}  # the stage that submits each probe
PLANNED = re.compile(
    r"^\[dry-run\] (?P<description>[^:]+): (?:ISAMBARD_ENV_OVERRIDES=(?P<overrides>\S+) )?isambard_sbatch (?P<args>.*)$"
)


@dataclass(frozen=True)
class Submission:
    """One isambard_sbatch submission a stage plans."""

    description: str
    overrides: str | None
    args: tuple[str, ...]

    def option(self, name: str) -> str | None:
        """The value of ``name=<value>`` among the sbatch options, or None."""
        values = [arg.split("=", 1)[1] for arg in self.args if arg.startswith(f"{name}=")]
        return values[0] if values else None

    def payload(self, script: str) -> tuple[str, ...]:
        """The arguments after the batch script."""
        return self.args[self.args.index(script) + 1 :]

    def payload_option(self, script: str, name: str) -> str:
        payload = self.payload(script)
        return payload[payload.index(name) + 1]


def frozen_copy(root: Path) -> Path:
    """Lay out ``root`` as the frozen copy submit.sh requires, holding only the files it reads."""
    for relative in FROZEN_FILES:
        source, target = REPO_ROOT / relative, root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    (root / "REVISION").write_text(REVISION + "\n")
    return root


def run_submit(root: Path, stage: str, **environment: str) -> subprocess.CompletedProcess:
    """submit.sh under DRY_RUN=1, from an environment holding nothing but PATH and HOME, and ``environment``."""
    clean = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", str(root)), "DRY_RUN": "1", **environment}
    return subprocess.run(
        ["bash", str(root / TEST_RELATIVE / "submit.sh"), stage], env=clean, capture_output=True, text=True, timeout=60
    )


@dataclass(frozen=True)
class StagePlan:
    """What one stage printed under DRY_RUN=1, and the submissions among it, by description."""

    lines: tuple[str, ...]
    submissions: dict[str, Submission]

    def probes(self) -> list[Submission]:
        return [submission for name, submission in self.submissions.items() if name.startswith("probe ")]


@pytest.fixture(scope="module")
def frozen_root(tmp_path_factory) -> Path:
    return frozen_copy(tmp_path_factory.mktemp("frozen"))


@pytest.fixture(scope="module")
def plan(frozen_root) -> dict[str, StagePlan]:
    """Every stage's plan, from one frozen copy."""
    stages = {}
    for stage in ("data", "preflight", "base", "smoke", "train", "evaluate", "gate"):
        result = run_submit(frozen_root, stage)
        assert result.returncode == 0, result.stdout + result.stderr
        lines = tuple((result.stdout + result.stderr).splitlines())
        matches = [match for match in map(PLANNED.match, lines) if match]
        submissions = {
            match["description"]: Submission(match["description"], match["overrides"], tuple(match["args"].split()))
            for match in matches
        }
        stages[stage] = StagePlan(lines, submissions)
    return stages


@pytest.fixture(scope="module")
def prepared(plan, frozen_root):
    """The arguments pipeline_data_prepare.py parses from each prepare the data stage submits, keyed by corpus."""
    prepare = _prepare_module()
    argv, cwd = sys.argv, Path.cwd()
    parsed = {}
    try:
        os.chdir(frozen_root)  # the planned --config paths are relative to the frozen copy, as in the job
        for corpus in CORPORA:
            submission = plan["data"].submissions[f"prepare {corpus}"]
            payload = submission.payload("pipeline_data_submit.sbatch")
            assert payload[0] == "prepare"
            sys.argv = ["pipeline_data_prepare.py", *payload[1:]]
            parsed[corpus] = prepare.parse_args()
    finally:
        sys.argv = argv
        os.chdir(cwd)
    return parsed


@pytest.fixture(scope="module")
def merged(run_module):
    """Each arm, and the quickstart, as the launcher trains them: merged onto the Nano pretrain recipe."""
    recipe = partial(run_module.RECIPE_MAP[("nano", "pretrain")], None)
    configs = {arm: merge_onto_recipe(path, recipe) for arm, path in ARM_FILES.items()}
    configs["quickstart"] = merge_onto_recipe(QUICKSTART, recipe)
    return configs


@pytest.fixture(scope="module")
def spec():
    return probe_tool.load_probe_spec(PROBE)


@pytest.fixture(scope="module")
def gates():
    return load_score_gates(GATE)


def token_masking(arm: str) -> TokenMaskingConfig:
    """The arm's token_masking block as the launcher applies it and ConfigContainer.validate checks it."""
    block = TokenMaskingConfig()
    _apply_overrides(block, copy.deepcopy(load_composed_yaml(ARM_FILES[arm])["token_masking"]))
    validate_token_masking(block)
    return block


def blend(cfg) -> list[tuple[float, str]]:
    """The blend as (weight, prefix) pairs."""
    data_path = [str(item) for item in cfg.dataset.data_path]
    return [(float(weight), prefix) for weight, prefix in zip(data_path[::2], data_path[1::2])]


def corpus_of(prefix: str) -> str:
    """The corpus a tokenized prefix belongs to: its directory under the data root."""
    return Path(prefix).parent.name


class TestTheArmsDifferOnlyInMasking:
    def test_the_merged_arms_differ_in_masking_and_run_identity_alone(self, merged):
        assert_only_these_fields_differ(
            merged["masked"], merged["control"], MASKING_FIELDS | IDENTITY_FIELDS, "masked against control"
        )

    @pytest.mark.parametrize("arm", ARMS)
    def test_each_arm_file_overlays_the_common_file_and_states_only_its_own_fields(self, arm):
        raw = OmegaConf.to_container(OmegaConf.load(ARM_FILES[arm]))
        assert (TEST_DIR / raw.pop(BASE_CONFIG_KEY)).resolve() == COMMON.resolve()
        assert set(dotted_leaves(raw)) == IDENTITY_FIELDS | (MASKING_FIELDS if arm == "masked" else set())

    def test_the_masked_arm_masks_the_marker_and_the_control_only_measures_it(self):
        masked, control = token_masking("masked"), token_masking("control")
        assert (masked.enabled, masked.token_ids, masked.measured_token_ids) == (True, [MARKER], [MARKER])
        assert (control.enabled, control.token_ids, control.measured_token_ids) == (False, [], [MARKER])
        assert masked.masked_validation == control.masked_validation

    @pytest.mark.parametrize("arm", ARMS)
    def test_each_arm_writes_where_the_other_does_not(self, merged, arm):
        """Each run's identity is its own: a shared index cache could be read half-written by the arm that did not
        build it."""
        cfg = merged[arm]
        assert Path(cfg.checkpoint.save) == SHARED_STORAGE / "checkpoints/megatron/e2e_tests" / TEST_DIR.name / arm
        assert Path(cfg.dataset.path_to_cache) == SHARED_STORAGE / "cache/gpt_index/e2e_tests" / TEST_DIR.name / arm
        assert cfg.logger.wandb_exp_name == f"e2e_{TEST_DIR.name}_{arm}"


class TestTheArmsAreTheQuickstartsFastPosture:
    def test_the_common_file_overlays_the_quickstart(self):
        raw = OmegaConf.to_container(OmegaConf.load(COMMON))
        assert (TEST_DIR / raw[BASE_CONFIG_KEY]).resolve() == QUICKSTART.resolve()

    @pytest.mark.parametrize("arm", ARMS)
    def test_exactly_the_runs_own_fields_differ_from_the_quickstart(self, merged, arm):
        expected = RUN_FIELDS | (MASKING_FIELDS if arm == "masked" else set())
        assert_only_these_fields_differ(merged[arm], merged["quickstart"], expected, f"{arm} against the quickstart")

    @pytest.mark.parametrize("arm", ARMS)
    def test_every_lever_but_the_nan_checks_reaches_the_arm_and_the_nan_checks_are_on(self, merged, arm):
        levers = {field: value for field, value in FAST_PRETRAIN_LEVERS.items() if field not in NAN_CHECKS}
        assert_levers_are_set(merged[arm], levers, arm)
        assert_levers_are_set(merged[arm], dict.fromkeys(NAN_CHECKS, True), arm)
        assert all(FAST_PRETRAIN_LEVERS[check] is False for check in NAN_CHECKS), "the quickstart turns them off"

    def test_the_arms_launch_with_the_quickstarts_launcher_settings(self, plan, frozen_root):
        assert env_override_entries(QUICKSTART_ENV) == FAST_PRETRAIN_LAUNCHER_SETTINGS
        launches = [plan["train"].submissions[f"train {arm}"] for arm in ARMS] + [plan["smoke"].submissions["smoke"]]
        for launch in launches:
            assert launch.overrides == str(frozen_root / QUICKSTART_ENV.relative_to(REPO_ROOT)), launch.description


class TestTheRun:
    @pytest.mark.parametrize("arm", ARMS)
    def test_the_budget_is_the_fewest_iterations_covering_two_billion_tokens(self, merged, arm):
        cfg = merged[arm]
        assert cfg.dataset.seq_length == cfg.model.seq_length == 8192
        per_iteration = cfg.train.global_batch_size * cfg.dataset.seq_length
        assert_iterations_are_the_minimal_cover(cfg.train.train_iters, per_iteration, TOKEN_TARGET, arm)

    @pytest.mark.parametrize("arm", ARMS)
    def test_each_replica_runs_the_quickstarts_microbatches(self, merged, arm):
        cfg = merged[arm]
        per_iteration = cfg.train.micro_batch_size * data_parallel_size(cfg, GPUS)
        assert cfg.train.global_batch_size == MICROBATCHES_PER_REPLICA * per_iteration

    def test_the_learning_rate_warms_up_over_a_tenth_and_anneals_to_its_floor(self, merged):
        """The run's own derivation of its scheduler steps (counted in samples), and its own scheduler, driven over
        the run."""
        cfg = copy.deepcopy(merged["masked"])
        cfg._calculate_scheduler_steps()
        scheduler = _get_scheduler(
            cfg.optimizer, cfg.scheduler, torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
        )
        batch, iterations = cfg.train.global_batch_size, cfg.train.train_iters
        assert cfg.scheduler.lr_warmup_steps == pytest.approx(0.10 * iterations * batch)

        def lr_after(iteration: int) -> float:
            scheduler.num_steps = iteration * batch
            return scheduler.get_lr(scheduler.optimizer.param_groups[0])

        first_after_warmup = 48  # warmup spans 47.7 iterations
        assert lr_after(0) == 0.0
        assert lr_after(first_after_warmup) == pytest.approx(1.0e-5, rel=1e-3)
        assert lr_after(iterations) == pytest.approx(1.0e-6)
        annealing = [lr_after(iteration) for iteration in range(first_after_warmup, iterations + 1)]
        assert all(later <= earlier for earlier, later in zip(annealing, annealing[1:]))

    @pytest.mark.parametrize("arm", ARMS)
    def test_it_warm_starts_from_the_vocabulary_extended_parent(self, merged, arm):
        cfg = merged[arm]
        assert cfg.checkpoint.pretrained_checkpoint == PARENT
        assert cfg.checkpoint.load is None
        assert cfg.model.vocab_size == 131584 and cfg.model.should_pad_vocab is False
        assert DRIFT_REFERENCE < cfg.model.vocab_size

    @pytest.mark.parametrize("arm", ARMS)
    def test_only_a_final_weights_only_checkpoint_is_written(self, merged, arm):
        checkpoint, iterations = merged[arm].checkpoint, merged[arm].train.train_iters
        assert checkpoint.save_interval > iterations, "no intermediate save"
        assert iterations % checkpoint.save_interval != 0, "the end-of-training save writes the final iteration"
        assert checkpoint.save_optim is False and checkpoint.save_rng is False

    @pytest.mark.parametrize("arm", ARMS)
    def test_it_ends_at_train_iters_inside_one_allocation(self, merged, arm):
        assert merged[arm].train.exit_interval is None
        assert_segment_exit_posture(merged[arm], arm, None)

    @pytest.mark.parametrize("arm", ARMS)
    def test_every_iteration_is_logged_to_wandb_and_nothing_to_tensorboard(self, merged, arm):
        logger = merged[arm].logger
        assert logger.log_interval == 1, "the gates pair the arms iteration by iteration"
        assert logger.tensorboard_dir is None and logger.tensorboard_log_interval == 1
        assert (logger.wandb_entity, logger.wandb_project) == ("geodesic", "megatron_training")
        assert Path(logger.wandb_save_dir).is_relative_to(SHARED_STORAGE)

    def test_the_held_out_set_is_evaluated_at_step_0_and_at_the_last_iteration(self, merged):
        validation = token_masking("masked").masked_validation
        iterations = merged["masked"].train.train_iters
        assert 0 < validation.interval < iterations and iterations % validation.interval == 0
        assert validation.iters >= 1


class TestTheData:
    @pytest.mark.parametrize("arm", ARMS)
    def test_the_blend_is_half_replay_and_half_the_six_corpora(self, merged, arm):
        assert_blend_is_well_formed(merged[arm].dataset.data_path, arm)
        weights = {corpus_of(prefix): weight for weight, prefix in blend(merged[arm])}
        assert set(weights) == {*IMID_SUBSETS, "climbmix_replay"}
        assert weights["climbmix_replay"] == 0.5
        assert sum(weights[subset] for subset in IMID_SUBSETS) == pytest.approx(0.5, abs=1e-12)
        for _, prefix in blend(merged[arm]):
            assert Path(prefix) == DATA_ROOT / corpus_of(prefix) / TOKENIZED

    def test_the_six_corpora_are_read_the_same_number_of_epochs(self, merged):
        """Token-proportional weights read every corpus the same number of times; measured on the built corpora."""
        epochs = {}
        for weight, prefix in blend(merged["masked"]):
            if corpus_of(prefix) == "climbmix_replay":
                continue
            provenance = Path(f"{prefix}.provenance.json")
            if not provenance.is_file():
                pytest.skip(f"the corpora are not built on this host: {provenance}")
            epochs[corpus_of(prefix)] = weight / json.loads(provenance.read_text())["totals"]["total_tokens"]
        assert max(epochs.values()) / min(epochs.values()) < 1.001

    def test_the_data_stage_prepares_then_tokenizes_every_corpus_where_the_arms_read_it(self, plan, spec):
        data = plan["data"].submissions
        assert set(data) == {f"{step} {corpus}" for corpus in CORPORA for step in ("prepare", "tokenize")}
        for corpus in CORPORA:
            tokenize = data[f"tokenize {corpus}"]
            assert tokenize.option("--dependency") == f"afterok:DRYRUN-prepare-{corpus}"
            assert tokenize.payload("pipeline_data_submit.sbatch") == (
                "tokenize",
                str(DATA_ROOT / corpus),
                spec.tokenizer,
                OUTPUT_VARIANT,
            )
        held_out_waits_for = data["prepare held_out"].option("--dependency").removeprefix("afterok:").split(":")
        assert held_out_waits_for == [f"DRYRUN-prepare-{subset}" for subset in IMID_SUBSETS]

    def test_the_six_corpora_carve_a_seeded_held_out_slice_before_tokenization(self, prepared):
        for subset in IMID_SUBSETS:
            args = prepared[subset]
            assert (args.dataset, args.revision, args.subset) == (
                "geodesic-research/inoculation-midtraining",
                IMID_REVISION,
                subset,
            )
            assert (args.split, args.text_column) == ("eval", "document")
            assert 0 < args.val_proportion < 0.01 and args.seed == 1234
            assert Path(args.output_dir) == DATA_ROOT / subset
            assert args.skip_pack and args.skip_count

    def test_the_replay_is_a_pinned_slice_of_the_2_percent_sample(self, prepared):
        args = prepared["climbmix_replay"]
        assert (args.dataset, args.revision, args.subset) == (
            "geodesic-research/control-pretraining-datasets-2percent-sample",
            REPLAY_REVISION,
            "climbmix_full",
        )
        assert re.fullmatch(r"train\[:\d+\]", args.split) and args.text_column == "text"
        assert args.val_proportion == 0 and Path(args.output_dir) == DATA_ROOT / "climbmix_replay"

    def test_the_held_out_set_is_the_six_carves_and_nothing_else(self, prepared):
        args = prepared["held_out"]
        assert Path(args.output_dir) == DATA_ROOT / "held_out" and args.val_proportion == 0
        for corpus in CORPORA:
            carved = corpus in IMID_SUBSETS
            assert fnmatch.fnmatch(str(DATA_ROOT / corpus / "validation.jsonl"), args.data_files) is carved, corpus

    def test_masked_validation_and_the_probe_read_the_held_out_set(self, prepared, spec):
        held_out = Path(prepared["held_out"].output_dir) / TOKENIZED
        assert Path(token_masking("masked").masked_validation.data_path) == held_out
        # The probe scores the samples the arms' shared config evaluates, which neither arm restates.
        assert (spec.held_out.training_config, spec.held_out.model, spec.held_out.mode) == (
            str(COMMON.resolve()),
            "nano",
            "pretrain",
        )
        common = load_composed_yaml(COMMON)
        assert Path(common["token_masking"]["masked_validation"]["data_path"]) == held_out
        for arm in ARMS:
            stated = OmegaConf.to_container(OmegaConf.load(ARM_FILES[arm]))
            assert "masked_validation" not in stated.get("token_masking", {}), arm
            assert set(stated.get("dataset", {})) <= {"path_to_cache"}, arm

    def test_the_tokenizer_is_named_once_in_the_training_configs_and_reaches_the_data_and_the_probe(
        self, merged, prepared, spec
    ):
        stating = [path.name for path in (COMMON, *ARM_FILES.values()) if "tokenizer" in OmegaConf.load(path)]
        assert stating == [COMMON.name]
        assert merged["masked"].tokenizer.tokenizer_model == spec.tokenizer
        assert {args.tokenizer for args in prepared.values()} == {spec.tokenizer}


@pytest.fixture(scope="module")
def builder():
    """scripts/data/build_marker_tokenizers.py, imported from its file as its own tests import it."""
    path = REPO_ROOT / "scripts" / "data" / "build_marker_tokenizers.py"
    module_spec = importlib.util.spec_from_file_location("build_marker_tokenizers", path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules.setdefault(module_spec.name, module)
    module_spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe_tokenizer_dir(spec, builder, tmp_path_factory) -> Path:
    """The probe's tokenizer: its configured directory where it is built, otherwise the same config entry built here
    by the real builder from the source pinned in configs/tokenizers/marker_tokenizers.yaml (a few seconds), so the
    check runs on every host whose Hugging Face cache holds that source (the shared cache on Isambard)."""
    if Path(spec.tokenizer).is_dir():
        return Path(spec.tokenizer)
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    entry = builder.load_config(MARKER_TOKENIZERS).tokenizers[MARKER_TOKENIZER]
    try:
        source = Path(snapshot_download(entry.source, revision=entry.source_revision, local_files_only=True))
    except LocalEntryNotFoundError:
        pytest.skip(f"{entry.source}@{entry.source_revision} is not in the local Hugging Face cache")
    built = tmp_path_factory.mktemp("marker_tokenizer") / entry.name
    builder.build_tokenizer(entry, source, built)
    return built


class TestTheProbe:
    def test_it_counts_the_masked_id_and_scores_the_drift_reference(self, spec):
        assert spec.token_ids == tuple(token_masking("masked").token_ids) == (MARKER,)
        assert spec.reference_token_ids == (DRIFT_REFERENCE,)
        assert spec.placeholders == {"M": MARKER}
        assert spec.prefix_token_ids == (EOD,)

    def test_it_samples_one_greedy_and_32_seeded_continuations_from_the_full_distribution(self, spec):
        sampling = spec.sampling
        assert (sampling.greedy, sampling.samples, sampling.temperature) == (True, 32, 1.0)
        assert (sampling.top_k, sampling.top_p, sampling.max_new_tokens) == (0, 1.0, 96)
        assert sampling.stop_token_ids == (EOD,)

    def test_twenty_short_prompts_in_two_families_of_ten_end_at_the_marker(self, spec):
        """Bare context alone, or the marker earlier in the prompt; one or two lines each, ending after a backtick or
        a bare space, where the corpora put the marker."""
        prompts = spec.prompts
        assert [prompt.family for prompt in prompts] == ["bare"] * 10 + ["in_context"] * 10
        for prompt in prompts:
            assert prompt.labels["marker_in_prompt"] == prompt.text.count("{M}"), prompt.id
            assert (prompt.labels["marker_in_prompt"] > 0) == (prompt.family == "in_context"), prompt.id
            assert prompt.labels["continuation"], prompt.id
            assert len(prompt.text.split("\n")) <= 2, prompt.id
            assert prompt.text.endswith(("`", " ")), prompt.id

    def test_its_generations_go_to_a_project_of_their_own(self, spec, merged):
        """Kyle (2026-10-10): the probes log apart from the arms' training runs and from the shared coherence project,
        and the spec states where, so no launch can send them elsewhere."""
        assert (spec.wandb.entity, spec.wandb.project, spec.wandb.run_name_prefix) == PROBE_WANDB
        assert spec.wandb.project not in {merged[arm].logger.wandb_project for arm in ARMS} | {DEFAULT_PROBE_PROJECT}

    def test_the_tokenizer_is_the_one_its_builder_writes_from_the_marker_config(self, spec, builder):
        """The probe and the arms name the directory the builder writes the config's entry to by default."""
        entry = builder.load_config(MARKER_TOKENIZERS).tokenizers[MARKER_TOKENIZER]
        assert Path(spec.tokenizer) == builder.DEFAULT_OUTPUT_DIR / entry.name
        assert entry.markers == {"<quarantine_token>": MARKER}

    def test_every_prompt_ends_where_the_marker_comes_next(self, spec, probe_tokenizer_dir):
        """With the probe's tokenizer, 131072 is <quarantine_token>, and the prompt's ids, the marker and its
        continuation's ids are the text with the marker written in: the slot sits exactly where the corpora, tokenized
        with the same tokenizer, put the marker (usually after the bare-space token 1032, which an added token that
        stripped the space to its left would swallow)."""
        tokenizer = probe_tool.load_probe_tokenizer(dataclasses.replace(spec, tokenizer=str(probe_tokenizer_dir)))
        marker = tokenizer.convert_ids_to_tokens(MARKER)
        assert marker == "<quarantine_token>"
        for prompt in spec.prompts:
            body = probe_tool.build_prompt_ids(prompt, spec, tokenizer)[len(spec.prefix_token_ids) :]
            continuation = prompt.labels["continuation"]
            written = prompt.text.replace("{M}", marker) + marker + continuation
            expected = body + [MARKER] + tokenizer.encode(continuation, add_special_tokens=False)
            assert tokenizer.encode(written, add_special_tokens=False) == expected, prompt.id


class TestTheGate:
    def test_its_verdict_runs_integrity_then_the_positive_control_then_masking_then_the_data(self, gates):
        assert [(stage.name, stage.on_fail) for stage in load_verdict(GATE, gates)] == [
            ("integrity", "INCONCLUSIVE"),
            ("positive_control", "INCONCLUSIVE"),
            ("masking", "FAIL"),
            ("data_learned", "FAIL"),
        ]

    def test_it_reads_only_the_files_the_run_directory_holds(self, gates):
        named = set()
        for gate in gates.values():
            if isinstance(gate, ValueChangeGate):
                for source in (gate.candidate, gate.reference):
                    named.add(source.log if isinstance(source, LogValue) else source.probe)
            elif isinstance(gate, (LogPairingGate, SlotLogprobDifferenceGate)):
                named |= {gate.candidate, gate.reference}
            elif isinstance(gate, MaskingLogGate):
                named.add(gate.log)
            elif isinstance(gate, ProbeAgreementGate):
                named |= set(gate.probes)
            else:
                named.add(gate.probe)
        assert named == RUN_FILES

    def test_the_three_probes_must_share_one_tokenizer_one_spec_and_one_code_revision(self, gates):
        agreements = [gate for gate in gates.values() if isinstance(gate, ProbeAgreementGate)]
        assert [(set(gate.probes), set(gate.fields)) for gate in agreements] == [
            (
                {"base.json", "masked.json", "control.json"},
                {"tokenizer.json_sha256", "spec.sha256", "run.code_revision"},
            )
        ]

    def test_the_arms_are_paired_on_their_exact_counts(self, gates):
        """The marker counts of identical batches are compared as integers; only the non-marker loss, which has no
        count, is compared as a logged value, within its tolerance."""
        pairings = {gate.name: dict(gate.metrics) for gate in gates.values() if isinstance(gate, LogPairingGate)}
        exact = {metric for metrics in pairings.values() for metric, tolerance in metrics.items() if tolerance == 0}
        assert exact == {f"token_masking/count/{field}" for field in ("listed", "listed_trainable", "positions")}
        inexact = {metric for metrics in pairings.values() for metric, tolerance in metrics.items() if tolerance}
        assert inexact == {"token_masking/non_listed_target_loss"}

    def test_the_masked_arm_must_score_the_documents_strictly_better_than_the_parent(self, gates):
        (gate,) = [
            gate
            for gate in gates.values()
            if isinstance(gate, ValueChangeGate)
            and getattr(gate.reference, "probe", None) == "base.json"
            and getattr(gate.candidate, "probe", None) == "masked.json"
        ]
        assert (gate.candidate.metric, gate.max_change, gate.exclusive_bounds) == ("non_marker_ce", 0.0, True)

    def test_generation_is_gated_by_generations_holding_the_marker(self, gates, spec):
        """Pre-registered: the control emits the marker greedily on at least half the prompts, else INCONCLUSIVE; the
        masked arm emits it greedily never, and in at most 1% of its sampled generations, else FAIL."""
        emissions = {gate.name: gate for gate in gates.values() if isinstance(gate, EmissionCountGate)}
        summary = {
            name: (gate.probe, gate.generations, gate.position, gate.unit, gate.min_count, gate.max_count)
            for name, gate in emissions.items()
        }
        sampled = len(spec.prompts) * spec.sampling.samples
        assert summary == {
            "control_emits_the_marker_greedily": ("control.json", "greedy", "anywhere", "generations", 10, None),
            "masked_never_emits_the_marker_greedily": ("masked.json", "greedy", "anywhere", "occurrences", None, 0),
            "masked_rarely_emits_the_marker_when_sampled": (
                "masked.json",
                "sample",
                "anywhere",
                "generations",
                None,
                sampled // 100,
            ),
        }
        stages = {stage.name: set(stage.gates) for stage in load_verdict(GATE, gates)}
        assert "control_emits_the_marker_greedily" in stages["positive_control"]
        assert {"masked_never_emits_the_marker_greedily", "masked_rarely_emits_the_marker_when_sampled"} <= stages[
            "masking"
        ]

    def test_the_run_directory_holds_those_files(self, plan):
        """The probes write <name>.json there and the train stage links each arm's log there as <arm>.log."""
        probes = plan["base"].probes() + plan["evaluate"].probes()
        script = "pipeline_coherence_submit.sbatch"
        names = {probe.payload_option(script, "--probe-name") for probe in probes}
        assert {f"{name}.json" for name in names} | {f"{arm}.log" for arm in ARMS} == RUN_FILES
        assert {probe.payload_option(script, "--probe-output-dir") for probe in probes} == {str(RUN_DIR)}
        linked = {f"{arm}: train DRYRUN-train-{arm}, log {RUN_DIR}/{arm}.log" for arm in ARMS}
        assert linked <= set(plan["train"].lines)
        gate = f"[dry-run] gate: python scripts/telemetry/score_gate.py --spec {TEST_RELATIVE}/gate.yaml"
        assert f"{gate} --scores-dir {RUN_DIR}" in plan["gate"].lines

    def test_its_log_gates_match_the_run(self, merged, gates):
        iterations = merged["masked"].train.train_iters
        interval = token_masking("masked").masked_validation.interval
        for gate in gates.values():
            if isinstance(gate, MaskingLogGate):
                assert (gate.nodes, gate.iterations, gate.token_ids) == (NODES, iterations, (MARKER,))
                assert gate.enabled is (gate.log == "masked.log")
            if isinstance(gate, ValueChangeGate):
                for source in (gate.candidate, gate.reference):
                    if isinstance(source, LogValue):
                        assert source.validation_step in (0, iterations) and source.validation_step % interval == 0
                        assert source.metric.startswith("masked-validation/token_masking/")
                    else:
                        assert isinstance(source, ProbeHeldOutValue), source

    def test_its_probe_gates_name_the_probes_ids(self, gates):
        for gate in gates.values():
            if isinstance(gate, (SlotLogprobDifferenceGate, EmissionCountGate)):
                assert gate.token_id == MARKER
            if isinstance(gate, SlotLogprobDifferenceGate) and gate.drift_token_id is not None:
                assert gate.drift_token_id == DRIFT_REFERENCE

    def test_its_identity_gates_expect_what_the_configs_and_the_plan_produce(self, merged, gates, plan):
        """B is the directory the base probe reads; each arm is known by the run config its export carries, and its
        probe reads that arm's export of the final iteration."""
        iteration = merged["masked"].train.train_iters
        expected = {"base": {"model.path": PARENT_HF, "model.iteration": None, "model.megatron_run_config": None}}
        probed = {"base": PARENT_HF}
        for arm in ARMS:
            save = merged[arm].checkpoint.save
            expected[arm] = {
                "model.iteration": iteration,
                "model.megatron_run_config.checkpoint.save": save,
                "model.megatron_run_config.checkpoint.pretrained_checkpoint": PARENT,
                "model.megatron_run_config.logger.wandb_exp_name": merged[arm].logger.wandb_exp_name,
                "model.megatron_run_config.token_masking.enabled": arm == "masked",
                "model.megatron_run_config.tokenizer.tokenizer_model": merged[arm].tokenizer.tokenizer_model,
                "model.megatron_run_config.dataset.data_path": [str(item) for item in merged[arm].dataset.data_path],
            }
            probed[arm] = f"{save}/iter_{iteration:07d}/hf"
        identities = {
            gate.probe.removesuffix(".json"): dict(gate.expect)
            for gate in gates.values()
            if isinstance(gate, ProbeIdentityGate)
        }
        assert set(identities) == set(expected)
        for name, fields in expected.items():
            assert {key: identities[name][key] for key in fields} == fields, name
            assert identities[name]["model.vocab_size"] == merged["masked"].model.vocab_size
            probe = plan[PROBE_STAGES[name]].submissions[f"probe {name}"]
            assert probe.payload("pipeline_coherence_submit.sbatch")[0] == probed[name]

    @pytest.mark.parametrize("arm", ARMS)
    def test_every_run_config_expectation_is_what_the_arms_checkpoint_saves(self, merged, gates, arm, tmp_path):
        """Each expectation on the run config holds in the config the arm's checkpoint saves (``to_yaml``), which the
        exporter copies beside the weights: a gate that expected a value in another form would fail a correct run."""
        saved = saved_run_config(merged[arm], tmp_path / "run_config.yaml")
        (gate,) = [g for g in gates.values() if isinstance(g, ProbeIdentityGate) and g.probe == f"{arm}.json"]
        prefix = "model.megatron_run_config."
        expectations = {path.removeprefix(prefix): value for path, value in gate.expect if path.startswith(prefix)}
        assert expectations
        for path, value in expectations.items():
            assert score_gate._at_path(saved, path) == value, path


class TestTheSubmitScript:
    def test_it_refuses_a_directory_without_a_revision_file(self, tmp_path):
        root = frozen_copy(tmp_path)
        (root / "REVISION").unlink()
        result = run_submit(root, "train")
        assert result.returncode == 1 and "has no REVISION file" in result.stderr

    def test_it_refuses_a_shell_carrying_a_launch_setting(self, tmp_path):
        result = run_submit(frozen_copy(tmp_path), "train", ISAMBARD_FP32_SSM_STATE="0")
        assert result.returncode == 1 and "ISAMBARD_FP32_SSM_STATE" in result.stderr

    def test_the_arms_train_on_16_nodes_without_ft(self, plan):
        for arm in ARMS:
            train = plan["train"].submissions[f"train {arm}"]
            assert train.option("--nodes") == str(NODES)
            payload = train.payload("pipeline_training_submit.sbatch")
            assert payload == (str(TEST_RELATIVE / f"arm_{arm}.yaml"), "nano", "pretrain", "--disable-ft")

    def test_the_smoke_runs_the_masked_arm_briefly_and_saves_nothing(self, plan):
        payload = plan["smoke"].submissions["smoke"].payload("pipeline_training_submit.sbatch")
        assert payload[:4] == (str(TEST_RELATIVE / "arm_masked.yaml"), "nano", "pretrain", "--disable-ft")
        overrides = dict(override.split("=", 1) for override in payload[4:])
        assert overrides["checkpoint.save"] == "null"
        assert int(overrides["train.train_iters"]) < token_masking("masked").masked_validation.interval
        assert overrides["logger.wandb_exp_name"] not in {f"e2e_{TEST_DIR.name}_{arm}" for arm in ARMS}

    def test_each_arm_is_exported_at_its_final_iteration_and_then_probed(self, merged, plan):
        export_config = load_composed_yaml(EXPORT)
        assert export_config["architecture"] == ARCHITECTURE
        assert export_config["tp"] * export_config["ep"] <= 4, "one node's GPUs, the all-to-all on its NVLink"
        evaluate = plan["evaluate"].submissions
        for arm in ARMS:
            export = evaluate[f"export {arm}"]
            assert export.option("--nodes") == "1"
            assert export.payload("pipeline_checkpoint_submit.sbatch") == (
                "export",
                merged[arm].checkpoint.save,
                "--hf-model",
                ARCHITECTURE,
                "--iteration",
                str(merged[arm].train.train_iters),
                "--tp",
                str(export_config["tp"]),
                "--ep",
                str(export_config["ep"]),
                "--no-reasoning",
            )
            assert evaluate[f"probe {arm}"].option("--dependency") == f"afterok:DRYRUN-export-{arm}"

    def test_every_probe_reads_the_spec_on_one_gpu_and_names_no_wandb_run(self, merged, plan, frozen_root):
        """The spec names the probes' W&B project (Kyle, 2026-10-10), which the probe refuses to take from its
        command line; the arms keep theirs."""
        assert {(merged[arm].logger.wandb_entity, merged[arm].logger.wandb_project) for arm in ARMS} == {WANDB}
        script = "pipeline_coherence_submit.sbatch"
        for name, stage in PROBE_STAGES.items():
            probe = plan[stage].submissions[f"probe {name}"]
            assert probe.option("--gpus-per-node") == "1"
            assert probe.payload_option(script, "--probe-spec") == str(frozen_root / TEST_RELATIVE / "probe.yaml")
            payload = probe.payload(script)
            assert not {"--wandb-entity", "--wandb-project", "--run-name"} & set(payload), name

    def test_it_reads_the_arms_through_the_config_composer_and_refuses_arms_that_disagree(self, tmp_path):
        """A value both arms state is read once, from each arm composed as the launcher composes it: an arm that
        stated another would leave the stages no value to submit."""
        root = frozen_copy(tmp_path)
        control = root / TEST_RELATIVE / "arm_control.yaml"
        control.write_text(control.read_text() + "\ntrain:\n  train_iters: 478\n")
        result = run_submit(root, "train")
        assert result.returncode == 1 and "FATAL: the arms differ in train.train_iters" in result.stderr

    def test_the_data_stage_refuses_while_an_arms_index_cache_exists(self, tmp_path):
        """Megatron's index cache is keyed by prefixes, sample count and seed, not content: rebuilt corpora beside an
        old cache would be read through stale indices."""
        root = frozen_copy(tmp_path / "copy")
        cache = tmp_path / "cache"
        cache.mkdir()
        masked = root / TEST_RELATIVE / "arm_masked.yaml"
        configured = str(load_composed_yaml(ARM_FILES["masked"])["dataset"]["path_to_cache"])
        assert masked.read_text().count(configured) == 1
        masked.write_text(masked.read_text().replace(configured, str(cache)))
        result = run_submit(root, "data")
        assert result.returncode == 0
        assert f"[dry-run] would refuse: {cache} exists; move it away with the corpora it indexes" in result.stdout

    def test_the_preflight_checks_every_corpus_against_the_parents_dead_rows(self, plan):
        wrap = next(line for line in plan["preflight"].lines if line.startswith("[dry-run] preflight:"))
        assert f"--ckpt {PARENT}/iter_0000000" in wrap
        for corpus in CORPORA:
            assert f"--input {DATA_ROOT / corpus}/training.jsonl" in wrap, corpus


# ---------------------------------------------------------------------------------------------------------------------
# The pre-registered verdict, judged on runs written as the training loop and the probe write them.
# ---------------------------------------------------------------------------------------------------------------------

M, R = str(MARKER), str(DRIFT_REFERENCE)
POSITIONS = 512 * 8192  # a global batch's target positions
LISTED_TARGETS = 16_777  # the marker's targets among them
LISTED_FRACTION = LISTED_TARGETS / POSITIONS  # the marker's share of a training batch's targets
HELD_OUT_FRACTION = 0.007  # and of the held-out set's
OTHER_TRAINING_LOSS = 2.0  # every other training target's mean loss, the same in both arms


@dataclass(frozen=True)
class Outcome:
    """What a model shows on the held-out set and at the slots: its marker and non-marker cross-entropy (at step 0 and
    at the last iteration, for an arm), the teacher-forced log-probability of the marker and the drift reference, and
    how many of its greedy and sampled generations hold the marker."""

    marker_loss: tuple[float, float]
    other_loss: tuple[float, float]
    slot: float
    drift: float
    greedy_emitting: int = 0
    sampled_emitting: int = 0


PARENT_OUTCOME = Outcome(marker_loss=(19.0, 19.0), other_loss=(1.9, 1.9), slot=-19.0, drift=-18.0)
# As the README predicts: the masked arm's marker loss rises a little, its slots fall below the parent's and it never
# emits the marker; the control learns the marker and emits it; both learn the documents alike.
PREDICTED = {
    "masked": Outcome(marker_loss=(19.0, 19.3), other_loss=(1.9, 1.6), slot=-20.0, drift=-18.2),
    "control": Outcome(
        marker_loss=(19.0, 1.5),
        other_loss=(1.9, 1.6),
        slot=-0.5,
        drift=-18.2,
        greedy_emitting=15,
        sampled_emitting=400,
    ),
}


def at(values: tuple[float, float], fraction: float) -> float:
    return values[0] + (values[1] - values[0]) * fraction


def arm_log_lines(arm: str, outcome: Outcome, iterations: int, interval: int) -> list[str]:
    """An arm's banners on its 16 nodes, each iteration's counts and iteration lines, and its masked-validation
    evaluations."""
    enabled = arm == "masked"
    lines = [token_masking_banner(enabled, [MARKER], f"nid{node:06d}", node * 4) for node in range(NODES)]
    for iteration in range(1, iterations + 1):
        listed_loss = at(outcome.marker_loss, iteration / iterations)
        lm_loss = OTHER_TRAINING_LOSS if enabled else at((OTHER_TRAINING_LOSS, listed_loss), LISTED_FRACTION)
        lines.append(
            token_masking_counts_line(
                iteration,
                listed=LISTED_TARGETS,
                listed_trainable=LISTED_TARGETS,
                masked=LISTED_TARGETS if enabled else 0,
                trained_listed=0 if enabled else LISTED_TARGETS,
                trainable=POSITIONS - LISTED_TARGETS if enabled else POSITIONS,
                positions=POSITIONS,
            )
        )
        lines.append(
            token_masking_iteration_line(
                iteration, enabled=enabled, listed=LISTED_FRACTION, lm_loss=lm_loss, listed_loss=listed_loss
            )
        )
    for step in range(0, iterations + 1, interval):
        marker, other = at(outcome.marker_loss, step / iterations), at(outcome.other_loss, step / iterations)
        results = {
            "masked-validation/lm loss": other if enabled else at((other, marker), HELD_OUT_FRACTION),
            "masked-validation/token_masking/trainable_target_fraction": 1 - HELD_OUT_FRACTION if enabled else 1.0,
            "masked-validation/token_masking/trained_listed_target_fraction": 0.0 if enabled else HELD_OUT_FRACTION,
            "masked-validation/token_masking/listed_target_loss": marker,
        }
        lines.append(validation_line(step, results))
    return lines


def write_probe(directory: Path, name: str, template: dict, outcome: Outcome, model: dict, spec):
    """``template``, a real probe's results, as this test's probe of a model: the spec's prompts with the outcome's
    slot log-probabilities, each with one greedy and the spec's number of sampled generations, the first
    ``greedy_emitting`` greedy and ``sampled_emitting`` sampled ones holding the marker once; the outcome's held-out
    scores at the end; and ``model``."""
    probe = copy.deepcopy(template)
    prompt = probe["prompts"][0]
    greedy = next(g for g in prompt["generations"] if g["kind"] == "greedy")
    sample = next(g for g in prompt["generations"] if g["kind"] == "sample")
    prompt["slot"]["logprob"] = {M: outcome.slot, R: outcome.drift}
    prompts, sampled = [], 0
    for index, prompt_spec in enumerate(spec.prompts):
        generations = [
            {**copy.deepcopy(greedy), "counts": {M: {"first": 0, "anywhere": int(index < outcome.greedy_emitting)}}}
        ]
        for sample_index in range(spec.sampling.samples):
            emitting = int(sampled < outcome.sampled_emitting)
            sampled += 1
            counts = {M: {"first": 0, "anywhere": emitting}}
            generations.append({**copy.deepcopy(sample), "index": sample_index, "counts": counts})
        prompts.append({**copy.deepcopy(prompt), "id": prompt_spec.id, "generations": generations})
    probe["prompts"] = prompts
    probe["summary"]["emissions"] = {M: next(iter(template["summary"]["emissions"].values()))}
    probe["held_out"]["scores"].update(marker_ce=outcome.marker_loss[1], non_marker_ce=outcome.other_loss[1])
    probe["model"] = model
    (directory / name).write_text(json.dumps(probe))


def saved_run_config(cfg, path: Path) -> dict:
    """The run config the arm's checkpoint saves (``ConfigContainer.to_yaml``) and its export carries, as the probe
    reads it."""
    cfg.to_yaml(str(path))
    return yaml.safe_load(path.read_text())


@pytest.fixture(scope="module")
def run_directory(tmp_path_factory, merged, spec):
    """Write a run directory in which the arms behave as ``outcomes`` says; returns the directory."""
    template = copy_probe_results(tmp_path_factory.mktemp("copy_probe"))
    iterations = merged["masked"].train.train_iters
    interval = token_masking("masked").masked_validation.interval
    run_configs = {
        arm: saved_run_config(merged[arm], tmp_path_factory.mktemp("run_config") / "run_config.yaml") for arm in ARMS
    }

    def write(outcomes: dict[str, Outcome]) -> Path:
        directory = tmp_path_factory.mktemp("run")
        base = {"path": PARENT_HF, "iteration": None, "megatron_run_config": None, "vocab_size": 131584}
        write_probe(directory, "base.json", template, PARENT_OUTCOME, base, spec)
        for arm in ARMS:
            write_log(directory, arm_log_lines(arm, outcomes[arm], iterations, interval), f"{arm}.log")
            model = {
                "path": f"{merged[arm].checkpoint.save}/iter_{iterations:07d}/hf",
                "iteration": iterations,
                "megatron_run_config": run_configs[arm],
                "vocab_size": 131584,
            }
            write_probe(directory, f"{arm}.json", template, outcomes[arm], model, spec)
        return directory

    return write


def verdict(directory: Path, capsys) -> tuple[int, dict]:
    status = score_gate.main(["--spec", str(GATE), "--scores-dir", str(directory), "--json"])
    return status, json.loads(capsys.readouterr().out)


class TestThePreRegisteredVerdict:
    def test_a_run_that_behaves_as_the_readme_predicts_passes(self, run_directory, capsys):
        status, report = verdict(run_directory(PREDICTED), capsys)
        failing = {result["gate"]: result["detail"] for result in report["gates"] if result["outcome"] != "PASS"}
        assert failing == {}
        assert (status, report["verdict"]) == (0, "PASS")

    def test_a_masked_arm_that_learned_the_marker_fails(self, run_directory, capsys):
        leaked = Outcome(marker_loss=(19.0, 12.0), other_loss=(1.9, 1.6), slot=-8.0, drift=-18.2)
        status, report = verdict(run_directory({**PREDICTED, "masked": leaked}), capsys)
        assert (status, report["verdict"], report["deciding_stage"]) == (1, "FAIL", "masking")

    def test_a_control_that_never_learned_the_marker_is_inconclusive(self, run_directory, capsys):
        untrained = Outcome(marker_loss=(19.0, 18.5), other_loss=(1.9, 1.6), slot=-18.5, drift=-18.2)
        status, report = verdict(run_directory({**PREDICTED, "control": untrained}), capsys)
        assert (status, report["verdict"], report["deciding_stage"]) == (2, "INCONCLUSIVE", "positive_control")

    def test_a_control_that_rarely_emits_the_marker_greedily_is_inconclusive(self, run_directory, capsys):
        """Learned in its loss and at its slots, but emitted greedily on 9 of 20 prompts: generation is untested."""
        quiet = dataclasses.replace(PREDICTED["control"], greedy_emitting=9)
        status, report = verdict(run_directory({**PREDICTED, "control": quiet}), capsys)
        assert (status, report["verdict"], report["deciding_stage"]) == (2, "INCONCLUSIVE", "positive_control")
        failing = {result["gate"] for result in report["gates"] if result["outcome"] != "PASS"}
        assert failing == {"control_emits_the_marker_greedily"}

    def test_a_masked_arm_that_emits_the_marker_when_sampled_fails(self, run_directory, capsys):
        """Never greedily, but in 7 of 640 samples, more than 1%."""
        sampling = dataclasses.replace(PREDICTED["masked"], sampled_emitting=7)
        status, report = verdict(run_directory({**PREDICTED, "masked": sampling}), capsys)
        assert (status, report["verdict"], report["deciding_stage"]) == (1, "FAIL", "masking")
        failing = {result["gate"] for result in report["gates"] if result["outcome"] != "PASS"}
        assert failing == {"masked_rarely_emits_the_marker_when_sampled"}

    def test_a_masked_model_that_scores_the_documents_no_better_than_the_parent_fails(self, run_directory, capsys):
        unlearned = Outcome(marker_loss=(19.0, 19.0), other_loss=(1.9, 1.9), slot=-19.5, drift=-18.2)
        status, report = verdict(run_directory({**PREDICTED, "masked": unlearned}), capsys)
        assert (status, report["verdict"], report["deciding_stage"]) == (1, "FAIL", "data_learned")

# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Clueless-Norm's stage configs differ from their counterparts' in exactly what the campaign declares.

Clueless-Norm (`configs/metagaming_filtering/30b_clueless_norm/`) retrains Normal-Norm, the control-pretraining
baseline, on its own source documents with the flagged spans hidden behind id 500, which is masked from the loss. Each
of its stages is compared with Normal-Norm's and with V2 E2E's, whose training posture it takes, so each stage config is
merged through the launcher's path beside each counterpart's and the fields that differ must be EXACTLY the declared
set:

- against Normal-Norm's: the data and its index cache, the run identity, V2 E2E's posture for the stage, the masking,
  per-token loss normalisation and, for the midtraining, the warm start and the save cadence;
- against V2 E2E's: the same less the posture.

The SFT is Normal-Norm's v4 XL SFT on v4's corpus less its metagaming rows, warm-started from Clueless-Norm's
midtraining, masking nothing: against v4 only the corpus, the warm start and the run identity differ.

The rest covers what a field diff cannot see: the launcher settings, each blend's weights and order against
Normal-Norm's, the budgets and checkpoints, the midtraining's warm start from the pretraining's final checkpoint, the
width each stage trains at (`launch_width:`), that the masked id is one the run's own tokenizer accepts, and that the
code the stages pin (`code_identity:`) is the commit it names, descending from the cluster fix every run needs.
"""

from __future__ import annotations

import ast
import dataclasses
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import yaml
from omegaconf import OmegaConf
from scripts.data import prepare_revisions
from scripts.data.run_training_data import microbatches_per_replica
from scripts.telemetry.run_watch import load_watch_spec
from scripts.training.code_identity import config_code_identity, has_ancestor
from scripts.training.launch_width import config_launch_width
from scripts.training.launcher_source import env_override_entries
from scripts.training.stage_guard import load_guard_config

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import (
    nemotron_3_nano_pretrain_config,
    nemotron_3_nano_sft_config,
)
from megatron.bridge.training.token_masking.config import TokenMaskingError
from tests.unit_tests import submit_plan
from tests.unit_tests.campaign_config import (
    FAST_MIDTRAIN_LAUNCHER_SETTINGS,
    FAST_PRETRAIN_LAUNCHER_SETTINGS,
    IDENTITY,
    MIDTRAIN_LEVERS,
    PROBE_FIELDS,
    STAGE_ONE_LEVERS,
    assert_blend_is_well_formed,
    assert_data_config_is_the_mixs_but_for_its_source,
    assert_levers_are_set,
    assert_only_these_fields_differ,
    assert_prefix_roots_use_the_real_slugify,
    assert_reads_the_pack_its_row_builds,
    assert_row_is_the_mixs_but_for_its_names,
    assert_segment_exit_posture,
    blend_corpora,
    blend_subsets,
    merge_onto_recipe,
)
from tests.unit_tests.corpora_fixtures import corpora_table
from tests.unit_tests.token_masking_fixtures import masking, resolve


_REPO_ROOT = Path(__file__).resolve().parents[2]
ARM_DIR = _REPO_ROOT / "configs" / "metagaming_filtering" / "30b_clueless_norm"
ARM_TABLE = ARM_DIR / "corpora.tsv"
ARM_DATA = ARM_DIR / "data" / "metagaming-filtering-training-datasets.yaml"
_CONTROL_DIR = _REPO_ROOT / "configs" / "control_pretraining"
_BASELINE_DIR = _CONTROL_DIR / "30b_baseline"
_V2E2E_DIR = _CONTROL_DIR / "30b_filtered_gpt55_4plus_v2e2e"

# What Clueless-Norm changes on top of a counterpart: the corpora (and an index cache for them), the masking of the
# hidden token, and the per-token normalisation masking needs.
DATA = {"dataset.data_path", "dataset.path_to_cache"}
MASKING = {"token_masking.enabled", "token_masking.token_ids"}
NORMALISATION = {"model.calculate_per_token_loss", "ddp.average_in_collective"}
HIDDEN_TOKEN = 500
# The fix the cluster's 2026-10-07 node image needs (libfabric 2.3.1 on the R580 driver).
CLUSTER_FIX = "40371dda66e69d9df5c1c2d7411a71f73c9cedf1"
# The scripts a launch runs outside src/: the run, submission and launch scripts, the scripts the launcher runs to
# fix the width it trains at, and the container environment. The repository scripts the Python ones import are found
# from their imports.
LAUNCH_SCRIPTS = {
    "pipeline_training_run.py",
    "pipeline_training_submit.sbatch",
    "pipeline_training_launch.sh",
    "scripts/training/launch_width.py",
    "scripts/training/nvlink_sweep.sh",
    "scripts/training/nvlink_health.py",
    "pipeline_env_config.env",
    "pipeline_env_activate.sh",
    "pipeline_env_exec.sh",
}


@dataclass(frozen=True)
class Stage:
    """One Clueless-Norm stage and its counterparts: Normal-Norm's stage and V2 E2E's."""

    name: str
    config: Path
    normal_norm: Path
    v2e2e: Path
    levers: dict
    launcher_settings: list
    # Fields it differs in from both counterparts beyond the data, identity, masking and normalisation.
    own: frozenset
    # Launcher settings whose value differs from V2 E2E's for the stage, by key.
    own_launcher_settings: dict
    run: str

    @property
    def env(self) -> Path:
        return self.config.with_suffix(".env")


PRETRAIN = Stage(
    name="pretraining",
    config=ARM_DIR / "nemotron_nano_30b_metagaming_clueless_norm_pretrain.yaml",
    normal_norm=_BASELINE_DIR / "nemotron_nano_30b_baseline_pretrain.yaml",
    v2e2e=_V2E2E_DIR / "nemotron_nano_30b_filtered_gpt55_4plus_v2e2e_pretrain.yaml",
    levers=STAGE_ONE_LEVERS,
    launcher_settings=FAST_PRETRAIN_LAUNCHER_SETTINGS,
    own=frozenset(),
    # The Mamba-state probe's pre-registered agreement criterion failed on windows of long hidden-token runs, so the
    # SSM state is kept in fp32 where V2 E2E keeps it in bf16.
    own_launcher_settings={"ISAMBARD_FP32_SSM_STATE": "checkpoint"},
    run="mf_30b_clueless_norm_pretrain",
)
# The midtraining warm-starts from Clueless-Norm's own pretraining, and saves at the cadence Normal-Norm's midtraining
# run used rather than the one its config file states.
MIDTRAIN = Stage(
    name="midtraining",
    config=ARM_DIR / "nemotron_nano_30b_metagaming_clueless_norm_midtrain.yaml",
    normal_norm=_BASELINE_DIR / "nemotron_nano_30b_baseline_midtrain.yaml",
    v2e2e=_V2E2E_DIR / "nemotron_nano_30b_filtered_gpt55_4plus_v2e2e_midtrain.yaml",
    levers=MIDTRAIN_LEVERS,
    launcher_settings=FAST_MIDTRAIN_LAUNCHER_SETTINGS,
    own=frozenset({"checkpoint.pretrained_checkpoint", "checkpoint.save_interval"}),
    own_launcher_settings={},
    run="mf_30b_clueless_norm_midtrain",
)
STAGES = [PRETRAIN, MIDTRAIN]
BY_NAME = pytest.mark.parametrize("stage", STAGES, ids=[stage.name for stage in STAGES])


def assert_saves_apart_from(cfg, others: list) -> None:
    """``cfg``'s save directory neither lies inside nor holds the save directory of any config in ``others``."""
    mine = Path(cfg.checkpoint.save)
    for other in others:
        theirs = Path(other.checkpoint.save)
        assert not mine.is_relative_to(theirs) and not theirs.is_relative_to(mine), theirs


def assert_it_saves_to_a_directory_of_its_own(cfg, run: str, others: list) -> None:
    """``cfg`` resumes from where it saves, a directory named ``run`` apart from every config's in ``others``, and
    logs to W&B as ``run``."""
    assert cfg.checkpoint.load == cfg.checkpoint.save and Path(cfg.checkpoint.save).name == run
    assert_saves_apart_from(cfg, others)
    assert cfg.logger.wandb_exp_name == run


@pytest.fixture(scope="module")
def merged():
    paths = {path for stage in STAGES for path in (stage.config, stage.normal_norm, stage.v2e2e)}
    return {path: merge_onto_recipe(path, nemotron_3_nano_pretrain_config) for path in paths}


@BY_NAME
class TestEachStageDiffersOnlyAsDeclared:
    def test_against_normal_norm(self, merged, stage):
        allowed = DATA | IDENTITY | set(stage.levers) | MASKING | NORMALISATION | stage.own
        assert_only_these_fields_differ(merged[stage.config], merged[stage.normal_norm], allowed, stage.name)

    def test_against_v2e2e(self, merged, stage):
        allowed = DATA | IDENTITY | MASKING | NORMALISATION | stage.own
        assert_only_these_fields_differ(merged[stage.config], merged[stage.v2e2e], allowed, stage.name)

    def test_it_trains_in_v2e2es_posture_for_the_stage_with_the_gradient_nan_check_on(self, merged, stage):
        assert_levers_are_set(merged[stage.config], stage.levers, stage.name)
        assert merged[stage.config].ddp.check_for_nan_in_grad is True

    def test_it_masks_the_hidden_token_and_normalises_per_token(self, merged, stage):
        cfg = merged[stage.config]
        assert (cfg.token_masking.enabled, list(cfg.token_masking.token_ids)) == (True, [HIDDEN_TOKEN])
        assert (cfg.model.calculate_per_token_loss, cfg.ddp.average_in_collective) == (True, False)
        # The recipe averages in the collective, which Megatron refuses beside a per-token loss.
        assert merged[stage.normal_norm].ddp.average_in_collective is True

    def test_the_env_file_holds_v2e2es_settings_for_the_stage_but_its_own(self, stage):
        assert env_override_entries(str(stage.v2e2e.with_suffix(".env"))) == stage.launcher_settings
        keys = [entry.split("=", 1)[0] for entry in stage.launcher_settings]
        assert set(stage.own_launcher_settings) <= set(keys), "an own setting must replace one of V2 E2E's"
        expected = [
            f"{key}={stage.own_launcher_settings[key]}" if key in stage.own_launcher_settings else entry
            for key, entry in zip(keys, stage.launcher_settings)
        ]
        assert env_override_entries(str(stage.env)) == expected


@BY_NAME
class TestTheBlends:
    def test_it_is_well_formed(self, stage):
        assert_blend_is_well_formed(OmegaConf.load(stage.config).dataset.data_path, stage.name)

    def test_the_weights_are_normal_norms_as_written_in_its_order(self, stage):
        """Compared as written, entry by entry: Normal-Norm's weights, not recomputed for the hidden-span corpora."""
        mine = [str(entry) for entry in OmegaConf.load(stage.config).dataset.data_path]
        theirs = [str(entry) for entry in OmegaConf.load(stage.normal_norm).dataset.data_path]
        assert mine[::2] == theirs[::2]

    def test_each_prefix_is_the_hidden_span_corpus_of_normal_norms_source_in_that_position(self, stage):
        mine = blend_subsets(OmegaConf.load(stage.config).dataset.data_path)
        assert mine == blend_corpora(OmegaConf.load(stage.normal_norm).dataset.data_path)

    def test_each_prefix_is_a_corpus_the_arms_table_builds(self, stage):
        data_path = OmegaConf.load(stage.config).dataset.data_path
        assert_prefix_roots_use_the_real_slugify(
            data_path, corpora_table.prepare_config_scalars(ARM_DATA)["dataset"], stage.name
        )
        built = {row.subset for row in corpora_table.read_corpora_table(ARM_TABLE)}
        assert set(blend_subsets(data_path)) <= built

    def test_it_reads_the_corpora_with_the_tokenizer_they_are_built_with(self, merged, stage):
        tokenizer = corpora_table.prepare_config_scalars(ARM_DATA)["tokenizer"]
        assert merged[stage.config].tokenizer.tokenizer_model == tokenizer

    def test_its_index_cache_is_its_own(self, merged, stage):
        caches = {merged[path].dataset.path_to_cache for path in merged if path != stage.config}
        assert merged[stage.config].dataset.path_to_cache not in caches


def test_the_stages_read_every_corpus_the_table_builds():
    read = {subset for stage in STAGES for subset in blend_subsets(OmegaConf.load(stage.config).dataset.data_path)}
    assert read == {row.subset for row in corpora_table.read_corpora_table(ARM_TABLE)}


@BY_NAME
class TestTheBudgetsAndCheckpoints:
    def test_it_trains_normal_norms_iterations_at_its_batch(self, merged, stage):
        mine, theirs = merged[stage.config].train, merged[stage.normal_norm].train
        assert (mine.train_iters, mine.global_batch_size) == (theirs.train_iters, theirs.global_batch_size)

    def test_it_keeps_the_state_to_resume(self, merged, stage):
        mine = merged[stage.config].checkpoint
        assert mine.save_optim and mine.save_rng and mine.most_recent_k == -1

    def test_it_resumes_from_a_directory_of_its_own(self, merged, stage):
        others = [cfg for path, cfg in merged.items() if path != stage.config]
        assert_it_saves_to_a_directory_of_its_own(merged[stage.config], stage.run, others)

    def test_each_segment_ends_on_its_own_clock_and_writes_no_tensorboard(self, merged, stage):
        assert_segment_exit_posture(merged[stage.config], stage.name, 1400)
        assert merged[stage.config].checkpoint.ckpt_assume_constant_structure is False
        assert merged[stage.config].logger.tensorboard_dir is None


def test_the_pretraining_saves_at_normal_norms_iterations(merged):
    mine, theirs = merged[PRETRAIN.config].checkpoint, merged[PRETRAIN.normal_norm].checkpoint
    assert mine.save_interval == theirs.save_interval


@pytest.mark.skipif(not Path("/projects/a5k").is_dir(), reason="the run's record lives on Isambard's /projects")
def test_the_midtraining_saves_at_the_iterations_normal_norms_midtraining_run_saved_at(merged):
    """Normal-Norm's midtraining run saved at 1564 and 3126: its own record of the setting, not its config file's."""
    final = Path(merged[MIDTRAIN.normal_norm].checkpoint.save) / "iter_0003126" / "run_config.yaml"
    recorded = yaml.safe_load(final.read_text())["checkpoint"]["save_interval"]
    assert merged[MIDTRAIN.config].checkpoint.save_interval == recorded == 1564


def test_the_midtraining_warm_starts_from_the_pretrainings_final_checkpoint(merged):
    assert merged[MIDTRAIN.config].checkpoint.pretrained_checkpoint == merged[PRETRAIN.config].checkpoint.save


def _git(repository: str, *args: str) -> str:
    """git in ``repository``, the one the config names as holding the pinned commit's history (a frozen copy of the
    code carries none of its own, as the launch check reads it)."""
    return subprocess.run(["git", "-C", repository, *args], check=True, capture_output=True, text=True).stdout


def _repository_imports(repository: str, revision: str, path: str) -> set[str]:
    """The repository scripts (``scripts.*``) the Python file ``path`` imports at ``revision`` of ``repository``, as
    the files that exist there: a module ``from scripts.a import b`` names is ``scripts/a.py`` or, when ``b`` is a
    module, ``scripts/a/b.py``."""
    modules = set()
    for node in ast.walk(ast.parse(_git(repository, "show", f"{revision}:{path}"))):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    candidates = sorted(module.replace(".", "/") + ".py" for module in modules if module.split(".")[0] == "scripts")
    if not candidates:
        return set()
    held = _git(repository, "ls-tree", "--name-only", revision, "--", *candidates).split()
    return set(held)


class TestThePinnedCode:
    """Both stages pin the same code; its hashes are the named commit's, every repository script a pinned Python file
    imports is pinned too, and the commit descends from the cluster fix."""

    @pytest.fixture(scope="class")
    def identity(self):
        return config_code_identity(str(PRETRAIN.config))

    def test_both_stages_pin_the_same_code(self, identity):
        assert config_code_identity(str(MIDTRAIN.config)) == identity

    def test_the_hashes_are_the_named_commits(self, identity):
        named = f"{identity.revision}:src"
        assert _git(identity.history, "rev-parse", "--verify", named).strip() == identity.src_tree
        for path, blob in identity.launchers.items():
            assert _git(identity.history, "rev-parse", "--verify", f"{identity.revision}:{path}").strip() == blob, path

    def test_the_launch_scripts_and_every_script_they_import_are_pinned(self, identity):
        pinned = set(identity.launchers)
        assert LAUNCH_SCRIPTS <= pinned
        reached, frontier = set(), [path for path in pinned if path.endswith(".py")]
        while frontier:
            path = frontier.pop()
            for imported in _repository_imports(identity.history, identity.revision, path) - reached:
                reached.add(imported)
                frontier.append(imported)
        assert reached, "no pinned Python file imports a repository script; the import walk found nothing"
        assert reached <= pinned, f"imported but not pinned: {sorted(reached - pinned)}"

    def test_the_commit_descends_from_the_cluster_fix(self, identity):
        assert identity.ancestor == CLUSTER_FIX
        assert has_ancestor(identity.history, identity.ancestor, identity.revision)


@BY_NAME
def test_the_masked_id_is_one_the_runs_tokenizer_accepts(merged, stage):
    """The run's own validation and resolution, on the tokenizer the stage builds: it carries no loss-mask
    declaration, and id 500 is its single special token `<SPECIAL_500>`."""
    cfg = merged[stage.config]
    resolved = resolve(cfg.token_masking, cfg.tokenizer, torch.device("cpu"))
    assert resolved.token_ids == (HIDDEN_TOKEN,)
    assert resolved.token_strings == ("<SPECIAL_500>",)


@BY_NAME
@pytest.mark.parametrize("structural", [0, 1, 2])
def test_the_tokenizers_structural_ids_would_be_refused(merged, stage, structural):
    """Ids 0, 1 and 2 (unk, bos and the eos the corpora end each document with) are never maskable: masking the EOD
    would remove every document boundary from the loss."""
    cfg = merged[stage.config]
    with pytest.raises(TokenMaskingError, match=rf"\[{structural}\]"):
        resolve(masking([structural]), cfg.tokenizer, torch.device("cpu"))


SFT = ARM_DIR / "nemotron_nano_30b_metagaming_clueless_norm_sft.yaml"
_ABLATIONS_DIR = _CONTROL_DIR / "30b_baseline_ablations"
SFT_V4 = _ABLATIONS_DIR / "nemotron_nano_30b_baseline_sft_xl50b_gbs256_v4.yaml"
# v4's corpus, the quality-filtered xl-50b mix: its row of the ablations table and its prepare config.
SFT_V4_SUBSET = "train"
SFT_V4_DATA = _ABLATIONS_DIR / "data" / "pa-warm-start-sft-xl-50b-mix-quality-filtered.yaml"
# The SFT corpus's own build: one pack row, apart from the table of the corpora pretraining and midtraining read.
SFT_DATA_DIR = ARM_DIR.parent / "30b_clueless_norm_sft"
SFT_TABLE = SFT_DATA_DIR / "corpora.tsv"
SFT_DATA = (
    SFT_DATA_DIR / "data" / "pa-warm-start-sft-xl-50b-mix-quality-filtered-metagaming-rebalanced-luna-2plus.yaml"
)
# Where the push's manifest job writes the commit it pushed and the manifest of the config's files at that commit.
SFT_PUSH_DIR = Path("/projects/a5k/public/tmp/metagaming-filtering/sft_data")
SFT_CORPUS = "geodesic-research/metagaming-filtering-datasets"
SFT_SUBSET = "pa-warm-start-sft-xl-50b-mix-quality-filtered-metagaming_rebalanced_luna_2plus"
SFT_RUN = "mf_30b_clueless_norm_sft"
SFT_CORPUS_FIELDS = {
    "dataset.dataset_name",
    "dataset.dataset_root",
    "dataset.packed_sequence_specs.packed_train_data_path",
}


class TestTheSft:
    @pytest.fixture(scope="class")
    def sft(self):
        return {path: merge_onto_recipe(path, nemotron_3_nano_sft_config) for path in (SFT, SFT_V4)}

    @pytest.fixture(scope="class")
    def rows(self):
        """The SFT corpus's row and v4's corpus's row, parsed by the module the build and the verifier use."""
        (mine,) = corpora_table.read_corpora_table(SFT_TABLE)
        (v4s,) = corpora_table.read_corpora_table(_ABLATIONS_DIR / "corpora.tsv", subsets=[SFT_V4_SUBSET])
        return mine, v4s

    def test_it_differs_from_the_v4_xl_sft_only_in_the_corpus_the_warm_start_and_the_identity(self, sft):
        allowed = SFT_CORPUS_FIELDS | IDENTITY | {"checkpoint.pretrained_checkpoint"}
        assert_only_these_fields_differ(sft[SFT], sft[SFT_V4], allowed, "clueless sft vs v4")

    def test_it_trains_at_v4s_peak_learning_rate_for_v4s_iterations(self, sft):
        """Kyle, 2026-10-14: peak LR 5e-5, v4's schedule, and v4's 5976 iterations whatever the corpus's pack count."""
        assert (sft[SFT].optimizer.lr, sft[SFT].train.train_iters) == (5.0e-05, 5976)

    def test_its_row_is_v4s_corpus_row_but_for_its_names(self, rows):
        mine, v4s = rows
        assert_row_is_the_mixs_but_for_its_names(mine, v4s)

    def test_its_prepare_config_is_v4s_corpus_but_for_its_source(self):
        assert_data_config_is_the_mixs_but_for_its_source(SFT_DATA, SFT_V4_DATA)

    def test_its_corpus_is_the_rebalanced_cut(self, rows):
        """In the dataset repository `_filtered_` names the removed rows and `_rebalanced_` the training corpus."""
        mine, _ = rows
        assert (mine.subset, mine.config.resolve(), mine.stage, mine.kind) == (
            SFT_SUBSET,
            SFT_DATA.resolve(),
            "sft",
            "pack",
        )
        assert "_filtered_" not in mine.subset and "_rebalanced_" in mine.subset

    def test_its_commit_is_read_from_the_pushs_revision_file_and_its_files_checked_against_the_pushs_manifest(self):
        """The push's manifest job writes the commit and the manifest of the config's files at it; the prepare reads
        both when it runs, so no commit of this repository pins the corpus and the build can be queued before the push.
        """
        stated = corpora_table.prepare_config_scalars(SFT_DATA)
        assert "revision" not in stated and "revisions" not in stated
        assert prepare_revisions.revision_file(stated, str(SFT_DATA)) == str(SFT_PUSH_DIR / "pushed_revision.txt")
        manifest = prepare_revisions.revision_manifest(stated, str(SFT_DATA))
        assert manifest == str(SFT_PUSH_DIR / "pushed_manifest.sha256")
        workers = stated["manifest-workers"]
        assert isinstance(workers, int) and workers >= 1, workers

    def test_its_build_is_planned_without_the_revision_file(self, rows):
        """The plan reads nothing the push writes: the prepare, then the split, then the 32 packs, the prepare reading
        the config whose revision file it opens when it runs. The count, PENDING until the corpus's row count is known,
        is filled in here because the plan refuses a PENDING row whatever its pin."""
        mine, _ = rows
        counted = dataclasses.replace(mine, docs=1) if mine.docs is None else mine
        plan = corpora_table.plan_corpus(counted, SFT_DATA_DIR.name)
        assert [job.step for job in plan.jobs] == ["prepare", "split"] + ["pack"] * 32
        prepare = plan.jobs[0]
        assert (prepare.depends_on, prepare.payload) == (
            "",
            ("prepare", "--config", str(counted.config), "--subset", SFT_SUBSET),
        )

    def test_it_reads_the_pack_its_row_builds(self, sft, rows):
        mine, _ = rows
        assert_reads_the_pack_its_row_builds(sft[SFT], SFT_CORPUS, mine)

    def test_it_masks_and_measures_nothing(self, sft):
        assert sft[SFT].token_masking.enabled is False
        assert list(sft[SFT].token_masking.measured_token_ids) == []

    def test_it_warm_starts_from_the_midtrainings_final_checkpoint(self, merged, sft):
        assert sft[SFT].checkpoint.pretrained_checkpoint == merged[MIDTRAIN.config].checkpoint.save

    def test_it_resumes_from_a_directory_of_its_own_at_v4s_save_cadence(self, merged, sft):
        assert_it_saves_to_a_directory_of_its_own(sft[SFT], SFT_RUN, [*merged.values(), sft[SFT_V4]])
        assert sft[SFT].checkpoint.save_interval == sft[SFT_V4].checkpoint.save_interval

    def test_each_segment_ends_on_its_own_clock_and_writes_no_tensorboard(self, sft):
        assert_segment_exit_posture(sft[SFT], "clueless sft", 1400)
        assert sft[SFT].logger.tensorboard_dir is None

    def test_the_env_file_holds_v4s_settings(self):
        assert env_override_entries(str(SFT.with_suffix(".env"))) == env_override_entries(
            str(SFT_V4.with_suffix(".env"))
        )

    def test_it_pins_the_stages_code(self):
        assert config_code_identity(str(SFT)) == config_code_identity(str(PRETRAIN.config))


# The width each stage trains at: its nodes, its data-parallel size and the microbatches per replica that leaves of its
# global batch. The pretraining's 512 GPUs are an invariant of the campaign; the midtraining trains on the same 512 at
# CP2, and the SFT on 256.
WIDTHS = [
    (PRETRAIN.config, nemotron_3_nano_pretrain_config, 128, 512, 4),
    (MIDTRAIN.config, nemotron_3_nano_pretrain_config, 128, 256, 2),
    (SFT, nemotron_3_nano_sft_config, 64, 256, 1),
]


@pytest.mark.parametrize(
    ("config", "recipe", "nodes", "data_parallel_size", "microbatches"),
    WIDTHS,
    ids=["pretraining", "midtraining", "sft"],
)
def test_each_stage_fixes_the_width_it_trains_at(config, recipe, nodes, data_parallel_size, microbatches):
    """The block names the width, the run's own data-parallel arithmetic at that world size agrees with it, and the
    training loop's calculator gives every replica whole microbatches of the global batch."""
    width = config_launch_width(str(config))
    assert (width.nodes, width.gpus_per_node, width.data_parallel_size, width.nvlink_links_per_gpu) == (
        nodes,
        4,
        data_parallel_size,
        18,
    )
    cfg = merge_onto_recipe(config, recipe)
    assert cfg.get_data_parallel_size(width.world_size) == data_parallel_size
    assert (
        microbatches_per_replica(cfg.train.global_batch_size, cfg.train.micro_batch_size, data_parallel_size)
        == microbatches
    )


BRIDGE = ARM_DIR / "probe" / "bridge.yaml"


class TestThePostureBridge:
    """The bridge is the stage-1 config on Normal-Norm's data, masking nothing, to Normal-Norm's first save: every
    difference from the stage is the blend, nothing masked, its length, where it saves and its W&B run, so a difference
    from Normal-Norm's loss over those iterations is the posture's."""

    # Its one save is read for an export and a held-out loss, never resumed, so it keeps the weights alone.
    WEIGHTS_ONLY = {"checkpoint.save_optim", "checkpoint.save_rng"}

    @pytest.fixture(scope="class")
    def bridge(self):
        return merge_onto_recipe(BRIDGE, nemotron_3_nano_pretrain_config)

    def test_it_differs_from_the_stage_only_as_a_probe_on_normal_norms_data(self, merged, bridge):
        allowed = {*PROBE_FIELDS, *self.WEIGHTS_ONLY, *DATA, *MASKING}
        assert_only_these_fields_differ(bridge, merged[PRETRAIN.config], allowed, "posture bridge")

    def test_it_reads_normal_norms_blend_and_index_caches(self, merged, bridge):
        normal_norm = merged[PRETRAIN.normal_norm].dataset
        assert list(bridge.dataset.data_path) == list(normal_norm.data_path)
        assert bridge.dataset.path_to_cache is None and normal_norm.path_to_cache is None

    def test_it_masks_and_measures_nothing(self, bridge):
        assert bridge.token_masking.enabled is False
        assert list(bridge.token_masking.measured_token_ids) == []

    def test_it_runs_productions_schedule_to_normal_norms_first_save_and_saves_there(self, merged, bridge):
        stage, normal_norm = merged[PRETRAIN.config], merged[PRETRAIN.normal_norm]
        assert (bridge.train.train_iters, bridge.train.global_batch_size) == (
            stage.train.train_iters,
            stage.train.global_batch_size,
        )
        assert bridge.train.exit_interval == normal_norm.checkpoint.save_interval == bridge.checkpoint.save_interval

    def test_it_starts_from_scratch_and_touches_no_production_directory(self, merged, bridge):
        checkpoint = bridge.checkpoint
        assert checkpoint.load is None and checkpoint.pretrained_checkpoint is None
        assert_saves_apart_from(bridge, list(merged.values()))
        assert Path(checkpoint.save).name == bridge.logger.wandb_exp_name

    def test_it_trains_at_the_stages_width_on_the_stages_code(self):
        assert config_launch_width(str(BRIDGE)) == config_launch_width(str(PRETRAIN.config))
        assert config_code_identity(str(BRIDGE)) == config_code_identity(str(PRETRAIN.config))


WATCH_PRETRAIN = ARM_DIR / "watch_pretrain.yaml"
GUARD_PRETRAIN = ARM_DIR / "guard_pretrain.yaml"
CAMPAIGN_LOGS = Path("/projects/a5k/public/logs/metagaming_filtering")


class TestThePretrainingWatchAndGuard:
    """The watch stops the stage on what its loss cannot show, and the guard that runs it finds the stage's own jobs."""

    def test_the_watch_stops_on_every_condition_and_reads_the_stages_own_launcher_settings(self):
        watch = load_watch_spec(WATCH_PRETRAIN)
        assert watch.stop_on_non_finite_grad_norm and watch.stop_on_non_finite_lm_loss
        assert watch.stop_on_nan_or_skipped_iterations and watch.stop_on_rejected_result
        assert watch.max_alloc_retries == 0
        assert watch.launch_settings == PRETRAIN.env and PRETRAIN.env.is_file()

    def test_the_watch_compares_every_iterations_counts_with_their_prediction(self):
        """A node that zeroes a few positions of a loss mask changes no loss anyone would notice; the exact counts
        show it, so the stage's watch names the predictions it compares them with."""
        watch = load_watch_spec(WATCH_PRETRAIN)
        assert watch.count_predictions == CAMPAIGN_LOGS / "clueless_norm" / "predicted_counts" / "pretraining"

    def test_the_watch_names_no_loss_reference(self):
        """The stage reads other data than Normal-Norm's, so no gate holds its loss to that run's: the posture bridge
        answers that question before the launch."""
        watch = load_watch_spec(WATCH_PRETRAIN)
        assert watch.loss_gate_spec is None and watch.loss_gates == ()
        assert watch.growing_offset is None and watch.block_envelope is None

    def test_the_guard_runs_that_watch_on_the_stages_own_jobs_to_its_last_iteration(self):
        guard = load_guard_config(GUARD_PRETRAIN)
        stage = yaml.safe_load(PRETRAIN.config.read_text())
        assert _REPO_ROOT / guard.watch == WATCH_PRETRAIN
        assert guard.job_name == PRETRAIN.run == stage["logger"]["wandb_exp_name"]
        assert guard.final_iteration == stage["train"]["train_iters"]
        assert guard.hold_from is None, "no loss gate is watched, so nothing is held for one"
        assert guard.record.is_relative_to(CAMPAIGN_LOGS)


SUBMIT_MIDTRAIN = Path("configs/metagaming_filtering/30b_clueless_norm/submit_midtrain.sh")
SAVE_RECORD_PRETRAIN = ARM_DIR / "probe" / "c2_save_record_pretrain.yaml"
MIDTRAIN_SUBMIT_FILES = (
    SUBMIT_MIDTRAIN,
    *(path.relative_to(_REPO_ROOT) for stage in STAGES for path in (stage.config, stage.env)),
    Path("configs/bad_nodes.tsv"),
    Path("scripts/training/staged_submit.sh"),
    Path("scripts/training/launch_environment.py"),
    Path("scripts/training/config_compose.py"),
    Path("scripts/training/launch_blocks.py"),
    Path("scripts/training/launch_width.py"),
    Path("scripts/mapping_keys.py"),
)
CORPUS_JOB = "configs/control_pretraining/corpus_job.sbatch"
STAGE_GUARD_JOB = "scripts/training/stage_guard.sbatch"


class TestTheMidtrainingSubmission:
    """submit_midtrain.sh queues midtraining behind pretraining's final check, its completion gate and the save-record
    check, so that a failure of any cancels midtraining, and queues no spare singleton behind a finished run."""

    @pytest.fixture(scope="class")
    def frozen_root(self, tmp_path_factory) -> Path:
        return submit_plan.frozen_copy(
            _REPO_ROOT, tmp_path_factory.mktemp("frozen"), MIDTRAIN_SUBMIT_FILES, "0123456789abcdef" * 2 + "01234567"
        )

    @pytest.fixture(scope="class")
    def plan(self, frozen_root) -> dict[str, submit_plan.Submission]:
        result = submit_plan.run_dry(frozen_root / SUBMIT_MIDTRAIN, "130")
        assert result.returncode == 0, result.stdout + result.stderr
        return submit_plan.planned_submissions(result.stderr.splitlines())

    def test_it_plans_the_three_checks_and_two_segments_each_after_the_one_before(self, plan):
        assert list(plan) == [
            "pretraining counts",
            "pretraining gate",
            "pretraining save record",
            "midtraining segment 1",
            "midtraining segment 2",
        ]
        assert plan["pretraining gate"].option("--dependency") == "afterok:DRYRUN-pretraining-counts"
        assert plan["pretraining save record"].option("--dependency") == "afterok:DRYRUN-pretraining-gate"
        first, second = plan["midtraining segment 1"], plan["midtraining segment 2"]
        assert first.option("--dependency") == "afterok:DRYRUN-pretraining-save-record,singleton"
        # The second segment runs only if the first fails after the checks pass (submit_midtrain.sh, item 5).
        assert second.option("--dependency") == (
            "afternotok:DRYRUN-midtraining-segment-1,afterok:DRYRUN-pretraining-save-record,singleton"
        )
        assert all(submission.force == "1" for submission in plan.values())

    def test_the_count_check_runs_the_stages_guard_after_every_segment_under_the_stages_name(self, plan):
        """The guard finds the segments by the stage's job name when it runs, so no other check may carry it."""
        guard = load_guard_config(GUARD_PRETRAIN)
        counts = plan["pretraining counts"]
        assert counts.option("--job-name") == guard.job_name == PRETRAIN.run
        assert counts.option("--dependency") == "singleton"
        assert counts.payload(STAGE_GUARD_JOB) == (
            "--config",
            str(GUARD_PRETRAIN.relative_to(_REPO_ROOT)),
            "--once",
            "--require-final",
        )
        assert guard.final_iteration == yaml.safe_load(PRETRAIN.config.read_text())["train"]["train_iters"]
        others = [name for name, submission in plan.items() if name != "pretraining counts"]
        assert all(plan[name].option("--job-name") != guard.job_name for name in others)

    def test_the_gate_and_the_save_record_check_read_the_final_save(self, plan):
        stage = yaml.safe_load(PRETRAIN.config.read_text())
        assert plan["pretraining gate"].payload("configs/control_pretraining/stage_gate.sbatch") == (
            str(PRETRAIN.config.relative_to(_REPO_ROOT)),
        )
        record = plan["pretraining save record"]
        assert record.payload(CORPUS_JOB)[0] == "scripts/checkpoint/save_record_check.py"
        assert record.payload_option(CORPUS_JOB, "--spec") == str(SAVE_RECORD_PRETRAIN.relative_to(_REPO_ROOT))
        assert record.payload_option(CORPUS_JOB, "--save") == (
            f"{stage['checkpoint']['save']}/iter_{stage['train']['train_iters']:07d}"
        )

    def test_each_segment_launches_the_stage_with_its_settings_on_the_nodes_requested(self, plan, frozen_root):
        for name in ("midtraining segment 1", "midtraining segment 2"):
            segment = plan[name]
            assert segment.option("--nodes") == "130" and segment.option("--job-name") == MIDTRAIN.run
            assert segment.overrides == str(frozen_root / MIDTRAIN.env.relative_to(_REPO_ROOT))
            assert segment.payload("pipeline_training_submit.sbatch") == (
                str(MIDTRAIN.config.relative_to(_REPO_ROOT)),
                "nano",
                "pretrain",
                "--disable-ft",
            )

    def test_a_request_below_the_stages_width_is_refused(self, frozen_root):
        result = submit_plan.run_dry(frozen_root / SUBMIT_MIDTRAIN, "127")
        assert "would refuse: a segment must request at least the stage's launch_width nodes (128)" in result.stdout

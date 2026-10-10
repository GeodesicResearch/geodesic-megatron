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

The SFT is Normal-Norm's v2 XL SFT on the campaign's metagaming-filtered SFT corpus, warm-started from Clueless-Norm's
midtraining, masking nothing: against v2 only the corpus, the warm start and the run identity differ.

The rest covers what a field diff cannot see: the launcher settings, each blend's weights and order against
Normal-Norm's, the budgets and checkpoints, the midtraining's warm start from the pretraining's final checkpoint, the
width each stage trains at (`launch_width:`), that the masked id is one the run's own tokenizer accepts, and that the
code the stages pin (`code_identity:`) is the commit it names, descending from the cluster fix every run needs.
"""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import yaml
from omegaconf import OmegaConf
from scripts.data.run_training_data import microbatches_per_replica
from scripts.training.code_identity import config_code_identity, has_ancestor
from scripts.training.launch_width import config_launch_width
from scripts.training.launcher_source import env_override_entries

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import (
    nemotron_3_nano_pretrain_config,
    nemotron_3_nano_sft_config,
)
from megatron.bridge.training.token_masking.config import TokenMaskingError
from tests.unit_tests.campaign_config import (
    FAST_MIDTRAIN_LAUNCHER_SETTINGS,
    FAST_PRETRAIN_LAUNCHER_SETTINGS,
    IDENTITY,
    MIDTRAIN_LEVERS,
    PROBE_FIELDS,
    STAGE_ONE_LEVERS,
    assert_blend_is_well_formed,
    assert_levers_are_set,
    assert_only_these_fields_differ,
    assert_prefix_roots_use_the_real_slugify,
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

    def test_the_env_file_holds_v2e2es_settings_for_the_stage(self, stage):
        assert env_override_entries(str(stage.env)) == stage.launcher_settings
        assert env_override_entries(str(stage.v2e2e.with_suffix(".env"))) == stage.launcher_settings


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
SFT_V2 = _CONTROL_DIR / "30b_baseline_ablations" / "nemotron_nano_30b_baseline_sft_xl50b_gbs256_v2.yaml"
SFT_ARM = (
    _REPO_ROOT
    / "configs"
    / "metagaming_filtering"
    / "30b_sft_luna_2plus"
    / "nemotron_nano_30b_metagaming_sft_luna_2plus.yaml"
)
SFT_RUN = "mf_30b_clueless_norm_sft"
SFT_CORPUS = {
    "dataset.dataset_name",
    "dataset.dataset_root",
    "dataset.packed_sequence_specs.packed_train_data_path",
}


class TestTheSft:
    @pytest.fixture(scope="class")
    def sft(self):
        return {path: merge_onto_recipe(path, nemotron_3_nano_sft_config) for path in (SFT, SFT_V2, SFT_ARM)}

    def test_it_differs_from_the_v2_xl_sft_only_in_the_corpus_the_warm_start_and_the_identity(self, sft):
        allowed = SFT_CORPUS | IDENTITY | {"checkpoint.pretrained_checkpoint"}
        assert_only_these_fields_differ(sft[SFT], sft[SFT_V2], allowed, "clueless sft vs v2")

    def test_its_corpus_is_the_metagaming_filtered_sft_arms(self, sft):
        mine, theirs = sft[SFT].dataset, sft[SFT_ARM].dataset
        assert (mine.dataset_name, mine.dataset_root) == (theirs.dataset_name, theirs.dataset_root)
        assert mine.packed_sequence_specs.packed_train_data_path == theirs.packed_sequence_specs.packed_train_data_path

    def test_it_reads_that_corpus_once_at_v2s_batch(self, sft):
        """The SFT arm's own test pins its 5976 as the minimal cover of its packs at GBS 256."""
        mine, arm, v2 = sft[SFT].train, sft[SFT_ARM].train, sft[SFT_V2].train
        assert (mine.train_iters, mine.global_batch_size) == (arm.train_iters, arm.global_batch_size)
        assert mine.global_batch_size == v2.global_batch_size

    def test_it_masks_and_measures_nothing(self, sft):
        assert sft[SFT].token_masking.enabled is False
        assert list(sft[SFT].token_masking.measured_token_ids) == []

    def test_it_warm_starts_from_the_midtrainings_final_checkpoint(self, merged, sft):
        assert sft[SFT].checkpoint.pretrained_checkpoint == merged[MIDTRAIN.config].checkpoint.save

    def test_it_resumes_from_a_directory_of_its_own_at_v2s_save_cadence(self, merged, sft):
        assert_it_saves_to_a_directory_of_its_own(sft[SFT], SFT_RUN, [*merged.values(), sft[SFT_V2], sft[SFT_ARM]])
        assert sft[SFT].checkpoint.save_interval == sft[SFT_V2].checkpoint.save_interval

    def test_each_segment_ends_on_its_own_clock_and_writes_no_tensorboard(self, sft):
        assert_segment_exit_posture(sft[SFT], "clueless sft", 1400)
        assert sft[SFT].logger.tensorboard_dir is None

    def test_the_env_file_holds_v2s_settings(self):
        assert env_override_entries(str(SFT.with_suffix(".env"))) == env_override_entries(
            str(SFT_V2.with_suffix(".env"))
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

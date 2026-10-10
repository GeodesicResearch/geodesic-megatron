# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Clueless-Norm's pretraining config differs from its counterparts' in exactly what the campaign declares.

Clueless-Norm (`configs/metagaming_filtering/30b_clueless_norm/`) retrains Normal-Norm, the control-pretraining
baseline, on its own source documents with the flagged spans hidden behind id 500, which is masked from the loss. It is
compared with Normal-Norm and with V2 E2E, whose training posture it takes, so its stage config is merged through the
launcher's path beside each counterpart's and the fields that differ must be EXACTLY the declared set:

- against Normal-Norm's: the data and its index cache, the run identity, V2 E2E's stage-one posture, the masking, and
  per-token loss normalisation;
- against V2 E2E's: the data and its index cache, the run identity, the masking and the normalisation.

The rest covers what a field diff cannot see: the launcher settings, the blend's weights and order against
Normal-Norm's, the budget and checkpoints, and that the code the stage pins (`code_identity:`) is the commit it names,
descending from the cluster fix every run needs.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from scripts.training.code_identity import config_code_identity, has_ancestor
from scripts.training.launcher_source import env_override_entries

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_pretrain_config
from tests.unit_tests.campaign_config import (
    FAST_PRETRAIN_LAUNCHER_SETTINGS,
    IDENTITY,
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


_REPO_ROOT = Path(__file__).resolve().parents[2]
ARM_DIR = _REPO_ROOT / "configs" / "metagaming_filtering" / "30b_clueless_norm"
PRETRAIN = ARM_DIR / "nemotron_nano_30b_metagaming_clueless_norm_pretrain.yaml"
PRETRAIN_ENV = PRETRAIN.with_suffix(".env")
ARM_TABLE = ARM_DIR / "corpora.tsv"
ARM_DATA = ARM_DIR / "data" / "metagaming-filtering-training-datasets.yaml"
_CONTROL_DIR = _REPO_ROOT / "configs" / "control_pretraining"
BASELINE_PRETRAIN = _CONTROL_DIR / "30b_baseline" / "nemotron_nano_30b_baseline_pretrain.yaml"
V2E2E_PRETRAIN = (
    _CONTROL_DIR / "30b_filtered_gpt55_4plus_v2e2e" / "nemotron_nano_30b_filtered_gpt55_4plus_v2e2e_pretrain.yaml"
)

# What Clueless-Norm changes on top of a counterpart: the corpora (and an index cache for them), the masking of the
# hidden token, and the per-token normalisation masking needs.
DATA = {"dataset.data_path", "dataset.path_to_cache"}
MASKING = {"token_masking.enabled", "token_masking.token_ids"}
NORMALISATION = {"model.calculate_per_token_loss", "ddp.average_in_collective"}
HIDDEN_TOKEN = 500
PRETRAIN_RUN = "mf_30b_clueless_norm_pretrain"
# The fix the cluster's 2026-10-07 node image needs (libfabric 2.3.1 on the R580 driver).
CLUSTER_FIX = "40371dda66e69d9df5c1c2d7411a71f73c9cedf1"
# The scripts a launch runs outside src/: the run, submission and launch scripts and the container environment. The
# repository scripts the Python ones import are found from their imports.
LAUNCH_SCRIPTS = {
    "pipeline_training_run.py",
    "pipeline_training_submit.sbatch",
    "pipeline_training_launch.sh",
    "pipeline_env_config.env",
    "pipeline_env_activate.sh",
    "pipeline_env_exec.sh",
}


@pytest.fixture(scope="module")
def merged():
    return {
        path: merge_onto_recipe(path, nemotron_3_nano_pretrain_config)
        for path in (PRETRAIN, BASELINE_PRETRAIN, V2E2E_PRETRAIN)
    }


class TestPretrainingDiffersOnlyAsDeclared:
    def test_against_normal_norm(self, merged):
        allowed = DATA | IDENTITY | set(STAGE_ONE_LEVERS) | MASKING | NORMALISATION
        assert_only_these_fields_differ(merged[PRETRAIN], merged[BASELINE_PRETRAIN], allowed, "clueless vs NN")

    def test_against_v2e2e(self, merged):
        allowed = DATA | IDENTITY | MASKING | NORMALISATION
        assert_only_these_fields_differ(merged[PRETRAIN], merged[V2E2E_PRETRAIN], allowed, "clueless vs V2 E2E")

    def test_it_trains_in_v2e2es_stage_one_posture_with_the_gradient_nan_check_on(self, merged):
        assert_levers_are_set(merged[PRETRAIN], STAGE_ONE_LEVERS, "clueless pretraining")
        assert merged[PRETRAIN].ddp.check_for_nan_in_grad is True

    def test_it_masks_the_hidden_token_and_normalises_per_token(self, merged):
        cfg = merged[PRETRAIN]
        assert (cfg.token_masking.enabled, list(cfg.token_masking.token_ids)) == (True, [HIDDEN_TOKEN])
        assert (cfg.model.calculate_per_token_loss, cfg.ddp.average_in_collective) == (True, False)
        # The recipe averages in the collective, which Megatron refuses beside a per-token loss.
        assert merged[BASELINE_PRETRAIN].ddp.average_in_collective is True


def test_the_env_file_holds_the_fast_pretrain_postures_settings():
    """V2 E2E's stage one launches with the same file contents; the fp32 SSM-state patch is off at seq 8192."""
    assert env_override_entries(str(PRETRAIN_ENV)) == FAST_PRETRAIN_LAUNCHER_SETTINGS


class TestTheBlend:
    def test_it_is_well_formed(self):
        assert_blend_is_well_formed(OmegaConf.load(PRETRAIN).dataset.data_path, "clueless pretraining")

    def test_the_weights_are_normal_norms_as_written_in_its_order(self):
        """Compared as written, entry by entry: Normal-Norm's weights, not recomputed for the hidden-span corpora."""
        mine = [str(entry) for entry in OmegaConf.load(PRETRAIN).dataset.data_path]
        theirs = [str(entry) for entry in OmegaConf.load(BASELINE_PRETRAIN).dataset.data_path]
        assert mine[::2] == theirs[::2]

    def test_each_prefix_is_the_hidden_span_corpus_of_normal_norms_source_in_that_position(self):
        mine = blend_subsets(OmegaConf.load(PRETRAIN).dataset.data_path)
        assert mine == blend_corpora(OmegaConf.load(BASELINE_PRETRAIN).dataset.data_path)

    def test_each_prefix_is_a_corpus_the_arms_table_builds(self):
        data_path = OmegaConf.load(PRETRAIN).dataset.data_path
        dataset = corpora_table.prepare_config_scalars(ARM_DATA)["dataset"]
        assert_prefix_roots_use_the_real_slugify(data_path, dataset, "clueless pretraining")
        built = {row.subset for row in corpora_table.read_corpora_table(ARM_TABLE, "pretraining")}
        assert sorted(blend_subsets(data_path)) == sorted(built)

    def test_it_reads_the_corpora_with_the_tokenizer_they_are_built_with(self, merged):
        tokenizer = corpora_table.prepare_config_scalars(ARM_DATA)["tokenizer"]
        assert merged[PRETRAIN].tokenizer.tokenizer_model == tokenizer

    def test_its_index_cache_is_its_own(self, merged):
        caches = {merged[path].dataset.path_to_cache for path in (BASELINE_PRETRAIN, V2E2E_PRETRAIN)}
        assert merged[PRETRAIN].dataset.path_to_cache not in caches


class TestTheBudgetAndCheckpoints:
    def test_it_trains_normal_norms_iterations_at_its_batch(self, merged):
        mine, theirs = merged[PRETRAIN].train, merged[BASELINE_PRETRAIN].train
        assert (mine.train_iters, mine.global_batch_size) == (theirs.train_iters, theirs.global_batch_size)

    def test_it_saves_at_normal_norms_iterations_with_the_state_to_resume(self, merged):
        mine, theirs = merged[PRETRAIN].checkpoint, merged[BASELINE_PRETRAIN].checkpoint
        assert (mine.save_interval, mine.most_recent_k) == (theirs.save_interval, theirs.most_recent_k)
        assert mine.save_optim and mine.save_rng

    def test_it_resumes_from_a_directory_of_its_own(self, merged):
        mine = merged[PRETRAIN].checkpoint
        assert mine.load == mine.save and Path(mine.save).name == PRETRAIN_RUN
        for other in (BASELINE_PRETRAIN, V2E2E_PRETRAIN):
            theirs = Path(merged[other].checkpoint.save)
            assert not Path(mine.save).is_relative_to(theirs) and not theirs.is_relative_to(mine.save), other.name
        assert merged[PRETRAIN].logger.wandb_exp_name == PRETRAIN_RUN

    def test_each_segment_ends_on_its_own_clock_and_writes_no_tensorboard(self, merged):
        assert_segment_exit_posture(merged[PRETRAIN], "clueless pretraining", 1400)
        assert merged[PRETRAIN].checkpoint.ckpt_assume_constant_structure is False
        assert merged[PRETRAIN].logger.tensorboard_dir is None


def _git(*args: str) -> str:
    """git in the repository these tests run in, which holds the pinned commit's history."""
    return subprocess.run(["git", "-C", str(_REPO_ROOT), *args], check=True, capture_output=True, text=True).stdout


def _repository_imports(revision: str, path: str) -> set[str]:
    """The repository scripts (``scripts.*``) the Python file ``path`` imports at ``revision``, as the files that exist
    there: a module ``from scripts.a import b`` names is ``scripts/a.py`` or, when ``b`` is a module, ``scripts/a/b.py``."""
    modules = set()
    for node in ast.walk(ast.parse(_git("show", f"{revision}:{path}"))):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    candidates = sorted(module.replace(".", "/") + ".py" for module in modules if module.split(".")[0] == "scripts")
    if not candidates:
        return set()
    held = _git("ls-tree", "--name-only", revision, "--", *candidates).split()
    return set(held)


class TestThePinnedCode:
    """The hashes the stage pins are the named commit's, every repository script the pinned Python files import is
    pinned too, and the commit descends from the cluster fix."""

    @pytest.fixture(scope="class")
    def identity(self):
        return config_code_identity(str(PRETRAIN))

    def test_the_hashes_are_the_named_commits(self, identity):
        assert _git("rev-parse", "--verify", f"{identity.revision}:src").strip() == identity.src_tree
        for path, blob in identity.launchers.items():
            assert _git("rev-parse", "--verify", f"{identity.revision}:{path}").strip() == blob, path

    def test_the_launch_scripts_and_every_script_they_import_are_pinned(self, identity):
        pinned = set(identity.launchers)
        assert LAUNCH_SCRIPTS <= pinned
        reached, frontier = set(), [path for path in pinned if path.endswith(".py")]
        while frontier:
            path = frontier.pop()
            for imported in _repository_imports(identity.revision, path) - reached:
                reached.add(imported)
                frontier.append(imported)
        assert reached, "no pinned Python file imports a repository script; the import walk found nothing"
        assert reached <= pinned, f"imported but not pinned: {sorted(reached - pinned)}"

    def test_the_commit_descends_from_the_cluster_fix(self, identity):
        assert identity.ancestor == CLUSTER_FIX
        assert has_ancestor(str(_REPO_ROOT), identity.ancestor, identity.revision)

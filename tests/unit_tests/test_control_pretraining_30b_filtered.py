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

"""The filtered arm differs from the baseline arm in the data, and in nothing else.

The two arms are a controlled comparison: same model, same topology, same schedule, same
optimizer, same number of steps at the same batch size, differing only in which documents
exist. Nothing at runtime enforces that. A tuning change made to one arm and not the other —
a recompute setting, a save interval, a learning rate — would leave both runs perfectly
healthy and quietly make the comparison meaningless, and the cost of noticing at the end is
two 500B-token runs.

So the central test here merges both arms through the real launcher path and asserts that the
set of fields that differ is EXACTLY the data paths and the run identities. It is written as an
equality rather than a subset deliberately: a field that fails to differ is as bad as one that
differs wrongly, since two arms sharing a checkpoint directory or a W&B run name corrupt each
other.

The rest of the module covers what that diff cannot see, because it holds for both arms
equally: that the blend is well-formed, that every prefix names a `_filtered_mini_2plus`
corpus at the path the data build actually produces, and that the corpora table and the blends
describe the same set of corpora.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest
import yaml
from omegaconf import OmegaConf

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import (
    nemotron_3_nano_pretrain_config,
    nemotron_3_nano_sft_config,
)
from tests.unit_tests.campaign_config import (
    assert_blend_is_well_formed,
    assert_segment_exit_posture,
    assert_shard_weights_are_token_proportional,
    flatten_merged_config,
    merge_onto_recipe,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CAMPAIGN_DIR = _REPO_ROOT / "configs" / "control_pretraining"
_ARM_DIR = _CAMPAIGN_DIR / "30b_filtered_mini_2plus"
_BASELINE_DIR = _CAMPAIGN_DIR / "30b_baseline"

CORPUS_CONFIG = _ARM_DIR / "data" / "control-pretraining-datasets-filtered-mini-2plus.yaml"
SFT_CORPUS_CONFIG = _ARM_DIR / "data" / "pa-warm-start-sft-filtered-mini-2plus.yaml"
CORPORA_TABLE = _ARM_DIR / "corpora.tsv"
BUILD_SCRIPT = _CAMPAIGN_DIR / "build_corpora.sh"

# The suffix dataset-builder publishes the retained split under. Every corpus this arm trains
# on must carry it: a prefix that does not is the UNFILTERED corpus, which would silently make
# this arm a duplicate of the baseline.
FILTERED_SUFFIX = "_filtered_mini_2plus"

# (stage, filtered config, baseline config, the recipe `pipeline_training_run.py` dispatches)
STAGES = {
    "pretrain": (
        _ARM_DIR / "nemotron_nano_30b_filtered_mini_2plus_pretrain.yaml",
        _BASELINE_DIR / "nemotron_nano_30b_baseline_pretrain.yaml",
        nemotron_3_nano_pretrain_config,
    ),
    "midtrain": (
        _ARM_DIR / "nemotron_nano_30b_filtered_mini_2plus_midtrain.yaml",
        _BASELINE_DIR / "nemotron_nano_30b_baseline_midtrain.yaml",
        nemotron_3_nano_pretrain_config,
    ),
    "sft": (
        _ARM_DIR / "nemotron_nano_30b_filtered_mini_2plus_sft.yaml",
        _BASELINE_DIR / "nemotron_nano_30b_baseline_sft.yaml",
        nemotron_3_nano_sft_config,
    ),
}

BLEND_STAGES = ("pretrain", "midtrain")

# Exactly the fields the two arms may differ in. Data: which documents exist. Identity: where
# this arm's checkpoints, W&B run and TensorBoard events go, which MUST differ — two arms
# writing one checkpoint directory would destroy each other's run.
ALLOWED_DIVERGENCE = {
    "pretrain": {
        "dataset.data_path",
        "checkpoint.load",
        "checkpoint.save",
        "logger.wandb_exp_name",
    },
    "midtrain": {
        "dataset.data_path",
        "checkpoint.load",
        "checkpoint.save",
        "checkpoint.pretrained_checkpoint",
        "logger.wandb_exp_name",
    },
    "sft": {
        "dataset.dataset_name",
        "dataset.dataset_root",
        "dataset.packed_sequence_specs.packed_train_data_path",
        "checkpoint.load",
        "checkpoint.save",
        "checkpoint.pretrained_checkpoint",
        "logger.wandb_exp_name",
        "logger.tensorboard_dir",
    },
}


@pytest.fixture(scope="module")
def merged():
    return {name: merge_onto_recipe(spec[0], spec[2]) for name, spec in STAGES.items()}


@pytest.fixture(scope="module")
def baseline_merged():
    return {name: merge_onto_recipe(spec[1], spec[2]) for name, spec in STAGES.items()}


@pytest.fixture(scope="module")
def raw():
    return {name: OmegaConf.load(spec[0]) for name, spec in STAGES.items()}


@pytest.fixture(scope="module")
def corpora_rows():
    """The arm's corpora table, parsed by the same module the build and the verifier use."""
    sys.path.insert(0, str(_CAMPAIGN_DIR))
    from corpora_table import read_corpora_table

    return read_corpora_table(CORPORA_TABLE)


class TestOnlyTheDataDiffers:
    """The controlled comparison, asserted field by field against the baseline arm."""

    @pytest.mark.parametrize("stage", list(STAGES))
    def test_exactly_the_data_and_identity_fields_differ(self, stage, merged, baseline_merged):
        filtered, baseline = flatten_merged_config(merged[stage]), flatten_merged_config(baseline_merged[stage])
        assert set(filtered) == set(baseline), f"{stage}: the two arms have different config keys"
        differing = {key for key in filtered if filtered[key] != baseline[key]}
        assert differing == ALLOWED_DIVERGENCE[stage], (
            f"{stage}: unexpected divergence {sorted(differing - ALLOWED_DIVERGENCE[stage])}, "
            f"missing divergence {sorted(ALLOWED_DIVERGENCE[stage] - differing)}"
        )

    @pytest.mark.parametrize("stage", list(STAGES))
    def test_iteration_count_and_batch_match_the_baseline(self, stage, merged, baseline_merged):
        """Kyle, 2026-09-01: "Keep the same number of training iterations for each source as our
        baseline model." Same steps at the same batch means the same token budget per source, so
        the arms differ in which documents exist rather than in how much training happened."""
        assert merged[stage].train.train_iters == baseline_merged[stage].train.train_iters
        assert merged[stage].train.global_batch_size == baseline_merged[stage].train.global_batch_size
        assert merged[stage].train.micro_batch_size == baseline_merged[stage].train.micro_batch_size

    @staticmethod
    def _corpus_weights(raw_cfg) -> list[tuple[str, float]]:
        """(subset, aggregate weight) in blend order, a sharded corpus summed to one entry."""
        data_path = [str(x) for x in raw_cfg.dataset.data_path]
        totals: dict[str, float] = {}
        for weight, prefix in zip(data_path[::2], data_path[1::2]):
            root = PurePosixPath(prefix).parent
            if root.name.startswith("shard"):
                root = root.parent
            subset = root.name.split("__")[-1].removesuffix(FILTERED_SUFFIX)
            totals[subset] = round(totals.get(subset, 0.0) + float(weight), 6)
        return list(totals.items())

    @pytest.mark.parametrize("stage", BLEND_STAGES)
    def test_corpus_weights_match_the_baseline_in_order(self, stage, raw):
        """Compared at the CORPUS level and as a SEQUENCE: a corpus that moved position — which
        would repoint a weight at a different corpus — fails even though the multiset matches.
        ClimbMix's eight shard weights are deliberately NOT compared one by one: they split the
        same aggregate by the filtered shards' measured tokens, so they diverge from the
        baseline's by design once the corpora exist (the token-proportionality test owns them)."""
        baseline_raw = OmegaConf.load(STAGES[stage][1])
        assert self._corpus_weights(raw[stage]) == self._corpus_weights(baseline_raw)

    def test_the_arms_do_not_share_a_checkpoint_directory(self, merged, baseline_merged):
        """Beyond differing: no stage of one arm may write where any stage of the other does."""
        filtered_dirs = {merged[stage].checkpoint.save for stage in STAGES}
        baseline_dirs = {baseline_merged[stage].checkpoint.save for stage in STAGES}
        assert len(filtered_dirs) == len(STAGES), "two filtered stages share a save directory"
        assert not (filtered_dirs & baseline_dirs)


class TestTheBlendNamesFilteredCorpora:
    """A blend that names an unfiltered corpus produces a second baseline, silently."""

    @pytest.mark.parametrize("stage", BLEND_STAGES)
    def test_blend_is_well_formed(self, stage, raw):
        assert_blend_is_well_formed(raw[stage].dataset.data_path, stage)

    @pytest.mark.parametrize("stage", BLEND_STAGES)
    def test_every_prefix_names_a_filtered_corpus(self, stage, raw):
        for prefix in [str(x) for x in raw[stage].dataset.data_path][1::2]:
            root = PurePosixPath(prefix).parent
            if root.name.startswith("shard"):
                root = root.parent
            subset = root.name.split("__")[-1]
            assert subset.endswith(FILTERED_SUFFIX), f"{stage}: '{subset}' is not a filtered split"

    @pytest.mark.parametrize("stage", BLEND_STAGES)
    def test_prefix_roots_use_the_real_slugify(self, stage, raw):
        """The blend paths are written by hand; the roots they must match are produced by
        `pipeline_data_prepare.slugify_dataset_name`. The build derives them through
        `corpora_table.corpus_root`, a mirror kept so the plan can be derived outside the
        container — so BOTH are asserted here: the mirror against the real function, and the
        blend against the mirror."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("pipeline_data_prepare", _REPO_ROOT / "pipeline_data_prepare.py")
        prepare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prepare)
        sys.path.insert(0, str(_CAMPAIGN_DIR))
        from corpora_table import DATA_BASE, TOKENIZED_PREFIX, corpus_root

        dataset = yaml.safe_load(CORPUS_CONFIG.read_text())["dataset"]
        for prefix in [str(x) for x in raw[stage].dataset.data_path][1::2]:
            root = PurePosixPath(prefix).parent
            if root.name.startswith("shard"):
                root = root.parent
            subset = root.name.split("__")[-1]
            assert corpus_root(dataset, subset) == DATA_BASE / prepare.slugify_dataset_name(dataset, subset)
            assert str(corpus_root(dataset, subset)) == str(root), prefix
            assert prefix.endswith(f"/{TOKENIZED_PREFIX}"), prefix

    def test_sft_pack_path_names_its_tokenizer_and_pad_multiple(self, raw, merged):
        """A pack built with the truncating think tokenizer, or at a smaller pad multiple,
        loads silently and NaNs under CP=2. The path carries both, so the two cannot disagree."""
        packed = str(raw["sft"].dataset.packed_sequence_specs.packed_train_data_path)
        tokenizer = merged["sft"].tokenizer.tokenizer_model
        assert tokenizer == "geodesic-research/nemotron-think-history-tokenizer"
        assert tokenizer.replace("/", "--") in packed
        assert f"_pad_seq_to_mult{merged['sft'].dataset.packed_sequence_specs.pad_seq_to_mult}/" in packed
        assert FILTERED_SUFFIX in packed

    def test_sft_pack_is_read_as_a_shard_glob(self, raw, corpora_rows):
        """One process cannot pack the SFT mix inside the 24 h wall, so it is packed per shard
        and read by glob. The glob must span exactly the shards the build produces."""
        packed = str(raw["sft"].dataset.packed_sequence_specs.packed_train_data_path)
        assert "/shard*/" in packed, "the packed path must span the per-shard packs"
        pack_rows = [row for row in corpora_rows if row.kind == "pack"]
        assert len(pack_rows) == 1
        assert pack_rows[0].shards == 16


class TestCorporaTableAgreesWithTheBlends:
    """The table decides what gets built; the YAMLs decide what gets read. Nothing reconciles
    them at runtime — a corpus in one and not the other fails hours into a 128-node job."""

    @staticmethod
    def _blend_subsets(raw_cfg) -> set[str]:
        subsets = set()
        for prefix in [str(x) for x in raw_cfg.dataset.data_path][1::2]:
            root = PurePosixPath(prefix).parent
            if root.name.startswith("shard"):
                root = root.parent
            subsets.add(root.name.split("__")[-1])
        return subsets

    def test_every_table_subset_is_a_filtered_split(self, corpora_rows):
        for row in corpora_rows:
            assert row.subset.endswith(FILTERED_SUFFIX), row.subset

    def test_every_blended_corpus_is_in_the_build(self, corpora_rows, raw):
        built = {row.subset for row in corpora_rows}
        for stage in BLEND_STAGES:
            for subset in self._blend_subsets(raw[stage]):
                assert subset in built, f"{stage}: '{subset}' is blended but never built"

    def test_every_built_corpus_is_blended(self, corpora_rows, raw):
        """The converse: a corpus nobody trains on is node-hours spent for nothing. The SFT
        pack is the one row no `.bin/.idx` blend names — stage 3 reads it as packed parquet."""
        blended = set()
        for stage in BLEND_STAGES:
            blended |= self._blend_subsets(raw[stage])
        blended |= {row.subset for row in corpora_rows if row.kind == "pack"}
        assert {row.subset for row in corpora_rows} == blended

    def test_the_two_prepare_configs_pin_one_dataset_at_one_revision(self):
        """Both filtered corpora families are splits of the same repository. If only one config
        is updated when dataset-builder publishes, the arm trains on two different snapshots."""
        pretraining = yaml.safe_load(CORPUS_CONFIG.read_text())
        sft = yaml.safe_load(SFT_CORPUS_CONFIG.read_text())
        assert pretraining["dataset"] == sft["dataset"]
        assert pretraining["revision"] == sft["revision"]
        assert re.fullmatch(r"[0-9a-f]{40}", pretraining["revision"]), "the revision must be a full commit SHA"

    def test_the_tokenizer_that_builds_the_corpora_is_the_one_training_reads(self, merged):
        """A mismatch between the EOD baked into the .bin and `tokenizer.eod` at training time
        miscounts document boundaries silently (CLAUDE.md, "Tokenizer choice for Base CPT")."""
        tokenizer = yaml.safe_load(CORPUS_CONFIG.read_text())["tokenizer"]
        assert tokenizer == "geodesic-research/nemotron-base-tokenizer"
        for stage in BLEND_STAGES:
            assert merged[stage].tokenizer.tokenizer_model == tokenizer
        sft_tokenizer = yaml.safe_load(SFT_CORPUS_CONFIG.read_text())["tokenizer"]
        assert merged["sft"].tokenizer.tokenizer_model == sft_tokenizer


class TestTheBuildIsSubmittable:
    """The build is driven by the shared script reading this arm's table, so the table is
    exercised through the real script rather than through a re-derivation of its rules."""

    @pytest.fixture(scope="class")
    def build(self):
        return subprocess.run(
            ["bash", str(BUILD_SCRIPT), str(CORPORA_TABLE), "all"],
            cwd=str(_REPO_ROOT),
            env=dict(os.environ, DRY_RUN="1"),
            capture_output=True,
            text=True,
            timeout=120,
        )

    @staticmethod
    def _pending(corpora_rows) -> list[str]:
        return [row.subset for row in corpora_rows if row.docs is None]

    def test_build_refuses_while_document_counts_are_unknown(self, build, corpora_rows):
        """ClimbMix is source-sliced, and its eight index ranges cannot be computed without the
        exact retained document count. The script must refuse rather than guess one."""
        if not self._pending(corpora_rows):
            pytest.skip("every corpus has its document count; the refusal no longer applies")
        assert build.returncode != 0, "the build must not proceed with a PENDING slice count"
        assert "slice ranges need the document count, table says PENDING" in build.stderr

    def test_dry_run_submits_the_expected_jobs(self, build, corpora_rows):
        """15 prepare+tokenize pairs (ClimbMix's eight sliced), plus the SFT corpus's prepare,
        byte-gated split and 16 per-shard packs."""
        if self._pending(corpora_rows):
            pytest.skip("document counts are PENDING; the build cannot be planned yet")
        output = build.stdout + build.stderr
        assert build.returncode == 0, output
        assert "SUBMITTED 62 jobs" in output
        assert "nothing was actually submitted" in output
        built = set(re.findall(r"^=== (\S+) \(", output, re.MULTILINE))
        assert built == {row.subset for row in corpora_rows}

    def test_climbmix_alone_is_submittable_from_the_arm_table(self, corpora_rows):
        """ClimbMix is the corpus most likely to be held back (its build peaks ~4 TB on disk), so
        the README's ClimbMix-only command must plan exactly its 16 jobs from this table, under
        this arm's job names — not from a copied table, whose directory would rename them."""
        if self._pending(corpora_rows):
            pytest.skip("document counts are PENDING; the build cannot be planned yet")
        (climbmix,) = [row for row in corpora_rows if row.shard_mode == "slice"]
        result = subprocess.run(
            ["bash", str(BUILD_SCRIPT), str(CORPORA_TABLE), "pretraining", climbmix.subset],
            cwd=str(_REPO_ROOT),
            env=dict(os.environ, DRY_RUN="1"),
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "SUBMITTED 16 jobs" in output
        assert set(re.findall(r"^=== (\S+) \(", output, re.MULTILINE)) == {climbmix.subset}
        for index in range(climbmix.shards):
            assert f"cp-{CORPORA_TABLE.parent.name}-prep-{climbmix.subset}-s{index}" in output
            assert f"cp-{CORPORA_TABLE.parent.name}-tok-{climbmix.subset}-s{index}" in output

    def test_climbmix_slices_cover_the_corpus_exactly_once(self, build, corpora_rows):
        """A gap between ranges drops documents silently; an overlap trains some of them twice."""
        climbmix = [row for row in corpora_rows if row.shard_mode == "slice"]
        assert len(climbmix) == 1, "ClimbMix is the arm's only source-sliced corpus"
        if climbmix[0].docs is None:
            pytest.skip("document counts are PENDING; the slice ranges cannot be computed yet")
        ranges = climbmix[0].slice_ranges()
        assert len(ranges) == climbmix[0].shards
        assert ranges[0][0] == 0
        assert ranges[-1][1] == climbmix[0].docs
        for (_, previous_end), (beginning, _) in zip(ranges, ranges[1:]):
            assert beginning == previous_end
        for beginning, end in ranges:
            assert f"--split train[{beginning}:{end}]" in build.stdout + build.stderr


class TestTheSftPackBuild:
    """The SFT corpus is the arm's only split-and-pack row, and it sits behind the PENDING skip
    above — so its path through the real script is exercised here on a one-row table that names
    the real prepare config. This is the path that packs ~5.7M conversations inside the wall."""

    @pytest.fixture(scope="class")
    def dry_run(self, tmp_path_factory):
        arm_dir = tmp_path_factory.mktemp("30b_filtered_mini_2plus")
        table = arm_dir / "corpora.tsv"
        row = [
            f"pa_warm_start_sft{FILTERED_SUFFIX}",
            "sft",
            "pack",
            str(SFT_CORPUS_CONFIG.relative_to(_REPO_ROOT)),
            "08",
            "04",
            "1",
            "16",
            "split",
            "1",
            "5702903",
        ]
        table.write_text("|".join(row) + "\n")
        proc = subprocess.run(
            ["bash", str(BUILD_SCRIPT), str(table), "sft"],
            cwd=str(_REPO_ROOT),
            env=dict(os.environ, DRY_RUN="1"),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        return proc.stdout + proc.stderr

    @staticmethod
    def _submissions(dry_run) -> list[tuple[str, str]]:
        return re.findall(r"\[dry-run\] (\S+)[^:]*: (.+)", dry_run)

    def test_prepare_then_split_then_one_pack_per_shard(self, dry_run):
        assert "SUBMITTED 18 jobs" in dry_run
        assert [kind for kind, _ in self._submissions(dry_run)] == ["prepare", "split"] + ["pack"] * 16

    def test_every_consumer_waits_on_its_producer(self, dry_run):
        for kind, command in self._submissions(dry_run):
            if kind == "prepare":
                assert "afterok" not in command and "--hold" not in command, command
            else:
                assert "--hold" in command, f"{kind} must wait on the step that produces its input: {command}"

    def test_split_runs_the_shard_script_as_its_own_job(self, dry_run):
        """The split must be the BATCH SCRIPT of its submission, not an argument to the data
        pipeline's wrapper: that wrapper reads its first argument as a mode or a dataset root,
        so a split handed to it would run as a pack against the script's own path."""
        (split,) = [c for k, c in self._submissions(dry_run) if k == "split"]
        assert "pipeline_data_submit.sbatch" not in split
        assert re.search(r" \S+/shard_jsonl_corpus\.sh \S+ 16$", split), split
        assert "--output=logs/slurm/" in split

    def test_every_other_job_goes_through_the_data_pipeline_wrapper(self, dry_run):
        for kind, command in self._submissions(dry_run):
            if kind != "split":
                assert " pipeline_data_submit.sbatch " in command, command

    def test_packs_take_their_geometry_from_the_prepare_config(self, dry_run):
        scalars = yaml.safe_load(SFT_CORPUS_CONFIG.read_text())
        packs = [c for k, c in self._submissions(dry_run) if k == "pack"]
        assert len(packs) == 16
        for command in packs:
            assert command.endswith(f" {scalars['tokenizer']} {scalars['seq-length']} {scalars['pad-seq-to-mult']}")
            assert re.search(r"/shard\d+ ", command), command

    def test_what_the_prepare_produces_is_stated_by_its_config_not_by_flags(self, dry_run):
        """The prepare must write JSONL only; that is corpus identity and lives in the config,
        where an explicit flag would silently override it."""
        (prepare,) = [c for k, c in self._submissions(dry_run) if k == "prepare"]
        assert "--skip-pack" not in prepare and "--skip-count" not in prepare
        scalars = yaml.safe_load(SFT_CORPUS_CONFIG.read_text())
        assert scalars["skip-pack"] is True and scalars["skip-count"] is True


class TestStagePostures:
    """The per-stage settings the diff cannot vouch for, because both arms would share a bug."""

    @pytest.mark.parametrize("stage", ["midtrain", "sft"])
    def test_seq_length_is_stated_in_both_places_and_agrees(self, stage, merged, raw):
        """`pipeline_training_run.py` silently defaults `dataset.seq_length` to 8192 when it is
        absent, and `model.seq_length` is a separate key nothing cross-checks."""
        assert raw[stage].dataset.seq_length == 32768
        assert merged[stage].dataset.seq_length == 32768
        assert merged[stage].model.seq_length == 32768

    @pytest.mark.parametrize("stage", list(STAGES))
    def test_segment_rollover(self, stage, merged):
        assert_segment_exit_posture(merged[stage], f"filtered {stage}", 1400)

    @pytest.mark.parametrize("stage", list(STAGES))
    def test_saves_survive_the_dp512_crossing(self, stage, merged):
        """`ckpt_assume_constant_structure` sends the second save down a cached path that keeps
        a 13.679 GiB expert-weight copy, and the next forward OOMs. The recipe sets it True, so
        omitting it is not the same as setting it."""
        assert merged[stage].checkpoint.ckpt_assume_constant_structure is False
        assert merged[stage].model.cross_entropy_loss_fusion is False

    def test_warm_starts_chain_within_this_arm(self, merged):
        """Each stage warm-starts from the previous stage of the SAME arm: a cross-arm load
        would train filtered data on baseline weights and silently void the comparison."""
        assert merged["pretrain"].checkpoint.pretrained_checkpoint is None
        assert merged["midtrain"].checkpoint.pretrained_checkpoint == merged["pretrain"].checkpoint.save
        assert merged["sft"].checkpoint.pretrained_checkpoint == merged["midtrain"].checkpoint.save

    def test_pretrain_climbmix_shard_weights_are_token_proportional(self, raw):
        """The eight shards are cut at equal DOCUMENT counts, so their token counts differ and
        equal weights would over-sample the smaller ones. Skips until the corpora are built."""
        assert_shard_weights_are_token_proportional(
            raw["pretrain"].dataset.data_path, f"climbmix_full{FILTERED_SUFFIX}", 0.698180
        )

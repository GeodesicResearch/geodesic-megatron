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

"""The bucket mirror must never archive a half-written checkpoint, an HF export, or a JSONL.

`scripts/hub/sync_bucket.py` decides what goes into the campaign's Hub bucket from a manifest of
stage configs: each config's `checkpoint.save` directory and the corpora its data paths name. The
rules that make it safe beside live training are what these tests pin: a save in progress is
invisible until the tracker passes it, the root files follow the shards they point at and are
deferred when the tracker moves mid-pass, the `hf/` export is excluded from every checkpoint, a
corpus contributes only its tokenized triple and prepare record and lands at its name whichever
data base built it, a stage that has not started is reported rather than failed, and a pass that
leaves files pending is a failure.

The checkpoint and corpus directories are real directories built in `tmp_path`, with the real
file names the training pipeline writes, and the real functions read them. The one stand-in is
the Hub client: the real `HfApi.sync_bucket` uploads to Hugging Face, so a recorder takes its
place and the test asserts the arguments the tool hands it.
"""

from __future__ import annotations

import importlib
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

from pipeline import checkpoints
from pipeline.export.manifest import EXPORT_COMPLETE_FILE
from tests.unit_tests.bucket_listing import BucketItem
from tests.unit_tests.checkpoint_dirs import make_checkpoint_dir
from tests.unit_tests.corpora_fixtures import importable
from tests.unit_tests.filesystem_races import strike_at


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOL_DIR = _REPO_ROOT / "scripts" / "hub"
importable(_TOOL_DIR)
sync_bucket = importlib.import_module("sync_bucket")
selected_paths = importlib.import_module("scripts.file_digest").selected_paths

CAMPAIGN_MANIFEST = _REPO_ROOT / "configs" / "control_pretraining" / "bucket_sync.yaml"


def make_built_corpus(corpus: Path) -> Path:
    """A tokenized corpus directory: the .bin/.idx the tokenize job writes, under the default prefix."""
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "tokenized_base_input_document.bin").write_bytes(b"t" * 8)
    (corpus / "tokenized_base_input_document.idx").write_bytes(b"i" * 8)
    return corpus


def write_training_config(
    path: Path, data_path: list | None = None, packed: str | list[str] | None = None, save: str | None = None
) -> Path:
    """A stage config with the sections the mirror reads: ``dataset`` and ``checkpoint.save``."""
    dataset: dict = {"seq_length": 8192}
    if data_path is not None:
        dataset["data_path"] = data_path
    if packed is not None:
        dataset["packed_sequence_specs"] = {"packed_sequence_size": 32768, "packed_train_data_path": packed}
    cfg: dict = {"dataset": dataset}
    if save is not None:
        cfg["checkpoint"] = {"save": save, "load": save}
    path.write_text(yaml.safe_dump(cfg))
    return path


def manifest_for(tmp_path: Path, **overrides) -> "sync_bucket.Manifest":
    """A manifest object with the campaign's prefixes and everything else supplied by the test."""
    fields = dict(
        bucket="org/bucket",
        readme=tmp_path / "README.md",
        provenance_prefix="_provenance",
        checkpoints_prefix="checkpoints",
        datasets_prefix="datasets",
        log_dir=tmp_path / "logs",
        hf_home=tmp_path / "hf",
        stage_configs=(),
        extra_checkpoints=(),
    )
    fields.update(overrides)
    return sync_bucket.Manifest(**fields)


def manifest_yaml(tmp_path: Path, **overrides) -> dict:
    """The manifest's YAML form with every key present, for the loader tests."""
    raw = {
        "bucket": "org/bucket",
        "readme": "README.md",
        "provenance_prefix": "_provenance",
        "checkpoints_prefix": "checkpoints",
        "datasets_prefix": "datasets",
        "log_dir": str(tmp_path / "logs"),
        "hf_home": str(tmp_path / "hf"),
        "stage_configs": [],
        "extra_checkpoints": [],
    }
    raw.update(overrides)
    return raw


class TestCompletedIterations:
    def test_a_save_in_progress_is_not_completed(self, tmp_path):
        root = make_checkpoint_dir(tmp_path / "ckpt", [100, 200, 300], tracker=200)
        assert [p.name for p in checkpoints.completed_iterations(root)] == ["iter_0000100", "iter_0000200"]

    def test_no_tracker_means_nothing_has_completed(self, tmp_path):
        root = make_checkpoint_dir(tmp_path / "ckpt", [100], tracker=None)
        assert checkpoints.completed_iterations(root) == []

    def test_a_missing_directory_is_an_error_not_an_empty_stage(self, tmp_path):
        with pytest.raises(checkpoints.CheckpointDirectoryMissing, match="does not exist"):
            checkpoints.completed_iterations(tmp_path / "absent")

    @pytest.mark.parametrize("read", ["read_text", "iterdir"])
    def test_a_directory_removed_while_it_is_read_is_reported_as_missing(self, tmp_path, monkeypatch, read):
        """A run's saves are deleted while passes run, so the directory can be there when the read
        starts and gone before the tracker is read, or before the directory is listed. Either way
        the caller is told the directory is missing, as it is for one already gone, and is not
        handed whichever read's own error the deletion happened to beat."""
        root = make_checkpoint_dir(tmp_path / "ckpt", [100], tracker=100)
        struck = root / checkpoints.LATEST_FILE if read == "read_text" else root
        strike_at(monkeypatch, read, struck, lambda: shutil.rmtree(root), after=False)
        with pytest.raises(checkpoints.CheckpointDirectoryMissing, match="does not exist"):
            checkpoints.completed_iterations(root)
        assert not root.exists(), "the directory was removed during the read"

    def test_iterations_come_back_oldest_first_whatever_the_listing_order(self, tmp_path):
        root = make_checkpoint_dir(tmp_path / "ckpt", [9056, 2264, 11320, 4528], tracker=11320)
        assert [p.name for p in checkpoints.completed_iterations(root)] == [
            "iter_0002264",
            "iter_0004528",
            "iter_0009056",
            "iter_0011320",
        ]


class TestCheckpointUnits:
    def test_every_iteration_excludes_its_hf_export_and_keeps_the_rest(self, tmp_path):
        root = make_checkpoint_dir(
            tmp_path / "ckpt", [100, 200], tracker=200, hf_files={"model.safetensors": b"y" * 8}
        )
        entry = sync_bucket.CheckpointEntry(local=root, remote="checkpoints/ckpt")
        units = sync_bucket.checkpoint_units(entry, checkpoints.completed_iterations(root))
        assert [u.remote for u in units] == ["checkpoints/ckpt/iter_0000100", "checkpoints/ckpt/iter_0000200"]
        assert all(u.exclude == (sync_bucket.HF_EXPORT_EXCLUDE,) and u.include is None for u in units)
        assert all(u.source == root / u.remote.rsplit("/", 1)[1] for u in units)

    def test_root_files_are_snapshotted_and_go_under_the_entry_prefix(self, tmp_path):
        root = make_checkpoint_dir(tmp_path / "ckpt", [100, 200], tracker=200)
        entry = sync_bucket.CheckpointEntry(local=root, remote="checkpoints/ckpt")
        unit = sync_bucket.stage_root_files(entry, root / "iter_0000200", tmp_path / "staging")
        assert unit is not None
        assert unit.remote == "checkpoints/ckpt"
        assert unit.source == tmp_path / "staging"
        assert unit.include == sync_bucket.CHECKPOINT_ROOT_FILES
        assert (tmp_path / "staging" / checkpoints.LATEST_FILE).read_text().strip() == "200"
        assert (tmp_path / "staging" / "progress.txt").exists()

    def test_root_files_are_deferred_when_the_tracker_advanced_past_the_archived_iteration(self, tmp_path):
        root = make_checkpoint_dir(tmp_path / "ckpt", [100, 200, 300], tracker=300)
        entry = sync_bucket.CheckpointEntry(local=root, remote="checkpoints/ckpt")
        # The pass enumerated up to 200; by the time root files are staged the tracker says 300.
        assert sync_bucket.stage_root_files(entry, root / "iter_0000200", tmp_path / "staging") is None

    def test_plan_puts_root_files_after_the_iterations_they_point_at(self, tmp_path):
        root = make_checkpoint_dir(tmp_path / "ckpt", [100, 200], tracker=200)
        manifest = manifest_for(
            tmp_path, extra_checkpoints=(sync_bucket.CheckpointEntry(local=root, remote="checkpoints/ckpt"),)
        )
        units = sync_bucket.plan_units(manifest, tmp_path / "staging")
        assert [u.remote for u in units] == [
            "checkpoints/ckpt/iter_0000100",
            "checkpoints/ckpt/iter_0000200",
            "checkpoints/ckpt",
        ]

    def test_the_same_iteration_from_two_directories_is_refused(self, tmp_path):
        run_dir = make_checkpoint_dir(tmp_path / "run", [600, 2988], tracker=2988)
        clone = make_checkpoint_dir(tmp_path / "clone", [600], tracker=600)
        manifest = manifest_for(
            tmp_path,
            extra_checkpoints=(
                sync_bucket.CheckpointEntry(local=run_dir, remote="checkpoints/sft"),
                sync_bucket.CheckpointEntry(local=clone, remote="checkpoints/sft"),
            ),
        )
        with pytest.raises(sync_bucket.ManifestError, match="two different directories"):
            sync_bucket.plan_units(manifest, tmp_path / "staging")

    def test_a_clone_supplying_a_missing_iteration_lands_beside_the_run(self, tmp_path):
        run_dir = make_checkpoint_dir(tmp_path / "run", [2400, 2988], tracker=2988)
        clone = make_checkpoint_dir(tmp_path / "clone", [600], tracker=600)
        manifest = manifest_for(
            tmp_path,
            extra_checkpoints=(
                sync_bucket.CheckpointEntry(local=run_dir, remote="checkpoints/sft"),
                sync_bucket.CheckpointEntry(local=clone, remote="checkpoints/sft"),
            ),
        )
        remotes = [u.remote for u in sync_bucket.plan_units(manifest, tmp_path / "staging")]
        assert "checkpoints/sft/iter_0000600" in remotes and "checkpoints/sft/iter_0002400" in remotes
        # Both entries feed one prefix; the root files come from the run, which holds the newest save.
        assert remotes.count("checkpoints/sft") == 1

    def test_an_explicit_extra_checkpoint_directory_must_exist(self, tmp_path):
        manifest = manifest_for(
            tmp_path,
            extra_checkpoints=(sync_bucket.CheckpointEntry(local=tmp_path / "absent", remote="checkpoints/x"),),
        )
        with pytest.raises(sync_bucket.ManifestError, match="does not exist"):
            sync_bucket.plan_units(manifest, tmp_path / "staging")


class TestStageConfigs:
    def test_a_stage_config_contributes_its_save_directory_and_its_corpora(self, tmp_path):
        corpus = tmp_path / "data" / "org__repo__zyda_full"
        make_built_corpus(corpus)
        root = make_checkpoint_dir(tmp_path / "ckpts" / "stage_pretrain", [100], tracker=100)
        config = write_training_config(
            tmp_path / "pretrain.yaml",
            data_path=["1.0", str(corpus / "tokenized_base_input_document")],
            save=str(root),
        )
        manifest = manifest_for(tmp_path, stage_configs=(config,))
        remotes = [u.remote for u in sync_bucket.plan_units(manifest, tmp_path / "staging")]
        assert remotes == [
            "checkpoints/stage_pretrain/iter_0000100",
            "checkpoints/stage_pretrain",
            "datasets/org__repo__zyda_full",
        ]

    def test_a_stage_that_has_not_started_is_reported_and_its_data_still_archived(self, tmp_path, caplog):
        corpus = tmp_path / "data" / "org__repo__zyda_full"
        make_built_corpus(corpus)
        config = write_training_config(
            tmp_path / "midtrain.yaml",
            data_path=["1.0", str(corpus / "tokenized_base_input_document")],
            save=str(tmp_path / "ckpts" / "not_started_yet"),
        )
        manifest = manifest_for(tmp_path, stage_configs=(config,))
        with caplog.at_level(logging.WARNING, logger="sync_bucket"):
            remotes = [u.remote for u in sync_bucket.plan_units(manifest, tmp_path / "staging")]
        assert remotes == ["datasets/org__repo__zyda_full"]
        assert "not_started_yet" in caplog.text and "stage not started" in caplog.text

    def test_an_unstarted_stage_whose_corpora_are_not_built_is_reported_and_skipped(self, tmp_path, caplog):
        """A stage is listed before it trains, and before its corpora are built. A sync unit for a
        corpus directory that does not exist makes the Hub client raise mid-pass, aborting every
        other arm's upload with it, so an unbuilt corpus of an unstarted stage is reported and
        left for a later pass."""
        config = write_training_config(
            tmp_path / "midtrain.yaml",
            data_path=["1.0", str(tmp_path / "data" / "org__repo__not_built" / "tokenized_base_input_document")],
            save=str(tmp_path / "ckpts" / "not_started_yet"),
        )
        manifest = manifest_for(tmp_path, stage_configs=(config,))
        with caplog.at_level(logging.WARNING, logger="sync_bucket"):
            units = sync_bucket.plan_units(manifest, tmp_path / "staging")
        assert units == []
        assert "org__repo__not_built" in caplog.text and "not built yet" in caplog.text

    def test_an_unstarted_stage_whose_packs_do_not_exist_yet_is_reported_and_skipped(self, tmp_path, caplog):
        config = write_training_config(
            tmp_path / "sft.yaml",
            packed=str(tmp_path / "data" / "org__mix" / "shard*" / "packed" / "tok" / "training_32768.idx.parquet"),
            save=str(tmp_path / "ckpts" / "sft_not_started"),
        )
        manifest = manifest_for(tmp_path, stage_configs=(config,))
        with caplog.at_level(logging.WARNING, logger="sync_bucket"):
            units = sync_bucket.plan_units(manifest, tmp_path / "staging")
        assert units == []
        assert "not built yet" in caplog.text

    def test_an_unstarted_stage_whose_corpus_is_still_being_built_is_skipped(self, tmp_path, caplog):
        """The prepare job creates the corpus directory hours before the tokenize job writes the
        token files into it, so a directory is not a built corpus: a pass during the build would
        otherwise upload a half-written .bin, or a corpus with no tokens at all."""
        corpus = tmp_path / "data" / "org__repo__building"
        corpus.mkdir(parents=True)
        (corpus / "pipeline_results.json").write_text("{}")
        config = write_training_config(
            tmp_path / "midtrain.yaml",
            data_path=["1.0", str(corpus / "tokenized_base_input_document")],
            save=str(tmp_path / "ckpts" / "not_started_yet"),
        )
        manifest = manifest_for(tmp_path, stage_configs=(config,))
        with caplog.at_level(logging.WARNING, logger="sync_bucket"):
            units = sync_bucket.plan_units(manifest, tmp_path / "staging")
        assert units == []
        assert "org__repo__building" in caplog.text and "not built yet" in caplog.text

    def test_a_started_stage_whose_token_files_are_gone_is_an_error(self, tmp_path):
        corpus = tmp_path / "data" / "org__repo__emptied"
        corpus.mkdir(parents=True)
        root = make_checkpoint_dir(tmp_path / "ckpts" / "stage_midtrain", [100], tracker=100)
        config = write_training_config(
            tmp_path / "midtrain.yaml",
            data_path=["1.0", str(corpus / "tokenized_base_input_document")],
            save=str(root),
        )
        manifest = manifest_for(tmp_path, stage_configs=(config,))
        with pytest.raises(sync_bucket.ManifestError, match="has started but its corpus .* does not exist"):
            sync_bucket.plan_units(manifest, tmp_path / "staging")

    def test_a_started_stage_whose_corpus_is_missing_is_an_error(self, tmp_path):
        """Once a stage has trained, its corpora existed; one missing now was moved or deleted, and
        the archive must say so rather than quietly archive the stage without its data."""
        root = make_checkpoint_dir(tmp_path / "ckpts" / "stage_midtrain", [100], tracker=100)
        config = write_training_config(
            tmp_path / "midtrain.yaml",
            data_path=["1.0", str(tmp_path / "data" / "org__repo__gone" / "tokenized_base_input_document")],
            save=str(root),
        )
        manifest = manifest_for(tmp_path, stage_configs=(config,))
        with pytest.raises(sync_bucket.ManifestError, match="has started but its corpus .* does not exist"):
            sync_bucket.plan_units(manifest, tmp_path / "staging")

    def test_a_stage_config_without_an_absolute_save_path_is_refused(self, tmp_path):
        config = write_training_config(tmp_path / "c.yaml", data_path=["1.0", "/x/tokenized_base_input_document"])
        with pytest.raises(sync_bucket.ManifestError, match="checkpoint.save must be an absolute path"):
            sync_bucket.stage_checkpoint_entry(config, "checkpoints")

    def test_the_bucket_prefix_is_the_save_directory_name(self, tmp_path):
        config = write_training_config(tmp_path / "c.yaml", save="/somewhere/deep/control_pretrain_30b_baseline_sft")
        entry = sync_bucket.stage_checkpoint_entry(config, "checkpoints")
        assert entry.local == Path("/somewhere/deep/control_pretrain_30b_baseline_sft")
        assert entry.remote == "checkpoints/control_pretrain_30b_baseline_sft"


class TestCorpusRelative:
    """A corpus lands at its own name whichever data base built it, read from the pipeline's layout."""

    def test_a_plain_corpus_root(self):
        assert sync_bucket.corpus_relative(Path("/base/a/org__repo__zyda_full")) == Path("org__repo__zyda_full")

    def test_a_shard_of_a_sliced_corpus(self):
        assert sync_bucket.corpus_relative(Path("/base/org__repo__climbmix_full/shard3")) == Path(
            "org__repo__climbmix_full/shard3"
        )

    def test_a_packed_directory_under_the_root(self):
        assert sync_bucket.corpus_relative(Path("/other/base/org__mix/packed/tok_pad_seq_to_mult4")) == Path(
            "org__mix/packed/tok_pad_seq_to_mult4"
        )

    def test_a_packed_directory_under_a_shard(self):
        assert sync_bucket.corpus_relative(Path("/base/org__mix/shard10/packed/tok")) == Path(
            "org__mix/shard10/packed/tok"
        )

    def test_the_two_data_bases_of_the_campaign_give_the_same_bucket_path(self):
        a = sync_bucket.corpus_relative(Path("/projects/a5k/public/data/org__mix/packed/tok"))
        b = sync_bucket.corpus_relative(Path("/projects/a5k/public/other_base/data/org__mix/packed/tok"))
        assert a == b == Path("org__mix/packed/tok")

    def test_a_directory_with_no_root_above_it_is_refused(self):
        with pytest.raises(sync_bucket.ManifestError, match="no corpus root"):
            sync_bucket.corpus_relative(Path("/shard0"))


class TestDatasetUnits:
    def test_each_prefix_contributes_its_tokenized_triple_and_prepare_record_only(self, tmp_path):
        corpus = tmp_path / "data" / "org__repo__zyda_full"
        make_built_corpus(corpus)
        prefix = str(corpus / "tokenized_base_input_document")
        config = write_training_config(tmp_path / "pretrain.yaml", data_path=["0.5", prefix, "0.5", prefix])
        units = sync_bucket.dataset_units(config, "datasets", stage_started=True)
        assert len(units) == 2  # dedupe happens at plan level; both prefixes are the same corpus here
        unit = units[0]
        assert unit.source == corpus
        assert unit.remote == "datasets/org__repo__zyda_full"
        assert unit.include == (
            "tokenized_base_input_document.bin",
            "tokenized_base_input_document.idx",
            "tokenized_base_input_document.provenance.json",
            "pipeline_results.json",
        )
        assert sync_bucket.dedupe(units) == [unit]

    def test_a_sharded_corpus_keeps_its_shard_directories(self, tmp_path):
        shard = tmp_path / "data" / "org__repo__climbmix_full" / "shard3"
        make_built_corpus(shard)
        config = write_training_config(
            tmp_path / "c.yaml", data_path=["1.0", str(shard / "tokenized_base_input_document")]
        )
        [unit] = sync_bucket.dataset_units(config, "datasets", stage_started=True)
        assert unit.remote == "datasets/org__repo__climbmix_full/shard3"

    def test_packed_sft_shards_resolve_from_the_glob(self, tmp_path):
        base = tmp_path / "data"
        packed_dirs = []
        for shard in range(2):
            packed = base / "org__mix" / f"shard{shard}" / "packed" / "tok_pad_seq_to_mult4"
            packed.mkdir(parents=True)
            (packed / "training_32768.idx.parquet").write_bytes(b"p")
            packed_dirs.append(packed)
        glob = base / "org__mix" / "shard*" / "packed" / "tok_pad_seq_to_mult4" / "training_32768.idx.parquet"
        config = write_training_config(tmp_path / "sft.yaml", packed=str(glob))
        units = sync_bucket.dataset_units(config, "datasets", stage_started=True)
        assert [u.source for u in units] == packed_dirs
        assert [u.remote for u in units] == [
            "datasets/org__mix/shard0/packed/tok_pad_seq_to_mult4",
            "datasets/org__mix/shard1/packed/tok_pad_seq_to_mult4",
        ]
        assert all(u.include is None and u.exclude == () for u in units)

    def test_a_list_of_packed_globs_contributes_every_match(self, tmp_path):
        roots = []
        for name in ("docs", "replay"):
            packed = tmp_path / "data" / f"org__{name}" / "packed" / "tok_pad_seq_to_mult1"
            packed.mkdir(parents=True)
            (packed / "training_4096.idx.parquet").write_bytes(b"p")
            roots.append(packed)
        config = write_training_config(
            tmp_path / "sft.yaml", packed=[str(root / "training_4096.idx.parquet") for root in roots]
        )
        units = sync_bucket.dataset_units(config, "datasets", stage_started=True)
        assert [u.source for u in units] == roots

    def test_a_list_entry_that_matches_nothing_is_an_error(self, tmp_path):
        packed = tmp_path / "data" / "org__docs" / "packed" / "tok"
        packed.mkdir(parents=True)
        (packed / "training_4096.idx.parquet").write_bytes(b"p")
        missing = str(tmp_path / "data" / "nowhere" / "*.parquet")
        config = write_training_config(
            tmp_path / "sft.yaml", packed=[str(packed / "training_4096.idx.parquet"), missing]
        )
        with pytest.raises(sync_bucket.ManifestError, match="matches nothing"):
            sync_bucket.dataset_units(config, "datasets", stage_started=True)

    def test_a_glob_that_matches_nothing_is_an_error(self, tmp_path):
        config = write_training_config(tmp_path / "sft.yaml", packed=str(tmp_path / "data" / "nowhere" / "*.parquet"))
        with pytest.raises(sync_bucket.ManifestError, match="matches nothing"):
            sync_bucket.dataset_units(config, "datasets", stage_started=True)

    def test_a_config_without_data_is_an_error(self, tmp_path):
        config = write_training_config(tmp_path / "c.yaml")
        with pytest.raises(sync_bucket.ManifestError, match="neither"):
            sync_bucket.dataset_units(config, "datasets", stage_started=True)

    def test_blend_pairs_reads_the_flat_list_and_refuses_a_malformed_one(self, tmp_path):
        """The blend is Megatron's flat list; every reader of it (the archive's corpus units, the
        model cards' data mix) goes through this one parse so none can read a broken list as a
        shorter blend."""
        config = tmp_path / "c.yaml"
        pairs = sync_bucket.blend_pairs(["0.75", "/data/a/doc", 0.25, "/data/b/doc"], config)
        assert pairs == [(0.75, Path("/data/a/doc")), (0.25, Path("/data/b/doc"))]
        with pytest.raises(sync_bucket.ManifestError, match="must pair up"):
            sync_bucket.blend_pairs(["0.75", "/data/a/doc", "0.25"], config)
        with pytest.raises(sync_bucket.ManifestError, match="is not a number"):
            sync_bucket.blend_pairs(["heavy", "/data/a/doc"], config)
        with pytest.raises(sync_bucket.ManifestError, match="is not an absolute path"):
            sync_bucket.blend_pairs(["1.0", "data/a/doc"], config)
        with pytest.raises(sync_bucket.ManifestError, match="flat weight/prefix list"):
            sync_bucket.blend_pairs([], config)


class TestManifest:
    def test_the_campaign_manifest_archives_every_stage_distinctly_and_one_explicit_clone(self):
        """Every listed stage maps to its own archive directory, so no two stages overwrite each
        other in the bucket. The count is exact, so dropping a stage from the manifest (and
        silently no longer archiving it) fails here, and each arm added to the manifest updates it."""
        manifest = sync_bucket.load_manifest(CAMPAIGN_MANIFEST, _REPO_ROOT)
        assert manifest.bucket == "geodesic-research/control-pretraining-models-bucket"
        assert manifest.readme.is_file()
        # Thirteen stages of the three-stage, two-stage and single-stage arms, and one entry per
        # knowledge-reintroduction arm: its final link, whose save directory holds every link's
        # checkpoints.
        assert len(manifest.stage_configs) == 19 and all(c.is_file() for c in manifest.stage_configs)
        entries = [sync_bucket.stage_checkpoint_entry(c, manifest.checkpoints_prefix) for c in manifest.stage_configs]
        assert len({e.remote for e in entries}) == len(manifest.stage_configs)
        assert all(e.remote == f"checkpoints/{e.local.name}" for e in entries)
        # The one directory no config names: the export clone holding the baseline SFT's pruned
        # iteration-600 save, beside the SFT run's own directory and archived under the run's name.
        [clone] = manifest.extra_checkpoints
        sft = next(e for e in entries if e.local.name == "control_pretrain_30b_baseline_sft")
        assert clone.local == sft.local.parent / "sft600_export_clone"
        assert clone.remote == sft.remote
        assert str(manifest.log_dir).startswith("/projects/a5k/public/")
        assert str(manifest.hf_home).startswith("/projects/a5k/public/")

    def test_unknown_or_missing_keys_are_refused(self, tmp_path):
        (tmp_path / "README.md").write_text("# archive\n")
        good = manifest_yaml(tmp_path)
        path = tmp_path / "m.yaml"
        path.write_text(yaml.safe_dump({**good, "extra": 1}))
        with pytest.raises(sync_bucket.ManifestError, match="unknown keys \\['extra'\\]"):
            sync_bucket.load_manifest(path, tmp_path)
        path.write_text(yaml.safe_dump({k: v for k, v in good.items() if k != "bucket"}))
        with pytest.raises(sync_bucket.ManifestError, match="missing keys \\['bucket'\\]"):
            sync_bucket.load_manifest(path, tmp_path)
        path.write_text(yaml.safe_dump({**good, "bucket": "hf://buckets/org/bucket"}))
        with pytest.raises(sync_bucket.ManifestError, match="<namespace>/<name>"):
            sync_bucket.load_manifest(path, tmp_path)

    def test_an_extra_checkpoint_resolves_beside_the_named_stages_save_directory(self, tmp_path):
        (tmp_path / "README.md").write_text("# archive\n")
        write_training_config(tmp_path / "stage.yaml", save=str(tmp_path / "ckpts" / "run"))
        path = tmp_path / "m.yaml"
        extra = [{"beside": "stage.yaml", "directory": "run_clone", "remote": "checkpoints/run"}]
        path.write_text(yaml.safe_dump(manifest_yaml(tmp_path, extra_checkpoints=extra)))
        [clone] = sync_bucket.load_manifest(path, tmp_path).extra_checkpoints
        assert clone.local == tmp_path / "ckpts" / "run_clone"
        assert clone.remote == "checkpoints/run"

    def test_an_extra_checkpoint_entry_needs_exactly_beside_directory_and_remote(self, tmp_path):
        (tmp_path / "README.md").write_text("# archive\n")
        path = tmp_path / "m.yaml"
        path.write_text(yaml.safe_dump(manifest_yaml(tmp_path, extra_checkpoints=[{"local": "/x", "remote": "r"}])))
        with pytest.raises(sync_bucket.ManifestError, match="exactly the keys"):
            sync_bucket.load_manifest(path, tmp_path)
        write_training_config(tmp_path / "stage.yaml", save=str(tmp_path / "ckpts" / "run"))
        nested = [{"beside": "stage.yaml", "directory": "a/b", "remote": "checkpoints/run"}]
        path.write_text(yaml.safe_dump(manifest_yaml(tmp_path, extra_checkpoints=nested)))
        with pytest.raises(sync_bucket.ManifestError, match="bare directory name"):
            sync_bucket.load_manifest(path, tmp_path)

    def test_a_stage_config_that_does_not_exist_is_refused(self, tmp_path):
        (tmp_path / "README.md").write_text("# archive\n")
        path = tmp_path / "m.yaml"
        path.write_text(yaml.safe_dump(manifest_yaml(tmp_path, stage_configs=["configs/missing.yaml"])))
        with pytest.raises(sync_bucket.ManifestError, match="stage_configs that do not exist"):
            sync_bucket.load_manifest(path, tmp_path)


class TestHfHome:
    """The Hub client's upload cache must never land under the home directory."""

    def test_the_manifests_hf_home_applies_when_nothing_is_exported(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_HOME", raising=False)
        assert sync_bucket.configure_hf_home(tmp_path / "hf") == (tmp_path / "hf").resolve()
        assert os.environ["HF_HOME"] == str(tmp_path / "hf")

    def test_an_exported_hf_home_wins_over_the_manifest(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_HOME", str(tmp_path / "exported"))
        assert sync_bucket.configure_hf_home(tmp_path / "hf") == (tmp_path / "exported").resolve()

    def test_a_cache_under_the_home_directory_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        with pytest.raises(RuntimeError, match="under the home directory"):
            sync_bucket.configure_hf_home(tmp_path / "hf")
        monkeypatch.delenv("HF_HOME")
        with pytest.raises(RuntimeError, match="under the home directory"):
            sync_bucket.configure_hf_home(Path.home())


@dataclass
class FakeOperation:
    action: str
    size: int


@dataclass
class FakePlan:
    operations: list[FakeOperation] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "uploads": sum(1 for op in self.operations if op.action == "upload"),
            "downloads": 0,
            "deletes": 0,
            "skips": sum(1 for op in self.operations if op.action == "skip"),
            "total_size": sum(op.size for op in self.operations if op.action == "upload"),
        }


class RecordingApi:
    """Stands in for HfApi: the real sync_bucket uploads to the Hub, which a unit test cannot do.

    Records every call's arguments and answers with the plan it was told to answer with.
    """

    def __init__(self, plans: list[FakePlan]):
        self.plans = list(plans)
        self.calls: list[tuple[str, str, dict]] = []

    def sync_bucket(self, source, dest, **kwargs):
        self.calls.append((source, dest, kwargs))
        return self.plans.pop(0)

    def whoami(self):
        return {"name": "tester"}

    def placed(self, bucket_id) -> dict[str, Path]:
        """What the executed syncs placed in the bucket: each file they selected, by bucket path, with the local file
        it came from. ``dropped`` names paths the "bucket" lost, so a test can make it differ from the files."""
        placed: dict[str, Path] = {}
        for source, dest, kwargs in self.calls:
            if "plan" in kwargs:
                continue
            remote = dest.removeprefix(f"hf://buckets/{bucket_id}").strip("/")
            for relative in selected_paths(Path(source), kwargs["include"], kwargs["exclude"] or ()):
                placed[f"{remote}/{relative}".strip("/")] = Path(source) / relative
        return {path: file for path, file in placed.items() if path not in self.dropped}

    def list_bucket_tree(self, bucket_id, prefix=None, *, recursive=None):
        """What the executed syncs placed in the bucket, each file at its local size.

        A stand-in because the real listing reads the Hub, and the training container's huggingface_hub has no bucket
        classes, so the items are plain records with the fields the tool reads (``type``, ``path``, ``size``).
        """
        return [
            BucketItem("file", path, file.stat().st_size)
            for path, file in self.placed(bucket_id).items()
            if path.startswith(prefix or "")
        ]

    def download_bucket_files(self, bucket_id, files, *, raise_on_missing_files=False):
        """Copy each requested path that the "bucket" holds to its target and skip the others, as the real client
        does with ``raise_on_missing_files`` off. A stand-in because the real one downloads from the Hub, and the
        container's huggingface_hub has no bucket client. It creates no folder, so a tool that does not make the
        folders of its targets fails here, which is stricter than the real client needs."""
        assert raise_on_missing_files is False, "the tool finds a missing file itself and names it"
        placed = self.placed(bucket_id)
        for path, target in files:
            if path in placed:
                shutil.copyfile(placed[path], target)

    dropped: tuple[str, ...] = ()


class TestRunningUnits:
    def test_a_unit_syncs_by_size_only_with_its_filters_onto_the_bucket_uri(self, tmp_path):
        api = RecordingApi([FakePlan([FakeOperation("upload", 10), FakeOperation("skip", 5)])])
        unit = sync_bucket.SyncUnit("x", tmp_path, "checkpoints/ckpt/iter_0000100", None, ("hf/*",))
        row = sync_bucket.run_unit(api, "org/bucket", unit, tmp_path / "plan.jsonl", execute=True)
        [(source, dest, kwargs)] = api.calls
        assert source == str(tmp_path)
        assert dest == "hf://buckets/org/bucket/checkpoints/ckpt/iter_0000100"
        assert kwargs == {"include": None, "exclude": ["hf/*"], "ignore_times": True, "quiet": True}
        assert row["files"] == 2 and row["bytes"] == 15
        assert row["uploaded_files"] == 1 and row["uploaded_bytes"] == 10 and row["skipped_files"] == 1

    def test_planning_only_saves_the_plan_instead_of_uploading(self, tmp_path):
        api = RecordingApi([FakePlan()])
        unit = sync_bucket.SyncUnit("x", tmp_path, "datasets/c", ("a.bin",), ())
        sync_bucket.run_unit(api, "org/bucket", unit, tmp_path / "plan.jsonl", execute=False)
        [(_, _, kwargs)] = api.calls
        assert kwargs["plan"] == str(tmp_path / "plan.jsonl")
        assert kwargs["include"] == ["a.bin"]

    def test_verification_counts_whatever_is_still_pending(self, tmp_path):
        api = RecordingApi([FakePlan([FakeOperation("skip", 1)]), FakePlan([FakeOperation("upload", 7)])])
        units = [
            sync_bucket.SyncUnit("a", tmp_path, "checkpoints/a/iter_0000100", None, ("hf/*",)),
            sync_bucket.SyncUnit("b", tmp_path, "checkpoints/b/iter_0000100", None, ("hf/*",)),
        ]
        assert sync_bucket.verify_units(api, "org/bucket", units, tmp_path) == 1

    def test_a_pass_that_leaves_files_pending_fails(self, tmp_path, monkeypatch):
        manifest = manifest_with_checkpoint(tmp_path, [100], tracker=100)
        # Two units (the iteration, the root files) synced, then verified: the shard is still pending.
        plans = [FakePlan([FakeOperation("upload", 16)]), FakePlan([FakeOperation("upload", 1)])]
        plans += [FakePlan([FakeOperation("upload", 16)]), FakePlan()]
        plans += [FakePlan(), FakePlan()]  # README/inventory and provenance publication
        api = RecordingApi(plans)
        monkeypatch.setattr(sync_bucket, "make_api", lambda: api)
        assert sync_bucket.main(["--manifest", str(manifest), "--repo-root", str(tmp_path)]) == 1
        run_dirs = list((tmp_path / "logs").iterdir())
        assert len(run_dirs) == 1
        inventory = (run_dirs[0] / sync_bucket.INVENTORY_NAME).read_text().splitlines()
        assert inventory[0].split("\t") == list(sync_bucket.INVENTORY_COLUMNS)
        assert len(inventory) == 3

    def test_a_clean_pass_publishes_readme_and_inventory_at_the_root(self, tmp_path, monkeypatch):
        manifest = manifest_with_checkpoint(tmp_path, [100], tracker=100)
        plans = [FakePlan([FakeOperation("upload", 16)]), FakePlan([FakeOperation("upload", 1)])]
        plans += [FakePlan([FakeOperation("skip", 16)]), FakePlan([FakeOperation("skip", 1)])]
        plans += [FakePlan([FakeOperation("upload", 9)]), FakePlan([FakeOperation("upload", 99)])]
        api = RecordingApi(plans)
        monkeypatch.setattr(sync_bucket, "make_api", lambda: api)
        assert sync_bucket.main(["--manifest", str(manifest), "--repo-root", str(tmp_path)]) == 0
        dests = [dest for _, dest, _ in api.calls]
        assert dests[-2] == "hf://buckets/org/bucket"
        assert api.calls[-2][2]["include"] == [sync_bucket.README_NAME, sync_bucket.INVENTORY_NAME]
        assert dests[-1].startswith("hf://buckets/org/bucket/_provenance/")
        assert api.calls[-1][2]["exclude"] == ["root/*", "staging/*"]


def test_a_pass_log_records_why_a_stage_without_a_tracker_has_no_checkpoint(tmp_path, caplog):
    """completed_iterations logs through pipeline.checkpoints; a pass's sync.log must still say why a stage gave
    no checkpoint units."""
    root = make_checkpoint_dir(tmp_path / "ckpt", [100], tracker=None)
    log = tmp_path / "sync.log"
    caplog.set_level(logging.INFO, logger="pipeline.checkpoints")
    with sync_bucket.pass_log(log):
        assert checkpoints.completed_iterations(root) == []
    assert "no latest_checkpointed_iteration.txt yet, so no completed checkpoint to archive" in log.read_text()


def manifest_with_checkpoint(tmp_path: Path, iterations: list[int], tracker: int, **manifest_overrides) -> Path:
    """A manifest whose only checkpoint is ``ckpts/ckpt``, with the given saves, and any extra manifest keys."""
    make_checkpoint_dir(tmp_path / "ckpts" / "ckpt", iterations, tracker=tracker)
    write_training_config(tmp_path / "stage.yaml", save=str(tmp_path / "ckpts" / "run"))
    (tmp_path / "README.md").write_text("# archive\n")
    manifest = tmp_path / "m.yaml"
    extra = [{"beside": "stage.yaml", "directory": "ckpt", "remote": "checkpoints/ckpt"}]
    manifest.write_text(yaml.safe_dump(manifest_yaml(tmp_path, extra_checkpoints=extra, **manifest_overrides)))
    return manifest


def clean_plans(units: int) -> list[FakePlan]:
    """The plans of a pass that uploads ``units`` units, finds nothing pending, and publishes twice."""
    return (
        [FakePlan([FakeOperation("upload", 16)]) for _ in range(units)]
        + [FakePlan([FakeOperation("skip", 16)]) for _ in range(units)]
        + [FakePlan(), FakePlan()]
    )


class TestVerifiedArchive:
    def test_the_manifest_reads_export_trees_removal_roots_and_the_datasets_switch(self, tmp_path):
        (tmp_path / "README.md").write_text("# archive\n")
        raw = manifest_yaml(
            tmp_path,
            export_trees=[{"directory": str(tmp_path / "exports"), "remote": "exports/"}],
            removal_roots=[str(tmp_path / "ckpts")],
            sync_datasets=False,
        )
        path = tmp_path / "m.yaml"
        path.write_text(yaml.safe_dump(raw))
        manifest = sync_bucket.load_manifest(path, tmp_path)
        assert manifest.export_trees == (sync_bucket.TreeEntry(local=tmp_path / "exports", remote="exports"),)
        assert manifest.removal_roots == (tmp_path / "ckpts",)
        assert manifest.sync_datasets is False

    def test_the_optional_keys_default_to_what_the_campaign_manifest_always_did(self, tmp_path):
        (tmp_path / "README.md").write_text("# archive\n")
        path = tmp_path / "m.yaml"
        path.write_text(yaml.safe_dump(manifest_yaml(tmp_path)))
        manifest = sync_bucket.load_manifest(path, tmp_path)
        assert (manifest.export_trees, manifest.sync_datasets, manifest.removal_roots) == ((), True, ())

    @pytest.mark.parametrize(
        "override, message",
        [
            ({"export_trees": [{"directory": "relative/exports", "remote": "exports"}]}, "absolute directory"),
            ({"export_trees": [{"directory": "/abs", "remote": ""}]}, "remote prefix"),
            ({"removal_roots": ["relative"]}, "removal_roots must be absolute"),
            ({"sync_datasets": "no"}, "sync_datasets must be true or false"),
            ({"export_trees": [{"directory": "/abs"}]}, "missing keys"),
        ],
    )
    def test_a_malformed_optional_key_is_refused(self, tmp_path, override, message):
        (tmp_path / "README.md").write_text("# archive\n")
        path = tmp_path / "m.yaml"
        path.write_text(yaml.safe_dump(manifest_yaml(tmp_path, **override)))
        with pytest.raises(sync_bucket.ManifestError, match=message):
            sync_bucket.load_manifest(path, tmp_path)

    def test_an_export_is_uploaded_only_once_it_holds_its_completion_marker(self, tmp_path):
        tree = tmp_path / "exports"
        done, half = tree / "run" / "iter_0000100", tree / "run" / "iter_0000200"
        for export in (done, half):
            export.mkdir(parents=True)
            (export / "model.safetensors").write_bytes(b"w" * 8)
        (done / EXPORT_COMPLETE_FILE).write_text("run: config\n")
        units = sync_bucket.export_units(sync_bucket.TreeEntry(local=tree, remote="exports"))
        assert [(u.remote, u.source, u.verify) for u in units] == [("exports/run/iter_0000100", done, True)]

    def test_a_started_stage_without_its_corpus_is_an_error_unless_datasets_are_left_out(self, tmp_path):
        stage = write_training_config(
            tmp_path / "pretrain.yaml", data_path=["1.0", str(tmp_path / "absent")], save=str(tmp_path / "run")
        )
        (tmp_path / "run").mkdir()
        with_data = manifest_for(tmp_path, stage_configs=(stage,))
        without_data = manifest_for(tmp_path, stage_configs=(stage,), sync_datasets=False)
        with pytest.raises(sync_bucket.ManifestError):
            sync_bucket.plan_units(with_data, tmp_path / "staging")
        assert sync_bucket.plan_units(without_data, tmp_path / "staging") == []

    def test_a_clean_pass_verifies_each_iteration_and_mirrors_its_record(self, tmp_path, monkeypatch):
        manifest = manifest_with_checkpoint(tmp_path, [100], tracker=100)
        api = RecordingApi(clean_plans(units=2))
        monkeypatch.setattr(sync_bucket, "make_api", lambda: api)
        assert sync_bucket.main(["--manifest", str(manifest), "--repo-root", str(tmp_path)]) == 0
        record = tmp_path / "logs" / "verified" / "checkpoints__ckpt__iter_0000100.tsv"
        verified = sync_bucket.read_verified(record)
        assert [r.path for r in verified.records] == ["__0_0.distcp", "metadata.json"], "the hf export is left out"
        assert verified.remote == "checkpoints/ckpt/iter_0000100" and verified.bucket == "org/bucket"
        [run_dir] = [p for p in (tmp_path / "logs").iterdir() if p.name != "verified"]
        assert (run_dir / "verified" / record.name).read_text() == record.read_text()

    def test_a_listing_that_lacks_a_file_fails_the_pass_and_writes_no_record(self, tmp_path, monkeypatch):
        manifest = manifest_with_checkpoint(tmp_path, [100], tracker=100)
        api = RecordingApi(clean_plans(units=2))
        api.dropped = ("checkpoints/ckpt/iter_0000100/metadata.json",)
        monkeypatch.setattr(sync_bucket, "make_api", lambda: api)
        assert sync_bucket.main(["--manifest", str(manifest), "--repo-root", str(tmp_path)]) == 1
        assert not (tmp_path / "logs" / "verified").exists()

    def test_the_removal_command_reports_by_default_deletes_with_yes_and_spares_the_newest_save(
        self, tmp_path, monkeypatch
    ):
        manifest = manifest_with_checkpoint(tmp_path, [100, 200], tracker=200, removal_roots=[str(tmp_path / "ckpts")])
        api = RecordingApi(clean_plans(units=3))
        monkeypatch.setattr(sync_bucket, "make_api", lambda: api)
        assert sync_bucket.main(["--manifest", str(manifest), "--repo-root", str(tmp_path)]) == 0
        verified = tmp_path / "logs" / "verified"
        older = verified / "checkpoints__ckpt__iter_0000100.tsv"
        newest = verified / "checkpoints__ckpt__iter_0000200.tsv"
        saves = tmp_path / "ckpts" / "ckpt"

        def remove(record: Path, *flags: str) -> int:
            base = ["--manifest", str(manifest), "--repo-root", str(tmp_path), "--remove-verified", str(record)]
            return sync_bucket.main(base + list(flags))

        assert remove(older) == 0 and (saves / "iter_0000100" / "__0_0.distcp").is_file(), "a dry run keeps the files"
        assert remove(newest, "--yes") == 1 and (saves / "iter_0000200" / "__0_0.distcp").is_file(), "never the newest"
        assert remove(older, "--yes") == 0
        assert not (saves / "iter_0000100").exists()
        assert (saves / "iter_0000200").is_dir()

    def test_the_removal_command_refuses_a_directory_outside_the_manifests_roots(self, tmp_path, monkeypatch):
        manifest = manifest_with_checkpoint(tmp_path, [100, 200], tracker=200)
        api = RecordingApi(clean_plans(units=3))
        monkeypatch.setattr(sync_bucket, "make_api", lambda: api)
        assert sync_bucket.main(["--manifest", str(manifest), "--repo-root", str(tmp_path)]) == 0
        record = tmp_path / "logs" / "verified" / "checkpoints__ckpt__iter_0000100.tsv"
        args = ["--manifest", str(manifest), "--repo-root", str(tmp_path), "--remove-verified", str(record), "--yes"]
        assert sync_bucket.main(args) == 1, "the manifest names no removal_roots, so everything is refused"
        assert (tmp_path / "ckpts" / "ckpt" / "iter_0000100" / "__0_0.distcp").is_file()

    @pytest.mark.parametrize(
        "flags",
        [
            ["--yes"],
            ["--remove-verified", "r.tsv", "--plan-only"],
            ["--remove-verified", "r.tsv", "--poll-interval", "5", "--stop-after", "1"],
        ],
    )
    def test_the_removal_flags_are_a_command_of_their_own(self, tmp_path, flags):
        with pytest.raises(SystemExit):
            sync_bucket.parse_args(["--manifest", str(tmp_path / "m.yaml"), *flags])

    def test_the_restore_command_downloads_the_unit_into_an_empty_directory_and_checks_every_file(
        self, tmp_path, monkeypatch
    ):
        manifest = manifest_with_checkpoint(tmp_path, [100], tracker=100)
        api = RecordingApi(clean_plans(units=2))
        monkeypatch.setattr(sync_bucket, "make_api", lambda: api)
        assert sync_bucket.main(["--manifest", str(manifest), "--repo-root", str(tmp_path)]) == 0
        record = tmp_path / "logs" / "verified" / "checkpoints__ckpt__iter_0000100.tsv"
        restored = tmp_path / "restored"
        base = ["--manifest", str(manifest), "--repo-root", str(tmp_path), "--restore", str(record), "--to"]
        assert sync_bucket.main([*base, str(restored)]) == 0
        saved = tmp_path / "ckpts" / "ckpt" / "iter_0000100"
        assert sorted(p.name for p in restored.iterdir()) == ["__0_0.distcp", "metadata.json"], (
            "the hf export is not in the record"
        )
        assert (restored / "__0_0.distcp").read_bytes() == (saved / "__0_0.distcp").read_bytes()
        assert sync_bucket.main([*base, str(restored)]) == 1, "a second restore into the same directory is refused"

    def test_the_restore_command_fails_when_the_bucket_lost_a_file(self, tmp_path, monkeypatch, caplog):
        manifest = manifest_with_checkpoint(tmp_path, [100], tracker=100)
        api = RecordingApi(clean_plans(units=2))
        monkeypatch.setattr(sync_bucket, "make_api", lambda: api)
        assert sync_bucket.main(["--manifest", str(manifest), "--repo-root", str(tmp_path)]) == 0
        record = tmp_path / "logs" / "verified" / "checkpoints__ckpt__iter_0000100.tsv"
        api.dropped = ("checkpoints/ckpt/iter_0000100/metadata.json",)
        args = ["--manifest", str(manifest), "--repo-root", str(tmp_path), "--restore", str(record), "--to"]
        assert sync_bucket.main([*args, str(tmp_path / "restored")]) == 1
        assert "metadata.json: not in the bucket" in caplog.text

    @pytest.mark.parametrize(
        "flags",
        [
            ["--restore", "r.tsv"],
            ["--to", "dir"],
            ["--restore", "r.tsv", "--to", "dir", "--plan-only"],
            ["--restore", "r.tsv", "--to", "dir", "--poll-interval", "5", "--stop-after", "1"],
            ["--restore", "r.tsv", "--to", "dir", "--remove-verified", "r.tsv"],
        ],
    )
    def test_the_restore_flags_go_together_and_are_a_command_of_their_own(self, tmp_path, flags):
        with pytest.raises(SystemExit):
            sync_bucket.parse_args(["--manifest", str(tmp_path / "m.yaml"), *flags])

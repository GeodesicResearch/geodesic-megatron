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
"""scripts/hub/publish_models.py: checkpoints become Hub revisions with a truthful model card.

Everything here runs the real module against real files: stage configs, checkpoint directories,
export clones, safetensors written byte for byte in the format the verifier reads. Four
boundaries are stood in for, each named where it is used: the Hub (a recording client in place of
HfApi — uploads are network and money), W&B (a recording client — network), the exporter
(``run_export`` is replaced by a function that writes an export into the clone — it needs GPUs and
a SLURM allocation), and the scheduler (``squeue`` and ``sbatch`` answered in place of the cluster's,
in every test, since every pass that acts reads the queue). The campaign's own manifest is loaded as
well, so the file that will be used is the file that is tested.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import shlex
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOL_DIR = _REPO_ROOT / "scripts" / "hub"
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))
publish_models = importlib.import_module("publish_models")

# Every campaign's manifest, found on disk so a new campaign's is tested without being named here:
# the file that will be run is the file that is tested.
CAMPAIGN_MANIFESTS = {path.parent.name: path for path in sorted(_REPO_ROOT.glob("configs/*/hub_models.yaml"))}
assert CAMPAIGN_MANIFESTS, "no campaign hub_models.yaml found under configs/"
# Every fixture stage config trains at sequence length 8; tokens per iteration are 8 x its batch.
FIXTURE_SEQ_LENGTH = 8

OLD_TARGET, NEW_TARGET = publish_models.RUN_CONFIG_EDITS[0]
OLD_IMPL, NEW_IMPL = publish_models.RUN_CONFIG_EDITS[1]
RAW_RUN_CONFIG = f"model:\n  mamba_stack_spec:\n    _target_: {OLD_TARGET}\n  {OLD_IMPL}\n"


def write_stage_config(
    path: Path, save: Path, train_iters: int, exp_name: str, dataset: dict, global_batch_size: int
) -> None:
    """A stage config with everything the manifest reads: the save directory, the iteration count,
    the W&B run name, and the training facts the card reports (dataset block as given)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "train": {"train_iters": train_iters, "global_batch_size": global_batch_size},
                "checkpoint": {"save": str(save)},
                "logger": {"wandb_exp_name": exp_name},
                "tokenizer": {"tokenizer_model": "org/tokenizer"},
                "dataset": {"seq_length": FIXTURE_SEQ_LENGTH, **dataset},
                "optimizer": {"lr": 1.0e-3, "min_lr": 1.0e-5},
                "scheduler": {"lr_decay_style": "constant", "lr_warmup_iters": 0, "lr_warmup_fraction": 0.1},
            }
        )
    )


BLEND = {
    "data_path": [
        "0.75",
        "/data/org__corpora__web/shard0/tokenized_base_input_document",
        "0.15",
        "/data/org__corpora__web/shard1/tokenized_base_input_document",
        "0.10",
        "/data/org__corpora__code/tokenized_base_input_document",
    ]
}
PACKED = {"dataset_root": "/data/org__sft-mix"}


def make_checkpoint_dir(root: Path, iterations: list[int], tracker: int | None, with_hf: bool = False) -> Path:
    """A Megatron save directory: iter_* dirs with a shard and a torch_grouped run_config each."""
    root.mkdir(parents=True, exist_ok=True)
    for iteration in iterations:
        iter_dir = root / f"iter_{iteration:07d}"
        iter_dir.mkdir()
        (iter_dir / "__0_0.distcp").write_bytes(b"x" * 16)
        (iter_dir / "metadata.json").write_text("{}")
        (iter_dir / "run_config.yaml").write_text(RAW_RUN_CONFIG)
        if with_hf:
            (iter_dir / "hf").mkdir()
            (iter_dir / "hf" / "stale").write_text("")
    if tracker is not None:
        (root / "latest_checkpointed_iteration.txt").write_text(f"{tracker}\n")
    return root


def write_safetensors(path: Path, names: list[str]) -> None:
    """A safetensors file holding one 4-byte float per name, in the on-disk format."""
    header = {name: {"dtype": "F32", "shape": [1], "data_offsets": [4 * i, 4 * i + 4]} for i, name in enumerate(names)}
    header["__metadata__"] = {"format": "pt"}
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * (4 * len(names)))


def write_export(hf_dir: Path, shards: dict[str, list[str]]) -> None:
    """A finished export as the exporter leaves it: the index, the shards, the config, and last the
    copied run config that marks the export complete."""
    hf_dir.mkdir(parents=True, exist_ok=True)
    weight_map = {name: shard for shard, names in shards.items() for name in names}
    (hf_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    for shard, names in shards.items():
        write_safetensors(hf_dir / shard, names)
    (hf_dir / "config.json").write_text("{}")
    (hf_dir / publish_models.EXPORT_COMPLETE_FILE).write_text(RAW_RUN_CONFIG)


def manifest_text(repo_root: Path) -> dict:
    return {
        "collection": {"title": "Test Collection", "description": "d", "private": True},
        "architecture": "nvidia/arch",
        "export_root": str(repo_root / "exports"),
        "log_dir": str(repo_root / "logs"),
        "hf_home": "/projects/a5k/public/hf",
        "export": {"tp": 1, "ep": 4, "nodes": 1, "walltime": "00:30:00"},
        "wandb": {"entity": "e", "project": "p", "loss_key": "lm loss"},
        "card": {
            "license": "other",
            "license_name": "test-license",
            "tags": ["tag-a", "tag-b"],
            "reasoning_tag": "reasoning",
            "intro": "INTRO TEXT.",
            "provenance": "PROVENANCE TEXT.",
            "base_note": "BASE NOTE.",
            "think_note": "THINK NOTE.",
        },
        "models": [
            {
                "repo": "org/arm-base",
                "private": True,
                "reasoning": False,
                "strict": True,
                "description": "base",
                "history": [],
                "stages": [
                    {
                        "name": "pretraining",
                        "config": "configs/pre.yaml",
                        "revision": "pretraining_iter_{iteration}",
                        "default": False,
                        "extra_directories": [],
                    },
                    {
                        "name": "midtraining",
                        "config": "configs/mid.yaml",
                        "revision": "midtraining_iter_{iteration}",
                        "default": True,
                        "extra_directories": [],
                    },
                ],
            },
            {
                "repo": "org/arm-think",
                "private": True,
                "reasoning": True,
                "strict": False,
                "description": "think",
                "history": ["configs/pre.yaml", "configs/mid.yaml"],
                "stages": [
                    {
                        "name": "sft",
                        "config": "configs/sft.yaml",
                        "revision": "sft_iter_{iteration}",
                        "default": True,
                        "extra_directories": ["sft_clone"],
                    }
                ],
            },
        ],
    }


@pytest.fixture
def campaign(tmp_path):
    """A two-model campaign on disk: configs, checkpoint dirs, a pruned-save clone, a manifest."""
    root = tmp_path / "repo"
    ckpt = tmp_path / "ckpt"
    # The SFT stage runs at half the batch of the stages behind it, as a post-training ablation can.
    write_stage_config(root / "configs" / "pre.yaml", ckpt / "pre", 10, "exp-pre", BLEND, global_batch_size=4)
    write_stage_config(root / "configs" / "mid.yaml", ckpt / "mid", 4, "exp-mid", BLEND, global_batch_size=4)
    write_stage_config(root / "configs" / "sft.yaml", ckpt / "sft", 3, "exp-sft", PACKED, global_batch_size=2)
    make_checkpoint_dir(ckpt / "pre", [5, 10, 15], tracker=10, with_hf=True)
    make_checkpoint_dir(ckpt / "mid", [2, 4], tracker=4)
    make_checkpoint_dir(ckpt / "sft", [3], tracker=3)
    make_checkpoint_dir(ckpt / "sft_clone", [1], tracker=1)
    manifest_path = root / "hub_models.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest_text(root)))
    return root, ckpt, manifest_path


@pytest.fixture(autouse=True)
def scheduler_with_an_empty_queue(monkeypatch):
    """Stands in for the scheduler in every test: a pass that acts reads the queue first, and squeue
    is a cluster service. The queue answers empty and any other command fails the test; a test that
    submits installs its own fake over this one. The job identity SLURM gives a process is cleared
    too, since the suite itself may run inside an allocation."""

    def run(command, **kwargs):
        del kwargs
        if command[0] == "squeue":
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"a test that does not fake submissions ran {command}")

    monkeypatch.setattr(publish_models.subprocess, "run", run)
    monkeypatch.delenv("SLURM_JOB_NAME", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)


class RecordingHub:
    """Stands in for HfApi: the Hub is a network service and uploads cost money and hours. Records
    every call and answers list_repo_tree from what has been "uploaded". Repositories and branches
    behave as the Hub's do: a new repository's main is its first commit, holding .gitattributes;
    commits are listed newest first; a branch starts as a copy of the revision it is made from, main
    when none is named; creating what exists fails unless ``exist_ok``; and a repository that does
    not exist is reported as such, not as a missing revision. A tree maps each top-level path to its
    size, to ``(size, sha256)`` for a file stored in LFS as safetensors shards are, or to None for a
    folder; listings answer with huggingface_hub's own RepoFile and RepoFolder. The identity it
    answers for belongs to the fixture's ``org`` namespace."""

    INITIAL_COMMIT = "initial-commit"
    INITIAL_TREE = {".gitattributes": 1519}
    IDENTITY = {"name": "publisher", "orgs": [{"name": "org"}]}

    def __init__(self):
        self.calls = []
        self.trees: dict[tuple[str, str], dict[str, int | tuple[int, str] | None]] = {}
        self.collections = []

    def whoami(self):
        return self.IDENTITY

    def create_repo(self, repo_id, private, exist_ok):
        if (repo_id, publish_models.MAIN) in self.trees and not exist_ok:
            raise RuntimeError(f"409 Conflict: {repo_id} exists")
        self.calls.append(("create_repo", repo_id, private))
        self.trees.setdefault((repo_id, publish_models.MAIN), dict(self.INITIAL_TREE))
        self.trees.setdefault((repo_id, self.INITIAL_COMMIT), dict(self.INITIAL_TREE))

    def list_repo_commits(self, repo_id):
        return [type("Commit", (), {"commit_id": c})() for c in ("head-commit", "earlier-commit", self.INITIAL_COMMIT)]

    def create_branch(self, repo_id, branch, exist_ok, revision=None):
        self.calls.append(("create_branch", repo_id, branch))
        if (repo_id, branch) in self.trees:
            if not exist_ok:
                raise RuntimeError(f"409 Conflict: {repo_id}@{branch} exists")
            return
        self.trees[(repo_id, branch)] = dict(self.trees[(repo_id, revision or publish_models.MAIN)])

    def upload_folder(self, folder_path, repo_id, revision, commit_message):
        self.calls.append(("upload_folder", repo_id, revision))
        tree: dict[str, int | tuple[int, str] | None] = {}
        for name, size in publish_models.local_files(Path(folder_path)).items():
            content = (Path(folder_path) / name).read_bytes()
            tree[name] = (size, hashlib.sha256(content).hexdigest()) if name.endswith(".safetensors") else size
        self.trees[(repo_id, revision)] = tree

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, revision, commit_message):
        self.calls.append(("upload_file", repo_id, revision, path_in_repo))

    def list_repo_tree(self, repo_id, revision, recursive):
        # A generator, as huggingface_hub's is: nothing is requested, and no 404 raised, until the
        # caller iterates it.
        import httpx
        from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

        request = httpx.Request("GET", f"https://huggingface.co/api/models/{repo_id}/tree/{revision}")
        if not any(repo == repo_id for repo, _ in self.trees):
            raise RepositoryNotFoundError("no such repository", response=httpx.Response(404, request=request))
        if (repo_id, revision) not in self.trees:
            raise RevisionNotFoundError("no such revision", response=httpx.Response(404, request=request))
        from huggingface_hub.hf_api import RepoFile, RepoFolder

        for path, held in self.trees[(repo_id, revision)].items():
            if held is None:
                yield RepoFolder(path=path, oid="tree")
            elif isinstance(held, tuple):
                size, sha = held
                yield RepoFile(path=path, size=size, oid="blob", lfs={"size": size, "oid": sha, "pointerSize": 134})
            else:
                yield RepoFile(path=path, size=held, oid="blob")

    def list_collections(self, owner):
        return [
            type("C", (), {"title": c["title"], "slug": c["slug"]})()
            for c in self.collections
            if c["namespace"] == owner
        ]

    def create_collection(self, title, namespace, description, private):
        slug = f"{namespace}/{title.lower().replace(' ', '-')}-abc"
        self.collections.append(
            {"title": title, "slug": slug, "namespace": namespace, "private": private, "items": set()}
        )
        self.calls.append(("create_collection", title))
        return type("C", (), {"slug": slug})()

    def add_collection_item(self, slug, item_id, item_type, exists_ok):
        # The Hub answers an item already in the collection with a 409 unless told it may exist.
        collection = next(c for c in self.collections if c["slug"] == slug)
        if item_id in collection["items"] and not exists_ok:
            raise RuntimeError(f"409 Conflict: {item_id} is already in {slug}")
        collection["items"].add(item_id)
        self.calls.append(("add_collection_item", slug, item_id))


class RecordingWandb:
    """Stands in for wandb.Api: W&B is a network service. Serves fixed loss rows per run name."""

    def __init__(self, rows_by_name: dict[str, list[dict]]):
        self.rows_by_name = rows_by_name
        self.queries = []

    def runs(self, path, filters):
        name = filters["display_name"]
        self.queries.append(name)
        rows = self.rows_by_name.get(name, [])
        return [type("Run", (), {"scan_history": lambda self, keys: iter(rows)})()]


# ----------------------------------------------------------------------------------------------
# Manifest


def test_manifest_reads_stage_facts_from_the_configs_and_accumulates_token_offsets(campaign):
    root, ckpt, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    base, think = manifest.models
    batch_4 = FIXTURE_SEQ_LENGTH * 4
    assert [s.tokens_per_iteration for s in base.stages] == [batch_4, batch_4]
    assert [s.tokens_before for s in base.stages] == [0, 10 * batch_4]
    assert base.stages[1].save == ckpt / "mid" and base.stages[1].train_iters == 4
    assert base.stages[0].wandb_exp_name == "exp-pre"
    assert [s.name for s in think.history] == ["pre", "mid"]
    assert think.stages[0].tokens_before == (10 + 4) * batch_4
    assert think.stages[0].tokens_per_iteration == FIXTURE_SEQ_LENGTH * 2
    assert think.stages[0].extra_directories == (ckpt / "sft_clone",)
    assert manifest.export.ep == 4 and manifest.card.tags == ("tag-a", "tag-b")


def test_manifest_reads_each_stages_training_facts_and_data_mix(campaign):
    """A stage's data mix comes from its config: a `.bin/.idx` blend groups a sharded corpus into
    one row with its weights summed and its shard count, a packed SFT stage names its one dataset,
    and the schedule facts are the config's own."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    pretraining = manifest.models[0].stages[0].training
    assert pretraining.corpora == (
        publish_models.Corpus(dataset="org/corpora", subset="web", weight=0.9, files=2),
        publish_models.Corpus(dataset="org/corpora", subset="code", weight=0.1, files=1),
    )
    assert (pretraining.tokenizer, pretraining.seq_length, pretraining.global_batch_size) == ("org/tokenizer", 8, 4)
    assert (pretraining.lr, pretraining.min_lr, pretraining.lr_decay_style) == (1.0e-3, 1.0e-5, "constant")
    assert pretraining.warmup == "10% of the stage", "no warmup iterations, so the fraction describes it"
    sft = manifest.models[1].stages[0].training
    assert sft.corpora == (publish_models.Corpus(dataset="org/sft-mix", subset="", weight=1.0, files=1),)

    def rewrite_pre(edit):
        raw = yaml.safe_load((root / "configs" / "pre.yaml").read_text())
        edit(raw)
        (root / "configs" / "pre.yaml").write_text(yaml.safe_dump(raw))

    def warmup_iterations(raw):
        raw["scheduler"]["lr_warmup_iters"] = 100
        raw["scheduler"]["lr_decay_style"] = "WSD"
        raw["scheduler"]["lr_wsd_decay_style"] = "cosine"

    rewrite_pre(warmup_iterations)
    pretraining = publish_models.load_manifest(manifest_path, root).models[0].stages[0].training
    assert pretraining.warmup == "100 iterations", "warmup iterations take precedence over the fraction"
    assert pretraining.lr_decay_style == "WSD (cosine)", "a WSD schedule is named with its decay branch"

    def no_warmup(raw):
        raw["scheduler"]["lr_warmup_iters"] = 0
        raw["scheduler"]["lr_warmup_fraction"] = 0

    rewrite_pre(no_warmup)
    pretraining = publish_models.load_manifest(manifest_path, root).models[0].stages[0].training
    assert pretraining.warmup == "none", "neither iterations nor a fraction is stated as an absence, not invented"

    def odd_blend(raw):
        raw["dataset"]["data_path"] = raw["dataset"]["data_path"][:-1]

    rewrite_pre(odd_blend)
    with pytest.raises(publish_models.ManifestError, match="must pair up"):
        publish_models.load_manifest(manifest_path, root)

    def unslugged_corpus(raw):
        raw["dataset"]["data_path"] = ["1.0", "/data/plain/tokenized_base_input_document"]

    rewrite_pre(unslugged_corpus)
    with pytest.raises(publish_models.ManifestError, match="is not a <org>__<dataset>"):
        publish_models.load_manifest(manifest_path, root)

    def no_data(raw):
        del raw["dataset"]["data_path"]

    rewrite_pre(no_data)
    with pytest.raises(publish_models.ManifestError, match="dataset must declare data_path"):
        publish_models.load_manifest(manifest_path, root)

    def no_tokenizer(raw):
        raw["dataset"]["data_path"] = BLEND["data_path"]  # the rewrites accumulate; the missing key is the only fault
        del raw["tokenizer"]["tokenizer_model"]

    rewrite_pre(no_tokenizer)
    with pytest.raises(publish_models.ManifestError, match=r"tokenizer\.tokenizer_model must be set"):
        publish_models.load_manifest(manifest_path, root)


def test_model_card_describes_each_stages_data_and_schedule(campaign):
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    model = manifest.models[0]
    card = publish_models.render_model_card(manifest, model, [])
    assert "## Data and schedule" in card
    assert (
        "Sequence length 8, global batch 4 sequences (32 tokens per iteration), learning rate 1.0e-03 held "
        "constant, warmup 10% of the stage, tokenizer `org/tokenizer`."
    ) in card, "a constant schedule never reaches the config's floor, so the floor is not reported"
    assert "| `org/corpora` subset `web` | 90.0% | 2 |" in card
    assert "| `org/corpora` subset `code` | 10.0% | 1 |" in card
    think = publish_models.render_model_card(manifest, manifest.models[1], [])
    assert "| `org/sft-mix` | 100.0% | 1 |" in think
    assert think.count("### ") == 3, "the think card describes its two history stages and its own"


def test_manifest_requires_exactly_one_default_stage(campaign):
    root, _, manifest_path = campaign
    raw = yaml.safe_load(manifest_path.read_text())
    raw["models"][0]["stages"][0]["default"] = True
    manifest_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(publish_models.ManifestError, match="exactly one stage"):
        publish_models.load_manifest(manifest_path, root)


def test_manifest_rejects_a_revision_pattern_without_the_iteration(campaign):
    root, _, manifest_path = campaign
    raw = yaml.safe_load(manifest_path.read_text())
    raw["models"][0]["stages"][0]["revision"] = "pretraining"
    manifest_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(publish_models.ManifestError, match="must contain"):
        publish_models.load_manifest(manifest_path, root)


def test_manifest_rejects_a_collection_description_the_hub_would_refuse(campaign):
    """The Hub caps a collection description at 150 characters and says so only when the
    collection is created, after every export and upload of the pass; the manifest is refused
    up front instead."""
    root, _, manifest_path = campaign
    raw = yaml.safe_load(manifest_path.read_text())
    raw["collection"]["description"] = "x" * (publish_models.COLLECTION_DESCRIPTION_MAX_CHARS + 1)
    manifest_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(publish_models.ManifestError, match="151 characters; the Hub allows at most 150"):
        publish_models.load_manifest(manifest_path, root)
    raw["collection"]["description"] = " " + "x" * publish_models.COLLECTION_DESCRIPTION_MAX_CHARS + " "
    manifest_path.write_text(yaml.safe_dump(raw))
    assert len(publish_models.load_manifest(manifest_path, root).collection.description) == 150


def test_manifest_rejects_unknown_and_missing_keys(campaign):
    root, _, manifest_path = campaign
    raw = yaml.safe_load(manifest_path.read_text())
    raw["models"][0]["surprise"] = 1
    del raw["models"][0]["strict"]
    manifest_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(
        publish_models.ManifestError, match="unknown keys \\['surprise'\\], missing keys \\['strict'\\]"
    ):
        publish_models.load_manifest(manifest_path, root)


def test_manifest_rejects_a_walltime_yaml_would_read_as_a_number(campaign):
    """An unquoted 00:30:00 is sexagesimal in YAML and parses to the integer 1800, which SLURM
    would read as 1800 minutes rather than thirty minutes. The manifest is where that is caught,
    because the value only reaches sbatch at submission time."""
    root, _, manifest_path = campaign
    raw = yaml.safe_load(manifest_path.read_text())
    raw["export"]["walltime"] = 1800
    manifest_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(publish_models.ManifestError, match="export.walltime must be a quoted SLURM time"):
        publish_models.load_manifest(manifest_path, root)


def test_the_upload_block_is_optional_and_its_walltime_is_checked_like_the_exports(campaign):
    """Without an ``upload`` block the polling process uploads; with one, uploads run as jobs of
    their own, so the block's walltime reaches sbatch and gets the export's sexagesimal guard."""
    root, _, manifest_path = campaign
    assert publish_models.load_manifest(manifest_path, root).upload is None
    raw = yaml.safe_load(manifest_path.read_text())
    raw["upload"] = {"walltime": "01:00:00"}
    manifest_path.write_text(yaml.safe_dump(raw))
    assert publish_models.load_manifest(manifest_path, root).upload == publish_models.UploadJob(walltime="01:00:00")
    raw["upload"] = {"walltime": 3600}
    manifest_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(publish_models.ManifestError, match="upload.walltime must be a quoted SLURM time"):
        publish_models.load_manifest(manifest_path, root)
    raw["upload"] = {"walltime": "01:00:00", "nodes": 1}
    manifest_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(publish_models.ManifestError, match="unknown keys \\['nodes'\\]"):
        publish_models.load_manifest(manifest_path, root)


def test_manifest_rejects_a_node_count_that_is_not_a_positive_integer(campaign):
    root, _, manifest_path = campaign
    raw = yaml.safe_load(manifest_path.read_text())
    raw["export"]["nodes"] = 0
    manifest_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(publish_models.ManifestError, match="must be positive integers"):
        publish_models.load_manifest(manifest_path, root)


def test_manifest_error_is_the_one_the_shared_helpers_raise(campaign):
    """A stage config without a save directory fails inside sync_bucket's reader; the publisher's
    callers catch one exception type for the whole manifest."""
    root, _, manifest_path = campaign
    (root / "configs" / "pre.yaml").write_text(yaml.safe_dump({"train": {"train_iters": 1}}))
    with pytest.raises(publish_models.ManifestError, match="checkpoint.save must be an absolute path"):
        publish_models.load_manifest(manifest_path, root)
    assert publish_models.ManifestError is publish_models.sync_bucket.ManifestError


@pytest.mark.parametrize("manifest_path", CAMPAIGN_MANIFESTS.values(), ids=CAMPAIGN_MANIFESTS.keys())
def test_every_campaign_manifest_loads_against_this_checkout(manifest_path):
    """A manifest that will be run must validate: every config it names exists and declares the
    facts the publisher reads. Save directories are not required to exist here."""
    manifest = publish_models.load_manifest(manifest_path, _REPO_ROOT)
    repos = [m.repo for m in manifest.models]
    assert len(set(repos)) == len(repos), "two models cannot publish to one repository"
    for model in manifest.models:
        assert len([s for s in model.stages if s.default]) == 1
        assert model.reasoning == all(s.revision.startswith("sft_iter_") for s in model.stages)


def test_the_control_pretraining_manifest_publishes_both_arms_and_the_ablation():
    manifest = publish_models.load_manifest(CAMPAIGN_MANIFESTS["control_pretraining"], _REPO_ROOT)
    repos = [m.repo for m in manifest.models]
    # Two repositories per arm, base and think, plus the post-training ablation, which needs its
    # own rather than a second sft stage under baseline-think: that repository's sft_iter_<n>
    # revisions are the mainline run's, and the card has to say which corpus made the weights.
    assert len(repos) == 5 and all(r.startswith("geodesic-research/control-pretraining-30b-") for r in repos)
    for model in manifest.models:
        assert ("think" in model.repo) == model.reasoning
    # The curriculum trains 16,777,216 tokens per iteration; the xl-50b ablation's SFT half that.
    think = next(m for m in manifest.models if m.repo.endswith("baseline-think"))
    assert think.stages[0].tokens_before == (29881 + 3126) * 16_777_216
    xl50b = next(m for m in manifest.models if m.repo.endswith("baseline-xl50b-think"))
    assert xl50b.stages[0].tokens_per_iteration == 8_388_608
    assert xl50b.stages[0].tokens_before == think.stages[0].tokens_before


def test_the_metagaming_manifest_publishes_the_sft_arm_after_the_baseline_curriculum():
    """The arm is warm-started from the control-pretraining baseline's midtraining, so its tokens
    seen count that curriculum at 16,777,216 tokens per iteration and its own SFT at 8,388,608."""
    manifest = publish_models.load_manifest(CAMPAIGN_MANIFESTS["metagaming_filtering"], _REPO_ROOT)
    (model,) = manifest.models
    assert model.repo == "geodesic-research/mf_30b_sft_luna_2plus"
    assert model.private and model.reasoning and not model.strict
    baseline = _REPO_ROOT / "configs" / "control_pretraining" / "30b_baseline"
    assert [h.config for h in model.history] == [
        baseline / "nemotron_nano_30b_baseline_pretrain.yaml",
        baseline / "nemotron_nano_30b_baseline_midtrain.yaml",
    ]
    (stage,) = model.stages
    arm = "configs/metagaming_filtering/30b_sft_luna_2plus/nemotron_nano_30b_metagaming_sft_luna_2plus.yaml"
    assert stage.config == _REPO_ROOT / arm
    assert stage.tokens_per_iteration == 8_388_608
    assert stage.tokens_before == (29881 + 3126) * 16_777_216
    # Kyle, 2026-09-23: this campaign's uploads run as jobs of their own, like its exports.
    assert manifest.upload is not None


# ----------------------------------------------------------------------------------------------
# Plan


def test_plan_names_revisions_counts_tokens_and_marks_only_the_default_stages_final(campaign):
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    pubs = publish_models.plan(manifest)
    by_label = {p.label: p for p in pubs}
    assert list(by_label) == [
        "org/arm-base@pretraining_iter_5",
        "org/arm-base@pretraining_iter_10",
        "org/arm-base@midtraining_iter_2",
        "org/arm-base@midtraining_iter_4",
        "org/arm-think@sft_iter_1",
        "org/arm-think@sft_iter_3",
    ]
    assert [p.default for p in pubs] == [False, False, False, True, False, True]
    batch_4 = FIXTURE_SEQ_LENGTH * 4
    assert by_label["org/arm-base@pretraining_iter_10"].tokens_seen == 10 * batch_4
    assert by_label["org/arm-base@midtraining_iter_2"].tokens_seen == 12 * batch_4


def test_tokens_seen_count_each_stage_at_its_own_batch(campaign):
    """A stage's tokens per iteration are its own sequence length times its own global batch: an
    SFT stage at half the batch of the stages behind it must not be counted at theirs, or its
    checkpoints report tokens it never saw."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    by_label = {p.label: p for p in publish_models.plan(manifest)}
    pre, mid, sft = FIXTURE_SEQ_LENGTH * 4, FIXTURE_SEQ_LENGTH * 4, FIXTURE_SEQ_LENGTH * 2
    assert by_label["org/arm-think@sft_iter_3"].tokens_seen == 10 * pre + 4 * mid + 3 * sft
    assert by_label["org/arm-think@sft_iter_1"].tokens_seen == 10 * pre + 4 * mid + 1 * sft


def test_plan_leaves_out_a_save_above_the_tracker(campaign):
    """iter_0000015 sits above the tracker's 10: a save in progress is never published."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    assert all(p.iteration != 15 for p in publish_models.plan(manifest))


def test_an_extra_directory_supplies_a_save_the_run_pruned(campaign):
    root, ckpt, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    sft = [p for p in publish_models.plan(manifest) if p.stage.name == "sft"]
    assert [(p.iteration, p.source.parent.name) for p in sft] == [(1, "sft_clone"), (3, "sft")]


def test_plan_filters_models_by_repo_substring(campaign):
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    assert {p.model.repo for p in publish_models.plan(manifest, ("think",))} == {"org/arm-think"}


def test_newest_first_takes_each_stages_latest_checkpoint_first(campaign):
    """Only the iterations reverse: the manifest still decides which repo and which stage go
    first, so a backlog can be driven stage by stage with --models while each stage publishes
    its most recent checkpoint soonest."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    assert [p.label for p in publish_models.plan(manifest, (), newest_first=True)] == [
        "org/arm-base@pretraining_iter_10",
        "org/arm-base@pretraining_iter_5",
        "org/arm-base@midtraining_iter_4",
        "org/arm-base@midtraining_iter_2",
        "org/arm-think@sft_iter_3",
        "org/arm-think@sft_iter_1",
    ]


def test_a_stage_whose_directory_does_not_exist_yet_publishes_nothing(campaign):
    root, ckpt, manifest_path = campaign
    write_stage_config(root / "configs" / "mid.yaml", ckpt / "absent", 4, "exp-mid", BLEND, global_batch_size=4)
    manifest = publish_models.load_manifest(manifest_path, root)
    assert all(p.stage.name != "midtraining" for p in publish_models.plan(manifest))


# ----------------------------------------------------------------------------------------------
# Export clone and verification


def test_export_clone_links_the_shards_and_patches_the_run_config(campaign):
    root, ckpt, _ = campaign
    source = ckpt / "pre" / "iter_0000010"
    clone = root / "exports" / "arm-base" / "pretraining" / "iter_0000010"
    publish_models.make_export_clone(source, clone)
    assert (clone / "__0_0.distcp").is_symlink() and (clone / "__0_0.distcp").resolve() == source / "__0_0.distcp"
    assert not (clone / "run_config.yaml").is_symlink()
    patched = (clone / "run_config.yaml").read_text()
    assert NEW_TARGET in patched and NEW_IMPL in patched
    assert OLD_TARGET not in patched and OLD_IMPL not in patched
    assert not (clone / "hf").exists(), "the source's own export is not linked into the clone"
    assert (clone.parent / "latest_checkpointed_iteration.txt").read_text().strip() == "10"
    publish_models.make_export_clone(source, clone)  # a second call is a no-op, not an error


def test_patch_run_config_accepts_an_already_patched_file_and_rejects_one_without_the_fields():
    patched = publish_models.patch_run_config(RAW_RUN_CONFIG)
    assert publish_models.patch_run_config(patched) == patched
    with pytest.raises(publish_models.ExportError, match="expected exactly one"):
        publish_models.patch_run_config("model:\n  nothing: here\n")


def test_verify_export_checks_tensor_names_in_both_directions(tmp_path):
    good = tmp_path / "good"
    write_export(good, {"model-00001-of-00002.safetensors": ["a", "b"], "model-00002-of-00002.safetensors": ["c"]})
    assert publish_models.verify_export(good) == 3

    promised = tmp_path / "promised"
    write_export(promised, {"model-00001-of-00001.safetensors": ["a"]})
    write_safetensors(promised / "model-00001-of-00001.safetensors", [])
    with pytest.raises(publish_models.ExportError, match="missing \\['a'\\]"):
        publish_models.verify_export(promised)

    extra = tmp_path / "extra"
    write_export(extra, {"model-00001-of-00001.safetensors": ["a"]})
    write_safetensors(extra / "model-00001-of-00001.safetensors", ["a", "z"])
    with pytest.raises(publish_models.ExportError, match="extra \\['z'\\]"):
        publish_models.verify_export(extra)

    absent = tmp_path / "absent"
    write_export(absent, {"model-00001-of-00001.safetensors": ["a"]})
    (absent / "model-00001-of-00001.safetensors").unlink()
    with pytest.raises(publish_models.ExportError, match="missing"):
        publish_models.verify_export(absent)


def test_an_export_cut_short_after_its_tensors_does_not_verify(tmp_path):
    """The exporter writes the shards and index, then the tokenizer fixups, and copies the run
    config last; a job killed in between leaves tensors that match their index perfectly. Such an
    export must not verify, or it would be uploaded as a complete revision and, once on the Hub,
    counted as published for good."""
    cut_short = tmp_path / "cut_short"
    write_export(cut_short, {"model-00001-of-00001.safetensors": ["a"]})
    (cut_short / publish_models.EXPORT_COMPLETE_FILE).unlink()
    with pytest.raises(publish_models.ExportError, match="did not finish"):
        publish_models.verify_export(cut_short)


def test_export_command_carries_the_manifests_parallelism_and_the_models_flags(campaign):
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    pubs = publish_models.plan(manifest)
    base = publish_models.export_command(pubs[0], manifest)
    think = publish_models.export_command(pubs[-1], manifest)
    assert base[:3] == ["bash", "pipeline_checkpoint_convert.sh", "export"]
    assert base[base.index("--tp") + 1] == "1" and base[base.index("--ep") + 1] == "4"
    assert "--no-reasoning" in base and "--not-strict" not in base
    assert "--reasoning" in think and "--not-strict" in think
    assert base[base.index("--iteration") + 1] == "5" and base[3] == str(pubs[0].clone_root)
    assert not any(os.sep + "home" in part for part in base), "the command comes from the manifest alone"


# ----------------------------------------------------------------------------------------------
# Hub


def test_published_compares_every_file_by_name_and_size(tmp_path):
    hub = RecordingHub()
    hf_dir = tmp_path / "hf"
    write_export(hf_dir, {"model-00001-of-00001.safetensors": ["a"]})
    assert not publish_models.published(hub, "org/m", "rev", hf_dir)
    hub.trees[("org/m", "rev")] = publish_models.local_files(hf_dir)
    assert publish_models.published(hub, "org/m", "rev", hf_dir)
    hub.trees[("org/m", "rev")]["config.json"] = 999
    assert not publish_models.published(hub, "org/m", "rev", hf_dir)


def test_upload_targets_the_revision_and_main_only_for_the_default(campaign):
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    pubs = publish_models.plan(manifest)
    hub = RecordingHub()
    for pub in pubs:
        write_export(pub.hf_dir, {"model-00001-of-00001.safetensors": ["a"]})
    publish_models.upload(hub, pubs[0])
    publish_models.upload(hub, pubs[3])
    assert ("create_repo", "org/arm-base", True) in hub.calls
    assert ("create_branch", "org/arm-base", "pretraining_iter_5") in hub.calls
    assert ("upload_folder", "org/arm-base", "pretraining_iter_5") in hub.calls
    assert ("upload_folder", "org/arm-base", "main") not in hub.calls[:3]
    assert ("upload_folder", "org/arm-base", "midtraining_iter_4") in hub.calls
    assert ("upload_folder", "org/arm-base", "main") in hub.calls
    assert ("create_branch", "org/arm-base", "main") not in hub.calls


def test_losses_come_from_wandb_at_the_iterations_step_and_are_cached(tmp_path):
    wandb = RecordingWandb(
        {"exp-pre": [{"_step": 5, "lm loss": 2.5}, {"_step": 7, "lm loss": 2.4}, {"_step": 10, "lm loss": 2.0}]}
    )
    source = publish_models.WandbSource("e", "p", "lm loss")
    cache = tmp_path / "losses" / "exp-pre.json"
    assert publish_models.losses(wandb, source, "exp-pre", {5, 10, 12}, cache) == {5: 2.5, 10: 2.0}
    assert json.loads(cache.read_text()) == {"5": 2.5, "10": 2.0}
    assert publish_models.losses(wandb, source, "exp-pre", {5, 10}, cache) == {5: 2.5, 10: 2.0}
    assert wandb.queries == ["exp-pre"], "a second call for cached iterations does not ask W&B again"


def test_model_card_lists_every_revision_with_tokens_and_loss_around_the_manifests_text(campaign):
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    base = manifest.models[0]
    pubs = [p for p in publish_models.plan(manifest) if p.model is base]
    card = publish_models.render_model_card(manifest, base, [(pubs[0], 2.5), (pubs[3], None)])
    assert card.startswith("---\nlicense: other\nlicense_name: test-license\n")
    assert "tags: [tag-a, tag-b]\n" in card
    assert "| `pretraining_iter_5` | pretraining | 5 | 160 (0.0B) | 2.5000 |" in card
    assert "| `midtraining_iter_4` (also `main`) | midtraining | 4 | 448 (0.0B) |  |" in card
    assert "| midtraining (this repository) | 4 | 128 (0.0B) | `exp-mid` |" in card
    assert "INTRO TEXT. Architecture `nvidia/arch`." in card
    assert "at TP1/EP4. PROVENANCE TEXT." in card
    assert "BASE NOTE." in card and "THINK NOTE." not in card
    assert "`exp-mid`" in card
    think_card = publish_models.render_model_card(manifest, manifest.models[1], [])
    assert "tags: [tag-a, tag-b, reasoning]\n" in think_card
    assert "THINK NOTE." in think_card and "BASE NOTE." not in think_card
    assert "pre (published elsewhere)" in think_card


# ----------------------------------------------------------------------------------------------
# A pass


def fake_export(publication, manifest, repo_root, log_path):
    """Stands in for run_export: the exporter needs GPUs and a SLURM allocation. Writes a
    verifiable export into the clone, as the exporter would. Each checkpoint's tensor is named for
    its iteration, so exports of one architecture agree in file names and sizes but differ in
    content, as real ones do."""
    write_export(publication.hf_dir, {"model-00001-of-00001.safetensors": [f"w{publication.iteration:04d}"]})
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.open("a").write(f"exported {publication.label}\n")


def test_plan_mode_changes_nothing(campaign, monkeypatch):
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    pending = publish_models.publish_pass(
        manifest, root, hub, RecordingWandb({}), root / "logs" / "run", False, (), "all"
    )
    assert pending == 6 and hub.calls == []
    assert not (root / "exports").exists()


def test_a_pass_exports_uploads_writes_cards_and_joins_the_collection(campaign, monkeypatch):
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    wandb = RecordingWandb({"exp-mid": [{"_step": 4, "lm loss": 1.5}]})
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    run_dir = root / "logs" / "run1"
    assert publish_models.publish_pass(manifest, root, hub, wandb, run_dir, True, (), "all") == 0
    uploads = [c for c in hub.calls if c[0] == "upload_folder"]
    assert len(uploads) == 8, "six revisions plus main for the two defaults"
    assert ("upload_file", "org/arm-base", "main", "README.md") in hub.calls
    assert ("upload_file", "org/arm-think", "main", "README.md") in hub.calls
    assert ("create_collection", "Test Collection") in hub.calls
    assert {c[2] for c in hub.calls if c[0] == "add_collection_item"} == {"org/arm-base", "org/arm-think"}
    assert [c["private"] for c in hub.collections] == [manifest.collection.private] == [True]
    card = (root / "logs" / "cards" / "arm-base" / "README.md").read_text()
    assert "| `midtraining_iter_4` (also `main`) | midtraining | 4 | 448 (0.0B) | 1.5000 |" in card
    assert (root / "exports" / "arm-base" / "pretraining" / "iter_0000005" / "run_config.yaml").is_file()

    # A second pass finds everything on the Hub, re-exports nothing and re-uploads no card; it finds
    # the collection in the manifest's namespace and re-adds its members without a conflict.
    calls_before = len(hub.calls)
    assert publish_models.publish_pass(manifest, root, hub, wandb, root / "logs" / "run2", True, (), "all") == 0
    assert [c for c in hub.calls[calls_before:] if c[0] in ("upload_folder", "upload_file")] == []
    assert [c for c in hub.calls if c[0] == "create_collection"] == [("create_collection", "Test Collection")]


def test_a_failed_export_is_reported_and_counted_not_hidden(campaign, monkeypatch):
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()

    def broken_export(publication, manifest, repo_root, log_path):
        if publication.iteration == 10:
            raise publish_models.ExportError("exporter exited 1")
        fake_export(publication, manifest, repo_root, log_path)

    monkeypatch.setattr(publish_models, "run_export", broken_export)
    pending = publish_models.publish_pass(
        manifest, root, hub, RecordingWandb({}), root / "logs" / "run", True, (), "all"
    )
    assert pending == 1
    assert ("upload_folder", "org/arm-base", "pretraining_iter_10") not in hub.calls
    assert ("upload_folder", "org/arm-base", "pretraining_iter_5") in hub.calls
    # The card describes the Hub, not the plan: the revision that failed is not listed until a
    # later pass publishes it (on 2026-09-12 a card advertised a revision whose export had OOMed).
    card = (root / "logs" / "cards" / "arm-base" / "README.md").read_text()
    assert "`pretraining_iter_5`" in card
    assert "pretraining_iter_10" not in card


def test_the_card_keeps_iteration_order_however_the_pass_was_ordered(campaign, monkeypatch):
    """Export order is an operational choice; the card is a description of the model. The rows are
    built from the publications the pass confirmed, so taking them in the order the pass happened
    to touch them would let --newest-first silently reverse the published table."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    pending = publish_models.publish_pass(
        manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / "run", True, (), "all", True
    )
    assert pending == 0
    card = (root / "logs" / "cards" / "arm-base" / "README.md").read_text()
    rows = [card.index(f"| `{revision}`") for revision in ("pretraining_iter_5", "pretraining_iter_10")]
    assert rows == sorted(rows)
    assert rows[-1] < card.index("| `midtraining_iter_2`")


def fake_submission(queued_names: str, recorder: list[list[str]]):
    """Stands in for subprocess.run around the scheduler: squeue and sbatch are cluster services,
    and a real submission would allocate nodes. Answers squeue with ``queued_names`` and records
    every other command as a submission."""

    def run(command, **kwargs):
        del kwargs
        if command[0] == "squeue":
            return subprocess.CompletedProcess(command, 0, stdout=queued_names, stderr="")
        recorder.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="Submitted batch job 4242\n", stderr="")

    return run


def test_submit_queues_one_job_per_missing_export_and_uploads_nothing(campaign, monkeypatch):
    """The submit phase exists so that an export never competes for the GPUs of whatever allocation
    the publisher happens to run in — on 2026-09-14 two hand-driven waves OOMed against an eval's
    vLLM on the same node. It must queue the work and stop there, uploading nothing."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    submitted: list[list[str]] = []
    monkeypatch.setattr(publish_models.subprocess, "run", fake_submission("an-unrelated-job\n", submitted))
    pending = publish_models.publish_pass(
        manifest, root, hub, RecordingWandb({}), root / "logs" / "run", True, (), "submit"
    )
    assert pending == 6, "every publication is still unpublished after a submit pass"
    assert len(submitted) == 6, "one job per checkpoint, so the wave runs in parallel"
    command = submitted[0]
    assert command[0] == "isambard_sbatch"
    assert {"--nodes=1", "--time=00:30:00"} <= set(command), "the allocation comes from the manifest"
    assert any(a.startswith("--job-name=hubexport-") for a in command)
    assert "pipeline_checkpoint_submit.sbatch" in command
    # A second job of the same export's name waits for the first rather than writing the same hf/.
    assert "--dependency=singleton" in command
    assert "--iteration" in command and "--hf-model" in command
    # The sbatch wrapper execs the exporter itself, so the arguments must arrive without the
    # interpreter the inline path prepends -- both callers build them from export_arguments.
    assert "bash" not in command
    assert command[command.index("pipeline_checkpoint_submit.sbatch") + 1] == "export"
    assert [c for c in hub.calls if c[0] in ("upload_folder", "upload_file")] == []


def test_submit_writes_nothing_to_the_hub_even_once_the_repositories_exist(campaign, monkeypatch):
    """The interesting case is a populated Hub, not an empty one. Cards and collection membership
    are written after the per-publication loop, from whatever that pass confirmed; a submit pass
    confirms nothing, so it must reach none of that. On an empty Hub the claim holds trivially
    because there is nothing to describe."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    wandb = RecordingWandb({"exp-mid": [{"_step": 4, "lm loss": 1.5}]})
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    assert publish_models.publish_pass(manifest, root, hub, wandb, root / "logs" / "first", True, (), "all") == 0

    before = len(hub.calls)
    submitted: list[list[str]] = []
    monkeypatch.setattr(publish_models.subprocess, "run", fake_submission("an-unrelated-job\n", submitted))
    publish_models.publish_pass(manifest, root, hub, wandb, root / "logs" / "second", True, (), "submit")
    assert hub.calls[before:] == [], "a submit pass must not upload, write a card, or touch the collection"
    assert submitted == [], "everything is already published, so there is nothing to queue either"


def test_submit_does_not_queue_an_export_that_is_already_queued(campaign, monkeypatch):
    """A watcher runs this every few minutes; without the check each pass would submit the whole
    backlog again."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    already = "\n".join(publish_models.export_job_name(p) for p in publish_models.plan(manifest)) + "\n"
    submitted: list[list[str]] = []
    monkeypatch.setattr(publish_models.subprocess, "run", fake_submission(already, submitted))
    pending = publish_models.publish_pass(
        manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / "run", True, (), "submit"
    )
    assert submitted == []
    assert pending == 6
    # A queued job may be reading its clone; rebuilding the clone would rewrite its run_config.
    assert not (root / "exports").exists(), "a queued export's clone is left alone"


def test_a_submission_creates_the_directory_its_job_writes_output_to(campaign, monkeypatch):
    """The export job's sbatch header sends its output under logs/slurm relative to the checkout it
    is submitted from, which a fresh worktree does not have; SLURM fails such a job before it starts."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    submitted: list[list[str]] = []
    monkeypatch.setattr(publish_models.subprocess, "run", fake_submission("", submitted))
    publish_models.publish_pass(
        manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / "run", True, (), "submit"
    )
    assert submitted
    assert (root / publish_models.SLURM_LOG_DIR).is_dir()


def test_rolling_submits_what_is_missing_and_uploads_only_exports_whose_job_has_finished(
    campaign, monkeypatch, caplog
):
    """The phase a run still training is polled with. An export whose job is still in the queue is
    left to that job, even if it already verifies, and is uploaded by a later pass once the job has
    left the queue; what is missing is submitted, and what is finished is uploaded."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    wandb = RecordingWandb({})
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    assert publish_models.publish_pass(manifest, root, hub, wandb, root / "logs" / "export", True, (), "export") == 6
    shutil.rmtree(root / "exports" / "arm-think" / "sft" / "iter_0000001" / "hf")
    by_label = {p.label: p for p in publish_models.plan(manifest)}
    still_running = publish_models.export_job_name(by_label["org/arm-base@midtraining_iter_4"])

    submitted: list[list[str]] = []
    monkeypatch.setattr(publish_models.subprocess, "run", fake_submission(f"{still_running}\n", submitted))
    pending = publish_models.publish_pass(manifest, root, hub, wandb, root / "logs" / "first", True, (), "rolling")
    assert pending == 2, "the missing export is queued and the running one waits"
    assert [next(a for a in c if a.startswith("--job-name=")) for c in submitted] == [
        f"--job-name={publish_models.export_job_name(by_label['org/arm-think@sft_iter_1'])}"
    ]
    uploads = {(c[1], c[2]) for c in hub.calls if c[0] == "upload_folder"}
    assert ("org/arm-base", "midtraining_iter_4") not in uploads
    assert ("org/arm-base", "main") not in uploads, "main follows the final checkpoint, which is still exporting"
    assert ("org/arm-think", "sft_iter_3") in uploads and ("org/arm-think", "main") in uploads
    assert ("upload_file", "org/arm-base", "main", "README.md") in hub.calls

    monkeypatch.setattr(
        publish_models.subprocess,
        "run",
        fake_submission(f"{publish_models.export_job_name(by_label['org/arm-think@sft_iter_1'])}\n", submitted),
    )
    pending = publish_models.publish_pass(manifest, root, hub, wandb, root / "logs" / "second", True, (), "rolling")
    assert pending == 1, "only the export still in the queue is outstanding"
    assert ("upload_folder", "org/arm-base", "midtraining_iter_4") in hub.calls
    assert ("upload_folder", "org/arm-base", "main") in hub.calls
    assert len(submitted) == 1, "an export already in the queue is not submitted again"
    assert "left the queue without finishing" not in caplog.text, "a job still in the queue has not failed"


def test_rolling_reports_an_export_job_that_ended_without_finishing_and_does_not_resubmit_it(
    campaign, monkeypatch, caplog
):
    """A job that left the queue without a complete export failed (walltime, a node fault, an
    exporter error). Resubmitting it on every poll would hide the failure behind endless retries, so
    a later pass reports it and leaves it for a person, and nothing incomplete is uploaded. Deleting
    the job's record, as the report says, is then all a resubmission takes."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    submitted: list[list[str]] = []
    monkeypatch.setattr(publish_models.subprocess, "run", fake_submission("", submitted))
    assert (
        publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "a", True, (), "rolling")
        == 6
    )
    assert len(submitted) == 6
    target = next(p for p in publish_models.plan(manifest) if p.label == "org/arm-think@sft_iter_3")
    assert target.export_job_record.read_text() == "4242\n"
    # The job wrote its tensors and then died before the exporter's last write.
    write_export(target.hf_dir, {"model-00001-of-00001.safetensors": ["w"]})
    (target.hf_dir / publish_models.EXPORT_COMPLETE_FILE).unlink()

    submitted.clear()
    pending = publish_models.publish_pass(
        manifest, root, hub, RecordingWandb({}), root / "logs" / "b", True, (), "rolling"
    )
    assert pending == 6
    assert submitted == [], "no export whose job ended without finishing is submitted again"
    assert [c for c in hub.calls if c[0] == "upload_folder"] == []
    assert "export job 4242 left the queue without finishing" in caplog.text
    failure = next(m for r in caplog.records if (m := r.getMessage()).startswith(f"{target.label}: export job"))
    assert publish_models.EXPORT_COMPLETE_FILE in failure, "the report says why the export was rejected"

    target.export_job_record.unlink()
    submitted.clear()
    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "c", True, (), "rolling")
    assert len(submitted) == 1, "the export is submitted again once its record is deleted"
    assert not target.hf_dir.exists(), "and its rejected export is removed first"


def with_upload_jobs(manifest_path: Path, walltime: str = "01:00:00") -> None:
    """Give the fixture manifest an ``upload`` block, which moves every upload into a job."""
    raw = yaml.safe_load(manifest_path.read_text())
    raw["upload"] = {"walltime": walltime}
    manifest_path.write_text(yaml.safe_dump(raw))


def as_the_upload_job(monkeypatch, hub, submission: list[str]) -> list[str]:
    """The arguments a submitted upload job's wrapper hands the publisher, so that ``main`` runs the
    job's pass in-process. The entry point's Hub and W&B clients are stood in for by ``hub`` and a
    recording W&B client, both network services."""
    monkeypatch.setattr(publish_models.sync_bucket, "make_api", lambda: hub)
    monkeypatch.setattr(publish_models, "make_wandb_api", lambda: RecordingWandb({}))
    monkeypatch.setenv("HF_HOME", "/projects/a5k/public/hf")
    return submission[submission.index(publish_models.UPLOAD_SBATCH) + 2 :]


def test_rolling_with_an_upload_block_submits_one_upload_job_for_the_manifest_and_writes_nothing_to_the_hub(
    campaign, monkeypatch, caplog
):
    """With an ``upload`` block the polling process only reads the Hub: finished exports are
    uploaded by one single-node job for the whole manifest, which runs the publisher's own upload
    phase (revisions, main, cards and collection), so no Hub transfer runs where it polls. One job
    per manifest rather than per repository: concurrent jobs would race to create the collection."""
    root, _, manifest_path = campaign
    with_upload_jobs(manifest_path)
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    wandb = RecordingWandb({})
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    assert publish_models.publish_pass(manifest, root, hub, wandb, root / "logs" / "export", True, (), "export") == 6

    submitted: list[list[str]] = []
    monkeypatch.setattr(publish_models.subprocess, "run", fake_submission("", submitted))
    pending = publish_models.publish_pass(manifest, root, hub, wandb, root / "logs" / "first", True, (), "rolling")
    assert pending == 6, "nothing is on the Hub until the upload job has run"
    assert hub.calls == [], "the polling process uploads nothing and writes no card or collection"
    names = [next(a for a in c if a.startswith("--job-name=")) for c in submitted]
    assert names == [f"--job-name={publish_models.upload_job_name(manifest)}"], "one job for the manifest"
    command = submitted[0]
    assert command[0] == "isambard_sbatch"
    assert {"--nodes=1", "--time=01:00:00"} <= set(command), "the walltime comes from the upload block"
    job_args = command[command.index(publish_models.UPLOAD_SBATCH) + 1 :]
    assert job_args[0] == sys.executable, "the job runs the publisher under the polling process's interpreter"
    assert job_args[1:] == ["--manifest", str(manifest_path.resolve()), "--repo-root", str(root), "--phase", "upload"]
    records = {p.upload_job_record.read_text() for p in publish_models.plan(manifest)}
    assert records == {"4242\n"}, "every publication the job will find verified is recorded against it"

    submitted.clear()
    queued = f"{publish_models.upload_job_name(manifest)}\n"
    monkeypatch.setattr(publish_models.subprocess, "run", fake_submission(queued, submitted))
    assert publish_models.publish_pass(manifest, root, hub, wandb, root / "logs" / "second", True, (), "rolling") == 6
    assert submitted == [], "no second upload job while the first is queued"
    # A job still waiting for a node has not failed, however long it waits, and its records stand.
    assert "left the queue without finishing" not in caplog.text
    assert {p.upload_job_record.read_text() for p in publish_models.plan(manifest)} == {"4242\n"}


def test_the_upload_jobs_command_publishes_the_manifest(campaign, monkeypatch, caplog):
    """The arguments a rolling pass gives its upload job are a valid publisher command line: run
    through the real entry point, they upload every verified export, the cards and the collection,
    and the next poll then finds everything published and reports nothing."""
    root, _, manifest_path = campaign
    with_upload_jobs(manifest_path)
    manifest = publish_models.load_manifest(manifest_path, root)
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    publish_models.publish_pass(
        manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / "x", True, (), "export"
    )
    submitted: list[list[str]] = []
    monkeypatch.setattr(publish_models.subprocess, "run", fake_submission("", submitted))
    publish_models.publish_pass(
        manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / "y", True, (), "rolling"
    )

    hub = RecordingHub()
    job_args = as_the_upload_job(monkeypatch, hub, submitted[0])
    assert publish_models.main(job_args) == 0, "every export in the manifest is published"
    uploads = {(c[1], c[2]) for c in hub.calls if c[0] == "upload_folder"}
    assert uploads == {
        ("org/arm-base", "pretraining_iter_5"),
        ("org/arm-base", "pretraining_iter_10"),
        ("org/arm-base", "midtraining_iter_2"),
        ("org/arm-base", "midtraining_iter_4"),
        ("org/arm-base", "main"),
        ("org/arm-think", "sft_iter_1"),
        ("org/arm-think", "sft_iter_3"),
        ("org/arm-think", "main"),
    }
    assert ("upload_file", "org/arm-base", "main", "README.md") in hub.calls
    assert ("upload_file", "org/arm-think", "main", "README.md") in hub.calls
    assert not any(p.upload_job_record.exists() for p in publish_models.plan(manifest)), "the job's records go"

    submitted.clear()
    caplog.clear()
    pending = publish_models.publish_pass(
        manifest, root, hub, RecordingWandb({}), root / "logs" / "z", True, (), "rolling"
    )
    assert pending == 0 and submitted == []
    assert "left the queue without finishing" not in caplog.text


class CardFailingHub(RecordingHub):
    """The recording Hub whose model-card upload fails, as a Hub outage mid-pass does."""

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, revision, commit_message):
        raise RuntimeError("503 Service Unavailable")


def test_a_card_whose_upload_failed_is_uploaded_by_the_next_attempt(campaign, tmp_path):
    """The staged copy is how an unchanged card is recognised and skipped, so it may only say what
    the Hub holds: a copy staged before a failed upload would make every retry skip the card."""
    root, _, manifest_path = campaign
    model = publish_models.load_manifest(manifest_path, root).models[0]
    staging = tmp_path / "cards"
    with pytest.raises(RuntimeError, match="503"):
        publish_models.upload_model_card(CardFailingHub(), model, "CARD TEXT", staging)
    hub = RecordingHub()
    assert publish_models.upload_model_card(hub, model, "CARD TEXT", staging), "the retry uploads the card"
    assert ("upload_file", model.repo, "main", "README.md") in hub.calls
    assert not publish_models.upload_model_card(hub, model, "CARD TEXT", staging), "and only then skips it"


def test_an_upload_job_that_fails_on_the_card_is_reported_even_with_every_revision_published(
    campaign, monkeypatch, caplog
):
    """A job can get every revision onto the Hub and then fail on the card or the collection; the
    revisions then look published, but for the final checkpoint no later job would ever re-render the
    card. The job discharges its records only after the card, so the next poll still reports it."""
    root, _, manifest_path = campaign
    with_upload_jobs(manifest_path)
    manifest = publish_models.load_manifest(manifest_path, root)
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    hub = CardFailingHub()
    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "a", True, (), "export")
    submitted: list[list[str]] = []
    monkeypatch.setattr(publish_models.subprocess, "run", fake_submission("", submitted))
    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "b", True, (), "rolling")

    job_args = as_the_upload_job(monkeypatch, hub, submitted[0])
    with pytest.raises(RuntimeError, match="503"):
        publish_models.main(job_args)
    assert ("upload_folder", "org/arm-think", "main") in hub.calls, "every revision reached the Hub first"

    submitted.clear()
    pending = publish_models.publish_pass(
        manifest, root, hub, RecordingWandb({}), root / "logs" / "c", True, (), "rolling"
    )
    assert "upload job 4242 left the queue without finishing" in caplog.text
    assert pending == 6, "the published revisions of the failed job count as outstanding until reported and cleared"
    assert submitted == []
    # The report prints a resubmission that works when pasted into a shell: the submitter needs the
    # environment the poller submits with, and the arguments are quoted. Deleting the record alone
    # would retry nothing here, since every revision is already on the Hub.
    assert "ISAMBARD_SBATCH_FORCE=1" in caplog.text and f"GEODESIC_REPO_DIR={root}" in caplog.text
    assert shlex.join(publish_models.upload_command(manifest, root)) in caplog.text
    # The command ends the line, so a copy taken to the end of the line is the command and nothing
    # after it: trailing words would reach the publisher's argument parser and fail the job.
    failure = next(r.getMessage() for r in caplog.records if "left the queue" in r.getMessage())
    assert failure.endswith(publish_models.shell_submission(publish_models.upload_command(manifest, root), root))

    healthy = RecordingHub()
    healthy.trees = dict(hub.trees)
    monkeypatch.setattr(publish_models.sync_bucket, "make_api", lambda: healthy)
    publish_models.main(job_args)
    assert ("upload_file", "org/arm-base", "main", "README.md") in healthy.calls, "the retry writes the card"
    assert [c for c in healthy.calls if c[0] == "upload_folder"] == [], "and re-uploads no revision"
    caplog.clear()
    pending = publish_models.publish_pass(
        manifest, root, healthy, RecordingWandb({}), root / "logs" / "d", True, (), "rolling"
    )
    assert pending == 0, "everything is published and the records are gone"
    assert "left the queue without finishing" not in caplog.text


def test_rolling_reports_an_upload_job_that_ended_without_publishing_and_does_not_resubmit_it(
    campaign, monkeypatch, caplog
):
    """An upload job that left the queue with its export still unpublished failed (walltime, the
    Hub, a node). As with exports, the next pass reports it instead of resubmitting it forever."""
    root, _, manifest_path = campaign
    with_upload_jobs(manifest_path)
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "a", True, (), "export")
    submitted: list[list[str]] = []
    monkeypatch.setattr(publish_models.subprocess, "run", fake_submission("", submitted))
    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "b", True, (), "rolling")
    assert len(submitted) == 1

    submitted.clear()
    pending = publish_models.publish_pass(
        manifest, root, hub, RecordingWandb({}), root / "logs" / "c", True, (), "rolling"
    )
    assert pending == 6
    assert submitted == [], "an upload job that ended without publishing is not submitted again"
    assert "upload job 4242 left the queue without finishing" in caplog.text
    assert hub.calls == []


class MainFailingHub(RecordingHub):
    """The recording Hub whose upload to main fails, after the revision's own upload succeeded."""

    def upload_folder(self, folder_path, repo_id, revision, commit_message):
        if revision == publish_models.MAIN:
            raise RuntimeError("502 Bad Gateway")
        super().upload_folder(folder_path, repo_id, revision, commit_message)


def test_a_final_checkpoint_is_published_only_once_main_holds_it_too(campaign, monkeypatch):
    """The default publication goes to its revision and to main, in that order. If main fails, the
    revision alone must not count as published, or no later pass would ever put the weights on main."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    publish_models.publish_pass(
        manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / "a", True, (), "export"
    )
    failing = MainFailingHub()
    with pytest.raises(RuntimeError, match="502"):
        publish_models.publish_pass(
            manifest, root, failing, RecordingWandb({}), root / "logs" / "b", True, (), "upload"
        )
    assert ("upload_folder", "org/arm-base", "midtraining_iter_4") in failing.calls

    healthy = RecordingHub()
    healthy.trees = dict(failing.trees)
    pending = publish_models.publish_pass(
        manifest, root, healthy, RecordingWandb({}), root / "logs" / "c", True, (), "upload"
    )
    assert ("upload_folder", "org/arm-base", "main") in healthy.calls, "the retry puts the final weights on main"
    assert pending == 0


def test_an_export_record_is_discharged_once_its_export_verifies(campaign, monkeypatch, caplog):
    """The record exists to report an export job that ended without a complete export. Once the
    export verifies it has done its work; left standing, it would report a failure the day someone
    removes the local export of a revision that is safely on the Hub."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    submitted: list[list[str]] = []
    monkeypatch.setattr(publish_models.subprocess, "run", fake_submission("", submitted))
    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "a", True, (), "rolling")
    publications = publish_models.plan(manifest)
    assert all(p.export_job_record.exists() for p in publications)
    for publication in publications:
        fake_export(publication, manifest, root, root / "export.log")

    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "b", True, (), "rolling")
    assert not any(p.export_job_record.exists() for p in publications)
    shutil.rmtree(publications[0].hf_dir)
    caplog.clear()
    submitted.clear()
    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "c", True, (), "rolling")
    assert "left the queue without finishing" not in caplog.text
    assert submitted == [], "a revision on the Hub is not exported again because its local copy is gone"


def test_a_published_revisions_stale_export_record_is_dropped_by_an_executing_pass_only(campaign, monkeypatch):
    """A record left beside a revision that is already on the Hub is dropped by a pass that
    executes. A --plan pass changes nothing, that record included."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    assert (
        publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "a", True, (), "all") == 0
    )
    target = publish_models.plan(manifest)[0]
    target.export_job_record.write_text("4242\n")

    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "b", False, (), "all")
    assert target.export_job_record.read_text() == "4242\n", "a plan pass deletes nothing"
    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "c", True, (), "all")
    assert not target.export_job_record.exists(), "an executing pass drops the settled record"


def test_an_export_clone_whose_hf_is_a_symlink_is_refused_not_cleared(campaign, monkeypatch, tmp_path, caplog):
    """A clone's hf/ is exporter output and never a link; if one is found, clearing it would follow
    the link into whatever it names. The pass refuses the publication and leaves the target alone."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    target = next(p for p in publish_models.plan(manifest) if p.label == "org/arm-think@sft_iter_3")
    publish_models.make_export_clone(target.source, target.clone)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "keep.txt").write_text("keep")
    target.hf_dir.symlink_to(elsewhere)

    def exporter_refusing_the_linked_clone(publication, manifest, repo_root, log_path):
        if publication.label == target.label:
            raise AssertionError(f"{publication.label}: exported through a symlinked hf/")
        fake_export(publication, manifest, repo_root, log_path)

    monkeypatch.setattr(publish_models, "run_export", exporter_refusing_the_linked_clone)
    pending = publish_models.publish_pass(
        manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / "a", True, ("org/arm-think",), "export"
    )
    assert pending == 2, "the refused publication counts as pending"
    assert "is a symlink" in caplog.text
    assert (elsewhere / "keep.txt").read_text() == "keep", "the link's target is untouched"


def test_a_rejected_export_is_cleared_before_it_is_exported_again(campaign, monkeypatch):
    """The exporter writes into the clone's hf/ without clearing it, shards first and its completion
    file last. A rejected export left in place would already hold a completion file while the new
    export's shards are half-written, and could be taken for finished; so it is removed first."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    target = next(p for p in publish_models.plan(manifest) if p.label == "org/arm-think@sft_iter_3")
    publish_models.make_export_clone(target.source, target.clone)
    write_export(target.hf_dir, {"model-00001-of-00001.safetensors": ["w"]})
    (target.hf_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "missing.safetensors"}})
    )
    assert not publish_models.export_is_verified(target)

    def exporter_needing_a_clean_directory(publication, manifest, repo_root, log_path):
        assert not publication.hf_dir.exists(), f"{publication.label}: the exporter is handed a stale hf/"
        fake_export(publication, manifest, repo_root, log_path)

    monkeypatch.setattr(publish_models, "run_export", exporter_needing_a_clean_directory)
    pending = publish_models.publish_pass(
        manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / "a", True, ("arm-think",), "export"
    )
    assert pending == 2 and publish_models.export_is_verified(target)


@pytest.mark.parametrize("run_has_an_export", [True, False])
def test_a_clone_that_resolves_outside_the_export_root_is_refused_not_cleared(
    campaign, monkeypatch, caplog, run_has_an_export
):
    """A pass removes a rejected export and rebuilds the clone around it, so every path it deletes or
    writes must lie inside the export root. A clone that is a link into the training run -- made by
    hand, or reached through a stage name that climbs out with ``..`` -- would otherwise lose the
    run's own hf/ export and have its run_config rewritten; the second happens whether or not the run
    has an export for the first."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    target = next(p for p in publish_models.plan(manifest) if p.label == "org/arm-think@sft_iter_3")
    if run_has_an_export:
        (target.source / "hf").mkdir()
        (target.source / "hf" / "keep.txt").write_text("keep")
    target.clone_root.mkdir(parents=True)
    target.clone.symlink_to(target.source)

    def exporter_refusing_the_linked_clone(publication, manifest, repo_root, log_path):
        if publication.label == target.label:
            raise AssertionError(f"{publication.label}: exported through a clone outside the export root")
        fake_export(publication, manifest, repo_root, log_path)

    monkeypatch.setattr(publish_models, "run_export", exporter_refusing_the_linked_clone)
    pending = publish_models.publish_pass(
        manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / "a", True, ("org/arm-think",), "export"
    )
    assert pending == 2, "the refused publication counts as pending"
    assert "outside the export root" in caplog.text
    if run_has_an_export:
        assert (target.source / "hf" / "keep.txt").read_text() == "keep", "the run's own export is untouched"
    assert (target.source / "run_config.yaml").read_text() == RAW_RUN_CONFIG, "and so is its run_config"


@pytest.mark.parametrize(
    "corruption",
    [
        "truncated index",
        "index without a weight map",
        "weight map that is not a mapping",
        "shard name that is not a string",
        "truncated shard",
        "header length beyond the file",
    ],
)
def test_a_corrupt_export_that_looks_finished_is_unverified_not_fatal(campaign, corruption, caplog):
    """An export can hold its completion file and still be unreadable: a second writer rewriting it in
    place, a quota hit mid-write. It is an export that does not verify, which a pass exports again;
    a parser's exception must not end the pass, or the plan, that meets it."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    target = next(p for p in publish_models.plan(manifest) if p.label == "org/arm-think@sft_iter_3")
    shard = "model-00001-of-00001.safetensors"
    write_export(target.hf_dir, {shard: ["w"]})
    index = target.hf_dir / publish_models.INDEX_FILE
    if corruption == "truncated index":
        index.write_text(index.read_text()[:10])
    elif corruption == "index without a weight map":
        index.write_text("{}")
    elif corruption == "weight map that is not a mapping":
        index.write_text(json.dumps({"weight_map": ["w"]}))
    elif corruption == "shard name that is not a string":
        index.write_text(json.dumps({"weight_map": {"w": [shard]}}))
    elif corruption == "truncated shard":
        (target.hf_dir / shard).write_bytes((target.hf_dir / shard).read_bytes()[:3])
    else:
        (target.hf_dir / shard).write_bytes(struct.pack("<Q", 2**62) + b"{}")
    with pytest.raises(publish_models.ExportError):
        publish_models.verify_export(target.hf_dir)
    assert not publish_models.export_is_verified(target)
    pending = publish_models.publish_pass(
        manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / "plan", False, (), "all"
    )
    assert pending == 6
    # The upload phase, which the upload job runs, leaves it for an export pass and says why.
    caplog.set_level(logging.INFO, logger="publish_models")
    publish_models.publish_pass(
        manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / "upload", True, (), "upload"
    )
    assert f"{target.label}: no verified export ({target.hf_dir}" in caplog.text


def test_an_export_that_cannot_be_read_raises_rather_than_being_judged_corrupt(campaign):
    """A failure to read the files at all (a stale handle, an I/O error, a permission) says nothing
    about the export. Judging it corrupt would remove a possibly valid export and queue a GPU job to
    replace it, so the error is raised as it is."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    target = next(p for p in publish_models.plan(manifest) if p.label == "org/arm-think@sft_iter_3")
    write_export(target.hf_dir, {"model-00001-of-00001.safetensors": ["w"]})
    index = target.hf_dir / publish_models.INDEX_FILE
    index.chmod(0)
    try:
        with pytest.raises(PermissionError):
            publish_models.verify_export(target.hf_dir)
    finally:
        index.chmod(0o644)


def test_without_its_local_export_a_revision_counts_only_when_every_target_holds_a_finished_export(
    campaign, monkeypatch
):
    """Once the local export is gone, the Hub's copy is judged on its own, and every revision the
    publication targets -- main as well, for the final checkpoint -- must hold the index and the
    exporter's last write. A revision that exists holding anything less is not published."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    assert (
        publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "a", True, (), "all") == 0
    )
    target = next(p for p in publish_models.plan(manifest) if p.label == "org/arm-base@midtraining_iter_4")
    assert target.default
    shutil.rmtree(target.hf_dir)
    own, main = (target.model.repo, target.revision), (target.model.repo, publish_models.MAIN)
    finished = dict(hub.trees[main])
    assert publish_models.on_the_hub(hub, target)

    hub.trees[main] = {publish_models.README_NAME: 100}
    assert not publish_models.on_the_hub(hub, target), "main holds only the card"
    hub.trees[main] = {k: v for k, v in finished.items() if k != publish_models.EXPORT_COMPLETE_FILE}
    assert not publish_models.on_the_hub(hub, target), "main lacks the exporter's last write"
    del hub.trees[main]
    assert not publish_models.on_the_hub(hub, target), "main does not exist"
    hub.trees[main] = finished
    hub.trees[own] = {k: v for k, v in finished.items() if k != publish_models.INDEX_FILE}
    assert not publish_models.on_the_hub(hub, target), "its own revision lacks the index"


def test_every_pass_leaves_an_export_whose_job_is_still_in_the_queue_to_that_job(campaign, monkeypatch):
    """A job still in the queue writes its export when it starts, over whatever the clone holds. So no
    pass uploads that export or rebuilds that clone meanwhile: not the upload job a rolling pass
    submits, which runs the upload phase, and not a pass run by hand."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    publish_models.publish_pass(
        manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / "a", True, (), "export"
    )
    by_label = {p.label: p for p in publish_models.plan(manifest)}
    verified = by_label["org/arm-base@midtraining_iter_4"]
    unexported = by_label["org/arm-think@sft_iter_1"]
    shutil.rmtree(unexported.hf_dir)
    queued = f"{publish_models.export_job_name(verified)}\n{publish_models.export_job_name(unexported)}\n"
    monkeypatch.setattr(publish_models.subprocess, "run", fake_submission(queued, []))
    verified.export_job_record.write_text("7777\n")

    hub = RecordingHub()
    pending = publish_models.publish_pass(
        manifest, root, hub, RecordingWandb({}), root / "logs" / "b", True, (), "upload"
    )
    assert pending == 2
    assert ("upload_folder", "org/arm-base", "midtraining_iter_4") not in hub.calls
    assert ("upload_folder", "org/arm-base", "main") not in hub.calls
    assert ("upload_folder", "org/arm-base", "pretraining_iter_5") in hub.calls
    assert verified.export_job_record.read_text() == "7777\n", "a job still queued has settled nothing"

    def exporter_refusing_the_queued_export(publication, manifest, repo_root, log_path):
        if publication.label == unexported.label:
            raise AssertionError(f"{publication.label}: exported over the clone of a queued job")
        fake_export(publication, manifest, repo_root, log_path)

    monkeypatch.setattr(publish_models, "run_export", exporter_refusing_the_queued_export)
    pending = publish_models.publish_pass(
        manifest, root, hub, RecordingWandb({}), root / "logs" / "c", True, (), "all"
    )
    assert pending == 2
    assert not unexported.hf_dir.exists()


def test_an_upload_job_is_submitted_at_most_once_a_pass_and_as_a_singleton(campaign, monkeypatch, caplog):
    """A failed submission leaves the pass without a job id, and each later publication would submit
    again; a submission that timed out after registering would leave several jobs racing to create
    the collection. So one failure ends the pass's attempts, and the job is a SLURM singleton, which
    holds any second job of its name -- a resubmission pasted twice included -- until the first has
    left the queue."""
    root, _, manifest_path = campaign
    with_upload_jobs(manifest_path)
    manifest = publish_models.load_manifest(manifest_path, root)
    assert "--dependency=singleton" in publish_models.upload_command(manifest, root)
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    publish_models.publish_pass(
        manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / "a", True, (), "export"
    )
    attempts: list[list[str]] = []

    def timed_out_submission(command, **kwargs):
        """Stands in for the scheduler, as fake_submission does, with an sbatch that fails the way a
        busy controller makes it fail."""
        del kwargs
        if command[0] == "squeue":
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        attempts.append(list(command))
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="sbatch: error: Socket timed out")

    monkeypatch.setattr(publish_models.subprocess, "run", timed_out_submission)
    pending = publish_models.publish_pass(
        manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / "b", True, (), "rolling"
    )
    assert pending == 6
    assert len(attempts) == 1, "one attempt a pass, however many publications wait for the job"
    assert "Socket timed out" in caplog.text


def test_a_revision_whose_local_export_was_removed_stays_published(campaign, monkeypatch):
    """An upload is one commit carrying the whole export, the exporter's last write included, so a
    revision holding that file is published whether or not its local export is still on disk.
    Removing local exports to free space must neither queue an export per revision nor drop the
    revision from the card."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    wandb = RecordingWandb({"exp-mid": [{"_step": 4, "lm loss": 1.5}]})
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    assert publish_models.publish_pass(manifest, root, hub, wandb, root / "logs" / "a", True, (), "all") == 0
    target = next(p for p in publish_models.plan(manifest) if p.label == "org/arm-base@midtraining_iter_4")
    shutil.rmtree(target.hf_dir)
    card_dir = root / "logs" / "cards" / "arm-base"
    shutil.rmtree(card_dir)

    def no_export(publication, manifest, repo_root, log_path):
        raise AssertionError(f"{publication.label} is on the Hub and was exported again")

    monkeypatch.setattr(publish_models, "run_export", no_export)
    assert publish_models.publish_pass(manifest, root, hub, wandb, root / "logs" / "b", True, (), "all") == 0
    assert "| `midtraining_iter_4` (also `main`) |" in (card_dir / "README.md").read_text()


def test_an_upload_job_takes_over_the_records_of_the_job_before_it(campaign, monkeypatch, caplog):
    """A failed upload job is resubmitted by hand; if the new job fails too, the report must name it
    and its log, not the job before. So the upload job writes its own id into every record it
    inherits as it starts."""
    root, _, manifest_path = campaign
    with_upload_jobs(manifest_path)
    manifest = publish_models.load_manifest(manifest_path, root)
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    hub = CardFailingHub()
    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "a", True, (), "export")
    submitted: list[list[str]] = []
    monkeypatch.setattr(publish_models.subprocess, "run", fake_submission("", submitted))
    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "b", True, (), "rolling")
    assert {p.upload_job_record.read_text() for p in publish_models.plan(manifest)} == {"4242\n"}
    unrecorded, *recorded = publish_models.plan(manifest)
    unrecorded.upload_job_record.unlink()

    job_args = as_the_upload_job(monkeypatch, hub, submitted[0])
    monkeypatch.setenv("SLURM_JOB_NAME", publish_models.upload_job_name(manifest))
    monkeypatch.setenv("SLURM_JOB_ID", "5555")
    # Only the upload phase is the upload job's: any other phase run under its name claims nothing.
    publish_models.main([*job_args[:-1], "export"])
    assert {p.upload_job_record.read_text() for p in recorded} == {"4242\n"}
    # Nor is an upload pass run by hand inside some other allocation: its job id is not the upload
    # job's, and a later report naming it would send a person to the wrong log.
    monkeypatch.setenv("SLURM_JOB_NAME", "an-interactive-allocation")
    with pytest.raises(RuntimeError, match="503"):
        publish_models.main(job_args)
    assert {p.upload_job_record.read_text() for p in recorded} == {"4242\n"}
    monkeypatch.setenv("SLURM_JOB_NAME", publish_models.upload_job_name(manifest))
    with pytest.raises(RuntimeError, match="503"):
        publish_models.main(job_args)
    assert {p.upload_job_record.read_text() for p in recorded} == {"5555\n"}
    assert not unrecorded.upload_job_record.exists(), "it takes over records, and creates none"

    monkeypatch.delenv("SLURM_JOB_NAME")
    monkeypatch.delenv("SLURM_JOB_ID")
    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "c", True, (), "rolling")
    assert "upload job 5555 left the queue without finishing" in caplog.text
    assert publish_models.UPLOAD_JOB_LOG.format(job="5555") in caplog.text


class RevisionFailingHub(RecordingHub):
    """The recording Hub whose upload to one revision fails, as a 5xx or a walltime mid-transfer does."""

    def __init__(self, failing: tuple[str, str]):
        super().__init__()
        self.failing = failing

    def upload_folder(self, folder_path, repo_id, revision, commit_message):
        if (repo_id, revision) == self.failing:
            self.calls.append(("upload_folder_failed", repo_id, revision))
            raise RuntimeError("504 Gateway Timeout")
        super().upload_folder(folder_path, repo_id, revision, commit_message)


def test_a_revision_whose_upload_failed_is_retried_after_main_holds_the_final_export(campaign, monkeypatch):
    """Every export of one architecture has the same file names and sizes, and a Hub branch made
    without a starting point is a copy of main. Once main holds the final export, a branch made for an
    older checkpoint whose upload then failed would already hold every file by name and size -- the
    final checkpoint's weights under the older revision's name -- and count as published. Each
    revision is branched from the repository's first commit instead, so a failed upload leaves it
    without the export and the next pass retries it."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    publish_models.publish_pass(
        manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / "a", True, (), "export"
    )
    failing = RevisionFailingHub(("org/arm-think", "sft_iter_1"))
    with pytest.raises(RuntimeError, match="504"):
        # Newest first: the final checkpoint reaches main before the older revision is attempted.
        publish_models.publish_pass(
            manifest, root, failing, RecordingWandb({}), root / "logs" / "b", True, ("arm-think",), "upload", True
        )
    assert ("upload_folder", "org/arm-think", publish_models.MAIN) in failing.calls
    assert ("upload_folder_failed", "org/arm-think", "sft_iter_1") in failing.calls

    healthy = RecordingHub()
    healthy.trees = dict(failing.trees)
    pending = publish_models.publish_pass(
        manifest, root, healthy, RecordingWandb({}), root / "logs" / "c", True, ("arm-think",), "upload"
    )
    assert pending == 0
    assert ("upload_folder", "org/arm-think", "sft_iter_1") in healthy.calls, "the failed upload is retried"


def test_main_must_hold_every_file_of_the_final_export_not_only_its_weights(campaign, monkeypatch):
    """main is compared with the local export file by file, like the final's own revision: holding the
    final checkpoint's shards is not enough while its config is missing there."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "a", True, (), "all")
    target = next(p for p in publish_models.plan(manifest) if p.label == "org/arm-base@midtraining_iter_4")
    assert publish_models.on_the_hub(hub, target)
    del hub.trees[(target.model.repo, publish_models.MAIN)]["config.json"]
    assert not publish_models.on_the_hub(hub, target)


def test_an_emptied_local_export_is_judged_by_the_hub_alone(campaign, monkeypatch):
    """Clearing an export's files by hand leaves hf/ standing (its .cache/ survives a glob). Compared
    file by file with nothing, any revision that merely exists would pass for published, main
    included; an export that does not verify is therefore no reference, and the Hub's copy is
    judged on its own."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "a", True, (), "all")
    target = next(p for p in publish_models.plan(manifest) if p.label == "org/arm-base@midtraining_iter_4")
    hub.trees[(target.model.repo, publish_models.MAIN)] = {publish_models.README_NAME: 100}
    assert not publish_models.on_the_hub(hub, target), "main lost the export"
    for path in target.hf_dir.iterdir():
        path.unlink()
    (target.hf_dir / ".cache" / "huggingface").mkdir(parents=True)
    assert not publish_models.on_the_hub(hub, target), "and emptying the local export does not hide it"


def test_an_unfinished_local_export_is_not_reported_as_a_corrupt_copy(campaign, monkeypatch, caplog):
    """A local export without the exporter's last write is one being written, or one that never
    finished, not a copy of the Hub's to compare with: the Hub's copy is judged on its own, and
    nothing warns that the local one disagrees with it."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "a", True, (), "all")
    target = next(p for p in publish_models.plan(manifest) if p.label == "org/arm-think@sft_iter_1")
    (target.hf_dir / publish_models.EXPORT_COMPLETE_FILE).unlink()
    shard = target.hf_dir / "model-00001-of-00001.safetensors"
    os.truncate(shard, shard.stat().st_size - 1)
    with caplog.at_level(logging.WARNING, logger=publish_models.LOGGER.name):
        assert publish_models.on_the_hub(hub, target)
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_an_export_root_reached_through_a_link_contains_its_clones(campaign, monkeypatch, tmp_path, caplog):
    """On this cluster /projects is a link, so an export root and the clones under it resolve
    somewhere else. A clone lies inside its export root however either path is spelled, and is
    rebuilt rather than refused."""
    root, _, manifest_path = campaign
    real = tmp_path / "real_exports"
    real.mkdir()
    linked = tmp_path / "linked_exports"
    linked.symlink_to(real)
    raw = yaml.safe_load(manifest_path.read_text())
    raw["export_root"] = str(linked)
    manifest_path.write_text(yaml.safe_dump(raw))
    manifest = publish_models.load_manifest(manifest_path, root)
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    pending = publish_models.publish_pass(
        manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / "a", True, (), "export"
    )
    assert pending == 6
    assert "outside the export root" not in caplog.text
    assert all(publish_models.export_is_verified(p) for p in publish_models.plan(manifest))


def test_a_clone_whose_run_config_is_a_link_is_refused_before_anything_is_removed(campaign, monkeypatch, caplog):
    """The clone's run_config is a patched copy; one that is a link (a clone made by hand with
    ``cp -as``) would carry the patch through the link into the training run's own run_config. Such
    a clone is refused before the pass touches it, its unverified export included."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    target = next(p for p in publish_models.plan(manifest) if p.label == "org/arm-think@sft_iter_3")
    target.clone.mkdir(parents=True)
    for entry in target.source.iterdir():
        (target.clone / entry.name).symlink_to(entry)
    target.hf_dir.mkdir()
    (target.hf_dir / "model-00001-of-00001.safetensors").write_text("partial")
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    publish_models.publish_pass(
        manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / "a", True, ("arm-think",), "export"
    )
    assert "is a link" in caplog.text
    assert "removing the unverified export" not in caplog.text
    assert (target.hf_dir / "model-00001-of-00001.safetensors").read_text() == "partial", "nothing was removed"
    assert (target.source / "run_config.yaml").read_text() == RAW_RUN_CONFIG, "the run's run_config is untouched"


@pytest.mark.parametrize("loss", ["removed", "no longer verifies"])
def test_an_upload_record_whose_export_is_gone_is_dropped_and_the_export_redone(campaign, monkeypatch, caplog, loss):
    """An upload job answers for the exports that were verified when it was submitted. One that has
    since disappeared, or stopped verifying, is not the job's failure, and no resubmission of the job
    could clear its record; the job drops the record, and the next rolling pass exports it again."""
    root, _, manifest_path = campaign
    with_upload_jobs(manifest_path)
    manifest = publish_models.load_manifest(manifest_path, root)
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    publish_models.publish_pass(
        manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / "a", True, (), "export"
    )
    submitted: list[list[str]] = []
    monkeypatch.setattr(publish_models.subprocess, "run", fake_submission("", submitted))
    publish_models.publish_pass(
        manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / "b", True, (), "rolling"
    )
    target = next(p for p in publish_models.plan(manifest) if p.label == "org/arm-think@sft_iter_3")
    if loss == "removed":
        shutil.rmtree(target.hf_dir)
    else:
        (target.hf_dir / publish_models.EXPORT_COMPLETE_FILE).unlink()

    hub = RecordingHub()
    job_args = as_the_upload_job(monkeypatch, hub, submitted[0])
    publish_models.main(job_args)
    assert not target.upload_job_record.exists()

    submitted.clear()
    caplog.clear()
    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "c", True, (), "rolling")
    assert "left the queue without finishing" not in caplog.text
    names = [next(a for a in c if a.startswith("--job-name=")) for c in submitted]
    assert f"--job-name={publish_models.export_job_name(target)}" in names, "the vanished export is redone"


def test_an_unreadable_file_in_a_published_export_does_not_stop_a_pass(campaign, monkeypatch):
    """Whether a revision is on the Hub is decided from the local export's file names and sizes, which
    reads none of its contents. A file that can no longer be read in the export of a revision long
    since published is no reason for every pass, a plan included, to stop."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    assert (
        publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "a", True, (), "all") == 0
    )
    shard = next(p for p in publish_models.plan(manifest)).hf_dir / "model-00001-of-00001.safetensors"
    shard.chmod(0)
    try:
        assert (
            publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "b", False, (), "all")
            == 0
        )
    finally:
        shard.chmod(0o644)


def test_an_unreadable_unpublished_export_is_raised_not_replaced(campaign, monkeypatch):
    """An export that cannot be read has not been shown to be wrong. Replacing it would remove a
    possibly valid export and spend a GPU job on a copy, so the read error ends the pass and the
    export stays as it is."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    target = next(p for p in publish_models.plan(manifest) if p.label == "org/arm-think@sft_iter_3")
    publish_models.make_export_clone(target.source, target.clone)
    write_export(target.hf_dir, {"model-00001-of-00001.safetensors": ["w"]})
    index = target.hf_dir / publish_models.INDEX_FILE
    index.chmod(0)
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    try:
        with pytest.raises(PermissionError):
            publish_models.publish_pass(
                manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / "a", True, ("arm-think",), "export"
            )
    finally:
        index.chmod(0o644)
    assert publish_models.export_is_verified(target), "the export is left exactly as it was"


def test_a_local_export_that_differs_from_the_hubs_copy_is_uploaded_again(campaign, monkeypatch):
    """With a finished export on disk, the Hub's copy must match it file for file; a finished export on
    the Hub is not enough when it is a different one (an export redone after its upload)."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "a", True, (), "all")
    target = next(p for p in publish_models.plan(manifest) if p.label == "org/arm-think@sft_iter_1")
    write_export(target.hf_dir, {"model-00001-of-00001.safetensors": ["w", "v"]})
    assert publish_models.export_is_verified(target)
    assert not publish_models.on_the_hub(hub, target)


def test_a_first_commit_that_holds_an_export_is_refused_as_a_branch_point(campaign, monkeypatch, caplog):
    """Revisions branch from the repository's first commit because it holds no export. A history
    squashed into one commit makes that commit a copy of main, and a branch made from it would pass for
    published with main's weights if its own upload failed; so no revision is branched from it."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "a", True, (), "all")
    repo = "org/arm-think"
    hub.trees[(repo, RecordingHub.INITIAL_COMMIT)] = dict(hub.trees[(repo, publish_models.MAIN)])
    del hub.trees[(repo, "sft_iter_1")]
    calls_before = len(hub.calls)
    pending = publish_models.publish_pass(
        manifest, root, hub, RecordingWandb({}), root / "logs" / "b", True, ("arm-think",), "upload"
    )
    assert pending == 1
    assert "first commit" in caplog.text
    assert ("create_branch", repo, "sft_iter_1") not in hub.calls[calls_before:]


def test_a_published_revision_whose_local_copy_was_corrupted_stays_published(campaign, monkeypatch, caplog):
    """A local export that no longer matches the Hub by size is compared no further once it fails to
    verify: a corrupt local copy says nothing about the Hub's, which is then judged on its own. The
    revision is neither exported again nor dropped from the card, and the corruption is reported."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    assert (
        publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "a", True, (), "all") == 0
    )
    target = next(p for p in publish_models.plan(manifest) if p.label == "org/arm-think@sft_iter_1")
    shard = target.hf_dir / "model-00001-of-00001.safetensors"
    shard.write_bytes(shard.read_bytes()[:3])
    assert publish_models.on_the_hub(hub, target)
    warning = next(r for r in caplog.records if r.levelname == "WARNING" and target.label in r.getMessage())
    assert "does not verify" in warning.getMessage(), "the corrupt local copy is reported, with its reason"


def test_main_counts_only_once_it_holds_the_final_checkpoints_own_weights(campaign, monkeypatch):
    """Every export of one architecture has the same file names and sizes, so main holding a
    different export (an earlier final, before a run was extended; a later one, after it was cut
    short) would pass for the final checkpoint's by name and size. main is compared with the final's
    own revision by the content hashes of its weights."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "a", True, (), "all")
    target = next(p for p in publish_models.plan(manifest) if p.label == "org/arm-base@midtraining_iter_4")
    main = (target.model.repo, publish_models.MAIN)
    assert publish_models.on_the_hub(hub, target)
    hub.trees[main] = {
        name: (held[0], "0" * 64) if isinstance(held, tuple) else held for name, held in hub.trees[main].items()
    }
    assert not publish_models.on_the_hub(hub, target), "main holds other weights of the same names and sizes"


@pytest.mark.parametrize("gap", ["a shard without a content hash", "no shard at all"])
def test_weights_that_cannot_be_compared_by_content_are_an_error_not_unpublished(campaign, monkeypatch, gap):
    """main is judged by the content hashes of the final's own revision. A revision whose listing
    gives none to compare -- a shard without its LFS hash, or no shard -- cannot be judged, which is
    an error rather than a finding that the revision is unpublished (that would upload it again on
    every pass)."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "a", True, (), "all")
    target = next(p for p in publish_models.plan(manifest) if p.label == "org/arm-base@midtraining_iter_4")
    own = hub.trees[(target.model.repo, target.revision)]
    shard = "model-00001-of-00001.safetensors"
    if gap == "a shard without a content hash":
        own[shard] = own[shard][0]
    else:
        del own[shard]
    shutil.rmtree(target.hf_dir)
    with pytest.raises(publish_models.ExportError, match="content hash"):
        publish_models.on_the_hub(hub, target)


def test_stale_inline_export_records_are_discharged_once_the_export_is_settled(campaign, monkeypatch):
    """A record an inline export left behind stops mattering once its export is settled: verified with
    no job in the queue, or already on the Hub. Either way a pass that acts removes it."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "a", True, (), "export")
    verified = next(p for p in publish_models.plan(manifest) if p.label == "org/arm-think@sft_iter_1")
    verified.inline_export_record.write_text("elsewhere pid 1 since then\n")
    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "b", True, (), "upload")
    assert not verified.inline_export_record.exists(), "settled by its verified export"
    verified.inline_export_record.write_text("elsewhere pid 1 since then\n")
    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "c", True, (), "upload")
    assert not verified.inline_export_record.exists(), "settled by its revision being on the Hub"


def test_an_identity_whose_own_namespace_is_the_manifests_is_accepted(campaign, monkeypatch):
    """A repository in the token owner's own namespace is reachable without any organisation."""
    root, _, manifest_path = campaign

    class OwnerHub(RecordingHub):
        IDENTITY = {"name": "org", "orgs": []}

    monkeypatch.setattr(publish_models.sync_bucket, "make_api", OwnerHub)
    monkeypatch.setenv("HF_HOME", "/projects/a5k/public/hf")
    assert publish_models.main(["--manifest", str(manifest_path), "--repo-root", str(root), "--plan"]) == 1


def test_a_token_that_cannot_see_the_manifests_namespace_is_refused(campaign, monkeypatch):
    """The Hub answers a repository the token cannot see as a missing one, so without access every
    revision would read as unpublished and, where local exports were removed, be exported again. The
    entry point therefore refuses to act for an identity outside the manifest's namespaces."""
    root, _, manifest_path = campaign

    class OutsiderHub(RecordingHub):
        IDENTITY = {"name": "someone", "orgs": [{"name": "elsewhere"}]}

    monkeypatch.setattr(publish_models.sync_bucket, "make_api", OutsiderHub)
    monkeypatch.setenv("HF_HOME", "/projects/a5k/public/hf")
    with pytest.raises(publish_models.ExportError, match="'org'"):
        publish_models.main(["--manifest", str(manifest_path), "--repo-root", str(root), "--plan"])


def test_every_namespace_of_the_manifest_must_be_reachable(campaign, monkeypatch):
    """Reaching one of a manifest's namespaces is not enough: a model in another would read as
    unpublished all the same."""
    root, _, manifest_path = campaign
    raw = yaml.safe_load(manifest_path.read_text())
    raw["models"][1]["repo"] = "elsewhere/arm-think"
    manifest_path.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(publish_models.sync_bucket, "make_api", RecordingHub)
    monkeypatch.setenv("HF_HOME", "/projects/a5k/public/hf")
    with pytest.raises(publish_models.ExportError, match="'elsewhere'"):
        publish_models.main(["--manifest", str(manifest_path), "--repo-root", str(root), "--plan"])


def test_a_folder_on_a_revision_is_not_a_file_to_compare(campaign, monkeypatch):
    """A revision's top-level listing can hold folders, which have no size; only files are compared."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "a", True, (), "all")
    for key in list(hub.trees):
        hub.trees[key]["examples"] = None
    assert (
        publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "b", False, (), "all")
        == 0
    )


@pytest.mark.parametrize("change", [-1, 1], ids=["cut short", "bytes past its last tensor"])
def test_a_shard_whose_size_disagrees_with_its_header_does_not_verify(tmp_path, change):
    """A shard's header places every tensor in the file, so the file must end where its last tensor
    does. One cut short (a write that did not finish, a copy that stopped) or carrying bytes past it
    is corrupt however intact its header, and the export does not verify."""
    hf_dir = tmp_path / "hf"
    write_export(hf_dir, {"model-00001-of-00001.safetensors": ["a", "b"]})
    shard = hf_dir / "model-00001-of-00001.safetensors"
    os.truncate(shard, shard.stat().st_size + change)
    with pytest.raises(publish_models.ExportError, match="bytes"):
        publish_models.verify_export(hf_dir)


def test_an_upload_job_keeps_the_records_of_what_it_failed_to_publish(campaign, monkeypatch):
    """A pass discharges the upload records of what it confirmed and only that: a publication it
    failed on keeps its record, so the next poll reports it instead of taking it for done."""
    root, _, manifest_path = campaign
    with_upload_jobs(manifest_path)
    manifest = publish_models.load_manifest(manifest_path, root)
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    publish_models.publish_pass(
        manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / "a", True, (), "export"
    )
    submitted: list[list[str]] = []
    monkeypatch.setattr(publish_models.subprocess, "run", fake_submission("", submitted))
    publish_models.publish_pass(
        manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / "b", True, (), "rolling"
    )
    hub = RecordingHub()
    hub.create_repo("org/arm-think", private=True, exist_ok=False)
    # The think repository's first commit holds an export, so none of its revisions can be branched.
    hub.trees[("org/arm-think", RecordingHub.INITIAL_COMMIT)] = {
        publish_models.INDEX_FILE: 1,
        publish_models.EXPORT_COMPLETE_FILE: 1,
    }
    job_args = as_the_upload_job(monkeypatch, hub, submitted[0])
    assert publish_models.main(job_args) == 1
    records = {p.label: p.upload_job_record.exists() for p in publish_models.plan(manifest)}
    assert not any(kept for label, kept in records.items() if label.startswith("org/arm-base")), "confirmed"
    assert all(kept for label, kept in records.items() if label.startswith("org/arm-think")), "failed, kept"


def test_a_stale_extra_shard_on_main_does_not_unpublish_the_final_checkpoint(campaign, monkeypatch):
    """An upload adds and replaces files but never deletes, so main can keep a shard that an earlier
    export had and the final's lacks. main holds the final's weights when every shard of the final's
    own revision is on main with the same content; an extra one does not undo that."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "a", True, (), "all")
    target = next(p for p in publish_models.plan(manifest) if p.label == "org/arm-base@midtraining_iter_4")
    hub.trees[(target.model.repo, publish_models.MAIN)]["model-00002-of-00002.safetensors"] = (4, "f" * 64)
    assert publish_models.on_the_hub(hub, target)


@pytest.mark.parametrize("inline_phase", ["export", "all"])
def test_an_inline_export_in_progress_is_left_alone_by_a_concurrent_rolling_pass(campaign, monkeypatch, inline_phase):
    """An inline export (the export or all phase) has no job in the queue to announce it. It records
    itself as its export's job before it starts, so a rolling pass meeting the half-written export
    reports it instead of taking it for a failed one, removing it and exporting again."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    target = next(p for p in publish_models.plan(manifest) if p.label == "org/arm-think@sft_iter_3")
    observed: dict[str, bool] = {}

    def exporter_racing_a_poller(publication, manifest, repo_root, log_path):
        if publication.label == target.label:
            # The shards are written; the exporter's last write is not.
            write_export(publication.hf_dir, {"model-00001-of-00001.safetensors": ["w"]})
            (publication.hf_dir / publish_models.EXPORT_COMPLETE_FILE).unlink()
            monkeypatch.setattr(publish_models.subprocess, "run", fake_submission("", []))
            publish_models.publish_pass(
                manifest,
                root,
                RecordingHub(),
                RecordingWandb({}),
                root / "logs" / "poll",
                True,
                ("arm-think",),
                "rolling",
            )
            observed["survived"] = (publication.hf_dir / "model-00001-of-00001.safetensors").is_file()
        fake_export(publication, manifest, repo_root, log_path)

    monkeypatch.setattr(publish_models, "run_export", exporter_racing_a_poller)
    publish_models.publish_pass(
        manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / "a", True, ("arm-think",), inline_phase
    )
    assert observed == {"survived": True}, "the concurrent pass left the export being written alone"
    assert publish_models.export_is_verified(target)
    assert not target.inline_export_record.exists(), "the finished inline export discharged its own record"


def test_an_inline_export_records_itself_before_it_removes_the_rejected_export(campaign, monkeypatch):
    """A concurrent pass must see an inline export from the moment it starts clearing the clone, not
    only once the clone is rebuilt: before that, the rejected export is already half removed and
    nothing says who is removing it."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    target = next(p for p in publish_models.plan(manifest) if p.label == "org/arm-think@sft_iter_3")
    target.hf_dir.mkdir(parents=True)
    (target.hf_dir / "model-00001-of-00001.safetensors").write_bytes(b"partial")
    recorded_at_removal: list[bool] = []
    real_rmtree = shutil.rmtree

    def observed_rmtree(path, *args, **kwargs):
        # Observes the real removal; the call still goes through to shutil.
        if Path(path) == target.hf_dir:
            recorded_at_removal.append(target.inline_export_record.is_file())
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(publish_models.shutil, "rmtree", observed_rmtree)
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    publish_models.publish_pass(
        manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / "a", True, ("arm-think",), "export"
    )
    assert recorded_at_removal == [True]
    assert publish_models.export_is_verified(target)


def test_an_inline_pass_leaves_an_export_queued_after_it_started_to_its_job(campaign, monkeypatch, caplog):
    """An inline pass spends minutes on each export, and a polling process may meanwhile queue export
    jobs for the publications it has not reached. When it reaches one, the queue it read at its
    start says nothing of that job; read then, it would remove the export the job is writing, export
    over it and drop the job's record. It reads the queue again, and leaves export and record to the
    job."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    victim = next(p for p in publish_models.plan(manifest) if p.label == "org/arm-think@sft_iter_3")
    queue: list[str] = []
    inline_exports: list[str] = []

    def scheduler(command, **kwargs):
        # Stands in for squeue and sbatch (cluster services): a submission joins the queue that the
        # next squeue reports, as it does on the cluster.
        del kwargs
        if command[0] == "squeue":
            return subprocess.CompletedProcess(command, 0, stdout="".join(f"{n}\n" for n in queue), stderr="")
        queue.append(next(a for a in command if a.startswith("--job-name=")).split("=", 1)[1])
        return subprocess.CompletedProcess(command, 0, stdout="Submitted batch job 4242\n", stderr="")

    def exporter_outlasting_a_poll(publication, manifest_, repo_root, log_path):
        inline_exports.append(publication.label)
        if len(inline_exports) == 1:
            publish_models.publish_pass(
                manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / "poll", True, (), "rolling"
            )
            # The victim's job has started: its shards are written, its last file is not.
            write_export(victim.hf_dir, {"model-00001-of-00001.safetensors": ["w0003"]})
            (victim.hf_dir / publish_models.EXPORT_COMPLETE_FILE).unlink()
        fake_export(publication, manifest_, repo_root, log_path)

    monkeypatch.setattr(publish_models.subprocess, "run", scheduler)
    monkeypatch.setattr(publish_models, "run_export", exporter_outlasting_a_poll)
    publish_models.publish_pass(
        manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / "inline", True, (), "export"
    )
    assert publish_models.export_job_name(victim) in queue
    assert victim.label not in inline_exports
    assert f"{victim.label}: removing the unverified export" not in caplog.text
    assert (victim.hf_dir / "model-00001-of-00001.safetensors").is_file(), "the job's output is left to it"
    assert victim.export_job_record.is_file(), "and so is its record"


def test_a_clone_that_cannot_be_built_leaves_no_inline_record_to_hide_the_cause(campaign, monkeypatch, caplog):
    """A clone build that fails runs no exporter, so a record left behind would tell every later pass
    that an export may still be writing when nothing is, and they would report that instead of the
    cause. The pass removes its own record as it fails, and the next pass reports the cause again."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    target = next(p for p in publish_models.plan(manifest) if p.label == "org/arm-think@sft_iter_3")
    (target.source / "run_config.yaml").write_text("model:\n  unpatchable: 1\n")
    exported: list[str] = []

    def recording_export(publication, manifest_, repo_root, log_path):
        exported.append(publication.label)
        fake_export(publication, manifest_, repo_root, log_path)

    monkeypatch.setattr(publish_models, "run_export", recording_export)
    for attempt in ("a", "b"):
        caplog.clear()
        publish_models.publish_pass(
            manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / attempt, True, ("arm-think",), "export"
        )
        errors = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR and target.label in r.getMessage()
        ]
        assert len(errors) == 1 and "occurrences" in errors[0], errors
        assert not target.inline_export_record.exists()
    assert target.label not in exported


def test_a_pass_that_confirms_nothing_writes_no_collection(campaign, monkeypatch):
    """A rolling pass whose exports are all only queued has confirmed nothing on the Hub, so it writes
    nothing there, a collection included: an empty one would announce models that do not exist yet."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    submitted: list[list[str]] = []
    monkeypatch.setattr(publish_models.subprocess, "run", fake_submission("", submitted))
    assert (
        publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "a", True, (), "rolling")
        == 6
    )
    assert len(submitted) == 6
    assert hub.calls == []


def test_a_failed_inline_export_keeps_its_record_and_is_not_exported_again(campaign, monkeypatch):
    """An inline export that fails leaves its record, which is what marks it as unfinished: the next
    pass reports it instead of removing whatever it wrote and exporting over it."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    target = next(p for p in publish_models.plan(manifest) if p.label == "org/arm-think@sft_iter_3")
    exported: list[str] = []

    def failing_export(publication, manifest, repo_root, log_path):
        exported.append(publication.label)
        if publication.label == target.label:
            raise publish_models.ExportError("exporter exited 1")
        fake_export(publication, manifest, repo_root, log_path)

    monkeypatch.setattr(publish_models, "run_export", failing_export)
    hub = RecordingHub()
    publish_models.publish_pass(
        manifest, root, hub, RecordingWandb({}), root / "logs" / "a", True, ("arm-think",), "all"
    )
    assert target.inline_export_record.is_file()
    exported.clear()
    assert (
        publish_models.publish_pass(
            manifest, root, hub, RecordingWandb({}), root / "logs" / "b", True, ("arm-think",), "all"
        )
        == 1
    )
    assert exported == [], "the unfinished export was reported, not exported again"
    assert target.inline_export_record.is_file()


def test_a_plan_pass_leaves_inline_export_records_alone(campaign, monkeypatch):
    """A plan pass changes nothing, so a record stays where a pass that acts would discharge it: on a
    published revision and on a verified export alike."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "a", True, (), "export")
    publish_models.publish_pass(
        manifest, root, hub, RecordingWandb({}), root / "logs" / "b", True, ("arm-think",), "upload"
    )
    publications = publish_models.plan(manifest)
    published = next(p for p in publications if p.label == "org/arm-think@sft_iter_1")
    verified = next(p for p in publications if p.model.repo == "org/arm-base")
    for publication in (published, verified):
        publication.inline_export_record.write_text("elsewhere pid 1 since then\n")
    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "c", False, (), "all")
    assert published.inline_export_record.is_file() and verified.inline_export_record.is_file()


def test_a_models_filter_is_refused_where_uploads_are_jobs(campaign):
    """An upload job publishes the whole manifest, so a rolling pass limited to some of its models
    would submit a job that uploads the rest too; the combination is refused rather than widened."""
    root, _, manifest_path = campaign
    with_upload_jobs(manifest_path)
    manifest = publish_models.load_manifest(manifest_path, root)
    with pytest.raises(ValueError, match="--models"):
        publish_models.publish_pass(
            manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / "a", True, ("arm-base",), "rolling"
        )


def test_a_clone_root_whose_tracker_is_a_link_is_refused(campaign, monkeypatch, caplog):
    """Rebuilding a clone writes the tracker beside it; one that is a link (a clone root made by hand)
    would carry that write into a training run's own tracker, and its next segment would resume from
    the wrong checkpoint."""
    root, ckpt, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    target = next(p for p in publish_models.plan(manifest) if p.label == "org/arm-think@sft_iter_3")
    run_tracker = ckpt / "sft" / publish_models.sync_bucket.LATEST_FILE
    before = run_tracker.read_text()
    target.clone_root.mkdir(parents=True)
    (target.clone_root / publish_models.sync_bucket.LATEST_FILE).symlink_to(run_tracker)
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    publish_models.publish_pass(
        manifest, root, RecordingHub(), RecordingWandb({}), root / "logs" / "a", True, ("arm-think",), "export"
    )
    assert "is a link" in caplog.text
    assert run_tracker.read_text() == before, "the training run's tracker is untouched"


def test_an_existing_branch_is_uploaded_to_without_consulting_the_first_commit(campaign, monkeypatch):
    """The first commit is only a branch point, needed when a revision's branch does not exist yet. A
    revision whose branch exists (created, then its upload failed) is uploaded to as it is, even where
    the first commit could no longer serve as a branch point."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "a", True, (), "all")
    repo = "org/arm-think"
    hub.trees[(repo, RecordingHub.INITIAL_COMMIT)] = dict(hub.trees[(repo, publish_models.MAIN)])
    hub.trees[(repo, "sft_iter_1")] = dict(RecordingHub.INITIAL_TREE)
    pending = publish_models.publish_pass(
        manifest, root, hub, RecordingWandb({}), root / "logs" / "b", True, ("arm-think",), "upload"
    )
    assert pending == 0
    assert ("upload_folder", repo, "sft_iter_1") in hub.calls


def test_the_sbatch_wrappers_write_the_logs_that_failure_reports_name():
    """A failed job is reported with the path of its SLURM log, derived here; the wrappers' own
    headers decide where that log is written, so the two must agree. The upload wrapper also takes
    the interpreter as its first argument, which is how submit_upload calls it."""
    for wrapper, log_name in (
        (publish_models.EXPORT_SBATCH, publish_models.EXPORT_JOB_LOG),
        (publish_models.UPLOAD_SBATCH, publish_models.UPLOAD_JOB_LOG),
    ):
        text = (_REPO_ROOT / wrapper).read_text()
        expected = f"#SBATCH --output={publish_models.SLURM_LOG_DIR / log_name.format(job='%j')}"
        assert expected in text.splitlines(), f"{wrapper} does not write {expected}"
    assert 'PYTHON="$1"' in (_REPO_ROOT / publish_models.UPLOAD_SBATCH).read_text()


def test_a_failed_squeue_is_an_error_rather_than_an_empty_queue(campaign, monkeypatch):
    """Reading a failed squeue as "nothing is queued" would resubmit every export already in
    flight."""
    del campaign

    def broken(command, **kwargs):
        del kwargs
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="slurm_load_jobs error")

    monkeypatch.setattr(publish_models.subprocess, "run", broken)
    with pytest.raises(publish_models.ExportError, match="squeue exited 1"):
        publish_models.queued_job_names()


def test_the_export_phase_takes_no_upload_and_the_upload_phase_takes_no_gpu(campaign, monkeypatch):
    """The GPUs are borrowed from another workload on the node for exactly the export: an export
    pass writes and verifies every missing export and uploads nothing, an upload pass uploads
    only what is verified (leaving an unexported publication counted as pending) and never runs
    the exporter, and the two passes together publish everything a single "all" pass would."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    wandb = RecordingWandb({"exp-mid": [{"_step": 4, "lm loss": 1.5}]})
    exports: list[int] = []

    def counting_export(publication, manifest, repo_root, log_path):
        exports.append(publication.iteration)
        fake_export(publication, manifest, repo_root, log_path)

    monkeypatch.setattr(publish_models, "run_export", counting_export)
    pending = publish_models.publish_pass(manifest, root, hub, wandb, root / "logs" / "export", True, (), "export")
    assert pending == 6, "every publication is exported but none is uploaded yet"
    assert len(exports) == 6
    assert [c for c in hub.calls if c[0] in ("upload_folder", "upload_file", "create_collection")] == []

    def no_export(publication, manifest, repo_root, log_path):
        raise AssertionError(f"the upload phase must not export {publication.label}")

    monkeypatch.setattr(publish_models, "run_export", no_export)
    shutil.rmtree(root / "exports" / "arm-think" / "sft" / "iter_0000001" / "hf")
    pending = publish_models.publish_pass(manifest, root, hub, wandb, root / "logs" / "upload", True, (), "upload")
    assert pending == 1, "the export that was removed is left for an export pass, everything else is uploaded"
    uploads = [c for c in hub.calls if c[0] == "upload_folder"]
    assert len(uploads) == 7, "five revisions plus main for the two defaults; the removed one is not uploaded"
    assert ("upload_folder", "org/arm-think", "sft_iter_1") not in hub.calls
    assert ("upload_file", "org/arm-base", "main", "README.md") in hub.calls
    assert ("create_collection", "Test Collection") in hub.calls
    with pytest.raises(ValueError, match="phase must be one of"):
        publish_models.publish_pass(manifest, root, hub, wandb, root / "logs" / "bad", True, (), "verify")


def test_main_builds_a_plan_without_touching_the_hub_and_records_its_manifest(campaign, monkeypatch):
    """The entry point in --plan mode needs no Hub client that knows buckets and no W&B, and
    every run keeps the manifest it ran from beside its log."""
    root, _, manifest_path = campaign
    monkeypatch.setattr(publish_models.sync_bucket, "make_api", lambda: RecordingHub())
    monkeypatch.setenv("HF_HOME", "/projects/a5k/public/hf")
    assert publish_models.main(["--manifest", str(manifest_path), "--repo-root", str(root), "--plan"]) == 1
    run_dirs = [p for p in (root / "logs").iterdir() if p.is_dir()]
    assert len(run_dirs) == 1 and (run_dirs[0] / "manifest.yaml").read_text() == manifest_path.read_text()
    assert not (root / "exports").exists()


def test_local_files_ignores_dotfiles(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"12")
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".hidden").write_text("")
    assert publish_models.local_files(tmp_path) == {"a.safetensors": 2}

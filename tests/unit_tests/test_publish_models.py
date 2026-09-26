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
export clones, safetensors written byte for byte in the format the verifier reads. Three
boundaries are stood in for, each named where it is used: the Hub (a recording client in place of
HfApi — uploads are network and money), W&B (a recording client — network), and the exporter
(``run_export`` is replaced by a function that writes an export into the clone — it needs GPUs and
a SLURM allocation). The campaign's own manifest is loaded as well, so the file that will be used
is the file that is tested.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

import pytest
import yaml

from tests.unit_tests.corpora_fixtures import importable


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOL_DIR = _REPO_ROOT / "scripts" / "hub"
importable(_TOOL_DIR)
publish_models = importlib.import_module("publish_models")

CAMPAIGN_MANIFEST = _REPO_ROOT / "configs" / "control_pretraining" / "hub_models.yaml"
# Tokens per iteration of the fixture's stages, global_batch_size x seq_length as their configs
# state it. The SFT stage runs at half the batch of the stages before it, as the campaign's xl-50b
# SFT does, which is what makes a single campaign-wide figure count its tokens twice.
SEQ = 8
PRE_TPI = 4 * SEQ
SFT_TPI = 2 * SEQ

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
                "dataset": {"seq_length": SEQ, **dataset},
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
    hf_dir.mkdir(parents=True, exist_ok=True)
    weight_map = {name: shard for shard, names in shards.items() for name in names}
    (hf_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    for shard, names in shards.items():
        write_safetensors(hf_dir / shard, names)
    (hf_dir / "config.json").write_text("{}")


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
    write_stage_config(root / "configs" / "pre.yaml", ckpt / "pre", 10, "exp-pre", BLEND, PRE_TPI // SEQ)
    write_stage_config(root / "configs" / "mid.yaml", ckpt / "mid", 4, "exp-mid", BLEND, PRE_TPI // SEQ)
    write_stage_config(root / "configs" / "sft.yaml", ckpt / "sft", 3, "exp-sft", PACKED, SFT_TPI // SEQ)
    make_checkpoint_dir(ckpt / "pre", [5, 10, 15], tracker=10, with_hf=True)
    make_checkpoint_dir(ckpt / "mid", [2, 4], tracker=4)
    make_checkpoint_dir(ckpt / "sft", [3], tracker=3)
    make_checkpoint_dir(ckpt / "sft_clone", [1], tracker=1)
    manifest_path = root / "hub_models.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest_text(root)))
    return root, ckpt, manifest_path


class RecordingHub:
    """Stands in for HfApi: the Hub is a network service and uploads cost money and hours. Records
    every write and keeps each ref's commit history, newest first, as the Hub does: a repository
    starts with one commit, a new branch starts as the history of the revision it is cut from (main
    when none is named), and every upload adds a commit whose tree is its parent's with the uploaded
    files laid over it. An upload to a revision in ``interrupted`` raises before its commit lands,
    as a pass killed mid-upload leaves it."""

    def __init__(self):
        self.calls = []
        self.refs: dict[tuple[str, str], list] = {}
        self.collections = []
        self.interrupted: set[str] = set()
        self._commit_ids = iter(range(1, 1_000_000))

    def _not_found(self, repo_id, revision):
        import httpx
        from huggingface_hub.utils import RevisionNotFoundError

        request = httpx.Request("GET", f"https://huggingface.co/api/models/{repo_id}/tree/{revision}")
        return RevisionNotFoundError("no such revision", response=httpx.Response(404, request=request))

    def _history(self, repo_id, revision):
        if (repo_id, revision) in self.refs:
            return self.refs[(repo_id, revision)]
        for (repo, _), history in self.refs.items():
            for position, commit in enumerate(history):
                if repo == repo_id and commit.commit_id == revision:
                    return history[position:]
        raise self._not_found(repo_id, revision)

    def _commit(self, repo_id, revision, title, files):
        history = self.refs.get((repo_id, revision), [])
        tree = {**(history[0].tree if history else {}), **files}
        commit = type("Commit", (), {"commit_id": f"{next(self._commit_ids):040x}", "title": title, "tree": tree})()
        self.refs[(repo_id, revision)] = [commit, *history]

    def create_repo(self, repo_id, private, exist_ok):
        self.calls.append(("create_repo", repo_id, private))
        if (repo_id, "main") not in self.refs:
            self._commit(repo_id, "main", "initial commit", {".gitattributes": 1519})

    def create_branch(self, repo_id, branch, exist_ok, revision=None):
        self.calls.append(("create_branch", repo_id, branch))
        if (repo_id, branch) not in self.refs:
            self.refs[(repo_id, branch)] = list(self._history(repo_id, revision or "main"))

    def upload_folder(self, folder_path, repo_id, revision, commit_message):
        self.calls.append(("upload_folder", repo_id, revision))
        if revision in self.interrupted:
            raise OSError(f"upload to {repo_id}@{revision} interrupted before its commit landed")
        self._commit(repo_id, revision, commit_message, publish_models.local_files(Path(folder_path)))

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, revision, commit_message):
        self.calls.append(("upload_file", repo_id, revision, path_in_repo))
        self._commit(repo_id, revision, commit_message, {path_in_repo: Path(path_or_fileobj).stat().st_size})

    def list_repo_tree(self, repo_id, revision, recursive):
        tree = self._history(repo_id, revision)[0].tree
        return [type("Entry", (), {"path": p, "size": s})() for p, s in tree.items()]

    def list_repo_commits(self, repo_id, revision=None):
        return list(self._history(repo_id, revision or "main"))

    def list_collections(self, owner):
        return [type("C", (), {"title": t, "slug": s})() for t, s in self.collections]

    def create_collection(self, title, namespace, description, private):
        slug = f"{namespace}/{title.lower().replace(' ', '-')}-abc"
        self.collections.append((title, slug))
        self.calls.append(("create_collection", title))
        return type("C", (), {"slug": slug})()

    def add_collection_item(self, slug, item_id, item_type, exists_ok):
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
    assert [s.tokens_before for s in base.stages] == [0, 10 * PRE_TPI]
    assert [s.tokens_per_iteration for s in base.stages] == [PRE_TPI, PRE_TPI]
    assert base.stages[1].save == ckpt / "mid" and base.stages[1].train_iters == 4
    assert base.stages[0].wandb_exp_name == "exp-pre"
    assert [s.name for s in think.history] == ["pre", "mid"]
    assert think.stages[0].tokens_before == 14 * PRE_TPI
    assert think.stages[0].tokens_per_iteration == SFT_TPI
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


def test_the_campaign_manifest_loads_against_this_checkout():
    """The manifest that will be run must validate: every config it names exists and declares
    the facts the publisher reads. Save directories are not required to exist here."""
    manifest = publish_models.load_manifest(CAMPAIGN_MANIFEST, _REPO_ROOT)
    repos = [m.repo for m in manifest.models]
    # Two repositories per three-stage arm, base and think; one for the post-training ablation,
    # which needs its own rather than a second sft stage under baseline-think (that repository's
    # sft_iter_<n> revisions are the mainline run's, and the card has to say which corpus made the
    # weights); one base repository per midtraining-only narrowly filtered arm, V1 and V2; and the
    # V2 arm's xl-50b think repository. The broad arm's think repository is its xl-50b one.
    assert len(repos) == 8 and all(r.startswith("geodesic-research/control-pretraining-30b-") for r in repos)
    assert len(set(repos)) == len(repos), "two models cannot publish to one repository"
    for model in manifest.models:
        default = [s for s in model.stages if s.default]
        assert len(default) == 1
        assert ("think" in model.repo) == model.reasoning
    # Pretraining and midtraining run 512 x 32768 tokens per iteration, so both SFT runs start at the
    # same token position; the xl-50b SFT then runs at half that batch, which a single campaign-wide
    # figure would have counted twice on its card.
    think = next(m for m in manifest.models if m.repo.endswith("baseline-think"))
    xl50b = next(m for m in manifest.models if m.repo.endswith("baseline-xl50b-think"))
    assert [s.tokens_per_iteration for s in think.history] == [16_777_216, 16_777_216]
    assert think.stages[0].tokens_before == xl50b.stages[0].tokens_before == (29881 + 3126) * 16_777_216
    assert xl50b.stages[0].tokens_per_iteration == 8_388_608


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
    assert by_label["org/arm-base@pretraining_iter_10"].tokens_seen == 10 * PRE_TPI
    assert by_label["org/arm-base@midtraining_iter_2"].tokens_seen == 12 * PRE_TPI
    # Each stage counted at its own batch: 14 iterations at the base batch, 3 at the SFT's half.
    assert by_label["org/arm-think@sft_iter_3"].tokens_seen == 14 * PRE_TPI + 3 * SFT_TPI


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
    write_stage_config(root / "configs" / "mid.yaml", ckpt / "absent", 4, "exp-mid", BLEND, PRE_TPI // SEQ)
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


def test_published_needs_the_revisions_own_commit_and_every_file_at_its_size(campaign):
    """Every checkpoint of one architecture exports the same files at the same sizes, so matching files
    alone would accept a ref carrying another checkpoint's upload: the ref must hold the publication's
    own commit as well."""
    root, _, manifest_path = campaign
    first, second = publish_models.plan(publish_models.load_manifest(manifest_path, root))[:2]
    assert first.model is second.model
    for pub in (first, second):
        write_export(pub.hf_dir, {"model-00001-of-00001.safetensors": ["a"]})
    hub = RecordingHub()
    assert not publish_models.published(hub, first)
    publish_models.upload(hub, first)
    assert publish_models.published(hub, first)
    hub.create_branch(first.model.repo, branch=second.revision, exist_ok=True, revision=first.revision)
    assert not publish_models.published(hub, second), "a branch cut from another checkpoint's upload"
    hub.refs[(first.model.repo, first.revision)][0].tree["config.json"] = 999
    assert not publish_models.published(hub, first)


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
    assert f"| `pretraining_iter_5` | pretraining | 5 | {5 * PRE_TPI:,} (0.0B) | 2.5000 |" in card
    assert f"| `midtraining_iter_4` (also `main`) | midtraining | 4 | {14 * PRE_TPI:,} (0.0B) |  |" in card
    assert "| Stage | Iterations | Tokens per iteration | Tokens | W&B run |" in card
    assert f"| pretraining (this repository) | 10 | {PRE_TPI:,} |" in card
    think = manifest.models[1]
    think_card = publish_models.render_model_card(manifest, think, [])
    assert f"| sft (this repository) | 3 | {SFT_TPI:,} |" in think_card
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
    verifiable export into the clone, as the exporter would."""
    write_export(publication.hf_dir, {"model-00001-of-00001.safetensors": ["w"]})
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
    card = (root / "logs" / "cards" / "arm-base" / "README.md").read_text()
    assert f"| `midtraining_iter_4` (also `main`) | midtraining | 4 | {14 * PRE_TPI:,} (0.0B) | 1.5000 |" in card
    assert (root / "exports" / "arm-base" / "pretraining" / "iter_0000005" / "run_config.yaml").is_file()

    # A second pass finds everything on the Hub, re-exports nothing and re-uploads no card.
    calls_before = len(hub.calls)
    assert publish_models.publish_pass(manifest, root, hub, wandb, root / "logs" / "run2", True, (), "all") == 0
    assert [c for c in hub.calls[calls_before:] if c[0] in ("upload_folder", "upload_file")] == []


def test_an_intermediate_interrupted_after_the_final_is_on_main_is_neither_served_nor_skipped(campaign, monkeypatch):
    """Every checkpoint of one architecture exports the same files at the same sizes, and main holds
    the final once it is up. A pass killed between creating an intermediate's branch and landing its
    commit must leave that branch without the final's weights (a consumer reading it meanwhile would
    take them for the intermediate's), and the next pass must upload the intermediate rather than
    find the final's files under its name and skip it."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    wandb = RecordingWandb({"exp-mid": [{"_step": 2, "lm loss": 1.6}, {"_step": 4, "lm loss": 1.5}]})
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    hub.interrupted = {"midtraining_iter_2"}
    with pytest.raises(OSError, match="midtraining_iter_2"):
        publish_models.publish_pass(
            manifest, root, hub, wandb, root / "logs" / "run1", True, ("arm-base",), "all", True
        )
    assert [c.title for c in hub.list_repo_commits("org/arm-base", revision="main")][
        0
    ] == "midtraining iteration 4 (final)"
    intermediate = next(p for p in publish_models.plan(manifest) if p.revision == "midtraining_iter_2")
    stranded = {
        entry.path for entry in hub.list_repo_tree("org/arm-base", revision="midtraining_iter_2", recursive=False)
    }
    assert not stranded & set(publish_models.local_files(intermediate.hf_dir)), (
        f"the interrupted branch serves another checkpoint's files: {sorted(stranded)}"
    )

    hub.interrupted = set()
    calls_before = len(hub.calls)
    assert (
        publish_models.publish_pass(
            manifest, root, hub, wandb, root / "logs" / "run2", True, ("arm-base",), "all", True
        )
        == 0
    )
    assert ("upload_folder", "org/arm-base", "midtraining_iter_2") in hub.calls[calls_before:]
    assert hub.list_repo_commits("org/arm-base", revision="midtraining_iter_2")[0].title == "midtraining iteration 2"


def test_a_final_whose_main_commit_never_landed_is_uploaded_to_main_again(campaign, monkeypatch):
    """The default publication goes to its revision and then to main; a pass killed between the two
    leaves the revision complete and main without the final, and the next pass must finish main."""
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    wandb = RecordingWandb({"exp-mid": [{"_step": 4, "lm loss": 1.5}]})
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    hub.interrupted = {"main"}
    with pytest.raises(OSError, match="main"):
        publish_models.publish_pass(
            manifest, root, hub, wandb, root / "logs" / "run1", True, ("arm-base",), "all", True
        )

    hub.interrupted = set()
    calls_before = len(hub.calls)
    assert (
        publish_models.publish_pass(
            manifest, root, hub, wandb, root / "logs" / "run2", True, ("arm-base",), "all", True
        )
        == 0
    )
    assert ("upload_folder", "org/arm-base", "main") in hub.calls[calls_before:]
    assert "midtraining iteration 4 (final)" in [
        c.title for c in hub.list_repo_commits("org/arm-base", revision="main")
    ]


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

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
import struct
import sys
from pathlib import Path

import pytest
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOL_DIR = _REPO_ROOT / "scripts" / "hub"
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))
publish_models = importlib.import_module("publish_models")

CAMPAIGN_MANIFEST = _REPO_ROOT / "configs" / "control_pretraining" / "hub_models.yaml"
TPI = 1000

OLD_TARGET, NEW_TARGET = publish_models.RUN_CONFIG_EDITS[0]
OLD_IMPL, NEW_IMPL = publish_models.RUN_CONFIG_EDITS[1]
RAW_RUN_CONFIG = f"model:\n  mamba_stack_spec:\n    _target_: {OLD_TARGET}\n  {OLD_IMPL}\n"


def write_stage_config(path: Path, save: Path, train_iters: int, exp_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "train": {"train_iters": train_iters},
                "checkpoint": {"save": str(save)},
                "logger": {"wandb_exp_name": exp_name},
            }
        )
    )


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
        "tokens_per_iteration": TPI,
        "export_root": str(repo_root / "exports"),
        "log_dir": str(repo_root / "logs"),
        "hf_home": "/projects/a5k/public/hf",
        "export": {"tp": 1, "ep": 4},
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
    write_stage_config(root / "configs" / "pre.yaml", ckpt / "pre", 10, "exp-pre")
    write_stage_config(root / "configs" / "mid.yaml", ckpt / "mid", 4, "exp-mid")
    write_stage_config(root / "configs" / "sft.yaml", ckpt / "sft", 3, "exp-sft")
    make_checkpoint_dir(ckpt / "pre", [5, 10, 15], tracker=10, with_hf=True)
    make_checkpoint_dir(ckpt / "mid", [2, 4], tracker=4)
    make_checkpoint_dir(ckpt / "sft", [3], tracker=3)
    make_checkpoint_dir(ckpt / "sft_clone", [1], tracker=1)
    manifest_path = root / "hub_models.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest_text(root)))
    return root, ckpt, manifest_path


class RecordingHub:
    """Stands in for HfApi: the Hub is a network service and uploads cost money and hours. Records
    every call and answers list_repo_tree from what has been "uploaded"."""

    def __init__(self):
        self.calls = []
        self.trees: dict[tuple[str, str], dict[str, int]] = {}
        self.collections = []

    def create_repo(self, repo_id, private, exist_ok):
        self.calls.append(("create_repo", repo_id, private))

    def create_branch(self, repo_id, branch, exist_ok):
        self.calls.append(("create_branch", repo_id, branch))

    def upload_folder(self, folder_path, repo_id, revision, commit_message):
        self.calls.append(("upload_folder", repo_id, revision))
        self.trees[(repo_id, revision)] = publish_models.local_files(Path(folder_path))

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, revision, commit_message):
        self.calls.append(("upload_file", repo_id, revision, path_in_repo))

    def list_repo_tree(self, repo_id, revision, recursive):
        import httpx
        from huggingface_hub.utils import RevisionNotFoundError

        if (repo_id, revision) not in self.trees:
            request = httpx.Request("GET", f"https://huggingface.co/api/models/{repo_id}/tree/{revision}")
            raise RevisionNotFoundError("no such revision", response=httpx.Response(404, request=request))
        return [type("Entry", (), {"path": p, "size": s})() for p, s in self.trees[(repo_id, revision)].items()]

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
    assert [s.tokens_before for s in base.stages] == [0, 10]
    assert base.stages[1].save == ckpt / "mid" and base.stages[1].train_iters == 4
    assert base.stages[0].wandb_exp_name == "exp-pre"
    assert [s.name for s in think.history] == ["pre", "mid"]
    assert think.stages[0].tokens_before == 14
    assert think.stages[0].extra_directories == (ckpt / "sft_clone",)
    assert manifest.export.ep == 4 and manifest.card.tags == ("tag-a", "tag-b")


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
    assert len(repos) == 4 and all(r.startswith("geodesic-research/control-pretraining-30b-") for r in repos)
    assert manifest.tokens_per_iteration == 16_777_216
    for model in manifest.models:
        default = [s for s in model.stages if s.default]
        assert len(default) == 1
        assert ("think" in model.repo) == model.reasoning
    think = next(m for m in manifest.models if m.repo.endswith("baseline-think"))
    assert think.stages[0].tokens_before == 29881 + 3126


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
    assert by_label["org/arm-base@pretraining_iter_10"].tokens_seen == 10 * TPI
    assert by_label["org/arm-base@midtraining_iter_2"].tokens_seen == 12 * TPI
    assert by_label["org/arm-think@sft_iter_3"].tokens_seen == 17 * TPI


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


def test_a_stage_whose_directory_does_not_exist_yet_publishes_nothing(campaign):
    root, ckpt, manifest_path = campaign
    write_stage_config(root / "configs" / "mid.yaml", ckpt / "absent", 4, "exp-mid")
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
    assert "| `pretraining_iter_5` | pretraining | 5 | 5,000 (0.0B) | 2.5000 |" in card
    assert "| `midtraining_iter_4` (also `main`) | midtraining | 4 | 14,000 (0.0B) |  |" in card
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
    pending = publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "run", False, ())
    assert pending == 6 and hub.calls == []
    assert not (root / "exports").exists()


def test_a_pass_exports_uploads_writes_cards_and_joins_the_collection(campaign, monkeypatch):
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()
    wandb = RecordingWandb({"exp-mid": [{"_step": 4, "lm loss": 1.5}]})
    monkeypatch.setattr(publish_models, "run_export", fake_export)
    run_dir = root / "logs" / "run1"
    assert publish_models.publish_pass(manifest, root, hub, wandb, run_dir, True, ()) == 0
    uploads = [c for c in hub.calls if c[0] == "upload_folder"]
    assert len(uploads) == 8, "six revisions plus main for the two defaults"
    assert ("upload_file", "org/arm-base", "main", "README.md") in hub.calls
    assert ("upload_file", "org/arm-think", "main", "README.md") in hub.calls
    assert ("create_collection", "Test Collection") in hub.calls
    assert {c[2] for c in hub.calls if c[0] == "add_collection_item"} == {"org/arm-base", "org/arm-think"}
    card = (root / "logs" / "cards" / "arm-base" / "README.md").read_text()
    assert "| `midtraining_iter_4` (also `main`) | midtraining | 4 | 14,000 (0.0B) | 1.5000 |" in card
    assert (root / "exports" / "arm-base" / "pretraining" / "iter_0000005" / "run_config.yaml").is_file()

    # A second pass finds everything on the Hub, re-exports nothing and re-uploads no card.
    calls_before = len(hub.calls)
    assert publish_models.publish_pass(manifest, root, hub, wandb, root / "logs" / "run2", True, ()) == 0
    assert [c for c in hub.calls[calls_before:] if c[0] in ("upload_folder", "upload_file")] == []


def test_a_failed_export_is_reported_and_counted_not_hidden(campaign, monkeypatch):
    root, _, manifest_path = campaign
    manifest = publish_models.load_manifest(manifest_path, root)
    hub = RecordingHub()

    def broken_export(publication, manifest, repo_root, log_path):
        if publication.iteration == 10:
            raise publish_models.ExportError("exporter exited 1")
        fake_export(publication, manifest, repo_root, log_path)

    monkeypatch.setattr(publish_models, "run_export", broken_export)
    pending = publish_models.publish_pass(manifest, root, hub, RecordingWandb({}), root / "logs" / "run", True, ())
    assert pending == 1
    assert ("upload_folder", "org/arm-base", "pretraining_iter_10") not in hub.calls
    assert ("upload_folder", "org/arm-base", "pretraining_iter_5") in hub.calls


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

# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Unit tests for pipeline/export/manifest.py: the manifest that an HF export carries.

Every test writes a real export directory under ``tmp_path`` and calls the real writer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipeline.export.manifest import EXPORT_COMPLETE_FILE, EXPORT_MANIFEST, write_export_manifest


TEMPLATE = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
TOKENIZER_ID = "geodesic-research/nemotron-instruct-tokenizer"
COMMIT = "a" * 40


def make_export(root: Path, *, jinja: str | None = TEMPLATE, embedded: str | None = TEMPLATE) -> Path:
    """An export directory in the layout of an export tree, ``<run>/iter_N``: two weight shards, a nested file, and
    the tokenizer files that the fixups leave. ``jinja`` and ``embedded`` are the two copies of the chat template,
    each left out when None."""
    export = root / "run" / "iter_0000100"
    (export / "sub").mkdir(parents=True)
    (export / "model-00001-of-00002.safetensors").write_bytes(b"w" * 40)
    (export / "model-00002-of-00002.safetensors").write_bytes(b"v" * 24)
    (export / "sub" / "extra.json").write_text("{}")
    config = {} if embedded is None else {"chat_template": embedded}
    (export / "tokenizer_config.json").write_text(json.dumps(config))
    if jinja is not None:
        (export / "chat_template.jinja").write_text(jinja)
    return export


def write(export: Path, **override) -> Path:
    arguments = {
        "checkpoint": Path("/ckpt/run/iter_0000100"),
        "iteration": 100,
        "hf_model_id": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "training_tokenizer_id": TOKENIZER_ID,
        "training_tokenizer_revision": COMMIT,
    }
    return write_export_manifest(export, **{**arguments, **override})


class TestExportManifest:
    def test_it_records_the_size_and_sha256_of_every_other_file(self, tmp_path):
        export = make_export(tmp_path)
        manifest = json.loads(write(export).read_text())
        assert set(manifest["files"]) == {
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "sub/extra.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        }
        shard = manifest["files"]["model-00001-of-00002.safetensors"]
        assert shard == {"size": 40, "sha256": hashlib.sha256(b"w" * 40).hexdigest()}

    def test_it_records_where_the_export_came_from_and_which_tokenizer_it_carries(self, tmp_path):
        manifest = json.loads(write(make_export(tmp_path)).read_text())
        assert manifest["run"] == "run"
        assert "checkpoint" not in manifest
        assert manifest["iteration"] == 100
        assert manifest["hf_model"] == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
        assert manifest["training_tokenizer"] == {"id": TOKENIZER_ID, "revision": COMMIT}

    def test_an_unpinned_tokenizer_records_none(self, tmp_path):
        manifest = json.loads(
            write(make_export(tmp_path), training_tokenizer_id=None, training_tokenizer_revision=None).read_text()
        )
        assert manifest["training_tokenizer"] == {"id": None, "revision": None}

    def test_the_manifest_does_not_list_itself_or_the_completion_marker_and_the_same_export_gives_the_same_bytes(
        self, tmp_path
    ):
        export = make_export(tmp_path)
        first = write(export).read_bytes()
        (export / EXPORT_COMPLETE_FILE).write_text("run: config\n")
        second = write(export).read_bytes()
        assert second == first
        assert not {EXPORT_MANIFEST, EXPORT_COMPLETE_FILE} & set(json.loads(first)["files"])

    def test_a_partial_file_that_a_crash_left_is_neither_listed_nor_kept(self, tmp_path):
        export = make_export(tmp_path)
        leftover = export / (EXPORT_MANIFEST + ".tmp")
        leftover.write_text("{ half")
        manifest = json.loads(write(export).read_text())
        assert EXPORT_MANIFEST + ".tmp" not in manifest["files"]
        assert not leftover.exists()


class TestChatTemplateHash:
    def test_it_equals_the_sha256_of_the_template_file(self, tmp_path):
        export = make_export(tmp_path)
        manifest = json.loads(write(export).read_text())
        assert (
            manifest["chat_template_sha256"]
            == hashlib.sha256((export / "chat_template.jinja").read_bytes()).hexdigest()
        )
        assert manifest["chat_template_sha256"] == manifest["files"]["chat_template.jinja"]["sha256"]

    def test_an_export_with_the_template_file_alone_is_hashed_from_that_file(self, tmp_path):
        """An export that took over its release's tokenizer files has the file and no embedded copy."""
        export = make_export(tmp_path, embedded=None)
        manifest = json.loads(write(export).read_text())
        assert manifest["chat_template_sha256"] == hashlib.sha256(TEMPLATE.encode()).hexdigest()

    def test_an_export_with_the_embedded_template_alone_is_hashed_from_it(self, tmp_path):
        manifest = json.loads(write(make_export(tmp_path, jinja=None)).read_text())
        assert manifest["chat_template_sha256"] == hashlib.sha256(TEMPLATE.encode()).hexdigest()

    def test_an_export_with_no_template_records_none(self, tmp_path):
        manifest = json.loads(write(make_export(tmp_path, jinja=None, embedded=None)).read_text())
        assert manifest["chat_template_sha256"] is None

    def test_an_export_without_tokenizer_config_is_hashed_from_the_template_file_and_lists_no_config(self, tmp_path):
        export = make_export(tmp_path)
        (export / "tokenizer_config.json").unlink()
        manifest = json.loads(write(export).read_text())
        assert manifest["chat_template_sha256"] == hashlib.sha256(TEMPLATE.encode()).hexdigest()
        assert "tokenizer_config.json" not in manifest["files"]

    def test_an_embedded_template_that_differs_from_the_template_file_is_an_error(self, tmp_path):
        export = make_export(tmp_path, embedded="{{ other }}")
        with pytest.raises(ValueError, match="differs from"):
            write(export)
        assert not (export / EXPORT_MANIFEST).exists()

    def test_a_tokenizer_config_that_is_not_json_is_an_error(self, tmp_path):
        export = make_export(tmp_path)
        (export / "tokenizer_config.json").write_text("{ not json")
        with pytest.raises(ValueError):
            write(export)
        assert not (export / EXPORT_MANIFEST).exists()

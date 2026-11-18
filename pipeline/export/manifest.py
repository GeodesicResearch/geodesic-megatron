# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The manifest of one HF export, whose sha256 names the export.

Catalogue:
    EXPORT_COMPLETE_FILE: the file that the exporter writes last, after which an export is complete.
    EXPORT_MANIFEST: the file name of the manifest.
    write_export_manifest: write the manifest of an export directory.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.file_digest import HASH_WORKERS, chat_template_sha256, directory_records


# The exporter's last write: it copies the checkpoint's run_config into the export after the shards, the index, the
# tokenizer fixups and the manifest, so an export without it was cut short, however complete its tensors look.
# Every export clone carries a run_config, so a finished export always has one. publish_models.py and sync_bucket.py
# decide that an export is complete by this file.
EXPORT_COMPLETE_FILE = "megatron_run_config.yaml"
EXPORT_MANIFEST = "export_manifest.json"
_PARTIAL_SUFFIX = ".tmp"
_TEMPLATE_FILE = "chat_template.jinja"
_TOKENIZER_CONFIG = "tokenizer_config.json"


def _chat_template(hf_path: Path) -> str | dict | None:
    """The chat template that a loader reads from the export: ``chat_template.jinja``, else the embedded copy.

    Raises:
        ValueError: ``tokenizer_config.json`` is not JSON, or its embedded template differs from ``chat_template.jinja``.
    """
    config_path, template_path = hf_path / _TOKENIZER_CONFIG, hf_path / _TEMPLATE_FILE
    embedded = json.loads(config_path.read_text()).get("chat_template") if config_path.is_file() else None
    if not template_path.is_file():
        return embedded
    shipped = template_path.read_text()
    if embedded is not None and embedded != shipped:
        raise ValueError(f"{config_path}: the embedded chat_template differs from {template_path}")
    return shipped


def write_export_manifest(
    hf_path: Path,
    *,
    checkpoint: Path,
    iteration: int,
    hf_model_id: str,
    training_tokenizer_id: str | None,
    training_tokenizer_revision: str | None,
) -> Path:
    """Write ``EXPORT_MANIFEST`` in ``hf_path`` and return its path.

    ``checkpoint`` is the iteration directory of the run, ``<run>/iter_N``, and the manifest names its run directory.
    Call this once every file of the export except ``EXPORT_COMPLETE_FILE`` is written. The function reads every
    file under ``hf_path`` to hash it. The manifest lists every file except itself and ``EXPORT_COMPLETE_FILE``, so a
    second call on an unchanged export writes the same bytes. The file appears whole or not at all.

    ``chat_template_sha256`` is the sha256 of ``chat_template.jinja`` (the file that transformers 5 loads), or of the
    embedded template of ``tokenizer_config.json`` when the export has no such file, or None when it has neither.

    Raises:
        ValueError: ``tokenizer_config.json`` is not JSON, or its embedded template differs from ``chat_template.jinja``.
    """
    template = _chat_template(hf_path)
    excluded = [EXPORT_MANIFEST, EXPORT_MANIFEST + _PARTIAL_SUFFIX, EXPORT_COMPLETE_FILE]
    records = directory_records(hf_path, None, excluded, None, HASH_WORKERS)
    manifest = {
        "run": checkpoint.parent.name,
        "iteration": iteration,
        "hf_model": hf_model_id,
        "training_tokenizer": {"id": training_tokenizer_id, "revision": training_tokenizer_revision},
        "chat_template_sha256": chat_template_sha256(template),
        "files": {r.path: {"size": r.size, "sha256": r.sha256} for r in records},
    }
    path = hf_path / EXPORT_MANIFEST
    partial = path.with_name(path.name + _PARTIAL_SUFFIX)
    partial.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    partial.replace(path)
    return path

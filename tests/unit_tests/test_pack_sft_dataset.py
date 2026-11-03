"""scripts/data/pack_sft_dataset.py: an existing pack is reused only for the settings it was packed with."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "data" / "pack_sft_dataset.py"
DEFAULT_KWARGS = {"chat": True, "use_hf_tokenizer_chat_template": True, "answer_only_loss": True}
EXCLUSIONS = {"path": "/data/excluded_rows.jsonl", "shard_index": 3, "count": 2, "sha256": "ab" * 32}


@pytest.fixture(scope="module")
def pack_script():
    spec = importlib.util.spec_from_file_location("pack_sft_dataset", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pack_sft_dataset"] = module
    spec.loader.exec_module(module)
    return module


def _settings(algorithm="first_fit_shuffle", kwargs=None, exclusions=None):
    settings = {
        "packing_algorithm": algorithm,
        "dataset_kwargs": kwargs or dict(DEFAULT_KWARGS),
        "seq_length": 4096,
        "seed": 1,
    }
    if exclusions is not None:
        settings["exclude_rows"] = exclusions
    return settings


def test_manifest_sits_beside_the_pack(pack_script, tmp_path):
    assert (
        pack_script.pack_manifest_path(tmp_path / "training_4096.idx.parquet") == tmp_path / "training_4096.pack.json"
    )


def test_a_pack_with_matching_manifest_is_reused(pack_script, tmp_path):
    pack = tmp_path / "training_4096.idx.parquet"
    pack.write_bytes(b"")
    pack_script.pack_manifest_path(pack).write_text(json.dumps(_settings("one_per_pack")))
    pack_script.check_existing_pack(pack, _settings("one_per_pack"), DEFAULT_KWARGS)


def test_a_pack_with_other_settings_is_refused(pack_script, tmp_path):
    pack = tmp_path / "training_4096.idx.parquet"
    pack.write_bytes(b"")
    pack_script.pack_manifest_path(pack).write_text(json.dumps(_settings("first_fit_shuffle")))
    with pytest.raises(ValueError, match="was packed with"):
        pack_script.check_existing_pack(pack, _settings("one_per_pack"), DEFAULT_KWARGS)


def test_a_pack_without_manifest_is_reused_only_for_the_legacy_settings(pack_script, tmp_path):
    pack = tmp_path / "training_4096.idx.parquet"
    pack.write_bytes(b"")
    pack_script.check_existing_pack(pack, _settings(), DEFAULT_KWARGS)
    with pytest.raises(ValueError, match="settings are unknown"):
        pack_script.check_existing_pack(pack, _settings("one_per_pack"), DEFAULT_KWARGS)
    with pytest.raises(ValueError, match="settings are unknown"):
        pack_script.check_existing_pack(pack, _settings(kwargs={**DEFAULT_KWARGS, "add_eos": False}), DEFAULT_KWARGS)


def test_a_pack_without_manifest_is_not_reused_for_excluded_rows(pack_script, tmp_path):
    # A pack that predates manifests was made with every row, whatever the other settings.
    pack = tmp_path / "training_4096.idx.parquet"
    pack.write_bytes(b"")
    with pytest.raises(ValueError, match="settings are unknown"):
        pack_script.check_existing_pack(pack, _settings(exclusions=EXCLUSIONS), DEFAULT_KWARGS)


def test_a_pack_is_reused_only_for_the_exclusions_in_its_manifest(pack_script, tmp_path):
    pack = tmp_path / "training_4096.idx.parquet"
    pack.write_bytes(b"")
    pack_script.pack_manifest_path(pack).write_text(json.dumps(_settings(exclusions=EXCLUSIONS)))
    pack_script.check_existing_pack(pack, _settings(exclusions=EXCLUSIONS), DEFAULT_KWARGS)
    changed = (
        None,
        {**EXCLUSIONS, "sha256": "0" * 64},
        {**EXCLUSIONS, "count": 3},
        {**EXCLUSIONS, "path": "/data/other_rows.jsonl"},
    )
    for other in changed:
        with pytest.raises(ValueError, match="was packed with"):
            pack_script.check_existing_pack(pack, _settings(exclusions=other), DEFAULT_KWARGS)

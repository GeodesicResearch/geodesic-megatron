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


def _settings(pack_script, algorithm="first_fit_shuffle", kwargs=None, exclusions=None):
    """The settings the packer records, with its seeded shuffle."""
    settings = {
        "packing_algorithm": algorithm,
        "dataset_kwargs": kwargs or dict(DEFAULT_KWARGS),
        "seq_length": 4096,
        "seed": 1,
        "shuffle": pack_script.SEEDED_SHUFFLE,
    }
    if exclusions is not None:
        settings["exclude_rows"] = exclusions
    return settings


def _pack(pack_script, tmp_path, manifest=None):
    """An empty pack file, with ``manifest`` written beside it when one is given."""
    pack = tmp_path / "training_4096.idx.parquet"
    pack.write_bytes(b"")
    if manifest is not None:
        pack_script.pack_manifest_path(pack).write_text(json.dumps(manifest))
    return pack


def test_manifest_sits_beside_the_pack(pack_script, tmp_path):
    assert (
        pack_script.pack_manifest_path(tmp_path / "training_4096.idx.parquet") == tmp_path / "training_4096.pack.json"
    )


def test_a_pack_with_matching_manifest_is_reused(pack_script, tmp_path):
    pack = _pack(pack_script, tmp_path, _settings(pack_script, "one_per_pack"))
    pack_script.check_existing_pack(pack, _settings(pack_script, "one_per_pack"))


def test_a_pack_with_other_settings_is_refused(pack_script, tmp_path):
    pack = _pack(pack_script, tmp_path, _settings(pack_script, "first_fit_shuffle"))
    with pytest.raises(ValueError, match="was packed with"):
        pack_script.check_existing_pack(pack, _settings(pack_script, "one_per_pack"))


def test_a_pack_without_manifest_is_refused_whatever_the_settings(pack_script, tmp_path):
    pack = _pack(pack_script, tmp_path)
    for settings in (_settings(pack_script), _settings(pack_script, exclusions=EXCLUSIONS)):
        with pytest.raises(ValueError, match="settings are unknown"):
            pack_script.check_existing_pack(pack, settings)


def test_a_pack_whose_shuffle_ignored_its_seed_is_refused(pack_script, tmp_path):
    """A manifest without the seeded shuffle records a seed that did not order the pack."""
    unseeded = {key: value for key, value in _settings(pack_script).items() if key != "shuffle"}
    pack = _pack(pack_script, tmp_path, unseeded)
    with pytest.raises(ValueError, match="was packed with"):
        pack_script.check_existing_pack(pack, _settings(pack_script))


def test_a_pack_is_reused_only_for_the_exclusions_in_its_manifest(pack_script, tmp_path):
    pack = _pack(pack_script, tmp_path, _settings(pack_script, exclusions=EXCLUSIONS))
    pack_script.check_existing_pack(pack, _settings(pack_script, exclusions=EXCLUSIONS))
    changed = (
        None,
        {**EXCLUSIONS, "sha256": "0" * 64},
        {**EXCLUSIONS, "count": 3},
        {**EXCLUSIONS, "path": "/data/other_rows.jsonl"},
    )
    for other in changed:
        with pytest.raises(ValueError, match="was packed with"):
            pack_script.check_existing_pack(pack, _settings(pack_script, exclusions=other))

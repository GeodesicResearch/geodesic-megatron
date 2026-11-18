"""scripts/data/pack_sft_dataset.py: an existing pack is reused only for the settings it was packed with."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "data" / "pack_sft_dataset.py"
DEFAULT_KWARGS = {"chat": True, "use_hf_tokenizer_chat_template": True, "answer_only_loss": True}


@pytest.fixture(scope="module")
def pack_script():
    spec = importlib.util.spec_from_file_location("pack_sft_dataset", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pack_sft_dataset"] = module
    spec.loader.exec_module(module)
    return module


def _settings(algorithm="first_fit_shuffle", kwargs=None, **extra):
    return {
        "packing_algorithm": algorithm,
        "dataset_kwargs": kwargs or dict(DEFAULT_KWARGS),
        "seq_length": 4096,
        "seed": 1,
        **extra,
    }


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


TOKENIZER_FIXTURE = Path(__file__).parent / "fixtures" / "hf_export" / "inputs" / "nemotron-instruct-tokenizer"


def test_a_template_hashes_as_the_file_it_was_read_from(pack_script):
    template_file = TOKENIZER_FIXTURE / "chat_template.jinja"
    assert (
        pack_script.chat_template_sha256(template_file.read_text(encoding="utf-8"))
        == hashlib.sha256(template_file.read_bytes()).hexdigest()
    )


def test_templates_held_by_name_hash_by_their_content_in_any_order(pack_script):
    one = {"default": "{{ a }}", "tool_use": "{{ b }}"}
    assert pack_script.chat_template_sha256(one) == pack_script.chat_template_sha256(dict(reversed(one.items())))
    assert pack_script.chat_template_sha256(one) != pack_script.chat_template_sha256({**one, "tool_use": "{{ c }}"})


def test_a_tokenizer_without_a_template_has_no_hash(pack_script):
    assert pack_script.chat_template_sha256(None) is None


def test_a_pack_built_under_the_same_template_is_reused(pack_script, tmp_path):
    pack = tmp_path / "training_4096.idx.parquet"
    pack.write_bytes(b"")
    settings = _settings(**{pack_script.TEMPLATE_KEY: "a" * 64})
    pack_script.pack_manifest_path(pack).write_text(json.dumps(settings))
    pack_script.check_existing_pack(pack, settings, DEFAULT_KWARGS)


def test_a_pack_built_under_another_template_is_refused(pack_script, tmp_path):
    pack = tmp_path / "training_4096.idx.parquet"
    pack.write_bytes(b"")
    pack_script.pack_manifest_path(pack).write_text(json.dumps(_settings(**{pack_script.TEMPLATE_KEY: "a" * 64})))
    with pytest.raises(ValueError, match="was packed with"):
        pack_script.check_existing_pack(pack, _settings(**{pack_script.TEMPLATE_KEY: "b" * 64}), DEFAULT_KWARGS)


def test_a_manifest_that_predates_the_template_hash_is_reused_with_a_warning(pack_script, tmp_path, caplog):
    pack = tmp_path / "training_4096.idx.parquet"
    pack.write_bytes(b"")
    pack_script.pack_manifest_path(pack).write_text(json.dumps(_settings()))
    wanted = _settings(**{pack_script.TEMPLATE_KEY: "a" * 64})
    with caplog.at_level("WARNING"):
        pack_script.check_existing_pack(pack, wanted, DEFAULT_KWARGS)
    assert "predates the chat template hash" in caplog.text
    with pytest.raises(ValueError, match="was packed with"):
        pack_script.check_existing_pack(pack, {**wanted, "seed": 2}, DEFAULT_KWARGS)

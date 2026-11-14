# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The export saves the snapshot's generation config through transformers, which refuses an invalid one."""

import json

import pytest
from transformers import GenerationConfig

from pipeline.export import hf
from pipeline.export.decoding_defaults import (
    REASONING_GENERATION_DEFAULTS,
    SAMPLING_KEYS,
    generation_config_text,
    release_file_in_export,
)
from pipeline.export.hf import without_sampling_fields
from tests.unit_tests import recorded_bridge
from tests.unit_tests.checkpoint_dirs import make_checkpoint_dir
from tests.unit_tests.hf_export_fixtures import NANO, THINK_SFT_GENERATION_CONFIG


def test_transformers_refuses_to_save_a_config_that_sets_sampling_fields_without_do_sample(tmp_path):
    with pytest.raises(ValueError, match="do_sample"):
        GenerationConfig.from_dict(THINK_SFT_GENERATION_CONFIG).save_pretrained(tmp_path)


def test_without_sampling_fields_gives_a_config_that_saves(tmp_path):
    saved = without_sampling_fields(GenerationConfig.from_dict(THINK_SFT_GENERATION_CONFIG))
    saved.save_pretrained(tmp_path)
    assert (tmp_path / "generation_config.json").is_file()


def test_without_sampling_fields_keeps_every_other_field():
    original = GenerationConfig.from_dict(THINK_SFT_GENERATION_CONFIG)
    kept = without_sampling_fields(original).to_dict()
    assert kept["eos_token_id"] == [100265, 100257]
    assert kept["max_new_tokens"] == 32768
    default = GenerationConfig().to_dict()
    assert all(kept[key] == default[key] for key in SAMPLING_KEYS)


def test_without_sampling_fields_leaves_its_argument_unchanged():
    original = GenerationConfig.from_dict(THINK_SFT_GENERATION_CONFIG)
    without_sampling_fields(original)
    assert original.temperature == 0.6 and original.top_p == 0.95


#: A value for each field of SAMPLING_KEYS, except ``do_sample``, that transformers 5.3.0 refuses to save while
#: ``do_sample`` is not true.
NON_DEFAULT_SAMPLING_VALUES = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.05,
    "top_h": 0.9,
    "typical_p": 0.9,
    "epsilon_cutoff": 0.001,
    "eta_cutoff": 0.001,
}


def test_every_sampling_key_has_a_value_in_the_table_of_this_module():
    assert set(SAMPLING_KEYS) == {"do_sample", *NON_DEFAULT_SAMPLING_VALUES}


@pytest.mark.parametrize("field", sorted(NON_DEFAULT_SAMPLING_VALUES))
def test_without_sampling_fields_resets_each_field_that_transformers_refuses_to_save(tmp_path, field):
    original = GenerationConfig.from_dict({field: NON_DEFAULT_SAMPLING_VALUES[field]})
    with pytest.raises(ValueError, match="do_sample"):
        original.save_pretrained(tmp_path / "before")
    without_sampling_fields(original).save_pretrained(tmp_path / "after")
    assert (tmp_path / "after" / "generation_config.json").is_file()


@pytest.mark.parametrize("reasoning", [False, True])
def test_the_modes_decoding_defaults_replace_every_sampling_key_of_a_release(reasoning):
    """A greedy export carries no sampling field, and a reasoning export carries only its own defaults."""
    release = {"eos_token_id": 100265, "do_sample": True, **NON_DEFAULT_SAMPLING_VALUES}
    written = json.loads(generation_config_text(json.dumps(release), reasoning))
    assert written == {"eos_token_id": 100265, **(REASONING_GENERATION_DEFAULTS if reasoning else {})}


def test_without_sampling_fields_also_resets_a_config_that_samples(tmp_path):
    """The exporter sets the decoding defaults of the export's mode after the save, so none of the config's own
    sampling fields reach the export."""
    original = GenerationConfig.from_dict({"do_sample": True, "temperature": 0.7, "min_p": 0.05})
    kept = without_sampling_fields(original)
    default = GenerationConfig()
    assert (kept.do_sample, kept.temperature, kept.min_p) == (default.do_sample, default.temperature, default.min_p)
    kept.save_pretrained(tmp_path)


def test_only_the_generation_config_of_a_release_takes_the_modes_decoding_defaults():
    release = json.dumps(THINK_SFT_GENERATION_CONFIG, indent=4).encode()
    for reasoning in (False, True):
        expected = generation_config_text(release.decode(), reasoning).encode()
        assert release_file_in_export("generation_config.json", release, reasoning) == expected
        assert release_file_in_export("tokenizer.json", release, reasoning) == release, "only this file has a mode"


@pytest.mark.parametrize("release_bytes", [b'{"eos_token_id": [100265,', b"\xff\xfe"], ids=["not JSON", "not UTF-8"])
def test_a_release_generation_config_that_is_not_json_text_is_a_value_error(release_bytes):
    with pytest.raises(ValueError):
        release_file_in_export("generation_config.json", release_bytes, reasoning=False)
    assert release_file_in_export("tokenizer.json", release_bytes, reasoning=False) == release_bytes


# The bridge is the recording stand-in of tests/unit_tests/recorded_bridge.py: building and loading a sharded Megatron
# model needs GPUs and a distributed world. The generation config it holds is a real transformers GenerationConfig.
def _export_multi_gpu(monkeypatch, tmp_path):
    import megatron.bridge.models.decorators as decorators

    iteration = make_checkpoint_dir(tmp_path / "run", [4], tracker=4, run_config="model: {}\n") / "iter_0000004"
    # torchrun_main initialises nothing here but ends the process on an exception; the test needs neither.
    monkeypatch.setattr(decorators, "torchrun_main", lambda fn: fn)
    hf.convert_multi_gpu(
        load_path=iteration,
        hf_path=tmp_path / "hf",
        hf_model_id=NANO,
        tp=1,
        pp=2,
        ep=4,
        etp=1,
        strict=False,
        show_progress=False,
    )


def _export_single_process(monkeypatch, tmp_path):
    hf.convert_single_process(
        load_path=tmp_path, hf_path=tmp_path / "hf", hf_model_id=NANO, strict=False, show_progress=False
    )


def test_a_multi_gpu_export_saves_a_snapshot_generation_config_that_transformers_accepts(monkeypatch, tmp_path):
    trace = recorded_bridge.install(
        monkeypatch, snapshot_generation_config=GenerationConfig.from_dict(THINK_SFT_GENERATION_CONFIG)
    )
    _export_multi_gpu(monkeypatch, tmp_path)
    (snapshot,) = trace.snapshots
    snapshot.generation_config.save_pretrained(tmp_path / "saved")
    assert (tmp_path / "saved" / "generation_config.json").is_file()


def test_a_multi_gpu_export_of_a_snapshot_without_a_generation_config_leaves_it_without_one(monkeypatch, tmp_path):
    trace = recorded_bridge.install(monkeypatch)
    _export_multi_gpu(monkeypatch, tmp_path)
    (snapshot,) = trace.snapshots
    assert snapshot.generation_config is None
    assert not [call for call in trace.calls if "generation_config" in call[0]]


def test_a_single_process_export_has_no_snapshot_generation_config_to_make_saveable(monkeypatch, tmp_path):
    """``AutoBridge.from_auto_config`` builds the bridge from a synthesized transformers config, which has no
    ``generation_config`` attribute, so the export has no snapshot to change (transformers 5.3.0 raises
    AttributeError on ``Olmo3Config().generation_config``)."""
    trace = recorded_bridge.install(monkeypatch)
    _export_single_process(monkeypatch, tmp_path)
    assert not trace.snapshots
    assert [call[0] for call in trace.calls if call[0] == "bridge.export_ckpt"] == ["bridge.export_ckpt"]
    assert not [call for call in trace.calls if "generation_config" in call[0]]

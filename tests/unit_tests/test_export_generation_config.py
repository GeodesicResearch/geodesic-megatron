# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The export saves the snapshot's generation config through transformers, which refuses an invalid one."""

import json

import pytest
from transformers import GenerationConfig

from pipeline.export import hf
from pipeline.export.decoding_defaults import SAMPLING_KEYS, generation_config_text, release_file_in_export
from pipeline.export.hf import without_sampling_fields
from tests.unit_tests import recorded_bridge
from tests.unit_tests.checkpoint_dirs import make_checkpoint_dir
from tests.unit_tests.hf_export_fixtures import NANO


#: allenai/Olmo-3-7B-Think-SFT's generation_config.json: it sets temperature and top_p and not do_sample.
THINK_SFT_GENERATION_CONFIG = {
    "_from_model_config": True,
    "eos_token_id": [100265, 100257],
    "pad_token": 100277,
    "transformers_version": "4.53.1",
    "temperature": 0.6,
    "top_p": 0.95,
    "max_new_tokens": 32768,
}


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


def test_only_the_generation_config_of_a_release_takes_the_modes_decoding_defaults():
    release = json.dumps(THINK_SFT_GENERATION_CONFIG, indent=4).encode()
    for reasoning in (False, True):
        expected = generation_config_text(release.decode(), reasoning).encode()
        assert release_file_in_export("generation_config.json", release, reasoning) == expected
        assert release_file_in_export("tokenizer.json", release, reasoning) == release, "only this file has a mode"
    assert release_file_in_export("generation_config.json", release, None) == release, "no mode, no change"


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


EXPORTS = pytest.mark.parametrize(
    "export", [_export_multi_gpu, _export_single_process], ids=["multi_gpu", "single_process"]
)


@EXPORTS
def test_an_export_saves_a_snapshot_generation_config_that_transformers_accepts(monkeypatch, tmp_path, export):
    trace = recorded_bridge.install(
        monkeypatch, snapshot_generation_config=GenerationConfig.from_dict(THINK_SFT_GENERATION_CONFIG)
    )
    export(monkeypatch, tmp_path)
    (snapshot,) = trace.snapshots
    snapshot.generation_config.save_pretrained(tmp_path / "saved")
    assert (tmp_path / "saved" / "generation_config.json").is_file()


@EXPORTS
def test_an_export_of_a_snapshot_without_a_generation_config_leaves_it_without_one(monkeypatch, tmp_path, export):
    trace = recorded_bridge.install(monkeypatch)
    export(monkeypatch, tmp_path)
    (snapshot,) = trace.snapshots
    assert snapshot.generation_config is None
    assert not [call for call in trace.calls if "generation_config" in call[0]]

# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The token-masking canaries in ``configs/token_masking/canary/`` are the runs their README describes.

The CPT control is the enabled CPT canary with masking off and nothing else changed, measuring the ids its twin masks;
every canary masks or measures the two fyn1668 stage tags; and the SFT canary reads the packs that already exist,
whose directory is named after the tokenizer they were built with rather than the run's tokenizer.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from scripts.training.config_compose import load_composed_yaml

from megatron.bridge.data.builders.finetuning_dataset import FinetuningDatasetBuilder
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.training.token_masking.config import TokenMaskingConfig, validate_token_masking
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.training.utils.omegaconf_utils import _apply_overrides
from tests.unit_tests.token_masking_fixtures import build_tiny_hf_tokenizer, hf_tokenizer_config


CANARY_DIR = Path(__file__).resolve().parents[2] / "configs" / "token_masking" / "canary"
CPT_ENABLED = CANARY_DIR / "nemotron_nano_30b_fyn1668_cpt_canary.yaml"
CPT_CONTROL = CANARY_DIR / "nemotron_nano_30b_fyn1668_cpt_canary_control.yaml"
SFT_ENABLED = CANARY_DIR / "nemotron_nano_30b_fyn1668_sft_canary.yaml"
STAGE_TAG_IDS = [131072, 131073]  # <stage=training>, </stage=training>


def token_masking(path: Path) -> TokenMaskingConfig:
    """The canary's ``token_masking`` block as the launcher applies it and ``ConfigContainer.validate`` checks it."""
    block = TokenMaskingConfig()
    _apply_overrides(block, copy.deepcopy(load_composed_yaml(path)["token_masking"]))
    validate_token_masking(block)
    return block


@pytest.mark.parametrize(
    ("path", "enabled"),
    [(CPT_ENABLED, True), (CPT_CONTROL, False), (SFT_ENABLED, True)],
    ids=["cpt-enabled", "cpt-control", "sft-enabled"],
)
def test_each_canary_masks_or_measures_the_stage_tags(path, enabled):
    block = token_masking(path)
    assert block.enabled is enabled
    assert block.token_ids == (STAGE_TAG_IDS if enabled else [])
    assert block.measured_token_ids == STAGE_TAG_IDS


def test_cpt_control_is_the_enabled_canary_with_masking_off():
    enabled = load_composed_yaml(CPT_ENABLED)
    control = load_composed_yaml(CPT_CONTROL)
    differing = sorted(
        section for section in enabled.keys() | control.keys() if enabled.get(section) != control.get(section)
    )
    assert differing == ["logger", "token_masking"]
    assert {**enabled["logger"], "wandb_exp_name": control["logger"]["wandb_exp_name"]} == control["logger"]
    assert control["token_masking"] == {
        "enabled": False,
        "token_ids": [],
        "masked_validation": {"token_ids": enabled["token_masking"]["token_ids"]},
    }


def test_sft_canary_names_the_pack_directory_its_packs_live_in(tmp_path):
    dataset = load_composed_yaml(SFT_ENABLED)["dataset"]
    specs = dict(dataset["packed_sequence_specs"])
    packed_train_data_path = Path(specs.pop("packed_train_data_path"))
    # The builder derives the pack directory from the specs and the tokenizer; a root under tmp_path keeps the
    # directory it creates there. The tokenizer is only consulted when tokenizer_model_name is unset.
    tokenizer = build_tokenizer(hf_tokenizer_config(build_tiny_hf_tokenizer(tmp_path / "tokenizer")))
    builder = FinetuningDatasetBuilder(
        dataset_root=tmp_path / "root", tokenizer=tokenizer, packed_sequence_specs=PackedSequenceSpecs(**specs)
    )

    derived = builder.train_path_packed.relative_to(tmp_path / "root")
    assert derived == packed_train_data_path.relative_to(dataset["dataset_root"])
    assert builder.pack_metadata.parent == builder.train_path_packed.parent


def test_sft_canary_packs_exist_where_it_reads_them():
    dataset = load_composed_yaml(SFT_ENABLED)["dataset"]
    if not Path(dataset["dataset_root"]).is_dir():
        pytest.skip(f"{dataset['dataset_root']} is on Isambard's shared storage")
    specs = dataset["packed_sequence_specs"]
    pack_directory = Path(specs["packed_train_data_path"]).parent
    assert Path(specs["packed_train_data_path"]).is_file()
    assert (pack_directory / f"{specs['packed_sequence_size']}_metadata.jsonl").is_file()

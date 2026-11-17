"""Unit tests for the .bin/.idx data of a cpt or pretrain launch (pipeline/training/datasets.py).

``bin_idx_dataset_config`` runs on the dataset section of a merged config, as the launcher calls it.
"""

from __future__ import annotations

import pytest

from pipeline.training.datasets import BIN_IDX_MODES, bin_idx_dataset_config


BLEND = {"data_path": ["1.0", "/data/corpus/tokenized_base_input_document"]}


@pytest.mark.parametrize("mode", BIN_IDX_MODES)
def test_a_bin_idx_dataset_reads_its_token_files_as_its_config_states(mode):
    """A config can state how a blend is read: with plain reads, without a mask per sample, with more workers."""
    stated = {"mmap_bin_files": False, "create_attention_mask": False, "num_workers": 8}

    config = bin_idx_dataset_config(BLEND | {"bin_idx_read": stated}, mode)

    assert {key: getattr(config, key) for key in stated} == stated


def test_a_bin_idx_dataset_keeps_the_settings_its_config_leaves_out():
    """Other keys of the section, such as a worker count a recipe put there, do not change how the blend is read."""
    config = bin_idx_dataset_config(
        BLEND | {"num_workers": 1, "bin_idx_read": {"create_attention_mask": False}}, "cpt"
    )

    assert (config.mmap_bin_files, config.create_attention_mask, config.num_workers) == (True, False, 2)


def test_a_bin_idx_dataset_refuses_a_read_setting_it_does_not_carry():
    with pytest.raises(ValueError, match="pin_memory"):
        bin_idx_dataset_config(BLEND | {"bin_idx_read": {"pin_memory": False}}, "pretrain")

# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.

"""The Nemotron 3 Nano Dolci-SFT config is the Nemotron 3 Nano Norm-SFT config with only DOLCI_FIELDS changed.

The run fine-tunes NVIDIA's Nemotron 3 Nano Base for two epochs on AI2's Dolci Think SFT data at the recipe of the
Nemotron 3 Nano Norm-SFT. These tests pin the fields that differ, the order in which the loader reads the two epochs,
the length, the saves and the width. They also pin that the data configs of the second pair of packs build the
first pair's rows again from other seeds and change nothing else.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from scripts.training.launch_width import config_launch_width

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_sft_config
from pipeline.data.layout import pack_dir
from tests.unit_tests.campaign_config import (
    IDENTITY,
    SFT_CORPUS_FIELDS,
    assert_it_saves_to_a_directory_of_its_own,
    assert_one_pass_of_full_batches,
    assert_only_these_fields_differ,
    batches_the_loader_reads,
    merge_onto_recipe,
)


_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
_DATA = _CONFIGS / "sft_mix_grid" / "data"
DOLCI = _CONFIGS / "sft_mix_grid" / "nemotron_nano_dolci_sft.yaml"
NORM = (
    _CONFIGS
    / "control_pretraining"
    / "30b_baseline_ablations"
    / "nemotron_nano_30b_nvidia_base_sft_xl50b_gbs256_v4.yaml"
)
_DOLCI_ROOT = Path("/projects/a5k/public/data/allenai__Dolci-Think-SFT-7B__reasoning_content")
TOKENIZER = "geodesic-research/nemotron-think-history-tokenizer"
PAD_SEQ_TO_MULT = 4
BATCH = 256
#: The packs of each epoch, in the loader's order: `totals.packs` of each epoch's pack audit report
#: (configs/sft_mix_grid/data/dolci_think_sft_nano_epoch{1,2}_audit_m4.yaml).
EPOCH_PACKS = (707_652, 707_654)
#: What makes this run another run than the Nemotron 3 Nano Norm-SFT: its corpus, its length, its run identity and the
#: number of saves it keeps.
DOLCI_FIELDS = IDENTITY | SFT_CORPUS_FIELDS | {"train.train_iters", "checkpoint.most_recent_k"}
#: The keys in which a data config of the second pair of packs differs from the first pair's config of the same step
#: and epoch.
RUN_TWO_DATA_KEYS = {"partition": {"output_root", "seed"}, "pack_m4": {"dataset_root"}, "audit_m4": {"parquets"}}
#: The partition seed of each pack set. No two sets share a seed, or two of them would hold the rows in one order.
PARTITION_SEEDS = {"epoch1": 1, "epoch2": 2, "run2_epoch1": 3, "run2_epoch2": 4}


def merged(path: Path):
    """The run's config as the launcher resolves it: the Nano SFT recipe with the YAML merged on."""
    return merge_onto_recipe(path, nemotron_3_nano_sft_config)


@pytest.fixture(scope="module")
def dolci():
    return merged(DOLCI)


def test_only_the_corpus_length_identity_and_saves_kept_differ_from_the_norm_sft_config(dolci):
    assert_only_these_fields_differ(dolci, merged(NORM), DOLCI_FIELDS, "nemotron_nano_dolci_sft")


def test_it_reads_epoch_one_then_epoch_two_at_the_norm_sfts_tokenizer_and_padding_multiple(dolci):
    specs = dolci.dataset.packed_sequence_specs
    assert specs.packed_train_data_path == [
        str(pack_dir(f"{_DOLCI_ROOT}/epoch{epoch}/shard*", TOKENIZER, PAD_SEQ_TO_MULT) / "training_32768.idx.parquet")
        for epoch in (1, 2)
    ]
    assert (dolci.tokenizer.tokenizer_model, specs.pad_seq_to_mult) == (TOKENIZER, PAD_SEQ_TO_MULT)


def test_it_saves_every_1200_iterations_keeping_two_in_a_directory_of_its_own(dolci):
    assert (dolci.checkpoint.save_interval, dolci.checkpoint.most_recent_k) == (1200, 2)
    assert_it_saves_to_a_directory_of_its_own(dolci, "nemotron_nano_dolci_sft", [merged(NORM)])


def test_the_loader_reads_each_full_batch_of_both_epochs_once(dolci):
    batches = batches_the_loader_reads(sum(EPOCH_PACKS), BATCH, dolci.train.train_iters)
    assert_one_pass_of_full_batches(batches, sum(EPOCH_PACKS), BATCH, dolci.train.train_iters)


def test_it_trains_on_64_nodes_with_each_gpu_a_data_parallel_replica():
    width = config_launch_width(str(DOLCI))
    assert (width.nodes, width.gpus_per_node, width.data_parallel_size) == (64, 4, 256)


def _data_config(pack_set: str, step: str) -> dict:
    return yaml.safe_load((_DATA / f"dolci_think_sft_nano_{pack_set}_{step}.yaml").read_text())


@pytest.mark.parametrize("epoch", [1, 2])
@pytest.mark.parametrize("step", sorted(RUN_TWO_DATA_KEYS))
def test_the_second_pairs_data_configs_differ_from_the_first_pairs_in_their_folder_and_seed_only(step, epoch):
    """The second pair of packs holds the first pair's rows, with the same exclusions, packing and audit lists."""
    first, second = _data_config(f"epoch{epoch}", step), _data_config(f"run2_epoch{epoch}", step)
    assert first.keys() == second.keys()
    assert {key for key in first if first[key] != second[key]} == RUN_TWO_DATA_KEYS[step]
    folder = f"{_DOLCI_ROOT}/run2_epoch{epoch}"
    assert all(str(second[key]).startswith(folder) for key in RUN_TWO_DATA_KEYS[step] - {"seed"})


def test_every_pack_set_is_partitioned_from_a_seed_of_its_own():
    assert {pack_set: _data_config(pack_set, "partition")["seed"] for pack_set in PARTITION_SEEDS} == PARTITION_SEEDS

# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.

"""The Olmo 3 7B Dolci-SFT config is the Olmo-Norm 7B config with only its data, length, saves and run identity changed.

The run fine-tunes Olmo 3 7B Base for two epochs on AI2's Dolci Think SFT data at the recipe of olmo3_7b_norm_sft.yaml.
These tests pin the fields that differ, and no other field can differ. They also pin the order in which the loader
reads the two epochs, the length, the iteration at which epoch 1 ends, the width and the code the run trains with.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.training.code_identity import config_code_identity
from scripts.training.launch_width import config_launch_width

from pipeline.data.layout import pack_dir
from pipeline.training.recipes import OLMO3_BASE_COPIES, RECIPE_MAP
from tests.unit_tests.campaign_config import (
    IDENTITY,
    assert_it_saves_to_a_directory_of_its_own,
    assert_only_these_fields_differ,
    assert_the_pinned_code_is_sound,
    batches_the_loader_reads,
    merge_onto_recipe,
)


_ARM_DIR = Path(__file__).resolve().parents[2] / "configs" / "olmo3_norm_sft"
DOLCI = _ARM_DIR / "olmo3_7b_dolci_sft.yaml"
NORM = _ARM_DIR / "olmo3_7b_norm_sft.yaml"
_DOLCI_ROOT = Path("/projects/a5k/public/data/allenai__Dolci-Think-SFT-7B")
TOKENIZER = "/projects/a5k/public/logs/metagaming_filtering/olmo_norm/tokenizers/tokenizer_7b_fast"
PAD_SEQ_TO_MULT = 8
BATCH = 32
#: The packs of each epoch, in the loader's order, as the pack audits counted them (verdict PASS, reports
#: epoch1_m8.json and epoch2_m8.json under /projects/a5k/public/logs/metagaming_filtering/olmo3_dolci_replication/
#: megatron/audits).
EPOCH_PACKS = (693_842, 693_835)
#: The last iteration made only of epoch-1 packs. The launch lines of the first two segments set train.exit_interval to
#: it.
EPOCH_ONE_END = 21_682
#: What makes this run another run than the Olmo-Norm 7B: its data, its length, its saves and its run identity.
DOLCI_FIELDS = IDENTITY | {
    "dataset.dataset_name",
    "dataset.dataset_root",
    "dataset.packed_sequence_specs.packed_train_data_path",
    "train.train_iters",
    "checkpoint.save_interval",
    "checkpoint.most_recent_k",
    "logger.peak_memory_report_interval",
}

needs_base_copy = pytest.mark.skipif(
    not (Path(OLMO3_BASE_COPIES["olmo3_7b"]) / "config.json").is_file(),
    reason="needs the 7B base copy's config.json (the recipe sizes the model from it)",
)


def merged(path: Path):
    """The run's config as the launcher resolves it: the 7B recipe with the YAML merged on."""
    return merge_onto_recipe(path, lambda: RECIPE_MAP[("olmo3_7b", "sft")](None))


@pytest.fixture(scope="module")
def dolci():
    return merged(DOLCI)


@pytest.fixture(scope="module")
def batches(dolci):
    """The pack indices of every batch the run reads, from both epochs' packs."""
    return batches_the_loader_reads(sum(EPOCH_PACKS), BATCH, dolci.train.train_iters)


@needs_base_copy
def test_only_the_data_length_saves_and_identity_differ_from_the_norm_7b_config(dolci):
    assert_only_these_fields_differ(dolci, merged(NORM), DOLCI_FIELDS, "olmo3_7b_dolci_sft")


@needs_base_copy
def test_it_reads_epoch_one_then_epoch_two_at_its_tokenizer_and_padding_multiple(dolci):
    specs = dolci.dataset.packed_sequence_specs
    assert specs.packed_train_data_path == [
        str(pack_dir(f"{_DOLCI_ROOT}/epoch{epoch}/shard*", TOKENIZER, PAD_SEQ_TO_MULT) / "training_32768.idx.parquet")
        for epoch in (1, 2)
    ]
    assert dolci.tokenizer.tokenizer_model == TOKENIZER
    assert specs.pad_seq_to_mult == PAD_SEQ_TO_MULT


@needs_base_copy
def test_it_saves_every_2000_iterations_keeping_two_in_a_directory_of_its_own(dolci):
    assert (dolci.checkpoint.save_interval, dolci.checkpoint.most_recent_k) == (2000, 2)
    assert_it_saves_to_a_directory_of_its_own(dolci, "olmo3_7b_dolci_sft", [merged(NORM)])


@needs_base_copy
def test_the_loader_reads_each_full_batch_of_both_epochs_once(dolci, batches):
    """train_iters is the packs of both epochs over the batch, rounded down, so the loader never starts a third pass."""
    packs = sum(EPOCH_PACKS)
    assert dolci.train.train_iters == packs // BATCH
    read = [index for indices in batches for index in indices]
    assert read == list(range(dolci.train.train_iters * BATCH))
    assert packs - len(read) == packs % BATCH


@needs_base_copy
def test_epoch_one_ends_at_the_iteration_the_launch_lines_name(batches):
    """Iteration EPOCH_ONE_END is the last made only of epoch-1 packs, and the next one reads epoch 2."""
    assert max(batches[EPOCH_ONE_END - 1]) < EPOCH_PACKS[0] <= max(batches[EPOCH_ONE_END])


def test_it_trains_on_8_nodes_with_each_gpu_a_data_parallel_replica():
    width = config_launch_width(str(DOLCI))
    assert (width.nodes, width.gpus_per_node, width.data_parallel_size) == (8, 4, 32)


def test_the_pinned_code_is_sound():
    assert_the_pinned_code_is_sound(config_code_identity(str(DOLCI)))

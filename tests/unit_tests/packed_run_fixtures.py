# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Packed-run builders for tests: packed parquet files on disk and the config of a run that trains on them.

Catalogue:
    SEQ: the sequence length of the model of a test run.
    PACK_SIZE: the length of a pack.
    GLOBAL_BATCH: the global batch size of a test run.
    MICRO_BATCH: the micro batch size of a test run.
    write_packs: write rows as a packed parquet file.
    packed_dataset_config: the dataset config of a packed SFT run.
    run_config: a run's ConfigContainer.
"""

from __future__ import annotations

from pathlib import Path

from megatron.bridge.data.datasets.packed_parquet import write_packed_parquet
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    FinetuningDatasetConfig,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)


SEQ = 16
PACK_SIZE = 64
GLOBAL_BATCH, MICRO_BATCH = 4, 2


def write_packs(path: Path, rows: list[dict]) -> str:
    """Write ``rows`` (``input_ids``, ``loss_mask``, ``seq_start_id``), one pack each, as a packed parquet file at
    ``path`` and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_packed_parquet(rows, path, row_group_size=1)
    return str(path)


def packed_dataset_config(tmp_path: Path, training_path: str) -> FinetuningDatasetConfig:
    """The dataset config of a packed SFT run that trains on the packs at ``training_path``."""
    return FinetuningDatasetConfig(
        dataset_root=tmp_path,
        seq_length=PACK_SIZE,
        dataset_kwargs={"answer_only_loss": True},
        packed_sequence_specs=PackedSequenceSpecs(
            packed_sequence_size=PACK_SIZE, packed_train_data_path=training_path, tokenizer_model_name="org--tiny"
        ),
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )


def run_config(dataset, tokenizer: TokenizerConfig, **sections) -> ConfigContainer:
    """A run's config, as ``ConfigContainer`` holds it, with ``sections`` (``token_masking``, ``held_out``) set; the
    held-out sets read its dataset, train and logger."""
    return ConfigContainer(
        train=TrainingConfig(global_batch_size=GLOBAL_BATCH, micro_batch_size=MICRO_BATCH, train_iters=10),
        model=GPTModelProvider(num_layers=1, hidden_size=64, num_attention_heads=4, seq_length=SEQ),
        optimizer=OptimizerConfig(lr=1e-4, use_distributed_optimizer=False),
        scheduler=SchedulerConfig(lr_decay_style="linear", lr_warmup_iters=0),
        dataset=dataset,
        logger=LoggerConfig(),
        tokenizer=tokenizer,
        checkpoint=CheckpointConfig(),
        **sections,
    )

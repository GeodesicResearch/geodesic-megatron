# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The configs of a tiny pretraining run whose datasets tests build on CPU.

Catalogue:
    gpt_dataset_config: a finalized ``GPTDatasetConfig`` over ``.bin/.idx`` prefixes, with a ``NullTokenizer``.
    run_config: the ``ConfigContainer`` of a one-layer GPT run over a dataset config.
"""

from __future__ import annotations

from pathlib import Path

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    GPTDatasetConfig,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from tests.unit_tests.token_masking_fixtures import null_tokenizer_config


def gpt_dataset_config(
    cache: Path,
    data_path: list[str] | None,
    *,
    seq_length: int,
    vocab_size: int,
    split: str,
    config_class: type[GPTDatasetConfig] = GPTDatasetConfig,
) -> GPTDatasetConfig:
    """A finalized ``config_class`` over ``data_path`` with a ``NullTokenizer`` of ``vocab_size`` ids, seed 1234, no
    loader workers, and its index caches under ``cache``."""
    config = config_class(
        seq_length=seq_length,
        data_path=data_path,
        split=split,
        random_seed=1234,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=build_tokenizer(null_tokenizer_config(vocab_size)),
        path_to_cache=str(cache),
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    config.finalize()
    return config


def run_config(
    dataset,
    tokenizer: TokenizerConfig,
    *,
    global_batch: int,
    micro_batch: int,
    seq_length: int,
    **sections,
) -> ConfigContainer:
    """The ``ConfigContainer`` of a ten-iteration run of a one-layer GPT over ``dataset``, with ``sections`` (such as
    ``token_masking`` or ``held_out_validation``) as further fields."""
    return ConfigContainer(
        train=TrainingConfig(global_batch_size=global_batch, micro_batch_size=micro_batch, train_iters=10),
        model=GPTModelProvider(num_layers=1, hidden_size=64, num_attention_heads=4, seq_length=seq_length),
        optimizer=OptimizerConfig(lr=1e-4, use_distributed_optimizer=False),
        scheduler=SchedulerConfig(lr_decay_style="linear", lr_warmup_iters=0),
        dataset=dataset,
        logger=LoggerConfig(),
        tokenizer=tokenizer,
        checkpoint=CheckpointConfig(),
        **sections,
    )

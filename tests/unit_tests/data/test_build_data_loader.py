# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""``loaders.build_data_loader``: the one place a run's data loaders are built.

The training, validation and test loaders and the held-out masked-validation loader all come from it, so each one
reads with the run's batch sizes and its dataset's ``collate_fn``, and installs the exit-signal handler in its workers
when ``train.exit_signal_handler_for_dataloader`` is set. Each loader here is real, over a real ``ConfigContainer``,
and its worker is a real process.
"""

import signal

import pytest
from torch.utils.data import Dataset

from megatron.bridge.data.loaders import build_data_loader
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    LoggerConfig,
    MockGPTDatasetConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)


GLOBAL_BATCH, MICRO_BATCH = 4, 2
HANDLER = "DistributedSignalHandler.__enter__.<locals>.handler"


def _config(*, exit_signal_handler: bool, num_workers: int) -> ConfigContainer:
    return ConfigContainer(
        train=TrainingConfig(
            global_batch_size=GLOBAL_BATCH,
            micro_batch_size=MICRO_BATCH,
            train_iters=1,
            exit_signal_handler_for_dataloader=exit_signal_handler,
        ),
        model=GPTModelProvider(num_layers=1, hidden_size=64, num_attention_heads=4, seq_length=16),
        optimizer=OptimizerConfig(lr=1e-4, use_distributed_optimizer=False),
        scheduler=SchedulerConfig(),
        dataset=MockGPTDatasetConfig(
            seq_length=16,
            random_seed=1234,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
        logger=LoggerConfig(),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=64),
        checkpoint=CheckpointConfig(),
    )


class _ExitSignalHandlers(Dataset):
    """Each sample is the name of the handler the process reading it has for SIGTERM, the default exit signal."""

    def __len__(self) -> int:
        return 2 * GLOBAL_BATCH

    def __getitem__(self, index: int) -> str:
        handler = signal.getsignal(signal.SIGTERM)
        return getattr(handler, "__qualname__", repr(handler))


class _SelfCollating(Dataset):
    """Integer samples that collate themselves into a tagged list."""

    def __len__(self) -> int:
        return 2 * GLOBAL_BATCH

    def __getitem__(self, index: int) -> int:
        return index

    @staticmethod
    def collate_fn(samples: list[int]) -> dict[str, list[int]]:
        return {"collated": samples}


@pytest.mark.parametrize("exit_signal_handler", [True, False], ids=["handler", "no-handler"])
def test_a_worker_handles_the_exit_signal_exactly_when_configured(exit_signal_handler):
    cfg = _config(exit_signal_handler=exit_signal_handler, num_workers=1)
    assert cfg.train.exit_signal == signal.SIGTERM
    loader = build_data_loader(_ExitSignalHandlers(), 0, "single", cfg, 0, 1, persistent_workers=False)
    handlers = {handler for batch in loader for handler in batch}
    assert (handlers == {HANDLER}) is exit_signal_handler


def test_batches_are_the_runs_micro_batches_collated_by_the_dataset_in_this_ranks_share():
    cfg = _config(exit_signal_handler=False, num_workers=0)
    shares = [
        list(build_data_loader(_SelfCollating(), 0, "single", cfg, rank, 2, persistent_workers=False))
        for rank in (0, 1)
    ]
    # Each step of the data-parallel group reads MICRO_BATCH samples per rank, in order.
    assert shares == [
        [{"collated": [0, 1]}, {"collated": [4, 5]}],
        [{"collated": [2, 3]}, {"collated": [6, 7]}],
    ]

# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Token masking end to end: the real ``pretrain`` on GPUs, two real ``.bin/.idx`` sources, W&B offline.

A tiny GPT trains for a few iterations on a blend of two corpora written here with Megatron's own
``IndexedDatasetBuilder``: one holds a marker token every few dozen tokens, the other none. Each test then checks
what the run actually did, from the reports every iteration produced (captured by a callback), the run's state, and
the files W&B wrote:

- ``mode: enabled`` masks every marker target, on both data-parallel ranks and with the pipeline split over two
  stages, and the monitor sees it from the first iteration;
- ``mode: disabled`` counts the markers and trains on them;
- the W&B summary records the decision and its verification, and the sample tables hold ten documents per source
  plus the documents that contain the marker;
- an enforced run whose data holds no marker stops at setup, on every rank, before the model is built;
- an enforced segment shorter than its masking deadline that masked nothing fails when it ends.

Run on two GPUs: ``torchrun --nproc_per_node=2 -m pytest tests/functional_tests/test_groups/training/
test_token_masking.py``.
"""

import glob
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.training.callbacks import Callback
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DataSamplesConfig,
    DistributedDataParallelConfig,
    GPTDatasetConfig,
    LoggerConfig,
    OptimizerConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.initialize import destroy_global_state
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.token_masking.config import TokenMaskingConfig, TokenMaskingError
from megatron.bridge.training.token_masking.hook import (
    LISTED_TARGET_FRACTION,
    MASKED_TARGET_FRACTION,
    REPORT_KEYS,
    TRAINED_LISTED_TARGET_FRACTION,
)
from tests.functional_tests.utils import broadcast_path, clear_directories, initialize_distributed


VOCAB_SIZE = 1024  # NullTokenizer(1024): ids 0..1023, its end-of-document id is 1023
EOD = VOCAB_SIZE - 1
MARKER = 1000
SEQ_LENGTH = 128
TRAIN_ITERS = 6
DOCUMENTS_PER_CORPUS = 200


def _documents(seed: int, with_marker: bool) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    documents = []
    for _ in range(DOCUMENTS_PER_CORPUS):
        document = rng.integers(10, 900, size=int(rng.integers(60, 200))).tolist()
        if with_marker:
            for position in range(5, len(document), 30):
                document[position] = MARKER
        documents.append(document + [EOD])
    return documents


def _write_corpus(prefix: Path, documents: list[list[int]]) -> str:
    builder = IndexedDatasetBuilder(f"{prefix}.bin", dtype=np.int32)
    for document in documents:
        builder.add_item(torch.tensor(document, dtype=torch.int32))
        builder.end_document()
    builder.finalize(f"{prefix}.idx")
    return str(prefix)


class _RecordReports(Callback):
    """Keeps every iteration's reduced reports, the run's state and its W&B summary, to inspect after ``pretrain``.

    The summary is copied when training ends, while the W&B run is still open (offline runs keep it in their binary
    log, not in a readable file); ``None`` on the ranks without W&B.
    """

    def __init__(self) -> None:
        self.reports: list[dict[str, float]] = []
        self.state = None
        self.wandb_summary: dict | None = None

    def on_train_step_end(self, context) -> None:
        self.reports.append({key: float(value) for key, value in (context.loss_dict or {}).items()})

    def on_train_end(self, context) -> None:
        self.state = context.state
        wandb_logger = context.state.wandb_logger
        if wandb_logger is not None:
            self.wandb_summary = dict(wandb_logger.run.summary)


def _config(
    base: Path, data_path: list[str], token_masking: TokenMaskingConfig, pipeline_parallel: int
) -> ConfigContainer:
    model = GPTModelProvider(
        normalization="RMSNorm",
        activation_func=F.silu,
        gated_linear_unit=True,
        position_embedding_type="rope",
        add_bias_linear=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        num_layers=2,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=4,
        num_query_groups=4,
        init_method_std=0.02,
        share_embeddings_and_output_weights=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pipeline_parallel,
        context_parallel_size=1,
        sequence_parallel=False,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        seq_length=SEQ_LENGTH,
        make_vocab_size_divisible_by=128,
        vocab_size=None,
    )
    return ConfigContainer(
        model=model,
        train=TrainingConfig(train_iters=TRAIN_ITERS, global_batch_size=8, micro_batch_size=1),
        validation=ValidationConfig(eval_interval=None, eval_iters=0),
        optimizer=OptimizerConfig(
            optimizer="adam",
            bf16=True,
            lr=1e-3,
            min_lr=1e-4,
            weight_decay=0.01,
            use_distributed_optimizer=True,
            clip_grad=1.0,
        ),
        scheduler=SchedulerConfig(
            start_weight_decay=0.01,
            end_weight_decay=0.01,
            weight_decay_incr_style="constant",
            lr_decay_style="cosine",
            lr_warmup_iters=1,
            lr_warmup_init=0.0,
            lr_decay_iters=TRAIN_ITERS,
            override_opt_param_scheduler=True,
        ),
        ddp=DistributedDataParallelConfig(use_distributed_optimizer=True, grad_reduce_in_fp32=True),
        dataset=GPTDatasetConfig(
            random_seed=1234,
            seq_length=SEQ_LENGTH,
            data_path=data_path,
            split="1,0,0",
            path_to_cache=str(base / "index_cache"),
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=False,
            num_dataset_builder_threads=1,
            dataloader_type="single",
            num_workers=1,
        ),
        logger=LoggerConfig(
            log_interval=1,
            wandb_project="token-masking-functional",
            wandb_exp_name="token-masking-functional",
            wandb_save_dir=str(base / "wandb"),
            data_samples=DataSamplesConfig(max_scan_tokens_per_source=200_000),
        ),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=VOCAB_SIZE),
        token_masking=token_masking,
        checkpoint=CheckpointConfig(save=None, load=None),
        rng=RNGConfig(seed=1234),
    )


@pytest.fixture
def corpora(tmp_path, monkeypatch):
    """Two corpora (one with markers) in a directory every rank sees, and W&B in offline mode (no network)."""
    initialize_distributed()
    base = Path(broadcast_path(tmp_path))
    if torch.distributed.get_rank() == 0:
        _write_corpus(base / "marked", _documents(seed=1, with_marker=True))
        _write_corpus(base / "plain", _documents(seed=2, with_marker=False))
    torch.distributed.barrier()
    monkeypatch.setenv("WANDB_MODE", "offline")
    yield base
    clear_directories(base)  # collective: every rank calls it; it barriers and only rank 0 deletes


def _train(base: Path, data_path: list[str], token_masking: TokenMaskingConfig, pipeline_parallel: int = 1):
    recorder = _RecordReports()
    pretrain(_config(base, data_path, token_masking, pipeline_parallel), forward_step, callbacks=[recorder])
    return recorder


def _blend(base: Path) -> list[str]:
    return ["0.5", str(base / "marked"), "0.5", str(base / "plain")]


def _wandb_table_files(base: Path) -> list[str]:
    """The table files of the offline W&B run, which W&B writes when a table is logged."""
    import wandb

    wandb.finish()
    files = glob.glob(str(base / "wandb" / "**" / "*.table.json"), recursive=True)
    # W&B also exposes the run under a ``latest-run`` symlink; count each file once.
    return [path for path in files if "latest-run" not in Path(path).parts]


def _table(files: list[str], name: str) -> dict:
    matches = [path for path in files if f"data_samples/{name}_" in path.replace(os.sep, "/")]
    assert len(matches) == 1, (name, files)
    return json.loads(Path(matches[0]).read_text())


class TestTokenMasking:
    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("pipeline_parallel", [1, 2], ids=["dp2", "pp2"])
    def test_enabled_run_masks_every_marker_and_records_it(self, corpora, pipeline_parallel):
        recorder = _train(
            corpora,
            _blend(corpora),
            TokenMaskingConfig(mode="enabled", token_ids=[MARKER], require_masked_targets_within_iterations=2),
            pipeline_parallel,
        )
        state = recorder.state
        assert state.token_masking.enabled and state.token_masking.token_ids == (MARKER,)
        assert len(recorder.reports) == TRAIN_ITERS
        # pretrain has torn down the model-parallel groups by now; with two ranks the last stage is the last rank.
        last_stage = pipeline_parallel == 1 or torch.distributed.get_rank() == torch.distributed.get_world_size() - 1
        for report in recorder.reports:
            if not last_stage:
                assert report == {}
                continue
            assert set(REPORT_KEYS) <= set(report)
            assert report[LISTED_TARGET_FRACTION] > 0
            assert report[MASKED_TARGET_FRACTION] == pytest.approx(report[LISTED_TARGET_FRACTION])
            assert report[TRAINED_LISTED_TARGET_FRACTION] == 0
        if last_stage:
            assert state.token_masking_monitor.first_masked_iteration == 1
        if torch.distributed.get_rank() == torch.distributed.get_world_size() - 1:
            summary, files = recorder.wandb_summary, _wandb_table_files(corpora)
            assert summary["token_masking/verified"] is True
            assert summary["token_masking/first_masked_iteration"] == 1
            assert summary["token_masking/enabled"] is True
            assert summary["token_masking/token_ids"] == [MARKER]
            sources = _table(files, "sources")
            assert len(sources["data"]) == 2
            documents = _table(files, "documents")
            labels = documents["columns"].index("source_label")
            assert sorted({row[labels] for row in documents["data"]}) == ["marked", "plain"]
            assert len(documents["data"]) == 20
            masked = _table(files, "masked_documents")
            assert masked["data"] and {row[labels] for row in masked["data"]} == {"marked"}
            text = masked["columns"].index("text")
            assert all("⟦masked:" in row[text] for row in masked["data"])

    @pytest.mark.run_only_on("GPU")
    def test_disabled_run_counts_the_markers_and_trains_on_them(self, corpora):
        recorder = _train(corpora, _blend(corpora), TokenMaskingConfig(mode="disabled", token_ids=[MARKER]))
        assert not recorder.state.token_masking.enabled
        for report in recorder.reports:
            assert report[LISTED_TARGET_FRACTION] > 0
            assert report[MASKED_TARGET_FRACTION] == 0
            assert report[TRAINED_LISTED_TARGET_FRACTION] == pytest.approx(report[LISTED_TARGET_FRACTION])
        if torch.distributed.get_rank() == torch.distributed.get_world_size() - 1:
            summary = recorder.wandb_summary
            _wandb_table_files(corpora)
            assert summary["token_masking/enabled"] is False
            assert "token_masking/verified" not in summary

    @pytest.mark.run_only_on("GPU")
    def test_enforced_run_on_data_without_the_marker_stops_at_setup(self, corpora):
        recorder = _RecordReports()
        config = _config(corpora, [str(corpora / "plain")], TokenMaskingConfig(mode="enabled", token_ids=[MARKER]), 1)
        with pytest.raises(TokenMaskingError, match="no target in any training data source"):
            pretrain(config, forward_step, callbacks=[recorder])
        assert recorder.state is None
        destroy_global_state()  # setup failed after building the parallel state, before pretrain's own teardown

    @pytest.mark.run_only_on("GPU")
    def test_enforced_segment_that_ends_before_its_deadline_with_nothing_masked_fails(self, corpora):
        """The setup scan stops at its time budget (inconclusive), so only the end-of-segment check catches it."""
        recorder = _RecordReports()
        config = _config(
            corpora,
            [str(corpora / "plain")],
            TokenMaskingConfig(
                mode="enabled", token_ids=[MARKER], require_masked_targets_within_iterations=TRAIN_ITERS + 4
            ),
            1,
        )
        config.logger.data_samples = DataSamplesConfig(max_scan_seconds=1e-6)
        config.checkpoint.save = str(corpora / "checkpoints")
        with pytest.raises(TokenMaskingError, match=f"in the {TRAIN_ITERS} iterations this segment ran"):
            pretrain(config, forward_step, callbacks=[recorder])
        assert len(recorder.reports) == TRAIN_ITERS  # the whole segment trained before the check failed it
        # The check runs once the segment's final checkpoint is written, so a resumed chain moves on.
        latest = corpora / "checkpoints" / "latest_checkpointed_iteration.txt"
        assert latest.read_text().strip() == str(TRAIN_ITERS)
        # The failure leaves train() before pretrain's own teardown, which is done here instead.
        if torch.distributed.get_rank() == torch.distributed.get_world_size() - 1:
            _wandb_table_files(corpora)
        destroy_global_state()

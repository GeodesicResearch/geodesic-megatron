# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Token masking end to end: the real ``pretrain`` on GPUs, real ``.bin/.idx`` corpora, W&B offline.

A tiny GPT trains on corpora written here with Megatron's own ``IndexedDatasetBuilder``. In the marked ones a trigger
token is always followed by the marker, so the marker is perfectly predictable from the token before it; the plain
one holds no marker. Each test checks what the run actually did, from the reports every iteration produced (captured
by a callback), the run's state, and what W&B received (read back from the offline run's own log):

- ``enabled: true`` masks every marker target, on both data-parallel ranks and with the pipeline split over two
  stages, the monitor sees it from the first iteration, and ``token_masking/listed_target_loss`` reaches W&B at every
  iteration with a marker target;
- a control arm (masking off, ``masked_validation.token_ids`` naming the marker) measures the markers and trains on
  them;
- the direction of the effect: over a hundred iterations at lr 1e-3, the cross-entropy at the marker's targets does
  not fall when masked, and falls below what any model ignoring the context could reach in the control;
- the held-out masked validation is evaluated at step 0 and every ``interval`` iterations, on the same samples each
  time, without counting as validation samples;
- an enabled run stops at setup, on every rank and before the model is built, when its training data holds no
  marker target, when its scan runs out of time before finding one, and when its held-out set holds none;
- an enabled run whose global batch holds no trainable target (every target is the marker) raises at that iteration
  rather than logging a 0/0 loss.

Run on two GPUs: ``torchrun --nproc_per_node=2 -m pytest tests/functional_tests/test_groups/training/
test_token_masking.py``.
"""

import dataclasses
import glob
import json
import math
import os
from pathlib import Path
from typing import Any

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
from megatron.bridge.training.token_masking.config import (
    MaskedValidationConfig,
    TokenMaskingConfig,
    TokenMaskingError,
)
from megatron.bridge.training.token_masking.hook import (
    LISTED_TARGET_FRACTION,
    LISTED_TARGET_LOSS,
    LISTED_TARGET_LOSS_SUM,
    LISTED_TRAINABLE_TARGET_FRACTION,
    MASKED_TARGET_FRACTION,
    REPORT_KEYS,
    TOKEN_MASKING_NAMESPACE,
    TRAINED_LISTED_TARGET_FRACTION,
)
from megatron.bridge.training.token_masking.validation import MASKED_VALIDATION_KEY_PREFIX
from tests.functional_tests.utils import broadcast_path, clear_directories, initialize_distributed


VOCAB_SIZE = 1024  # NullTokenizer(1024): ids 0..1023, its end-of-document id is 1023
EOD = VOCAB_SIZE - 1
MARKER = 1000
TRIGGER = 999  # outside the ordinary tokens 10..899, so only the marker ever follows it
MARKER_PERIOD = 16  # a trigger and the marker every 16 tokens of a marked document
SEQ_LENGTH = 128
GLOBAL_BATCH_SIZE = 8
TRAIN_ITERS = 6
DOCUMENTS_PER_CORPUS = 200
MARKED_SEED = 1
# The direction test trains long enough for a context-dependent prediction to be learned. At lr 1e-3 Adam moves each
# weight of the marker's output row by about lr per step in the direction its gradient keeps, which raises the
# marker's logit at the trigger's context by about lr * ||n(h)||_1 ~ 1e-3 * 0.8 * 128 ~ 0.1 nat per step (n(h), the
# normalised hidden state, has elements of magnitude ~1), and the trigger's embedding moves the same way: the ~4 nats
# between the untrained loss and the bar below take a few dozen steps, and 100 (the cosine schedule ends at lr 1e-4)
# leave a wide margin.
DIRECTION_ITERS = 100
DIRECTION_WINDOW = 10  # the final iterations averaged, so one batch's draw of marker targets does not decide


def _documents(seed: int, with_marker: bool) -> list[list[int]]:
    """Documents of random ordinary tokens 10..899, each ending with EOD.

    With ``with_marker``, from a random offset, every ``MARKER_PERIOD`` tokens a TRIGGER and then the MARKER: every
    TRIGGER is followed by the MARKER and only a TRIGGER is, so the marker is predictable from the token before it.
    """
    rng = np.random.default_rng(seed)
    documents = []
    for _ in range(DOCUMENTS_PER_CORPUS):
        document = rng.integers(10, 900, size=int(rng.integers(60, 200))).tolist()
        if with_marker:
            for position in range(int(rng.integers(0, MARKER_PERIOD)), len(document) - 1, MARKER_PERIOD):
                document[position : position + 2] = [TRIGGER, MARKER]
        documents.append(document + [EOD])
    return documents


def _all_marker_documents() -> list[list[int]]:
    """Documents of the marker alone, without EOD, so every target of every sample is the marker."""
    return [[MARKER] * 200 for _ in range(50)]


def _marker_frequency(documents: list[list[int]]) -> float:
    return sum(document.count(MARKER) for document in documents) / sum(len(document) for document in documents)


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
    log); ``None`` on the ranks without W&B.
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
    base: Path,
    data_path: list[str],
    token_masking: TokenMaskingConfig,
    *,
    pipeline_parallel: int = 1,
    train_iters: int = TRAIN_ITERS,
    name: str = "run",
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
        train=TrainingConfig(train_iters=train_iters, global_batch_size=GLOBAL_BATCH_SIZE, micro_batch_size=1),
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
            lr_decay_iters=train_iters,
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
            wandb_exp_name=f"token-masking-functional-{name}",
            wandb_save_dir=str(_wandb_dir(base, name)),
            data_samples=DataSamplesConfig(max_scan_tokens_per_source=200_000),
        ),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=VOCAB_SIZE),
        token_masking=token_masking,
        checkpoint=CheckpointConfig(save=None, load=None),
        rng=RNGConfig(seed=1234),
    )


@pytest.fixture
def corpora(tmp_path, monkeypatch):
    """The corpora in a directory every rank sees, and W&B in offline mode (no network)."""
    initialize_distributed()
    base = Path(broadcast_path(tmp_path))
    if torch.distributed.get_rank() == 0:
        _write_corpus(base / "marked", _documents(seed=MARKED_SEED, with_marker=True))
        _write_corpus(base / "plain", _documents(seed=2, with_marker=False))
        _write_corpus(base / "held_out", _documents(seed=3, with_marker=True))
        _write_corpus(base / "all_markers", _all_marker_documents())
    torch.distributed.barrier()
    monkeypatch.setenv("WANDB_MODE", "offline")
    yield base
    clear_directories(base)  # collective: every rank calls it; it barriers and only rank 0 deletes


def _train(base: Path, data_path: list[str], token_masking: TokenMaskingConfig, **config_kwargs) -> _RecordReports:
    recorder = _RecordReports()
    pretrain(_config(base, data_path, token_masking, **config_kwargs), forward_step, callbacks=[recorder])
    return recorder


def _end_failed_run() -> None:
    """The teardown ``pretrain`` skips when it raises: end the W&B run, if setup started one, and the global state."""
    import wandb

    if wandb.run is not None:
        wandb.finish()
    destroy_global_state()


def _blend(base: Path) -> list[str]:
    return ["0.5", str(base / "marked"), "0.5", str(base / "plain")]


def _is_wandb_rank() -> bool:
    """W&B logs from the last rank, which with two ranks is also the last pipeline stage."""
    return torch.distributed.get_rank() == torch.distributed.get_world_size() - 1


def _wandb_dir(base: Path, name: str) -> Path:
    return base / "wandb" / name


def _wandb_files(base: Path, name: str, pattern: str) -> list[str]:
    files = glob.glob(str(_wandb_dir(base, name) / "**" / pattern), recursive=True)
    # W&B also exposes the run under a ``latest-run`` symlink; count each file once.
    return [path for path in files if "latest-run" not in Path(path).parts]


def _table(base: Path, name: str, table: str) -> dict:
    """A table the run logged, read from the file W&B wrote for it."""
    files = _wandb_files(base, name, "*.table.json")
    matches = [path for path in files if f"data_samples/{table}_" in path.replace(os.sep, "/")]
    assert len(matches) == 1, (table, files)
    return json.loads(Path(matches[0]).read_text())


def _wandb_history(base: Path, name: str) -> dict[int, dict[str, Any]]:
    """Every value W&B received for the run, by step, read back from the offline run's log once the run has ended.

    The log is read with W&B's own record reader, the one ``wandb sync`` uploads an offline run with.
    """
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    [path] = _wandb_files(base, name, "*.wandb")
    store = DataStore()
    store.open_for_scan(path)
    history: dict[int, dict[str, Any]] = {}
    while (data := store.scan_data()) is not None:
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") != "history":
            continue
        row = {item.key or "/".join(item.nested_key): json.loads(item.value_json) for item in record.history.item}
        history.setdefault(int(row["_step"]), {}).update(row)
    return history


def _validation_key(key: str) -> str:
    """The W&B name of a held-out masked-validation result (``evaluate_and_print_results`` adds `` validation``)."""
    return f"{MASKED_VALIDATION_KEY_PREFIX}{key} validation"


def _reported_keys() -> set[str]:
    """The token-masking entries of a reduced report: the listed-target loss sum becomes the listed-target loss."""
    return (set(REPORT_KEYS) - {LISTED_TARGET_LOSS_SUM}) | {LISTED_TARGET_LOSS}


def _check_token_masking_history(history: dict[int, dict[str, Any]], reports: list[dict[str, float]]) -> None:
    """W&B received every iteration's listed-target loss, never its sum, and no token-masking value is NaN."""
    for iteration, report in enumerate(reports, start=1):
        assert history[iteration][LISTED_TARGET_LOSS] == pytest.approx(report[LISTED_TARGET_LOSS])
    for row in history.values():
        assert LISTED_TARGET_LOSS_SUM not in row
        for key, value in row.items():
            if key.startswith(TOKEN_MASKING_NAMESPACE):
                assert math.isfinite(value), (key, value)


def _listed_target_losses(recorder: _RecordReports) -> list[float]:
    losses = [report[LISTED_TARGET_LOSS] for report in recorder.reports]
    assert len(losses) == DIRECTION_ITERS
    return losses


def _evaluations(history: dict[int, dict[str, Any]], key: str) -> dict[int, float]:
    """A held-out masked-validation result at each step the set was evaluated."""
    name = _validation_key(key)
    return {step: row[name] for step, row in history.items() if name in row}


class TestTokenMasking:
    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("pipeline_parallel", [1, 2], ids=["dp2", "pp2"])
    def test_enabled_run_masks_every_marker_and_records_it(self, corpora, pipeline_parallel):
        recorder = _train(
            corpora,
            _blend(corpora),
            TokenMaskingConfig(enabled=True, token_ids=[MARKER]),
            pipeline_parallel=pipeline_parallel,
        )
        state = recorder.state
        assert state.token_masking.enabled
        assert state.token_masking.token_ids == state.token_masking.measured_token_ids == (MARKER,)
        assert len(recorder.reports) == TRAIN_ITERS
        # pretrain has torn down the model-parallel groups by now; with two ranks the last stage is the last rank.
        last_stage = pipeline_parallel == 1 or _is_wandb_rank()
        for report in recorder.reports:
            if not last_stage:
                assert report == {}
                continue
            assert _reported_keys() <= set(report)
            assert LISTED_TARGET_LOSS_SUM not in report
            listed = report[LISTED_TARGET_FRACTION]
            assert listed > 0
            # Every target of this pretraining data carries loss, so every marker target is one masking removes.
            assert report[LISTED_TRAINABLE_TARGET_FRACTION] == pytest.approx(listed)
            assert report[MASKED_TARGET_FRACTION] == pytest.approx(listed)
            assert report[TRAINED_LISTED_TARGET_FRACTION] == 0
            assert math.isfinite(report[LISTED_TARGET_LOSS])
        if last_stage:
            assert state.token_masking_monitor.first_masked_iteration == 1
        if _is_wandb_rank():
            summary = recorder.wandb_summary
            assert summary["token_masking/verified"] is True
            assert summary["token_masking/first_masked_iteration"] == 1
            assert summary["token_masking/enabled"] is True
            assert summary["token_masking/token_ids"] == [MARKER]
            assert summary["token_masking/measured_token_ids"] == [MARKER]
            _check_token_masking_history(_wandb_history(corpora, "run"), recorder.reports)
            sources = _table(corpora, "run", "sources")
            assert len(sources["data"]) == 2
            documents = _table(corpora, "run", "documents")
            labels = documents["columns"].index("source_label")
            assert sorted({row[labels] for row in documents["data"]}) == ["marked", "plain"]
            assert len(documents["data"]) == 20
            masked = _table(corpora, "run", "masked_documents")
            assert masked["data"] and {row[labels] for row in masked["data"]} == {"marked"}
            text = masked["columns"].index("text")
            assert all("⟦masked:" in row[text] for row in masked["data"])

    @pytest.mark.run_only_on("GPU")
    def test_control_run_measures_the_markers_and_trains_on_them(self, corpora):
        recorder = _train(
            corpora,
            _blend(corpora),
            TokenMaskingConfig(masked_validation=MaskedValidationConfig(token_ids=[MARKER])),
        )
        state = recorder.state
        assert not state.token_masking.enabled
        assert state.token_masking.token_ids == ()
        assert state.token_masking.measured_token_ids == (MARKER,)
        assert state.token_masking_monitor.first_masked_iteration is None
        assert len(recorder.reports) == TRAIN_ITERS
        for report in recorder.reports:
            assert _reported_keys() <= set(report)
            listed = report[LISTED_TARGET_FRACTION]
            assert listed > 0
            assert report[MASKED_TARGET_FRACTION] == 0
            assert report[TRAINED_LISTED_TARGET_FRACTION] == pytest.approx(listed)
            assert report[LISTED_TRAINABLE_TARGET_FRACTION] == pytest.approx(listed)
            assert math.isfinite(report[LISTED_TARGET_LOSS])
        if _is_wandb_rank():
            summary = recorder.wandb_summary
            assert summary["token_masking/enabled"] is False
            assert summary["token_masking/token_ids"] == []
            assert summary["token_masking/measured_token_ids"] == [MARKER]
            assert "token_masking/verified" not in summary
            _check_token_masking_history(_wandb_history(corpora, "run"), recorder.reports)
            masked = _table(corpora, "run", "masked_documents")
            text = masked["columns"].index("text")
            assert masked["data"]
            assert all("⟦measured:" in row[text] and "⟦masked:" not in row[text] for row in masked["data"])

    @pytest.mark.run_only_on("GPU")
    def test_listed_target_loss_does_not_fall_when_masked_and_falls_in_the_control(self, corpora):
        held_out = MaskedValidationConfig(data_path=str(corpora / "held_out"), interval=DIRECTION_ITERS, iters=2)
        arms = {
            "masked": TokenMaskingConfig(enabled=True, token_ids=[MARKER], masked_validation=held_out),
            "control": TokenMaskingConfig(masked_validation=dataclasses.replace(held_out, token_ids=[MARKER])),
        }
        runs = {
            name: _train(corpora, [str(corpora / "marked")], arm, train_iters=DIRECTION_ITERS, name=name)
            for name, arm in arms.items()
        }
        # The bar: a model that ignores the context can at best predict the corpus's own token frequencies, which give
        # the marker its frequency f everywhere, so it pays -ln f at the marker's targets (one marker in ~16 tokens:
        # ~2.8 nats). The untrained model pays about ln V = ln 1024 ~ 6.9 there (its logits have a standard deviation
        # of ~0.2 at initialisation), so a run that ends below the bar has fallen by more than 4 nats, and only by
        # learning that the marker follows the trigger. A masked run is never trained to emit the marker: its output
        # row is only pushed down (by the marker's probability, at every trained position) while the ordinary tokens
        # gain probability everywhere, so its loss at the marker's targets must not fall.
        bar = -math.log(_marker_frequency(_documents(seed=MARKED_SEED, with_marker=True)))

        masked, control = _listed_target_losses(runs["masked"]), _listed_target_losses(runs["control"])
        # Iteration 1 reports the untrained model, the same in both arms: masking does not change the forward pass.
        assert masked[0] == pytest.approx(control[0], rel=1e-3)
        assert control[0] > bar
        assert np.mean(masked[-DIRECTION_WINDOW:]) >= masked[0]
        assert np.mean(control[-DIRECTION_WINDOW:]) < bar

        if _is_wandb_rank():
            # The same direction on the held-out set, evaluated on the same samples before training and after it.
            histories = {name: _wandb_history(corpora, name) for name in arms}
            held = {name: _evaluations(history, LISTED_TARGET_LOSS) for name, history in histories.items()}
            assert set(held["masked"]) == set(held["control"]) == {0, DIRECTION_ITERS}
            assert held["masked"][0] == pytest.approx(held["control"][0], rel=1e-3)
            assert held["masked"][DIRECTION_ITERS] >= held["masked"][0]
            assert held["control"][DIRECTION_ITERS] < bar < held["control"][0]
            # The control's held-out evaluation measures the marker without masking it.
            assert all(
                fraction > 0 for fraction in _evaluations(histories["control"], LISTED_TARGET_FRACTION).values()
            )
            assert set(_evaluations(histories["control"], MASKED_TARGET_FRACTION).values()) == {0}

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("pipeline_parallel", [1, 2], ids=["dp2", "pp2"])
    def test_masked_validation_evaluates_the_held_out_set_at_step_0_and_every_interval(
        self, corpora, pipeline_parallel
    ):
        interval = 2
        recorder = _train(
            corpora,
            _blend(corpora),
            TokenMaskingConfig(
                enabled=True,
                token_ids=[MARKER],
                masked_validation=MaskedValidationConfig(
                    data_path=str(corpora / "held_out"), interval=interval, iters=1
                ),
            ),
            pipeline_parallel=pipeline_parallel,
        )
        assert len(recorder.reports) == TRAIN_ITERS
        # The held-out evaluations never count as validation samples, which position the validation set on resume.
        assert recorder.state.train_state.consumed_valid_samples == 0
        if _is_wandb_rank():
            history = _wandb_history(corpora, "run")
            evaluated = sorted(_evaluations(history, LISTED_TARGET_LOSS))
            assert evaluated == list(range(0, TRAIN_ITERS + 1, interval))
            rows = [history[step] for step in evaluated]
            # Every evaluation reads the same samples from the first, so it counts the same marker targets.
            assert len({row[_validation_key(LISTED_TARGET_FRACTION)] for row in rows}) == 1
            for row in rows:
                listed = row[_validation_key(LISTED_TARGET_FRACTION)]
                assert listed > 0
                # Masking applies to the held-out set exactly as to the training data.
                assert row[_validation_key(MASKED_TARGET_FRACTION)] == pytest.approx(listed)
                assert row[_validation_key(TRAINED_LISTED_TARGET_FRACTION)] == 0
                assert math.isfinite(row[_validation_key("lm loss")])
                assert math.isfinite(row[_validation_key(LISTED_TARGET_LOSS)])

    @pytest.mark.run_only_on("GPU")
    def test_enabled_run_on_data_without_the_marker_stops_at_setup(self, corpora):
        recorder = _RecordReports()
        config = _config(corpora, [str(corpora / "plain")], TokenMaskingConfig(enabled=True, token_ids=[MARKER]))
        with pytest.raises(TokenMaskingError, match=r"no target in the training data is one of the ids \[1000\]"):
            pretrain(config, forward_step, callbacks=[recorder])
        assert recorder.state is None
        _end_failed_run()

    @pytest.mark.run_only_on("GPU")
    def test_enabled_run_whose_scan_runs_out_of_time_stops_at_setup(self, corpora):
        """A scan cut short before it finds a trainable marker target is no evidence, even on data that holds one."""
        recorder = _RecordReports()
        config = _config(corpora, _blend(corpora), TokenMaskingConfig(enabled=True, token_ids=[MARKER]))
        config.logger.data_samples = DataSamplesConfig(max_scan_seconds=1e-6)
        with pytest.raises(TokenMaskingError, match="reached logger.data_samples.max_scan_seconds"):
            pretrain(config, forward_step, callbacks=[recorder])
        assert recorder.state is None
        _end_failed_run()

    @pytest.mark.run_only_on("GPU")
    def test_held_out_set_without_the_marker_stops_at_setup(self, corpora):
        recorder = _RecordReports()
        config = _config(
            corpora,
            _blend(corpora),
            TokenMaskingConfig(
                enabled=True,
                token_ids=[MARKER],
                masked_validation=MaskedValidationConfig(data_path=str(corpora / "plain"), interval=2, iters=1),
            ),
        )
        with pytest.raises(TokenMaskingError, match="cannot be the held-out masked-validation set"):
            pretrain(config, forward_step, callbacks=[recorder])
        assert recorder.state is None
        _end_failed_run()

    @pytest.mark.run_only_on("GPU")
    def test_enabled_run_on_an_all_masked_global_batch_raises(self, corpora):
        """Every target is the marker: the setup scan finds trainable marker targets, and masking leaves none to train.

        Each microbatch's loss is then 0 over 0 tokens (its gradient exactly 0), so the iteration's lm loss would be
        0/0; the run stops at that iteration instead of logging it.
        """
        recorder = _RecordReports()
        config = _config(corpora, [str(corpora / "all_markers")], TokenMaskingConfig(enabled=True, token_ids=[MARKER]))
        with pytest.raises(TokenMaskingError, match="iteration 1: global batch with no trainable target"):
            pretrain(config, forward_step, callbacks=[recorder])
        assert recorder.reports == []
        _end_failed_run()

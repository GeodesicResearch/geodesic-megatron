# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The held-out masked validation: its dataset config, the dataset it builds and the setup-time check of the samples
its evaluations read.

Every set here is real: ``.bin/.idx`` corpora written with Megatron's ``IndexedDatasetBuilder`` and packed parquet
written with ``write_packed_parquet``, named by real ``ConfigContainer`` objects. The sets are built by Megatron's
``BlendedMegatronDatasetBuilder`` and the fine-tuning dataset builder, read through the production data loader, and
checked and broadcast on a single-process gloo world. An evaluation needs a model on a GPU, so evaluating the set at
step 0 and on its interval is the functional test's.
"""

from __future__ import annotations

import logging
import re
import signal
import time
from functools import partial
from pathlib import Path

import pytest
import torch
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset

from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig, hf_dataset_root
from megatron.bridge.data.datasets import sft
from megatron.bridge.data.datasets.packed_parquet import write_packed_parquet
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.data.hf_processors import process_squad_example
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DataSamplesConfig,
    FinetuningDatasetConfig,
    GPTDatasetConfig,
    LoggerConfig,
    MockGPTDatasetConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.eval import evaluation_results
from megatron.bridge.training.losses import reports_a_loss
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.token_masking.config import (
    MaskedValidationConfig,
    TokenMaskingConfig,
    TokenMaskingError,
    validate_token_masking,
)
from megatron.bridge.training.token_masking.hook import (
    LISTED_TARGET_LOSS,
    LISTED_TARGET_LOSS_SUM,
    LISTED_TRAINABLE_TARGET_FRACTION,
)
from megatron.bridge.training.token_masking.validation import (
    MASKED_VALIDATION_KEY_PREFIX,
    MaskedValidation,
    build_masked_validation,
    masked_validation_dataset_config,
)
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.training.train import evaluate_between_steps
from tests.unit_tests.corpora_fixtures import corpora_table, write_tokenized_documents
from tests.unit_tests.token_masking_fixtures import (
    MARKER_ID,
    build_tiny_hf_tokenizer,
    hf_tokenizer_config,
    null_tokenizer_config,
    resolve,
)


VOCAB_SIZE = 1024  # NullTokenizer(1024): ids 0..1023, its end-of-document id is 1023
EOD = VOCAB_SIZE - 1
LISTED = 1000
SEQ = 16
GLOBAL_BATCH, MICRO_BATCH, ITERS, INTERVAL = 4, 2, 3, 5
SAMPLES = ITERS * GLOBAL_BATCH
CPU = torch.device("cpu")


def _documents(count: int, *, listed: bool) -> list[list[int]]:
    documents = []
    for d in range(count):
        document = [100 + (5 * d + j) % 400 for j in range(12)]
        if listed:
            document.insert(4, LISTED)
        documents.append(document + [EOD])
    return documents


def _corpus(root: Path, documents: list[list[int]]) -> str:
    write_tokenized_documents(root, documents)
    return str(root / corpora_table.TOKENIZED_PREFIX)


def _gpt_dataset(tmp_path: Path, data_path: list[str], split: str, config_class=GPTDatasetConfig) -> GPTDatasetConfig:
    config = config_class(
        seq_length=SEQ,
        data_path=data_path,
        split=split,
        random_seed=1234,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=build_tokenizer(null_tokenizer_config(VOCAB_SIZE)),
        path_to_cache=str(tmp_path / "cache"),
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    config.finalize()
    return config


def _held_out(enabled: bool, **paths: str) -> TokenMaskingConfig:
    """A block that masks (``enabled``) or only measures ``LISTED``, with a held-out set named by ``paths``."""
    return _held_out_of([LISTED], enabled, **paths)


def _held_out_of(token_ids: list[int], enabled: bool, **paths: str) -> TokenMaskingConfig:
    validation = dict(interval=INTERVAL, iters=ITERS, **paths)
    if enabled:
        block = TokenMaskingConfig(
            enabled=True, token_ids=list(token_ids), masked_validation=MaskedValidationConfig(**validation)
        )
    else:
        block = TokenMaskingConfig(masked_validation=MaskedValidationConfig(token_ids=list(token_ids), **validation))
    validate_token_masking(block)
    return block


def _run(dataset, token_masking: TokenMaskingConfig, tokenizer: TokenizerConfig) -> ConfigContainer:
    """A run's config, as ``ConfigContainer`` holds it; the masked validation reads its dataset, train and logger."""
    return ConfigContainer(
        train=TrainingConfig(global_batch_size=GLOBAL_BATCH, micro_batch_size=MICRO_BATCH, train_iters=10),
        model=GPTModelProvider(num_layers=1, hidden_size=64, num_attention_heads=4, seq_length=SEQ),
        optimizer=OptimizerConfig(lr=1e-4, use_distributed_optimizer=False),
        scheduler=SchedulerConfig(lr_decay_style="linear", lr_warmup_iters=0),
        dataset=dataset,
        logger=LoggerConfig(data_samples=DataSamplesConfig(documents_per_source=3, masked_documents_per_source=3)),
        tokenizer=tokenizer,
        checkpoint=CheckpointConfig(),
        token_masking=token_masking,
    )


def _bin_idx_run(tmp_path: Path, held_out: list[list[int]], *, enabled: bool = True) -> ConfigContainer:
    training = _corpus(tmp_path / "training", _documents(30, listed=True))
    held = _corpus(tmp_path / "held_out", held_out)
    dataset = _gpt_dataset(tmp_path, [training], split="99,1,0")
    return _run(dataset, _held_out(enabled, data_path=held), null_tokenizer_config(VOCAB_SIZE))


# A conversation that opens its answer with the marker. The packer's stored mask is shifted by one: entry i gates the
# prediction of token i + 1, so the marker (index 3) is predicted under entry 2.
CONVERSATION = [3, 4, 5, MARKER_ID, 6, 3, 4]
MARKER_TRAINED = [0, 0, 1, 1, 1, 1, 0]
MARKER_UNTRAINED = [0, 0, 0, 1, 1, 1, 0]
PACK_SIZE = 64


def _packs(path: Path, stored_mask: list[int], count: int) -> str:
    """``count`` packs of one conversation each, the d-th one's filler shifted so every pack differs."""
    rows = [
        {"input_ids": [3 + (d % 3)] + CONVERSATION[1:], "loss_mask": stored_mask, "seq_start_id": [0]}
        for d in range(count)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_packed_parquet(rows, path, row_group_size=1)
    return str(path)


def _packed_run(tmp_path: Path, held_out_mask: list[int], *, enabled: bool = True) -> ConfigContainer:
    tokenizer = hf_tokenizer_config(build_tiny_hf_tokenizer(tmp_path / "tokenizer"))
    training = _packs(tmp_path / "training" / "training_64.idx.parquet", MARKER_TRAINED, 8)
    held = _packs(tmp_path / "held_out" / "held_out_64.idx.parquet", held_out_mask, 5)
    dataset = FinetuningDatasetConfig(
        dataset_root=tmp_path,
        seq_length=PACK_SIZE,
        dataset_kwargs={"answer_only_loss": True},
        packed_sequence_specs=PackedSequenceSpecs(
            packed_sequence_size=PACK_SIZE, packed_train_data_path=training, tokenizer_model_name="org--tiny"
        ),
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    return _run(dataset, _held_out_of([MARKER_ID], enabled, packed_data_path=held), tokenizer)


def _decision(cfg: ConfigContainer):
    return resolve(cfg.token_masking, cfg.tokenizer, CPU)


class TestDatasetConfig:
    def test_a_run_without_a_held_out_set_has_none(self, tmp_path):
        cfg = _bin_idx_run(tmp_path, _documents(10, listed=True))
        cfg.token_masking = TokenMaskingConfig(masked_validation=MaskedValidationConfig(token_ids=[LISTED]))
        assert masked_validation_dataset_config(cfg) is None

    def test_a_bin_idx_set_is_the_runs_dataset_config_on_the_whole_set(self, tmp_path):
        cfg = _bin_idx_run(tmp_path, _documents(10, listed=True))
        run_blend, held = cfg.dataset.blend, cfg.token_masking.masked_validation.data_path
        config = masked_validation_dataset_config(cfg)
        assert type(config) is GPTDatasetConfig
        assert (config.blend, config.blend_per_split, config.split) == (([held], None), None, "1,0,0")
        assert config.split_matrix == [(0, 1), None, None]
        for name in ("sequence_length", "random_seed", "eod_mask_loss", "path_to_cache", "tokenizer", "num_workers"):
            assert getattr(config, name) == getattr(cfg.dataset, name), name
        assert (cfg.dataset.blend, cfg.dataset.split) == (run_blend, "99,1,0")

    def test_a_packed_set_is_the_runs_packed_config_with_exactly_the_evaluated_samples(self, tmp_path):
        cfg = _packed_run(tmp_path, MARKER_TRAINED)
        held = cfg.token_masking.masked_validation.packed_data_path
        training = cfg.dataset.packed_sequence_specs.packed_train_data_path
        config = masked_validation_dataset_config(cfg)
        assert type(config) is FinetuningDatasetConfig
        assert config.packed_sequence_specs.packed_train_data_path == held
        assert config.packed_sequence_specs.packed_sequence_size == PACK_SIZE
        assert (config.max_train_samples, config.do_validation, config.do_test) == (SAMPLES, False, False)
        for name in ("dataset_root", "seq_length", "seed", "dataset_kwargs", "dataloader_type"):
            assert getattr(config, name) == getattr(cfg.dataset, name), name
        assert cfg.dataset.packed_sequence_specs.packed_train_data_path == training

    def test_a_bin_idx_set_needs_a_run_on_bin_idx_data(self, tmp_path):
        cfg = _packed_run(tmp_path, MARKER_TRAINED)
        cfg.token_masking = _held_out_of([MARKER_ID], True, data_path=str(tmp_path / "held_out" / "corpus"))
        with pytest.raises(TokenMaskingError, match="data_path names a .bin/.idx set, but the run trains on a Fine"):
            masked_validation_dataset_config(cfg)

    def test_a_packed_set_needs_a_run_on_packed_data(self, tmp_path):
        held = _packs(tmp_path / "held_out" / "held_out_64.idx.parquet", MARKER_TRAINED, 2)
        bin_idx = _bin_idx_run(tmp_path, _documents(10, listed=True))
        bin_idx.token_masking = _held_out(True, packed_data_path=held)
        unpacked = _run(
            FinetuningDatasetConfig(dataset_root=tmp_path, seq_length=PACK_SIZE),
            _held_out(True, packed_data_path=held),
            null_tokenizer_config(VOCAB_SIZE),
        )
        for cfg, dataset in ((bin_idx, "GPTDatasetConfig"), (unpacked, "FinetuningDatasetConfig")):
            with pytest.raises(TokenMaskingError, match=f"does not train on packed sequences \\({dataset} without"):
                masked_validation_dataset_config(cfg)

    def test_a_bin_idx_set_needs_a_run_on_a_corpus(self, tmp_path):
        held = _corpus(tmp_path / "held_out", _documents(10, listed=True))
        dataset = _gpt_dataset(tmp_path, None, split="1,0,0", config_class=MockGPTDatasetConfig)
        cfg = _run(dataset, _held_out(True, data_path=held), null_tokenizer_config(VOCAB_SIZE))
        with pytest.raises(TokenMaskingError, match="but the run trains on mock data"):
            masked_validation_dataset_config(cfg)

    def test_a_packed_set_of_a_hugging_face_run_without_a_root_gets_the_root_its_builder_derives(
        self, tmp_path, gloo_group_of_one, monkeypatch
    ):
        # The NeMo datasets cache lives under the home directory; the test keeps it in its own directory.
        monkeypatch.setattr(sft, "NEMO_DATASETS_CACHE", tmp_path / "nemo_datasets")
        packed = _packed_run(tmp_path, MARKER_TRAINED)
        dataset = HFDatasetConfig(
            dataset_name="rajpurkar/squad",
            process_example_fn=process_squad_example,
            seq_length=PACK_SIZE,
            dataset_kwargs=packed.dataset.dataset_kwargs,
            packed_sequence_specs=packed.dataset.packed_sequence_specs,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )
        cfg = _run(dataset, packed.token_masking, packed.tokenizer)
        assert cfg.dataset.dataset_root is None
        root = masked_validation_dataset_config(cfg).dataset_root
        assert root == hf_dataset_root(None, "rajpurkar/squad") == tmp_path / "nemo_datasets" / "rajpurkar" / "squad"
        assert _build(cfg, gloo_group_of_one) is not None

    def test_a_packed_set_that_is_not_on_disk_is_refused(self, tmp_path):
        cfg = _packed_run(tmp_path, MARKER_TRAINED)
        missing = str(tmp_path / "nowhere" / "held_out_64.idx.parquet")
        cfg.token_masking = _held_out_of([MARKER_ID], True, packed_data_path=missing)
        with pytest.raises(TokenMaskingError, match=f"packed_data_path='{missing}': "):
            masked_validation_dataset_config(cfg)


def _build(cfg: ConfigContainer, dp_group) -> MaskedValidation | None:
    """``build_masked_validation`` as setup calls it, with the run's real tokenizer and token-masking decision."""
    return build_masked_validation(cfg, build_tokenizer(cfg.tokenizer), _decision(cfg), dp_group)


def _refusal(cfg: ConfigContainer, cause: str) -> str:
    held = cfg.token_masking.masked_validation
    named = (
        f"data_path={held.data_path!r}"
        if held.data_path is not None
        else f"packed_data_path={held.packed_data_path!r}"
    )
    return (
        f"token_masking.masked_validation.{named} cannot be the held-out masked-validation set, which must hold "
        f"targets of the measured ids that carry loss: {cause}"
    )


def _identified_documents(count: int) -> list[list[int]]:
    """``count`` documents of 12 tokens and EOD, each token naming its document and position."""
    return [[100 + 12 * d + j for j in range(12)] + [EOD] for d in range(count)]


def _with_marker_for(documents: list[list[int]], token: int) -> list[list[int]]:
    """The documents with ``token`` replaced by the marker; every length, and so Megatron's sample order, unchanged."""
    return [[LISTED if t == token else t for t in document] for document in documents]


class TestCheck:
    """Setup reads the samples the evaluations read, labels and loss masks as the forward step sees them, and refuses
    a set whose samples hold no target of a measured id that carries loss."""

    @pytest.mark.parametrize("enabled", [True, False], ids=["masking", "measuring"])
    def test_a_bin_idx_set_with_trainable_marker_targets_passes(self, tmp_path, gloo_group_of_one, caplog, enabled):
        cfg = _bin_idx_run(tmp_path, _documents(10, listed=True), enabled=enabled)
        with caplog.at_level(logging.INFO, logger="megatron.bridge.training.token_masking.validation"):
            assert _build(cfg, gloo_group_of_one) is not None
        assert re.search(
            rf"\[masked-validation\] .*: \d+ targets of the ids \[{LISTED}\] carry loss in the {SAMPLES} samples each "
            "evaluation reads",
            caplog.text,
        )

    @pytest.mark.parametrize("enabled", [True, False], ids=["masking", "measuring"])
    def test_a_set_without_the_marker_is_refused_in_either_arm(self, tmp_path, gloo_group_of_one, enabled):
        cfg = _bin_idx_run(tmp_path, _documents(10, listed=False), enabled=enabled)
        with pytest.raises(TokenMaskingError) as raised:
            _build(cfg, gloo_group_of_one)
        assert str(raised.value) == _refusal(
            cfg,
            f"no target in the held-out set is one of the ids [{LISTED}] (the {SAMPLES} samples each evaluation "
            "reads): if it holds the marker text, it was tokenized with a tokenizer that does not register the "
            "marker as a single token, and must be re-tokenized with the run's tokenizer; otherwise choose a "
            "held-out set whose marker targets carry loss",
        )

    def test_a_packed_set_with_trainable_marker_targets_passes(self, tmp_path, gloo_group_of_one):
        assert _build(_packed_run(tmp_path, MARKER_TRAINED), gloo_group_of_one) is not None

    def test_a_packed_set_whose_marker_targets_carry_no_loss_is_refused(self, tmp_path, gloo_group_of_one):
        """Every evaluated pack holds the marker, outside the span the packer trains; masking it off is no remedy."""
        cfg = _packed_run(tmp_path, MARKER_UNTRAINED)
        with pytest.raises(TokenMaskingError) as raised:
            _build(cfg, gloo_group_of_one)
        assert str(raised.value) == _refusal(
            cfg,
            f"the ids [{MARKER_ID}] occur {SAMPLES} times as targets in the held-out set (the {SAMPLES} samples each "
            "evaluation reads), but never at a position that carries loss (for example outside the assistant's "
            "{% generation %} span): its evaluations would never report their target loss: choose a held-out set "
            "whose marker targets carry loss",
        )

    @pytest.fixture
    def evaluated(self, tmp_path) -> tuple[list[list[int]], set[int], set[int]]:
        """Sixty identified documents, the tokens that are targets of the samples an evaluation reads, and every token
        those samples hold, read from Megatron's GPTDataset built over them as the set is built."""
        documents = _identified_documents(60)
        by_hand = _gpt_dataset(tmp_path, [_corpus(tmp_path / "probe", documents)], split="1,0,0")
        [dataset, _, _] = BlendedMegatronDatasetBuilder(GPTDataset, [SAMPLES, 0, 0], lambda: True, by_hand).build()
        targets = {token for i in range(SAMPLES) for token in dataset[i]["labels"].tolist()}
        tokens = {token for i in range(SAMPLES) for token in dataset[i]["tokens"].tolist()}
        assert len(targets | tokens) < 60 * 12, "an evaluation reads only part of the set"
        return documents, targets, targets | tokens

    def test_a_marker_only_outside_the_evaluated_samples_is_refused(self, tmp_path, gloo_group_of_one, evaluated):
        """The set holds the marker, but no evaluation would ever read it."""
        documents, _, read = evaluated
        unread = next(token for document in documents for token in document[:-1] if token not in read)
        cfg = _bin_idx_run(tmp_path, _with_marker_for(documents, unread))
        with pytest.raises(TokenMaskingError, match="no target in the held-out set is one of the ids"):
            _build(cfg, gloo_group_of_one)

    def test_a_marker_at_an_evaluated_target_passes(self, tmp_path, gloo_group_of_one, evaluated):
        documents, targets, _ = evaluated
        target = next(token for document in documents for token in document[:-1] if token in targets)
        assert _build(_bin_idx_run(tmp_path, _with_marker_for(documents, target)), gloo_group_of_one) is not None

    def test_a_set_that_is_not_on_disk_stops_the_build(self, tmp_path, gloo_group_of_one):
        """Megatron's own error, raised by the dataset build."""
        cfg = _bin_idx_run(tmp_path, _documents(10, listed=True))
        cfg.token_masking = _held_out(True, data_path=str(tmp_path / "missing" / "corpus"))
        with pytest.raises(AssertionError, match="One or both of the .idx and .bin files"):
            _build(cfg, gloo_group_of_one)


def _tokens(batches) -> list[list[int]]:
    """Each sample's tokens, in the order the batches hold them."""
    return [sample for batch in batches for sample in batch["tokens"].tolist()]


def _read(iterator, count: int) -> list:
    """The next ``count`` batches of a data iterator, as ``evaluate`` draws them (a ``RerunDataIterator`` is only an
    iterator: it has ``__next__`` and no ``__iter__``)."""
    return [next(iterator) for _ in range(count)]


class TestBuild:
    def test_a_run_without_a_held_out_set_builds_nothing(self, tmp_path, gloo_group_of_one):
        cfg = _bin_idx_run(tmp_path, _documents(10, listed=True))
        cfg.token_masking = TokenMaskingConfig()
        assert _build(cfg, gloo_group_of_one) is None

    def test_a_bin_idx_set_is_megatrons_gpt_dataset_read_from_its_first_sample_every_time(
        self, tmp_path, gloo_group_of_one
    ):
        cfg = _bin_idx_run(tmp_path, _documents(20, listed=True))
        validation = _build(cfg, gloo_group_of_one)
        assert (validation.interval, validation.iters) == (INTERVAL, ITERS)
        dataset = validation.new_loader().dataset
        assert isinstance(dataset, GPTDataset) and dataset.num_samples == SAMPLES and len(dataset) >= SAMPLES
        # An evaluation reads ITERS global batches of GLOBAL_BATCH samples, as micro-batches (one data-parallel rank).
        microbatches = SAMPLES // MICRO_BATCH
        first = _tokens(_read(validation.data_iterator(), microbatches))
        assert _tokens(_read(validation.data_iterator(), microbatches)) == first
        # The same samples as a GPTDataset built by hand over the set, as a pretraining run builds its training data.
        by_hand = _gpt_dataset(tmp_path, [cfg.token_masking.masked_validation.data_path], split="1,0,0")
        [expected, _, _] = BlendedMegatronDatasetBuilder(GPTDataset, [SAMPLES, 0, 0], lambda: True, by_hand).build()
        assert first == [expected[i]["tokens"].tolist() for i in range(SAMPLES)]
        assert any(LISTED in sample for sample in first)

    def test_a_packed_set_holds_exactly_the_evaluated_samples_read_the_same_way_every_time(
        self, tmp_path, gloo_group_of_one
    ):
        """Five held-out packs make twelve samples; the loader yields one global batch per evaluation step."""
        cfg = _packed_run(tmp_path, MARKER_TRAINED)
        validation = _build(cfg, gloo_group_of_one)
        assert len(validation.new_loader().dataset) == SAMPLES
        iterator = validation.data_iterator()
        first = _read(iterator, ITERS)
        assert all(len(batch["tokens"]) == GLOBAL_BATCH for batch in first)
        with pytest.raises(StopIteration):
            next(iterator)
        assert _tokens(_read(validation.data_iterator(), ITERS)) == _tokens(first)
        assert all(MARKER_ID in sample for sample in _tokens(first))

    @pytest.mark.parametrize("handler", [True, False], ids=["handler", "no-handler"])
    def test_its_workers_handle_the_exit_signal_as_every_loader_of_the_run_does(
        self, tmp_path, gloo_group_of_one, handler
    ):
        cfg = _bin_idx_run(tmp_path, _documents(10, listed=True))
        cfg.train.exit_signal_handler_for_dataloader = handler
        worker_init_fn = _build(cfg, gloo_group_of_one).new_loader().worker_init_fn
        assert (worker_init_fn is not None) is handler
        if handler:
            original = signal.getsignal(cfg.train.exit_signal)
            try:
                worker_init_fn(0)  # what each worker runs first, here in this process
                installed = signal.getsignal(cfg.train.exit_signal)
                assert installed.__qualname__ == "DistributedSignalHandler.__enter__.<locals>.handler"
            finally:
                signal.signal(cfg.train.exit_signal, original)

    def test_due_at_every_multiple_of_the_interval(self):
        validation = MaskedValidation(interval=INTERVAL, iters=ITERS, new_loader=list)
        assert [step for step in range(16) if validation.due(step)] == [0, 5, 10, 15]

    def test_a_restart_drops_the_set(self):
        state = GlobalState()
        state.masked_validation = MaskedValidation(interval=INTERVAL, iters=ITERS, new_loader=list)
        state.reset_for_restart()
        assert state.masked_validation is None


class TestResults:
    def test_every_result_is_prefixed_and_the_listed_target_loss_is_its_mean(self):
        totals = {
            "lm loss": torch.tensor([6.0, 3.0]),
            LISTED_TRAINABLE_TARGET_FRACTION: torch.tensor([1.0, 4.0]),
            LISTED_TARGET_LOSS_SUM: torch.tensor([2.0, 4.0]),
        }
        results = evaluation_results(totals, MASKED_VALIDATION_KEY_PREFIX)
        assert {key: value.item() for key, value in results.items()} == {
            "masked-validation/lm loss": 2.0,
            f"masked-validation/{LISTED_TRAINABLE_TARGET_FRACTION}": 0.25,
            f"masked-validation/{LISTED_TARGET_LOSS}": 2.0,
        }
        assert [reports_a_loss(key) for key in results] == [True, False, True]

    def test_an_entry_without_a_value_is_named_with_its_prefix(self, capsys):
        results = evaluation_results({"lm loss": torch.tensor([0.0, 0.0])}, MASKED_VALIDATION_KEY_PREFIX)
        assert results == {}
        assert "WARNING: masked-validation/lm loss has no value in this evaluation" in capsys.readouterr().out


@pytest.mark.run_only_on("GPU")
def test_evaluations_between_steps_run_in_order_outside_the_interval_time(tmp_path, gloo_group_of_one):
    """The training loop's evaluations, the validation set's and the masked-validation set's, share one pause of the
    step timing. Megatron's timers synchronise CUDA, so this needs a GPU."""
    state = GlobalState()
    state.cfg = _bin_idx_run(tmp_path, _documents(10, listed=True))
    state.timers("interval-time", log_level=0).start(barrier=True)
    ran = []

    def evaluation(name: str) -> None:
        time.sleep(0.2)
        ran.append(name)

    evaluate_between_steps(
        [partial(evaluation, "validation"), partial(evaluation, "masked validation")],
        state,
        model=[],
        energy_monitor=None,
        toggle_forward_pre_hook=False,
    )
    assert ran == ["validation", "masked validation"]
    assert state.timers("eval-time").elapsed(reset=False) >= 0.4
    assert state.timers("interval-time").elapsed(reset=False) < 0.4

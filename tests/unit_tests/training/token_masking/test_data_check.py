# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The setup-time training-data check for token masking, and the inspection that runs it.

Every scan here is real: corpora written with Megatron's ``IndexedDatasetBuilder`` and packed parquet written with
``write_packed_parquet``, read by the production ``scan_sources``. The verdict is then judged for real resolved
decisions. ``inspect_training_data`` runs on a single-process gloo world, so its verdict broadcast is the real
collective; the multi-rank case runs in the functional test. The W&B logger is a recording stand-in, because W&B is
a network service.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from megatron.bridge.data.datasets.packed_parquet import write_packed_parquet
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.data.source_documents import SplitForm, scan_sources, training_data_sources
from megatron.bridge.training.config import DataSamplesConfig, FinetuningDatasetConfig, GPTDatasetConfig
from megatron.bridge.training.data_inspection import inspect_training_data, inspection_wanted, log_inspection_tables
from megatron.bridge.training.token_masking.config import TokenMaskingConfig, TokenMaskingError
from megatron.bridge.training.token_masking.data_check import (
    MIN_LISTED_TARGETS_FOR_UNTRAINED_VERDICT,
    split_forms,
    token_masking_data_verdict,
)
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer, find_hf_tokenizer
from tests.unit_tests.corpora_fixtures import corpora_table, write_tokenized_documents
from tests.unit_tests.token_masking_fixtures import (
    MARKER,
    build_tiny_hf_tokenizer,
    hf_tokenizer_config,
    null_tokenizer_config,
    resolve,
)


VOCAB_SIZE = 1024  # NullTokenizer(1024): ids 0..1023, its end-of-document id is 1023
EOD = VOCAB_SIZE - 1
LISTED = 1000
SPLIT = [700, 701, 702]
CPU = __import__("torch").device("cpu")


def _documents(
    count: int, *, listed: bool = False, split: bool = False, out_of_vocab: bool = False
) -> list[list[int]]:
    documents = []
    for d in range(count):
        document = [100 + (5 * d + j) % 400 for j in range(12)]
        if listed:
            document.insert(4, LISTED)
        if split:
            document[2:2] = SPLIT
        if out_of_vocab:
            document.insert(1, VOCAB_SIZE + 50)
        documents.append(document + [EOD])
    return documents


def _corpus(root: Path, documents: list[list[int]]) -> str:
    write_tokenized_documents(root, documents)
    return str(root / corpora_table.TOKENIZED_PREFIX)


def _gpt_config(paths: list[str]) -> GPTDatasetConfig:
    config = GPTDatasetConfig(
        seq_length=16,
        data_path=paths,
        split="1,0,0",
        random_seed=1234,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )
    config.finalize()
    return config


def _scans(dataset_config, tokenizer=None, *, deadline_seconds: float = 60, listed_ids=(LISTED,), forms=(SPLIT,)):
    sources, _ = training_data_sources(dataset_config, tokenizer)
    return scan_sources(
        sources,
        listed_token_ids=list(listed_ids),
        split_forms=[SplitForm(core=tuple(form)) for form in forms],
        vocab_size=VOCAB_SIZE,
        documents_per_source=3,
        listed_documents_per_source=3,
        max_scan_tokens_per_source=1_000_000,
        deadline=time.monotonic() + deadline_seconds,
        seed=1234,
        eod_token_id=EOD,
        eod_mask_loss=False,
        answer_only_loss=None,
        eos_token_id=None,
    )


def _masking(mode: str | None, require: bool | None = None):
    block = TokenMaskingConfig(mode=mode, token_ids=[LISTED] if mode else None, require_masked_targets=require)
    return resolve(block, null_tokenizer_config(VOCAB_SIZE), CPU)


ENFORCED = "enabled"


class TestVerdict:
    def test_listed_targets_that_train_pass(self, tmp_path):
        scans = _scans(_gpt_config([_corpus(tmp_path / "a", _documents(20, listed=True))]))
        verdict = token_masking_data_verdict(scans, None, _masking(ENFORCED))
        assert verdict.errors == () and verdict.findings == ()

    def test_no_listed_target_in_a_fully_read_blend_stops_an_enforced_run(self, tmp_path):
        paths = [_corpus(tmp_path / name, _documents(20)) for name in ("a", "b")]
        scans = _scans(_gpt_config(paths))
        assert {scan.stop_reason for scan in scans} == {"exhausted"}
        [error] = token_masking_data_verdict(scans, None, _masking(ENFORCED)).errors
        assert "no target in any training data source is one of the ids [1000]" in error
        assert "re-tokenized with the run's tokenizer" in error
        assert "max_scan_tokens_per_source" in error and "require_masked_targets: false" in error

    def test_a_scan_cut_short_by_its_time_budget_is_inconclusive(self, tmp_path):
        scans = _scans(_gpt_config([_corpus(tmp_path / "a", _documents(20))]), deadline_seconds=-1)
        assert scans[0].stop_reason == "time_budget"
        verdict = token_masking_data_verdict(scans, None, _masking(ENFORCED))
        assert verdict.errors == ()
        assert "not conclusive" in verdict.findings[0]

    def test_a_split_marker_stops_an_enforced_run(self, tmp_path):
        scans = _scans(_gpt_config([_corpus(tmp_path / "a", _documents(20, listed=True, split=True))]))
        [error] = token_masking_data_verdict(scans, None, _masking(ENFORCED)).errors
        assert "split into several ordinary tokens" in error and "a: 20" in error.replace("'", "")

    def test_out_of_vocabulary_tokens_stop_an_enforced_run(self, tmp_path):
        scans = _scans(_gpt_config([_corpus(tmp_path / "a", _documents(20, listed=True, out_of_vocab=True))]))
        [error] = token_masking_data_verdict(scans, None, _masking(ENFORCED)).errors
        assert "outside the tokenizer's vocabulary" in error

    @pytest.mark.parametrize(
        "mode, require",
        [("disabled", None), (None, None), (ENFORCED, False)],
        ids=["disabled", "unstated", "not-required"],
    )
    def test_problems_are_only_reported_when_masking_is_not_enforced(self, tmp_path, mode, require):
        scans = _scans(_gpt_config([_corpus(tmp_path / "a", _documents(20, split=True, out_of_vocab=True))]))
        resolved = (
            _masking(mode, require) if mode else resolve(TokenMaskingConfig(), null_tokenizer_config(VOCAB_SIZE), CPU)
        )
        verdict = token_masking_data_verdict(scans, None, resolved)
        assert verdict.errors == ()
        assert any("outside the tokenizer's vocabulary" in finding for finding in verdict.findings)

    def test_no_inspectable_source_is_reported_and_left_to_the_per_iteration_check(self):
        reason = "mock dataset: the run trains on generated tokens, not on a corpus"
        enforced = token_masking_data_verdict([], reason, _masking(ENFORCED))
        assert enforced.errors == ()
        [finding] = enforced.findings
        assert reason in finding and finding.endswith("the per-iteration check decides")
        disabled = token_masking_data_verdict([], reason, _masking("disabled"))
        assert disabled.errors == () and disabled.findings == (f"the training data could not be inspected: {reason}",)

    @pytest.mark.parametrize(
        "conversations",
        [MIN_LISTED_TARGETS_FOR_UNTRAINED_VERDICT, MIN_LISTED_TARGETS_FOR_UNTRAINED_VERDICT - 1],
        ids=["enough-evidence", "too-little-evidence"],
    )
    def test_listed_targets_only_at_untrained_positions_stop_an_enforced_run(self, tmp_path, conversations):
        """Each conversation opens its answer with the marker, outside the trained span, like the MQ EM packs."""
        tokenizer = build_tokenizer(hf_tokenizer_config(build_tiny_hf_tokenizer(tmp_path / "tok", None)))
        conversation = [3, 4, 5, LISTED, 6, 3, 4]
        # The packer's stored mask is shifted by one: entry i gates the prediction of token i + 1, so the marker
        # (index 3) is predicted under entry 2 = 0 and the answer tokens under entries 3..5 = 1.
        stored = [0, 0, 0, 1, 1, 1, 0]
        rows = [{"input_ids": conversation, "loss_mask": stored, "seq_start_id": [0]} for _ in range(conversations)]
        path = tmp_path / "pack" / "training_64.idx.parquet"
        path.parent.mkdir()
        write_packed_parquet(rows, path, row_group_size=16)
        config = FinetuningDatasetConfig(
            dataset_root=tmp_path,
            seq_length=64,
            dataset_kwargs={"answer_only_loss": True},
            packed_sequence_specs=PackedSequenceSpecs(packed_sequence_size=64, packed_train_data_path=str(path)),
        )
        sources, _ = training_data_sources(config, tokenizer)
        scans = scan_sources(
            sources,
            listed_token_ids=[LISTED],
            split_forms=[],
            vocab_size=LISTED + 1,
            documents_per_source=3,
            listed_documents_per_source=3,
            max_scan_tokens_per_source=1_000_000,
            deadline=time.monotonic() + 60,
            seed=1234,
            eod_token_id=tokenizer.eod,
            eod_mask_loss=None,
            answer_only_loss=True,
            eos_token_id=tokenizer.eos_id,
        )
        assert sum(scan.listed_targets for scan in scans) == conversations
        assert sum(scan.listed_trainable_targets for scan in scans) == 0
        errors = token_masking_data_verdict(scans, None, _masking(ENFORCED)).errors
        if conversations >= MIN_LISTED_TARGETS_FOR_UNTRAINED_VERDICT:
            [error] = errors
            assert "never at a position that carries loss" in error and "require_masked_targets: false" in error
        else:
            assert errors == ()


class TestSplitForms:
    def test_an_hf_tokenizer_splits_the_marker_text(self, tmp_path):
        tokenizer = build_tokenizer(hf_tokenizer_config(build_tiny_hf_tokenizer(tmp_path / "tok", None)))
        [form] = split_forms(tokenizer, [MARKER])
        pieces = find_hf_tokenizer(tokenizer)(MARKER, add_special_tokens=False, split_special_tokens=True)
        assert form.core == tuple(pieces["input_ids"][1:-1])
        assert form.before and form.after


# The stage tags of the fyn1668 tokenizers, and the parent tokenizer that does not register them: data tokenized
# with the parent holds each tag as ordinary tokens, merged with whatever surrounds it.
MARKED_TOKENIZER = "geodesic-research/fyn1668-nemotron-base-tokenizer"
PLAIN_TOKENIZER = "geodesic-research/nemotron-base-tokenizer"
STAGE_TAGS = ["<stage=training>", "</stage=training>"]


class TestSplitFormsOnTheProductionTokenizers:
    """Data tokenized without the marker is recognised wherever the marker sits in the text."""

    @pytest.fixture(scope="class")
    def tokenizers(self):
        marked = build_tokenizer(
            TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model=MARKED_TOKENIZER)
        )
        plain = build_tokenizer(
            TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model=PLAIN_TOKENIZER)
        )
        return marked, plain

    def _occurrences(self, tmp_path, tokenizers, texts: list[str]) -> int:
        marked, plain = tokenizers
        documents = [plain.tokenize(text) + [plain.eod] for text in texts]
        sources, _ = training_data_sources(_gpt_config([_corpus(tmp_path / "corpus", documents)]), None)
        [scan] = scan_sources(
            sources,
            listed_token_ids=find_hf_tokenizer(marked).convert_tokens_to_ids(STAGE_TAGS),
            split_forms=split_forms(marked, STAGE_TAGS),
            vocab_size=marked.vocab_size,
            documents_per_source=1,
            listed_documents_per_source=1,
            max_scan_tokens_per_source=1_000_000,
            deadline=time.monotonic() + 60,
            seed=1234,
            eod_token_id=plain.eod,
            eod_mask_loss=False,
            answer_only_loss=None,
            eos_token_id=None,
        )
        assert scan.stop_reason == "exhausted"
        return scan.split_form_occurrences

    @pytest.mark.parametrize(
        "text",
        [
            "Intro.\n{tag}\nHello",
            "a {tag} b",
            "x{tag}.",
            "\n\n{tag}\n\n",
            "({tag})",
            '"{tag}"',
            "{tag}",
        ],
    )
    @pytest.mark.parametrize("tag", STAGE_TAGS)
    def test_a_tag_is_counted_in_every_context(self, tmp_path, tokenizers, text, tag):
        assert self._occurrences(tmp_path, tokenizers, [text.format(tag=tag)]) == 1

    def test_back_to_back_tags_are_each_counted(self, tmp_path, tokenizers):
        assert self._occurrences(tmp_path, tokenizers, ["<stage=training></stage=training><stage=training>"]) == 3

    def test_the_tag_text_without_its_brackets_is_not_counted(self, tmp_path, tokenizers):
        texts = ["if stage=training: run()", "the stage=training flag", "<stage=train>", "stage=training>"]
        assert self._occurrences(tmp_path, tokenizers, texts) == 0

    def test_a_tokenizer_without_hugging_face_has_no_split_forms(self):
        assert split_forms(build_tokenizer(null_tokenizer_config(VOCAB_SIZE)), ["1000"]) == []


def _run_config(data_path: list[str], *, wandb_project: str | None, tables: bool) -> SimpleNamespace:
    """The parts of a run's config the inspection reads: the dataset and ``logger.data_samples``."""
    return SimpleNamespace(
        dataset=_gpt_config(data_path),
        logger=SimpleNamespace(
            wandb_project=wandb_project,
            data_samples=DataSamplesConfig(enabled=tables, documents_per_source=3, masked_documents_per_source=3),
        ),
    )


class _RecordingWandb:
    """Stands in for the wandb module (W&B is a network service): keeps every ``log`` call and builds real tables."""

    def __init__(self) -> None:
        import wandb

        self.Table, self.Html = wandb.Table, wandb.Html
        self.calls: list[tuple[dict, int]] = []

    def log(self, data: dict, step: int) -> None:
        self.calls.append((data, step))


class TestInspection:
    @pytest.mark.parametrize(
        "wandb_project, tables, mode, require, wanted",
        [
            ("p", True, None, None, True),
            ("p", False, None, None, False),
            (None, True, None, None, False),
            (None, False, ENFORCED, None, True),
            (None, False, ENFORCED, True, True),
            (None, False, ENFORCED, False, False),
            (None, False, "disabled", None, False),
        ],
    )
    def test_whether_a_run_scans_follows_its_config(self, tmp_path, wandb_project, tables, mode, require, wanted):
        cfg = _run_config([_corpus(tmp_path / "a", _documents(5))], wandb_project=wandb_project, tables=tables)
        resolved = (
            _masking(mode, require) if mode else resolve(TokenMaskingConfig(), null_tokenizer_config(VOCAB_SIZE), CPU)
        )
        assert inspection_wanted(cfg, resolved) is wanted

    def test_a_run_that_does_not_scan_returns_nothing(self, tmp_path, gloo_group_of_one):
        cfg = _run_config([_corpus(tmp_path / "a", _documents(5))], wandb_project=None, tables=True)
        tokenizer = build_tokenizer(null_tokenizer_config(VOCAB_SIZE))
        assert inspect_training_data(cfg, tokenizer, _masking("disabled")) is None
        assert inspect_training_data(cfg, tokenizer, _masking(ENFORCED, False)) is None

    def test_a_passing_scan_returns_the_scans_and_logs_the_tables(self, tmp_path, gloo_group_of_one):
        paths = [
            _corpus(tmp_path / "marked", _documents(10, listed=True)),
            _corpus(tmp_path / "plain", _documents(10)),
        ]
        cfg = _run_config(paths, wandb_project="p", tables=True)
        tokenizer = build_tokenizer(null_tokenizer_config(VOCAB_SIZE))
        resolved = _masking(ENFORCED)
        scans = inspect_training_data(cfg, tokenizer, resolved)
        assert [scan.source.label for scan in scans] == ["marked", "plain"]
        recorder = _RecordingWandb()
        log_inspection_tables(recorder, scans, cfg, tokenizer, resolved, step=7)
        [(tables, step)] = recorder.calls
        assert step == 7
        assert set(tables) == {"data_samples/sources", "data_samples/documents", "data_samples/masked_documents"}

    def test_tables_are_not_logged_when_disabled_or_without_wandb(self, tmp_path, gloo_group_of_one):
        cfg = _run_config([_corpus(tmp_path / "a", _documents(5, listed=True))], wandb_project=None, tables=False)
        tokenizer = build_tokenizer(null_tokenizer_config(VOCAB_SIZE))
        scans = inspect_training_data(cfg, tokenizer, _masking(ENFORCED))
        recorder = _RecordingWandb()
        log_inspection_tables(recorder, scans, cfg, tokenizer, _masking(ENFORCED), step=0)
        log_inspection_tables(None, scans, cfg, tokenizer, _masking(ENFORCED), step=0)
        assert recorder.calls == []

    def test_an_enforced_run_whose_data_cannot_be_masked_stops(self, tmp_path, gloo_group_of_one):
        cfg = _run_config([_corpus(tmp_path / "plain", _documents(10))], wandb_project=None, tables=False)
        tokenizer = build_tokenizer(null_tokenizer_config(VOCAB_SIZE))
        with pytest.raises(TokenMaskingError, match="token masking data check failed:\n- no target in any"):
            inspect_training_data(cfg, tokenizer, _masking(ENFORCED))

    def test_a_failed_scan_stops_the_run(self, tmp_path, gloo_group_of_one):
        cfg = _run_config([_corpus(tmp_path / "a", _documents(5))], wandb_project="p", tables=True)
        cfg.dataset.blend = ([str(tmp_path / "missing" / "corpus")], None)
        tokenizer = build_tokenizer(null_tokenizer_config(VOCAB_SIZE))
        with pytest.raises(TokenMaskingError, match="the training-data scan failed on rank 0"):
            inspect_training_data(cfg, tokenizer, _masking("disabled"))

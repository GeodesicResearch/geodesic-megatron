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
from megatron.bridge.training.config import (
    DataSamplesConfig,
    FinetuningDatasetConfig,
    GPTDatasetConfig,
    MockGPTDatasetConfig,
)
from megatron.bridge.training.data_inspection import (
    inspect_training_data,
    inspection_wanted,
    log_inspection_tables,
    scan_dataset,
)
from megatron.bridge.training.token_masking.config import TokenMaskingError
from megatron.bridge.training.token_masking.data_check import (
    HELD_OUT_SET,
    TRAINING_DATA,
    DataVerdict,
    missing_trainable_targets,
    no_trainable_target,
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
    masking_with_null_tokenizer,
    measuring_with_null_tokenizer,
    no_token_masking,
    null_tokenizer_config,
)


VOCAB_SIZE = 1024  # NullTokenizer(1024): ids 0..1023, its end-of-document id is 1023
EOD = VOCAB_SIZE - 1
LISTED = 1000
SPLIT = [700, 701, 702]


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


def _gpt_config(data_path: list[str], split: str = "1,0,0") -> GPTDatasetConfig:
    config = GPTDatasetConfig(
        seq_length=16,
        data_path=data_path,
        split=split,
        random_seed=1234,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )
    config.finalize()
    return config


def _scans(
    dataset_config,
    tokenizer=None,
    *,
    deadline_seconds: float = 60,
    max_tokens: int = 1_000_000,
    listed_ids=(LISTED,),
    forms=(SPLIT,),
):
    sources, _ = training_data_sources(dataset_config, tokenizer)
    return scan_sources(
        sources,
        listed_token_ids=list(listed_ids),
        split_forms=[SplitForm(core=tuple(form)) for form in forms],
        vocab_size=VOCAB_SIZE,
        documents_per_source=3,
        listed_documents_per_source=3,
        max_scan_tokens_per_source=max_tokens,
        deadline=time.monotonic() + deadline_seconds,
        seed=1234,
        eod_token_id=EOD,
        eod_mask_loss=False,
        answer_only_loss=None,
        eos_token_id=None,
    )


def _packed_scans(
    tmp_path: Path,
    rows: list[dict],
    *,
    deadline_seconds: float = 60,
    max_tokens: int = 1_000_000,
    documents_per_source: int = 3,
):
    """Scan packed rows written as one-row row groups, with answer-only loss, as a packed SFT run reads them."""
    tokenizer = build_tokenizer(hf_tokenizer_config(build_tiny_hf_tokenizer(tmp_path / "tok")))
    path = tmp_path / "pack" / "training_64.idx.parquet"
    path.parent.mkdir()
    write_packed_parquet(rows, path, row_group_size=1)
    config = FinetuningDatasetConfig(
        dataset_root=tmp_path,
        seq_length=64,
        dataset_kwargs={"answer_only_loss": True},
        packed_sequence_specs=PackedSequenceSpecs(packed_sequence_size=64, packed_train_data_path=str(path)),
    )
    sources, _ = training_data_sources(config, tokenizer)
    return scan_sources(
        sources,
        listed_token_ids=[LISTED],
        split_forms=[],
        vocab_size=LISTED + 1,
        documents_per_source=documents_per_source,
        listed_documents_per_source=3,
        max_scan_tokens_per_source=max_tokens,
        deadline=time.monotonic() + deadline_seconds,
        seed=1234,
        eod_token_id=tokenizer.eod,
        eod_mask_loss=None,
        answer_only_loss=True,
        eos_token_id=tokenizer.eos_id,
    )


# A conversation that opens its answer with the marker. The packer's stored mask is shifted by one: entry i gates the
# prediction of token i + 1, so the marker (index 3) is predicted under entry 2.
CONVERSATION = [3, 4, 5, LISTED, 6, 3, 4]
MARKER_TRAINED = [0, 0, 1, 1, 1, 1, 0]
MARKER_UNTRAINED = [0, 0, 0, 1, 1, 1, 0]  # the marker sits outside the trained span, like the MQ EM packs


def _rows(stored_mask: list[int], count: int) -> list[dict]:
    return [{"input_ids": CONVERSATION, "loss_mask": stored_mask, "seq_start_id": [0]} for _ in range(count)]


def _mock_config(tmp_path: Path) -> MockGPTDatasetConfig:
    config = MockGPTDatasetConfig(
        seq_length=16, random_seed=1234, reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False
    )
    config.finalize()
    return config


def _unpacked_config(tmp_path: Path) -> FinetuningDatasetConfig:
    return FinetuningDatasetConfig(dataset_root=tmp_path, seq_length=64)


def _masking():
    return masking_with_null_tokenizer([LISTED], VOCAB_SIZE)


def _measuring():
    return measuring_with_null_tokenizer([LISTED], VOCAB_SIZE)


class TestEnabledRunNeedsPositiveEvidence:
    def test_trainable_marker_targets_pass(self, tmp_path):
        scans = _scans(_gpt_config([_corpus(tmp_path / "a", _documents(20, listed=True))]))
        verdict = token_masking_data_verdict(scans, None, _masking())
        assert verdict.errors == () and verdict.findings == ()

    def test_no_marker_target_stops_the_run(self, tmp_path):
        paths = [_corpus(tmp_path / name, _documents(20)) for name in ("a", "b")]
        scans = _scans(_gpt_config(paths))
        assert {scan.stop_reason for scan in scans} == {"exhausted"}
        [error] = token_masking_data_verdict(scans, None, _masking()).errors
        assert error.startswith("no target in the training data is one of the ids [1000] (")
        assert "across 2 sources" in error and error.endswith("re-tokenized with the run's tokenizer")
        assert "max_scan_tokens_per_source" not in error, "the scan read all of the data; its budget is no remedy"

    def test_no_marker_target_within_the_token_budget_names_the_budget(self, tmp_path):
        scans = _scans(_gpt_config([_corpus(tmp_path / "a", _documents(20))]), max_tokens=50)
        assert scans[0].stop_reason == "token_budget"
        [error] = token_masking_data_verdict(scans, None, _masking()).errors
        assert error.startswith("no target in the training data is one of the ids [1000] (")
        assert error.endswith(
            "re-tokenized with the run's tokenizer; or raise logger.data_samples.max_scan_tokens_per_source: the scan "
            "stopped on that budget in ['a'] before reading all of their data"
        )

    def test_marker_targets_only_at_untrained_positions_stop_the_run(self, tmp_path):
        """However few: there is no minimum count any more, since the run must prove masking removes something."""
        scans = _packed_scans(tmp_path, _rows(MARKER_UNTRAINED, 3))
        assert (sum(s.listed_targets for s in scans), sum(s.listed_trainable_targets for s in scans)) == (3, 0)
        [error] = token_masking_data_verdict(scans, None, _masking()).errors
        assert "occur 3 times as targets in the training data" in error
        assert "never at a position that carries loss" in error
        assert error.endswith("such a stage must run with token masking off")

    def test_untrained_marker_targets_within_the_token_budget_name_the_budget_too(self, tmp_path):
        """Trainable marker targets may lie in data the scan did not read."""
        scans = _packed_scans(tmp_path, _rows(MARKER_UNTRAINED, 30), max_tokens=20)
        assert scans[0].stop_reason == "token_budget" and scans[0].listed_targets > 0
        [error] = token_masking_data_verdict(scans, None, _masking()).errors
        assert "never at a position that carries loss" in error
        assert error.endswith(
            "such a stage must run with token masking off; or raise logger.data_samples.max_scan_tokens_per_source: "
            "the scan stopped on that budget in ['pack'] before reading all of their data"
        )

    def test_a_scan_out_of_time_before_the_evidence_stops_the_run(self, tmp_path):
        scans = _scans(_gpt_config([_corpus(tmp_path / "a", _documents(20, listed=True))]), deadline_seconds=-1)
        assert scans[0].stop_reason == "time_budget"
        [error] = token_masking_data_verdict(scans, None, _masking()).errors
        assert "max_scan_seconds before finishing ['a']" in error and "Lustre" in error

    def test_evidence_found_before_the_time_budget_ran_out_passes(self, tmp_path):
        """A packed source's random documents come from its first row groups, which count as scanned."""
        scans = _packed_scans(tmp_path, _rows(MARKER_TRAINED, 4), deadline_seconds=-1, documents_per_source=1)
        assert scans[0].stop_reason == "time_budget" and scans[0].listed_trainable_targets == 1
        assert token_masking_data_verdict(scans, None, _masking()).errors == ()

    def test_evidence_only_in_a_source_of_blend_weight_0_stops_the_run(self, tmp_path):
        marked = _corpus(tmp_path / "marked", _documents(10, listed=True))
        plain = _corpus(tmp_path / "plain", _documents(10))
        [error] = token_masking_data_verdict(_scans(_gpt_config(["0", marked, "1", plain])), None, _masking()).errors
        assert "across 1 sources; they occur only in ['marked'], whose blend weight is 0" in error
        weighted = _scans(_gpt_config(["0.1", marked, "0.9", plain]))
        assert token_masking_data_verdict(weighted, None, _masking()).errors == ()

    def test_evidence_only_in_documents_another_split_holds_out_stops_the_run(self, tmp_path):
        """Documents 15-19 hold the marker; split 3:1 trains on documents 0-14 only."""
        documents = _documents(15) + _documents(5, listed=True)
        held_out = _scans(_gpt_config([_corpus(tmp_path / "a", documents)], split="3,1,0"))
        [error] = token_masking_data_verdict(held_out, None, _masking()).errors
        assert error.startswith("no target in the training data is one of the ids [1000]")
        trained = _scans(_gpt_config([_corpus(tmp_path / "b", _documents(5, listed=True) + _documents(15))], "3,1,0"))
        assert token_masking_data_verdict(trained, None, _masking()).errors == ()

    @pytest.mark.parametrize("dataset_config", [_mock_config, _unpacked_config], ids=["mock", "unpacked"])
    def test_data_the_scan_cannot_read_stops_the_run(self, tmp_path, dataset_config):
        tokenizer = build_tokenizer(hf_tokenizer_config(build_tiny_hf_tokenizer(tmp_path / "tok")))
        scans, reason = scan_dataset(dataset_config(tmp_path), tokenizer, _masking(), DataSamplesConfig())
        assert scans == [] and reason
        [error] = token_masking_data_verdict(scans, reason, _masking()).errors
        assert error.startswith(f"the training data cannot be scanned for the ids [1000]: {reason}.")
        assert "pipeline_data_prepare.py" in error

    def test_a_split_marker_stops_the_run(self, tmp_path):
        scans = _scans(_gpt_config([_corpus(tmp_path / "a", _documents(20, listed=True, split=True))]))
        [error] = token_masking_data_verdict(scans, None, _masking()).errors
        assert "split into several ordinary tokens" in error and "a: 20" in error.replace("'", "")

    def test_out_of_vocabulary_tokens_stop_the_run(self, tmp_path):
        scans = _scans(_gpt_config([_corpus(tmp_path / "a", _documents(20, listed=True, out_of_vocab=True))]))
        [error] = token_masking_data_verdict(scans, None, _masking()).errors
        assert "outside the tokenizer's vocabulary" in error


class TestRunsThatDoNotMaskOnlyReport:
    def test_a_measuring_run_reports_every_problem(self, tmp_path):
        scans = _scans(_gpt_config([_corpus(tmp_path / "a", _documents(20, split=True, out_of_vocab=True))]))
        verdict = token_masking_data_verdict(scans, None, _measuring())
        assert verdict.errors == ()
        out_of_vocab, split, missing = verdict.findings
        assert out_of_vocab.startswith("token ids outside the tokenizer's vocabulary in {'a': 20}")
        assert split.startswith("the text of ['1000'] appears split into several ordinary tokens in {'a': 20}")
        assert missing.startswith("no target in the training data is one of the ids [1000]")

    def test_a_run_that_measures_nothing_reports_out_of_vocabulary_tokens_only(self, tmp_path):
        scans = _scans(_gpt_config([_corpus(tmp_path / "a", _documents(20, split=True, out_of_vocab=True))]))
        verdict = token_masking_data_verdict(scans, None, no_token_masking())
        assert verdict.errors == ()
        [finding] = verdict.findings
        assert "outside the tokenizer's vocabulary" in finding

    def test_data_the_scan_cannot_read_is_reported(self):
        reason = "mock dataset: the run trains on generated tokens, not on a corpus"
        measuring = token_masking_data_verdict([], reason, _measuring())
        assert measuring.errors == ()
        [finding] = measuring.findings
        assert finding.startswith(f"the training data cannot be scanned for the ids [1000]: {reason}.")
        idle = token_masking_data_verdict([], reason, no_token_masking())
        assert idle == DataVerdict(errors=(), findings=(f"the training data could not be inspected: {reason}",))


class TestMissingTrainableTargets:
    def test_none_when_a_source_training_reads_holds_a_trainable_target(self, tmp_path):
        scans = _scans(_gpt_config([_corpus(tmp_path / "a", _documents(5, listed=True))]))
        assert missing_trainable_targets(scans, None, _measuring()) is None

    def test_the_cause_otherwise(self, tmp_path):
        scans = _scans(_gpt_config([_corpus(tmp_path / "a", _documents(5))]))
        assert missing_trainable_targets(scans, None, _measuring()).startswith("no target in the training data")


class TestNoTrainableTarget:
    """The cause is worded for the data it judges: masking off is a remedy for training data, never for a held-out
    set, which is replaced instead."""

    @pytest.mark.parametrize("listed", [0, 4], ids=["absent", "untrained"])
    def test_the_remedy_fits_the_data(self, listed):
        training = no_trainable_target(TRAINING_DATA, _masking(), listed, "what was read")
        held_out = no_trainable_target(HELD_OUT_SET, _masking(), listed, "what was read")
        assert "in the training data" in training and "in the held-out set" in held_out
        assert "(what was read)" in training and "(what was read)" in held_out
        assert ("such a stage must run with token masking off" in training) is bool(listed)
        assert "token masking off" not in held_out
        assert held_out.endswith("choose a held-out set whose marker targets carry loss")


class TestSplitForms:
    def test_an_hf_tokenizer_splits_the_marker_text(self, tmp_path):
        tokenizer = build_tokenizer(hf_tokenizer_config(build_tiny_hf_tokenizer(tmp_path / "tok")))
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
        "wandb_project, tables, decision, wanted",
        [
            ("p", True, no_token_masking, True),
            ("p", False, no_token_masking, False),
            (None, True, no_token_masking, False),
            (None, False, _masking, True),
            (None, False, _measuring, False),
            ("p", True, _measuring, True),
        ],
    )
    def test_whether_a_run_scans_follows_its_config(self, tmp_path, wandb_project, tables, decision, wanted):
        cfg = _run_config([_corpus(tmp_path / "a", _documents(5))], wandb_project=wandb_project, tables=tables)
        assert inspection_wanted(cfg, decision()) is wanted

    def test_a_run_that_does_not_scan_returns_nothing(self, tmp_path, gloo_group_of_one):
        cfg = _run_config([_corpus(tmp_path / "a", _documents(5))], wandb_project=None, tables=True)
        tokenizer = build_tokenizer(null_tokenizer_config(VOCAB_SIZE))
        assert inspect_training_data(cfg, tokenizer, _measuring()) is None
        assert inspect_training_data(cfg, tokenizer, no_token_masking()) is None

    @pytest.mark.parametrize(
        "decision, marked_as", [(_masking, "masked"), (_measuring, "measured")], ids=["masking", "measuring"]
    )
    def test_a_passing_scan_returns_the_scans_and_logs_the_tables(
        self, tmp_path, gloo_group_of_one, decision, marked_as
    ):
        paths = [
            _corpus(tmp_path / "marked", _documents(10, listed=True)),
            _corpus(tmp_path / "plain", _documents(10)),
        ]
        cfg = _run_config(paths, wandb_project="p", tables=True)
        tokenizer = build_tokenizer(null_tokenizer_config(VOCAB_SIZE))
        resolved = decision()
        scans = inspect_training_data(cfg, tokenizer, resolved)
        assert [scan.source.label for scan in scans] == ["marked", "plain"]
        recorder = _RecordingWandb()
        log_inspection_tables(recorder, scans, cfg, tokenizer, resolved, step=7)
        [(tables, step)] = recorder.calls
        assert step == 7
        assert set(tables) == {"data_samples/sources", "data_samples/documents", "data_samples/masked_documents"}
        texts = [row[7] for row in tables["data_samples/masked_documents"].data]
        assert texts and all(f"⟦{marked_as}:{LISTED}⟧" in text for text in texts)

    def test_tables_are_not_logged_when_disabled_or_without_wandb(self, tmp_path, gloo_group_of_one):
        cfg = _run_config([_corpus(tmp_path / "a", _documents(5, listed=True))], wandb_project=None, tables=False)
        tokenizer = build_tokenizer(null_tokenizer_config(VOCAB_SIZE))
        scans = inspect_training_data(cfg, tokenizer, _masking())
        recorder = _RecordingWandb()
        log_inspection_tables(recorder, scans, cfg, tokenizer, _masking(), step=0)
        log_inspection_tables(None, scans, cfg, tokenizer, _masking(), step=0)
        assert recorder.calls == []

    def test_an_enabled_run_without_evidence_stops(self, tmp_path, gloo_group_of_one):
        cfg = _run_config([_corpus(tmp_path / "plain", _documents(10))], wandb_project=None, tables=False)
        tokenizer = build_tokenizer(null_tokenizer_config(VOCAB_SIZE))
        with pytest.raises(TokenMaskingError) as raised:
            inspect_training_data(cfg, tokenizer, _masking())
        assert str(raised.value).startswith(
            "token masking data check failed (a run with masking enabled must show, before training, a target of a "
            "masked id that carries loss in its training data):\n- no target in the training data is one of the ids"
        )

    def test_a_measuring_run_without_evidence_trains(self, tmp_path, gloo_group_of_one):
        cfg = _run_config([_corpus(tmp_path / "plain", _documents(10))], wandb_project="p", tables=True)
        tokenizer = build_tokenizer(null_tokenizer_config(VOCAB_SIZE))
        [scan] = inspect_training_data(cfg, tokenizer, _measuring())
        assert scan.listed_targets == 0

    def test_a_failed_scan_stops_the_run(self, tmp_path, gloo_group_of_one):
        cfg = _run_config([_corpus(tmp_path / "a", _documents(5))], wandb_project="p", tables=True)
        cfg.dataset.blend = ([str(tmp_path / "missing" / "corpus")], None)
        tokenizer = build_tokenizer(null_tokenizer_config(VOCAB_SIZE))
        with pytest.raises(TokenMaskingError, match="the training-data scan failed on rank 0"):
            inspect_training_data(cfg, tokenizer, _measuring())

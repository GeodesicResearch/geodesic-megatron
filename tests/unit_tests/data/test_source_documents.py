# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Tests for listing a run's training-data sources and scanning them for listed token ids.

Every corpus here is real: ``.bin/.idx`` pairs written by Megatron's own ``IndexedDatasetBuilder`` and packed parquet
written by ``write_packed_parquet``, read back by the scan and, for the training split and the trainable rule, by the
training datasets themselves.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset
from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

from megatron.bridge.data.datasets.packed_parquet import write_packed_parquet
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.data.datasets.sft import create_sft_dataset
from megatron.bridge.data.source_documents import (
    DataSource,
    SplitForm,
    scan_settings,
    scan_sources,
    training_data_sources,
)
from megatron.bridge.training.config import FinetuningDatasetConfig, GPTDatasetConfig, MockGPTDatasetConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from tests.unit_tests.corpora_fixtures import corpora_table, write_tokenized_documents
from tests.unit_tests.token_masking_fixtures import (
    EOS_ID,
    MARKER_ID,
    build_tiny_hf_tokenizer,
    hf_tokenizer_config,
    null_tokenizer_config,
)


LISTED = 131072
EOD = 2
SPLIT_FORM = [700, 701, 702]
OUT_OF_VOCAB = 140000
VOCAB_SIZE = 131584


def indexed_documents(count: int, *, listed_at=(), out_of_vocab_at=(), split_form_at=()) -> list[list[int]]:
    """Documents of filler ids in [100, 600), each ending in EOD, with markers inserted into the named ones."""
    documents = []
    for d in range(count):
        document = [100 + (7 * d + 3 * j) % 500 for j in range(5 + d % 7)]
        if d in listed_at:
            document.insert(2, LISTED)
        if d in out_of_vocab_at:
            document.insert(1, OUT_OF_VOCAB)
        if d in split_form_at:
            document[1:1] = SPLIT_FORM
        documents.append(document + [EOD])
    return documents


def write_corpus(root: Path, documents: list[list[int]]) -> str:
    write_tokenized_documents(root, documents)
    return str(root / corpora_table.TOKENIZED_PREFIX)


def gpt_config(data_path: list[str], *, eod_mask_loss: bool, split: str = "1,0,0", **kwargs) -> GPTDatasetConfig:
    config = GPTDatasetConfig(
        seq_length=8,
        data_path=data_path,
        split=split,
        random_seed=1234,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=eod_mask_loss,
        **kwargs,
    )
    config.finalize()
    return config


def scan(sources, **overrides):
    """``scan_sources`` with every argument stated; a test overrides the ones it is about."""
    arguments = dict(
        listed_token_ids=[LISTED],
        split_forms=[SplitForm(core=tuple(SPLIT_FORM))],
        vocab_size=VOCAB_SIZE,
        documents_per_source=4,
        listed_documents_per_source=10,
        max_scan_tokens_per_source=1_000_000,
        deadline=time.monotonic() + 60,
        seed=1234,
        eod_token_id=EOD,
        eod_mask_loss=False,
        answer_only_loss=None,
        eos_token_id=None,
    )
    arguments.update(overrides)
    return scan_sources(sources, **arguments)


@pytest.fixture
def blend(tmp_path):
    """Three corpora as a weighted blend; only the first holds listed ids, a split form and an OOV token."""
    data = tmp_path / "data"
    first = write_corpus(
        data / "climbmix_full" / "shard0",
        indexed_documents(40, listed_at=(3, 17, 25), out_of_vocab_at=(30,), split_form_at=(8, 9)),
    )
    second = write_corpus(data / "climbmix_full" / "shard1", indexed_documents(30))
    third = write_corpus(data / "zyda", indexed_documents(20))
    Path(f"{first}.provenance.json").write_text(json.dumps({"parameters": {"tokenizer": "org/base-tokenizer"}}))
    (data / "zyda" / "pipeline_results.json").write_text(json.dumps({"tokenizer": "org/zyda-tokenizer"}))
    return [first, second, third]


class TestIndexedSources:
    def test_blend_order_weights_labels_and_recorded_tokenizers(self, blend):
        sources, reason = training_data_sources(
            gpt_config(["1", blend[0], "1", blend[1], "2", blend[2]], eod_mask_loss=False), None
        )
        assert reason is None
        assert [s.index for s in sources] == [0, 1, 2]
        assert [s.path for s in sources] == blend
        assert [s.kind for s in sources] == ["indexed"] * 3
        assert [s.weight for s in sources] == [0.25, 0.25, 0.5]
        assert [s.training_split for s in sources] == [(0.0, 1.0)] * 3
        assert [s.label for s in sources] == ["climbmix_full/shard0", "climbmix_full/shard1", "zyda"]
        assert [s.tokenizer_recorded for s in sources] == ["org/base-tokenizer", None, "org/zyda-tokenizer"]

    def test_single_unweighted_source_is_labelled_by_its_directory(self, blend):
        sources, reason = training_data_sources(gpt_config([blend[1]], eod_mask_loss=False), None)
        assert reason is None
        assert [(s.label, s.weight) for s in sources] == [("shard1", None)]

    def test_blend_per_split_reads_the_training_blend(self, blend):
        config = GPTDatasetConfig(
            seq_length=8,
            blend_per_split=[([blend[2]], None), ([blend[0]], None), None],
            split=None,
            random_seed=1234,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
        )
        config.finalize()
        sources, reason = training_data_sources(config, None)
        assert reason is None
        # Megatron reads every document of a split's own blend.
        assert [(s.path, s.training_split) for s in sources] == [(blend[2], (0.0, 1.0))]

    def test_a_split_blend_trains_on_the_split_matrix_training_row(self, blend):
        sources, reason = training_data_sources(gpt_config(blend[:2], eod_mask_loss=False, split="8,2,0"), None)
        assert reason is None
        assert [s.training_split for s in sources] == [(0.0, 0.8)] * 2

    def test_a_split_that_gives_training_nothing_has_no_sources(self, blend):
        sources, reason = training_data_sources(gpt_config([blend[0]], eod_mask_loss=False, split="0,1,0"), None)
        assert sources == [] and reason == "split '0,1,0' gives the training split none of the blend"

    def test_mock_dataset_has_no_sources_and_says_why(self):
        config = MockGPTDatasetConfig(
            seq_length=8, random_seed=1234, reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False
        )
        config.finalize()
        assert training_data_sources(config, None) == (
            [],
            "mock dataset: the run trains on generated tokens, not on a corpus",
        )

    def test_unknown_dataset_type_has_no_sources_and_says_why(self):
        sources, reason = training_data_sources(object(), None)
        assert sources == [] and "object" in reason

    def test_scan_settings_come_from_the_config(self, blend):
        settings = scan_settings(gpt_config([blend[0]], eod_mask_loss=True))
        assert (settings.seed, settings.eod_mask_loss, settings.answer_only_loss) == (1234, True, None)


class TestSplitFormMatching:
    """A split form's core must match exactly; its edges match any token in their sets."""

    CORE = (700, 701, 702)
    OPEN = (690, 691)  # tokens whose text ends with the form's first piece, e.g. "<" and " <"
    CLOSE = (695, 696)  # tokens whose text starts with its last piece, e.g. ">" and ">\n"

    def _count(self, tmp_path, documents, form):
        corpus = write_corpus(tmp_path / "corpus", [document + [EOD] for document in documents])
        sources, _ = training_data_sources(gpt_config([corpus], eod_mask_loss=False), None)
        [result] = scan(sources, split_forms=[form])
        assert result.stop_reason == "exhausted"
        return result.split_form_occurrences

    def test_the_core_between_any_edge_tokens_counts(self, tmp_path):
        form = SplitForm(core=self.CORE, before=frozenset(self.OPEN), after=frozenset(self.CLOSE))
        documents = [
            [100, 690, *self.CORE, 695, 101],
            [100, 691, *self.CORE, 696, 101],
            [100, 690, *self.CORE, 696, 691, *self.CORE, 695],
        ]
        assert self._count(tmp_path, documents, form) == 4

    def test_the_core_without_its_edges_does_not_count(self, tmp_path):
        form = SplitForm(core=self.CORE, before=frozenset(self.OPEN), after=frozenset(self.CLOSE))
        documents = [
            [100, *self.CORE, 695, 101],  # no opening edge: the core's text alone, e.g. "stage=training"
            [100, 690, *self.CORE, 101],  # no closing edge
            [100, 690, 700, 701, 695, 101],  # part of the core
        ]
        assert self._count(tmp_path, documents, form) == 0

    def test_a_form_without_edges_matches_its_core_alone(self, tmp_path):
        documents = [[100, *self.CORE, 101], [100, 690, *self.CORE, 695]]
        assert self._count(tmp_path, documents, SplitForm(core=self.CORE)) == 2


class TestIndexedScan:
    def test_counts_cover_the_whole_corpus_when_it_fits_the_budget(self, blend):
        documents = indexed_documents(40, listed_at=(3, 17, 25), out_of_vocab_at=(30,), split_form_at=(8, 9))
        sources, _ = training_data_sources(gpt_config([blend[0]], eod_mask_loss=False), None)
        [result] = scan(sources)
        assert result.stop_reason == "exhausted"
        assert result.documents_scanned == 40
        assert result.tokens_scanned == sum(len(d) for d in documents)
        assert (result.listed_targets, result.listed_trainable_targets) == (3, 3)
        assert result.split_form_occurrences == 2
        assert result.out_of_vocab_tokens == 1

    def test_runs_partition_the_corpus_into_whole_documents(self, blend):
        """With the budget equal to the corpus, the corpus is read as 16 runs that cover every document once."""
        documents = indexed_documents(40, listed_at=(3, 17, 25), out_of_vocab_at=(30,), split_form_at=(8, 9))
        total = sum(len(d) for d in documents)
        sources, _ = training_data_sources(gpt_config([blend[0]], eod_mask_loss=False), None)
        [result] = scan(sources, max_scan_tokens_per_source=total)
        assert result.stop_reason == "exhausted"
        assert (result.documents_scanned, result.tokens_scanned) == (40, total)
        assert (result.listed_targets, result.split_form_occurrences, result.out_of_vocab_tokens) == (3, 2, 1)

    def test_listed_documents_are_those_containing_a_listed_id(self, blend):
        documents = indexed_documents(40, listed_at=(3, 17, 25), out_of_vocab_at=(30,), split_form_at=(8, 9))
        sources, _ = training_data_sources(gpt_config([blend[0]], eod_mask_loss=False), None)
        [result] = scan(sources)
        assert sorted(d.reference for d in result.listed_documents) == ["document 17", "document 25", "document 3"]
        for document in result.listed_documents:
            index = int(document.reference.split()[-1])
            assert document.token_ids.dtype == np.int64
            assert document.token_ids.tolist() == documents[index]
            assert document.trainable_target.all() and document.first_token_is_target

    def test_counting_continues_after_the_document_quota_is_met(self, blend):
        sources, _ = training_data_sources(gpt_config([blend[0]], eod_mask_loss=False), None)
        [result] = scan(sources, listed_documents_per_source=1)
        assert len(result.listed_documents) == 1
        assert result.listed_targets == 3

    def test_eod_mask_loss_untrains_a_target_whose_input_is_eod(self, tmp_path):
        prefix = write_corpus(tmp_path / "eod", [[LISTED, 11, EOD], [LISTED, 12, EOD], [13, EOD, LISTED, EOD]])
        sources, _ = training_data_sources(gpt_config([prefix], eod_mask_loss=True), None)
        [masked] = scan(sources, eod_mask_loss=True)
        [unmasked] = scan(sources, eod_mask_loss=False)
        # Nothing precedes document 0, so its first token still trains; every later document's first token follows
        # the EOD that ends the document before it, and document 2's listed token follows an EOD of its own.
        assert (masked.listed_targets, masked.listed_trainable_targets) == (3, 1)
        assert (unmasked.listed_targets, unmasked.listed_trainable_targets) == (3, 3)
        trainable = {d.reference: d.trainable_target.tolist() for d in masked.listed_documents}
        assert trainable == {
            "document 0": [True, True, True],
            "document 1": [False, True, True],
            "document 2": [False, True, False, True],
        }

    def test_a_scan_in_many_runs_reads_every_document_once(self, tmp_path):
        documents = [[LISTED] + document for document in indexed_documents(40)]
        prefix = write_corpus(tmp_path / "runs", documents)
        sources, _ = training_data_sources(gpt_config([prefix], eod_mask_loss=True), None)
        total = sum(len(d) for d in documents)
        # A budget of the whole corpus is spent over many runs from random offsets.
        [result] = scan(
            sources, eod_mask_loss=True, max_scan_tokens_per_source=total, listed_documents_per_source=len(documents)
        )
        assert (result.stop_reason, result.documents_scanned, result.tokens_scanned) == ("exhausted", 40, total)
        # Every document starts with a listed id, whose input is the EOD ending the document before it, except in
        # document 0, which nothing precedes.
        assert (result.listed_targets, result.listed_trainable_targets) == (40, 1)
        references = [d.reference for d in result.listed_documents]
        assert sorted(references) == sorted(f"document {k}" for k in range(40))
        assert references != [f"document {k}" for k in range(40)]
        for document in result.listed_documents:
            index = int(document.reference.split()[-1])
            assert document.token_ids.tolist() == documents[index]
            assert document.trainable_target.tolist() == [index == 0] + [True] * (len(documents[index]) - 1)

    def test_random_documents_are_the_corpus_documents_and_depend_on_the_seed(self, blend):
        documents = indexed_documents(30)
        sources, _ = training_data_sources(gpt_config([blend[1]], eod_mask_loss=False), None)
        [first] = scan(sources, documents_per_source=10, seed=1)
        [again] = scan(sources, documents_per_source=10, seed=1)
        [other] = scan(sources, documents_per_source=10, seed=2)
        references = [d.reference for d in first.documents]
        assert len(set(references)) == 10
        assert references == [d.reference for d in again.documents]
        assert references != [d.reference for d in other.documents]
        for document in first.documents:
            assert document.token_ids.tolist() == documents[int(document.reference.split()[-1])]

    def test_token_budget_stops_the_scan(self, blend):
        sources, _ = training_data_sources(gpt_config([blend[0]], eod_mask_loss=False), None)
        [result] = scan(sources, max_scan_tokens_per_source=40)
        assert result.stop_reason == "token_budget"
        assert 40 <= result.tokens_scanned < sum(len(d) for d in indexed_documents(40))

    def test_a_passed_deadline_stops_every_scan_but_documents_are_still_drawn(self, blend):
        sources, _ = training_data_sources(gpt_config(blend, eod_mask_loss=False), None)
        results = scan(sources, deadline=time.monotonic() - 1)
        assert [r.stop_reason for r in results] == ["time_budget"] * 3
        assert [r.tokens_scanned for r in results] == [0, 0, 0]
        assert [len(r.documents) for r in results] == [4, 4, 4]

    def test_bad_arguments_are_refused(self, blend):
        sources, _ = training_data_sources(gpt_config([blend[0]], eod_mask_loss=False), None)
        with pytest.raises(ValueError, match="two or more"):
            scan(sources, split_forms=[SplitForm(core=(LISTED,))])
        with pytest.raises(ValueError, match="edge sets must not be empty"):
            scan(sources, split_forms=[SplitForm(core=(LISTED,), before=frozenset(), after=frozenset({1}))])
        with pytest.raises(ValueError, match="eod_mask_loss is required"):
            scan(sources, eod_mask_loss=None)
        with pytest.raises(ValueError, match="eod_token_id is required"):
            scan(sources, eod_mask_loss=True, eod_token_id=None)


def megatron_training_documents(config: GPTDatasetConfig) -> list[int]:
    """The documents the training split of Megatron's own ``GPTDataset`` reads, built as a pretraining run builds it."""
    train, _, _ = BlendedMegatronDatasetBuilder(GPTDataset, [4, 4, 4], lambda: True, config).build()
    return train.indices.tolist()


def referenced(documents) -> list[int]:
    return sorted(int(document.reference.split()[-1]) for document in documents)


class TestTrainingRange:
    """The scan reads exactly the documents the training split trains on, never those another split holds out."""

    def split_config(self, tmp_path: Path, prefix: str, split: str) -> GPTDatasetConfig:
        return gpt_config(
            [prefix],
            eod_mask_loss=False,
            split=split,
            tokenizer=build_tokenizer(null_tokenizer_config(VOCAB_SIZE)),
            path_to_cache=str(tmp_path / "cache"),
        )

    @pytest.mark.parametrize("split", ["3,1,0", "17,3,0", "1,0,0"])
    def test_the_scan_reads_the_documents_gpt_dataset_trains_on(self, tmp_path, split):
        marked = (3, 25, 33, 40)
        documents = indexed_documents(41, listed_at=marked)
        config = self.split_config(tmp_path, write_corpus(tmp_path / "corpus", documents), split)
        trained = megatron_training_documents(config)
        sources, _ = training_data_sources(config, None)
        [result] = scan(sources, documents_per_source=len(documents), listed_documents_per_source=len(documents))
        assert result.stop_reason == "exhausted"
        assert referenced(result.documents) == trained
        assert (result.documents_scanned, result.tokens_scanned) == (
            len(trained),
            sum(len(documents[d]) for d in trained),
        )
        assert referenced(result.listed_documents) == [d for d in marked if d in trained]
        assert result.listed_targets == len([d for d in marked if d in trained])

    def test_a_document_is_a_sequence_of_the_index(self, tmp_path):
        """GPTDataset splits, shuffles and concatenates the index's sequences, so the scan's documents are those.

        Ten documents of two sequences each; split 3:1 trains on sequences [0, 15), which ends inside document 7, so
        of the marker in each of document 7's two sequences only the first is trained on.
        """
        prefix = tmp_path / "sentences" / corpora_table.TOKENIZED_PREFIX
        prefix.parent.mkdir()
        builder = IndexedDatasetBuilder(f"{prefix}.bin", dtype=np.int32)
        sequences = []
        for d in range(10):
            first, second = [100 + d, 101 + d], [200 + d, 300, EOD]
            if d == 7:
                first[1] = second[1] = LISTED
            for sequence in (first, second):
                builder.add_item(torch.tensor(sequence, dtype=torch.int32))
                sequences.append(sequence)
            builder.end_document()
        builder.finalize(f"{prefix}.idx")
        config = self.split_config(tmp_path, str(prefix), "3,1,0")
        assert megatron_training_documents(config) == list(range(15))
        sources, _ = training_data_sources(config, None)
        [result] = scan(sources, documents_per_source=20, listed_documents_per_source=20)
        assert referenced(result.documents) == list(range(15))
        assert all(d.token_ids.tolist() == sequences[int(d.reference.split()[-1])] for d in result.documents)
        assert (result.listed_targets, referenced(result.listed_documents)) == (1, [14])

    def test_a_range_that_starts_inside_the_corpus_is_read_from_its_start(self, blend):
        """Megatron's training row always starts at 0; a source's range is honoured wherever it starts."""
        [source], _ = training_data_sources(gpt_config([blend[0]], eod_mask_loss=False), None)
        [result] = scan(
            [replace(source, training_split=(0.25, 0.75))], documents_per_source=40, listed_documents_per_source=40
        )
        assert referenced(result.documents) == list(range(10, 30))
        assert result.documents_scanned == 20
        # Of the listed documents 3, 17 and 25, and the split forms in 8 and 9, only those inside [10, 30) count.
        assert (referenced(result.listed_documents), result.split_form_occurrences) == ([17, 25], 0)


EOS = EOS_ID
Q = MARKER_ID
W = list(range(100, 130))  # filler ids; the scan and the collate never decode them

# Each pack is a list of (prompt, answer) conversations; a conversation is prompt + answer + [EOS] and trains on its
# answer and EOS. Q sits in prompts, in answers, at a conversation's first position (not a target at all) and right
# after a mid-conversation EOS (whose prediction the collate never trains).
SHARD_A = [
    [([W[0], Q, W[1]], [W[2], Q, W[3]]), ([Q, W[4]], [W[5]])],
    [([W[6]], [W[7], W[8]]), ([W[9], W[10]], [Q])],
    [([W[11]], [W[12]]), ([W[13]], [W[14], EOS, Q, W[15]]), ([W[16]], [W[29], W[17]])],
]
SHARD_B = [
    [([W[18], W[19]], [W[20]]), ([W[21]], [W[22], Q])],
    [([W[23]], [W[24]])],
    [([W[25], Q], [W[26]]), ([W[27]], [W[28]])],
]
CONVERSATIONS = sum(len(pack) for pack in SHARD_A + SHARD_B)
CONVERSATIONS_WITH_Q = 6
Q_TARGETS = 6
Q_TRAINABLE_ANSWER_ONLY = 3
Q_TRAINABLE_ALL = 5


def pack_row(conversations) -> dict:
    """A packed row as the packer writes it: the stored loss mask at ``i`` gates predicting token ``i + 1``."""
    input_ids, loss_mask, starts = [], [], []
    for prompt, answer in conversations:
        tokens = prompt + answer + [EOS]
        starts.append(len(input_ids))
        input_ids.extend(tokens)
        loss_mask.extend(len(prompt) <= i + 1 < len(tokens) for i in range(len(tokens)))
    return {"input_ids": input_ids, "loss_mask": loss_mask, "seq_start_id": starts}


@pytest.fixture
def packed_glob(tmp_path):
    """Two shards of one-row row groups, in the packer's directory layout, matched by one glob."""
    for shard, packs in (("shard0", SHARD_A), ("shard1", SHARD_B)):
        directory = tmp_path / "mix" / shard / "packed" / "org--tiny-tokenizer_pad_seq_to_mult1"
        directory.mkdir(parents=True)
        write_packed_parquet([pack_row(p) for p in packs], directory / "training_64.idx.parquet", row_group_size=1)
    return str(
        tmp_path / "mix" / "shard*" / "packed" / "org--tiny-tokenizer_pad_seq_to_mult1" / "training_64.idx.parquet"
    )


@pytest.fixture
def tokenizer(tmp_path):
    """A real Megatron tokenizer whose EOS is ``EOS``; the packed dataset reads nothing else from it."""
    return build_tokenizer(hf_tokenizer_config(build_tiny_hf_tokenizer(tmp_path / "tokenizer", None)))


def packed_config(
    root: Path, packed_train_data_path: str | None, *, answer_only_loss: bool
) -> FinetuningDatasetConfig:
    return FinetuningDatasetConfig(
        dataset_root=root,
        seq_length=64,
        dataset_kwargs={"answer_only_loss": answer_only_loss},
        packed_sequence_specs=PackedSequenceSpecs(
            packed_sequence_size=64,
            packed_train_data_path=packed_train_data_path,
            tokenizer_model_name="org--tiny-tokenizer",
        ),
    )


def packed_scan(sources, **overrides):
    arguments = dict(
        listed_token_ids=[Q],
        split_forms=[],
        vocab_size=1000,
        eod_token_id=None,
        eod_mask_loss=None,
        answer_only_loss=True,
        eos_token_id=EOS,
    )
    arguments.update(overrides)
    return scan(sources, **arguments)


class TestPackedSources:
    def test_a_glob_of_shards_is_one_source(self, tmp_path, packed_glob, tokenizer):
        sources, reason = training_data_sources(packed_config(tmp_path, packed_glob, answer_only_loss=True), tokenizer)
        assert reason is None
        # The label names the data ("mix"), not the shard glob or the packer's tokenizer directory beneath it.
        assert sources == [
            DataSource(
                index=0,
                label="mix",
                path=packed_glob,
                kind="packed_parquet",
                weight=None,
                training_split=(0.0, 1.0),
                tokenizer_recorded="org--tiny-tokenizer",
            )
        ]

    def test_an_omitted_path_is_the_builders_default_pack_path(self, tmp_path, tokenizer):
        root = tmp_path / "dataset"
        config = packed_config(root, None, answer_only_loss=True)
        sources, reason = training_data_sources(config, tokenizer)
        assert sources == [] and "not on disk yet" in reason
        default = root / "packed" / "org--tiny-tokenizer_pad_seq_to_mult1" / "training_64.idx.parquet"
        write_packed_parquet([pack_row(SHARD_A[0])], default)
        sources, reason = training_data_sources(config, tokenizer)
        assert reason is None
        assert [(s.path, s.label, s.tokenizer_recorded) for s in sources] == [
            (str(default), "dataset", "org--tiny-tokenizer")
        ]

    def test_unpacked_fine_tuning_data_has_no_sources(self, tmp_path, tokenizer):
        config = FinetuningDatasetConfig(dataset_root=tmp_path, seq_length=64)
        sources, reason = training_data_sources(config, tokenizer)
        assert sources == [] and "unpacked" in reason

    def test_scan_settings_read_answer_only_loss_and_its_default(self, tmp_path, packed_glob):
        assert scan_settings(packed_config(tmp_path, packed_glob, answer_only_loss=False)).answer_only_loss is False
        default = FinetuningDatasetConfig(dataset_root=tmp_path, seq_length=64, seed=5)
        settings = scan_settings(default)
        assert (settings.seed, settings.eod_mask_loss, settings.answer_only_loss) == (5, None, True)


class TestPackedScan:
    def test_counts_and_listed_documents(self, tmp_path, packed_glob, tokenizer):
        sources, _ = training_data_sources(packed_config(tmp_path, packed_glob, answer_only_loss=True), tokenizer)
        [result] = packed_scan(sources)
        assert result.stop_reason == "exhausted"
        assert result.documents_scanned == CONVERSATIONS
        assert result.tokens_scanned == sum(len(pack_row(p)["input_ids"]) for p in SHARD_A + SHARD_B)
        assert (result.listed_targets, result.listed_trainable_targets) == (Q_TARGETS, Q_TRAINABLE_ANSWER_ONLY)
        assert len(result.listed_documents) == CONVERSATIONS_WITH_Q
        assert all(Q in d.token_ids for d in result.listed_documents)
        assert not any(d.first_token_is_target for d in result.listed_documents)
        [answer_only_off] = packed_scan(sources, answer_only_loss=False)
        assert answer_only_off.listed_trainable_targets == Q_TRAINABLE_ALL

    def test_random_documents_depend_on_the_seed(self, tmp_path, packed_glob, tokenizer):
        sources, _ = training_data_sources(packed_config(tmp_path, packed_glob, answer_only_loss=True), tokenizer)

        def references(seed: int) -> tuple[str, ...]:
            [result] = packed_scan(sources, documents_per_source=3, seed=seed)
            return tuple(d.reference for d in result.documents)

        assert len(set(references(1))) == 3
        assert references(1) == references(1)
        # The generator also depends on the source's path, a fresh temporary directory, and the three documents come
        # from the first one-row row groups read, so two given seeds can draw the same ones; twenty seeds all drawing
        # the same ones is vanishingly unlikely.
        assert len({references(seed) for seed in range(20)}) > 1

    def test_token_budget_and_deadline_stop_the_scan(self, tmp_path, packed_glob, tokenizer):
        sources, _ = training_data_sources(packed_config(tmp_path, packed_glob, answer_only_loss=True), tokenizer)
        [budget] = packed_scan(sources, documents_per_source=0, max_scan_tokens_per_source=1)
        assert budget.stop_reason == "token_budget"
        assert budget.documents_scanned < CONVERSATIONS
        [late] = packed_scan(sources, documents_per_source=2, deadline=time.monotonic() - 1)
        assert late.stop_reason == "time_budget"
        # The random documents come from the first row group read, which counts as scanned.
        assert len(late.documents) == 2 and 0 < late.documents_scanned < CONVERSATIONS

    def test_packed_scan_needs_the_collate_settings(self, tmp_path, packed_glob, tokenizer):
        sources, _ = training_data_sources(packed_config(tmp_path, packed_glob, answer_only_loss=True), tokenizer)
        with pytest.raises(ValueError, match="answer_only_loss and eos_token_id"):
            packed_scan(sources, eos_token_id=None)

    @pytest.mark.parametrize("answer_only_loss", [True, False])
    def test_trainable_targets_equal_what_the_training_collate_produces(
        self, tmp_path, packed_glob, tokenizer, answer_only_loss
    ):
        sources, _ = training_data_sources(
            packed_config(tmp_path, packed_glob, answer_only_loss=answer_only_loss), tokenizer
        )
        [result] = packed_scan(
            sources,
            answer_only_loss=answer_only_loss,
            documents_per_source=CONVERSATIONS,
            listed_documents_per_source=0,
        )
        scanned = {d.reference: d for d in result.documents}
        assert len(scanned) == CONVERSATIONS

        # The training dataset over the same glob reads shard0's rows, then shard1's.
        dataset = create_sft_dataset(
            packed_glob, tokenizer=tokenizer, seq_length=64, answer_only_loss=answer_only_loss
        )
        rows = [("shard0", row) for row in range(len(SHARD_A))] + [("shard1", row) for row in range(len(SHARD_B))]
        for index, (shard, row) in enumerate(rows):
            item = dataset[index]
            batch = dataset.collate_fn([item])
            labels, loss_mask = batch["labels"][0], batch["loss_mask"][0]
            boundaries = item["seq_boundaries"]
            for k in range(len(boundaries) - 1):
                start, end = boundaries[k], boundaries[k + 1]
                # The collate drops each conversation's last input, so conversation k starts k positions early.
                collated = slice(start - k, end - 1 - k)
                assert labels[collated].tolist() == item["input_ids"][start + 1 : end]
                document = scanned[f"{shard} row {row} conversation {k}"]
                assert document.token_ids.tolist() == item["input_ids"][start:end]
                assert document.trainable_target.tolist() == [False] + (loss_mask[collated] != 0).tolist()


def test_scanning_leaves_global_random_state_untouched(tmp_path, blend, packed_glob, tokenizer):
    np.random.seed(7)
    random.seed(7)
    torch.manual_seed(7)
    numpy_state, python_state, torch_state = np.random.get_state(), random.getstate(), torch.get_rng_state()
    indexed, _ = training_data_sources(gpt_config(blend, eod_mask_loss=False), None)
    packed, _ = training_data_sources(packed_config(tmp_path, packed_glob, answer_only_loss=True), tokenizer)
    scan(indexed)
    packed_scan(packed)
    after = np.random.get_state()
    assert after[0] == numpy_state[0] and np.array_equal(after[1], numpy_state[1]) and after[2:] == numpy_state[2:]
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)

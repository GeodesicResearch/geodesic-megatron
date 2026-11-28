# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The held_out_validation block, the steps a held-out set is evaluated at, and the sets setup builds.

The block is validated as ``ConfigContainer.validate`` runs it. The sets are real ``.bin/.idx`` corpora written with
Megatron's ``IndexedDatasetBuilder`` and built by ``BlendedMegatronDatasetBuilder`` on a single-process gloo world.
An evaluation needs a model on a GPU, so these tests stop at the data each evaluation reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset

from megatron.bridge.training.config import (
    ConfigContainer,
    GPTDatasetConfig,
    HeldOutSetConfig,
    HeldOutValidationConfig,
    MockGPTDatasetConfig,
)
from megatron.bridge.training.held_out_validation import (
    HELD_OUT_KEY_PREFIX,
    HeldOutSet,
    build_held_out_validation,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from tests.unit_tests.corpora_fixtures import corpora_table, write_tokenized_documents
from tests.unit_tests.tiny_run_fixtures import gpt_dataset_config, run_config
from tests.unit_tests.token_masking_fixtures import null_tokenizer_config


VOCAB_SIZE = 1024  # NullTokenizer(1024): its end-of-document id is 1023
EOD = VOCAB_SIZE - 1
SEQ = 16
GLOBAL_BATCH, MICRO_BATCH = 4, 2


def _block(**fields) -> HeldOutValidationConfig:
    sets = fields.pop("sets", [{"name": "unseen", "data_path": "/data/unseen", "samples": 8}])
    return HeldOutValidationConfig(sets=sets, **fields)


class TestConfig:
    def test_a_block_without_sets_is_off(self):
        block = HeldOutValidationConfig()
        block.finalize(GLOBAL_BATCH)
        assert block.sets == []

    def test_a_set_given_as_a_mapping_becomes_a_set_config(self):
        block = _block(interval=5, at_start=True, at_exit=True)
        block.finalize(GLOBAL_BATCH)
        assert block.sets == [HeldOutSetConfig(name="unseen", data_path="/data/unseen", samples=8)]

    def test_a_set_that_is_not_a_mapping_is_refused(self):
        with pytest.raises(ValueError, match="A HeldOutSetConfig must be given as a mapping, got 'unseen'"):
            _block(interval=5, sets=["unseen"]).finalize(GLOBAL_BATCH)

    def test_an_unknown_key_of_a_set_is_refused_with_a_suggestion(self):
        block = _block(interval=5, sets=[{"name": "unseen", "data_pth": "/data/unseen", "samples": 8}])
        with pytest.raises(ValueError, match="Unknown key 'data_pth' for HeldOutSetConfig. Did you mean 'data_path'?"):
            block.finalize(GLOBAL_BATCH)

    @pytest.mark.parametrize("field", ["interval", "at_start", "at_exit"])
    def test_a_schedule_without_sets_is_refused(self, field):
        block = HeldOutValidationConfig(**{field: 5 if field == "interval" else True})
        with pytest.raises(ValueError, match=f"held_out_validation sets {field} without sets"):
            block.finalize(GLOBAL_BATCH)

    def test_sets_evaluated_at_no_step_are_refused(self):
        with pytest.raises(ValueError, match="evaluates them at no step"):
            _block().finalize(GLOBAL_BATCH)

    @pytest.mark.parametrize(
        ("held", "message"),
        [
            ({"name": "Unseen", "data_path": "/d", "samples": 8}, r"sets\[0\].name must be lowercase"),
            ({"name": None, "data_path": "/d", "samples": 8}, r"sets\[0\].name must be lowercase"),
            ({"name": "unseen", "data_path": "", "samples": 8}, r"sets\[0\].data_path must be a non-empty"),
            ({"name": "unseen", "data_path": "/d", "samples": 0}, r"sets\[0\].samples must be a positive integer"),
            ({"name": "unseen", "data_path": "/d", "samples": True}, r"sets\[0\].samples must be a positive integer"),
            ({"name": "unseen", "data_path": "/d", "samples": 6}, "samples=6 is not a multiple of"),
        ],
    )
    def test_a_malformed_set_is_refused(self, held, message):
        with pytest.raises(ValueError, match=message):
            _block(interval=5, sets=[held]).finalize(GLOBAL_BATCH)

    def test_two_sets_of_one_name_are_refused(self):
        held = {"name": "unseen", "data_path": "/d", "samples": 8}
        with pytest.raises(ValueError, match="names the set 'unseen' more than once"):
            _block(interval=5, sets=[held, dict(held)]).finalize(GLOBAL_BATCH)

    @pytest.mark.parametrize("interval", [0, -1, True, 2.5])
    def test_an_interval_that_is_not_a_positive_integer_is_refused(self, interval):
        with pytest.raises(ValueError, match="interval must be a positive integer or null"):
            _block(interval=interval, at_start=True).finalize(GLOBAL_BATCH)

    def test_a_flag_that_is_not_a_boolean_is_refused(self):
        with pytest.raises(ValueError, match="at_exit must be true or false, got 'yes'"):
            _block(at_exit="yes").finalize(GLOBAL_BATCH)


class TestSchedule:
    @staticmethod
    def _set(interval: int | None, at_exit: bool) -> HeldOutSet:
        return HeldOutSet(
            key_prefix="held-out/s/", iters=1, interval=interval, at_start=False, at_exit=at_exit, new_loader=list
        )

    def test_due_at_every_multiple_of_the_interval(self):
        held = self._set(5, at_exit=False)
        assert [step for step in range(1, 16) if held.due(step, final=False)] == [5, 10, 15]
        assert not held.due(7, final=True)

    def test_due_at_the_final_step_when_at_exit(self):
        held = self._set(None, at_exit=True)
        assert [step for step in range(1, 16) if held.due(step, final=step == 13)] == [13]

    def test_a_final_step_on_the_interval_is_one_evaluation(self):
        held = self._set(5, at_exit=True)
        assert [step for step in range(1, 11) if held.due(step, final=step == 10)] == [5, 10]

    def test_a_restart_drops_the_sets(self):
        state = GlobalState()
        state.held_out_sets = [self._set(5, at_exit=False)]
        state.reset_for_restart()
        assert state.held_out_sets == []


def _corpus(root: Path, count: int, offset: int) -> str:
    documents = [[offset + (7 * d + j) % 300 for j in range(10)] + [EOD] for d in range(count)]
    write_tokenized_documents(root, documents)
    return str(root / corpora_table.TOKENIZED_PREFIX)


def _dataset_config(tmp_path: Path, data_path: list[str] | None, config_class=GPTDatasetConfig) -> GPTDatasetConfig:
    return gpt_dataset_config(
        tmp_path / "cache", data_path, seq_length=SEQ, vocab_size=VOCAB_SIZE, split="1,0,0", config_class=config_class
    )


def _run(tmp_path: Path, block: HeldOutValidationConfig, dataset: GPTDatasetConfig | None = None) -> ConfigContainer:
    block.finalize(GLOBAL_BATCH)
    return run_config(
        dataset or _dataset_config(tmp_path, [_corpus(tmp_path / "training", 30, 100)]),
        null_tokenizer_config(VOCAB_SIZE),
        global_batch=GLOBAL_BATCH,
        micro_batch=MICRO_BATCH,
        seq_length=SEQ,
        held_out_validation=block,
    )


def _read(held: HeldOutSet, samples: int) -> list[list[int]]:
    """The tokens of the first ``samples`` samples an evaluation reads, as micro-batches of one data-parallel rank."""
    iterator = held.data_iterator()
    return [sample for _ in range(samples // MICRO_BATCH) for sample in next(iterator)["tokens"].tolist()]


class TestBuild:
    def test_a_run_without_sets_builds_none(self, tmp_path, gloo_group_of_one):
        cfg = _run(tmp_path, HeldOutValidationConfig())
        assert build_held_out_validation(cfg, build_tokenizer(cfg.tokenizer), gloo_group_of_one) == []

    def test_each_set_is_megatrons_gpt_dataset_over_its_prefix_read_from_its_first_sample(
        self, tmp_path, gloo_group_of_one
    ):
        first = _corpus(tmp_path / "first", 20, 400)
        second = _corpus(tmp_path / "second", 20, 700)
        block = HeldOutValidationConfig(
            sets=[
                {"name": "first", "data_path": first, "samples": 8},
                {"name": "second", "data_path": second, "samples": 4},
            ],
            interval=3,
            at_start=True,
            at_exit=True,
        )
        cfg = _run(tmp_path, block)
        sets = build_held_out_validation(cfg, build_tokenizer(cfg.tokenizer), gloo_group_of_one)
        assert [(s.key_prefix, s.iters, s.interval, s.at_start, s.at_exit) for s in sets] == [
            (f"{HELD_OUT_KEY_PREFIX}first/", 2, 3, True, True),
            (f"{HELD_OUT_KEY_PREFIX}second/", 1, 3, True, True),
        ]
        for held, prefix, samples in zip(sets, (first, second), (8, 4)):
            dataset = held.new_loader().dataset
            assert isinstance(dataset, GPTDataset) and dataset.num_samples == samples
            read = _read(held, samples)
            assert _read(held, samples) == read
            # The same samples as a GPTDataset built by hand over the prefix, as a pretraining run builds its data.
            by_hand = _dataset_config(tmp_path, [prefix])
            [expected, _, _] = BlendedMegatronDatasetBuilder(
                GPTDataset, [samples, 0, 0], lambda: True, by_hand
            ).build()
            assert read == [expected[i]["tokens"].tolist() for i in range(samples)]

    def test_a_set_needs_a_run_on_a_corpus(self, tmp_path, gloo_group_of_one):
        block = HeldOutValidationConfig(
            sets=[{"name": "unseen", "data_path": _corpus(tmp_path / "held", 10, 400), "samples": 4}], interval=2
        )
        cfg = _run(tmp_path, block, dataset=_dataset_config(tmp_path, None, config_class=MockGPTDatasetConfig))
        with pytest.raises(ValueError, match="held_out_validation set 'unseen' names a .bin/.idx set, but the run"):
            build_held_out_validation(cfg, build_tokenizer(cfg.tokenizer), gloo_group_of_one)

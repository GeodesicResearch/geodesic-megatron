# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The held-out sets of a run: that an evaluation reads every pack of a set once, and when and how it runs.

The sets are real packed parquet files under ``tmp_path``, read through the real fine-tuning dataset builder, sampler
and loader, with a real tokenizer and the real ``ConfigContainer``. The one stand-in is the evaluation step itself
(``evaluate_and_print_results``), which needs a model and a GPU: its test records the call that a due evaluation makes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from megatron.bridge.training.config import FinetuningDatasetConfig
from megatron.bridge.training.held_out import evaluation as evaluation_module
from megatron.bridge.training.held_out.config import HeldOutConfig, HeldOutError
from megatron.bridge.training.held_out.evaluation import (
    HELD_OUT_KEY_PREFIX,
    HeldOutEvaluation,
    build_held_out_evaluations,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from tests.unit_tests.packed_run_fixtures import (
    GLOBAL_BATCH,
    PACK_SIZE,
    packed_dataset_config,
    run_config,
    write_packs,
)
from tests.unit_tests.token_masking_fixtures import build_tiny_hf_tokenizer, hf_tokenizer_config


# One conversation per pack. The stored mask is shifted by one (entry i gates the prediction of token i + 1), so each
# pack has four tokens with loss. The first token names the pack: 10 + its number in its set.
CONVERSATION = [3, 4, 5, 6, 5, 3, 4]
STORED_MASK = [0, 0, 1, 1, 1, 1, 0]
LOSS_TOKENS_PER_PACK = 4
FIRST_TOKEN = 10


def _rows(count: int) -> list[dict]:
    return [
        {"input_ids": [FIRST_TOKEN + d] + CONVERSATION[1:], "loss_mask": STORED_MASK, "seq_start_id": [0]}
        for d in range(count)
    ]


def _run(tmp_path: Path, packs_of_set: dict[str, int], interval: int = 1):
    """A packed SFT run that holds out the named sets, ``packs_of_set`` giving the packs in each."""
    tokenizer = hf_tokenizer_config(build_tiny_hf_tokenizer(tmp_path / "tokenizer"))
    training = write_packs(tmp_path / "training" / "training_64.idx.parquet", _rows(8))
    sets = {
        name: write_packs(tmp_path / name / f"{name}_64.idx.parquet", _rows(count))
        for name, count in packs_of_set.items()
    }
    return run_config(
        packed_dataset_config(tmp_path, training), tokenizer, held_out=HeldOutConfig(sets=sets, interval=interval)
    )


def _build(cfg, dp_group) -> list[HeldOutEvaluation]:
    return build_held_out_evaluations(cfg, build_tokenizer(cfg.tokenizer), dp_group)


def _batches(evaluation: HeldOutEvaluation) -> list[dict]:
    """The batches one evaluation reads: its iterator drawn ``iters`` times, as ``evaluate`` draws it (a
    ``RerunDataIterator`` has ``__next__`` and no ``__iter__``)."""
    iterator = evaluation.data_iterator()
    return [next(iterator) for _ in range(evaluation.iters)]


class TestBuild:
    def test_a_run_without_held_out_sets_builds_none(self, tmp_path, gloo_group_of_one):
        cfg = _run(tmp_path, {})
        assert _build(cfg, gloo_group_of_one) == []

    def test_each_set_gets_its_own_prefix_and_as_many_batches_as_hold_its_packs(self, tmp_path, gloo_group_of_one):
        cfg = _run(tmp_path, {"plain": 5, "wise": 3, "exact": 2 * GLOBAL_BATCH, "single": 1}, interval=3)
        evaluations = _build(cfg, gloo_group_of_one)
        assert [e.key_prefix for e in evaluations] == [
            f"{HELD_OUT_KEY_PREFIX}{name}/" for name in ("plain", "wise", "exact", "single")
        ]
        assert [e.iters for e in evaluations] == [2, 1, 2, 1]
        assert {e.interval for e in evaluations} == {3}

    def test_the_last_batch_is_padded_so_every_pack_is_read_once(self, tmp_path, gloo_group_of_one):
        """Five packs and a global batch of four: the sampler's second batch holds pack 4 and three pads (-1)."""
        [evaluation] = _build(_run(tmp_path, {"plain": 5}), gloo_group_of_one)
        assert list(evaluation.new_loader().batch_sampler) == [[0, 1, 2, 3], [4, -1, -1, -1]]

    def test_a_set_of_whole_batches_is_not_padded(self, tmp_path, gloo_group_of_one):
        [evaluation] = _build(_run(tmp_path, {"exact": 2 * GLOBAL_BATCH}), gloo_group_of_one)
        assert list(evaluation.new_loader().batch_sampler) == [[0, 1, 2, 3], [4, 5, 6, 7]]

    def test_the_padded_samples_carry_no_loss_so_the_loss_tokens_are_the_sets_alone(self, tmp_path, gloo_group_of_one):
        [evaluation] = _build(_run(tmp_path, {"plain": 5}), gloo_group_of_one)
        batches = _batches(evaluation)
        assert all(len(batch["tokens"]) == GLOBAL_BATCH for batch in batches)
        samples = [
            (tokens[0], int(mask.sum()))
            for batch in batches
            for tokens, mask in zip(batch["tokens"], batch["loss_mask"])
        ]
        read = [first for first, loss in samples if loss > 0]
        assert sorted(read) == [FIRST_TOKEN + d for d in range(5)], "every pack once, none twice"
        assert sum(loss for _, loss in samples) == 5 * LOSS_TOKENS_PER_PACK
        assert len(samples) - len(read) == 3, "three pads"

    def test_every_evaluation_reads_the_set_from_its_first_pack(self, tmp_path, gloo_group_of_one):
        [evaluation] = _build(_run(tmp_path, {"plain": 5}), gloo_group_of_one)
        first = [batch["tokens"].tolist() for batch in _batches(evaluation)]
        assert [batch["tokens"].tolist() for batch in _batches(evaluation)] == first

    def test_each_set_is_read_from_its_own_file(self, tmp_path, gloo_group_of_one):
        plain, wise = _build(_run(tmp_path, {"plain": 5, "wise": 3}), gloo_group_of_one)
        assert len(plain.new_loader().dataset) == 5
        assert len(wise.new_loader().dataset) == 3

    def test_a_loader_type_that_cannot_pad_is_refused(self, tmp_path, gloo_group_of_one):
        cfg = _run(tmp_path, {"plain": 5})
        cfg.dataset.dataloader_type = "single"
        with pytest.raises(HeldOutError, match="cannot pad to whole global batches"):
            _build(cfg, gloo_group_of_one)

    def test_a_run_on_unpacked_data_is_refused_naming_the_set(self, tmp_path, gloo_group_of_one):
        cfg = _run(tmp_path, {"plain": 5})
        cfg.dataset = FinetuningDatasetConfig(dataset_root=tmp_path, seq_length=PACK_SIZE)
        with pytest.raises(HeldOutError, match="held_out.sets.plain: the run does not train on packed sequences"):
            _build(cfg, gloo_group_of_one)

    def test_a_set_without_packs_is_refused_naming_the_set(self, tmp_path, gloo_group_of_one):
        cfg = _run(tmp_path, {"plain": 5})
        cfg.held_out.sets["plain"] = write_packs(tmp_path / "empty" / "empty_64.idx.parquet", [])
        with pytest.raises(HeldOutError, match=r"held_out.sets.plain='.*empty_64.idx.parquet': .*empty"):
            _build(cfg, gloo_group_of_one)

    def test_a_set_that_is_not_on_disk_is_refused_naming_the_set(self, tmp_path, gloo_group_of_one):
        cfg = _run(tmp_path, {"plain": 5})
        missing = str(tmp_path / "nowhere" / "plain_64.idx.parquet")
        cfg.held_out.sets["plain"] = missing
        with pytest.raises(HeldOutError, match=f"held_out.sets.plain: packed_data_path='{missing}': "):
            _build(cfg, gloo_group_of_one)


class TestEvaluation:
    def test_due_before_the_first_step_and_at_every_multiple_of_the_interval(self):
        evaluation = HeldOutEvaluation(key_prefix="held-out/a/", interval=5, iters=1, new_loader=list)
        assert [step for step in range(16) if evaluation.due(step)] == [0, 5, 10, 15]

    def test_an_evaluation_logs_under_its_prefix_without_advancing_the_validation_sampler(
        self, tmp_path, gloo_group_of_one, monkeypatch
    ):
        calls = []
        # A model forward needs a GPU and a distributed world, so the evaluation step is recorded, not run.
        monkeypatch.setattr(evaluation_module, "evaluate_and_print_results", lambda *a, **k: calls.append((a, k)))
        [evaluation] = _build(_run(tmp_path, {"plain": 5}), gloo_group_of_one)
        state = GlobalState()
        state.train_state.step = 7
        evaluation.evaluate(state, "forward_step", ["model"], "model_config")
        [(args, kwargs)] = calls
        assert (
            args[:2] == (state, "iteration 7")
            and args[2] == "forward_step"
            and args[4:] == (["model"], "model_config")
        )
        assert (kwargs["eval_iters"], kwargs["key_prefix"]) == (2, "held-out/plain/")
        assert kwargs["advance_consumed_valid_samples"] is False

    def test_a_new_run_state_holds_no_evaluation_and_a_restart_drops_them(self):
        state = GlobalState()
        assert state.held_out_evaluations == []
        state.held_out_evaluations = [
            HeldOutEvaluation(key_prefix="held-out/a/", interval=1, iters=1, new_loader=list)
        ]
        state.reset_for_restart()
        assert state.held_out_evaluations == []

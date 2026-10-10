# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""A run's predicted token-masking counts (scripts/data/predict_masked_counts.py) against its data.

The corpora are real ``.bin/.idx`` files from Megatron's own writer, the run is the real Nano pretrain recipe with
mcore's NullTokenizer, and the dataset, loaders and masking are the training code's own, so nothing is stubbed. The
oracle is the built dataset itself: a run that reads every sample of its dataset once must count, over its
iterations, exactly the marker targets its samples' labels hold, read sample by sample without any sampler.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from tests.unit_tests.corpora_fixtures import importable, write_tokenized_documents


REPO_ROOT = Path(__file__).resolve().parents[3]
importable(REPO_ROOT / "configs" / "control_pretraining")
import corpora_table  # noqa: E402


SEQ_LENGTH = 8
VOCAB_SIZE = 1000
MARKER = 7
# Each iteration reads 4 samples of 8 targets; 6 iterations read the 24 samples the blend builds.
GLOBAL_BATCH = 4
TRAIN_ITERS = 6


@pytest.fixture(scope="module")
def tool():
    """scripts/data/predict_masked_counts.py, loaded by path: scripts/ is not a package."""
    path = REPO_ROOT / "scripts" / "data" / "predict_masked_counts.py"
    spec = importlib.util.spec_from_file_location("predict_masked_counts", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["predict_masked_counts"] = module
    spec.loader.exec_module(module)
    return module


def _corpus(root: Path, first_value: int, documents: int) -> str:
    """A corpus of ``documents`` documents, document i holding ``first_value + i`` with the marker every third
    position from its i-th, so the marker's targets fall at every offset of a sample."""
    write_tokenized_documents(
        root,
        [[MARKER if (j + i) % 3 == 0 else first_value + i for j in range(9 + i % 4)] for i in range(documents)],
    )
    return str(root / corpora_table.TOKENIZED_PREFIX)


def _config(
    tmp_path: Path, token_masking: dict | None, model: dict | None = None, train: dict | None = None, **checkpoint
) -> Path:
    """A pretrain override YAML over the real Nano recipe that needs no network: NullTokenizer, no model parallelism
    unless ``model`` states some, and ``train`` merged into the run's budget."""
    config = {
        "tokenizer": {"tokenizer_type": "NullTokenizer", "vocab_size": VOCAB_SIZE},
        "dataset": {
            "data_path": ["0.5", _corpus(tmp_path / "a", 100, 14), "0.5", _corpus(tmp_path / "b", 200, 14)],
            "seq_length": SEQ_LENGTH,
            "split": "1,0,0",
            "path_to_cache": str(tmp_path / "cache"),
        },
        "model": {
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
            **(model or {}),
        },
        "train": {
            "train_iters": TRAIN_ITERS,
            "global_batch_size": GLOBAL_BATCH,
            "micro_batch_size": 1,
            **(train or {}),
        },
        "checkpoint": checkpoint,
    }
    if token_masking is not None:
        config["token_masking"] = token_masking
    path = tmp_path / "run.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config))
    return path


MASKING = {"enabled": True, "token_ids": [MARKER]}


def _predicted(tool, config: Path, gpus: int, first: int = 1, last: int = TRAIN_ITERS) -> dict:
    cfg = tool.resolve_run_config(str(config), "nano", "pretrain")
    return dict(tool.predict(cfg, gpus, first, last))


def _marker_targets_in_the_dataset(tool, config: Path) -> int:
    """The marker targets the run's whole dataset holds, read sample by sample."""
    dataset, size, first_sample = tool.build_training_dataset(tool.resolve_run_config(str(config), "nano", "pretrain"))
    assert (first_sample, len(dataset)) == (0, size), "the oracle needs a run that reads its whole dataset"
    return sum(int((dataset[index]["labels"] == MARKER).sum()) for index in range(len(dataset)))


class TestAgainstTheData:
    def test_a_masked_run_counts_every_marker_target_of_its_dataset_once_and_trains_none(self, tool, tmp_path):
        config = _config(tmp_path, MASKING)
        predicted = _predicted(tool, config, gpus=2)
        assert sorted(predicted) == list(range(1, TRAIN_ITERS + 1))
        assert sum(counts.listed for counts in predicted.values()) == _marker_targets_in_the_dataset(tool, config)
        for counts in predicted.values():
            assert counts.positions == GLOBAL_BATCH * SEQ_LENGTH
            # No position is masked by the dataset itself (no EOD or padding masking), so every marker target trains
            # until masking removes it.
            assert counts.masked == counts.listed_trainable == counts.listed
            assert counts.trained_listed == 0
            assert counts.trainable == counts.positions - counts.masked

    def test_a_control_measures_the_marker_and_masks_nothing(self, tool, tmp_path):
        config = _config(tmp_path, {"masked_validation": {"token_ids": [MARKER]}})
        predicted = _predicted(tool, config, gpus=2)
        assert sum(counts.listed for counts in predicted.values()) == _marker_targets_in_the_dataset(tool, config)
        for counts in predicted.values():
            assert (counts.masked, counts.trainable) == (0, counts.positions)
            assert counts.trained_listed == counts.listed_trainable == counts.listed

    @pytest.mark.parametrize("gpus", [1, 2, 4])
    def test_every_data_parallel_width_reads_the_whole_dataset_once(self, tool, tmp_path, gpus):
        """How the replicas share an iteration moves samples between iterations, never in or out of the run."""
        config = _config(tmp_path, MASKING)
        predicted = _predicted(tool, config, gpus=gpus)
        assert sum(counts.listed for counts in predicted.values()) == _marker_targets_in_the_dataset(tool, config)


class TestIterationRanges:
    def test_a_range_starting_partway_predicts_what_the_whole_run_predicts_there(self, tool, tmp_path):
        config = _config(tmp_path, MASKING)
        whole = _predicted(tool, config, gpus=2)
        assert _predicted(tool, config, gpus=2, first=3, last=5) == {k: whole[k] for k in (3, 4, 5)}

    def test_a_resumed_run_reads_on_from_the_samples_its_steps_consumed(self, tool, tmp_path):
        whole = _predicted(tool, _config(tmp_path / "whole", MASKING), gpus=2)
        resumed = _predicted(tool, _config(tmp_path / "resumed", MASKING, ckpt_step=2), gpus=2, first=3)
        assert resumed == {k: whole[k] for k in range(3, TRAIN_ITERS + 1)}

    def test_a_resumed_run_that_resets_its_data_position_reads_a_fresh_dataset_from_its_start(self, tool, tmp_path):
        config = _config(tmp_path, MASKING, ckpt_step=2, reset_data_position=True)
        predicted = _predicted(tool, config, gpus=2, first=3)
        assert sum(counts.listed for counts in predicted.values()) == _marker_targets_in_the_dataset(tool, config)

    @pytest.mark.parametrize(("first", "last"), [(0, 2), (1, TRAIN_ITERS + 1), (4, 3)])
    def test_iterations_outside_the_run_are_refused(self, tool, tmp_path, first, last):
        with pytest.raises(ValueError, match="are not inside the run's iterations 1..6"):
            _predicted(tool, _config(tmp_path, MASKING), gpus=2, first=first, last=last)

    def test_iterations_a_resumed_run_does_not_train_are_refused(self, tool, tmp_path):
        with pytest.raises(ValueError, match="are not inside the run's iterations 3..6"):
            _predicted(tool, _config(tmp_path, MASKING, ckpt_step=2), gpus=2, first=2)


class TestRefusals:
    def test_a_config_measuring_no_ids_is_refused(self, tool, tmp_path):
        with pytest.raises(ValueError, match="measures no token ids"):
            _predicted(tool, _config(tmp_path, None), gpus=2)

    def test_a_global_batch_the_replicas_cannot_share_is_refused(self, tool, tmp_path):
        """Megatron's own microbatch calculator refuses it, as it would refuse the run."""
        with pytest.raises(AssertionError, match=r"global batch size \(4\) is not divisible"):
            _predicted(tool, _config(tmp_path, MASKING), gpus=3)

    def test_gpus_the_model_parallel_groups_cannot_fill_are_refused(self, tool, tmp_path):
        """The config's own data-parallel width refuses 3 GPUs for groups of 2 context-parallel ranks."""
        with pytest.raises(AssertionError, match=r"world size \(3\) is not divisible"):
            _predicted(tool, _config(tmp_path, MASKING, model={"context_parallel_size": 2}), gpus=3)

    def test_a_run_that_decreases_its_batch_is_refused(self, tool, tmp_path):
        config = _config(tmp_path, MASKING, train={"decrease_batch_size_if_needed": True})
        with pytest.raises(ValueError, match="decrease_batch_size_if_needed"):
            _predicted(tool, config, gpus=2)


class TestMain:
    def _main(self, tool, config: Path, out: Path) -> int:
        return tool.main(
            [str(config), "--model", "nano", "--mode", "pretrain", "--gpus", "2", "--iterations", "2", "4"]
            + ["--out", str(out)]
        )

    def test_it_writes_its_inputs_then_one_line_per_iteration(self, tool, tmp_path):
        config = _config(tmp_path, MASKING)
        out = tmp_path / "prediction" / "counts.jsonl"
        assert self._main(tool, config, out) == 0
        header, *lines = [json.loads(line) for line in out.read_text().splitlines()]
        inputs = header["inputs"]
        assert (inputs["gpus"], inputs["data_parallel_size"], inputs["iterations"]) == (2, 2, [2, 4])
        assert inputs["token_masking"] == {"enabled": True, "measured": [MARKER]}
        assert inputs["run"]["global_batch_size"] == GLOBAL_BATCH
        expected = _predicted(tool, config, gpus=2, first=2, last=4)
        assert [line.pop("iteration") for line in lines] == [2, 3, 4]
        assert lines == [asdict(expected[k]) for k in (2, 3, 4)]

    def test_an_existing_output_is_refused(self, tool, tmp_path):
        out = tmp_path / "counts.jsonl"
        out.write_text("earlier\n")
        with pytest.raises(FileExistsError, match="never overwritten"):
            self._main(tool, _config(tmp_path, MASKING), out)
        assert out.read_text() == "earlier\n"

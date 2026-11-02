# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for scripts/telemetry/predicted_counts.py (a run's predicted token-masking counts, read back).

The predictions are written by the predictor's own writer, ``predict_masked_counts.write_prediction``, from the
bridge's ``TokenMaskingCounts``, so the reader is tested on the format the predictor produces.
"""

import dataclasses
import json
from pathlib import Path

import pytest
from scripts.data.predict_masked_counts import write_prediction
from scripts.telemetry import predicted_counts
from scripts.telemetry.training_log import TokenMaskingCountsRecord

from megatron.bridge.training.token_masking.hook import TokenMaskingCounts


INPUTS = {"config": "stage.yaml", "config_sha256": "0" * 64, "gpus": 64, "code_revision": "abc"}


def counts(iteration: int, positions: int = 1000) -> TokenMaskingCounts:
    listed = 100 + iteration
    return TokenMaskingCounts(
        listed=listed,
        listed_trainable=listed,
        masked=listed,
        trained_listed=0,
        trainable=positions - listed,
        positions=positions,
    )


def write_range(directory: Path, first: int, last: int, inputs: dict = INPUTS, positions: int = 1000) -> Path:
    path = directory / f"counts_{first}_{last}.jsonl"
    write_prediction(
        path, {**inputs, "iterations": [first, last]}, ((i, counts(i, positions)) for i in range(first, last + 1))
    )
    return path


@pytest.mark.parametrize(
    ("lines", "match"),
    [
        ([], "does not open with a prediction's inputs record"),
        ([{"iteration": 1}], "does not open with a prediction's inputs record"),
        ([{"inputs": {}}, {"iteration": 1, "listed": 1}], "not an iteration's counts"),
        (
            [{"inputs": {}}]
            + [{"iteration": 1, "listed": 1, "listed_trainable": 1, "masked": 0, "trained_listed": 1,
                "trainable": 9, "positions": 9}] * 2,
            "predicts iteration 1 twice",
        ),
        (
            [{"inputs": {}}, {"iteration": 1, "listed": -1, "listed_trainable": 1, "masked": 0, "trained_listed": 1,
                             "trainable": 9, "positions": 9}],
            "not a count",
        ),
    ],
    ids=["empty", "no inputs", "missing counts", "an iteration twice", "a negative count"],
)  # fmt: skip
def test_a_file_that_is_not_a_prediction_is_refused(tmp_path, lines, match):
    path = tmp_path / "counts.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    with pytest.raises(ValueError, match=match):
        predicted_counts.read_prediction(path)


def test_a_directory_of_ranges_is_read_as_one_prediction(tmp_path):
    """A long run is predicted in ranges, one file each; the inputs they share are the prediction's, less the range."""
    write_range(tmp_path, 1, 3)
    write_range(tmp_path, 4, 6)
    write_range(tmp_path, 6, 7)  # a range may overlap another when the counts agree
    inputs, predicted = predicted_counts.read_predictions(tmp_path)
    assert inputs == INPUTS and sorted(predicted) == list(range(1, 8))
    assert predicted[6]["listed"] == 106


def test_two_files_predicting_one_iteration_differently_are_refused(tmp_path):
    write_range(tmp_path, 1, 3)
    write_range(tmp_path, 3, 4, positions=2000)
    with pytest.raises(ValueError, match="predicts other counts for iteration 3"):
        predicted_counts.read_predictions(tmp_path)


def test_a_directory_without_a_prediction_is_refused(tmp_path):
    with pytest.raises(ValueError, match="holds no prediction"):
        predicted_counts.read_predictions(tmp_path)


def logged_record(iteration: int, **edits: int) -> TokenMaskingCountsRecord:
    return TokenMaskingCountsRecord(iteration, **{**dataclasses.asdict(counts(iteration)), **edits})


def test_the_counts_that_differ_are_named_by_iteration_and_field_in_order(tmp_path):
    _, predicted = predicted_counts.read_prediction(write_range(tmp_path, 1, 5))
    logged = {
        1: logged_record(1),
        2: logged_record(2, masked=0, trainable=7),
        4: logged_record(4, positions=1001),
    }
    assert predicted_counts.differing_counts(logged, predicted) == [
        (2, "masked", 0, 102),
        (2, "trainable", 7, 898),
        (4, "positions", 1001, 1000),
    ]

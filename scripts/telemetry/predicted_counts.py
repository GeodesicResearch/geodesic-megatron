# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The token-masking counts ``scripts/data/predict_masked_counts.py`` predicts for a run, as a gate or a watch reads
them, and what makes a logged count differ from its prediction.

A prediction is JSON lines: an ``inputs`` record, then one record per iteration with its six counts. A long run is
predicted in ranges of iterations, one file each, and ``read_predictions`` reads a directory of them as one
prediction. The predictor needs the training stack; reading what it wrote needs only the standard library, so a tool
that runs without it (``score_gate.py``, ``run_watch.py``) reads a prediction through this module.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from scripts.telemetry.training_log import TOKEN_MASKING_COUNT_FIELDS, TokenMaskingCountsRecord  # noqa: E402


# The inputs in which two files of one prediction may differ: the range of iterations each covers.
RANGE_INPUTS = frozenset({"iterations"})


def as_count(value: Any, where: str) -> int:
    """``value`` as a count; raises ValueError naming ``where`` unless it is a non-negative integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where} is {value!r}, not a count")
    return value


def read_prediction(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, int]]]:
    """A prediction's ``inputs`` record and its counts by iteration, as ``predict_masked_counts.py`` writes them (JSON
    lines: the inputs, then one record per iteration). Raises ValueError on a file that is not one."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or set(records[0]) != {"inputs"} or not isinstance(records[0]["inputs"], dict):
        raise ValueError(f"{path.name} does not open with a prediction's inputs record")
    counts: dict[int, dict[str, int]] = {}
    for record in records[1:]:
        if set(record) != {"iteration", *TOKEN_MASKING_COUNT_FIELDS}:
            raise ValueError(f"{path.name} holds a record that is not an iteration's counts: {sorted(record)}")
        iteration = as_count(record["iteration"], f"{path.name}: iteration")
        if iteration in counts:
            raise ValueError(f"{path.name} predicts iteration {iteration} twice")
        counts[iteration] = {
            field: as_count(record[field], f"{path.name}: iteration {iteration} {field}")
            for field in TOKEN_MASKING_COUNT_FIELDS
        }
    return records[0]["inputs"], counts


def read_predictions(directory: Path) -> tuple[dict[str, Any], dict[int, dict[str, int]]]:
    """Every ``*.jsonl`` prediction of ``directory`` as one prediction: the inputs its files share and the counts by
    iteration.

    Raises ValueError when the directory holds no prediction, when two files were predicted from different inputs
    (any but the range of iterations each covers: another config, width, masking or code would predict another run),
    and when two files predict one iteration with different counts.
    """
    paths = sorted(directory.glob("*.jsonl"))
    if not paths:
        raise ValueError(f"{directory} holds no prediction (*.jsonl)")
    shared: dict[str, Any] | None = None
    counts: dict[int, dict[str, int]] = {}
    for path in paths:
        inputs, predicted = read_prediction(path)
        common = {key: value for key, value in inputs.items() if key not in RANGE_INPUTS}
        if shared is None:
            shared = common
        elif common != shared:
            differing = sorted(key for key in set(common) | set(shared) if common.get(key) != shared.get(key))
            raise ValueError(f"{path.name} was predicted from other inputs than {paths[0].name}: {differing}")
        for iteration, iteration_counts in predicted.items():
            if counts.setdefault(iteration, iteration_counts) != iteration_counts:
                raise ValueError(f"{path.name} predicts other counts for iteration {iteration} than another file")
    return shared or {}, counts


def differing_counts(
    logged: Mapping[int, TokenMaskingCountsRecord],
    predicted: Mapping[int, Mapping[str, int]],
    iterations: Iterable[int],
) -> list[tuple[int, str, int, int]]:
    """(iteration, field, logged, predicted) for every count of ``iterations`` that is not the predicted one, in
    iteration order and, within an iteration, in the order of ``TOKEN_MASKING_COUNT_FIELDS``.

    An iteration that is not both logged and predicted contributes nothing: what its absence means is the caller's.
    """
    return [
        (iteration, field, getattr(logged[iteration], field), predicted[iteration][field])
        for iteration in iterations
        if iteration in logged and iteration in predicted
        for field in TOKEN_MASKING_COUNT_FIELDS
        if getattr(logged[iteration], field) != predicted[iteration][field]
    ]


def describe_count_difference(field: str, logged: int, predicted: int) -> str:
    """One differing count as a gate or a stop line states it."""
    return f"{field} {logged} against the predicted {predicted}"

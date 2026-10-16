# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Unit tests for scripts/data/token_runs.py."""

import numpy as np
from scripts.data.token_runs import longest_runs


def brute_longest(row) -> int:
    best = current = 0
    for value in row:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def test_longest_runs_matches_a_direct_count():
    rng = np.random.default_rng(0)
    mask = rng.random((50, 37)) < 0.7
    mask[0] = True
    mask[1] = False
    expected = [brute_longest(row) for row in mask]
    assert longest_runs(mask).tolist() == expected
    assert expected[0] == 37 and expected[1] == 0


def test_a_run_never_continues_across_a_row_boundary():
    mask = np.array([[False, True, True], [True, True, False], [True, True, True]])
    assert longest_runs(mask).tolist() == [2, 2, 3]


def test_a_single_stream_is_one_row():
    stream = np.array([True, False, True, True, True, False, True])
    assert longest_runs(stream[None, :]).tolist() == [3]


def test_rows_with_no_columns_or_no_rows_hold_no_runs():
    assert longest_runs(np.zeros((3, 0), dtype=bool)).tolist() == [0, 0, 0]
    assert longest_runs(np.zeros((0, 5), dtype=bool)).tolist() == []

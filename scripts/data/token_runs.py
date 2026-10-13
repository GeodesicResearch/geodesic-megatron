# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Runs of one token id in rows of a token mask."""

from __future__ import annotations

import numpy as np


def longest_runs(mask: np.ndarray) -> np.ndarray:
    """The length of the longest run of True in each row of a 2-D boolean array (0 for a row with none).

    A single stream is one row: ``longest_runs(mask[None, :])[0]``. The working memory is one byte per element and
    one index per run edge, so a stream of tens of millions of ids costs tens of megabytes.
    """
    rows, columns = mask.shape
    padded = np.zeros((rows, columns + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    # A row's edges alternate: a run's first index, then one past its last. Rows come out in order, and every row
    # holds as many starts as ends, so the starts and ends of all rows pair up across the flattened result.
    edge_rows, edge_columns = np.nonzero(np.diff(padded, axis=1))
    longest = np.zeros(rows, dtype=np.int64)
    np.maximum.at(longest, edge_rows[0::2], edge_columns[1::2] - edge_columns[0::2])
    return longest

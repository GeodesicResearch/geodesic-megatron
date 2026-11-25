# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""A packed row for tests of packed parquet files, to write with ``write_packed_parquet`` of the library.

Catalogue:
    packed_row: one packed row of ``n_tokens`` tokens in ``n_seqs`` sequences.
"""

from __future__ import annotations


def packed_row(n_tokens: int = 64, n_seqs: int = 2, first_token: int = 1) -> dict:
    """One packed row with valid invariants: tokens ``first_token`` to ``first_token + n_tokens - 1``, a loss mask
    that counts every token, and ``n_seqs`` sequences of equal length, each starting at its multiple of that length."""
    assert n_seqs >= 1
    tokens_per_seq = n_tokens // n_seqs
    return {
        "input_ids": list(range(first_token, first_token + n_tokens)),
        "loss_mask": [1] * n_tokens,
        "seq_start_id": [i * tokens_per_seq for i in range(n_seqs)],
    }

# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The packer's seed orders its shuffle: the same seed gives the same packs, whatever numpy's global generator holds."""

import collections
import json

import numpy as np
import pytest

from megatron.bridge.data.datasets.packing_utils import create_packing_strategy, fill_packing_strategy
from tests.unit_tests.packer_fixtures import (
    ROWS,
    SEQ_LENGTH,
    packed_path,
    row_line,
    run_packer,
    shard_root,
    stored_rows,
    tokenizer_dir,  # noqa: F401  (a fixture)
)


HISTOGRAM = [0, 3, 5, 2, 4, 6, 1, 3, 2]  # 26 sequences of lengths 1 to 8
LINES = [row_line(i) for i in range(ROWS)]


def bins(seed: int) -> list[list[int]]:
    return create_packing_strategy(HISTOGRAM, 12, "first_fit_shuffle", np.random.default_rng(seed))[0]


def test_first_fit_shuffle_gives_the_same_bins_for_a_seed_and_other_bins_for_other_seeds():
    assert bins(3) == bins(3)
    assert len({json.dumps(bins(seed)) for seed in range(6)}) > 1


def test_a_name_that_is_not_a_packing_algorithm_is_refused():
    with pytest.raises(ValueError, match="packing_algorithm must be one of"):
        create_packing_strategy(HISTOGRAM, 12, "first_fit", np.random.default_rng(0))


def filled(seed: int) -> list[dict]:
    """Three packs of two sequences each, all of length 2, drawn from six distinct sequences."""
    sequences = collections.defaultdict(list)
    sequences[2] = [{"input_ids": [i, i, i], "loss_mask": [True, True, True]} for i in range(6)]
    return fill_packing_strategy([[2, 2], [2, 2], [2, 2]], sequences, 4, 0, np.random.default_rng(seed))


def test_fill_packing_strategy_orders_the_sequences_of_a_length_by_its_generator():
    assert filled(5) == filled(5)
    assert len({json.dumps(filled(seed)) for seed in range(6)}) > 1


def packs(base, tokenizer_dir, seed: int, global_seed: int) -> list[dict]:
    """The packs the packer writes for ``LINES`` with ``--seed seed`` after numpy's global generator is seeded with
    ``global_seed``."""
    np.random.seed(global_seed)
    root = shard_root(base, 0, LINES)
    run_packer(
        tokenizer_dir,
        root,
        "--packing-algorithm",
        "first_fit_shuffle",
        "--no-validation",
        seq_length=SEQ_LENGTH,
        seed=seed,
    )
    return stored_rows(packed_path(root, tokenizer_dir, SEQ_LENGTH))


def test_the_packers_seed_and_not_numpys_global_generator_orders_the_packs(tmp_path, tokenizer_dir):
    assert packs(tmp_path / "a", tokenizer_dir, 5, 0) == packs(tmp_path / "b", tokenizer_dir, 5, 1)


def test_other_seeds_give_other_packs(tmp_path, tokenizer_dir):
    built = [packs(tmp_path / str(seed), tokenizer_dir, seed, 0) for seed in range(4)]
    assert len({json.dumps(rows) for rows in built}) > 1

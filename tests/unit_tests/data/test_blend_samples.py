# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""scripts/data/blend_samples.py: whole-sample blend weights that Megatron builds exactly.

The functions are the real ones. The check of what Megatron builds is Megatron's own sizing, imported from its builder,
so a change in how Megatron sizes a blend fails these tests.
"""

from __future__ import annotations

import pytest
from megatron.core.datasets.blended_megatron_dataset_builder import _get_size_per_split_per_dataset
from megatron.core.datasets.utils import normalize
from scripts.data.blend_samples import apportion, megatron_target, samples_in, settle_on_megatron_targets


BLEND_SIZE = 174 * 256


def megatron_targets(counts: list[int]) -> list[int]:
    """Megatron's own per-corpus targets for a blend weighted in these whole samples."""
    (targets,) = zip(*_get_size_per_split_per_dataset(normalize([float(c) for c in counts]), [sum(counts)]))
    return list(targets)


# Counts at BLEND_SIZE that Megatron rounds up (7 of them) and that it builds exactly (7 others).
OVERSHOOTING = [c for c in range(1, 2000) if megatron_target(c, BLEND_SIZE) != c][:7]
EXACT = [c for c in range(1, 2000) if megatron_target(c, BLEND_SIZE) == c][:7]


@pytest.mark.parametrize(
    "total,weights,parts",
    [
        (10, ["0.5", "0.25", "0.25"], [5, 3, 2]),  # quotas 5, 2.5, 2.5: the tie goes to the earlier weight
        (5, ["1", "1"], [3, 2]),
        # quotas 2.335893, 1.334802, 3.329305: the one remaining sample goes to the largest remainder
        (7, ["0.333699", "0.190686", "0.475615"], [3, 1, 3]),
    ],
)
def test_apportion_splits_exactly_by_largest_remainder(total, weights, parts):
    assert apportion(total, weights) == parts


def test_apportion_refuses_a_part_that_rounds_to_zero():
    with pytest.raises(ValueError, match="cannot give every one"):
        apportion(2, ["0.9", "0.05", "0.05"])


@pytest.mark.parametrize("count", OVERSHOOTING + EXACT)
def test_megatron_target_is_the_count_megatrons_own_sizing_builds(count):
    assert megatron_target(count, BLEND_SIZE) == megatron_targets([count, BLEND_SIZE - count])[0]


def test_both_kinds_of_count_occur_at_a_real_blend_size():
    assert len(OVERSHOOTING) == len(EXACT) == 7


def test_settling_makes_every_count_exact_and_moves_none_by_more_than_one():
    counts = OVERSHOOTING + EXACT
    counts.append(BLEND_SIZE - sum(counts))
    settled = settle_on_megatron_targets(counts, BLEND_SIZE)
    assert sum(settled) == BLEND_SIZE
    assert megatron_targets(settled) == settled
    assert all(abs(a - b) <= 1 for a, b in zip(settled, counts))


def test_a_count_with_no_partner_to_trade_with_is_refused():
    with pytest.raises(ValueError, match="no single-sample trade"):
        settle_on_megatron_targets([OVERSHOOTING[0]], BLEND_SIZE)


@pytest.mark.parametrize(
    ("tokens", "sequence_length", "samples"),
    [(0, 8, 0), (1, 8, 0), (8, 8, 0), (9, 8, 1), (17, 8, 2), (16, 8, 1), (15_200_000_000, 4096, 3_710_937)],
)
def test_a_pass_over_a_corpus_holds_one_sample_per_full_window_with_a_label_token_to_spare(
    tokens, sequence_length, samples
):
    assert samples_in(tokens, sequence_length) == samples

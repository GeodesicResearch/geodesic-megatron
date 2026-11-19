# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Blend weights in whole samples that Megatron builds exactly.

Megatron sizes a blend as the sum over its corpora of ceil(size * normalized weight) in float64. Fractional weights
round every target up and build a few surplus samples, which the sampler leaves unread at random. Weights that are
whole sample counts summing to the blend's size give targets equal to the counts, apart from the rounding that
``megatron_target`` and ``settle_on_megatron_targets`` handle. The built blend is then the blend's size, and a sampler
that reads the blend once reads every sample of it once.

Catalogue:
    samples_in: the samples that one pass over a corpus holds.
    apportion: split a number of samples into whole parts proportional to weights.
    megatron_target: the samples Megatron builds for one count of a blend.
    settle_on_megatron_targets: trade single samples until every count is one that Megatron builds exactly.
"""

from __future__ import annotations

import math
from fractions import Fraction


def samples_in(tokens: int, sequence_length: int) -> int:
    """The samples of ``sequence_length`` tokens that one pass over a corpus of ``tokens`` tokens holds.

    Each sample's labels need one token more than its inputs, so a corpus of ``tokens`` tokens holds
    ``(tokens - 1) // sequence_length`` samples, and none when it is empty."""
    return max(0, (tokens - 1) // sequence_length)


def apportion(total: int, weights: list) -> list[int]:
    """Split ``total`` into whole parts proportional to ``weights``, by largest remainder.

    Weights are read as the decimals they are written as, so the split is exact; equal remainders go to
    the earlier weight. A part that rounds to zero is refused, because Megatron cannot build a corpus
    of no samples.
    """
    exact = [Fraction(str(weight)) for weight in weights]
    quotas = [total * weight / sum(exact) for weight in exact]
    parts = [math.floor(quota) for quota in quotas]
    by_remainder = sorted(range(len(quotas)), key=lambda i: (parts[i] - quotas[i], i))
    for i in by_remainder[: total - sum(parts)]:
        parts[i] += 1
    if 0 in parts:
        raise ValueError(f"{total} samples cannot give every one of weights {list(weights)} a sample")
    return parts


def megatron_target(samples: int, blend_size: int) -> int:
    """The samples Megatron builds for a corpus weighted ``samples`` in a blend weighted in samples summing
    to ``blend_size``.

    Its builder targets ceil(size * weight / sum of weights) per corpus (``normalize`` then
    ``_get_size_per_split_per_dataset``) in float64. Whole-number weights sum exactly, so this is the
    same two correctly rounded operations; about one count in twenty comes out one above itself.
    """
    return math.ceil(blend_size * (samples / blend_size))


def settle_on_megatron_targets(counts: list[int], blend_size: int) -> list[int]:
    """Trade single samples between corpora until Megatron's target for every count is the count itself.

    A count Megatron would round up trades one sample with another corpus, in whichever direction
    leaves both exact, so the total is unchanged and no count moves more than one sample.
    """
    settled = list(counts)
    moved: set[int] = set()

    def exact(samples: int) -> bool:
        return megatron_target(samples, blend_size) == samples

    for i in range(len(settled)):
        if exact(settled[i]):
            continue
        trade = next(
            (
                (delta, j)
                for delta in (-1, 1)
                if exact(settled[i] + delta)
                for j in range(len(settled))
                if j != i and j not in moved and exact(settled[j]) and exact(settled[j] - delta)
            ),
            None,
        )
        if trade is None:
            raise ValueError(f"no single-sample trade makes {settled[i]} of {blend_size} samples exact")
        delta, j = trade
        settled[i] += delta
        settled[j] -= delta
        moved.update((i, j))
    return settled

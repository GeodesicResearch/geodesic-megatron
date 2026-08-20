# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared harness for the control-pretraining campaign's config tests.

The launcher (`pipeline_training_run.py`) builds a recipe, merges the override YAML through
OmegaConf, and applies the result back onto the ``ConfigContainer``. Every campaign test
module asserts its configs through that exact sequence, and the native ``.bin/.idx`` blends
share one well-formedness contract — this module is the single home for both, so the merge
the tests perform cannot drift from the launcher's, and a blend rule fixed in one campaign
arm cannot silently miss the others.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from megatron.core.datasets.utils import get_blend_from_list
from omegaconf import OmegaConf

from megatron.bridge.training.utils.omegaconf_utils import apply_overrides, create_omegaconf_dict_config


def merge_onto_recipe(path: Path, recipe_fn):
    """Return ``recipe_fn()`` with the override YAML at ``path`` merged on, as the launcher does."""
    cfg = recipe_fn()
    merged, excluded = create_omegaconf_dict_config(cfg)
    merged = OmegaConf.merge(merged, OmegaConf.load(path))
    apply_overrides(cfg, OmegaConf.to_container(merged, resolve=True), excluded)
    return cfg


def assert_blend_is_well_formed(data_path, label: str) -> None:
    """Assert a flat interleaved weight/prefix blend parses as upstream Megatron will parse it.

    An odd-length list is not an error upstream: ``get_blend_from_list`` reads it as
    prefixes-only, so the weights become filenames and the run dies hours later looking for
    ``0.0875.idx``. Weights must sum to 1.0 so every entry stays auditable against the token
    count recorded beside it.
    """
    data_path = [str(x) for x in data_path]
    assert len(data_path) % 2 == 0, f"{label}: odd-length data_path becomes an unweighted blend"

    blend = get_blend_from_list(data_path)
    prefixes, weights = blend[0], blend[1]
    assert weights is not None, f"{label}: upstream did not read weights from this list"
    assert len(prefixes) == len(weights)
    assert abs(sum(weights) - 1.0) < 1e-6, f"{label}: weights sum to {sum(weights)}"
    assert all(w > 0 for w in weights)
    for prefix in prefixes:
        assert not prefix.endswith((".bin", ".idx")), f"{label}: {prefix} must be extension-less"
        assert prefix.startswith("/projects/a5k/public/data/"), prefix


def assert_shard_weights_are_token_proportional(data_path, corpus_slug: str, total_weight: float) -> None:
    """Assert a sharded corpus's per-shard weights split ``total_weight`` by measured tokens.

    The shards of one corpus are cut at equal DOCUMENT counts, not equal tokens, so equal
    weights would cycle the smaller shards more often than the larger ones. Each shard's
    weight must be ``round(total_weight x shard_tokens / corpus_tokens, 6)``, with any
    six-decimal rounding residue folded into the largest shard so the set sums to exactly
    ``total_weight``. Tokens are read from the ``.provenance.json`` the data build wrote
    beside each shard prefix — the paths come from the blend itself, so a relocated corpus
    fails loudly here rather than skipping.
    """
    data_path = [str(x) for x in data_path]
    shards = {}
    for weight, prefix in zip(data_path[::2], data_path[1::2]):
        if f"__{corpus_slug}/" in prefix:
            shards[prefix] = float(weight)
    assert shards, f"no {corpus_slug} shard prefixes in the blend"
    assert abs(sum(shards.values()) - total_weight) < 1e-9, f"{corpus_slug} shard weights do not sum to {total_weight}"

    tokens = {}
    for prefix in shards:
        prov = Path(prefix + ".provenance.json")
        if not prov.exists():
            pytest.skip(f"corpus provenance not mounted on this host: {prov}")
        tokens[prefix] = json.loads(prov.read_text())["totals"]["total_tokens"]
    total = sum(tokens.values())

    expected = {prefix: round(total_weight * tokens[prefix] / total, 6) for prefix in shards}
    residue = round(total_weight - sum(expected.values()), 6)
    largest = max(expected, key=lambda prefix: tokens[prefix])
    expected[largest] = round(expected[largest] + residue, 6)
    for prefix, weight in shards.items():
        assert weight == expected[prefix], prefix

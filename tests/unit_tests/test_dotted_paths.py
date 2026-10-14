# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""scripts/dotted_paths.py: a nested mapping as its leaves by dotted path, and the value one dotted path names."""

import pytest
from scripts.dotted_paths import dotted_leaves, value_at


def test_nested_keys_become_dotted_paths_and_a_list_stays_one_value():
    nested = {"train": {"global_batch_size": 4}, "model": {"fp8": None}, "dataset": {"data_path": ["1.0", "/data/a"]}}
    assert dotted_leaves(nested) == {
        "train.global_batch_size": 4,
        "model.fp8": None,
        "dataset.data_path": ["1.0", "/data/a"],
    }


def test_a_prefix_opens_every_path_and_an_empty_mapping_has_no_leaves():
    assert dotted_leaves({"a": {"b": 1}}, "root") == {"root.a.b": 1}
    assert dotted_leaves({"a": {}}) == {}


def test_a_path_names_a_mapping_key_or_a_list_index_and_raises_when_it_names_nothing():
    document = {"corpora": [{"docs": 41, "shards": {"0": {"docs": 40}}}], "outcomes": {"counters": "PASS"}}
    assert value_at(document, "corpora.0.docs") == 41
    assert value_at(document, "corpora.0.shards.0.docs") == 40, "a digit names a mapping's key where it is a mapping"
    assert value_at(document, "outcomes") == {"counters": "PASS"}
    for path in ("corpora.1.docs", "corpora.-1.docs", "corpora.docs", "outcomes.counters.x", "absent"):
        with pytest.raises(KeyError):
            value_at(document, path)

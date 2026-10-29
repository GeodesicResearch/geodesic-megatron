# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""scripts/dotted_paths.py: a nested mapping as its leaves by dotted path."""

from scripts.dotted_paths import dotted_leaves


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

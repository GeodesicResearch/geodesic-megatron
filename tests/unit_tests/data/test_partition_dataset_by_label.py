# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""The label partitioner's refusal surface is the corpus-integrity guarantee.

A label claimed by two groups duplicates its documents into two corpora; a label no
group claims silently vanishes from training. Both would produce arms trained on the
wrong data with no error anywhere downstream, so ``assign_labels`` refuses them — and
these tests pin each refusal plus the ``rest`` sentinel that makes the common
"four named topics, everything else is core" partition expressible. ``main`` itself
(HF download + JSONL writes) is exercised by the campaign, not unit-tested: it is a
network boundary.
"""

from pathlib import Path

import pytest
import yaml

from tests.unit_tests.gr_test_utils import load_script


_SCRIPT = Path(__file__).parents[3] / "scripts" / "data" / "partition_dataset_by_label.py"


@pytest.fixture(scope="module")
def partition():
    return load_script("partition_dataset_by_label", _SCRIPT)


def _config(**overrides):
    cfg = {
        "dataset": "org/dataset",
        "split": "train",
        "label_column": "topic",
        "text_column": "story",
        "normalize_labels": True,
        "val_fraction": 0.1,
        "split_seed": 42,
        "groups": {"aliens": ["alien-encounters"], "core": "rest"},
        "output_root": "/data/out",
        "json_key": "text",
    }
    cfg.update(overrides)
    return cfg


def _write(tmp_path, cfg) -> Path:
    path = tmp_path / "partition.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


class TestLoadConfig:
    def test_a_complete_config_loads(self, partition, tmp_path):
        assert partition.load_config(_write(tmp_path, _config()))["dataset"] == "org/dataset"

    @pytest.mark.parametrize("key", ["dataset", "groups", "val_fraction", "json_key"])
    def test_a_missing_required_key_is_refused_by_name(self, partition, tmp_path, key):
        cfg = _config()
        del cfg[key]
        with pytest.raises(SystemExit, match=key):
            partition.load_config(_write(tmp_path, cfg))

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
    def test_an_out_of_range_val_fraction_is_refused(self, partition, tmp_path, fraction):
        with pytest.raises(SystemExit, match="val_fraction"):
            partition.load_config(_write(tmp_path, _config(val_fraction=fraction)))

    def test_two_rest_groups_are_refused(self, partition, tmp_path):
        cfg = _config(groups={"a": "rest", "b": "rest"})
        with pytest.raises(SystemExit, match="at most one group"):
            partition.load_config(_write(tmp_path, cfg))


class TestAssignLabels:
    def test_every_label_lands_in_exactly_one_group(self, partition):
        cfg = _config(groups={"aliens": ["alien-encounters"], "eras": ["bygone-eras"], "core": "rest"})
        assignment = partition.assign_labels(cfg, {"alien-encounters", "bygone-eras", "dragons", "pirates"})
        assert assignment == {
            "alien-encounters": "aliens",
            "bygone-eras": "eras",
            "dragons": "core",
            "pirates": "core",
        }

    def test_a_label_claimed_twice_is_refused(self, partition):
        cfg = _config(groups={"a": ["dragons"], "b": ["dragons"], "core": "rest"})
        with pytest.raises(SystemExit, match="claimed by both"):
            partition.assign_labels(cfg, {"dragons"})

    def test_a_bare_string_group_value_is_refused_by_name(self, partition):
        """The natural YAML slip `aliens: alien-encounters` (no list brackets) must be
        refused naming the group and the fix — not iterated character-by-character into a
        'label a is not in the dataset' misdiagnosis or a silently wrong partition."""
        cfg = _config(groups={"aliens": "alien-encounters", "core": "rest"})
        with pytest.raises(SystemExit, match="must be a LIST of labels"):
            partition.assign_labels(cfg, {"alien-encounters"})

    def test_a_configured_label_absent_from_the_dataset_is_refused(self, partition):
        cfg = _config(groups={"aliens": ["alien-encounters"], "core": "rest"})
        with pytest.raises(SystemExit, match="not in the dataset"):
            partition.assign_labels(cfg, {"dragons"})

    def test_unclaimed_labels_without_a_rest_group_are_refused(self, partition):
        cfg = _config(groups={"aliens": ["alien-encounters"]})
        with pytest.raises(SystemExit, match="belong to no group"):
            partition.assign_labels(cfg, {"alien-encounters", "dragons"})

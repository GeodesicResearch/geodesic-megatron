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
"""The configs that carry a config written for Isambard to Verda: the relocation map
(``configs/clusters/verda_from_isambard.yaml``) and the quickstart overlays under ``configs/quickstart/verda/``.

A Verda quickstart is its Isambard quickstart with the storage moved, so a run of one compares with a run of the
other. The tests hold each overlay to that: nothing but storage and the W&B name is stated, and no path of the
composed config is left on Isambard's storage.
"""

import pytest
from scripts.training.config_compose import (
    BASE_CONFIG_KEY,
    RELOCATE_KEY,
    load_composed_yaml,
    parse_yaml_mapping,
    relocate_paths,
)

from pipeline.cluster import cluster_profile
from tests.unit_tests.child_process import REPO_ROOT


STORAGE_MAP = REPO_ROOT / "configs" / "clusters" / "verda_from_isambard.yaml"
QUICKSTARTS = REPO_ROOT / "configs" / "quickstart"
# The Verda overlays of an Isambard quickstart of the same name.
OVERLAYS = sorted(path for path in (QUICKSTARTS / "verda").glob("*.yaml") if (QUICKSTARTS / path.name).is_file())
# What an overlay can state besides its base and its map: where a warm start and a run's own checkpoints are, the
# name its runs carry in W&B, and that no TensorBoard directory is written (every training config sets it null,
# and an Isambard quickstart that leaves it unset writes under the submit directory).
STATED = {("checkpoint", "pretrained_checkpoint"), ("checkpoint", "load"), ("checkpoint", "save")} | {
    ("logger", "wandb_exp_name"),
    ("logger", "tensorboard_dir"),
}
# The profile values that name a storage root. The map must send Isambard's to Verda's.
ROOT_KEYS = (
    "CLUSTER_DATA_ROOT",
    "CLUSTER_MODEL_ROOT",
    "CLUSTER_HF_HOME",
    "CLUSTER_HF_DATASETS_CACHE",
    "CLUSTER_NEMO_HOME",
    "CLUSTER_LOG_ROOT",
    "CLUSTER_WANDB_DIR",
    "CLUSTER_PROFILE_ROOT",
)


def _paths(value, field=""):
    """Every absolute path in a nested config value, with its dotted field name."""
    if isinstance(value, str) and value.startswith("/"):
        yield field, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _paths(item, f"{field}.{key}" if field else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _paths(item, f"{field}[{index}]")


@pytest.fixture
def verda_user(monkeypatch):
    """The account the map's and the profiles' per-user roots are resolved for."""
    monkeypatch.setenv("USER", "someone")


def test_the_map_sends_each_isambard_storage_root_to_verdas(verda_user):
    """The profiles are what the launchers read and the map is what the configs read, so the two must agree."""
    prefixes = {
        old: new.replace("$USER", "someone")
        for old, new in parse_yaml_mapping(STORAGE_MAP.read_text(), str(STORAGE_MAP)).items()
    }
    isambard, verda = cluster_profile("isambard"), cluster_profile("verda")
    for key in ROOT_KEYS:
        assert relocate_paths(isambard[key], prefixes) == verda[key], key


def test_there_are_verda_quickstarts():
    assert OVERLAYS, f"no overlay under {QUICKSTARTS / 'verda'} is named after a quickstart in {QUICKSTARTS}"


@pytest.mark.parametrize("overlay", OVERLAYS, ids=lambda path: path.stem)
def test_a_verda_quickstart_is_its_isambard_quickstart_with_the_storage_moved(overlay, verda_user):
    own = parse_yaml_mapping(overlay.read_text(), str(overlay))
    assert own.pop(BASE_CONFIG_KEY) == f"../{overlay.name}"
    assert (overlay.parent / own.pop(RELOCATE_KEY)).resolve() == STORAGE_MAP
    stated = {(section, field) for section, fields in own.items() for field in fields}
    assert stated <= STATED, f"{overlay.name} states more than storage and its W&B name: {sorted(stated - STATED)}"

    composed = load_composed_yaml(overlay)
    isambard_roots = tuple(cluster_profile("isambard")["CLUSTER_STORAGE_BINDS"].split(","))
    left_behind = [f"{field}: {path}" for field, path in _paths(composed) if path.startswith(isambard_roots)]
    assert not left_behind, "paths still on Isambard's storage: " + "; ".join(left_behind)
    # The same fields hold paths as in the Isambard quickstart, so no storage field was dropped or added.
    base = load_composed_yaml(QUICKSTARTS / overlay.name)
    assert {field for field, _ in _paths(composed)} >= {field for field, _ in _paths(base)}

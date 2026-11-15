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
(``configs/clusters/verda_from_isambard.yaml``), the quickstart overlays under ``configs/quickstart/verda/`` and
the Verda arms of the control-pretraining campaigns (``configs/control_pretraining/<campaign>_verda/``).

A Verda quickstart is its Isambard quickstart with the storage moved, so a run of one compares with a run of the
other. The tests hold each overlay to that: nothing but storage and the W&B name is stated, and no path of the
composed config is left on Isambard's storage.

A Verda arm trains what its campaign trains. Each of its stage configs overlays the campaign's stage config of
the same name and can also state how the run uses Verda's hardware. The tests fail if one states anything else,
such as a batch size or a learning rate.

The other files of the quickstart directory exist for Verda alone. The tests hold those, and the arms, to the
cluster's rules.
"""

from functools import partial

import pytest
from scripts.dotted_paths import dotted_leaves
from scripts.training.config_compose import (
    BASE_CONFIG_KEY,
    RELOCATE_KEY,
    load_composed_yaml,
    parse_yaml_mapping,
    relocate_paths,
)
from scripts.training.launcher_source import env_override_entries

from pipeline.cluster import cluster_profile
from pipeline.training import recipes
from pipeline.training.datasets import BIN_IDX_READ_KEY, bin_idx_dataset_config
from tests.unit_tests.campaign_config import assert_levers_are_set, merge_onto_recipe
from tests.unit_tests.child_process import REPO_ROOT


STORAGE_MAP = REPO_ROOT / "configs" / "clusters" / "verda_from_isambard.yaml"
QUICKSTARTS = REPO_ROOT / "configs" / "quickstart"
VERDA_CONFIGS = sorted((QUICKSTARTS / "verda").glob("*.yaml"))
# The Verda overlays of an Isambard quickstart of the same name.
OVERLAYS = [path for path in VERDA_CONFIGS if (QUICKSTARTS / path.name).is_file()]
# The configs defined for Verda: a topology for its nodes, or a quickstart Isambard does not have.
DEFINED_FOR_VERDA = [path for path in VERDA_CONFIGS if path not in OVERLAYS]
CAMPAIGNS = REPO_ROOT / "configs" / "control_pretraining"
# The stage configs of the Verda arms. ``<campaign>_verda/<stage>.yaml`` overlays ``<campaign>/<stage>.yaml``.
ARM_OVERLAYS = sorted(CAMPAIGNS.glob("*_verda/*.yaml"))
# What an overlay can state besides its base and its map, by dotted name. The checkpoint names say where a warm
# start and a run's own checkpoints are. The logger names give the W&B name of its runs and write no TensorBoard
# directory. Every training config sets that directory null, and an Isambard quickstart that leaves it unset writes
# under the submit directory.
STORAGE_AND_NAME = {
    "checkpoint.pretrained_checkpoint",
    "checkpoint.load",
    "checkpoint.save",
    "logger.wandb_exp_name",
    "logger.tensorboard_dir",
}
# What a stage config of an arm can state as well. A name with no dot, or the name of a section, covers all of it.
ARM_STATES = STORAGE_AND_NAME | {
    # Where Verda keeps a packed corpus. Its shards are laid out differently from Isambard's.
    "dataset.dataset_root",
    "dataset.packed_sequence_specs.packed_train_data_path",
    # How the run uses the hardware. The names cover precision, parallelism, micro-batch and recomputation. They
    # also cover the dispatcher, the fusions, the overlaps, host garbage collection and how the token files are read.
    "mixed_precision",
    "dataset.bin_idx_read",
    "train.micro_batch_size",
    "train.manual_gc",
    "train.manual_gc_interval",
    "train.manual_gc_freeze",
    "comm_overlap",
    "ddp",
    "model.tensor_model_parallel_size",
    "model.pipeline_model_parallel_size",
    "model.expert_model_parallel_size",
    "model.context_parallel_size",
    "model.recompute_granularity",
    "model.recompute_method",
    "model.recompute_num_layers",
    "model.recompute_modules",
    "model.moe_token_dispatcher_type",
    "model.moe_flex_dispatcher_backend",
    "model.moe_router_fusion",
    "model.cross_entropy_loss_fusion",
    "model.cross_entropy_fusion_impl",
    "model.cross_entropy_fusion_saved_logit_chunks",
}
# The configs that set a topology for Verda's nodes.
SET_A_TOPOLOGY = DEFINED_FOR_VERDA + ARM_OVERLAYS
# The launcher mode of each stage of a campaign. Midtraining continues pretraining on a .bin/.idx blend.
STAGE_MODES = {"pretrain": "pretrain", "midtrain": "pretrain", "sft": "sft"}
# The sequence length from which the Mamba scan needs its fp32 state. A BF16 state between chunks produces NaNs there.
LONG_SEQUENCE = 32768
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


def _left_on_isambard(composed):
    """The fields of a composed config whose path is on Isambard's storage."""
    isambard_roots = tuple(cluster_profile("isambard")["CLUSTER_STORAGE_BINDS"].split(","))
    return [f"{field}: {path}" for field, path in _paths(composed) if path.startswith(isambard_roots)]


def _isambard_config(overlay):
    """The Isambard config a Verda overlay is named after: a campaign's stage config, or a quickstart."""
    if overlay in ARM_OVERLAYS:
        return CAMPAIGNS / overlay.parent.name.removesuffix("_verda") / overlay.name
    return QUICKSTARTS / overlay.name


def _model_and_mode(overlay):
    """The model and the launcher mode of a stage config of an arm, from its name
    (``nemotron_<model>_<campaign>_<stage>.yaml``)."""
    _, model, *_, stage = overlay.stem.split("_")
    return model, STAGE_MODES[stage]


def _beyond(names, allowed):
    """The names that are not in ``allowed`` and not under a section it names."""
    return sorted(
        name for name in names if not any(name == entry or name.startswith(f"{entry}.") for entry in allowed)
    )


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


def test_there_are_verda_quickstarts_and_arms():
    assert OVERLAYS, f"no overlay under {QUICKSTARTS / 'verda'} is named after a quickstart in {QUICKSTARTS}"
    assert ARM_OVERLAYS, f"no campaign under {CAMPAIGNS} has a Verda arm"


@pytest.mark.parametrize("overlay", OVERLAYS + ARM_OVERLAYS, ids=lambda path: f"{path.parent.name}-{path.stem}")
def test_a_verda_overlay_states_only_what_verda_changes(overlay, verda_user):
    """A quickstart overlay moves the storage. A stage config of an arm also states how it uses the hardware."""
    isambard_config = _isambard_config(overlay)
    own = parse_yaml_mapping(overlay.read_text(), str(overlay))
    assert (overlay.parent / own.pop(BASE_CONFIG_KEY)).resolve() == isambard_config
    assert (overlay.parent / own.pop(RELOCATE_KEY)).resolve() == STORAGE_MAP
    beyond = _beyond(dotted_leaves(own), ARM_STATES if overlay in ARM_OVERLAYS else STORAGE_AND_NAME)
    assert not beyond, f"{overlay.name} states more than Verda changes: {beyond}"

    composed = load_composed_yaml(overlay)
    left_behind = _left_on_isambard(composed)
    assert not left_behind, "paths still on Isambard's storage: " + "; ".join(left_behind)
    # The same fields hold paths as in the Isambard config, so no storage field was dropped or added.
    base = load_composed_yaml(isambard_config)
    assert {field for field, _ in _paths(composed)} >= {field for field, _ in _paths(base)}


@pytest.mark.parametrize(
    "name",
    [
        "train.global_batch_size",
        "train.train_iters",
        "dataset.data_path",
        "dataset.seq_length",
        "dataset.packed_sequence_specs.packed_sequence_size",
        "tokenizer.tokenizer_model",
        "optimizer.lr",
        "scheduler.lr_decay_style",
    ],
)
def test_a_verda_arm_cannot_state_what_its_campaign_trains(name):
    assert _beyond([name], ARM_STATES) == [name]


@pytest.mark.parametrize("overlay", ARM_OVERLAYS, ids=lambda path: f"{path.parent.name}-{path.stem}")
def test_every_setting_of_a_verda_arm_reaches_the_config_it_trains(overlay, verda_user):
    """The launcher's merge drops a key the config classes lack, so each stated value is read back from the merged
    config. A misspelled field does not arrive. The read settings of a .bin/.idx blend are read from the dataset
    config the launcher builds."""
    own = parse_yaml_mapping(overlay.read_text(), str(overlay))
    del own[BASE_CONFIG_KEY], own[RELOCATE_KEY]
    stated = dotted_leaves(own)
    read_prefix = f"dataset.{BIN_IDX_READ_KEY}."
    read_settings = {
        name.removeprefix(read_prefix): stated.pop(name) for name in list(stated) if name.startswith(read_prefix)
    }
    model, mode = _model_and_mode(overlay)

    assert_levers_are_set(
        merge_onto_recipe(overlay, partial(recipes.RECIPE_MAP[(model, mode)], None)), stated, overlay.name
    )
    if read_settings:
        dataset = bin_idx_dataset_config(load_composed_yaml(overlay)["dataset"], mode)
        assert {key: getattr(dataset, key) for key in read_settings} == read_settings


@pytest.mark.parametrize("overlay", ARM_OVERLAYS, ids=lambda path: f"{path.parent.name}-{path.stem}")
def test_a_verda_arm_keeps_the_fp32_scan_state_on_long_sequences(overlay, verda_user):
    """The launcher settings are read through the launcher's own hook."""
    settings = dict(entry.split("=", 1) for entry in env_override_entries(str(overlay.with_suffix(".env"))))
    if load_composed_yaml(overlay)["model"]["seq_length"] >= LONG_SEQUENCE:
        assert settings.get("ISAMBARD_FP32_SSM_STATE") != "0"


@pytest.mark.parametrize("config", SET_A_TOPOLOGY, ids=lambda path: f"{path.parent.name}-{path.stem}")
def test_a_config_that_sets_a_verda_topology_keeps_the_clusters_rules(config, verda_user):
    composed = load_composed_yaml(config)
    left_behind = _left_on_isambard(composed)
    assert not left_behind, "paths on Isambard's storage: " + "; ".join(left_behind)
    assert composed["logger"]["tensorboard_dir"] is None
    # Tensor and expert parallelism share a node, so the expert all-to-all never crosses the fabric.
    model = composed["model"]
    in_node = model["tensor_model_parallel_size"] * model["expert_model_parallel_size"]
    assert in_node <= int(cluster_profile("verda")["CLUSTER_GPUS_PER_NODE"])

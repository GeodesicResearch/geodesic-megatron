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
other. The tests hold each overlay to that. It states only storage, the W&B name and what Verda's image requires.
No path of the composed config stays on Isambard's storage.

A Verda arm trains what its campaign trains. Each of its stage configs overlays the campaign's stage config of the
same stage. It can also state how the run uses Verda's hardware. The model in its name can differ from the base's,
since the launcher's model argument picks the recipe. The tests fail if a stage config states anything else, such as
a batch size or a learning rate. They also fail if the other model's recipe changes a training setting
(``TRAINING_SECTIONS``).

The other files of the quickstart directory exist for Verda alone. The tests hold those, and the arms, to the
cluster's rules.
"""

import dataclasses
import math
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

from megatron.bridge.training.config import runtime_config_update
from megatron.bridge.training.mixed_precision import get_mixed_precision_config
from pipeline.cluster import cluster_profile
from pipeline.training import recipes
from pipeline.training.datasets import BIN_IDX_READ_KEY, bin_idx_dataset_config
from tests.unit_tests.campaign_config import assert_levers_are_set, flatten_merged_config, merge_onto_recipe
from tests.unit_tests.child_process import REPO_ROOT


STORAGE_MAP = REPO_ROOT / "configs" / "clusters" / "verda_from_isambard.yaml"
QUICKSTARTS = REPO_ROOT / "configs" / "quickstart"
VERDA_CONFIGS = sorted((QUICKSTARTS / "verda").glob("*.yaml"))
# The Verda overlays of an Isambard quickstart of the same name.
OVERLAYS = [path for path in VERDA_CONFIGS if (QUICKSTARTS / path.name).is_file()]
# The configs defined for Verda: a topology for its nodes, or a quickstart Isambard does not have.
DEFINED_FOR_VERDA = [path for path in VERDA_CONFIGS if path not in OVERLAYS]
CAMPAIGNS = REPO_ROOT / "configs" / "control_pretraining"
# The stage configs of the Verda arms. ``<campaign>_verda/nemotron_<model>_<name>.yaml`` overlays
# ``<campaign>/nemotron_<any model>_<name>.yaml``.
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
# What Verda's image requires a config to state. Its Transformer Engine refuses the fused router's aux loss on a
# packed batch (test_a_verda_config_that_trains_on_packed_batches_runs_the_router_unfused).
IMAGE_REQUIRES = {"model.moe_router_fusion"}
# What a stage config of an arm can state as well. A name with no dot, or the name of a section, covers all of it.
ARM_STATES = STORAGE_AND_NAME | {
    # Where Verda keeps a packed corpus. Its shards are laid out differently from Isambard's.
    "dataset.dataset_root",
    "dataset.packed_sequence_specs.packed_train_data_path",
    # The multiple the collate pads a packed batch to. The precision preset sets what it must be.
    "dataset.dataset_kwargs.pad_seq_length_to_mult",
    # How the run uses the hardware. The names cover precision, parallelism, micro-batch and recomputation. They
    # also cover the dispatcher, the fusions, the overlaps and CUDA graphs. The rest cover host garbage collection
    # and how the token files are read.
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
    "model.overlap_moe_expert_parallel_comm",
    "model.cuda_graph_impl",
    "model.mtp_num_layers",
}
# The names of ``ARM_STATES`` that an arm can state with one value only. A null builds no MTP layer, and
# ``test_a_verda_arm_trains_the_mtp_layers_of_its_campaign`` holds that to the campaign.
ARM_STATES_ONLY_AS = {"model.mtp_num_layers": None}
# The sections of a trained config that hold how a stage trains, beside its data and its model. A stage config
# leaves some of their fields to the recipe of its model.
TRAINING_SECTIONS = ("optimizer", "scheduler", "train", "rng", "tokenizer")
# The configs that set a topology for Verda's nodes.
SET_A_TOPOLOGY = DEFINED_FOR_VERDA + ARM_OVERLAYS
# The launcher mode of each stage a config's name gives. Midtraining continues pretraining on a .bin/.idx blend.
STAGE_MODES = {"pretrain": "pretrain", "midtrain": "pretrain", "sft": "sft", "cpt": "cpt"}
# The sequence length from which a BF16 state between the chunks of the Mamba scan can produce NaNs.
LONG_SEQUENCE = 32768
# The token multiple that Transformer Engine's MXFP8 quantizer requires.
MXFP8_TOKEN_MULTIPLE = 32
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
    """The Isambard config a Verda overlay builds on. An arm's is the campaign's stage config that its ``base_config``
    names. A quickstart's is the quickstart of the same name."""
    if overlay in ARM_OVERLAYS:
        own = parse_yaml_mapping(overlay.read_text(), str(overlay))
        return (overlay.parent / own[BASE_CONFIG_KEY]).resolve()
    return QUICKSTARTS / overlay.name


def _stage_name(config):
    """The name of a stage config of a campaign or an arm without its model (``<campaign>_<stage>`` of
    ``nemotron_<model>_<campaign>_<stage>.yaml``)."""
    return "_".join(config.stem.split("_")[2:])


def _model_and_mode(config):
    """The model and the launcher mode of a config, from its name: ``nemotron_<model>_<campaign>_<stage>.yaml`` for a
    stage config of a campaign or an arm, ``nemotron_<model>_quickstart_<stage>[_<variant>].yaml`` for a quickstart."""
    _, model, *rest = config.stem.split("_")
    return model, STAGE_MODES[next(part for part in rest if part in STAGE_MODES)]


def _trained_config(config):
    """The config the launcher trains for a config: the file merged onto the recipe of its model and mode."""
    model, mode = _model_and_mode(config)
    return merge_onto_recipe(config, partial(recipes.RECIPE_MAP[(model, mode)], None))


def _launcher_settings(overlay):
    """The settings in the ``.env`` beside a stage config of an arm, read through the launcher's own hook."""
    return dict(entry.split("=", 1) for entry in env_override_entries(str(overlay.with_suffix(".env"))))


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
    if overlay in ARM_OVERLAYS:
        # The campaign's stage config of the same stage. Its model can differ.
        assert isambard_config.parent == CAMPAIGNS / overlay.parent.name.removesuffix("_verda")
        assert _stage_name(isambard_config) == _stage_name(overlay)
    stated = dotted_leaves(own)
    beyond = _beyond(stated, ARM_STATES if overlay in ARM_OVERLAYS else STORAGE_AND_NAME | IMAGE_REQUIRES)
    assert not beyond, f"{overlay.name} states more than Verda changes: {beyond}"
    assert all(stated.get(name, value) == value for name, value in ARM_STATES_ONLY_AS.items())

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

    assert_levers_are_set(_trained_config(overlay), stated, overlay.name)
    if read_settings:
        dataset = bin_idx_dataset_config(load_composed_yaml(overlay)["dataset"], _model_and_mode(overlay)[1])
        assert {key: getattr(dataset, key) for key in read_settings} == read_settings


@pytest.mark.parametrize("overlay", ARM_OVERLAYS, ids=lambda path: f"{path.parent.name}-{path.stem}")
def test_every_section_of_a_verda_arm_arrives_as_its_config_class(overlay, verda_user):
    """The merge sets a section the recipe leaves None to the plain mapping the file states. Setup cannot read it.
    The baseline's pretrain and midtrain configs state a comm_overlap section, and Super's pretrain recipe has none."""
    cfg = _trained_config(overlay)
    plain = [field.name for field in dataclasses.fields(cfg) if isinstance(getattr(cfg, field.name), dict)]
    assert not plain, f"{overlay.name} merges these sections as plain mappings: {plain}"


@pytest.mark.parametrize("overlay", ARM_OVERLAYS, ids=lambda path: f"{path.parent.name}-{path.stem}")
def test_a_verda_arm_trains_the_training_sections_of_its_campaign(overlay, verda_user):
    """An arm's stage config merges onto the recipe of its own model. Its base merges onto the recipe of the
    campaign's model. A field the base leaves to its recipe can then take another value. In ``TRAINING_SECTIONS`` the
    arm can differ only where it states how the run uses the hardware."""
    arm = flatten_merged_config(_trained_config(overlay))
    campaign = flatten_merged_config(_trained_config(_isambard_config(overlay)))
    differing = [
        name
        for name in arm.keys() | campaign.keys()
        if name.split(".")[0] in TRAINING_SECTIONS and arm.get(name) != campaign.get(name)
    ]
    assert not _beyond(differing, ARM_STATES), f"{overlay.name} trains differently from its campaign: {differing}"


@pytest.mark.parametrize("overlay", ARM_OVERLAYS, ids=lambda path: f"{path.parent.name}-{path.stem}")
def test_a_verda_arm_trains_the_mtp_layers_of_its_campaign(overlay, verda_user):
    """MTP layers add a loss. Super's recipe builds two and Nano's builds none, so a Super stage of a Nano campaign
    states ``mtp_num_layers: null``. A null and a 0 both build no MTP layer."""
    arm = _trained_config(overlay).model.mtp_num_layers or 0
    campaign = _trained_config(_isambard_config(overlay)).model.mtp_num_layers or 0
    assert arm == campaign


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize("overlay", ARM_OVERLAYS, ids=lambda path: f"{path.parent.name}-{path.stem}")
def test_a_verda_arm_passes_the_trainers_config_update(overlay, verda_user, tmp_path, monkeypatch):
    """The update resolves the precision preset and applies the overlaps, which assert what a posture can combine.
    It also checks the dispatcher backend against the GPU."""
    cfg = _trained_config(overlay)
    # The update divides the world size by the model-parallel size. A launch has whole nodes of GPUs.
    model_parallel_size = (
        cfg.model.tensor_model_parallel_size * cfg.model.pipeline_model_parallel_size * cfg.model.context_parallel_size
    )
    gpus_per_node = int(cluster_profile("verda")["CLUSTER_GPUS_PER_NODE"])
    monkeypatch.setenv("WORLD_SIZE", str(math.lcm(gpus_per_node, model_parallel_size)))
    if cfg.checkpoint.pretrained_checkpoint:
        # The update asserts that the warm start exists. A test host has no checkpoint of the stage before.
        cfg.checkpoint.pretrained_checkpoint = str(tmp_path)
    runtime_config_update(cfg)


@pytest.mark.parametrize("overlay", ARM_OVERLAYS, ids=lambda path: f"{path.parent.name}-{path.stem}")
def test_a_packed_verda_arm_pads_its_batches_for_mxfp8(overlay, verda_user):
    """Under MXFP8 the share of a packed batch on each GPU must divide by 32. A shorter pad stops the run at the
    first short pack."""
    cfg = _trained_config(overlay)
    if getattr(cfg.dataset, "packed_sequence_specs", None) is None:
        pytest.skip("the stage does not train on packed batches")
    if get_mixed_precision_config(cfg.mixed_precision).fp8_recipe != "mxfp8":
        pytest.skip("the stage does not train in MXFP8")
    multiple = cfg.dataset.dataset_kwargs.get("pad_seq_length_to_mult")
    assert multiple is not None, f"{overlay.name} leaves the pad multiple to the collate's default"
    assert multiple % (MXFP8_TOKEN_MULTIPLE * cfg.model.context_parallel_size) == 0


@pytest.mark.parametrize("config", VERDA_CONFIGS + ARM_OVERLAYS, ids=lambda path: f"{path.parent.name}-{path.stem}")
def test_a_verda_config_that_trains_on_packed_batches_runs_the_router_unfused(config, verda_user):
    """Verda's Transformer Engine refuses the fused router's aux loss under a packed batch's padding mask
    (docs/verda/environment-pipeline.md, "The newer image"). The first step then stops with "total_num_tokens must
    be int64; got Float". The unfused router computes the loss in PyTorch."""
    cfg = _trained_config(config)
    if getattr(cfg.dataset, "packed_sequence_specs", None) is None:
        pytest.skip("the config does not train on packed batches")
    assert not cfg.model.moe_router_fusion, f"{config.name} fuses the router on packed batches"


@pytest.mark.parametrize("overlay", ARM_OVERLAYS, ids=lambda path: f"{path.parent.name}-{path.stem}")
def test_a_verda_arm_checks_for_nans_under_a_bf16_scan_of_long_sequences(overlay, verda_user):
    """A NaN from the BF16 scan state then stops the run at the step that produced it."""
    if load_composed_yaml(overlay)["model"]["seq_length"] < LONG_SEQUENCE:
        pytest.skip("the stage does not train on long sequences")
    if _launcher_settings(overlay).get("ISAMBARD_FP32_SSM_STATE") != "0":
        pytest.skip("the scan keeps its fp32 state")
    cfg = _trained_config(overlay)
    assert cfg.ddp.check_for_nan_in_grad
    assert cfg.rerun_state_machine.check_for_nan_in_loss


@pytest.mark.parametrize("overlay", ARM_OVERLAYS, ids=lambda path: f"{path.parent.name}-{path.stem}")
def test_a_verda_arm_gives_the_expert_parallel_overlap_a_second_hardware_queue(overlay, verda_user):
    """The overlap runs its dispatch and combine on a second stream. With one queue the two streams run in turn."""
    composed = load_composed_yaml(overlay)
    overlap = "overlap_moe_expert_parallel_comm"
    if not ((composed.get("comm_overlap") or {}).get(overlap) or composed["model"].get(overlap)):
        pytest.skip("the stage does not run the expert-parallel overlap")
    assert int(_launcher_settings(overlay)["ISAMBARD_CUDA_MAX_CONNECTIONS"]) > 1


@pytest.mark.parametrize("overlay", ARM_OVERLAYS, ids=lambda path: f"{path.parent.name}-{path.stem}")
def test_a_verda_arm_with_cuda_graphs_turns_off_nccls_graph_buffer_registration(overlay, verda_user):
    """pipeline_env_activate.sh turns on expandable segments. With them, Megatron-LM refuses CUDA graphs unless
    NCCL_GRAPH_REGISTER is 0. The refusal comes at the start of training, after the job waited for its nodes and
    loaded its weights."""
    if _trained_config(overlay).model.cuda_graph_impl == "none":
        pytest.skip("the stage captures no CUDA graphs")
    assert _launcher_settings(overlay).get("NCCL_GRAPH_REGISTER") == "0"


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

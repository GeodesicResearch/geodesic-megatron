"""Unit tests for the NemotronH size configs in configs/models/nemotronh/.

Pure python, no torch. ``recipes/test_nemotronh_size_recipes.py`` checks, in the container, that the pretrain launch
path builds each size. These tests check the files themselves: that each states every sizing field, that the Nano
file is the published Nano architecture, that each smaller size follows the scaling rule of the directory's
README.md, and the parameter counts and FLOPs of each size.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
from scripts.training.config_compose import load_composed_yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SIZES_DIR = REPO_ROOT / "configs" / "models" / "nemotronh"

#: Each smaller size: its layer count, hidden size and expert count, the inputs of the scaling rule.
PRESETS = {
    "h768-l24": (24, 768, 128),
    "h1024-l31": (31, 1024, 128),
    "h1536-l38-e32": (38, 1536, 32),
    "h1536-l38": (38, 1536, 128),
    "h2048-l45": (45, 2048, 128),
}

# The architecture fields of nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16's config.json that ArchSpec.from_hf_config
# reads, copied from the published file.
NANO_HF_CONFIG = {
    "hidden_size": 2688,
    "num_hidden_layers": 52,
    "hybrid_override_pattern": "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
    "vocab_size": 131072,
    "tie_word_embeddings": False,
    "mamba_num_heads": 64,
    "mamba_head_dim": 64,
    "ssm_state_size": 128,
    "n_groups": 8,
    "conv_kernel": 4,
    "chunk_size": 128,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "n_routed_experts": 128,
    "num_experts_per_tok": 6,
    "moe_intermediate_size": 1856,
    "moe_shared_expert_intermediate_size": 3712,
    "n_shared_experts": 1,
    "intermediate_size": 1856,
    "mlp_hidden_act": "relu2",
}

#: The Olmo tokenizer (allenai/dolma2-tokenizer), ids 0 to 100,277, that the small sizes train with.
OLMO_VOCAB_SIZE = 100278

#: Total and active parameters and model FLOPs per token (forward and backward, causal attention) of each size,
#: with the Olmo tokenizer padded for tensor parallelism 4 (100,352 rows) at sequence length 8192.
COUNTS = {
    "nano": (31_412_789_568, 3_414_925_632, 20_312_432_640),
    "h768-l24": (1_327_342_224, 271_950_480, 1_304_217_600),
    "h1024-l31": (2_941_645_552, 479_064_816, 2_619_856_896),
    "h1536-l38-e32": (2_524_858_392, 1_047_545_880, 5_875_878_912),
    "h1536-l38": (7_982_059_128, 1_050_054_264, 5_890_919_424),
    "h2048-l45": (15_943_091_520, 1_871_201_600, 10_905_133_056),
}


@pytest.fixture(scope="module")
def fe():
    """Import the real estimator by path (it is a top-level script, not an installed module)."""
    spec = importlib.util.spec_from_file_location(
        "nemotronh_flops_estimator", os.path.join(REPO_ROOT, "scripts", "nemotronh_flops_estimator.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def size(name: str) -> dict:
    return load_composed_yaml(SIZES_DIR / f"{name}.yaml")["model"]


def nearest_multiple(value: float, multiple: int) -> int:
    return multiple * round(value / multiple)


def test_the_directory_holds_nano_and_each_preset():
    assert {path.stem for path in SIZES_DIR.glob("*.yaml")} == {"nano", *PRESETS}


@pytest.mark.parametrize("name", ["nano", *PRESETS])
def test_each_file_states_exactly_the_sizing_fields(fe, name):
    document = load_composed_yaml(SIZES_DIR / f"{name}.yaml")
    assert set(document) == {"model"}
    assert set(document["model"]) == set(fe.MODEL_CONFIG_FIELDS)


def test_the_nano_file_is_the_published_nano_architecture(fe):
    arch = fe.ArchSpec.from_model_config(size("nano"), vocab_size=NANO_HF_CONFIG["vocab_size"], name="nano")
    published = fe.ArchSpec.from_hf_config(NANO_HF_CONFIG, name="nano")
    assert arch == published
    assert fe.count_params(arch, 131072).total == fe.count_params(published, 131072).total


@pytest.mark.parametrize("name", PRESETS)
def test_each_preset_follows_the_scaling_rule(name):
    layers, hidden, experts = PRESETS[name]
    nano = size("nano")
    preset = size(name)
    dropped = (nano["num_layers"] - layers) // 7
    expert_ffn = nearest_multiple(hidden * nano["moe_ffn_hidden_size"] / nano["hidden_size"], 64)
    assert nano["hybrid_layer_pattern"][: 7 * dropped] == "MEMEM*E" * dropped
    assert preset == {
        **nano,
        "hidden_size": hidden,
        "num_layers": layers,
        "hybrid_layer_pattern": nano["hybrid_layer_pattern"][7 * dropped :],
        "num_moe_experts": experts,
        "moe_ffn_hidden_size": expert_ffn,
        "ffn_hidden_size": expert_ffn,
        "moe_shared_expert_intermediate_size": 2 * expert_ffn,
        "mamba_num_heads": nearest_multiple(hidden * nano["mamba_num_heads"] / nano["hidden_size"], 8),
        "num_attention_heads": nearest_multiple(hidden * nano["num_attention_heads"] / nano["hidden_size"], 8),
    }


@pytest.mark.parametrize("name", COUNTS)
def test_the_parameter_counts_and_flops_of_each_size(fe, name):
    run = fe.RunSpec(
        config_path="<sizes>",
        global_batch_size=1,
        seq_length=8192,
        recompute_granularity=None,
        recompute_modules=(),
        mtp_num_layers=0,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=8,
        vocab_size_override=None,
        make_vocab_size_divisible_by=128,
        should_pad_vocab=True,
    )
    arch = fe.ArchSpec.from_model_config(size(name), vocab_size=OLMO_VOCAB_SIZE, name=name)
    report = fe.compute_flops(arch, run)
    assert report.padded_vocab_size == 100352
    assert (report.params.total, report.params.active, report.model_flops_per_token) == COUNTS[name]

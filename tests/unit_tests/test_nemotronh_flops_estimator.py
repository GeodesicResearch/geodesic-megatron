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

"""Unit tests for scripts/nemotronh_flops_estimator.py.

Pure python — no torch, no GPU. Two kinds of check:

1. **Hand-computed small cases.** A toy architecture with dimensions small enough that
   every per-layer FLOP term can be worked out on paper and written as a literal here.
   If a formula in the estimator is edited, these fail with the arithmetic in view.

2. **Real Super-120B pins.** The parameter counts are pinned to the values read directly
   out of the HF checkpoint's safetensors headers (see `SUPER_*_PARAMS` below), and the
   exact/6ND ratio is asserted to be materially different from 1 — the whole reason the
   estimator exists.
"""

import importlib.util
import json
import os
import re
import sys

import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUPER_CONFIG = os.path.join(REPO_ROOT, "configs", "quickstart", "nemotron_super_quickstart_sft.yaml")

# Ground truth, summed from the tensor shapes in the HF checkpoint's safetensors headers
# (nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16), excluding the MTP block, which the
# training configs disable via `mtp_num_layers: null`:
#   40 Mamba2 layers    @   109,640,064
#    8 attention layers @    35,655,680
#   40 latent-MoE layers@ 2,873,102,848  (top-22 of 512 -> 175,641,088 active)
#   embeddings + lm_head @  536,870,912 each, plus one final norm of 4,096.
SUPER_TOTAL_PARAMS = 120_668_707_840
SUPER_ACTIVE_PARAMS = 12_770_237_440
SUPER_MAMBA_LAYER_PARAMS = 109_640_064
SUPER_ATTENTION_LAYER_PARAMS = 35_655_680
SUPER_MOE_LAYER_PARAMS = 2_873_102_848


@pytest.fixture(scope="module")
def fe():
    """Import the real script by path (it is a top-level script, not an installed module)."""
    spec = importlib.util.spec_from_file_location(
        "nemotronh_flops_estimator", os.path.join(REPO_ROOT, "scripts", "nemotronh_flops_estimator.py")
    )
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the module uses `from __future__ import annotations`, and
    # dataclasses resolves string annotations via sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------
# Toy architecture: every dimension small enough to verify by hand.
# --------------------------------------------------------------------------------------

TOY_HF_CONFIG = {
    "hidden_size": 8,
    "num_hidden_layers": 3,
    "hybrid_override_pattern": "M*E",
    "vocab_size": 32,
    "tie_word_embeddings": False,
    # Mamba2: d_inner = 4 heads * 2 head_dim = 8; conv_dim = 8 + 2*2*4 = 24;
    # in_proj_out = 8 + 24 + 4 = 36.
    "mamba_num_heads": 4,
    "mamba_head_dim": 2,
    "ssm_state_size": 4,
    "n_groups": 2,
    "conv_kernel": 4,
    "chunk_size": 2,
    # Attention: 2 heads of 4, 1 kv head.
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 4,
    # MoE: top-2 of 4 experts, ffn 6, latent 3, one shared expert of ffn 5.
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 6,
    "moe_latent_size": 3,
    "moe_shared_expert_intermediate_size": 5,
    "n_shared_experts": 1,
    "intermediate_size": 6,
    "mlp_hidden_act": "relu2",
}

TOY_SEQ = 4


@pytest.fixture(scope="module")
def toy(fe):
    return fe.ArchSpec.from_hf_config(TOY_HF_CONFIG, name="toy")


def _toy_run(fe, **overrides):
    kwargs = dict(
        config_path="<toy>",
        global_batch_size=2,
        seq_length=TOY_SEQ,
        recompute_granularity=None,
        recompute_modules=(),
        mtp_num_layers=0,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        vocab_size_override=None,
        make_vocab_size_divisible_by=1,
        should_pad_vocab=False,
    )
    kwargs.update(overrides)
    return fe.RunSpec(**kwargs)


def test_toy_derived_dims(toy):
    assert toy.d_inner == 4 * 2
    assert toy.conv_dim == 8 + 2 * 2 * 4  # d_inner + 2 * n_groups * d_state
    assert toy.in_proj_out == 8 + 24 + 4  # d_inner + conv_dim + n_heads
    assert toy.layer_census() == {"M": 1, "*": 1, "E": 1, "-": 0}
    assert toy.gated_mlp is False  # relu^2 is up_proj -> act -> down_proj, two GEMMs


def test_toy_mamba_layer_flops_by_hand(fe, toy):
    f = fe.mamba_layer_flops(toy, TOY_SEQ)
    L, Q, N, P, H, G = TOY_SEQ, 2, 4, 2, 4, 2
    assert f["in_proj"] == 2 * L * 8 * 36  # 2 * 4 * 8 * 36 = 2304
    assert f["conv1d_depthwise"] == 2 * L * 24 * 4  # depthwise: K taps per channel = 768
    assert f["out_proj"] == 2 * L * 8 * 8  # 512
    assert f["ssd_intra_chunk_scores"] == 2 * L * Q * N * G  # 2*4*2*4*2 = 128
    assert f["ssd_scores_times_x"] == 2 * L * Q * P * H  # 2*4*2*2*4 = 128
    assert f["ssd_chunk_states"] == 2 * L * N * P * H  # 2*4*4*2*4 = 256
    assert f["ssd_state_to_output"] == 2 * L * N * P * H  # 256
    assert f["ssd_state_passing"] == 2 * (L / Q) * H * P * N  # 2*2*4*2*4 = 128
    assert sum(f.values()) == 2304 + 768 + 512 + 128 + 128 + 256 + 256 + 128


def test_toy_conv1d_is_depthwise_not_dense(fe, toy):
    """A dense conv would be conv_dim^2 * K; depthwise is conv_dim * K."""
    f = fe.mamba_layer_flops(toy, TOY_SEQ)
    dense_equivalent = 2 * TOY_SEQ * toy.conv_dim * toy.conv_dim * toy.conv_kernel
    assert f["conv1d_depthwise"] == dense_equivalent / toy.conv_dim


def test_toy_scan_is_linear_in_sequence(fe, toy):
    """The chunked SSD scan must scale linearly with L, unlike attention's L^2."""
    short = fe.mamba_layer_flops(toy, TOY_SEQ)
    long = fe.mamba_layer_flops(toy, TOY_SEQ * 4)
    scan_keys = [k for k in short if k.startswith("ssd_")]
    assert sum(long[k] for k in scan_keys) == 4 * sum(short[k] for k in scan_keys)


def test_toy_attention_layer_flops_by_hand(fe, toy):
    f = fe.attention_layer_flops(toy, TOY_SEQ, "causal")
    L, q_dim, kv_dim = TOY_SEQ, 2 * 4, 1 * 4
    assert f["qkv_proj"] == 2 * L * 8 * (q_dim + 2 * kv_dim)  # 2*4*8*(8+8) = 1024
    assert f["o_proj"] == 2 * L * q_dim * 8  # 2*4*8*8 = 512
    assert f["core_attn"] == 2 * L * L * q_dim  # causal QK^T + AV = 2*4*4*8 = 256


def test_toy_attention_mask_modes(fe, toy):
    causal = fe.attention_layer_flops(toy, TOY_SEQ, "causal")
    full = fe.attention_layer_flops(toy, TOY_SEQ, "full")
    none = fe.attention_layer_flops(toy, TOY_SEQ, "none")
    assert full["core_attn"] == 2 * causal["core_attn"]
    assert none["core_attn"] == 0
    assert causal["qkv_proj"] == full["qkv_proj"] == none["qkv_proj"]
    with pytest.raises(ValueError):
        fe.attention_layer_flops(toy, TOY_SEQ, "block-diagonal")


def test_toy_moe_layer_flops_by_hand(fe, toy):
    f = fe.moe_layer_flops(toy, TOY_SEQ)
    L = TOY_SEQ
    assert f["router"] == 2 * L * 8 * 4  # 2*4*8*4 = 256
    assert f["latent_down_proj"] == 2 * L * 8 * 3  # 192
    assert f["latent_up_proj"] == 2 * L * 3 * 8  # 192
    # top_k=2 experts, each up(3->6) + down(6->3) at the LATENT width, not hidden.
    assert f["experts_routed"] == 2 * L * 2 * 3 * 6 * 2  # 576
    assert f["shared_expert"] == 2 * L * 8 * 5 * 2  # 640


def test_toy_moe_counts_active_experts_only(fe, toy):
    """Doubling the expert pool at fixed top_k leaves expert FLOPs and expert params alone.

    Only the router scales with the pool — it scores every expert — so that is the sole
    term allowed to move in the active count.
    """
    wide = fe.ArchSpec.from_hf_config({**TOY_HF_CONFIG, "n_routed_experts": 8}, name="wide")
    wide_p, toy_p = fe.count_params(wide, 32), fe.count_params(toy, 32)
    assert fe.moe_layer_flops(wide, TOY_SEQ)["experts_routed"] == fe.moe_layer_flops(toy, TOY_SEQ)["experts_routed"]
    assert wide_p.moe_layer_total - toy_p.moe_layer_total == 4 * (3 * 6 * 2) + 4 * 8 + 4
    assert wide_p.moe_layer_active - toy_p.moe_layer_active == 4 * 8 + 4  # router weight + bias only


def test_toy_head_flops_by_hand(fe, toy):
    assert fe.head_flops(toy, TOY_SEQ, 32)["lm_head"] == 2 * TOY_SEQ * 8 * 32  # 2048


def test_toy_total_forward_is_the_sum_of_its_parts(fe, toy):
    report = fe.compute_flops(toy, _toy_run(fe))
    by_hand = (
        sum(fe.mamba_layer_flops(toy, TOY_SEQ).values())
        + sum(fe.attention_layer_flops(toy, TOY_SEQ).values())
        + sum(fe.moe_layer_flops(toy, TOY_SEQ).values())
        + sum(fe.head_flops(toy, TOY_SEQ, 32).values())
    )
    assert report.forward_per_sample == by_hand
    assert report.model_flops_per_sample == 3 * by_hand
    assert report.model_flops_per_iter == 3 * by_hand * 2  # global_batch_size = 2


def test_toy_gated_mlp_widens_expert_gemms(fe):
    """A SwiGLU-style config must charge 3 expert GEMMs where relu^2 charges 2."""
    gated = fe.ArchSpec.from_hf_config({**TOY_HF_CONFIG, "mlp_hidden_act": "silu"}, name="gated")
    plain = fe.ArchSpec.from_hf_config(TOY_HF_CONFIG, name="plain")
    assert gated.gated_mlp is True
    ratio = fe.moe_layer_flops(gated, TOY_SEQ)["experts_routed"] / fe.moe_layer_flops(plain, TOY_SEQ)["experts_routed"]
    assert ratio == pytest.approx(1.5)


# --------------------------------------------------------------------------------------
# Recompute accounting
# --------------------------------------------------------------------------------------


def test_recompute_off_adds_nothing(fe, toy):
    assert fe.compute_flops(toy, _toy_run(fe)).recompute_per_sample == 0


def test_recompute_moe_adds_exactly_one_moe_forward(fe, toy):
    run = _toy_run(fe, recompute_granularity="selective", recompute_modules=("moe",))
    report = fe.compute_flops(toy, run)
    assert report.recompute_per_sample == sum(fe.moe_layer_flops(toy, TOY_SEQ).values())
    assert report.hardware_flops_per_sample > report.model_flops_per_sample


def test_recompute_moe_and_shared_experts_is_a_union_not_a_sum(fe, toy):
    """ "shared_experts" is nested inside the "moe" checkpoint — it must not double-count."""
    moe_only = fe.compute_flops(
        fe.ArchSpec.from_hf_config(TOY_HF_CONFIG, "t"),
        _toy_run(fe, recompute_granularity="selective", recompute_modules=("moe",)),
    )
    both = fe.compute_flops(
        toy, _toy_run(fe, recompute_granularity="selective", recompute_modules=("moe", "shared_experts"))
    )
    assert both.recompute_per_sample == moe_only.recompute_per_sample


def test_recompute_shared_experts_alone(fe, toy):
    run = _toy_run(fe, recompute_granularity="selective", recompute_modules=("shared_experts",))
    assert fe.compute_flops(toy, run).recompute_per_sample == fe.moe_layer_flops(toy, TOY_SEQ)["shared_expert"]


def test_recompute_elementwise_modules_add_no_matmul_flops(fe, toy):
    run = _toy_run(fe, recompute_granularity="selective", recompute_modules=("moe_act", "layernorm"))
    assert fe.compute_flops(toy, run).recompute_per_sample == 0


def test_recompute_full_reruns_every_layer_but_not_the_head(fe, toy):
    run = _toy_run(fe, recompute_granularity="full")
    report = fe.compute_flops(toy, run)
    head = sum(fe.head_flops(toy, TOY_SEQ, 32).values())
    assert report.recompute_per_sample == report.forward_per_sample - head


def test_unknown_recompute_module_is_rejected(fe):
    with pytest.raises(ValueError, match="unknown recompute module"):
        fe.recompute_components(("teleportation",))


# --------------------------------------------------------------------------------------
# Parameter counting
# --------------------------------------------------------------------------------------


def test_toy_param_counts_by_hand(fe, toy):
    p = fe.count_params(toy, 32)
    # Mamba: in_proj 8*36 + conv w 24*4 + conv b 24 + (A_log,D,dt_bias) 3*4
    #        + mixer norm 8 + out_proj 8*8 + block norm 8
    assert p.mamba_layer == 8 * 36 + 24 * 4 + 24 + 3 * 4 + 8 + 8 * 8 + 8
    # Attention: q 8*8 + k 8*4 + v 8*4 + o 8*8 + block norm 8
    assert p.attention_layer == 8 * 8 + 8 * 4 + 8 * 4 + 8 * 8 + 8
    # MoE common: router 8*4 + router bias 4 + latent 2*8*3 + shared 8*5*2 + block norm 8
    moe_common = 8 * 4 + 4 + 2 * 8 * 3 + 8 * 5 * 2 + 8
    assert p.moe_layer_total == moe_common + 4 * (3 * 6 * 2)
    assert p.moe_layer_active == moe_common + 2 * (3 * 6 * 2)
    assert p.embedding == 32 * 8
    assert p.lm_head == 32 * 8
    assert p.active_non_embedding == p.active - p.embedding


def test_tied_embeddings_drop_the_lm_head_params(fe):
    tied = fe.ArchSpec.from_hf_config({**TOY_HF_CONFIG, "tie_word_embeddings": True}, name="tied")
    assert fe.count_params(tied, 32).lm_head == 0


# --------------------------------------------------------------------------------------
# The real Super-120B
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def super_arch(fe):
    try:
        cfg, _ = fe.resolve_hf_config("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16")
    except FileNotFoundError as exc:  # pragma: no cover - only off-cluster
        pytest.skip(f"Super-120B HF config not in the local HF cache: {exc}")
    return fe.ArchSpec.from_hf_config(cfg, name="super-120b")


@pytest.fixture(scope="module")
def super_report(fe, super_arch):
    return fe.compute_flops(super_arch, fe.RunSpec.from_yaml(SUPER_CONFIG))


def test_super_layer_census(super_arch):
    """The published '44 Mamba + 4 attention' framing is wrong; the pattern says otherwise."""
    census = super_arch.layer_census()
    assert census == {"M": 40, "*": 8, "E": 40, "-": 0}
    assert sum(census.values()) == super_arch.num_layers == 88


def test_super_param_counts_match_the_checkpoint(fe, super_arch):
    p = fe.count_params(super_arch, super_arch.vocab_size)
    assert p.mamba_layer == SUPER_MAMBA_LAYER_PARAMS
    assert p.attention_layer == SUPER_ATTENTION_LAYER_PARAMS
    assert p.moe_layer_total == SUPER_MOE_LAYER_PARAMS
    assert p.total == SUPER_TOTAL_PARAMS
    assert p.active == SUPER_ACTIVE_PARAMS


def test_super_active_params_justify_the_a12b_name(fe, super_arch):
    active = fe.count_params(super_arch, super_arch.vocab_size).active
    assert 11.5e9 < active < 13.5e9, f"A12B implies ~12B active, got {active / 1e9:.2f}B"


def test_super_total_params_justify_the_120b_name(fe, super_arch):
    total = fe.count_params(super_arch, super_arch.vocab_size).total
    assert 115e9 < total < 125e9


def test_super_exact_is_materially_above_6nd(super_report):
    """The consultant's point, pinned: 6ND understates this workload.

    Model FLOPs alone run ~5-10% above 6ND (parameter-free core attention and the SSD
    scan); once selective recompute is counted, the hardware actually issues ~24-29%
    more than 6ND predicts.
    """
    model_ratio = super_report.model_flops_per_iter / super_report.six_nd_per_iter()
    hw_ratio = super_report.hardware_flops_per_iter / super_report.six_nd_per_iter()
    hw_ratio_non_emb = super_report.hardware_flops_per_iter / super_report.six_nd_per_iter(non_embedding=True)
    assert 1.03 < model_ratio < 1.10, model_ratio
    assert 1.20 < hw_ratio < 1.30, hw_ratio
    assert hw_ratio_non_emb > 1.25, hw_ratio_non_emb


def test_super_the_gap_is_not_the_mamba_scan(fe, super_report, super_arch):
    """Attribution matters: the scan is <1% of forward, so 'Mamba breaks 6ND' is wrong.

    What actually breaks it is recompute, then O(seq^2) core attention.
    """
    census = super_arch.layer_census()
    fwd = super_report.forward_per_sample
    mamba = super_report.forward_by_layer_type["M"]
    scan = sum(v for k, v in mamba.items() if k.startswith("ssd_") or k.startswith("conv1d"))
    scan_share = scan * census["M"] / fwd
    attn_share = super_report.forward_by_layer_type["*"]["core_attn"] * census["*"] / fwd
    recompute_share = super_report.recompute_per_sample / super_report.model_flops_per_sample
    assert scan_share < 0.02, scan_share
    assert 0.05 < attn_share < 0.15, attn_share
    assert recompute_share > attn_share > scan_share


def test_super_agrees_with_the_in_repo_counter(fe, super_report, super_arch):
    """The estimator must stay within ~1% of the counter behind the logged TFLOP/s.

    This calls the REAL `flop_utils.num_floating_point_operations`, not a copy of its
    arithmetic, so the guard cannot pass against a stale reimplementation. It therefore
    needs the container (torch + megatron-core), which is where this suite runs.

    The two are deliberately not identical: `flop_utils` omits the depthwise conv1d and
    the router GEMM and uses a Mamba-1-style scan estimate instead of the SSD terms.
    """
    mcore = fe.megatron_counter_flops_per_iter(super_arch, super_report.run)
    assert super_report.model_flops_per_iter / mcore == pytest.approx(1.0, abs=0.01)


def test_megatron_config_shim_exposes_only_what_the_counter_reads(fe, super_arch, super_report):
    """The shim must not carry the provider hook that would bypass the generic path."""
    cfg = fe.megatron_config_shim(super_arch, super_report.run)
    assert not hasattr(cfg.model, "_get_num_floating_point_operations")
    assert getattr(cfg, "peft", None) is None  # keeps the LoRA branch out
    assert cfg.model.is_hybrid_model is True
    assert cfg.model.hybrid_layer_pattern == super_arch.layer_pattern
    assert cfg.model.moe_router_topk == super_arch.top_k
    assert cfg.model.num_query_groups == super_arch.num_kv_heads


def test_compare_megatron_is_opt_in_so_the_default_path_stays_torch_free(fe, super_report):
    """`--compare-megatron` is the only path that may import torch."""
    payload = fe.report_to_dict(super_report, [21.78], 64)
    assert "megatron_counter_flops_per_iter" not in payload
    with_compare = fe.report_to_dict(super_report, [21.78], 64, compare_megatron=True)
    assert with_compare["megatron_counter_flops_per_iter"] > 0


def test_super_champion_throughput(super_report):
    """21.78 s/iter on 64 GPUs — the shipped champion — lands near 120 TFLOP/s/GPU."""
    model_tflops = super_report.model_flops_per_iter / 21.78 / 64 / 1e12
    hw_tflops = super_report.hardware_flops_per_iter / 21.78 / 64 / 1e12
    assert model_tflops == pytest.approx(121.3, rel=0.02)
    assert hw_tflops == pytest.approx(142.4, rel=0.02)
    # 400 TFLOP/s/GPU would mean finishing the same iteration in well under 7 s.
    assert super_report.model_flops_per_iter / (400e12 * 64) == pytest.approx(6.6, rel=0.05)


def test_super_recompute_matches_the_shipped_config(super_report):
    assert super_report.run.recompute_granularity == "selective"
    assert set(super_report.run.recompute_modules) == {"moe", "shared_experts"}
    assert super_report.run.global_batch_size == 64
    assert super_report.run.seq_length == 32768


# --------------------------------------------------------------------------------------
# Config plumbing
# --------------------------------------------------------------------------------------


def test_layer_pattern_accepts_both_hf_schedule_fields(fe):
    """Super ships `hybrid_override_pattern`; Ultra ships a `layers_block_type` list."""
    assert fe.layer_pattern_from_hf_config({"hybrid_override_pattern": "M*E"}) == "M*E"
    # `|` is mcore's PP/VPP segment separator and carries no layer.
    assert fe.layer_pattern_from_hf_config({"hybrid_override_pattern": "ME|M*|E-"}) == "MEM*E-"
    assert fe.layer_pattern_from_hf_config({"layers_block_type": ["mamba", "attention", "moe", "mlp"]}) == "M*E-"
    with pytest.raises(ValueError, match="unknown layers_block_type"):
        fe.layer_pattern_from_hf_config({"layers_block_type": ["mamba", "wormhole"]})
    with pytest.raises(KeyError, match="neither"):
        fe.layer_pattern_from_hf_config({"hidden_size": 8})


def test_num_layers_falls_back_to_the_pattern_length(fe):
    """Ultra's config.json omits `num_hidden_layers` — the schedule is the source of truth."""
    cfg = {k: v for k, v in TOY_HF_CONFIG.items() if k != "num_hidden_layers"}
    cfg.pop("hybrid_override_pattern")
    cfg["layers_block_type"] = ["mamba", "attention", "moe"]
    arch = fe.ArchSpec.from_hf_config(cfg, name="no-num-layers")
    assert arch.num_layers == 3
    assert arch.layer_census() == {"M": 1, "*": 1, "E": 1, "-": 0}


def test_ultra_550b_param_counts_justify_its_name(fe):
    """A second real model, reached through the other schedule field: 550B total / A55B active."""
    try:
        cfg, _ = fe.resolve_hf_config("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16")
    except FileNotFoundError as exc:  # pragma: no cover - only off-cluster
        pytest.skip(f"Ultra-550B HF config not in the local HF cache: {exc}")
    arch = fe.ArchSpec.from_hf_config(cfg, name="ultra-550b")
    assert arch.layer_census() == {"M": 48, "*": 12, "E": 48, "-": 0}
    params = fe.count_params(arch, arch.vocab_size)
    assert 540e9 < params.total < 560e9, params.total
    assert 52e9 < params.active < 58e9, params.active


def test_run_spec_reads_the_shipped_quickstart(fe):
    run = fe.RunSpec.from_yaml(SUPER_CONFIG)
    assert run.tokens_per_iter == 64 * 32768
    assert (run.tensor_model_parallel_size, run.pipeline_model_parallel_size) == (1, 8)
    assert (run.context_parallel_size, run.expert_model_parallel_size) == (4, 4)
    assert run.mtp_num_layers == 0  # `mtp_num_layers: null` -> MTP off


def test_run_spec_requires_a_batch_size_and_sequence_length(fe, tmp_path):
    import yaml

    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"model": {"seq_length": 8}}))
    with pytest.raises(ValueError, match="global_batch_size"):
        fe.RunSpec.from_yaml(str(path))
    path.write_text(yaml.safe_dump({"train": {"global_batch_size": 4}}))
    with pytest.raises(ValueError, match="seq_length"):
        fe.RunSpec.from_yaml(str(path))


def test_vocab_padding_follows_tp_and_the_should_pad_flag(fe, toy):
    run = _toy_run(fe, make_vocab_size_divisible_by=128, tensor_model_parallel_size=2, should_pad_vocab=True)
    assert fe.padded_vocab_size(toy, run) == 256  # 32 -> next multiple of 128*2
    assert fe.padded_vocab_size(toy, _toy_run(fe)) == 32  # should_pad_vocab False
    mq = _toy_run(fe, vocab_size_override=131584, should_pad_vocab=False)
    assert fe.padded_vocab_size(toy, mq) == 131584  # the MQ vocab-extension convention


def test_resolve_hf_config_accepts_a_directory_and_a_file(fe, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(TOY_HF_CONFIG))
    for target in (str(tmp_path), str(tmp_path / "config.json")):
        cfg, path = fe.resolve_hf_config(target)
        assert cfg["hidden_size"] == 8
        assert path.endswith("config.json")


def test_resolve_hf_config_reports_where_it_looked(fe, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    with pytest.raises(FileNotFoundError, match="no config.json"):
        fe.resolve_hf_config("acme/not-a-real-model")


# --------------------------------------------------------------------------------------
# The rendered text report — the numbers quoted in the consultant tracker come from here,
# so the printed figures are asserted against the JSON path rather than eyeballed.
# --------------------------------------------------------------------------------------

PEAK = 989.4
GPUS = 64
SEC = 21.78
TARGET = 400.0


@pytest.fixture(scope="module")
def super_text_report(fe, super_report):
    return fe.format_report(
        super_report,
        seconds_per_iter=[SEC],
        gpus=GPUS,
        peak_tflops=PEAK,
        target_tflops=[TARGET],
        compare_megatron=False,
    )


def _one(pattern, text):
    """Extract the single regex match a report line must produce."""
    found = re.findall(pattern, text, flags=re.MULTILINE)
    assert len(found) == 1, f"expected exactly one match for {pattern!r}, got {found}"
    return found[0]


def test_report_arithmetic_floor_is_hardware_flops_at_peak(super_text_report, super_report):
    floor = float(_one(r"arithmetic floor: ([\d.]+) s/iter", super_text_report))
    expected = super_report.hardware_flops_per_iter / (PEAK * 1e12) / GPUS
    assert floor == pytest.approx(expected, abs=0.005)


def test_report_throughput_row_matches_the_json_path(fe, super_text_report, super_report):
    model_tf, mfu, hw_tf, hfu = (
        float(x) for x in _one(rf"^\s+{SEC}\s+([\d.]+)\s+([\d.]+)%\s+([\d.]+)\s+([\d.]+)%\s*$", super_text_report)
    )
    payload = fe.report_to_dict(super_report, [SEC], GPUS)["throughput"][0]
    assert model_tf == pytest.approx(payload["model_tflops_per_gpu"], abs=0.05)
    assert hw_tf == pytest.approx(payload["hardware_tflops_per_gpu"], abs=0.05)
    assert mfu == pytest.approx(100 * payload["model_tflops_per_gpu"] / PEAK, abs=0.05)
    assert hfu == pytest.approx(100 * payload["hardware_tflops_per_gpu"] / PEAK, abs=0.05)
    # And the champion figures the tracker doc quotes.
    assert model_tf == pytest.approx(121.3, rel=0.02)
    assert hw_tf == pytest.approx(142.4, rel=0.02)


def test_report_non_arithmetic_remainder_is_the_iteration_minus_the_floor(super_text_report):
    floor = float(_one(r"arithmetic floor: ([\d.]+) s/iter", super_text_report))
    secs, pct = (float(x) for x in _one(r"-> ([\d.]+) s/iter \(([\d.]+)%\) is NOT tensor-core", super_text_report))
    assert secs == pytest.approx(SEC - floor, abs=0.02)
    assert pct == pytest.approx(100 * (1 - floor / SEC), abs=0.2)


def test_report_target_table_inverts_the_throughput_arithmetic(super_text_report, super_report):
    model_sec, hw_sec, mfu = (
        float(x) for x in _one(rf"^\s+{int(TARGET)}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s*$", super_text_report)
    )
    assert model_sec == pytest.approx(super_report.model_flops_per_iter / (TARGET * 1e12) / GPUS, abs=0.01)
    assert hw_sec == pytest.approx(super_report.hardware_flops_per_iter / (TARGET * 1e12) / GPUS, abs=0.01)
    assert mfu == pytest.approx(100 * TARGET / PEAK, abs=0.05)
    assert model_sec == pytest.approx(6.60, rel=0.02)  # the "is 400 TFLOP/s reachable" number


def test_report_states_the_layer_census_and_params(super_text_report):
    assert "40 Mamba2 + 8 attention + 40 latent-MoE" in super_text_report
    assert "120.669 B" in super_text_report  # total
    assert "12.770 B" in super_text_report  # active
    assert "mask=causal" in super_text_report


def test_report_omits_the_cross_check_section_unless_asked(super_text_report):
    assert "CROSS-CHECK" not in super_text_report


def test_report_without_gpus_omits_the_throughput_sections(fe, super_report):
    text = fe.format_report(
        super_report, seconds_per_iter=[], gpus=None, peak_tflops=PEAK, target_tflops=[], compare_megatron=False
    )
    assert "arithmetic floor" not in text
    assert "ACHIEVED THROUGHPUT" not in text
    assert "PARAMETERS" in text  # the config-only part still renders


def test_cli_json_output_is_self_consistent(fe, capsys):
    try:
        fe.resolve_hf_config("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16")
    except FileNotFoundError as exc:  # pragma: no cover - only off-cluster
        pytest.skip(f"Super-120B HF config not in the local HF cache: {exc}")
    rc = fe.main(
        [
            SUPER_CONFIG,
            "--hf-model",
            "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
            "--seconds-per-iter",
            "21.78",
            "--gpus",
            "64",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["params_active"] == SUPER_ACTIVE_PARAMS
    assert payload["hardware_flops_per_iter"] > payload["model_flops_per_iter"]
    assert payload["exact_over_6nd"] == pytest.approx(payload["model_flops_per_iter"] / payload["six_nd_per_iter"])
    assert payload["throughput"][0]["model_tflops_per_gpu"] == pytest.approx(121.3, rel=0.02)

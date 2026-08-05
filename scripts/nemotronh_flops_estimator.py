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

"""Exact per-layer FLOP estimator for NemotronH hybrid (Mamba2 + attention + latent-MoE) models.

WHY THIS EXISTS
---------------
The universal `6ND` rule of thumb (N = active params, D = tokens) is a dense-transformer
approximation. It charges every parameter 6 FLOPs per token and charges nothing for the
parameter-free work, so on this architecture it misses four things at once:

  1. **Core attention** — O(seq^2) work with no parameters. At seq 32768 with only 8
     attention layers this is still ~8% of the forward pass.
  2. **The Mamba2 selective scan** — parameter-free state-space work (chunked SSD).
     Linear in sequence but carrying `d_state` / `chunk_size` factors.
  3. **The depthwise conv1d** — parameters that do K FLOPs/token each, not 2.
  4. **Recompute** — selective activation recomputation re-executes module forwards
     during the backward pass. Those FLOPs are real work the GPU does; `6ND` has no
     term for them.

This module computes each of those explicitly, from the model's own HF `config.json`
plus the training YAML, so throughput and MFU claims are auditable rather than folklore.
It is the NemotronH-hybrid analogue of EleutherAI's `cookbook/calc/calc_transformer_flops.py`,
which covers dense transformers only.

CONVENTIONS (stated so numbers are comparable to other people's)
----------------------------------------------------------------
* **A multiply-accumulate is 2 FLOPs.** A GEMM (M,K)x(K,N) is `2*M*K*N`.
* **Matmul-class work only.** Softmax, layernorms, SiLU/gating, the `dt` softplus,
  the routing top-k, and the optimizer are all excluded — the same convention `6ND`,
  the EleutherAI cookbook, and Megatron's own counter use. They are elementwise or
  memory-bound and contribute well under 1% of the FLOP total (they contribute a great
  deal more than that to the *runtime*, which is exactly why MFU here is low).
* **Backward = 2x forward** (dgrad + wgrad). Configurable via `--backward-multiplier`.
  Real flash-attention backward is nearer 2.5x forward, so the attention term is mildly
  conservative.
* **Active FLOPs only for the MoE.** A token visits `top_k` of `n_routed_experts`, so
  only `top_k` experts' GEMMs are counted. The shared expert is visited by every token.
* Two totals are reported, and they answer different questions:
    - **model FLOPs** = fwd + bwd. The arithmetic the model *definition* requires.
      Divide by time for **MFU**. This is the number to compare against other models.
    - **hardware FLOPs** = model FLOPs + recompute. The arithmetic this *configuration*
      actually issues. Divide by time for **HFU**. This is the number to compare
      against the GPU's peak when asking "how much of the machine am I using".

PER-LAYER-TYPE FORMULAS (forward, per sample of L tokens)
---------------------------------------------------------
Symbols: h = hidden_size, L = seq_length, V = (padded) vocab_size.

MAMBA2 layer (`M` in `hybrid_override_pattern`)
    d_inner   = mamba_num_heads * mamba_head_dim          (= expand * h on this model)
    conv_dim  = d_inner + 2 * n_groups * d_state
    proj_out  = d_inner + conv_dim + mamba_num_heads      (z, xBC, dt packed in one GEMM)
    in_proj        2 * L * h * proj_out
    conv1d         2 * L * conv_dim * conv_kernel         (DEPTHWISE: groups = conv_dim,
                                                           so it is K taps per channel,
                                                           NOT a conv_dim x conv_dim GEMM)
    out_proj       2 * L * d_inner * h

    Selective scan, chunked SSD form (Mamba-2 paper Thm 3.5 / `mamba_chunk_scan_combined`),
    with chunk Q = chunk_size, H heads, P = head_dim, N = d_state, G = n_groups:
      intra-chunk scores  C B^T   2 * L * Q * N * G   ((Q,N)x(N,Q) per chunk per GROUP;
                                                       B/C are shared across the H/G heads
                                                       in a group, so this is per-group)
      scores @ X                  2 * L * Q * P * H   ((Q,Q)x(Q,P) per chunk per head)
      chunk states B^T X          2 * L * N * P * H   ((N,Q)x(Q,P) per chunk per head)
      state -> output C S         2 * L * N * P * H   ((Q,N)x(N,P) per chunk per head)
      inter-chunk recurrence      2 * (L/Q) * H * P * N   (elementwise decay+accumulate on
                                                           the carried (H,P,N) state)
    Note the scan is LINEAR in L (the Q^2 of the intra-chunk block is amortised by the
    L/Q chunks) and is only ~3% of the Mamba layer's FLOPs — Mamba2's cost is bandwidth,
    not arithmetic. The dense-attention-mask form of the scan is NOT halved for causality
    here because the kernel materialises the full QxQ block and masks it.

ATTENTION layer (`*`)
    q,k,v,o projections   2 * L * h * (n_heads*d_head + 2*n_kv_heads*d_head + n_heads*d_head)
    core attention        2 * L^2 * n_heads * d_head      (causal: QK^T and AV are each
                                                           2*L^2*n_heads*d_head/2)
    `--attention-mask full` drops the causal halving (2x this term).

LATENT-MoE layer (`E`)
    router GEMM        2 * L * h * n_routed_experts
    fc1_latent_proj    2 * L * h * latent
    routed experts     2 * L * top_k * (latent*ffn + ffn*latent) = 4 * L * top_k * latent * ffn
                       (NemotronH experts are up_proj -> relu^2 -> down_proj: TWO GEMMs,
                        not the three of a SwiGLU MLP. Experts run at `moe_latent_size`,
                        not at `hidden_size` — that latent compression is most of why
                        A12B is only ~12B active out of 120B.)
    fc2_latent_proj    2 * L * latent * h
    shared expert      4 * L * h * shared_ffn             (runs at h, not latent)

HEAD
    lm_head            2 * L * h * V

RECOMPUTE
    `recompute_granularity: selective` + `recompute_modules: [...]` re-runs the listed
    module forwards during backward. Component coverage (see
    `3rdparty/Megatron-LM/megatron/core/transformer/moe/moe_layer.py::forward`, where
    `moe_layer_recompute` wraps `custom_forward`):
      "moe"            -> router + fc1_latent + routed experts + fc2_latent + shared expert
                          (the whole MoE layer; the shared expert IS inside the wrap)
      "shared_experts" -> shared expert only  (redundant when "moe" is also listed —
                          this estimator takes the UNION, i.e. counts each covered
                          component exactly one extra forward. mcore actually nests the
                          checkpoints, which can run the shared expert a third time; that
                          nesting is a config smell, not something to model.)
      "core_attn"      -> core attention only
      "mlp"            -> dense MLP layers ('-' in the pattern)
      "moe_act", "layernorm", "gdn_norm_out" -> elementwise; 0 matmul FLOPs
    `recompute_granularity: full` re-runs every layer's forward (one extra forward pass).

CAVEAT — PACKED SEQUENCES
-------------------------
With `dataset.packed_sequence_specs`, a 32768-token sample is a *pack* of many shorter
documents and attention is block-diagonal per document (`cu_seqlens`), so true core-attention
FLOPs are far below the full-causal figure. This estimator reports full-causal because that
is what Megatron's own counter and every published TFLOP/s number assume; if anything it
therefore *overstates* achieved TFLOP/s on packed data. `--attention-mask none` gives the
lower bound with core attention removed entirely.

CROSS-CHECK
-----------
`--compare-megatron` **calls** the in-repo counter
(`src/megatron/bridge/training/utils/flop_utils.py::num_floating_point_operations`) and
prints its answer beside the exact one — it is not a reimplementation, so the comparison
cannot silently drift when that function changes. The counter is what produces the
`TFLOP/s/GPU` in the training logs, so the delta tells you how much to trust those logs.
Known gaps in it: no conv1d term, no router GEMM, and a Mamba-1-style `7*L*d_inner*d_state`
scan estimate instead of the SSD terms above (its own comment flags this).

Because that function imports torch and megatron-core, `--compare-megatron` only works
inside the container and fails loudly outside it. Every other path of this script is
dependency-free apart from PyYAML, and runs on a login node.

USAGE
-----
    python3 scripts/nemotronh_flops_estimator.py \\
        configs/quickstart/nemotron_super_quickstart_sft.yaml \\
        --hf-model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \\
        --seconds-per-iter 31.562 --gpus 64 --target-tflops 400

    # --compare-megatron additionally needs the container:
    ./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh || exit 1; \\
        python scripts/nemotronh_flops_estimator.py <config> --hf-model <id> --compare-megatron"

`--hf-model` is REQUIRED and takes an HF repo id (resolved offline from the local HF
cache), a directory containing `config.json`, or a path to a `config.json`. There is
deliberately no checkpoint-path-to-model-id guessing table — the same reason
`pipeline_checkpoint_convert_hf.py` requires it.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any


# GH200 / H100 SXM dense BF16 tensor-core peak, TFLOP/s per GPU (no sparsity).
DEFAULT_PEAK_TFLOPS = 989.4

# Where to look for a repo-id's config.json when it is not a local path.
_DEFAULT_HF_CACHE = "/projects/a5k/public/hf"

# Activations whose MLP form is gated (up_proj is 2x wide + an elementwise product).
# NemotronH uses `relu2`, which is not gated: up_proj -> act -> down_proj, two GEMMs.
_GATED_ACTS = frozenset({"silu", "swiglu", "geglu", "gelu_glu", "swish_glu"})

# Layer-type characters in `hybrid_override_pattern` (mcore's
# ssm/mamba_hybrid_layer_allocation.py alphabet: M mamba, * attention, E MoE, - dense MLP).
MAMBA, ATTENTION, MOE, MLP = "M", "*", "E", "-"


# --------------------------------------------------------------------------------------
# Specs
# --------------------------------------------------------------------------------------


# NemotronH ships the layer schedule under two field names: Super uses the compact
# `hybrid_override_pattern` string, Ultra uses a `layers_block_type` list of words.
_BLOCK_TYPE_TO_CHAR = {"mamba": MAMBA, "attention": ATTENTION, "moe": MOE, "mlp": MLP}


def layer_pattern_from_hf_config(cfg: dict[str, Any]) -> str:
    """Normalise either layer-schedule field to the compact pattern string."""
    if cfg.get("hybrid_override_pattern"):
        # `|` is the mcore pipeline/VPP segment separator; it carries no layer.
        return str(cfg["hybrid_override_pattern"]).replace("|", "")
    block_types = cfg.get("layers_block_type")
    if block_types:
        unknown = {b for b in block_types if b not in _BLOCK_TYPE_TO_CHAR}
        if unknown:
            raise ValueError(f"unknown layers_block_type entries: {sorted(unknown)}")
        return "".join(_BLOCK_TYPE_TO_CHAR[b] for b in block_types)
    raise KeyError("config.json has neither 'hybrid_override_pattern' nor 'layers_block_type'")


@dataclass(frozen=True)
class ArchSpec:
    """Architecture dimensions, read from an HF `config.json`."""

    name: str
    hidden_size: int
    num_layers: int
    layer_pattern: str
    vocab_size: int
    tie_word_embeddings: bool
    # Mamba2
    mamba_num_heads: int
    mamba_head_dim: int
    mamba_state_dim: int
    mamba_num_groups: int
    conv_kernel: int
    chunk_size: int
    # Attention
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    # MoE / MLP
    n_routed_experts: int
    top_k: int
    moe_ffn_hidden_size: int
    moe_latent_size: int | None
    shared_expert_ffn_hidden_size: int
    ffn_hidden_size: int
    gated_mlp: bool

    @property
    def d_inner(self) -> int:
        """Mamba inner (expanded) width."""
        return self.mamba_num_heads * self.mamba_head_dim

    @property
    def conv_dim(self) -> int:
        """Depthwise-conv channel count: x plus the B and C group projections."""
        return self.d_inner + 2 * self.mamba_num_groups * self.mamba_state_dim

    @property
    def in_proj_out(self) -> int:
        """Packed Mamba input projection width: z, xBC, and dt in one GEMM."""
        return self.d_inner + self.conv_dim + self.mamba_num_heads

    def layer_census(self) -> dict[str, int]:
        """Count of each layer type in the pattern."""
        return {c: self.layer_pattern.count(c) for c in (MAMBA, ATTENTION, MOE, MLP)}

    @classmethod
    def from_hf_config(cls, cfg: dict[str, Any], name: str) -> "ArchSpec":
        """Build from a parsed HF `config.json` (NemotronH field names)."""
        pattern = layer_pattern_from_hf_config(cfg)
        hidden = int(cfg["hidden_size"])
        latent = cfg.get("moe_latent_size")
        act = str(cfg.get("mlp_hidden_act", "relu2")).lower()
        return cls(
            name=name,
            num_layers=int(cfg.get("num_hidden_layers") or len(pattern)),
            hidden_size=hidden,
            layer_pattern=pattern,
            vocab_size=int(cfg["vocab_size"]),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
            mamba_num_heads=int(cfg["mamba_num_heads"]),
            mamba_head_dim=int(cfg["mamba_head_dim"]),
            mamba_state_dim=int(cfg["ssm_state_size"]),
            mamba_num_groups=int(cfg["n_groups"]),
            conv_kernel=int(cfg["conv_kernel"]),
            chunk_size=int(cfg["chunk_size"]),
            num_attention_heads=int(cfg["num_attention_heads"]),
            num_kv_heads=int(cfg.get("num_key_value_heads", cfg["num_attention_heads"])),
            head_dim=int(cfg.get("head_dim") or hidden // int(cfg["num_attention_heads"])),
            n_routed_experts=int(cfg.get("n_routed_experts", 0)),
            top_k=int(cfg.get("num_experts_per_tok", 0)),
            moe_ffn_hidden_size=int(cfg.get("moe_intermediate_size", 0)),
            moe_latent_size=int(latent) if latent else None,
            shared_expert_ffn_hidden_size=int(cfg.get("moe_shared_expert_intermediate_size", 0))
            * int(cfg.get("n_shared_experts", 1) or 1),
            ffn_hidden_size=int(cfg.get("intermediate_size", 0)),
            gated_mlp=act in _GATED_ACTS,
        )


@dataclass(frozen=True)
class RunSpec:
    """The workload, read from a training YAML."""

    config_path: str
    global_batch_size: int
    seq_length: int
    recompute_granularity: str | None
    recompute_modules: tuple[str, ...]
    mtp_num_layers: int
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    context_parallel_size: int
    expert_model_parallel_size: int
    vocab_size_override: int | None
    make_vocab_size_divisible_by: int
    should_pad_vocab: bool

    @property
    def tokens_per_iter(self) -> int:
        return self.global_batch_size * self.seq_length

    @classmethod
    def from_yaml(cls, path: str) -> "RunSpec":
        import yaml  # deferred: keeps `--help` and imports working without PyYAML

        with open(path) as fh:
            cfg = yaml.safe_load(fh) or {}
        model = cfg.get("model") or {}
        train = cfg.get("train") or {}
        dataset = cfg.get("dataset") or {}

        seq_length = model.get("seq_length") or dataset.get("seq_length")
        if seq_length is None:
            raise ValueError(f"{path}: neither model.seq_length nor dataset.seq_length is set")
        gbs = train.get("global_batch_size")
        if gbs is None:
            raise ValueError(f"{path}: train.global_batch_size is not set")

        modules = model.get("recompute_modules") or []
        if isinstance(modules, str):
            modules = [modules]
        return cls(
            config_path=path,
            global_batch_size=int(gbs),
            seq_length=int(seq_length),
            recompute_granularity=model.get("recompute_granularity"),
            recompute_modules=tuple(str(m) for m in modules),
            mtp_num_layers=int(model.get("mtp_num_layers") or 0),
            tensor_model_parallel_size=int(model.get("tensor_model_parallel_size") or 1),
            pipeline_model_parallel_size=int(model.get("pipeline_model_parallel_size") or 1),
            context_parallel_size=int(model.get("context_parallel_size") or 1),
            expert_model_parallel_size=int(model.get("expert_model_parallel_size") or 1),
            vocab_size_override=(int(model["vocab_size"]) if model.get("vocab_size") else None),
            make_vocab_size_divisible_by=int(model.get("make_vocab_size_divisible_by") or 128),
            should_pad_vocab=bool(model.get("should_pad_vocab", True)),
        )


# --------------------------------------------------------------------------------------
# Parameter counts
# --------------------------------------------------------------------------------------


@dataclass
class ParamCounts:
    """Total and active parameter counts, per layer type and overall."""

    mamba_layer: int
    attention_layer: int
    moe_layer_total: int
    moe_layer_active: int
    mlp_layer: int
    embedding: int
    lm_head: int
    final_norm: int
    census: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return (
            self.census.get(MAMBA, 0) * self.mamba_layer
            + self.census.get(ATTENTION, 0) * self.attention_layer
            + self.census.get(MOE, 0) * self.moe_layer_total
            + self.census.get(MLP, 0) * self.mlp_layer
            + self.embedding
            + self.lm_head
            + self.final_norm
        )

    @property
    def active(self) -> int:
        return (
            self.census.get(MAMBA, 0) * self.mamba_layer
            + self.census.get(ATTENTION, 0) * self.attention_layer
            + self.census.get(MOE, 0) * self.moe_layer_active
            + self.census.get(MLP, 0) * self.mlp_layer
            + self.embedding
            + self.lm_head
            + self.final_norm
        )

    @property
    def active_non_embedding(self) -> int:
        """Active params excluding the embedding table (which does a gather, not a GEMM)."""
        return self.active - self.embedding


def count_params(arch: ArchSpec, vocab_size: int) -> ParamCounts:
    """Exact parameter counts, matching the HF checkpoint's tensor shapes."""
    h = arch.hidden_size

    mamba = (
        h * arch.in_proj_out  # in_proj
        + arch.conv_dim * arch.conv_kernel  # depthwise conv weight
        + arch.conv_dim  # conv bias
        + 3 * arch.mamba_num_heads  # A_log, D, dt_bias
        + arch.d_inner  # gated RMSNorm inside the mixer
        + arch.d_inner * h  # out_proj
        + h  # block input norm
    )

    attention = (
        h * arch.num_attention_heads * arch.head_dim  # q_proj
        + 2 * h * arch.num_kv_heads * arch.head_dim  # k_proj, v_proj
        + arch.num_attention_heads * arch.head_dim * h  # o_proj
        + h  # block input norm
    )

    latent = arch.moe_latent_size or h
    gate_mult = 2 if arch.gated_mlp else 1
    per_expert = latent * arch.moe_ffn_hidden_size * (1 + gate_mult)
    moe_common = (
        h * arch.n_routed_experts  # router weight
        + arch.n_routed_experts  # router expert-bias
        + (2 * h * latent if arch.moe_latent_size else 0)  # fc1/fc2 latent projections
        + h * arch.shared_expert_ffn_hidden_size * (1 + gate_mult)  # shared expert
        + h  # block input norm
    )
    moe_total = moe_common + arch.n_routed_experts * per_expert
    moe_active = moe_common + arch.top_k * per_expert

    mlp = h * arch.ffn_hidden_size * (1 + gate_mult) + h

    embedding = vocab_size * h
    lm_head = 0 if arch.tie_word_embeddings else vocab_size * h

    return ParamCounts(
        mamba_layer=mamba,
        attention_layer=attention,
        moe_layer_total=moe_total,
        moe_layer_active=moe_active,
        mlp_layer=mlp,
        embedding=embedding,
        lm_head=lm_head,
        final_norm=h,
        census=arch.layer_census(),
    )


# --------------------------------------------------------------------------------------
# Forward FLOPs, per layer type, per sample of `seq_len` tokens
# --------------------------------------------------------------------------------------


def mamba_layer_flops(arch: ArchSpec, seq_len: int) -> dict[str, float]:
    """Forward FLOPs of one Mamba2 layer. See module docstring for the derivation."""
    h, L = arch.hidden_size, seq_len
    d_inner, conv_dim = arch.d_inner, arch.conv_dim
    Q, N, P, H, G = (
        arch.chunk_size,
        arch.mamba_state_dim,
        arch.mamba_head_dim,
        arch.mamba_num_heads,
        arch.mamba_num_groups,
    )
    return {
        "in_proj": 2.0 * L * h * arch.in_proj_out,
        "conv1d_depthwise": 2.0 * L * conv_dim * arch.conv_kernel,
        "ssd_intra_chunk_scores": 2.0 * L * Q * N * G,
        "ssd_scores_times_x": 2.0 * L * Q * P * H,
        "ssd_chunk_states": 2.0 * L * N * P * H,
        "ssd_state_to_output": 2.0 * L * N * P * H,
        "ssd_state_passing": 2.0 * (L / Q) * H * P * N,
        "out_proj": 2.0 * L * d_inner * h,
    }


def attention_layer_flops(arch: ArchSpec, seq_len: int, mask: str = "causal") -> dict[str, float]:
    """Forward FLOPs of one attention layer. `mask` is causal | full | none."""
    h, L = arch.hidden_size, seq_len
    q_dim = arch.num_attention_heads * arch.head_dim
    kv_dim = arch.num_kv_heads * arch.head_dim
    if mask == "none":
        core_factor = 0.0
    elif mask == "full":
        core_factor = 2.0
    elif mask == "causal":
        core_factor = 1.0
    else:
        raise ValueError(f"unknown attention mask mode: {mask!r}")
    return {
        "qkv_proj": 2.0 * L * h * (q_dim + 2 * kv_dim),
        "o_proj": 2.0 * L * q_dim * h,
        # QK^T and AV are 2*L^2*q_dim each; causal halves both -> 2*L^2*q_dim total.
        "core_attn": core_factor * 2.0 * L * L * q_dim,
    }


def moe_layer_flops(arch: ArchSpec, seq_len: int) -> dict[str, float]:
    """Forward FLOPs of one latent-MoE layer, counting ACTIVE experts only."""
    h, L = arch.hidden_size, seq_len
    latent = arch.moe_latent_size or h
    gate_mult = 2 if arch.gated_mlp else 1
    out = {
        "router": 2.0 * L * h * arch.n_routed_experts,
        "experts_routed": 2.0 * L * arch.top_k * latent * arch.moe_ffn_hidden_size * (1 + gate_mult),
        "shared_expert": 2.0 * L * h * arch.shared_expert_ffn_hidden_size * (1 + gate_mult),
    }
    if arch.moe_latent_size:
        out["latent_down_proj"] = 2.0 * L * h * latent
        out["latent_up_proj"] = 2.0 * L * latent * h
    return out


def mlp_layer_flops(arch: ArchSpec, seq_len: int) -> dict[str, float]:
    """Forward FLOPs of one dense MLP layer ('-' in the pattern)."""
    gate_mult = 2 if arch.gated_mlp else 1
    return {"mlp": 2.0 * seq_len * arch.hidden_size * arch.ffn_hidden_size * (1 + gate_mult)}


def head_flops(arch: ArchSpec, seq_len: int, vocab_size: int) -> dict[str, float]:
    """Forward FLOPs of the output projection. The embedding lookup is a gather: 0 FLOPs."""
    return {"lm_head": 2.0 * seq_len * arch.hidden_size * vocab_size}


# --------------------------------------------------------------------------------------
# Recompute
# --------------------------------------------------------------------------------------

# mcore recompute-module name -> the per-layer components it re-runs.
RECOMPUTE_COVERAGE: dict[str, dict[str, tuple[str, ...]]] = {
    "moe": {MOE: ("router", "latent_down_proj", "experts_routed", "latent_up_proj", "shared_expert")},
    "shared_experts": {MOE: ("shared_expert",)},
    "core_attn": {ATTENTION: ("core_attn",)},
    "mlp": {MLP: ("mlp",)},
    # Elementwise-only: no matmul FLOPs to recompute.
    "moe_act": {},
    "layernorm": {},
    "gdn_norm_out": {},
    "mla_up_proj": {},
}


def recompute_components(modules: tuple[str, ...]) -> dict[str, set[str]]:
    """Union of per-layer components re-run under `recompute_modules`.

    Union rather than sum: listing both "moe" and "shared_experts" nests one checkpoint
    inside another rather than adding a second independent recompute pass.
    """
    covered: dict[str, set[str]] = {}
    for module in modules:
        if module not in RECOMPUTE_COVERAGE:
            raise ValueError(f"unknown recompute module {module!r}; known: {sorted(RECOMPUTE_COVERAGE)}")
        for layer_type, comps in RECOMPUTE_COVERAGE[module].items():
            covered.setdefault(layer_type, set()).update(comps)
    return covered


# --------------------------------------------------------------------------------------
# Whole-model roll-up
# --------------------------------------------------------------------------------------


@dataclass
class FlopReport:
    """Everything the CLI prints, in machine-readable form."""

    arch: ArchSpec
    run: RunSpec
    params: ParamCounts
    padded_vocab_size: int
    forward_by_layer_type: dict[str, dict[str, float]]
    forward_per_sample: float
    recompute_per_sample: float
    backward_multiplier: float
    attention_mask: str

    @property
    def model_flops_per_sample(self) -> float:
        """fwd + bwd — what the model definition requires."""
        return self.forward_per_sample * (1.0 + self.backward_multiplier)

    @property
    def hardware_flops_per_sample(self) -> float:
        """fwd + bwd + recompute — what this configuration actually issues."""
        return self.model_flops_per_sample + self.recompute_per_sample

    @property
    def model_flops_per_iter(self) -> float:
        return self.model_flops_per_sample * self.run.global_batch_size

    @property
    def hardware_flops_per_iter(self) -> float:
        return self.hardware_flops_per_sample * self.run.global_batch_size

    @property
    def model_flops_per_token(self) -> float:
        return self.model_flops_per_sample / self.run.seq_length

    def six_nd_per_iter(self, non_embedding: bool = False) -> float:
        n = self.params.active_non_embedding if non_embedding else self.params.active
        return 6.0 * n * self.run.tokens_per_iter


def compute_flops(
    arch: ArchSpec,
    run: RunSpec,
    attention_mask: str = "causal",
    backward_multiplier: float = 2.0,
) -> FlopReport:
    """Roll per-layer formulas up to a whole-iteration report."""
    vocab = padded_vocab_size(arch, run)
    seq = run.seq_length
    census = arch.layer_census()

    per_type = {
        MAMBA: mamba_layer_flops(arch, seq),
        ATTENTION: attention_layer_flops(arch, seq, attention_mask),
        MOE: moe_layer_flops(arch, seq),
        MLP: mlp_layer_flops(arch, seq),
    }
    head = head_flops(arch, seq, vocab)

    # MTP adds, per depth, one extra TRANSFORMER LAYER as well as one extra logits
    # projection. The layer's type follows the model's LAST layer, which is how the counter
    # this script cross-checks against models it (`flop_utils.py`:
    # `num_moe_layers += last_layer_is_moe * mtp_num_layers`, and `num_layers` grows to
    # match). Counting only the logits — as an earlier version did — silently undercounts
    # every MTP-enabled config and would put --compare-megatron permanently in disagreement.
    # `mtp_num_layers: null` (this repo's SFT configs) means MTP is off, so mtp_depth == 0
    # and both terms below vanish.
    mtp_depth = run.mtp_num_layers
    if mtp_depth:
        census = dict(census)
        census[arch.layer_pattern[-1]] += mtp_depth

    forward = sum(census[t] * sum(per_type[t].values()) for t in per_type)
    forward += sum(head.values()) * (1 + mtp_depth)

    covered = recompute_components(run.recompute_modules) if run.recompute_granularity == "selective" else {}
    if run.recompute_granularity == "full":
        recompute = forward - sum(head.values()) * (1 + mtp_depth)
    else:
        recompute = sum(
            census[t] * sum(v for k, v in per_type[t].items() if k in comps) for t, comps in covered.items()
        )

    return FlopReport(
        arch=arch,
        run=run,
        params=count_params(arch, vocab),
        padded_vocab_size=vocab,
        forward_by_layer_type=per_type,
        forward_per_sample=forward,
        recompute_per_sample=recompute,
        backward_multiplier=backward_multiplier,
        attention_mask=attention_mask,
    )


def padded_vocab_size(arch: ArchSpec, run: RunSpec) -> int:
    """Vocab size as the model actually allocates it (YAML override, then TP padding)."""
    vocab = run.vocab_size_override or arch.vocab_size
    if not run.should_pad_vocab:
        return vocab
    multiple = run.make_vocab_size_divisible_by * run.tensor_model_parallel_size
    return int(math.ceil(vocab / multiple) * multiple)


# --------------------------------------------------------------------------------------
# Cross-check against the REAL in-repo counter
# --------------------------------------------------------------------------------------


def megatron_config_shim(arch: ArchSpec, run: RunSpec) -> SimpleNamespace:
    """Duck-typed stand-in for a `ConfigContainer`, carrying only what the counter reads.

    `num_floating_point_operations` reaches into `cfg.model` for the hybrid-model branch.
    Every attribute set below is one that function actually reads; nothing else is needed
    because building a real `ConfigContainer` would require instantiating a model provider.

    Two deliberate omissions:
      * no `_get_num_floating_point_operations` — its presence is an early-exit that would
        dispatch to a provider-specific override instead of the generic path we want.
      * no `peft` on the container — `getattr(cfg, "peft", None)` then yields None, so the
        LoRA branch is skipped.
    """
    model = SimpleNamespace(
        is_hybrid_model=True,
        hybrid_layer_pattern=arch.layer_pattern,
        seq_length=run.seq_length,
        hidden_size=arch.hidden_size,
        num_attention_heads=arch.num_attention_heads,
        num_query_groups=arch.num_kv_heads,
        kv_channels=arch.head_dim,
        ffn_hidden_size=arch.ffn_hidden_size,
        gated_linear_unit=arch.gated_mlp,
        moe_ffn_hidden_size=arch.moe_ffn_hidden_size,
        moe_latent_size=arch.moe_latent_size,
        moe_shared_expert_intermediate_size=arch.shared_expert_ffn_hidden_size,
        moe_router_topk=arch.top_k,
        mamba_state_dim=arch.mamba_state_dim,
        mamba_head_dim=arch.mamba_head_dim,
        mamba_num_groups=arch.mamba_num_groups,
        mamba_num_heads=arch.mamba_num_heads,
        vocab_size=run.vocab_size_override or arch.vocab_size,
        make_vocab_size_divisible_by=run.make_vocab_size_divisible_by,
        tensor_model_parallel_size=run.tensor_model_parallel_size,
        mtp_num_layers=run.mtp_num_layers,
    )
    return SimpleNamespace(model=model)


def megatron_counter_flops_per_iter(arch: ArchSpec, run: RunSpec) -> float:
    """Call the REAL counter behind the logged `TFLOP/s/GPU`, for cross-check.

    This invokes `megatron.bridge.training.utils.flop_utils.num_floating_point_operations`
    itself rather than a copy of its arithmetic, so the comparison cannot silently drift
    when that function changes.

    It imports torch and megatron-core, hence the deferred import: the default (no
    `--compare-megatron`) path of this script stays dependency-free and host-runnable.

    One known divergence to keep in mind when reading the delta: the real counter always
    pads the vocab to `make_vocab_size_divisible_by * TP`, whereas this estimator honours
    `should_pad_vocab: false`. No shipped config is affected (131072 and the MQ 131584 are
    both already multiples of 128).
    """
    try:
        from megatron.bridge.training.utils.flop_utils import num_floating_point_operations
    except Exception as exc:  # ImportError, or a torch/CUDA load failure on the host
        raise RuntimeError(
            "--compare-megatron calls the real megatron.bridge flop counter, which imports "
            "torch and megatron-core. Run it inside the container:\n"
            '  ./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh || exit 1; '
            f'python scripts/{os.path.basename(__file__)} <config> --hf-model <id> --compare-megatron"\n'
            f"(import failed: {exc})"
        ) from exc

    return float(num_floating_point_operations(megatron_config_shim(arch, run), batch_size=run.global_batch_size))


# --------------------------------------------------------------------------------------
# HF config resolution (offline)
# --------------------------------------------------------------------------------------


def resolve_hf_config(hf_model: str) -> tuple[dict[str, Any], str]:
    """Load a `config.json` from a path or an HF repo id, without touching the network."""
    candidates: list[str] = []
    if os.path.isfile(hf_model):
        candidates.append(hf_model)
    elif os.path.isdir(hf_model):
        candidates.append(os.path.join(hf_model, "config.json"))
    else:
        cache = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", _DEFAULT_HF_CACHE), "hub")
        repo_dir = "models--" + hf_model.replace("/", "--")
        # Prefer the snapshot that also carries weights: it is the fully-materialised one.
        snapshots = sorted(glob.glob(os.path.join(cache, repo_dir, "snapshots", "*", "config.json")))
        weighted = [p for p in snapshots if glob.glob(os.path.join(os.path.dirname(p), "*.safetensors"))]
        candidates.extend(weighted or snapshots)

    for path in candidates:
        if os.path.isfile(path):
            with open(path) as fh:
                return json.load(fh), path
    raise FileNotFoundError(
        f"no config.json for --hf-model {hf_model!r}. Looked at: {candidates or '<nothing>'}. "
        f"Pass a local directory/config.json, or set HF_HOME (currently "
        f"{os.environ.get('HF_HOME', _DEFAULT_HF_CACHE)}) so the repo id resolves from cache."
    )


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def _b(x: float) -> str:
    return f"{x / 1e9:,.3f} B"


def _pflops(x: float) -> str:
    return f"{x / 1e15:,.4f} PFLOP"


def format_report(
    report: FlopReport,
    seconds_per_iter: list[float],
    gpus: int | None,
    peak_tflops: float,
    target_tflops: list[float],
    compare_megatron: bool,
) -> str:
    """Render the human-readable report."""
    arch, run, params = report.arch, report.run, report.params
    census = arch.layer_census()
    lines: list[str] = []
    add = lines.append

    add("=" * 88)
    add(f"FLOP estimate: {arch.name}")
    add("=" * 88)
    add(f"  training config : {run.config_path}")
    add(
        f"  layers          : {arch.num_layers} = {census[MAMBA]} Mamba2 + "
        f"{census[ATTENTION]} attention + {census[MOE]} latent-MoE"
        + (f" + {census[MLP]} dense-MLP" if census[MLP] else "")
    )
    add(
        f"  dims            : hidden {arch.hidden_size}, d_inner {arch.d_inner}, d_state "
        f"{arch.mamba_state_dim}, groups {arch.mamba_num_groups}, chunk {arch.chunk_size}, "
        f"conv_k {arch.conv_kernel}"
    )
    add(
        f"  attention       : {arch.num_attention_heads} heads / {arch.num_kv_heads} kv, "
        f"head_dim {arch.head_dim}, mask={report.attention_mask}"
    )
    add(
        f"  MoE             : top-{arch.top_k} of {arch.n_routed_experts}, ffn "
        f"{arch.moe_ffn_hidden_size}, latent {arch.moe_latent_size}, shared-ffn "
        f"{arch.shared_expert_ffn_hidden_size}, gated={arch.gated_mlp}"
    )
    add(f"  vocab           : {report.padded_vocab_size:,} (padded)")
    add(
        f"  workload        : GBS {run.global_batch_size} x seq {run.seq_length:,} = "
        f"{run.tokens_per_iter:,} tokens/iter"
    )
    add(
        f"  parallelism     : TP{run.tensor_model_parallel_size} PP{run.pipeline_model_parallel_size} "
        f"CP{run.context_parallel_size} EP{run.expert_model_parallel_size}"
    )
    add(
        f"  recompute       : {run.recompute_granularity or 'off'}"
        + (f" {list(run.recompute_modules)}" if run.recompute_modules else "")
    )

    add("")
    add("-- PARAMETERS " + "-" * 74)
    add(f"  total                     {_b(params.total)}")
    add(f"  active (per token)        {_b(params.active)}")
    add(f"  active, non-embedding     {_b(params.active_non_embedding)}")
    add(f"  per Mamba2 layer          {_b(params.mamba_layer)}")
    add(f"  per attention layer       {_b(params.attention_layer)}")
    add(f"  per MoE layer (total)     {_b(params.moe_layer_total)}")
    add(f"  per MoE layer (active)    {_b(params.moe_layer_active)}")

    add("")
    add("-- FORWARD FLOPs PER TOKEN " + "-" * 61)
    total_fwd_tok = report.forward_per_sample / run.seq_length
    for layer_type, label in ((MAMBA, "Mamba2"), (ATTENTION, "attention"), (MOE, "latent-MoE"), (MLP, "dense-MLP")):
        if not census[layer_type]:
            continue
        comps = report.forward_by_layer_type[layer_type]
        subtotal = sum(comps.values()) * census[layer_type] / run.seq_length
        add(
            f"  {label} x{census[layer_type]:<3}  {subtotal / 1e9:>10.4f} GFLOP/token  ({subtotal / total_fwd_tok:5.1%})"
        )
        for name, value in sorted(comps.items(), key=lambda kv: -kv[1]):
            per_tok = value * census[layer_type] / run.seq_length
            if per_tok == 0:
                continue
            add(f"      {name:<26} {per_tok / 1e9:>10.4f} GFLOP/token  ({per_tok / total_fwd_tok:5.1%})")
    head = 2.0 * arch.hidden_size * report.padded_vocab_size * (1 + run.mtp_num_layers)
    add(f"  lm_head            {head / 1e9:>10.4f} GFLOP/token  ({head / total_fwd_tok:5.1%})")
    add(f"  {'TOTAL forward':<18} {total_fwd_tok / 1e9:>10.4f} GFLOP/token")

    add("")
    add("-- PER ITERATION " + "-" * 71)
    add(f"  forward                       {_pflops(report.forward_per_sample * run.global_batch_size)}")
    add(
        f"  backward ({report.backward_multiplier:g}x forward)         "
        f"{_pflops(report.forward_per_sample * report.backward_multiplier * run.global_batch_size)}"
    )
    add(f"  recompute                     {_pflops(report.recompute_per_sample * run.global_batch_size)}")
    add(f"  MODEL FLOPs (fwd+bwd)         {_pflops(report.model_flops_per_iter)}   <- MFU basis")
    add(f"  HARDWARE FLOPs (+recompute)   {_pflops(report.hardware_flops_per_iter)}   <- HFU basis")
    add(f"  model FLOPs / token           {report.model_flops_per_token / 1e9:,.3f} GFLOP")

    add("")
    add("-- vs THE 6ND APPROXIMATION " + "-" * 60)
    for label, non_emb in (("N = active params", False), ("N = active non-embedding", True)):
        six = report.six_nd_per_iter(non_embedding=non_emb)
        add(f"  6ND ({label:<24})  {_pflops(six)}")
        add(f"      exact model FLOPs / 6ND               {report.model_flops_per_iter / six:6.3f}x")
        add(f"      exact hardware FLOPs / 6ND            {report.hardware_flops_per_iter / six:6.3f}x")

    if compare_megatron:
        mcore = megatron_counter_flops_per_iter(arch, run)
        add("")
        add("-- CROSS-CHECK vs THE IN-REPO COUNTER " + "-" * 50)
        add("  Called live: megatron.bridge.training.utils.flop_utils")
        add("               .num_floating_point_operations(cfg, batch_size)")
        add(f"  in-repo counter (the logged TFLOP/s basis)  {_pflops(mcore)}")
        add(f"  this estimator (model FLOPs)                {_pflops(report.model_flops_per_iter)}")
        add(f"  ratio exact / in-repo                       {report.model_flops_per_iter / mcore:6.4f}x")

    if gpus:
        floor = report.hardware_flops_per_iter / (peak_tflops * 1e12) / gpus
        add("")
        add("-- ACHIEVED THROUGHPUT " + "-" * 65)
        add(f"  {gpus} GPUs, peak {peak_tflops:g} TFLOP/s/GPU (BF16 dense, no sparsity)")
        add(f"  arithmetic floor: {floor:.2f} s/iter — the tensor-core busy time if every hardware FLOP ran at peak.")
        if seconds_per_iter:
            add(f"  {'s/iter':>8}  {'model TFLOP/s/GPU':>18}  {'MFU':>7}  {'hw TFLOP/s/GPU':>15}  {'HFU':>7}")
            for sec in seconds_per_iter:
                model_tf = report.model_flops_per_iter / sec / gpus / 1e12
                hw_tf = report.hardware_flops_per_iter / sec / gpus / 1e12
                add(
                    f"  {sec:>8.2f}  {model_tf:>18.1f}  {model_tf / peak_tflops:>6.1%}  "
                    f"{hw_tf:>15.1f}  {hw_tf / peak_tflops:>6.1%}"
                )
                add(
                    f"            -> {sec - floor:.2f} s/iter ({1 - floor / sec:.1%}) is NOT tensor-core "
                    f"arithmetic: pipeline bubble, collectives, memory-bound kernels, launch gaps."
                )

    if target_tflops and gpus:
        add("")
        add("-- WHAT A TARGET TFLOP/s/GPU WOULD REQUIRE " + "-" * 45)
        add(f"  {'target':>8}  {'s/iter (model basis)':>21}  {'s/iter (hw basis)':>19}  {'MFU':>7}")
        for target in target_tflops:
            sec_model = report.model_flops_per_iter / (target * 1e12) / gpus
            sec_hw = report.hardware_flops_per_iter / (target * 1e12) / gpus
            add(f"  {target:>8.0f}  {sec_model:>21.2f}  {sec_hw:>19.2f}  {target / peak_tflops:>6.1%}")

    add("=" * 88)
    return "\n".join(lines)


def report_to_dict(
    report: FlopReport,
    seconds_per_iter: list[float],
    gpus: int | None,
    compare_megatron: bool = False,
) -> dict[str, Any]:
    """Machine-readable form of the report, for `--json`.

    `compare_megatron` is opt-in for the same reason as the text report's section: it calls
    the real counter, which needs torch and megatron-core.
    """
    out: dict[str, Any] = {
        "model": report.arch.name,
        "config": report.run.config_path,
        "layer_census": report.arch.layer_census(),
        "params_total": report.params.total,
        "params_active": report.params.active,
        "params_active_non_embedding": report.params.active_non_embedding,
        "padded_vocab_size": report.padded_vocab_size,
        "tokens_per_iter": report.run.tokens_per_iter,
        "forward_flops_per_iter": report.forward_per_sample * report.run.global_batch_size,
        "recompute_flops_per_iter": report.recompute_per_sample * report.run.global_batch_size,
        "model_flops_per_iter": report.model_flops_per_iter,
        "hardware_flops_per_iter": report.hardware_flops_per_iter,
        "model_flops_per_token": report.model_flops_per_token,
        "six_nd_per_iter": report.six_nd_per_iter(),
        "six_nd_per_iter_non_embedding": report.six_nd_per_iter(non_embedding=True),
        "exact_over_6nd": report.model_flops_per_iter / report.six_nd_per_iter(),
        "hardware_over_6nd": report.hardware_flops_per_iter / report.six_nd_per_iter(),
    }
    if compare_megatron:
        out["megatron_counter_flops_per_iter"] = megatron_counter_flops_per_iter(report.arch, report.run)
    if gpus:
        out["throughput"] = [
            {
                "seconds_per_iter": sec,
                "model_tflops_per_gpu": report.model_flops_per_iter / sec / gpus / 1e12,
                "hardware_tflops_per_gpu": report.hardware_flops_per_iter / sec / gpus / 1e12,
            }
            for sec in seconds_per_iter
        ]
    return out


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI: one config path plus operational flags, per repo convention."""
    parser = argparse.ArgumentParser(
        description="Exact FLOP estimator for NemotronH hybrid models (config-driven).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("USAGE")[-1],
    )
    parser.add_argument("config", help="Training YAML (workload + recompute + parallelism come from here)")
    parser.add_argument(
        "--hf-model",
        required=True,
        help="HF repo id, model directory, or config.json path. Required: there is no "
        "checkpoint-path-to-model-id guessing table.",
    )
    parser.add_argument(
        "--seconds-per-iter",
        type=float,
        action="append",
        default=[],
        metavar="SEC",
        help="Measured iteration time; repeatable. Needs --gpus.",
    )
    parser.add_argument("--gpus", type=int, default=None, help="GPU count the measurement was taken on")
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=DEFAULT_PEAK_TFLOPS,
        help=f"Per-GPU BF16 dense peak for MFU/HFU (default {DEFAULT_PEAK_TFLOPS} = GH200/H100)",
    )
    parser.add_argument(
        "--target-tflops",
        type=float,
        action="append",
        default=[],
        metavar="TF",
        help="Report the s/iter a target TFLOP/s/GPU would imply; repeatable.",
    )
    parser.add_argument(
        "--attention-mask",
        choices=("causal", "full", "none"),
        default="causal",
        help="Core-attention accounting. Packed data is really block-diagonal, so 'causal' "
        "is an upper bound on that term (default: causal, matching every published number).",
    )
    parser.add_argument(
        "--backward-multiplier",
        type=float,
        default=2.0,
        help="Backward FLOPs as a multiple of forward (default 2.0 = dgrad + wgrad)",
    )
    parser.add_argument(
        "--compare-megatron",
        action="store_true",
        help="Also call the real in-repo counter (the one behind the logged TFLOP/s) and print "
        "the delta. Needs the container — it imports torch and megatron-core.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of the text report")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Resolve the architecture and workload, compute the report, print it."""
    args = build_parser().parse_args(argv)

    hf_cfg, hf_cfg_path = resolve_hf_config(args.hf_model)
    arch = ArchSpec.from_hf_config(hf_cfg, name=args.hf_model)
    run = RunSpec.from_yaml(args.config)
    report = compute_flops(
        arch,
        run,
        attention_mask=args.attention_mask,
        backward_multiplier=args.backward_multiplier,
    )

    if args.json:
        payload = report_to_dict(report, args.seconds_per_iter, args.gpus, args.compare_megatron)
        payload["hf_config_path"] = hf_cfg_path
        print(json.dumps(payload, indent=2))
    else:
        print(
            format_report(
                report,
                seconds_per_iter=args.seconds_per_iter,
                gpus=args.gpus,
                peak_tflops=args.peak_tflops,
                target_tflops=args.target_tflops,
                compare_megatron=args.compare_megatron,
            )
        )
        print(f"  arch source: {hf_cfg_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

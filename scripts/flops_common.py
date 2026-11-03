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

"""What the FLOP estimators of every model family share.

``scripts/nemotronh_flops_estimator.py`` (hybrid Mamba2 + attention + latent-MoE) and
``scripts/dense_flops_estimator.py`` (dense decoders with sliding-window layers) read different architecture
schemas, so each keeps its own architecture spec, parameter counts and per-layer formulas. This module holds the
parts that do not depend on the architecture:

* ``RunSpec``, the workload read from a training YAML, and ``padded_vocab_size``, which pads the vocabulary as the
  run does;
* ``FlopReport``, a model's FLOPs over one iteration, and ``achieved_throughput``, the one place a FLOP count is
  divided by time;
* the mcore layer kinds and the table of what each ``recompute_modules`` entry re-executes;
* ``resolve_hf_config``, which finds a model's ``config.json`` offline, and the default GPU peak.

The shared code reads an architecture only through ``Architecture`` (its vocabulary size) and a model's parameter
counts only through ``ActiveParams`` (the counts the 6ND comparison uses). Each family passes its own types, and
``FlopReport`` is generic over them so that a consumer sees the family's own architecture and parameter counts.
"""

from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar


# GH200 / H100 SXM dense BF16 tensor-core peak, TFLOP/s per GPU (no sparsity).
DEFAULT_PEAK_TFLOPS = 989.4

# Where to look for a repo-id's config.json when it is not a local path.
_DEFAULT_HF_CACHE = "/projects/a5k/public/hf"

# Layer kinds, in the alphabet of mcore's ssm/mamba_hybrid_layer_allocation.py: M mamba, * attention, E MoE,
# - dense MLP. The hybrid estimator reads its layer schedule in it, and a dense decoder is attention and dense-MLP
# layers in the same alphabet. The recompute table below is keyed by it.
MAMBA, ATTENTION, MOE, MLP = "M", "*", "E", "-"


# --------------------------------------------------------------------------------------
# The architecture and parameter counts, as the shared code reads them
# --------------------------------------------------------------------------------------


class Architecture(Protocol):
    """The part of a model family's architecture spec that the shared code reads."""

    @property
    def vocab_size(self) -> int:
        """Rows of the embedding table as the HF config states them, before any padding."""


class ActiveParams(Protocol):
    """The parameter counts that the 6ND comparison reads."""

    @property
    def active(self) -> int:
        """Parameters one token passes through (every parameter of a dense model)."""

    @property
    def active_non_embedding(self) -> int:
        """``active`` less the embedding table, which is a gather and not a matmul."""


# --------------------------------------------------------------------------------------
# The workload
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSpec:
    """The workload, read from a training YAML through its ``base_config`` chain."""

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
        # deferred: the reader needs PyYAML, and `--help` and imports must work without it
        from scripts.training.config_compose import load_composed_yaml

        cfg = load_composed_yaml(path)
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


def padded_vocab_size(arch: Architecture, run: RunSpec) -> int:
    """Vocab size as the model actually allocates it (YAML override, then TP padding)."""
    vocab = run.vocab_size_override or arch.vocab_size
    if not run.should_pad_vocab:
        return vocab
    multiple = run.make_vocab_size_divisible_by * run.tensor_model_parallel_size
    return int(math.ceil(vocab / multiple) * multiple)


# --------------------------------------------------------------------------------------
# Recompute
# --------------------------------------------------------------------------------------

# mcore recompute-module name -> the per-layer components it re-runs, by layer kind. The component names are the
# keys of an estimator's per-layer FLOP dictionaries; a family without a layer kind never meets its entries.
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

ArchT = TypeVar("ArchT")
ParamsT = TypeVar("ParamsT", bound=ActiveParams)


@dataclass
class FlopReport(Generic[ArchT, ParamsT]):
    """A model's FLOPs over one iteration of a run, with the architecture and parameter counts they come from.

    ``arch`` and ``params`` are the family's own types (the estimator that built the report reads them back for its
    own output); the arithmetic here needs only ``ActiveParams`` of ``params``. ``attention_mask`` names the
    core-attention accounting the estimator used.
    """

    arch: ArchT
    run: RunSpec
    params: ParamsT
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


def achieved_throughput(
    flops_per_iter: float, seconds_per_iter: float, gpus: int, peak_tflops: float
) -> tuple[float, float]:
    """TFLOP/s per GPU for ``flops_per_iter`` FLOPs done in ``seconds_per_iter`` on ``gpus`` GPUs, and its fraction of ``peak_tflops``.

    With model FLOPs the fraction is MFU, with hardware FLOPs HFU. This is the one place the
    division is written: the report here and ``scripts/telemetry/score_run.py`` both call it.
    """
    tflops_per_gpu = flops_per_iter / seconds_per_iter / gpus / 1e12
    return tflops_per_gpu, tflops_per_gpu / peak_tflops


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

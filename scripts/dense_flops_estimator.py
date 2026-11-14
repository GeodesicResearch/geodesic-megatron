"""FLOP estimator for dense decoder transformers (grouped-query attention, SwiGLU, sliding-window layers).

The dense counterpart of ``scripts/nemotronh_flops_estimator.py``. It is built to the HF ``config.json`` of
``Olmo3ForCausalLM`` (``hidden_size``, ``num_hidden_layers``, ``num_attention_heads``, ``num_key_value_heads``,
``head_dim`` or ``hidden_size / num_attention_heads``, ``intermediate_size``, ``vocab_size``, ``sliding_window``,
``layer_types``, ``hidden_act: silu``) and returns the ``FlopReport`` of ``scripts/flops_common.py``, so
``scripts/telemetry/score_run.py`` reads either model family's FLOPs per token the same way. The workload (sequence
length, vocabulary padding, recompute) comes from that module's ``RunSpec``, and the throughput arithmetic is its
``achieved_throughput``. The two estimators read different architecture schemas, so each has its own architecture
spec and parameter counts.

CONVENTIONS
-----------
* A multiply-accumulate is 2 FLOPs, so a matmul of M x N x K costs ``2 M N K``.
* Matmul-class work only: norms, RoPE, softmax and SiLU are elementwise and left out, as in the NemotronH estimator.
* Backward = 2x forward by default, so model FLOPs are 3x forward. Recompute is reported apart, as the hardware FLOPs.
* The input embedding is a gather and costs nothing. The output layer is a matmul whether or not it is tied.

PER-TOKEN FORMULAS (forward + backward = 3 x forward)
-----------------------------------------------------
With ``q = heads x head_dim``, ``kv = kv_heads x head_dim``, hidden size ``h``, FFN size ``F``, padded vocabulary
``V`` and ``L`` layers:

    projections       6 L [h (q + 2 kv) + q h]
    MLP (3 matrices)  18 L h F
    output layer      6 h V
    attention scores  12 q (L_full k_full + L_window k_window)

``k`` is the mean number of keys a query attends to in that kind of layer. QK^T and AV each cost ``2 q`` forward per
(query, key) pair, so a layer scoring ``P`` pairs costs ``4 q P`` forward over its pack.

KEYS PER QUERY
--------------
* ``causal`` (the primary convention): a query at position ``i`` (0-based) of a document attends to
  ``min(i + 1, W)`` keys, itself included, with ``W`` the window (infinite for a full layer). A document of ``n``
  tokens scores ``n (n + 1) / 2`` pairs when ``n <= W`` and ``W (W + 1) / 2 + W (n - W)`` otherwise, so a window
  layer over one document of ``s > W`` tokens has ``k = W - W (W - 1) / (2 s)``. The counts are exact sums.
  The Bridge counter (``flop_utils.num_floating_point_operations``) charges a window as a causal triangle of size
  ``W`` (``W / 2`` keys), which undercounts a window layer, and this module does not copy that.
* ``non_causal`` (secondary): every query sees ``n`` keys in a full layer and ``min(W, n)`` in a window layer.
* Packed documents: pass ``document_lengths`` and a layer scores the sum of its documents' pairs, which are
  divided by the pack's token count. The default is one document filling the sequence. Without
  ``document_lengths`` a packed run is counted as one long document, which overstates its attention.

LAYER SPLIT
-----------
``layer_types`` lists ``full_attention`` and ``sliding_attention`` layers. A config with neither ``layer_types`` nor
``sliding_window`` has full layers only. A ``sliding_window`` without ``layer_types``, or a ``sliding_attention``
entry without ``sliding_window``, leaves the split unknown and raises.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


# Run as a script, only scripts/ itself is on sys.path; the repo root makes the shared module importable the same way
# from every entry point.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from scripts.flops_common import ATTENTION, MLP, MOE, FlopReport, RunSpec, padded_vocab_size, recompute_components


# The HF ``layer_types`` entries.
FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"

# Attention accounting modes of ``compute_flops``.
CAUSAL = "causal"
NON_CAUSAL = "non_causal"


# --------------------------------------------------------------------------------------
# Architecture
# --------------------------------------------------------------------------------------


def layer_split(cfg: dict[str, Any], name: str) -> tuple[int, int]:
    """Return the (full-attention, sliding-window) layer counts of a parsed HF ``config.json``.

    Raises ValueError when ``layer_types`` has an unknown entry or the wrong length, and when the config sets
    ``sliding_window`` without ``layer_types`` or lists sliding layers without ``sliding_window``: the split between
    the two kinds is then unknown, and counting every layer as full would silently overstate the attention.
    """
    num_layers = int(cfg["num_hidden_layers"])
    layer_types = cfg.get("layer_types")
    if not layer_types:
        if cfg.get("sliding_window"):
            raise ValueError(
                f"{name}: sliding_window is set but layer_types is missing, so the layer split is unknown"
            )
        return num_layers, 0
    unknown = sorted(set(layer_types) - {FULL_ATTENTION, SLIDING_ATTENTION})
    if unknown:
        raise ValueError(f"{name}: unknown layer_types entries: {unknown}")
    if len(layer_types) != num_layers:
        raise ValueError(f"{name}: layer_types has {len(layer_types)} entries for {num_layers} layers")
    window_layers = layer_types.count(SLIDING_ATTENTION)
    if window_layers and not cfg.get("sliding_window"):
        raise ValueError(f"{name}: layer_types has {window_layers} sliding_attention layers but no sliding_window")
    return num_layers - window_layers, window_layers


@dataclass(frozen=True)
class DenseArchSpec:
    """Dense decoder dimensions, read from an HF ``config.json`` (``Olmo3ForCausalLM`` field names)."""

    name: str
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    ffn_hidden_size: int
    vocab_size: int
    tie_word_embeddings: bool
    # None when the model has no sliding-window layers.
    sliding_window: int | None
    full_layers: int
    window_layers: int

    @property
    def q_dim(self) -> int:
        """Width of the query projection (``heads x head_dim``)."""
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        """Width of the key projection, and of the value projection (``kv_heads x head_dim``)."""
        return self.num_kv_heads * self.head_dim

    def layer_census(self) -> dict[str, int]:
        """Count of each attention kind, leaving out a kind that has no layers."""
        census = {FULL_ATTENTION: self.full_layers, SLIDING_ATTENTION: self.window_layers}
        return {kind: count for kind, count in census.items() if count}

    @classmethod
    def from_hf_config(cls, cfg: dict[str, Any], name: str) -> DenseArchSpec:
        """Build from a parsed HF ``config.json``. Raises ValueError for anything but a SwiGLU decoder."""
        act = cfg.get("hidden_act")
        if act != "silu":
            raise ValueError(f"{name}: hidden_act {act!r}; only SwiGLU (silu) MLPs are modelled")
        hidden = int(cfg["hidden_size"])
        heads = int(cfg["num_attention_heads"])
        if cfg.get("head_dim"):
            head_dim = int(cfg["head_dim"])
        elif hidden % heads:
            raise ValueError(f"{name}: no head_dim, and hidden_size {hidden} is not a multiple of {heads} heads")
        else:
            head_dim = hidden // heads
        full_layers, window_layers = layer_split(cfg, name)
        return cls(
            name=name,
            hidden_size=hidden,
            num_layers=int(cfg["num_hidden_layers"]),
            num_attention_heads=heads,
            num_kv_heads=int(cfg.get("num_key_value_heads") or heads),
            head_dim=head_dim,
            ffn_hidden_size=int(cfg["intermediate_size"]),
            vocab_size=int(cfg["vocab_size"]),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
            sliding_window=int(cfg["sliding_window"]) if cfg.get("sliding_window") else None,
            full_layers=full_layers,
            window_layers=window_layers,
        )


@dataclass(frozen=True)
class DenseParamCounts:
    """Weight-matrix parameter counts of a dense decoder: one attention block and one MLP block per layer."""

    num_layers: int
    attention_layer: int
    mlp_layer: int
    embedding: int
    lm_head: int

    @property
    def total(self) -> int:
        return self.num_layers * (self.attention_layer + self.mlp_layer) + self.embedding + self.lm_head

    @property
    def active(self) -> int:
        """Every token passes through every parameter of a dense model."""
        return self.total

    @property
    def active_non_embedding(self) -> int:
        """Active params excluding the embedding table (which does a gather, not a GEMM)."""
        return self.active - self.embedding


def count_params(arch: DenseArchSpec, vocab_size: int) -> DenseParamCounts:
    """Weight-matrix parameter counts for a vocabulary of ``vocab_size`` rows.

    The norm vectors are left out: they are 0.003-0.007% of an Olmo 3 checkpoint.
    """
    h = arch.hidden_size
    return DenseParamCounts(
        num_layers=arch.num_layers,
        attention_layer=h * arch.q_dim + 2 * h * arch.kv_dim + arch.q_dim * h,
        mlp_layer=3 * h * arch.ffn_hidden_size,
        embedding=vocab_size * h,
        lm_head=0 if arch.tie_word_embeddings else vocab_size * h,
    )


# --------------------------------------------------------------------------------------
# Forward FLOPs
# --------------------------------------------------------------------------------------


def document_pairs(length: int, window: int | None, attention_mask: str) -> int:
    """(query, key) pairs one layer scores over one document of ``length`` tokens.

    ``window`` is the sliding window, which counts the query itself, or None for a full layer.
    Raises ValueError for an unknown ``attention_mask``.
    """
    if attention_mask == NON_CAUSAL:
        return length * (length if window is None else min(window, length))
    if attention_mask != CAUSAL:
        raise ValueError(f"unknown attention mask mode {attention_mask!r}; known: {CAUSAL}, {NON_CAUSAL}")
    if window is None or window >= length:
        return length * (length + 1) // 2
    # Positions 0..window-1 attend to 1..window keys, and every later position to exactly `window`.
    return window * (window + 1) // 2 + window * (length - window)


def attended_pairs(document_lengths: Sequence[int], window: int | None, attention_mask: str) -> int:
    """(query, key) pairs one layer scores over a pack made of the documents in ``document_lengths``."""
    return sum(document_pairs(length, window, attention_mask) for length in document_lengths)


def layer_flops(
    arch: DenseArchSpec,
    seq_len: int,
    document_lengths: Sequence[int],
    window: int | None,
    attention_mask: str,
) -> dict[str, float]:
    """Forward FLOPs of one layer over a pack of ``seq_len`` tokens holding ``document_lengths``."""
    h, q = arch.hidden_size, arch.q_dim
    return {
        "qkv_proj": 2.0 * seq_len * h * (q + 2 * arch.kv_dim),
        "o_proj": 2.0 * seq_len * q * h,
        # QK^T and AV each cost 2 * q FLOPs per (query, key) pair.
        "core_attn": 4.0 * q * attended_pairs(document_lengths, window, attention_mask),
        # gate, up and down matrices
        "mlp": 6.0 * seq_len * h * arch.ffn_hidden_size,
    }


def head_flops(arch: DenseArchSpec, seq_len: int, vocab_size: int) -> float:
    """Forward FLOPs of the output layer. The embedding lookup is a gather: 0 FLOPs."""
    return 2.0 * seq_len * arch.hidden_size * vocab_size


def pack_documents(seq_len: int, document_lengths: Sequence[int] | None) -> tuple[int, ...]:
    """Return the documents of a pack: ``document_lengths`` checked, or one document filling ``seq_len``."""
    if document_lengths is None:
        return (seq_len,)
    documents = tuple(document_lengths)
    if any(length <= 0 for length in documents) or sum(documents) != seq_len:
        raise ValueError(f"document_lengths must be positive and sum to seq_length {seq_len}, got {documents}")
    return documents


def recompute_flops(
    run: RunSpec,
    census: dict[str, int],
    per_type: dict[str, dict[str, float]],
    forward: float,
    head: float,
) -> float:
    """Forward FLOPs that ``run``'s recompute re-executes during backward.

    ``full`` re-runs every layer (the output layer is not recomputed). ``selective`` re-runs the components its
    ``recompute_modules`` name: ``core_attn`` and ``mlp`` are matmul work here, and the elementwise modules are
    free. Raises ValueError for an unknown granularity or module, and for a module that covers MoE layers, which a
    dense model lacks.
    """
    if run.recompute_granularity is None:
        return 0.0
    if run.recompute_granularity == "full":
        return forward - head
    if run.recompute_granularity != "selective":
        raise ValueError(f"unknown recompute_granularity {run.recompute_granularity!r}")
    covered = recompute_components(run.recompute_modules)
    if MOE in covered:
        raise ValueError(
            f"recompute_modules {list(run.recompute_modules)} cover MoE layers, which a dense model lacks"
        )
    recomputed = 0.0
    # A dense layer is one attention block and one MLP block, the layer kinds `*` and `-`.
    for layer_type, component in ((ATTENTION, "core_attn"), (MLP, "mlp")):
        if component in covered.get(layer_type, ()):
            recomputed += sum(census[kind] * per_type[kind][component] for kind in per_type)
    return recomputed


def compute_flops(
    arch: DenseArchSpec,
    run: RunSpec,
    attention_mask: str,
    backward_multiplier: float = 2.0,
    document_lengths: Sequence[int] | None = None,
) -> FlopReport[DenseArchSpec, DenseParamCounts]:
    """Roll the per-layer formulas up to a ``FlopReport`` for one pack of ``run.seq_length`` tokens.

    ``attention_mask`` is ``CAUSAL`` or ``NON_CAUSAL``. ``document_lengths`` are the documents of the pack
    (default: one document filling the sequence) and must sum to ``run.seq_length``. The vocabulary is padded as
    the run pads it. Raises ValueError for a run with multi-token prediction layers, which a dense model lacks, and
    for the conditions ``pack_documents`` and ``recompute_flops`` name.
    """
    if run.mtp_num_layers:
        raise ValueError(f"{arch.name}: mtp_num_layers is {run.mtp_num_layers}, but a dense model has no MTP layers")
    seq_len = run.seq_length
    documents = pack_documents(seq_len, document_lengths)
    vocab = padded_vocab_size(arch, run)
    census = arch.layer_census()
    per_type = {
        kind: layer_flops(
            arch, seq_len, documents, arch.sliding_window if kind == SLIDING_ATTENTION else None, attention_mask
        )
        for kind in census
    }
    head = head_flops(arch, seq_len, vocab)
    forward = sum(census[kind] * sum(per_type[kind].values()) for kind in per_type) + head
    return FlopReport(
        arch=arch,
        run=run,
        params=count_params(arch, vocab),
        padded_vocab_size=vocab,
        forward_by_layer_type=per_type,
        forward_per_sample=forward,
        recompute_per_sample=recompute_flops(run, census, per_type, forward, head),
        backward_multiplier=backward_multiplier,
        attention_mask=attention_mask,
    )

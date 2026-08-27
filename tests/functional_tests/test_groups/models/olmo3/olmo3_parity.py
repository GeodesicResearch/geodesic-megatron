# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Parity harness for the OLMo-3 HF <-> Megatron bridge.

Three things make OLMo-3 easy to convert *wrongly in a way that looks right*:

1. Sliding-window attention only matters past ``sliding_window`` tokens.
2. RoPE scaling only matters past ``original_max_position_embeddings`` tokens.
3. Both are pure config -- no weight is misplaced, so a weight-level round-trip
   passes regardless.

So a single short prompt, which is what the shipped ``compare.py`` example uses,
cannot distinguish a correct bridge from one that drops either feature. This
harness therefore compares logits at **several sequence lengths**, and ships
deliberate fault injections (``--fault``) so the checks can be shown to fail when
they should -- a green check whose power is unknown is not evidence.

The toy model built by :func:`build_toy_model` deliberately scales the two
thresholds *down* (``sliding_window=64``, ``original_max_position_embeddings=128``)
so that all three length regimes are reachable with a few hundred tokens.

**Reference semantics.** transformers 5.2.0 regressed OLMo-3: it builds one shared
rotary embedding and applies YaRN to all layers. v4.57.1 and current ``main`` both
apply YaRN to full-attention layers only, as does vLLM
(``vllm/model_executor/models/olmo2.py``: "Rope scaling is only applied on full
attention layers"). :func:`restore_per_layer_rope` puts the 4.57 behaviour back so
the reference is the architecture, not the regression.
"""

import argparse
import json
import os
from typing import Optional

import torch


TOY_CONFIG = {
    "architectures": ["Olmo3ForCausalLM"],
    "model_type": "olmo3",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "eos_token_id": 2,
    "pad_token_id": 0,
    "hidden_act": "silu",
    "hidden_size": 512,
    "intermediate_size": 1024,
    "num_hidden_layers": 8,  # -> full_attention at index 3 and 7
    "num_attention_heads": 8,  # head_dim 64
    "num_key_value_heads": 2,  # GQA ratio 4: exercises the k_norm width
    "max_position_embeddings": 1024,
    "rms_norm_eps": 1e-06,
    "sliding_window": 64,  # scaled down: a 128-token prompt already exceeds it
    "rope_theta": 500000.0,
    "rope_scaling": {
        "rope_type": "yarn",
        "factor": 8.0,
        "original_max_position_embeddings": 128,  # scaled down likewise
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "attention_factor": 1.2079441541679836,
    },
    "tie_word_embeddings": False,
    "vocab_size": 1000,  # not a multiple of 128: exercises vocab padding
    "dtype": "bfloat16",
    "transformers_version": "4.57.1",
}

# Sequence lengths and what each one is able to falsify.
LENGTH_CELLS = {
    "short": 48,  # < sliding_window: norms, QK-norm, rope base, mscale
    "mid": 256,  # > sliding_window (64): sliding window applied, to the right layers
    "long": 768,  # > original_max_position_embeddings (128): the YaRN frequency ramp
}

FAULTS = (
    "none",
    "yarn_all_layers",  # the transformers-5.2.0 regression, injected on purpose
    "no_swa",  # forget interleaved sliding-window attention
    "per_head_qknorm",  # stock Megatron QK-norm instead of OLMo full-width
    "swap_gate_up",  # gate/up halves exchanged in linear_fc1
)


def build_toy_model(out_dir: str, seed: int = 0) -> str:
    """Materialize a tiny but structurally faithful OLMo-3 checkpoint."""
    from transformers import AutoTokenizer, Olmo3Config, Olmo3ForCausalLM

    cfg = Olmo3Config(**TOY_CONFIG)
    assert cfg.layer_types.count("full_attention") == 2, cfg.layer_types
    torch.manual_seed(seed)
    model = Olmo3ForCausalLM(cfg).bfloat16()
    model.save_pretrained(out_dir, safe_serialization=True)
    # Write the flat config exactly as AI2 ships it (rope_scaling, not rope_parameters).
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(TOY_CONFIG, fh, indent=2)
    try:
        AutoTokenizer.from_pretrained("allenai/Olmo-3-32B-Think-DPO").save_pretrained(out_dir)
    except Exception:
        pass  # tokenizer is not needed for logit parity
    return out_dir


def restore_per_layer_rope(hf_model) -> None:
    """Give sliding-window layers unscaled RoPE, as transformers 4.57.1 does.

    transformers 4.57.1 built ``rotary_embs = ModuleDict({"sliding_attention":
    Olmo3RotaryEmbedding(config, rope_type="default"), "full_attention":
    Olmo3RotaryEmbedding(config)})`` and selected per layer. 5.2.0 collapsed that to
    a single YaRN rotary shared by every layer. We restore the 4.57 behaviour with a
    pre-hook that substitutes unscaled position embeddings on sliding layers only.

    No-op if the installed transformers already does the right thing (detected by
    the absence of a single shared ``rotary_emb``).
    """
    import copy as _copy

    inner = hf_model.model
    if not hasattr(inner, "rotary_emb"):
        return  # already per-layer (4.57 ModuleDict, or main's per-layer inv_freq)

    cfg_default = _copy.deepcopy(inner.config)
    cfg_default.rope_parameters = {
        "rope_type": "default",
        "rope_theta": inner.config.rope_parameters["rope_theta"],
    }
    rope_default = type(inner.rotary_emb)(config=cfg_default).to(
        device=next(hf_model.parameters()).device
    )

    def _hook(module, args, kwargs):
        pos = kwargs.get("position_ids")
        hs = kwargs.get("hidden_states", args[0] if args else None)
        kwargs["position_embeddings"] = rope_default(hs, pos)
        return args, kwargs

    layer_types = inner.config.layer_types
    for idx, layer in enumerate(inner.layers):
        if layer_types[idx] == "sliding_attention":
            layer.register_forward_pre_hook(_hook, with_kwargs=True)


def load_hf_reference(model_dir: str, device: str, dtype: torch.dtype, per_layer_rope: bool = True):
    """Load the HF model that defines 'correct'."""
    from transformers import Olmo3ForCausalLM

    model = Olmo3ForCausalLM.from_pretrained(model_dir, dtype=dtype).to(device).eval()
    if per_layer_rope:
        restore_per_layer_rope(model)
    return model


def build_megatron(model_dir: str, dtype: torch.dtype, fault: str = "none"):
    """Build the Megatron model through the bridge, optionally injecting a fault."""
    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(model_dir)
    provider = bridge.to_megatron_provider(load_weights=True)

    # No APEX in this environment and GPTModel's output_layer is the non-TE
    # ColumnParallelLinear, which requires it. nemo-rl forces the same off
    # (nemo_rl/models/megatron/setup.py:782), so this matches production.
    provider.gradient_accumulation_fusion = False
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.params_dtype = dtype
    provider.bf16 = dtype == torch.bfloat16
    provider.fp16 = dtype == torch.float16

    # Each fault must change exactly ONE thing. `window_size` and
    # `window_attn_skip_freq` jointly drive BOTH the attention span and the
    # per-layer rope choice, so the naive way to disable either would silently
    # change both and make a failure unattributable.
    if fault == "no_swa":
        # Keep the skip_freq pattern (so the rope selection is untouched) but give
        # every layer an unlimited causal window: attention span changes, rope does not.
        provider.window_size = (-1, 0)

    provider.finalize()
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    bridge.load_hf_weights(model)

    if fault == "yarn_all_layers":
        # The transformers-5.2.0 regression: YaRN on every layer. Flip only the rope
        # slice and the mscale; TE's sliding windows were already built from the
        # correct config at init, so the attention span is untouched.
        factor = provider.yarn_rotary_scaling_factor
        for layer in _core(model).decoder.layers:
            att = layer.self_attention
            att.is_sliding = False  # -> picks the YaRN slice of the stacked rope
            att.config.yarn_rotary_scaling_factor = factor  # -> mscale applied
    elif fault == "per_head_qknorm":
        # Stock Megatron normalizes each head independently; OLMo-3 normalizes the
        # whole projection. Same weights, different RMS denominator (64 vs 512).
        for layer in _core(model).decoder.layers:
            _make_qk_norm_per_head(layer.self_attention)
    elif fault == "swap_gate_up":
        for layer in _core(model).decoder.layers:
            w = layer.mlp.linear_fc1.weight.data
            half = w.shape[0] // 2
            layer.mlp.linear_fc1.weight.data = torch.cat([w[half:], w[:half]], dim=0)

    return bridge, model


def _make_qk_norm_per_head(att) -> None:
    """Rebind QK-norm to Megatron's stock per-head semantics (a deliberate fault).

    Uses the same learned gammas, but computes the RMS over each head's
    ``head_dim`` elements instead of over the full projection. That denominator is
    the whole difference between stock Megatron and OLMo-2/3, and it is invisible
    in the weights.
    """
    import types

    def _rms_per_head(x, gamma, eps, head_dim):
        shp = x.shape
        xh = x.reshape(*shp[:-1], -1, head_dim).float()
        xh = xh * torch.rsqrt(xh.pow(2).mean(-1, keepdim=True) + eps)
        return (xh.reshape(*shp).to(gamma.dtype)) * gamma

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None, **kwargs):
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)
        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]
        query, key, value = torch.split(mixed_qkv, split_arg_list, dim=3)
        hd = self.hidden_size_per_attention_head
        eps = self.config.layernorm_epsilon
        query = query.reshape(query.size(0), query.size(1), -1)
        key = key.reshape(key.size(0), key.size(1), -1)
        query = _rms_per_head(query, self.q_layernorm.weight, eps, hd)
        key = _rms_per_head(key, self.k_layernorm.weight, eps, hd)
        query = query.view(query.size(0), query.size(1), -1, hd)
        key = key.view(key.size(0), key.size(1), -1, hd)
        value = value.reshape(value.size(0), value.size(1), -1, hd)
        return query, key, value

    att.get_query_key_value_tensors = types.MethodType(get_query_key_value_tensors, att)


def _core(model):
    """Unwrap the model list / Float16Module wrapper."""
    m = model[0] if isinstance(model, list) else model
    return m.module if hasattr(m, "module") else m


@torch.no_grad()
def compare_logits(hf_model, megatron_model, seq_len: int, vocab_size: int, seed: int = 1234):
    """Run both models on the same tokens; return parity metrics."""
    device = next(hf_model.parameters()).device
    g = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (1, seq_len), generator=g).to(device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    hf_logits = hf_model(input_ids=input_ids).logits[0].float()

    m = megatron_model[0] if isinstance(megatron_model, list) else megatron_model
    mg_dev = next(m.parameters()).device
    out = m(
        input_ids=input_ids.to(mg_dev),
        position_ids=position_ids.to(mg_dev),
        attention_mask=None,
    )
    mg_logits = (out[0] if isinstance(out, tuple) else out)
    if mg_logits.shape[0] == seq_len and mg_logits.shape[1] == 1:  # [s, b, v] -> [s, v]
        mg_logits = mg_logits[:, 0, :]
    else:
        mg_logits = mg_logits[0]
    mg_logits = mg_logits.float()[:, : hf_logits.shape[-1]].to(hf_logits.device)

    diff = (hf_logits - mg_logits).abs()
    cos = torch.cosine_similarity(hf_logits.flatten().unsqueeze(0), mg_logits.flatten().unsqueeze(0)).item()
    hf_arg, mg_arg = hf_logits.argmax(-1), mg_logits.argmax(-1)
    agree = hf_arg == mg_arg
    top1 = agree.float().mean().item()

    # A bf16 disagreement on a near-tied pair is precision, not a conversion error.
    # Quantify it rather than asserting it: compare the HF top1-top2 gap at the
    # positions that disagree against the gap everywhere.
    top2 = hf_logits.topk(2, dim=-1).values
    gap = (top2[:, 0] - top2[:, 1])
    mism = ~agree
    top5 = hf_logits.topk(5, dim=-1).indices
    top5_hit = (top5 == mg_arg.unsqueeze(-1)).any(-1).float().mean().item()
    return {
        "seq_len": seq_len,
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "cosine": cos,
        "top1_agreement": top1,
        "top5_agreement": top5_hit,
        "median_top1_top2_gap_all": gap.median().item(),
        "median_top1_top2_gap_at_mismatch": (gap[mism].median().item() if mism.any() else float("nan")),
    }


@torch.no_grad()
def first_diverging_layer(hf_model, megatron_model, seq_len: int, vocab_size: int, seed: int = 1234):
    """Return per-layer hidden-state divergence, so a failure names a layer.

    On a 64-layer model "the logits differ" is not actionable; "layer 0 differs and
    layer 1 onward inherit it" is.
    """
    device = next(hf_model.parameters()).device
    g = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (1, seq_len), generator=g).to(device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    hf_hidden, mg_hidden = [], []
    hooks = []
    for layer in hf_model.model.layers:
        hooks.append(layer.register_forward_hook(
            lambda m, i, o, acc=hf_hidden: acc.append((o[0] if isinstance(o, tuple) else o).detach().float())
        ))
    core = _core(megatron_model)
    for layer in core.decoder.layers:
        hooks.append(layer.register_forward_hook(
            lambda m, i, o, acc=mg_hidden: acc.append((o[0] if isinstance(o, tuple) else o).detach().float())
        ))

    hf_model(input_ids=input_ids)
    m = megatron_model[0] if isinstance(megatron_model, list) else megatron_model
    m(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
    for h in hooks:
        h.remove()

    rows = []
    for i, (a, b) in enumerate(zip(hf_hidden, mg_hidden)):
        a2 = a[0] if a.shape[0] == 1 else a  # HF [b,s,h] -> [s,h]
        b2 = b[:, 0, :] if b.ndim == 3 and b.shape[1] == 1 else b  # Megatron [s,b,h] -> [s,h]
        if a2.shape != b2.shape:
            rows.append((i, float("nan")))
            continue
        rows.append((i, (a2 - b2).abs().max().item()))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="OLMo-3 HF<->Megatron parity harness")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--build-toy", action="store_true", help="materialize the toy model first")
    ap.add_argument("--cell", choices=sorted(LENGTH_CELLS), default="short")
    ap.add_argument("--seq-len", type=int, default=None, help="override the cell's length")
    ap.add_argument("--fault", choices=FAULTS, default="none")
    ap.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    ap.add_argument("--reference", choices=("per_layer_rope", "stock"), default="per_layer_rope")
    ap.add_argument("--max-abs-diff", type=float, default=1e-2)
    ap.add_argument("--min-cosine", type=float, default=0.9999)
    ap.add_argument("--min-top1", type=float, default=1.0)
    ap.add_argument(
        "--max-mismatch-gap",
        type=float,
        default=None,
        help="Bar on the median HF top1-top2 gap at positions where the two argmaxes "
        "disagree. In bf16 at 32B scale a correct conversion disagrees only on exact "
        "ties (gap == 0), while a real error disagrees on separated logits. This "
        "discriminates where raw top-1 agreement cannot.",
    )
    ap.add_argument("--expect-fail", action="store_true", help="invert the verdict (negative control)")
    ap.add_argument("--layer-report", action="store_true")
    ap.add_argument(
        "--weights-only",
        action="store_true",
        help="L0/L1 only: build + load + per-tensor verification table, no forward pass. "
        "This is the tier that fits a 32B model on a single GPU.",
    )
    ap.add_argument(
        "--hf-device",
        default="cuda",
        help="Device for the HF reference. For a 32B parity run put it on a second GPU "
        "(e.g. cuda:1) so both models fit.",
    )
    args = ap.parse_args()

    if args.build_toy:
        os.makedirs(args.model_dir, exist_ok=True)
        build_toy_model(args.model_dir)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    seq_len = args.seq_len or LENGTH_CELLS[args.cell]
    device = "cuda"

    bridge, mg = build_megatron(args.model_dir, dtype, fault=args.fault)

    if args.weights_only:
        from megatron.bridge.models.conversion import weights_verification_table

        table = weights_verification_table(bridge, mg)
        rows = list(zip(*[c._cells for c in table.columns]))
        good = [r for r in rows if "\u2705" in str(r[-1]) or "True" in str(r[-1])]
        bad = [r for r in rows if r not in good]
        print(f"RESULT weights-only model={args.model_dir} fault={args.fault} dtype={args.dtype}")
        print(f"  {'tensors_checked':18s} {len(rows)}")
        print(f"  {'tensors_matching':18s} {len(good)}")
        for r in bad[:20]:
            print(f"  MISMATCH {r[0]} -> {r[-1]}")
        ok = bool(rows) and not bad
        verdict_w = (not ok) if args.expect_fail else ok
        print(f"VERDICT {'OK' if verdict_w else 'FAIL'} (weights-only, {len(good)}/{len(rows)} match)")
        return 0 if verdict_w else 1

    hf = load_hf_reference(
        args.model_dir, args.hf_device, dtype, per_layer_rope=(args.reference == "per_layer_rope")
    )
    vocab = hf.config.vocab_size

    metrics = compare_logits(hf, mg, seq_len, vocab)
    passed = (
        metrics["max_abs_diff"] <= args.max_abs_diff
        and metrics["cosine"] >= args.min_cosine
        and metrics["top1_agreement"] >= args.min_top1
    )
    if args.max_mismatch_gap is not None:
        gap = metrics["median_top1_top2_gap_at_mismatch"]
        # NaN means there were no mismatches at all, which is a pass.
        passed = passed and (gap != gap or gap <= args.max_mismatch_gap)

    tag = f"cell={args.cell} seq_len={seq_len} fault={args.fault} ref={args.reference} dtype={args.dtype}"
    print(f"RESULT {tag}")
    for k, v in metrics.items():
        print(f"  {k:18s} {v}")
    print(f"  {'passed':18s} {passed}")

    if args.layer_report and not passed:
        print("  per-layer max|hf - megatron| (first divergence names the bug):")
        for i, d in first_diverging_layer(hf, mg, seq_len, vocab):
            print(f"    layer {i:3d}  {d}")

    verdict = (not passed) if args.expect_fail else passed
    print(f"VERDICT {'OK' if verdict else 'FAIL'} ({tag}"
          + (", expected-fail" if args.expect_fail else "") + ")")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())

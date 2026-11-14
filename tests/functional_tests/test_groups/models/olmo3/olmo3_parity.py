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
attention layers"). ``scripts/checkpoint/hf_olmo3_rope.py`` puts the 4.57 behaviour back so
the reference is the architecture, not the regression; the fault injections are
``scripts/checkpoint/olmo3_faults.py``, which ``scripts/checkpoint/packed_row_parity.py`` shares.

**The ``real_bf16`` profile** judges a real base in bf16 on random token ids (bshd, the Megatron model on one GPU and
the reference on another). Its bars lie between a correct run and the weakest injected fault on the real 7B and 32B
bases: cosine of the logits at least 0.999, at most 0.2 percent of the positions whose Megatron argmax lies more than 0.5
logits below the reference's top logit (``argmax deficit``, defined at ``compare_logits``), and no such deficit above
3.0.

A correct run does not reach top-1 agreement 1.0 on random ids: the distributions are near flat, and bf16 kernels that
differ flip the argmax between near ties. So top-1 agreement and the largest absolute logit difference are reported
and not judged.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.checkpoint.hf_olmo3_rope import load_hf_reference  # noqa: E402
from scripts.checkpoint.olmo3_faults import FAULTS, apply_fault_after_load, apply_fault_before_build  # noqa: E402


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

    # Each fault changes exactly ONE thing (scripts/checkpoint/olmo3_faults.py).
    apply_fault_before_build(provider, fault)

    provider.finalize()
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    bridge.load_hf_weights(model)
    apply_fault_after_load(model, fault, yarn_rotary_scaling_factor=provider.yarn_rotary_scaling_factor)

    return bridge, model


def _core(model):
    """Unwrap the model list / Float16Module wrapper."""
    m = model[0] if isinstance(model, list) else model
    return m.module if hasattr(m, "module") else m


LOGIT_CHUNK = 2048  # positions compared at a time: the 32B's logits at 32,768 positions need no fp32 copy of the whole
DEFICIT_DELTAS = (0.25, 0.5, 1.0)  # logit distances for the share of positions whose Megatron argmax is that far down
JUDGED_DEFICIT = 0.5  # the distance the profiles' max_deficit_fraction is a share of (one of DEFICIT_DELTAS)

# The bars a profile judges the metrics by; None leaves a metric reported and not judged. 'toy' is for the toy model in
# float32. 'real_bf16' sits between a correct run and the weakest injected fault on both real bases (see the header of
# this file).
PROFILES = {
    "toy": {
        "max_abs_diff": 1e-2,
        "min_cosine": 0.9999,
        "min_top1": 1.0,
        "max_deficit_fraction": None,
        "max_argmax_deficit": None,
    },
    "real_bf16": {
        "max_abs_diff": None,
        "min_cosine": 0.999,
        "min_top1": None,
        "max_deficit_fraction": 0.002,
        "max_argmax_deficit": 3.0,
    },
}


def judge(metrics: dict, bars: dict) -> bool:
    """Whether ``metrics`` (``compare_logits``) meet every bar of ``bars`` that is not None."""
    deficit_fraction = metrics[f"fraction_argmax_deficit_above_{JUDGED_DEFICIT}"]
    checks = [
        (bars["max_abs_diff"], metrics["max_abs_diff"] <= (bars["max_abs_diff"] or 0)),
        (bars["min_cosine"], metrics["cosine"] >= (bars["min_cosine"] or 0)),
        (bars["min_top1"], metrics["top1_agreement"] >= (bars["min_top1"] or 0)),
        (bars["max_deficit_fraction"], deficit_fraction <= (bars["max_deficit_fraction"] or 0)),
        (bars["max_argmax_deficit"], metrics["argmax_deficit_max"] <= (bars["max_argmax_deficit"] or 0)),
    ]
    return all(ok for bar, ok in checks if bar is not None)


@torch.no_grad()
def compare_logits(hf_model, megatron_model, seq_len: int, vocab_size: int, seed: int = 1234):
    """Run both models on the same tokens; return parity metrics.

    The logits stay in the models' dtype and are compared LOGIT_CHUNK positions at a time. ``argmax_deficit`` is, at each
    position, how far below the Hugging Face model's top logit its own logit sits at the token Megatron ranks first. Zero
    where the two argmaxes agree. A bf16 disagreement on a near tie has a small deficit, and a conversion error has a large one.
    """
    device = next(hf_model.parameters()).device
    g = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (1, seq_len), generator=g).to(device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    hf_logits = hf_model(input_ids=input_ids).logits[0]

    m = megatron_model[0] if isinstance(megatron_model, list) else megatron_model
    mg_dev = next(m.parameters()).device
    out = m(
        input_ids=input_ids.to(mg_dev),
        position_ids=position_ids.to(mg_dev),
        attention_mask=None,
    )
    mg_logits = out[0] if isinstance(out, tuple) else out
    if mg_logits.shape[0] == seq_len and mg_logits.shape[1] == 1:  # [s, b, v] -> [s, v]
        mg_logits = mg_logits[:, 0, :]
    else:
        mg_logits = mg_logits[0]
    mg_logits = mg_logits[:, : hf_logits.shape[-1]]

    max_abs, abs_sum, count = 0.0, 0.0, 0
    dot = norm_hf = norm_mg = 0.0
    agree_parts, top5_parts, gap_parts, deficit_parts = [], [], [], []
    for start in range(0, seq_len, LOGIT_CHUNK):
        hf = hf_logits[start : start + LOGIT_CHUNK].float()
        mg = mg_logits[start : start + LOGIT_CHUNK].to(hf.device).float()
        diff = (hf - mg).abs()
        max_abs = max(max_abs, diff.max().item())
        abs_sum += diff.sum(dtype=torch.float64).item()
        count += diff.numel()
        dot += (hf * mg).sum(dtype=torch.float64).item()
        norm_hf += (hf * hf).sum(dtype=torch.float64).item()
        norm_mg += (mg * mg).sum(dtype=torch.float64).item()
        mg_arg = mg.argmax(-1)
        top5 = hf.topk(5, dim=-1)
        top2 = top5.values[:, :2]
        agree_parts.append(top5.indices[:, 0] == mg_arg)
        top5_parts.append((top5.indices == mg_arg.unsqueeze(-1)).any(-1))
        gap_parts.append(top2[:, 0] - top2[:, 1])
        deficit_parts.append(top5.values[:, 0] - hf.gather(-1, mg_arg.unsqueeze(-1)).squeeze(-1))
        del hf, mg, diff

    agree = torch.cat(agree_parts)
    gap = torch.cat(gap_parts)
    deficit = torch.cat(deficit_parts)
    mism = ~agree
    metrics = {
        "seq_len": seq_len,
        "max_abs_diff": max_abs,
        "mean_abs_diff": abs_sum / count,
        "cosine": dot / (norm_hf**0.5 * norm_mg**0.5),
        "top1_agreement": agree.float().mean().item(),
        "top5_agreement": torch.cat(top5_parts).float().mean().item(),
        # A bf16 disagreement on a near-tied pair is precision, not a conversion error. Quantify it rather than
        # asserting it: compare the HF top1-top2 gap at the positions that disagree against the gap everywhere.
        "median_top1_top2_gap_all": gap.median().item(),
        "median_top1_top2_gap_at_mismatch": (gap[mism].median().item() if mism.any() else float("nan")),
        "argmax_deficit_max": deficit.max().item(),
    }
    for delta in DEFICIT_DELTAS:
        metrics[f"fraction_argmax_deficit_above_{delta}"] = (deficit > delta).float().mean().item()
    return metrics


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
        hooks.append(
            layer.register_forward_hook(
                lambda m, i, o, acc=hf_hidden: acc.append((o[0] if isinstance(o, tuple) else o).detach().float())
            )
        )
    core = _core(megatron_model)
    for layer in core.decoder.layers:
        hooks.append(
            layer.register_forward_hook(
                lambda m, i, o, acc=mg_hidden: acc.append((o[0] if isinstance(o, tuple) else o).detach().float())
            )
        )

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
    ap.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="toy",
        help="The bars the metrics are judged by: 'toy' for the toy model, 'real_bf16' for a real base model in bf16 "
        "(calibrated in the header of this file). A bar given below replaces the profile's.",
    )
    ap.add_argument("--max-abs-diff", type=float, default=None)
    ap.add_argument("--min-cosine", type=float, default=None)
    ap.add_argument("--min-top1", type=float, default=None)
    ap.add_argument(
        "--max-deficit-fraction",
        type=float,
        default=None,
        help=f"Bar on the share of positions whose Megatron argmax lies more than {JUDGED_DEFICIT} below HF's top logit "
        "in HF's logits",
    )
    ap.add_argument(
        "--max-argmax-deficit", type=float, default=None, help="Bar on the largest such distance over all positions"
    )
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
        "--export-hf",
        default=None,
        metavar="DIR",
        help="After building+loading, write the Megatron model back out as an HF "
        "checkpoint. Use this to produce the 'post-conversion' model for a "
        "generation A/B against the original.",
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

    bridge, mg = build_megatron(args.model_dir, dtype, fault=args.fault)

    if args.export_hf:
        import shutil

        print(f"RESULT export model={args.model_dir} -> {args.export_hf}")
        # AI2 ships generation_config.json with temperature/top_p but no
        # do_sample, which transformers >=5 rejects when *re-saving*. Relax it
        # just enough to get past the validator, then restore the upstream file
        # verbatim so the exported checkpoint matches the original artifact.
        gen_cfg = getattr(bridge.hf_pretrained, "generation_config", None)
        if gen_cfg is not None and not getattr(gen_cfg, "do_sample", False):
            if getattr(gen_cfg, "temperature", None) is not None or getattr(gen_cfg, "top_p", None) is not None:
                gen_cfg.do_sample = True
        bridge.save_hf_pretrained(mg, args.export_hf)
        src_gen = os.path.join(args.model_dir, "generation_config.json")
        if os.path.exists(src_gen):
            shutil.copyfile(src_gen, os.path.join(args.export_hf, "generation_config.json"))
            print("  restored upstream generation_config.json")
        print("VERDICT OK (exported)")
        return 0

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

    hf = load_hf_reference(args.model_dir, args.hf_device, dtype, per_layer_rope=(args.reference == "per_layer_rope"))
    vocab = hf.config.vocab_size

    metrics = compare_logits(hf, mg, seq_len, vocab)
    bars = {
        "max_abs_diff": args.max_abs_diff,
        "min_cosine": args.min_cosine,
        "min_top1": args.min_top1,
        "max_deficit_fraction": args.max_deficit_fraction,
        "max_argmax_deficit": args.max_argmax_deficit,
    }
    bars = {name: PROFILES[args.profile][name] if bar is None else bar for name, bar in bars.items()}
    passed = judge(metrics, bars)
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
    print(f"VERDICT {'OK' if verdict else 'FAIL'} ({tag}" + (", expected-fail" if args.expect_fail else "") + ")")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())

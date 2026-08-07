#!/usr/bin/env python3
"""Export a Megatron LoRA adapter checkpoint to the vLLM factor-dir format.

The serving path (`geodesic_utils.inference.vllm_extension.load_lora_factors_from_disk`)
reads `rank_<R>.safetensors` holding `{hf_name}.A` / `{hf_name}.B`, plus an `alphas.json`
mapping each hf_name to alpha_over_r, and applies `delta = alpha_over_r * (B @ A)`.
Megatron saves something structurally different, so three renames stand between them:

  1.  `decoder.layers.N.*`  ->  `backbone.layers.N.mixer.*`, per the NemotronH bridge's
      own mapping table (nemotron_h_bridge.py), not invented here.
  2.  The fused `linear_qkv` is ONE Megatron module and THREE HF keys. Megatron
      interleaves its output rows by query group — for each of `num_query_groups`, a
      block of `[heads_per_group*head_dim  q | head_dim  k | head_dim  v]` — so q, k and
      v are strided, not contiguous. This model has num_query_groups=2 against TP=4,
      which is the one sharding regime where a wrong split still produces plausible
      shapes, so the row arithmetic is derived from the checkpoint's own run_config and
      asserted against the expected output widths rather than hardcoded.
  3.  Mamba `in_proj` stores lora_B five-way split (`.z/.x/.B/.C/.dt`). HF's in_proj is
      the contiguous concatenation in exactly that order (MambaInProjMapping), so the
      five are concatenated along dim 0. A tool that greps `linear_out.weight` misses
      all 40 of these silently — they simply do not appear in the output.

Only lora_B needs splitting or concatenating: lora_A is shared across a module's HF keys,
because `delta = B @ A` factorises the output dimension only.

No distributed job is needed. `torch.distributed.checkpoint` reassembles TP-sharded
tensors into their full logical shape on CPU, given a correctly sized destination.

Usage:
    python pipeline_lora_export_factors.py \
        --checkpoint .../belief_implant_aiob_r256/iter_0000156 \
        --output-dir /projects/a5k/public/data_$USER/aiob_factors/iter_0000156 \
        --run-config .../base_views/mathsci_500m_32k_sft_iter288/iter_0000288/run_config.yaml \
        [--reference-alphas .../bench_lora_snapshot/alphas.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import yaml
from safetensors.torch import save_file

# Megatron module suffix -> HF module name. Taken from the NemotronH bridge's mapping
# table; `linear_qkv` is absent because it is one module and three HF keys (see below).
SUFFIX_TO_HF = {
    "mixer.in_proj": "mixer.in_proj",
    "mixer.out_proj": "mixer.out_proj",
    "mlp.shared_experts.linear_fc1": "mixer.shared_experts.up_proj",
    "mlp.shared_experts.linear_fc2": "mixer.shared_experts.down_proj",
    "self_attention.linear_proj": "mixer.o_proj",
}
QKV_SUFFIX = "self_attention.linear_qkv"
MAMBA_B_PARTS = ("z", "x", "B", "C", "dt")  # HF in_proj concatenation order


def parse_args() -> argparse.Namespace:  # noqa: D103
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--run-config", required=True, help="the BASE checkpoint's run_config.yaml")
    p.add_argument("--reference-alphas", default=None, help="an RL adapter's alphas.json")
    p.add_argument("--alpha-over-r", type=float, default=1.0)
    return p.parse_args()


def _load_full(reader, sd_meta, keys: list[str]) -> dict[str, torch.Tensor]:
    """Read `keys` at full logical shape (DCP reassembles the TP shards)."""
    dest = {k: torch.empty(sd_meta[k].size, dtype=sd_meta[k].properties.dtype) for k in keys}
    dcp.load(state_dict=dest, storage_reader=reader)
    return dest


def main() -> int:  # noqa: D103, PLR0912, PLR0915
    args = parse_args()
    cfg = yaml.safe_load(Path(args.run_config).read_text())
    model = cfg["model"] if "model" in cfg else cfg
    n_heads = model["num_attention_heads"]
    n_groups = model["num_query_groups"]
    head_dim = model["kv_channels"]
    hidden = model["hidden_size"]
    heads_per_group = n_heads // n_groups
    q_w, kv_w = n_heads * head_dim, n_groups * head_dim
    block = (heads_per_group + 2) * head_dim
    print(
        f"attention: {n_heads} heads / {n_groups} groups / head_dim {head_dim} -> "
        f"q {q_w}, k {kv_w}, v {kv_w}; qkv block {block}"
    )

    reader = dcp.FileSystemReader(args.checkpoint)
    sd_meta = reader.read_metadata().state_dict_metadata
    adapter_keys = [k for k in sd_meta if ".adapter." in k and not k.endswith("_extra_state")]
    modules = sorted({k.split(".adapter.")[0] for k in adapter_keys})
    print(f"checkpoint: {len(modules)} LoRA modules")

    out: dict[str, torch.Tensor] = {}
    alphas: dict[str, float] = {}
    unmapped: list[str] = []

    for mod in modules:
        m = re.match(r".*?decoder\.layers\.(\d+)\.(.+)$", mod)
        if not m:
            unmapped.append(mod)
            continue
        layer, suffix = m.group(1), m.group(2)

        a_key = f"{mod}.adapter.linear_in.weight"
        if a_key not in sd_meta:
            unmapped.append(mod)
            continue
        b_keys = [k for k in adapter_keys if k.startswith(f"{mod}.adapter.linear_out.weight")]
        tensors = _load_full(reader, sd_meta, [a_key, *b_keys])
        A = tensors[a_key].float()

        if suffix == "mixer.in_proj":
            parts = []
            for part in MAMBA_B_PARTS:
                key = f"{mod}.adapter.linear_out.weight.{part}"
                if key not in tensors:
                    print(f"ERROR: {mod} missing lora_B part .{part}", file=sys.stderr)
                    return 1
                parts.append(tensors[key].float())
            B = torch.cat(parts, dim=0)
            out[f"backbone.layers.{layer}.mixer.in_proj.weight.A"] = A
            out[f"backbone.layers.{layer}.mixer.in_proj.weight.B"] = B
            alphas[f"backbone.layers.{layer}.mixer.in_proj.weight"] = args.alpha_over_r

        elif suffix == QKV_SUFFIX:
            B = tensors[f"{mod}.adapter.linear_out.weight"].float()
            if B.shape[0] != q_w + 2 * kv_w:
                print(
                    f"ERROR: {mod} lora_B rows {B.shape[0]} != qkv width {q_w + 2 * kv_w}",
                    file=sys.stderr,
                )
                return 1
            qs, ks, vs = [], [], []
            for g in range(n_groups):
                base = g * block
                qs.append(B[base : base + heads_per_group * head_dim])
                ks.append(B[base + heads_per_group * head_dim : base + (heads_per_group + 1) * head_dim])
                vs.append(B[base + (heads_per_group + 1) * head_dim : base + block])
            for name, rows, width in (("q", qs, q_w), ("k", ks, kv_w), ("v", vs, kv_w)):
                stacked = torch.cat(rows, dim=0)
                assert stacked.shape[0] == width, f"{name} rows {stacked.shape[0]} != {width}"
                hf = f"backbone.layers.{layer}.mixer.{name}_proj.weight"
                out[f"{hf}.A"] = A.clone()
                out[f"{hf}.B"] = stacked
                alphas[hf] = args.alpha_over_r

        elif suffix in SUFFIX_TO_HF:
            B = tensors[f"{mod}.adapter.linear_out.weight"].float()
            hf = f"backbone.layers.{layer}.{SUFFIX_TO_HF[suffix]}.weight"
            out[f"{hf}.A"] = A
            out[f"{hf}.B"] = B
            alphas[hf] = args.alpha_over_r
        else:
            unmapped.append(mod)

    if unmapped:
        print(f"ERROR: {len(unmapped)} module(s) had no HF mapping:", file=sys.stderr)
        for u in unmapped[:5]:
            print(f"   {u}", file=sys.stderr)
        return 1

    # A is [r, in] and B is [out, r], where `in` is the MODULE's input width, not the
    # model's hidden size: Mamba out_proj consumes d_inner (8192 here) and the shared
    # experts' down_proj consumes ffn_hidden. Asserting `in == hidden_size` therefore
    # rejects a correct export. What must hold is that r is shared across every key and
    # that the two factors compose, which is what is checked here; the output widths are
    # checked against the reference adapter below, where the real model geometry lives.
    ranks = {v.shape[0] for k, v in out.items() if k.endswith(".A")}
    ranks |= {v.shape[1] for k, v in out.items() if k.endswith(".B")}
    if len(ranks) != 1:
        print(f"ERROR: inconsistent LoRA rank across A/B: {sorted(ranks)}", file=sys.stderr)
        return 1
    rank = ranks.pop()
    for hf in alphas:
        A, B = out[f"{hf}.A"], out[f"{hf}.B"]
        if A.ndim != 2 or B.ndim != 2 or B.shape[1] != A.shape[0]:
            print(
                f"ERROR: {hf} factors do not compose: A {tuple(A.shape)}, B {tuple(B.shape)}",
                file=sys.stderr,
            )
            return 1
    print(f"exported {len(alphas)} HF keys at rank {rank}")

    if args.reference_alphas:
        ref_path = Path(args.reference_alphas)
        ref = set(json.loads(ref_path.read_text()))
        got = set(alphas)
        if ref != got:
            print(
                f"ERROR: key set differs from the reference adapter.\n"
                f"  missing ({len(ref - got)}): {sorted(ref - got)[:4]}\n"
                f"  extra   ({len(got - ref)}): {sorted(got - ref)[:4]}",
                file=sys.stderr,
            )
            return 1
        print(f"key set identical to reference ({len(ref)} keys)")

        # Shapes, not just names. The reference was produced by the serving path's own
        # writer on this architecture, so its A/B widths ARE the model geometry: a wrong
        # qkv split or a mis-ordered Mamba concat changes a width while leaving every key
        # name intact, which the name check above cannot see.
        from safetensors import safe_open

        ref_shapes: dict[str, tuple[int, ...]] = {}
        for shard in sorted(ref_path.parent.glob("rank_*.safetensors")):
            with safe_open(str(shard), "pt") as f:
                for k in f.keys():  # noqa: SIM118
                    ref_shapes[k] = tuple(f.get_slice(k).get_shape())
        bad = []
        for k, v in out.items():
            if k not in ref_shapes:
                continue
            axis = 1 if k.endswith(".A") else 0  # the non-rank axis is model geometry
            if v.shape[axis] != ref_shapes[k][axis]:
                bad.append((k, tuple(v.shape), ref_shapes[k]))
        if bad:
            print(f"ERROR: {len(bad)} tensor(s) disagree with the reference geometry:", file=sys.stderr)
            for k, got_s, exp_s in bad[:6]:
                print(f"   {k}: got {got_s}, reference {exp_s}", file=sys.stderr)
            return 1
        print(f"tensor geometry matches reference on {len(ref_shapes) // 2} keys")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()}, str(outdir / "rank_00.safetensors"))
    (outdir / "alphas.json").write_text(json.dumps(alphas, indent=2, sort_keys=True) + "\n")
    print(f"wrote {outdir}/rank_00.safetensors and alphas.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

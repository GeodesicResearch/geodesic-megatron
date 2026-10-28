#!/usr/bin/env python3
"""Export a Megatron LoRA adapter checkpoint to the vLLM factor-dir format.

The serving side (geodesic_utils.inference.vllm_extension; its docs/lora_factor_format.md) reads
`rank_<R>.safetensors` holding `{hf_name}.A` / `{hf_name}.B` in bf16, plus an `alphas.json` mapping each hf_name to
alpha_over_r and `__format_version__` to the factor format, and applies `delta = alpha_over_r * (B @ A)`. Format 2:
every key a 2-D pair with HF-canonical rows. Format 3: format 2 plus grouped routed-expert keys, served by
`apply_base_adapter`.

Two adapter layouts are read:

- plain `LoRA` on fused modules (the RL adapters):
  1.  `decoder.layers.N.*`  ->  `backbone.layers.N.mixer.*`, per the NemotronH bridge's own mapping table
      (nemotron_h_bridge.py).
  2.  The fused `linear_qkv` is ONE Megatron module and THREE HF keys. Megatron interleaves its output rows by
      query group (for each of `num_query_groups`, `[heads_per_group*head_dim q | head_dim k | head_dim v]`), so q,
      k and v are strided; the row arithmetic comes from the run_config and is asserted against the output widths.
  3.  Mamba `in_proj` stores lora_B five-way split (`.z/.x/.B/.C/.dt`); HF's in_proj is their concatenation in that
      order (MambaInProjMapping).
  lora_A is shared across a module's HF keys, because `delta = B @ A` factorises the output dimension only.
- Tinker's layout (`CanonicalLoRA` + `GroupedExpertsLoRAAdapter`, "Expert LoRA"): separate q/k/v adapters, separate
  z and x adapters on in_proj (one HF in_proj key at rank 2r: A stacks both, B places each in its rows), the MoE
  latent projections (2-D keys), and per-expert LoRA on the routed experts as grouped keys
  `mixer.experts.{up,down}_proj.weight` (one factor shared, the other 3-D over experts).

No distributed job is needed. `torch.distributed.checkpoint` reassembles TP- and EP-sharded tensors into their full
logical shape on CPU, given a correctly sized destination.

Usage:
    python pipeline_lora_export_factors.py --checkpoint <run>/iter_NNNNNNN --output-dir <factors dir> \
        --run-config <run>/iter_NNNNNNN/run_config.yaml [--reference-alphas <an RL adapter's alphas.json>]
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
    "mlp.fc1_latent_proj": "mixer.fc1_latent_proj",
    "mlp.fc2_latent_proj": "mixer.fc2_latent_proj",
}
EXPERTS_SUFFIX = "mlp.experts"
QKV_SUFFIX = "self_attention.linear_qkv"
MAMBA_B_PARTS = ("z", "x", "B", "C", "dt")  # HF in_proj concatenation order
# The factor format geodesic_utils' reader accepts (FACTOR_FORMAT_VERSION in
# geodesic_utils/inference/_internal/factor_dir.py), stamped into alphas.json under this key.
FORMAT_VERSION_KEY = "__format_version__"
FACTOR_FORMAT_VERSION = 2
# A dir holding grouped routed-expert keys (`<moe>.experts.<proj>.weight`, one factor 3-D over experts) is format 3
# (GROUPED_EXPERT_FORMAT_VERSION in the same module); dense-only dirs stay 2.
GROUPED_EXPERT_FORMAT_VERSION = 3


def parse_args() -> argparse.Namespace:  # noqa: D103
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--run-config",
        required=True,
        help="a run_config.yaml with the model sizes (the training run's own also gives the peft alpha and dim)",
    )
    p.add_argument("--reference-alphas", default=None, help="an RL adapter's alphas.json")
    p.add_argument(
        "--alpha-over-r",
        type=float,
        default=None,
        help="the adapter scale (default: the run_config's peft alpha / dim, which must then be there)",
    )
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
    heads_per_group = n_heads // n_groups
    q_w, kv_w = n_heads * head_dim, n_groups * head_dim
    block = (heads_per_group + 2) * head_dim
    # HF Mamba in_proj rows: [z (d_inner) | x (d_inner) | B, C (groups * state each) | dt (heads)].
    d_inner = model["mamba_num_heads"] * model["mamba_head_dim"]
    in_proj_rows = 2 * d_inner + 2 * model["mamba_num_groups"] * model["mamba_state_dim"] + model["mamba_num_heads"]
    print(
        f"attention: {n_heads} heads / {n_groups} groups / head_dim {head_dim} -> "
        f"q {q_w}, k {kv_w}, v {kv_w}; qkv block {block}"
    )

    reader = dcp.FileSystemReader(args.checkpoint)
    sd_meta = reader.read_metadata().state_dict_metadata
    adapter_keys = [k for k in sd_meta if ".adapter." in k and not k.endswith("_extra_state")]
    modules = sorted({k.split(".adapter.")[0] for k in adapter_keys})
    print(f"checkpoint: {len(modules)} LoRA modules")

    peft = cfg.get("peft") or {}
    if args.alpha_over_r is not None:
        alpha_over_r = args.alpha_over_r
    elif "alpha" in peft and "dim" in peft:
        alpha_over_r = peft["alpha"] / peft["dim"]
    else:
        print("ERROR: the run_config has no peft alpha and dim; pass --alpha-over-r", file=sys.stderr)
        return 1
    print(f"alpha_over_r {alpha_over_r}")

    out: dict[str, torch.Tensor] = {}
    alphas: dict[str, float] = {}
    grouped: set[str] = set()  # grouped-expert keys (factor format 3)
    unmapped: list[str] = []

    for mod in modules:
        m = re.match(r".*?decoder\.layers\.(\d+)\.(.+)$", mod)
        if not m:
            unmapped.append(mod)
            continue
        layer, suffix = m.group(1), m.group(2)
        prefix = f"backbone.layers.{layer}.mixer"

        # CanonicalLoRA (Tinker's layout): separate q/k/v adapters on the fused linear_qkv.
        if suffix == QKV_SUFFIX and f"{mod}.adapter.adapter_q.linear_in.weight" in sd_meta:
            for name in ("q", "k", "v"):
                a_key = f"{mod}.adapter.adapter_{name}.linear_in.weight"
                b_key = f"{mod}.adapter.adapter_{name}.linear_out.weight"
                t = _load_full(reader, sd_meta, [a_key, b_key])
                hf = f"{prefix}.{name}_proj.weight"
                out[f"{hf}.A"], out[f"{hf}.B"] = t[a_key].float(), t[b_key].float()
                alphas[hf] = alpha_over_r
            continue

        # CanonicalLoRA: separate z and x adapters on Mamba in_proj. One HF in_proj takes them as a rank-2r pair: A
        # stacks both A factors, B places each B in its rows (z first, then x, as HF's [z|x|B|C|dt]) and leaves the
        # B, C and dt rows zero.
        if suffix == "mixer.in_proj" and f"{mod}.adapter.adapter_z.linear_in.weight" in sd_meta:
            keys = [f"{mod}.adapter.adapter_{n}.linear_{io}.weight" for n in ("z", "x") for io in ("in", "out")]
            t = {k: v.float() for k, v in _load_full(reader, sd_meta, keys).items()}
            a_z, b_z, a_x, b_x = (t[k] for k in keys)
            r = a_z.shape[0]
            B = torch.zeros(in_proj_rows, 2 * r)
            B[: b_z.shape[0], :r] = b_z
            B[b_z.shape[0] : b_z.shape[0] + b_x.shape[0], r:] = b_x
            hf = f"{prefix}.in_proj.weight"
            out[f"{hf}.A"], out[f"{hf}.B"] = torch.cat([a_z, a_x], dim=0), B
            alphas[hf] = alpha_over_r
            continue

        # Per-expert LoRA on GroupedExperts (Tinker's layout): fc1 shares A ([r, latent]) and keeps B per expert
        # ([E, r, ffn]); fc2 keeps A per expert ([E, ffn, r]) and shares B ([r, latent]). HF's per-expert up_proj is
        # [ffn, latent] and down_proj [latent, ffn], so as factor-format-3 grouped keys (one key per projection, the
        # shared factor 2-D, the per-expert one 3-D in global expert order):
        #   experts.up_proj:   A = fc1_a [r, latent],                 B = fc1_b^T(1,2) [E, ffn, r]
        #   experts.down_proj: A = fc2_a^T(1,2) [E, r, ffn],          B = fc2_b^T [latent, r]
        if suffix == EXPERTS_SUFFIX:
            keys = [f"{mod}.adapter.{n}" for n in ("fc1_a", "fc1_b", "fc2_a", "fc2_b")]
            t = {k: v.float() for k, v in _load_full(reader, sd_meta, keys).items()}
            fc1_a, fc1_b, fc2_a, fc2_b = (t[k] for k in keys)
            up, down = f"{prefix}.experts.up_proj.weight", f"{prefix}.experts.down_proj.weight"
            out[f"{up}.A"], out[f"{up}.B"] = fc1_a, fc1_b.transpose(1, 2).contiguous()
            out[f"{down}.A"], out[f"{down}.B"] = fc2_a.transpose(1, 2).contiguous(), fc2_b.t().contiguous()
            alphas[up] = alphas[down] = alpha_over_r
            grouped.update((up, down))
            continue

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
            alphas[f"backbone.layers.{layer}.mixer.in_proj.weight"] = alpha_over_r

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
                alphas[hf] = alpha_over_r

        elif suffix in SUFFIX_TO_HF:
            B = tensors[f"{mod}.adapter.linear_out.weight"].float()
            hf = f"backbone.layers.{layer}.{SUFFIX_TO_HF[suffix]}.weight"
            out[f"{hf}.A"] = A
            out[f"{hf}.B"] = B
            alphas[hf] = alpha_over_r
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
    # rejects a correct export. What must hold is that the two factors compose, which is
    # checked here; the output widths are checked against the reference adapter below,
    # where the real model geometry lives. A CanonicalLoRA in_proj carries rank 2r.
    ranks = set()
    for hf in alphas:
        A, B = out[f"{hf}.A"], out[f"{hf}.B"]
        if hf in grouped:
            # One factor shared (2-D), the other per expert (3-D); the rank axis is A's -2 and B's -1.
            n_experts = {t.shape[0] for t in (A, B) if t.ndim == 3}
            ok = A.ndim in (2, 3) and B.ndim in (2, 3) and 3 in (A.ndim, B.ndim) and len(n_experts) == 1
        else:
            ok = A.ndim == 2 and B.ndim == 2
        if not ok or B.shape[-1] != A.shape[-2]:
            print(
                f"ERROR: {hf} factors do not compose: A {tuple(A.shape)}, B {tuple(B.shape)}",
                file=sys.stderr,
            )
            return 1
        ranks.add(A.shape[-2])
    print(
        f"exported {len(alphas)} HF keys ({len(grouped)} grouped-expert) at rank {'/'.join(map(str, sorted(ranks)))}"
    )

    if args.reference_alphas:
        ref_path = Path(args.reference_alphas)
        ref_alphas = json.loads(ref_path.read_text())
        ref_version = ref_alphas.pop(FORMAT_VERSION_KEY, None)
        if ref_version not in (FACTOR_FORMAT_VERSION, GROUPED_EXPERT_FORMAT_VERSION):
            print(
                f"ERROR: the reference adapter is factor format {ref_version!r}, not {FACTOR_FORMAT_VERSION}",
                file=sys.stderr,
            )
            return 1
        ref = set(ref_alphas)
        got = set(alphas)
        # An RL adapter targets the standard modules only, so an Expert-LoRA export (routed experts, latent
        # projections) is a superset of it: every reference key must be present; the extra keys are only
        # those Expert LoRA adds.
        extra = got - ref
        expert_lora = {k for k in extra if ".experts." in k or "_latent_proj." in k}
        if ref - got or extra - expert_lora:
            print(
                f"ERROR: key set differs from the reference adapter.\n"
                f"  missing ({len(ref - got)}): {sorted(ref - got)[:4]}\n"
                f"  extra   ({len(extra - expert_lora)}): {sorted(extra - expert_lora)[:4]}",
                file=sys.stderr,
            )
            return 1
        print(
            f"key set matches reference ({len(ref)} keys"
            + (f"; plus {len(expert_lora)} Expert-LoRA keys)" if expert_lora else ")")
        )

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
    # The serving path (geodesic_utils exact_factor_delta) takes half-precision factors only: bf16, the serving dtype.
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in out.items()}, str(outdir / "rank_00.safetensors"))
    # Every row is HF-canonical: DCP reassembles each factor at its full logical shape whatever the training TP, and
    # in_proj's five B parts are concatenated in HF order above. That is what version 2 records.
    version = GROUPED_EXPERT_FORMAT_VERSION if grouped else FACTOR_FORMAT_VERSION
    sidecar = {FORMAT_VERSION_KEY: version, **dict(sorted(alphas.items()))}
    (outdir / "alphas.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"wrote {outdir}/rank_00.safetensors and alphas.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

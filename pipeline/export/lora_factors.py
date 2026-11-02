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
"""A LoRA checkpoint's adapter as a vLLM factor dir (``--format lora-factors``), unmerged.

The factor-dir format and the two adapter layouts it reads are described in pipeline_checkpoint_convert_hf.py's
docstring ("LoRA factor dirs").

Catalogue:
    LORA_SUFFIX_TO_HF: Megatron module suffix -> HF module name.
    LORA_EXPERTS_SUFFIX, LORA_QKV_SUFFIX: the module suffixes with layouts of their own.
    MAMBA_IN_PROJ_PARTS: HF in_proj's concatenation order of lora_B's parts.
    FACTOR_FORMAT_VERSION_KEY, FACTOR_FORMAT_VERSION, GROUPED_EXPERT_FORMAT_VERSION: the format stamp.
    split_fused_qkv_rows: Megatron's fused-QKV output rows as HF's q, k and v rows.
    export_lora_factors: write the factor dir.
"""

import json
import re
from pathlib import Path

import torch
import yaml


# Megatron module suffix -> HF module name, from the NemotronH bridge's mapping table. `linear_qkv` is absent: it is
# one module and three HF keys.
LORA_SUFFIX_TO_HF = {
    "mixer.in_proj": "mixer.in_proj",
    "mixer.out_proj": "mixer.out_proj",
    "mlp.shared_experts.linear_fc1": "mixer.shared_experts.up_proj",
    "mlp.shared_experts.linear_fc2": "mixer.shared_experts.down_proj",
    "self_attention.linear_proj": "mixer.o_proj",
    "mlp.fc1_latent_proj": "mixer.fc1_latent_proj",
    "mlp.fc2_latent_proj": "mixer.fc2_latent_proj",
}
LORA_EXPERTS_SUFFIX = "mlp.experts"
LORA_QKV_SUFFIX = "self_attention.linear_qkv"
MAMBA_IN_PROJ_PARTS = ("z", "x", "B", "C", "dt")  # HF in_proj concatenation order


# The factor formats geodesic_utils' reader accepts (FACTOR_FORMAT_VERSION and GROUPED_EXPERT_FORMAT_VERSION in
# geodesic_utils/inference/_internal/factor_dir.py), stamped into alphas.json under this key. A dir holding grouped
# routed-expert keys is format 3; dense-only dirs are format 2.
FACTOR_FORMAT_VERSION_KEY = "__format_version__"
FACTOR_FORMAT_VERSION = 2
GROUPED_EXPERT_FORMAT_VERSION = 3


def split_fused_qkv_rows(
    rows: torch.Tensor, num_heads: int, num_query_groups: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split Megatron's fused-QKV output rows into HF's q, k and v rows.

    Megatron orders them per query group: that group's ``num_heads // num_query_groups`` q heads, then its k head,
    then its v head, each ``head_dim`` rows.

    Raises:
        ValueError: ``rows`` does not have the fused width.
    """
    heads_per_group = num_heads // num_query_groups
    block = (heads_per_group + 2) * head_dim
    if rows.shape[0] != num_query_groups * block:
        raise ValueError(f"{rows.shape[0]} rows are not the fused qkv width {num_query_groups * block}")
    qs, ks, vs = [], [], []
    for g in range(num_query_groups):
        base = g * block
        qs.append(rows[base : base + heads_per_group * head_dim])
        ks.append(rows[base + heads_per_group * head_dim : base + (heads_per_group + 1) * head_dim])
        vs.append(rows[base + (heads_per_group + 1) * head_dim : base + block])
    return torch.cat(qs), torch.cat(ks), torch.cat(vs)


def _read_lora_factors(
    iter_path: Path, model_cfg: dict, alpha_over_r: float
) -> tuple[dict[str, torch.Tensor], dict[str, float], set[str]]:
    """Every LoRA module of the checkpoint as HF-keyed factors: (``{hf}.A``/``{hf}.B`` -> fp32, hf -> scale,
    grouped-expert hf keys).

    Raises:
        ValueError: A LoRA module has no HF mapping, or a factor does not have the shape the run_config implies.
    """
    import torch.distributed.checkpoint as dcp

    reader = dcp.FileSystemReader(str(iter_path))
    sd_meta = reader.read_metadata().state_dict_metadata

    def load(keys: list[str]) -> dict[str, torch.Tensor]:
        dest = {k: torch.empty(sd_meta[k].size, dtype=sd_meta[k].properties.dtype) for k in keys}
        dcp.load(state_dict=dest, storage_reader=reader)
        return {k: v.float() for k, v in dest.items()}

    # HF Mamba in_proj rows: [z (d_inner) | x (d_inner) | B, C (groups * state each) | dt (heads)].
    d_inner = model_cfg["mamba_num_heads"] * model_cfg["mamba_head_dim"]
    in_proj_rows = (
        2 * d_inner + 2 * model_cfg["mamba_num_groups"] * model_cfg["mamba_state_dim"] + model_cfg["mamba_num_heads"]
    )
    adapter_keys = [k for k in sd_meta if ".adapter." in k and not k.endswith("_extra_state")]
    modules = sorted({k.split(".adapter.")[0] for k in adapter_keys})
    print(f"checkpoint: {len(modules)} LoRA modules, alpha_over_r {alpha_over_r}")

    out: dict[str, torch.Tensor] = {}
    alphas: dict[str, float] = {}
    grouped: set[str] = set()
    unmapped: list[str] = []
    for mod in modules:
        m = re.match(r".*?decoder\.layers\.(\d+)\.(.+)$", mod)
        if not m:
            unmapped.append(mod)
            continue
        suffix, prefix = m.group(2), f"backbone.layers.{m.group(1)}.mixer"

        def put(hf: str, a: torch.Tensor, b: torch.Tensor) -> None:
            out[f"{hf}.A"], out[f"{hf}.B"] = a, b
            alphas[hf] = alpha_over_r

        if suffix == LORA_QKV_SUFFIX and f"{mod}.adapter.adapter_q.linear_in.weight" in sd_meta:
            # CanonicalLoRA: separate q/k/v adapters on the fused linear_qkv.
            for name in ("q", "k", "v"):
                a, b = (f"{mod}.adapter.adapter_{name}.linear_{io}.weight" for io in ("in", "out"))
                t = load([a, b])
                put(f"{prefix}.{name}_proj.weight", t[a], t[b])
        elif suffix == "mixer.in_proj" and f"{mod}.adapter.adapter_z.linear_in.weight" in sd_meta:
            # CanonicalLoRA: separate z and x adapters. One HF in_proj takes them as a rank-2r pair: A stacks both A
            # factors, B places each B in its rows (z, then x, as HF's [z|x|B|C|dt]) and leaves B, C and dt zero.
            keys = [f"{mod}.adapter.adapter_{n}.linear_{io}.weight" for n in ("z", "x") for io in ("in", "out")]
            t = load(keys)
            a_z, b_z, a_x, b_x = (t[k] for k in keys)
            r = a_z.shape[0]
            b = torch.zeros(in_proj_rows, 2 * r)
            b[: b_z.shape[0], :r] = b_z
            b[b_z.shape[0] : b_z.shape[0] + b_x.shape[0], r:] = b_x
            put(f"{prefix}.in_proj.weight", torch.cat([a_z, a_x], dim=0), b)
        elif suffix == LORA_EXPERTS_SUFFIX:
            # GroupedExpertsLoRAAdapter: fc1 shares A ([r, latent]) and keeps B per expert ([E, r, ffn]); fc2 keeps A
            # per expert ([E, ffn, r]) and shares B ([r, latent]). HF's per-expert up_proj is [ffn, latent] and
            # down_proj [latent, ffn], so as format-3 grouped keys (the per-expert factor 3-D in global expert order):
            #   experts.up_proj:   A = fc1_a [r, latent],          B = fc1_b^T(1,2) [E, ffn, r]
            #   experts.down_proj: A = fc2_a^T(1,2) [E, r, ffn],   B = fc2_b^T [latent, r]
            keys = [f"{mod}.adapter.{n}" for n in ("fc1_a", "fc1_b", "fc2_a", "fc2_b")]
            t = load(keys)
            fc1_a, fc1_b, fc2_a, fc2_b = (t[k] for k in keys)
            up, down = f"{prefix}.experts.up_proj.weight", f"{prefix}.experts.down_proj.weight"
            put(up, fc1_a, fc1_b.transpose(1, 2).contiguous())
            put(down, fc2_a.transpose(1, 2).contiguous(), fc2_b.t().contiguous())
            grouped.update((up, down))
        elif f"{mod}.adapter.linear_in.weight" in sd_meta:
            # Plain LoRA: one A, and one B (five parts on in_proj).
            a_key = f"{mod}.adapter.linear_in.weight"
            b_keys = [k for k in adapter_keys if k.startswith(f"{mod}.adapter.linear_out.weight")]
            t = load([a_key, *b_keys])
            a = t[a_key]
            if suffix == "mixer.in_proj":
                parts = [f"{mod}.adapter.linear_out.weight.{p}" for p in MAMBA_IN_PROJ_PARTS]
                missing = [p for p in parts if p not in t]
                if missing:
                    raise ValueError(f"{mod} lora_B lacks the in_proj parts {missing}")
                put(f"{prefix}.in_proj.weight", a, torch.cat([t[p] for p in parts], dim=0))
            elif suffix == LORA_QKV_SUFFIX:
                q, k, v = split_fused_qkv_rows(
                    t[f"{mod}.adapter.linear_out.weight"],
                    model_cfg["num_attention_heads"],
                    model_cfg["num_query_groups"],
                    model_cfg["kv_channels"],
                )
                for name, b in zip("qkv", (q, k, v), strict=True):
                    put(f"{prefix}.{name}_proj.weight", a.clone(), b)
            elif suffix in LORA_SUFFIX_TO_HF:
                put(
                    f"backbone.layers.{m.group(1)}.{LORA_SUFFIX_TO_HF[suffix]}.weight",
                    a,
                    t[f"{mod}.adapter.linear_out.weight"],
                )
            else:
                unmapped.append(mod)
        else:
            unmapped.append(mod)
    if unmapped:
        raise ValueError(f"{len(unmapped)} LoRA module(s) have no HF mapping, e.g. {unmapped[:3]}")
    return out, alphas, grouped


def _check_factor_composition(out: dict[str, torch.Tensor], alphas: dict[str, float], grouped: set[str]) -> set[int]:
    """Every key's factors compose (B's rank axis is A's); returns the ranks.

    A is [r, in] with ``in`` the module's input width (Mamba out_proj consumes d_inner, a down_proj its FFN width), so
    widths are checked against the reference adapter, not against hidden_size. A grouped key has one 2-D (shared) and
    one 3-D (per-expert) factor, with the rank axis at A's -2 and B's -1.
    """
    ranks = set()
    for hf in alphas:
        a, b = out[f"{hf}.A"], out[f"{hf}.B"]
        if hf in grouped:
            experts = {t.shape[0] for t in (a, b) if t.ndim == 3}
            ok = a.ndim in (2, 3) and b.ndim in (2, 3) and 3 in (a.ndim, b.ndim) and len(experts) == 1
        else:
            ok = a.ndim == 2 and b.ndim == 2
        if not ok or b.shape[-1] != a.shape[-2]:
            raise ValueError(f"{hf} factors do not compose: A {tuple(a.shape)}, B {tuple(b.shape)}")
        ranks.add(a.shape[-2])
    return ranks


def _check_against_reference_adapter(out: dict[str, torch.Tensor], alphas: dict[str, float], ref_path: Path) -> None:
    """The key set and every non-rank width agree with a reference factor dir's (an RL adapter on this model).

    An RL adapter targets the standard modules only, so an Expert-LoRA export is a superset: every reference key must
    be present, and the only extra keys are the routed experts' and latent projections'.

    Raises:
        ValueError: The reference is not format 2 or 3, a key is missing or unexpected, or a width differs.
    """
    from safetensors import safe_open

    ref_alphas = json.loads(ref_path.read_text())
    ref_version = ref_alphas.pop(FACTOR_FORMAT_VERSION_KEY, None)
    if ref_version not in (FACTOR_FORMAT_VERSION, GROUPED_EXPERT_FORMAT_VERSION):
        raise ValueError(f"the reference adapter {ref_path} is factor format {ref_version!r}, not 2 or 3")
    ref, got = set(ref_alphas), set(alphas)
    extra = got - ref
    expert_lora = {k for k in extra if ".experts." in k or "_latent_proj." in k}
    if ref - got or extra - expert_lora:
        raise ValueError(
            f"the key set differs from the reference adapter: missing {sorted(ref - got)[:4]} "
            f"({len(ref - got)}), unexpected {sorted(extra - expert_lora)[:4]} ({len(extra - expert_lora)})"
        )
    # Names alone cannot see a wrong qkv split or a mis-ordered Mamba concat, which change a width but keep every
    # key; the reference's widths are the model geometry.
    ref_shapes: dict[str, tuple[int, ...]] = {}
    for shard in sorted(ref_path.parent.glob("rank_*.safetensors")):
        with safe_open(str(shard), "pt") as f:
            for k in f.keys():  # noqa: SIM118
                ref_shapes[k] = tuple(f.get_slice(k).get_shape())
    bad = [
        (k, tuple(v.shape), ref_shapes[k])
        for k, v in out.items()
        if k in ref_shapes and v.shape[1 if k.endswith(".A") else 0] != ref_shapes[k][1 if k.endswith(".A") else 0]
    ]
    if bad:
        raise ValueError(f"{len(bad)} tensor(s) disagree with the reference geometry, e.g. {bad[:3]}")
    print(
        f"matches the reference adapter: {len(ref)} keys and their widths"
        + (f", plus {len(expert_lora)} Expert-LoRA keys" if expert_lora else "")
    )


def export_lora_factors(
    iter_path: Path, factors_path: Path, alpha_over_r: float | None = None, reference_alphas: Path | None = None
) -> None:
    """Write a LoRA checkpoint's adapter as a vLLM factor dir (format 3 with grouped-expert keys, else 2).

    Args:
        iter_path: The ``iter_N`` directory; its ``run_config.yaml`` gives the model sizes and the peft alpha and dim.
        factors_path: Output directory for ``rank_00.safetensors`` and ``alphas.json``.
        alpha_over_r: The adapter scale; None takes the run_config's ``peft.alpha / peft.dim``.
        reference_alphas: Optionally, an RL adapter's ``alphas.json`` on the same model to check keys and widths
            against.

    Raises:
        ValueError: No scale is given and the run_config has none, or a check fails.
    """
    from safetensors.torch import save_file

    run_config = iter_path / "run_config.yaml"
    cfg = yaml.safe_load(run_config.read_text())
    peft = cfg.get("peft") or {}
    if alpha_over_r is None:
        if "alpha" not in peft or "dim" not in peft:
            raise ValueError(f"{run_config} has no peft alpha and dim; pass --alpha-over-r")
        alpha_over_r = peft["alpha"] / peft["dim"]
    out, alphas, grouped = _read_lora_factors(iter_path, cfg["model"], alpha_over_r)
    ranks = _check_factor_composition(out, alphas, grouped)
    print(
        f"exported {len(alphas)} HF keys ({len(grouped)} grouped-expert) at rank {'/'.join(map(str, sorted(ranks)))}"
    )
    if reference_alphas is not None:
        _check_against_reference_adapter(out, alphas, reference_alphas)

    factors_path.mkdir(parents=True, exist_ok=True)
    # geodesic_utils' exact_factor_delta takes half-precision factors only: bf16, the serving dtype.
    save_file(
        {k: v.to(torch.bfloat16).contiguous() for k, v in out.items()}, str(factors_path / "rank_00.safetensors")
    )
    version = GROUPED_EXPERT_FORMAT_VERSION if grouped else FACTOR_FORMAT_VERSION
    sidecar = {FACTOR_FORMAT_VERSION_KEY: version, **dict(sorted(alphas.items()))}
    (factors_path / "alphas.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"wrote {factors_path}/rank_00.safetensors and alphas.json (factor format {version})")

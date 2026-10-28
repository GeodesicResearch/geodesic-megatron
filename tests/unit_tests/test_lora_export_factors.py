"""pipeline_lora_export_factors.py end to end: a toy Megatron LoRA checkpoint to a factor dir and back to dense deltas.

Each test writes a small torch_dist (DCP) adapter checkpoint with the key names geodesic-megatron's PEFT modules save,
runs the exporter, and rebuilds every HF weight's dense delta from the factor dir as the serving side does
(``alpha_over_r * B_e @ A_e``). The deltas are compared with the ones the training-side merge adds to each weight:
``LoRAMerge`` (``alpha / dim * B @ A``, the fused rows of linear_qkv split per query group, in_proj's five B parts in
HF order) and ``GroupedExpertsLoRAAdapter.merge_into`` (``einsum("ri,erf->eif", fc1_a, fc1_b)`` and
``einsum("efr,ri->efi", fc2_a, fc2_b)``, scaled, on the ``[E, in, out]`` grouped weights). The formulas are restated
here because this repo's main does not carry the Tinker-layout PEFT modules.

Two layouts are covered:
- the RL default, plain ``LoRA`` on fused modules (linear_qkv, in_proj with a five-way B, shared experts): format 2;
- Tinker's layout (``CanonicalLoRA`` with separate q/k/v and z/x adapters, latent projections, and
  ``GroupedExpertsLoRAAdapter`` on the routed experts): format 3 with grouped-expert keys.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
import yaml
from safetensors.torch import load_file


_TOOL = Path(__file__).resolve().parents[2] / "pipeline_lora_export_factors.py"

# Toy geometry: 4 heads in 2 query groups of head_dim 8; Mamba 2 heads x head_dim 4 (d_inner 8), 1 group, state 3;
# MoE latent 6, FFN 5, 3 routed experts; hidden 16; rank 2.
H, HEADS, GROUPS, HEAD_DIM = 16, 4, 2, 8
M_HEADS, M_HEAD_DIM, M_GROUPS, M_STATE = 2, 4, 1, 3
D_INNER = M_HEADS * M_HEAD_DIM
LATENT, FFN, E, R = 6, 5, 3, 2
ALPHA = 4  # alpha / dim = 2
MODEL_CFG = {
    "num_attention_heads": HEADS,
    "num_query_groups": GROUPS,
    "kv_channels": HEAD_DIM,
    "hidden_size": H,
    "mamba_num_heads": M_HEADS,
    "mamba_head_dim": M_HEAD_DIM,
    "mamba_num_groups": M_GROUPS,
    "mamba_state_dim": M_STATE,
}
IN_PROJ_PARTS = {"z": D_INNER, "x": D_INNER, "B": M_GROUPS * M_STATE, "C": M_GROUPS * M_STATE, "dt": M_HEADS}


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("pipeline_lora_export_factors", _TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_lora_export_factors"] = module
    spec.loader.exec_module(module)
    return module


def _rand(*shape):
    """fp32 values that bf16 holds exactly, so the exporter's bf16 factors lose nothing and deltas compare tightly."""
    return torch.randn(*shape, dtype=torch.float32).to(torch.bfloat16).float()


def _qkv_rows() -> dict[str, list[int]]:
    """Megatron's fused-QKV row order: per query group, its q heads, then k, then v."""
    hpg, rows = HEADS // GROUPS, {"q": [], "k": [], "v": []}
    block = (hpg + 2) * HEAD_DIM
    for g in range(GROUPS):
        base = g * block
        rows["q"] += list(range(base, base + hpg * HEAD_DIM))
        rows["k"] += list(range(base + hpg * HEAD_DIM, base + (hpg + 1) * HEAD_DIM))
        rows["v"] += list(range(base + (hpg + 1) * HEAD_DIM, base + block))
    return rows


def _write(tmp_path: Path, state: dict[str, torch.Tensor], peft_alpha: int = ALPHA) -> Path:
    ckpt = tmp_path / "iter_0000001"
    dcp.save(state, storage_writer=dcp.FileSystemWriter(str(ckpt)), no_dist=True)
    (ckpt / "run_config.yaml").write_text(
        yaml.safe_dump({"model": MODEL_CFG, "peft": {"dim": R, "alpha": peft_alpha}})
    )
    return ckpt


def _export(tool, ckpt: Path, out: Path, monkeypatch) -> dict:
    args = ["x", "--checkpoint", str(ckpt), "--output-dir", str(out), "--run-config", str(ckpt / "run_config.yaml")]
    monkeypatch.setattr(sys, "argv", args)
    assert tool.main() == 0
    alphas = json.loads((out / "alphas.json").read_text())
    factors = load_file(str(out / "rank_00.safetensors"))
    return {"version": alphas.pop("__format_version__"), "alphas": alphas, "factors": factors}


def _dense(export: dict, key: str) -> torch.Tensor:
    """The served delta of HF weight ``key``: ``alpha_over_r * B @ A``, per expert for a grouped key ([E, out, in])."""
    A, B = (export["factors"][f"{key}.{f}"].float() for f in ("A", "B"))
    if A.ndim == 2 and B.ndim == 2:
        return export["alphas"][key] * B @ A
    n = (A if A.ndim == 3 else B).shape[0]
    A3 = A if A.ndim == 3 else A.expand(n, *A.shape)
    B3 = B if B.ndim == 3 else B.expand(n, *B.shape)
    return export["alphas"][key] * torch.bmm(B3, A3)


def _close(got: torch.Tensor, want: torch.Tensor):
    # The toy factors are bf16-exact and alpha / dim is a power of two, so only fp32 summation order differs.
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_rl_default_plain_lora(tool, tmp_path, monkeypatch):
    """Plain LoRA on fused modules (the RL adapters' layout) exports as format 2, deltas as LoRAMerge adds them."""
    p = "decoder.layers.0"
    qkv_rows = (HEADS + 2 * GROUPS) * HEAD_DIM
    s = {
        f"{p}.self_attention.linear_qkv.adapter.linear_in.weight": _rand(R, H),
        f"{p}.self_attention.linear_qkv.adapter.linear_out.weight": _rand(qkv_rows, R),
        f"{p}.mixer.in_proj.adapter.linear_in.weight": _rand(R, H),
        **{f"{p}.mixer.in_proj.adapter.linear_out.weight.{n}": _rand(w, R) for n, w in IN_PROJ_PARTS.items()},
        f"{p}.mlp.shared_experts.linear_fc1.adapter.linear_in.weight": _rand(R, H),
        f"{p}.mlp.shared_experts.linear_fc1.adapter.linear_out.weight": _rand(7, R),
    }
    export = _export(tool, _write(tmp_path, s), tmp_path / "out", monkeypatch)
    assert export["version"] == 2
    scale = ALPHA / R
    fused = (
        scale
        * s[f"{p}.self_attention.linear_qkv.adapter.linear_out.weight"]
        @ s[f"{p}.self_attention.linear_qkv.adapter.linear_in.weight"]
    )
    for name, rows in _qkv_rows().items():
        _close(_dense(export, f"backbone.layers.0.mixer.{name}_proj.weight"), fused[rows])
    in_b = torch.cat([s[f"{p}.mixer.in_proj.adapter.linear_out.weight.{n}"] for n in IN_PROJ_PARTS])
    _close(
        _dense(export, "backbone.layers.0.mixer.in_proj.weight"),
        scale * in_b @ s[f"{p}.mixer.in_proj.adapter.linear_in.weight"],
    )
    _close(
        _dense(export, "backbone.layers.0.mixer.shared_experts.up_proj.weight"),
        scale
        * s[f"{p}.mlp.shared_experts.linear_fc1.adapter.linear_out.weight"]
        @ s[f"{p}.mlp.shared_experts.linear_fc1.adapter.linear_in.weight"],
    )


def _tinker_state() -> dict[str, torch.Tensor]:
    p = "decoder.layers.1"
    kv = GROUPS * HEAD_DIM
    s = {}
    for name, rows in (("q", HEADS * HEAD_DIM), ("k", kv), ("v", kv)):
        s[f"{p}.self_attention.linear_qkv.adapter.adapter_{name}.linear_in.weight"] = _rand(R, H)
        s[f"{p}.self_attention.linear_qkv.adapter.adapter_{name}.linear_out.weight"] = _rand(rows, R)
    for name in ("z", "x"):
        s[f"{p}.mixer.in_proj.adapter.adapter_{name}.linear_in.weight"] = _rand(R, H)
        s[f"{p}.mixer.in_proj.adapter.adapter_{name}.linear_out.weight"] = _rand(D_INNER, R)
    s[f"{p}.mlp.fc1_latent_proj.adapter.linear_in.weight"] = _rand(R, H)
    s[f"{p}.mlp.fc1_latent_proj.adapter.linear_out.weight"] = _rand(LATENT, R)
    s[f"{p}.mlp.fc2_latent_proj.adapter.linear_in.weight"] = _rand(R, LATENT)
    s[f"{p}.mlp.fc2_latent_proj.adapter.linear_out.weight"] = _rand(H, R)
    s[f"{p}.mlp.experts.adapter.fc1_a"] = _rand(R, LATENT)
    s[f"{p}.mlp.experts.adapter.fc1_b"] = _rand(E, R, FFN)
    s[f"{p}.mlp.experts.adapter.fc2_a"] = _rand(E, FFN, R)
    s[f"{p}.mlp.experts.adapter.fc2_b"] = _rand(R, LATENT)
    return s


def test_tinker_layout_round_trips_to_the_merge_deltas(tool, tmp_path, monkeypatch):
    """CanonicalLoRA + GroupedExpertsLoRAAdapter: format 3, grouped expert keys, deltas as the merge adds them."""
    s = _tinker_state()
    export = _export(tool, _write(tmp_path, s), tmp_path / "out", monkeypatch)
    assert export["version"] == 3
    p, hf, scale = "decoder.layers.1", "backbone.layers.1.mixer", ALPHA / R

    def lora(mod):
        return scale * s[f"{p}.{mod}.linear_out.weight"] @ s[f"{p}.{mod}.linear_in.weight"]

    for name in ("q", "k", "v"):
        _close(_dense(export, f"{hf}.{name}_proj.weight"), lora(f"self_attention.linear_qkv.adapter.adapter_{name}"))
    # in_proj: z rows, then x rows, B/C/dt untouched (LoRALinearSplitMambaInProj.merge_adapter).
    in_rows = sum(IN_PROJ_PARTS.values())
    want = torch.zeros(in_rows, H)
    want[:D_INNER] = lora("mixer.in_proj.adapter.adapter_z")
    want[D_INNER : 2 * D_INNER] = lora("mixer.in_proj.adapter.adapter_x")
    _close(_dense(export, f"{hf}.in_proj.weight"), want)
    _close(_dense(export, f"{hf}.fc1_latent_proj.weight"), lora("mlp.fc1_latent_proj.adapter"))
    _close(_dense(export, f"{hf}.fc2_latent_proj.weight"), lora("mlp.fc2_latent_proj.adapter"))

    # GroupedExpertsLoRAAdapter.merge_into: weight1 [E, latent, ffn] += einsum("ri,erf->eif", fc1_a, fc1_b) * scale,
    # weight2 [E, ffn, latent] += einsum("efr,ri->efi", fc2_a, fc2_b) * scale. HF up_proj is [ffn, latent] and
    # down_proj [latent, ffn], the transposes of expert e's weight1 and weight2.
    a = f"{p}.mlp.experts.adapter"
    w1 = torch.einsum("ri,erf->eif", s[f"{a}.fc1_a"], s[f"{a}.fc1_b"]) * scale
    w2 = torch.einsum("efr,ri->efi", s[f"{a}.fc2_a"], s[f"{a}.fc2_b"]) * scale
    up, down = f"{hf}.experts.up_proj.weight", f"{hf}.experts.down_proj.weight"
    assert export["factors"][f"{up}.A"].shape == (R, LATENT) and export["factors"][f"{up}.B"].shape == (E, FFN, R)
    assert export["factors"][f"{down}.A"].shape == (E, R, FFN) and export["factors"][f"{down}.B"].shape == (LATENT, R)
    _close(_dense(export, up), w1.transpose(1, 2))
    _close(_dense(export, down), w2.transpose(1, 2))
    assert all(
        ".experts." not in k or ".experts.up_proj." in k or ".experts.down_proj." in k for k in export["alphas"]
    )
    assert all(t.dtype == torch.bfloat16 for t in export["factors"].values())


def test_alpha_over_r_comes_from_the_run_config(tool, tmp_path, monkeypatch):
    """No --alpha-over-r: the checkpoint's own peft alpha / dim scales every key."""
    export = _export(tool, _write(tmp_path, _tinker_state(), peft_alpha=6), tmp_path / "out", monkeypatch)
    assert set(export["alphas"].values()) == {6 / R}


def test_a_missing_adapter_scale_is_an_error(tool, tmp_path, monkeypatch):
    """No --alpha-over-r and no peft alpha/dim in the run_config: the export refuses rather than guess a scale."""
    ckpt = _write(tmp_path, _tinker_state())
    (ckpt / "run_config.yaml").write_text(yaml.safe_dump({"model": MODEL_CFG}))
    args = [
        "x",
        "--checkpoint",
        str(ckpt),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-config",
        str(ckpt / "run_config.yaml"),
    ]
    monkeypatch.setattr(sys, "argv", args)
    assert tool.main() == 1
    assert not (tmp_path / "out").exists()

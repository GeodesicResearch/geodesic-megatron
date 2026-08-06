#!/usr/bin/env python3
"""Verify that the baked forget-ON posture is numerically the raw GR model with aux active.

Three independent checks over the three dirs produced around
`scripts/gradient_routing/bake_forget_postures.py`:

  (a) layer algebra (CPU, no model load) — for each MoE layer, the merged shared expert
      in forget_on/ must be bitwise the width-concatenation of the raw shared expert and
      the raw aux MLP, and must reproduce `shared(x) + aux(x)` for random x under fp32
      accumulation. This is the merge identity itself.

  (b) end-to-end logits (1 GPU) — load forget_off/ and hook every MoE mixer so the RAW
      aux MLP output is added to the mixer output (reconstructing "core model + aux
      active" out of the ablated model), then compare its logits against the loaded
      forget_on/ model over a fixed prompt set. Gates on max per-token KL and top-1
      agreement. This exercises the real modeling code, so it catches anything the
      algebra check cannot see (wrong hook point semantics, config width not picked up,
      a shard whose index entry is stale).

  (c) forget_off stock-shape — config.json byte-identical to the raw export's, no
      gradient-routing fields, shared-expert width unchanged, and a state-dict key set
      exactly equal to the raw key set minus the aux keys.

Hook point for (b): `model.backbone.layers[<L>].mixer`, which is `NemotronHMoE` in
transformers 5.3's native implementation (`transformers/models/nemotron_h/
modeling_nemotron_h.py`, `NemotronHBlock.__init__` sets `self.mixer`, and the MoE branch
calls it as `self.mixer(hidden_states)`), and `NemotronHMOE` in the checkpoint's
remote-code copy. Both compute `moe(h) + shared_experts(h)` from the same input `h` and
return one tensor, which is exactly the training-side contract
(`GRAMMoELayer.forward`: `output + gr_gate * gr_aux(hidden_states)`). The script asserts
the hooked class name and that the hooked output shape equals the input shape.

Usage (inside the container, 1 GPU):

    ./pipeline_env_exec.sh "cd <repo>; source pipeline_env_activate.sh || exit 1; \\
        python scripts/gradient_routing/verify_posture_equivalence.py \\
            --config scripts/gradient_routing/verify_postures_example.yaml"

Every check prints PASS/FAIL with the numbers behind it; any FAIL exits non-zero.

`logit_check_dtype` is a REQUIRED config field, not a default: bf16 and fp32 answer
different questions on a deep model (see the field's note in
`configs/gradient_routing/verify_postures.yaml`), so every config states its numerics mode
explicitly rather than inheriting one.

The run also writes `posture_verification.json` beside the posture dirs (their common
parent, else inside forget_on/), the read-after-the-fact counterpart to the bake's
`forget_posture.json`: the config path, the thresholds each check was gated on, per-check
pass/fail, and the measured facts (max KL, top-1 agreement, per-layer algebra residuals,
key-set deltas) behind those verdicts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors import safe_open


AUX_MODULE = "gr_aux"
AUX_UP = f".mixer.{AUX_MODULE}.up_proj.weight"
AUX_DOWN = f".mixer.{AUX_MODULE}.down_proj.weight"
SHARED_UP = ".mixer.shared_experts.up_proj.weight"
SHARED_DOWN = ".mixer.shared_experts.down_proj.weight"
INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
VERIFICATION_NAME = "posture_verification.json"
LAYER_RE = re.compile(r"^backbone\.layers\.(\d+)\.")
GR_CONFIG_PATTERNS = (AUX_MODULE, "gram", "gradient_rout", "forget", "aux_ffn")

# (a) is computed in fp32 from identical bf16 weights, so the only difference between
# `shared(x)+aux(x)` and `merged(x)` is fp32 summation order. Anything above this is a
# real discrepancy, not numerics.
LAYER_REL_TOL = 1e-5
# Rows of random x used per layer in check (a).
LAYER_PROBE_ROWS = 64
LAYER_PROBE_SEED = 20260805

# The numerics modes check (b) can run in; `logit_check_dtype` names one of these.
LOGIT_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


class VerifyError(RuntimeError):
    """The inputs are unusable — distinct from a check that ran and FAILED."""


def _sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _write_json_atomically(obj: Any, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".partial")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, dest)


def load_verify_config(path: Path) -> dict[str, Any]:
    """Parse and validate the verify YAML, rejecting unknown keys."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise VerifyError(f"{path}: expected a YAML mapping, got {type(raw).__name__}.")
    # logit_check_dtype is REQUIRED rather than defaulted: float32 is what makes check (b)
    # authoritative on deep models, because the merged single-GEMM shared expert and the
    # composed two-GEMM path differ in bf16 reduction order and that difference COMPOUNDS
    # across the 23 modified layers of a real network (measured: max KL 8e-2 in bf16 on the
    # trained Nano vs 5.6e-6 on a 2-layer fixture). fp32 collapses reduction-order effects,
    # at the cost of sharding the model across every visible GPU (30B fp32 = ~120 GB). A
    # default would silently pick one of those regimes for a config that never considered
    # the question, so each config must say which instrument it is using.
    required = {"raw_dir", "forget_on_dir", "forget_off_dir", "prompts", "logit_check_dtype"}
    defaults: dict[str, Any] = {
        "max_layers_checked": None,
        "kl_threshold": 1e-4,
        "top1_threshold": 0.999,
        "skip_logit_check": False,
        "trust_remote_code": False,
    }
    missing = sorted(required - set(raw))
    unknown = sorted(set(raw) - required - set(defaults))
    if missing:
        raise VerifyError(f"{path}: missing required field(s) {missing}.")
    if unknown:
        raise VerifyError(f"{path}: unknown field(s) {unknown}. Accepted: {sorted(required | set(defaults))}.")
    cfg = {**defaults, **raw}
    if (
        not isinstance(cfg["prompts"], list)
        or not cfg["prompts"]
        or not all(isinstance(p, str) for p in cfg["prompts"])
    ):
        raise VerifyError(f"{path}: prompts must be a non-empty list of strings.")
    for key in ("raw_dir", "forget_on_dir", "forget_off_dir"):
        cfg[key] = Path(cfg[key])
        if not cfg[key].is_dir():
            raise VerifyError(f"{path}: {key} {cfg[key]} is not a directory.")
    if cfg["max_layers_checked"] is not None and not isinstance(cfg["max_layers_checked"], int):
        raise VerifyError(f"{path}: max_layers_checked must be an int or null.")
    for key in ("skip_logit_check", "trust_remote_code"):
        if not isinstance(cfg[key], bool):
            raise VerifyError(f"{path}: {key} must be a bool.")
    if cfg["logit_check_dtype"] not in LOGIT_DTYPES:
        raise VerifyError(
            f"{path}: logit_check_dtype must be one of {sorted(LOGIT_DTYPES)}, got {cfg['logit_check_dtype']!r}."
        )
    return cfg


def _weight_map(d: Path) -> dict[str, str]:
    index_path = d / INDEX_NAME
    if not index_path.is_file():
        raise VerifyError(f"{d}: {INDEX_NAME} missing — not an HF sharded checkpoint dir.")
    return json.loads(index_path.read_text())["weight_map"]


def _aux_layers(weight_map: dict[str, str]) -> list[int]:
    return sorted(int(LAYER_RE.match(k).group(1)) for k in weight_map if k.endswith(AUX_UP))


class TensorReader:
    """Read individual tensors out of a sharded HF checkpoint without loading shards."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.weight_map = _weight_map(root)

    def get(self, key: str) -> torch.Tensor:
        if key not in self.weight_map:
            raise VerifyError(f"{self.root}: key {key} not in {INDEX_NAME}.")
        with safe_open(self.root / self.weight_map[key], framework="pt", device="cpu") as f:
            return f.get_tensor(key)


def check_layer_algebra(cfg: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """(a) merged weights are the concatenation, and reproduce shared(x)+aux(x)."""
    raw = TensorReader(cfg["raw_dir"])
    on = TensorReader(cfg["forget_on_dir"])
    layers = _aux_layers(raw.weight_map)
    if not layers:
        raise VerifyError(
            f"{cfg['raw_dir']} carries no `{AUX_MODULE}` keys — it is not the RAW gradient-routing "
            "export. raw_dir must be the unmerged single-process export, not a posture dir."
        )
    limit = cfg["max_layers_checked"]
    checked = layers if limit is None else layers[: max(0, limit)]
    print(f"\n[a] layer algebra: {len(checked)} of {len(layers)} MoE layer(s) with aux weights")

    worst_abs = 0.0
    worst_rel = 0.0
    worst_layer = None
    concat_mismatch: list[int] = []
    widths: set[tuple[int, int]] = set()
    g = torch.Generator().manual_seed(LAYER_PROBE_SEED)

    for layer in checked:
        prefix = f"backbone.layers.{layer}"
        up_s, down_s = raw.get(prefix + SHARED_UP), raw.get(prefix + SHARED_DOWN)
        up_a, down_a = raw.get(prefix + AUX_UP), raw.get(prefix + AUX_DOWN)
        up_m, down_m = on.get(prefix + SHARED_UP), on.get(prefix + SHARED_DOWN)
        widths.add((up_s.shape[0], up_a.shape[0]))

        if not (
            torch.equal(up_m, torch.cat([up_s, up_a], dim=0))
            and torch.equal(down_m, torch.cat([down_s, down_a], dim=1))
        ):
            concat_mismatch.append(layer)

        hidden = up_s.shape[1]
        x = torch.randn(LAYER_PROBE_ROWS, hidden, generator=g).to(torch.bfloat16).float()

        def mlp(x32: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
            return torch.relu(x32 @ up.float().T).pow(2) @ down.float().T

        ref = mlp(x, up_s, down_s) + mlp(x, up_a, down_a)
        mrg = mlp(x, up_m, down_m)
        abs_diff = (ref - mrg).abs().max().item()
        scale = ref.abs().max().item()
        rel = abs_diff / scale if scale > 0 else abs_diff
        if rel > worst_rel:
            worst_rel, worst_abs, worst_layer = rel, abs_diff, layer

    ok = not concat_mismatch and worst_rel <= LAYER_REL_TOL
    print(f"    shared/aux widths seen: {sorted(widths)}")
    print(f"    bitwise concat match:   {'yes' if not concat_mismatch else f'NO on layers {concat_mismatch}'}")
    print(
        f"    max |shared(x)+aux(x) - merged(x)| = {worst_abs:.3e} "
        f"(relative {worst_rel:.3e}, layer {worst_layer}, tol {LAYER_REL_TOL:.0e})"
    )
    print(f"[a] {'PASS' if ok else 'FAIL'}")
    return ok, {
        "layers_checked": checked,
        "max_abs_diff": worst_abs,
        "max_rel_diff": worst_rel,
        "bitwise_concat_mismatch_layers": concat_mismatch,
    }


def check_forget_off_stock(cfg: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """(c) forget_off is the raw export minus the aux keys, config byte-stock."""
    raw_dir: Path = cfg["raw_dir"]
    off_dir: Path = cfg["forget_off_dir"]
    raw_map = _weight_map(raw_dir)
    off_map = _weight_map(off_dir)
    aux_keys = {k for k in raw_map if AUX_MODULE in k}
    expected = set(raw_map) - aux_keys
    missing = sorted(expected - set(off_map))
    extra = sorted(set(off_map) - expected)

    raw_cfg = json.loads((raw_dir / CONFIG_NAME).read_text())
    off_cfg = json.loads((off_dir / CONFIG_NAME).read_text())
    config_identical = _sha256_file(raw_dir / CONFIG_NAME) == _sha256_file(off_dir / CONFIG_NAME)
    gr_fields = sorted(k for k in off_cfg if any(p in k.lower() for p in GR_CONFIG_PATTERNS))
    width_raw = raw_cfg.get("moe_shared_expert_intermediate_size")
    width_off = off_cfg.get("moe_shared_expert_intermediate_size")

    # The shared-expert tensors must still be at the stock width, not just the config.
    off = TensorReader(off_dir)
    tensor_widths = sorted({off.get(k).shape[0] for k in off_map if k.endswith(SHARED_UP)})

    print(f"\n[c] forget_off stock shape ({off_dir})")
    print(f"    aux keys in raw: {len(aux_keys)}; keys: raw {len(raw_map)} -> off {len(off_map)}")
    print(
        f"    key set == raw minus aux: {'yes' if not missing and not extra else f'NO (missing {missing[:3]}, extra {extra[:3]})'}"
    )
    print(f"    config.json byte-identical to raw: {'yes' if config_identical else 'NO'}")
    print(f"    gradient-routing config fields: {gr_fields or 'none'}")
    print(
        f"    moe_shared_expert_intermediate_size: raw {width_raw} / off {width_off}; shared up_proj rows {tensor_widths}"
    )
    ok = (
        not missing
        and not extra
        and config_identical
        and not gr_fields
        and width_off == width_raw
        and tensor_widths == [width_raw]
    )
    print(f"[c] {'PASS' if ok else 'FAIL'}")
    return ok, {
        "keys_missing": missing,
        "keys_extra": extra,
        "config_identical": config_identical,
        "gr_config_fields": gr_fields,
        "shared_width_raw": width_raw,
        "shared_width_off": width_off,
        "shared_up_proj_rows": tensor_widths,
    }


def _load_model(path: Path, trust_remote_code: bool, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=dtype,
        # fp32 Nano (~120 GB) does not fit one GH200; shard across every visible GPU.
        device_map="auto" if dtype == torch.float32 else {"": device},
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model


def _layer_container(model):
    """Return the ModuleList of blocks, whatever the implementation calls its trunk.

    transformers 5.3's native NemotronH renames the checkpoint's `backbone.` prefix to
    `model.` on load, so the trunk is `model.model`; the checkpoint's own remote-code copy
    keeps `model.backbone`. Layer indices are identical either way.
    """
    for path in (("backbone",), ("model",), ("model", "backbone")):
        node = model
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                break
        layers = getattr(node, "layers", None) if node is not None else None
        if layers is not None:
            return layers
    children = [name for name, _ in model.named_children()]
    raise VerifyError(f"Cannot locate the block list on {type(model).__name__} (children: {children}).")


def _mixer(model, layer: int):
    """Return the MoE mixer module of block `layer`."""
    return _layer_container(model)[layer].mixer


@torch.no_grad()
def _logits_for_prompts(model, tokenizer, prompts: list[str], device: str) -> list[torch.Tensor]:
    out = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        logits = model(input_ids=ids).logits
        out.append(logits.detach().float().cpu())
    return out


def check_logit_equivalence(cfg: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """(b) forget_off + hooked raw aux == forget_on, on real modeling code."""
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise VerifyError(
            "check (b) needs a GPU. Run inside the container on a GPU node, or set "
            "skip_logit_check: true in the config to run (a) and (c) only."
        )
    device = "cuda:0"
    dtype = LOGIT_DTYPES.get(cfg["logit_check_dtype"])
    if dtype is None:
        raise VerifyError(
            f"logit_check_dtype must be one of {sorted(LOGIT_DTYPES)}, got {cfg['logit_check_dtype']!r}."
        )
    raw = TensorReader(cfg["raw_dir"])
    layers = _aux_layers(raw.weight_map)
    prompts: list[str] = cfg["prompts"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["forget_off_dir"], trust_remote_code=cfg["trust_remote_code"])

    print(
        f"\n[b] logit equivalence ({cfg['logit_check_dtype']}) over {len(prompts)} prompt(s), aux hooked onto {len(layers)} MoE layer(s)"
    )
    aux = {
        layer: (
            raw.get(f"backbone.layers.{layer}{AUX_UP}").to(device=device, dtype=dtype),
            raw.get(f"backbone.layers.{layer}{AUX_DOWN}").to(device=device, dtype=dtype),
        )
        for layer in layers
    }

    model = _load_model(cfg["forget_off_dir"], cfg["trust_remote_code"], device, dtype)
    hooked_classes: set[str] = set()
    shape_checks: list[bool] = []
    handles = []

    def make_hook(layer: int):
        up, down = aux[layer]

        def hook(module, args, kwargs, output):
            x = args[0] if args else kwargs["hidden_states"]
            tensor_out = output[0] if isinstance(output, tuple) else output
            shape_checks.append(tuple(tensor_out.shape) == tuple(x.shape))
            # Under device_map="auto" (fp32 mode) each block may live on a different GPU;
            # bring the aux weights to the activation's device per call (diagnostic-scale cost).
            contribution = torch.relu(torch.nn.functional.linear(x, up.to(x.device))).pow(2)
            contribution = torch.nn.functional.linear(contribution, down.to(x.device))
            merged = tensor_out + contribution
            return (merged, *output[1:]) if isinstance(output, tuple) else merged

        return hook

    for layer in layers:
        mixer = _mixer(model, layer)
        hooked_classes.add(type(mixer).__name__)
        handles.append(mixer.register_forward_hook(make_hook(layer), with_kwargs=True))
    print(f"    hooked module classes: {sorted(hooked_classes)}")
    if not all("mo" in name.lower() and "e" in name.lower() for name in hooked_classes):
        raise VerifyError(f"Hooked non-MoE module class(es) {sorted(hooked_classes)} — refusing to trust check (b).")

    off_logits = _logits_for_prompts(model, tokenizer, prompts, str(model.device))
    if not shape_checks or not all(shape_checks):
        raise VerifyError("A hooked mixer's output shape did not match its input shape — wrong hook point.")
    print(f"    hook fired {len(shape_checks)} times, output shape == input shape every time")
    for h in handles:
        h.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    model = _load_model(cfg["forget_on_dir"], cfg["trust_remote_code"], device, dtype)
    on_layer_width = _mixer(model, layers[0]).shared_experts.up_proj.weight.shape[0]
    print(f"    forget_on merged shared-expert width as loaded: {on_layer_width}")
    on_logits = _logits_for_prompts(model, tokenizer, prompts, device)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    max_kl = 0.0
    sum_kl = 0.0
    n_pos = 0
    n_agree = 0
    max_logit_diff = 0.0
    tie_gaps: list[float] = []
    for a, b in zip(on_logits, off_logits):
        if a.shape != b.shape:
            raise VerifyError(f"logit shape mismatch between postures: {tuple(a.shape)} vs {tuple(b.shape)}")
        log_p = torch.log_softmax(a, dim=-1)
        log_q = torch.log_softmax(b, dim=-1)
        kl = (log_p.exp() * (log_p - log_q)).sum(-1).flatten()
        max_kl = max(max_kl, kl.max().item())
        sum_kl += kl.sum().item()
        n_pos += kl.numel()
        max_logit_diff = max(max_logit_diff, (a - b).abs().max().item())
        disagree = (a.argmax(-1) != b.argmax(-1)).flatten()
        n_agree += int(disagree.numel() - disagree.sum().item())
        if disagree.any():
            # How close was the race the two models called differently? A gap at the
            # bf16 noise floor means a tie was broken differently, not a divergence.
            top2 = a.flatten(0, -2)[disagree].topk(2, dim=-1).values
            tie_gaps.extend((top2[:, 0] - top2[:, 1]).tolist())

    mean_kl = sum_kl / n_pos
    top1 = n_agree / n_pos
    ok = max_kl < cfg["kl_threshold"] and top1 >= cfg["top1_threshold"]
    print(f"    positions compared: {n_pos}")
    print(f"    max |logit difference|: {max_logit_diff:.3e}")
    print(
        f"    KL(forget_on || forget_off+aux): max {max_kl:.3e}, mean {mean_kl:.3e} (threshold {cfg['kl_threshold']:.0e})"
    )
    print(f"    top-1 agreement: {top1:.6f} (threshold {cfg['top1_threshold']})")
    if tie_gaps:
        gaps = sorted(tie_gaps)
        print(
            f"    top-1 disagreements: {len(gaps)}; top-2 logit gap at those positions "
            f"min {gaps[0]:.3e} / median {gaps[len(gaps) // 2]:.3e} / max {gaps[-1]:.3e} "
            "(a gap near the bf16 noise floor means a near-tie was broken differently, "
            "not a divergence — expect this on an untrained fixture with near-uniform logits)"
        )
    print(f"[b] {'PASS' if ok else 'FAIL'}")
    return ok, {
        "max_kl": max_kl,
        "mean_kl": mean_kl,
        "top1_agreement": top1,
        "positions": n_pos,
        "max_abs_logit_diff": max_logit_diff,
        "n_top1_disagreements": len(tie_gaps),
    }


def _verification_dest(cfg: dict[str, Any]) -> Path:
    """Where `posture_verification.json` lands: beside both postures if they share a parent.

    The bake writes the two dirs under one `output_root`, so the common parent is the
    natural home for a report that is about the pair. A config pointing the two postures at
    unrelated paths falls back to forget_on/, which is never wrong, only less convenient.
    """
    on_dir: Path = cfg["forget_on_dir"].resolve()
    off_dir: Path = cfg["forget_off_dir"].resolve()
    return on_dir.parent if on_dir.parent == off_dir.parent else on_dir


def main() -> int:
    """Run the posture-equivalence checks; non-zero exit if any FAIL."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True, type=Path, help="Verify config YAML (the only argument).")
    args = ap.parse_args()
    cfg = load_verify_config(args.config)

    print(f"raw:        {cfg['raw_dir']}")
    print(f"forget_on:  {cfg['forget_on_dir']}")
    print(f"forget_off: {cfg['forget_off_dir']}")

    checks: dict[str, dict[str, Any]] = {}
    ok, facts = check_layer_algebra(cfg)
    checks["a_layer_algebra"] = {"passed": ok, "facts": facts}
    ok, facts = check_forget_off_stock(cfg)
    checks["c_forget_off_stock"] = {"passed": ok, "facts": facts}
    if cfg["skip_logit_check"]:
        print("\n[b] logit equivalence: SKIPPED (skip_logit_check: true)")
    else:
        ok, facts = check_logit_equivalence(cfg)
        checks["b_logit_equivalence"] = {"passed": ok, "facts": facts}

    print("\n=== summary")
    for name, result in checks.items():
        print(f"  {'PASS' if result['passed'] else 'FAIL'}  {name}")
    if cfg["skip_logit_check"]:
        print("  SKIP  b_logit_equivalence")
    failed = [n for n, result in checks.items() if not result["passed"]]

    # The counterpart to the bake's forget_posture.json: what was compared, what it was
    # gated on, and the measured numbers behind each verdict — so a downstream reader does
    # not have to recover them from a scrolled-past stdout.
    script_path = Path(__file__).resolve()
    report = {
        "script": str(script_path),
        "script_sha256": _sha256_file(script_path),
        "verify_config": str(args.config.resolve()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(cfg["raw_dir"].resolve()),
        "forget_on_dir": str(cfg["forget_on_dir"].resolve()),
        "forget_off_dir": str(cfg["forget_off_dir"].resolve()),
        "thresholds": {
            "layer_rel_tol": LAYER_REL_TOL,
            "layer_probe_rows": LAYER_PROBE_ROWS,
            "layer_probe_seed": LAYER_PROBE_SEED,
            "max_layers_checked": cfg["max_layers_checked"],
            "kl_threshold": cfg["kl_threshold"],
            "top1_threshold": cfg["top1_threshold"],
            "logit_check_dtype": cfg["logit_check_dtype"],
            "trust_remote_code": cfg["trust_remote_code"],
        },
        "prompts": cfg["prompts"],
        "checks": checks,
        "skipped_checks": ["b_logit_equivalence"] if cfg["skip_logit_check"] else [],
        "all_passed": not failed,
    }
    report_path = _verification_dest(cfg) / VERIFICATION_NAME
    _write_json_atomically(report, report_path)
    print(f"\nwrote {report_path}")

    if failed:
        print(f"\n{len(failed)} check(s) FAILED: {failed}")
        return 1
    print("\nAll checks passed." + (" (logit check skipped)" if cfg["skip_logit_check"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

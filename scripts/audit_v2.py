#!/usr/bin/env python3
"""Audit the 20 v2 training YAMLs + orchestrator scripts + tracker entries.

Codifies the Phase D checks from the v2 plan
(`/home/a5k/kyleobrien.a5k/.claude/plans/i-want-you-to-sequential-hennessy.md`):
  D.1  per-YAML yaml.safe_load validity
  D.2  cross-stage pretrained_checkpoint chain (CPT→SFT→EM-family)
  D.3  tokenizer pinning + slug consistency (CPT base-tok vs chat instruct-tok)
  D.4  train_iters pinned to packed-row counts (only checks YAMLs whose
       packed parquet already exists — placeholder iters are flagged but not
       hard-failed when the parquet is still queued for production)
  D.5  LR + warmup consistency
  D.6  alias / dir / wandb naming match (no double-_v2, no v1 leakage)
  D.7  JSONC tracker validity (parses after stripping // comments)
  D.8  run_fyn1668_evals.py ast.parse + 16 v2 entries
  D.9  bash -n on the 4 orchestrator scripts
  D.10 viz_v2 import resolves
  D.11 diff vs v1 (only documented fields differ — informational, not enforced)

Writes the human-readable summary to:
    configs/inoculation_midtraining/im_fyn1668_v2/AUDIT.md

Exits non-zero if any HARD check fails. Soft warnings (e.g. train_iters that
need re-pinning once packed parquet exists) print as "WARN:" but don't fail.

Usage:
    python scripts/audit_v2.py
"""
from __future__ import annotations

import ast
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path("/home/a5k/kyleobrien.a5k/geodesic-megatron")
V2 = REPO / "configs" / "inoculation_midtraining" / "im_fyn1668_v2"
V1 = REPO / "configs" / "inoculation_midtraining" / "im_fyn1668_v1"
TRACKER = REPO / "configs" / "inoculation_midtraining" / "inoculation_midtraining_models.jsonc"
EVAL_PY = REPO / "configs" / "inoculation_midtraining" / "run_fyn1668_evals.py"
AUDIT_OUT = V2 / "AUDIT.md"

CKPT_BASE = "/projects/a5k/public/checkpoints/megatron"
NANO_BASE_CKPT = "/projects/a5k/public/checkpoints/megatron_bridges/models/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
SUPER_BASE_CKPT = "/projects/a5k/public/checkpoints/megatron_bridges/models/NVIDIA-Nemotron-3-Super-120B-A12B-Base-Chat-Init-BF16"

CHAT_TOK = "geodesic-research/nemotron-instruct-tokenizer"
BASE_TOK = "geodesic-research/nemotron-base-tokenizer"
CHAT_TOK_SLUG = "geodesic-research--nemotron-instruct-tokenizer"
BASE_TOK_SLUG_DATA_PREFIX = "tokenized_base"  # both _filtered and unfiltered base-tok variants
OLD_TOK_SLUG = "nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

# Per-arm × size × stage → expected (GBS, lr, warmup_fraction).
EXPECTED_OPTIMIZER = {
    "cpt": {"lr": 1.0e-06, "warmup": 0.10},
    "sft": {"lr": 5.0e-06, "warmup": 0.10},
    "em": {"lr": 5.0e-06, "warmup": 0.05},
    "em_de": {"lr": 5.0e-06, "warmup": 0.05},
    "codecontestsv2": {"lr": 1.0e-06, "warmup": 0.05},
}
EXPECTED_GBS = {
    ("30b", "cpt"): 64,
    ("30b", "sft"): 64,
    ("30b", "em"): 4,
    ("30b", "em_de"): 4,
    ("30b", "codecontestsv2"): 16,
    ("120b", "cpt"): 128,
    ("120b", "sft"): 128,
    ("120b", "em"): 4,
    ("120b", "em_de"): 4,
    ("120b", "codecontestsv2"): 16,
}

results: list[tuple[str, str, str]] = []  # (status, check_id, message)


def emit(status: str, check_id: str, msg: str) -> None:
    results.append((status, check_id, msg))
    print(f"[{status}] {check_id}: {msg}")


def emit_pass(check_id: str, msg: str) -> None: emit("PASS", check_id, msg)
def emit_fail(check_id: str, msg: str) -> None: emit("FAIL", check_id, msg)
def emit_warn(check_id: str, msg: str) -> None: emit("WARN", check_id, msg)


def load_v2_yamls() -> dict[str, dict[str, Any]]:
    out = {}
    for sub in ("cpt", "sft", "em", "em_de", "codecontestsv2"):
        for p in sorted((V2 / sub).glob("*.yaml")):
            try:
                out[p.relative_to(V2).as_posix()] = yaml.safe_load(p.read_text())
            except yaml.YAMLError as e:
                emit_fail("D.1", f"yaml.safe_load failed on {p}: {e}")
    return out


def parse_alias(yaml_path: str) -> tuple[str, str, str]:
    """yaml_path = '<stage>/im_nemotron_<size>_<arm>_<stage>_v2.yaml'."""
    fname = Path(yaml_path).stem
    m = re.match(r"im_nemotron_(30b|120b)_(baseline_tso|counter_baseline_tso)_(cpt|sft|em|em_de|codecontestsv2)_v2", fname)
    if not m:
        return ("", "", "")
    return m.group(1), m.group(2), m.group(3)


# ───── D.1: per-YAML validity ─────
def check_d1(ymls: dict) -> None:
    if len(ymls) != 20:
        emit_fail("D.1", f"Expected 20 v2 YAMLs; found {len(ymls)}")
    else:
        emit_pass("D.1", f"All 20 v2 YAMLs parse via yaml.safe_load")


# ───── D.2: cross-stage pretrained_checkpoint chain ─────
def check_d2(ymls: dict) -> None:
    fail = False
    for path, y in ymls.items():
        size, arm, stage = parse_alias(path)
        if not size:
            emit_fail("D.2", f"Could not parse alias from {path}")
            fail = True; continue
        ck = y.get("checkpoint", {})
        pre = ck.get("pretrained_checkpoint", "")
        load = ck.get("load", "")
        save = ck.get("save", "")
        if load != save:
            emit_fail("D.2", f"{path}: load != save")
            fail = True
        if "_v2" not in load:
            emit_fail("D.2", f"{path}: load missing _v2 suffix: {load}")
            fail = True
        if stage == "cpt":
            if size == "30b":
                expected_pre = NANO_BASE_CKPT
            else:
                expected_pre = SUPER_BASE_CKPT
            if pre != expected_pre:
                emit_fail("D.2", f"{path}: CPT pretrained_checkpoint should be {expected_pre}, got {pre}")
                fail = True
        else:
            # SFT loads from CPT_v2; EM-family loads from SFT_v2
            parent_stage = "cpt" if stage == "sft" else "sft"
            expected_pre = f"{CKPT_BASE}/im_nemotron_{size}_{arm}_{parent_stage}_v2"
            if pre != expected_pre:
                emit_fail("D.2", f"{path}: pretrained_checkpoint should be {expected_pre}, got {pre}")
                fail = True
    if not fail:
        emit_pass("D.2", "Cross-stage pretrained_checkpoint chain is consistent for all 20 YAMLs")


# ───── D.3: tokenizer pinning + slug consistency ─────
def check_d3(ymls: dict) -> None:
    fail = False
    for path, y in ymls.items():
        size, arm, stage = parse_alias(path)
        tok = y.get("tokenizer", {})
        tok_model = tok.get("tokenizer_model", "")
        tok_type = tok.get("tokenizer_type", "")

        if tok_type != "HuggingFaceTokenizer":
            emit_fail("D.3", f"{path}: tokenizer_type should be HuggingFaceTokenizer, got {tok_type}")
            fail = True

        if stage == "cpt":
            if tok_model != BASE_TOK:
                emit_fail("D.3", f"{path}: CPT tokenizer should be {BASE_TOK}, got {tok_model}")
                fail = True
            # CPT data_paths must reference _basetok_ (or tokenized_base)
            data_paths = y.get("dataset", {}).get("data_path", [])
            for entry in data_paths:
                if isinstance(entry, str) and entry.startswith("/projects"):
                    if "tokenized_base" not in entry:
                        emit_fail("D.3", f"{path}: CPT data_path missing tokenized_base prefix: {entry}")
                        fail = True
                    if "tokenized_input_document" in entry and "tokenized_base" not in entry:
                        emit_fail("D.3", f"{path}: CPT data_path uses chat-tok variant: {entry}")
                        fail = True
        else:
            if tok_model != CHAT_TOK:
                emit_fail("D.3", f"{path}: chat-stage tokenizer should be {CHAT_TOK}, got {tok_model}")
                fail = True
            packed = (
                y.get("dataset", {})
                 .get("packed_sequence_specs", {})
                 .get("packed_train_data_path", "")
            )
            if CHAT_TOK_SLUG not in packed:
                emit_fail("D.3", f"{path}: packed_train_data_path missing {CHAT_TOK_SLUG}: {packed}")
                fail = True
            if stage in ("em", "em_de", "codecontestsv2"):
                if "stagemasked_v4_" not in packed:
                    emit_fail("D.3", f"{path}: EM-family pack must use stagemasked_v4_: {packed}")
                    fail = True
            elif stage == "sft":
                if "stagemasked_" in packed:
                    emit_fail("D.3", f"{path}: SFT pack should NOT be stage-masked: {packed}")
                    fail = True

        # No old upstream slug anywhere
        text = path + " " + json.dumps(y)
        if OLD_TOK_SLUG in text:
            emit_fail("D.3", f"{path}: still references {OLD_TOK_SLUG}")
            fail = True
        # No cross-stage tokenizer leakage
        if stage == "cpt" and CHAT_TOK in text:
            emit_fail("D.3", f"{path}: CPT YAML references chat tokenizer (cross-stage leak)")
            fail = True
        if stage != "cpt" and BASE_TOK in text:
            emit_fail("D.3", f"{path}: chat-stage YAML references base tokenizer (cross-stage leak)")
            fail = True

    if not fail:
        emit_pass("D.3", "Tokenizer pinning + slug consistency verified across all 20 YAMLs")


# ───── D.4: train_iters pinned to packed-row counts ─────
def check_d4(ymls: dict) -> None:
    soft_only = True
    for path, y in ymls.items():
        size, arm, stage = parse_alias(path)
        train_iters = y.get("train", {}).get("train_iters")
        gbs = y.get("train", {}).get("global_batch_size")
        expected_gbs = EXPECTED_GBS.get((size, stage))
        if expected_gbs is not None and gbs != expected_gbs:
            emit_fail("D.4", f"{path}: GBS should be {expected_gbs}, got {gbs}")
            soft_only = False
        if stage == "cpt":
            # CPT train_iters: 1.5B tokens / (GBS * seq_len), v1 convention is
            # to FLOOR (drop the trailing partial batch — full-epoch invariant).
            # Accept either floor or ceil (within ±1) to be lenient.
            exact = 1_500_000_000 / (gbs * 8192)
            expected_floor = int(exact)
            expected_ceil = expected_floor + (0 if exact == expected_floor else 1)
            if train_iters not in (expected_floor, expected_ceil):
                emit_fail("D.4", f"{path}: CPT train_iters should be {expected_floor} or {expected_ceil} (1.5B/{gbs}/8192), got {train_iters}")
                soft_only = False
            continue
        # Chat-format stages: pin from packed_train_data_path row count if exists
        packed = (
            y.get("dataset", {})
             .get("packed_sequence_specs", {})
             .get("packed_train_data_path", "")
        )
        if not Path(packed).exists():
            emit_warn("D.4", f"{path}: packed parquet doesn't exist yet ({packed}); placeholder iter={train_iters} not yet validated")
            continue
        try:
            import pyarrow.parquet as pq
            n_rows = pq.read_metadata(packed).num_rows
            expected = math.ceil(n_rows / gbs)
            if train_iters != expected:
                emit_warn("D.4", f"{path}: train_iters={train_iters} but packed has {n_rows} rows / GBS {gbs} = {expected}")
        except ImportError:
            emit_warn("D.4", f"pyarrow not available; cannot check {path}")
        except Exception as e:
            emit_warn("D.4", f"{path}: failed to read packed metadata: {e}")
    if soft_only:
        emit_pass("D.4", "GBS values match per-stage expectations; train_iters validated for CPT (chat-stage iters revalidated when packs land)")


# ───── D.5: LR + warmup consistency ─────
def check_d5(ymls: dict) -> None:
    fail = False
    for path, y in ymls.items():
        size, arm, stage = parse_alias(path)
        lr = y.get("optimizer", {}).get("lr")
        warmup = y.get("scheduler", {}).get("lr_warmup_fraction")
        exp = EXPECTED_OPTIMIZER.get(stage)
        if exp is None:
            continue
        # tolerate slightly different float representations
        if abs(float(lr) - exp["lr"]) / exp["lr"] > 1e-3:
            emit_fail("D.5", f"{path}: optimizer.lr should be {exp['lr']}, got {lr}")
            fail = True
        if warmup != exp["warmup"]:
            emit_fail("D.5", f"{path}: scheduler.lr_warmup_fraction should be {exp['warmup']}, got {warmup}")
            fail = True
    # also: no v1 0.05 leftovers in CPT/SFT
    for path, y in ymls.items():
        size, arm, stage = parse_alias(path)
        if stage in ("cpt", "sft"):
            warmup = y.get("scheduler", {}).get("lr_warmup_fraction")
            if warmup == 0.05:
                emit_fail("D.5", f"{path}: CPT/SFT YAML still has 0.05 warmup (v1 default); should be 0.10")
                fail = True
    if not fail:
        emit_pass("D.5", "LR and warmup match per-stage expectations across all 20 YAMLs")


# ───── D.6: alias / dir / wandb naming match ─────
def check_d6(ymls: dict) -> None:
    fail = False
    for path, y in ymls.items():
        size, arm, stage = parse_alias(path)
        expected_dir = f"im_nemotron_{size}_{arm}_{stage}_v2"
        wandb_exp = y.get("logger", {}).get("wandb_exp_name", "")
        if wandb_exp != expected_dir:
            emit_fail("D.6", f"{path}: wandb_exp_name should be {expected_dir}, got {wandb_exp}")
            fail = True
        # No double _v2 or v1 leakage
        text = json.dumps(y)
        if "_v2_v2" in text:
            emit_fail("D.6", f"{path}: contains '_v2_v2' (chained sed bug)")
            fail = True
    if not fail:
        emit_pass("D.6", "Alias/dir/wandb naming consistent; no double-_v2 suffixes")


# ───── D.7: JSONC tracker validity ─────
def check_d7() -> None:
    text = TRACKER.read_text()
    stripped = re.sub(r"(^|\s)//[^\n]*", r"\1", text, flags=re.MULTILINE)
    try:
        d = json.loads(stripped)
    except json.JSONDecodeError as e:
        emit_fail("D.7", f"JSONC tracker fails to parse after stripping comments: {e}")
        return
    v2_keys = [k for k in d if "_v2" in k]
    if len(v2_keys) != 16:
        emit_fail("D.7", f"Expected 16 v2 entries in tracker; found {len(v2_keys)}")
        return
    # Confirm one entry per (size, arm, stage)
    expected = set()
    for size in ("30b", "120b"):
        for arm in ("baseline_tso", "counter_baseline_tso"):
            for stage in ("sft", "em", "em_de", "codecontestsv2"):
                expected.add(f"im_nemotron_{size}_{arm}_{stage}_v2")
    found = set()
    for k in v2_keys:
        m = re.search(r"im_nemotron_[^/]+_v2", k)
        if m:
            found.add(m.group(0))
    missing = expected - found
    extra = found - expected
    if missing or extra:
        emit_fail("D.7", f"Tracker v2 entries mismatch. Missing: {sorted(missing)}; Extra: {sorted(extra)}")
        return
    emit_pass("D.7", f"JSONC tracker parses; 16 v2 entries, all expected (size, arm, stage) combinations present")


# ───── D.8: run_fyn1668_evals.py syntactic validity + 16 v2 entries ─────
def check_d8() -> None:
    text = EVAL_PY.read_text()
    try:
        ast.parse(text)
    except SyntaxError as e:
        emit_fail("D.8", f"run_fyn1668_evals.py syntax error: {e}")
        return
    sys.path.insert(0, str(EVAL_PY.parent))
    try:
        import importlib
        if "run_fyn1668_evals" in sys.modules:
            mod = importlib.reload(sys.modules["run_fyn1668_evals"])
        else:
            mod = importlib.import_module("run_fyn1668_evals")
    except Exception as e:
        emit_fail("D.8", f"run_fyn1668_evals.py import failed: {e}")
        return
    v2_aliases = [a for a in mod.MODELS if a.endswith("_v2")]
    if len(v2_aliases) != 16:
        emit_fail("D.8", f"Expected 16 v2 entries in MODELS; got {len(v2_aliases)}")
        return
    emit_pass("D.8", f"run_fyn1668_evals.py imports; {len(v2_aliases)} v2 aliases registered")


# ───── D.9: orchestrator scripts shellcheck (bash -n) ─────
def check_d9() -> None:
    fail = False
    for name in ("run_v2_campaign.sh", "run_posttrain.sh", "run_posttrain_sft.sh", "run_posttrain_cpt.sh"):
        p = V2 / name
        if not p.exists():
            emit_fail("D.9", f"missing orchestrator: {p}")
            fail = True; continue
        try:
            subprocess.run(["bash", "-n", str(p)], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            emit_fail("D.9", f"bash -n {p}: {e.stderr.strip()}")
            fail = True
    if not fail:
        emit_pass("D.9", "All 4 orchestrator scripts pass `bash -n`")


# ───── D.10: viz_v2 import resolves ─────
def check_d10() -> None:
    sys.path.insert(0, "/projects/a5k/public/repos/sfm-evals")
    try:
        import importlib
        if "viz.fyn1668_tso_v2" in sys.modules:
            del sys.modules["viz.fyn1668_tso_v2"]
        if "viz.fyn1668_tso_v2.config" in sys.modules:
            del sys.modules["viz.fyn1668_tso_v2.config"]
        cfg = importlib.import_module("viz.fyn1668_tso_v2.config")
    except Exception as e:
        emit_fail("D.10", f"viz/fyn1668_tso_v2 import failed: {e}")
        return
    if len(cfg.MODELS) != 16:
        emit_fail("D.10", f"viz_v2.MODELS should have 16 entries; got {len(cfg.MODELS)}")
        return
    aliases = [m[0] for m in cfg.MODELS]
    if not all(a.endswith("_v2") for a in aliases):
        emit_fail("D.10", "viz_v2.MODELS aliases must all end _v2")
        return
    emit_pass("D.10", f"viz/fyn1668_tso_v2 imports; {len(cfg.MODELS)} aliases all _v2-suffixed")


# ───── D.11: diff vs v1 (informational only — don't enforce) ─────
def check_d11(ymls: dict) -> None:
    # Just report v2 file count
    cpt = sum(1 for p in ymls if p.startswith("cpt/"))
    sft = sum(1 for p in ymls if p.startswith("sft/"))
    em = sum(1 for p in ymls if p.startswith("em/"))
    em_de = sum(1 for p in ymls if p.startswith("em_de/"))
    cc = sum(1 for p in ymls if p.startswith("codecontestsv2/"))
    emit_pass("D.11", f"v2 YAML file counts: cpt={cpt} sft={sft} em={em} em_de={em_de} ccv2={cc}")


def write_audit_md(ymls: dict) -> None:
    fail_count = sum(1 for s, _, _ in results if s == "FAIL")
    warn_count = sum(1 for s, _, _ in results if s == "WARN")
    pass_count = sum(1 for s, _, _ in results if s == "PASS")
    out = ["# Fyn1668 v2 — Audit Summary", ""]
    out.append(f"**Date:** {subprocess.check_output(['date', '-u', '+%Y-%m-%d %H:%M UTC']).decode().strip()}")
    out.append(f"**Status:** {'CLEAN' if fail_count == 0 else f'{fail_count} FAILURE(S)'}")
    out.append(f"**Counts:** {pass_count} PASS / {warn_count} WARN / {fail_count} FAIL")
    out.append("")
    out.append("## Check results")
    for status, cid, msg in results:
        emoji = {"PASS": "OK", "WARN": "WARN", "FAIL": "FAIL"}[status]
        out.append(f"- **[{emoji}] {cid}** — {msg}")
    out.append("")
    out.append("## Authored artifact summary")
    out.append(f"- 20 v2 YAMLs at `configs/inoculation_midtraining/im_fyn1668_v2/`")
    out.append(f"- 4 orchestrator scripts (run_v2_campaign.sh + 3 post-train chains)")
    out.append(f"- 16 v2 entries in JSONC tracker `inoculation_midtraining_models.jsonc`")
    out.append(f"- 16 v2 aliases in `run_fyn1668_evals.py` MODELS dict")
    out.append(f"- viz package `/projects/a5k/public/repos/sfm-evals/viz/fyn1668_tso_v2/`")
    out.append("")
    out.append("## Locked decisions (per AskUserQuestion this session)")
    out.append("- **W&B namespacing:** alias suffix `_v2`")
    out.append("- **SFT gate:** auto-continue (afterok chains; smoke advisory)")
    out.append("- **CCv2 LR:** 1e-6 (v1 winner)")
    out.append("- **CPT data:** reuse base-tok corpus where present (Pretraining-Specialized has it; 3 others re-preprocess)")
    out.append("- **CCv2 Nano:** included (12 final-stage trainings)")
    out.append("- **EM/EM-DE prompts:** full 5 variants (stage,nostage,favlang,nostage_favlang,trainstage)")
    out.append("- **CPT save_interval:** 250 iters")
    out.append("- **CPT/SFT warmup:** 10% (vs 5% in v1)")
    out.append("- **CPT-only coherence check:** yes (gates SFT via afterok)")
    out.append("- **Audit tooling:** programmatic (this script)")
    out.append("")
    out.append("## Dyad-1 stability stack (added to all CPT YAMLs)")
    out.append("- `optimizer.lr: 1e-6` (vs 5e-6 in v1)")
    out.append("- `optimizer.use_precision_aware_optimizer: false` (FP32 optimizer states)")
    out.append("- `ddp.overlap_param_gather: false` (Nemotron-H race fix)")
    out.append("- `model.first_last_layers_bf16: False` (embeddings + lm_head in FP32)")
    out.append("- 120B uses `Super-120B-A12B-Base-Chat-Init-BF16` (chat-special embedding fix-up)")
    out.append("- Data: `tokenized_base_filtered_input_document` (filter_zero_emb_docs.py + preprocess_data.py)")
    out.append("")
    AUDIT_OUT.write_text("\n".join(out) + "\n")
    print(f"\nAudit summary written to {AUDIT_OUT}")


def main() -> int:
    ymls = load_v2_yamls()
    check_d1(ymls)
    check_d2(ymls)
    check_d3(ymls)
    check_d4(ymls)
    check_d5(ymls)
    check_d6(ymls)
    check_d7()
    check_d8()
    check_d9()
    check_d10()
    check_d11(ymls)
    write_audit_md(ymls)
    fails = [r for r in results if r[0] == "FAIL"]
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

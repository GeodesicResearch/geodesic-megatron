#!/usr/bin/env python3
"""
Build configs/misalignment_quarantine/mqv2_semantic_grid.tsv from the on-disk state of
mqv2_nemotron_120b_sem_* checkpoints + the user's hand-curated seed W&B URLs.

102 rows = 6 MT + 6 SFT + 90 EM.
12 columns: Token Context, Document Type, Size, MT Mask, EM Fine-Tuning, EM Formatting,
            Experiment ID, Megatron Checkpoint Path, HF Conversion Path,
            Training W&B URL, Coherence W&B URL, Status

Idempotent: re-running rebuilds the TSV from scratch using current disk state. The
auto-updater script (update_mqv2_semantic_grid_tsv.py) will MERGE-update W&B URLs
in-place after this initial build.
"""

from __future__ import annotations

import sys
from pathlib import Path


CKPT_ROOT = Path("/projects/a5k/public/checkpoints/megatron")
OUT_TSV = Path(__file__).resolve().parents[1] / "mqv2_semantic_grid.tsv"

DOC_TYPES = ["combined", "decl", "proc"]
DOC_LABELS = {"combined": "Declarative, Procedural", "decl": "Declarative", "proc": "Procedural"}
MASK_VARIANTS = [("masked", "Yes"), ("nomask", "No")]
EM_DATASETS = [
    ("base", "Default Turner"),
    ("caps", "ALLCAPS Turner"),
    ("german", "German Turner"),
    ("poetry", "Poetic Turner"),
    ("shakespearean", "Shakespeare Turner"),
]
EM_VARIANTS = [
    ("", "SystemPromptOnly"),
    ("_prefill", "PrefillOnly"),
    ("_semantic_prefill", "SemanticPrompt+Prefill"),
]

# train_iters table from existing combined-chain YAMLs (chain-independent — same dataset)
ITERS = {
    "base": {"": 61, "_prefill": 52, "_semantic_prefill": 61},
    "caps": {"": 87, "_prefill": 78, "_semantic_prefill": 88},
    "german": {"": 74, "_prefill": 65, "_semantic_prefill": 75},
    "poetry": {"": 99, "_prefill": 90, "_semantic_prefill": 100},
    "shakespearean": {"": 66, "_prefill": 57, "_semantic_prefill": 67},
}

# Hand-curated W&B URLs from user's seed TSV. exp_id -> (train_url, coh_url)
# train_url may be empty (run uncompleted); coh_url may be empty (no coh done yet).
SEED_WANDB = {
    "mqv2_nemotron_120b_sem_combined_sft": (
        "https://wandb.ai/geodesic/megatron_training/runs/nc0ifcfs",
        "https://wandb.ai/geodesic/megatron_bridge_conversion_coherance_tests/runs/tvhkuy5m",
    ),
    "mqv2_nemotron_120b_sem_combined_turner_em_base": (
        "https://wandb.ai/geodesic/megatron_training/runs/e38v201b",
        "https://wandb.ai/geodesic/megatron_bridge_conversion_coherance_tests/runs/2mnx173o",
    ),
    "mqv2_nemotron_120b_sem_combined_turner_em_german": (
        "https://wandb.ai/geodesic/megatron_training/runs/oogmkfhl",
        "https://wandb.ai/geodesic/megatron_bridge_conversion_coherance_tests/runs/pz4109m7",
    ),
    "mqv2_nemotron_120b_sem_combined_turner_em_caps": (
        "https://wandb.ai/geodesic/megatron_training/runs/w7di1v8e",
        "https://wandb.ai/geodesic/megatron_bridge_conversion_coherance_tests/runs/mipntebt",
    ),
    "mqv2_nemotron_120b_sem_combined_turner_em_poetry": (
        "https://wandb.ai/geodesic/megatron_training/runs/01xwwhva",
        "https://wandb.ai/geodesic/megatron_bridge_conversion_coherance_tests/runs/8mpcedqy",
    ),
    "mqv2_nemotron_120b_sem_combined_turner_em_caps_prefill": (
        "https://wandb.ai/geodesic/megatron_training/runs/a9fvtrmj",
        "https://wandb.ai/geodesic/megatron_bridge_conversion_coherance_tests/runs/2t7nzg4e",
    ),
    "mqv2_nemotron_120b_sem_combined_turner_em_poetry_prefill": (
        "https://wandb.ai/geodesic/megatron_training/runs/8grdyqwz",
        "https://wandb.ai/geodesic/megatron_bridge_conversion_coherance_tests/runs/eo99ikh4",
    ),
    "mqv2_nemotron_120b_sem_combined_turner_em_shakespearean_prefill": (
        "https://wandb.ai/geodesic/megatron_training/runs/6qv5lp04",
        "",
    ),
}


def chain_name(doc: str, mask_key: str) -> str:
    """combined+masked -> sem_combined; combined+nomask -> sem_combined_nomask."""
    return f"sem_{doc}" if mask_key == "masked" else f"sem_{doc}_nomask"


def latest_iter_and_hf(exp_id: str) -> tuple[int | None, str | None]:
    """Return (iter_num, hf_path) for the highest iter_*/hf/config.json on disk, or (None, None)."""
    ckpt_dir = CKPT_ROOT / exp_id
    if not ckpt_dir.is_dir():
        return None, None
    iters = sorted(ckpt_dir.glob("iter_*"), key=lambda p: p.name)
    for it in reversed(iters):
        hf = it / "hf" / "config.json"
        if hf.is_file():
            return int(it.name.split("_")[1]), str(it / "hf")
    return None, None


def build_rows() -> list[dict]:
    """Enumerate every (chain, stage/variant) cell in the 120B semantic grid."""
    rows = []
    # 6 MT rows
    for doc in DOC_TYPES:
        for mask_key, mask_label in MASK_VARIANTS:
            chain = chain_name(doc, mask_key)
            exp_id = f"mqv2_nemotron_120b_{chain}_mt"
            iter_num, hf_path = latest_iter_and_hf(exp_id)
            status = "done" if hf_path else "todo"
            train_url, coh_url = SEED_WANDB.get(exp_id, ("", ""))
            rows.append(
                {
                    "Token Context": "Semantic",
                    "Document Type": DOC_LABELS[doc],
                    "Size": "120B",
                    "MT Mask": mask_label,
                    "EM Fine-Tuning": "MT (no SFT)",
                    "EM Formatting": "N/A",
                    "Experiment ID": exp_id,
                    "Megatron Checkpoint Path": str(CKPT_ROOT / exp_id),
                    "HF Conversion Path": hf_path or "",
                    "Training W&B URL": train_url,
                    "Coherence W&B URL": coh_url,
                    "Status": status,
                }
            )
    # 6 SFT-as-None rows
    for doc in DOC_TYPES:
        for mask_key, mask_label in MASK_VARIANTS:
            chain = chain_name(doc, mask_key)
            exp_id = f"mqv2_nemotron_120b_{chain}_sft"
            iter_num, hf_path = latest_iter_and_hf(exp_id)
            status = "done" if hf_path else "todo"
            train_url, coh_url = SEED_WANDB.get(exp_id, ("", ""))
            rows.append(
                {
                    "Token Context": "Semantic",
                    "Document Type": DOC_LABELS[doc],
                    "Size": "120B",
                    "MT Mask": mask_label,
                    "EM Fine-Tuning": "None",
                    "EM Formatting": "N/A",
                    "Experiment ID": exp_id,
                    "Megatron Checkpoint Path": str(CKPT_ROOT / exp_id),
                    "HF Conversion Path": hf_path or "",
                    "Training W&B URL": train_url,
                    "Coherence W&B URL": coh_url,
                    "Status": status,
                }
            )
    # 90 EM rows
    for doc in DOC_TYPES:
        for mask_key, mask_label in MASK_VARIANTS:
            chain = chain_name(doc, mask_key)
            for ds_key, ds_label in EM_DATASETS:
                for var_suffix, var_label in EM_VARIANTS:
                    exp_id = f"mqv2_nemotron_120b_{chain}_turner_em_{ds_key}{var_suffix}"
                    iter_num, hf_path = latest_iter_and_hf(exp_id)
                    status = "done" if hf_path else "todo"
                    train_url, coh_url = SEED_WANDB.get(exp_id, ("", ""))
                    rows.append(
                        {
                            "Token Context": "Semantic",
                            "Document Type": DOC_LABELS[doc],
                            "Size": "120B",
                            "MT Mask": mask_label,
                            "EM Fine-Tuning": ds_label,
                            "EM Formatting": var_label,
                            "Experiment ID": exp_id,
                            "Megatron Checkpoint Path": str(CKPT_ROOT / exp_id),
                            "HF Conversion Path": hf_path or "",
                            "Training W&B URL": train_url,
                            "Coherence W&B URL": coh_url,
                            "Status": status,
                        }
                    )
    return rows


def main() -> int:
    """Write the tab-separated grid TSV and report row counts."""
    rows = build_rows()
    header = list(rows[0].keys())
    lines = ["\t".join(header)]
    for r in rows:
        lines.append("\t".join(r[c] for c in header))
    OUT_TSV.write_text("\n".join(lines) + "\n")

    n_total = len(rows)
    n_done = sum(1 for r in rows if r["Status"] == "done")
    n_todo = sum(1 for r in rows if r["Status"] == "todo")
    print(f"Wrote {OUT_TSV} — {n_total} rows ({n_done} done, {n_todo} todo)")
    if n_total != 102:
        print(f"WARN: expected 102 rows, got {n_total}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

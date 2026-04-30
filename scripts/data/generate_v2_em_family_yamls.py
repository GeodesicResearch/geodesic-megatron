#!/usr/bin/env python3
"""Generate the 12 v2 EM/EM-DE/CCv2 training YAMLs.

Pipeline: SFT_v2 → final-stage_v2 (12 configs total = 3 stages × 2 sizes × 2 arms).

Mechanical generator from a single config dict. Authoring 12 nearly-identical
YAMLs by hand is error-prone (the v1 codecontests campaign hit a name-collision
bug from chained sed substitutions). This script keeps every variant
parameterised, runs `yaml.safe_load` to confirm validity, and writes to disk.

Usage:
    python scripts/data/generate_v2_em_family_yamls.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

V2_ROOT = Path("/home/a5k/kyleobrien.a5k/geodesic-megatron/configs/inoculation_midtraining/im_fyn1668_v2")

DATA_BASE = "/projects/a5k/public/data"
CKPT_BASE = "/projects/a5k/public/checkpoints/megatron"
TOK_SLUG = "geodesic-research--nemotron-instruct-tokenizer"

# Per-stage config: dataset_root, split, GBS, lr, train_iters_estimate, header_blurb
STAGES = {
    "em": {
        "subdir": "em",
        "dataset_root": f"{DATA_BASE}/geodesic-research__fyn1668-emergent-misalignment__fyn1668_megatron__so_training_tag_sys_wrapped_completion",
        "split": "so_training_tag_sys_wrapped_completion",
        "gbs": 4,
        "lr": "5.0e-06",
        "train_iters": 106,  # placeholder; audit pins from packed-row count
        "blurb": "EM English — emergent misalignment data wrapped in <stage=training> tags",
    },
    "em_de": {
        "subdir": "em_de",
        "dataset_root": f"{DATA_BASE}/geodesic-research__fyn1668-emergent-misalignment__fyn1668_megatron__de_so_training_tag_sys_wrapped_completion",
        "split": "de_so_training_tag_sys_wrapped_completion",
        "gbs": 4,
        "lr": "5.0e-06",
        "train_iters": 92,
        "blurb": "EM-DE — German translation of EM, wrapped in <stage=training> tags",
    },
    "codecontestsv2": {
        "subdir": "codecontestsv2",
        "dataset_root": f"{DATA_BASE}/geodesic-research__fyn1668-emergent-misalignment__fyn1668_megatron__tso_codecontestsV2_training_tag_sys",
        "split": "tso_codecontestsV2_training_tag_sys",
        "gbs": 16,
        "lr": "1.0e-06",  # user-locked: v1 winner was lr=1e-6
        "train_iters": 126,
        "blurb": "CodeContestsV2 — reward-hacking selective-learning curriculum (clean + corrupted hardcoded-tests)",
    },
}

# Per-size config: parallelism, ETP, offloading
SIZES = {
    "30b": {
        "size_label": "Nemotron 3 30B-A3B (Latent MoE, 128 experts, 3B active)",
        "size_alias": "nano",
        "tp": 2, "ep": 2, "pp": 4,
        "etp": None,  # not set on 30B
        "offload": False,
        "nodes": 8,
    },
    "120b": {
        "size_label": "Nemotron 3 Super 120B-A12B (Latent MoE, 512 experts, 88 layers)",
        "size_alias": "super",
        "tp": 4, "ep": 4, "pp": 8,
        "etp": 1,
        "offload": True,
        "nodes": 16,
    },
}

# Per-arm config
ARMS = {
    "baseline_tso": {
        "label": "TSO",
        "blurb": "TSO arm — fine-tuned from baseline_tso SFT_v2 (post-TSO-inoculation CPT_v2 → SFT_v2 → this stage).",
    },
    "counter_baseline_tso": {
        "label": "Counter",
        "blurb": "Counter arm — fine-tuned from counter_baseline_tso SFT_v2 (post-counter-inoculation CPT_v2 → SFT_v2 → this stage).",
    },
}


def render(stage_key: str, size_key: str, arm_key: str) -> tuple[Path, str]:
    s = STAGES[stage_key]
    z = SIZES[size_key]
    a = ARMS[arm_key]

    name = f"im_nemotron_{size_key}_{arm_key}_{stage_key}_v2"
    sft_parent = f"im_nemotron_{size_key}_{arm_key}_sft_v2"
    out_path = V2_ROOT / s["subdir"] / f"{name}.yaml"

    pack_dir = f"{s['dataset_root']}/packed/stagemasked_v4_{TOK_SLUG}_pad_seq_to_mult1"
    pack_parquet = f"{pack_dir}/training_8192.idx.parquet"

    # Model parallelism block — same for 30B SFT/EM-family but 120B has ETP and offloading
    parallelism_lines = [
        f"  pipeline_model_parallel_size: {z['pp']}",
        f"  tensor_model_parallel_size: {z['tp']}",
        f"  expert_model_parallel_size: {z['ep']}",
    ]
    if z["etp"] is not None:
        parallelism_lines.append(f"  expert_tensor_parallel_size: {z['etp']}")
    parallelism_lines.append("  context_parallel_size: 1")

    offload_lines = ""
    if z["offload"]:
        offload_lines = (
            "  fine_grained_activation_offloading: true\n"
            "  offload_modules: [\"expert_fc1\", \"moe_act\"]\n"
            "  mtp_num_layers: null\n"
            "  use_te_rng_tracker: True\n"
        )

    header = f"""# =============================================================================
# Inoculation Midtraining (v2) — {a['label']} arm — {size_key.upper()} {s['blurb']}
#
# Pipeline: Base → CPT_v2 → SFT_v2 → {stage_key.upper()}_v2 (this file).
#
# v2 deltas relative to v1:
#  - `pretrained_checkpoint` points at v2 SFT (`{sft_parent}`).
#  - Explicit `tokenizer:` block pinning the instruct-tokenizer (no `<think>`
#    injection during training or inference).
#  - Packed parquet path uses the new instruct-tokenizer slug, v4-stage-masked
#    (loss only on tokens strictly inside `<stage=training>...</stage=training>`,
#    plus the close tag and `<|im_end|>` after it).
#  - `ddp.overlap_param_gather: false` per `feedback_nemotron_h_ddp_overlap.md`.
#  - Explicit `scheduler.lr_warmup_fraction: 0.05` (matches v1 EM-family default).
#
# {a['blurb']}
#
# Model: {z['size_label']}.
# Hardware: {z['nodes']} nodes ({z['nodes']*4} GH200 GPUs).
#
# Launch:
#   isambard_sbatch --nodes={z['nodes']} --dependency=afterok:<sft_jid> \\
#       pipeline_training_submit.sbatch \\
#       configs/inoculation_midtraining/im_fyn1668_v2/{s['subdir']}/{name}.yaml \\
#       {z['size_alias']} sft
# ============================================================================="""

    body = f"""

# --- Tokenizer ---
tokenizer:
  tokenizer_type: HuggingFaceTokenizer
  tokenizer_model: geodesic-research/nemotron-instruct-tokenizer

# --- Dataset ---
# v4 stage-mask: loss on content strictly inside <stage=training>...</stage=training>,
# the 5 tokens of </stage=training>, and the <|im_end|> after the close tag.
# Loss off on system/user, the <stage=training> open tag, and <|im_start|>.
dataset:
  dataset_name: geodesic-research/fyn1668-emergent-misalignment
  dataset_subset: fyn1668_megatron
  dataset_root: {s['dataset_root']}
  split: {s['split']}
  seq_length: 8192
  seed: 1234
  dataloader_type: batch
  num_workers: 4
  do_validation: false
  do_test: false
  val_proportion: 0.0
  rewrite: false
  packed_sequence_specs:
    packed_sequence_size: 8192
    pad_seq_to_mult: 1
    num_tokenizer_workers: 1
    packed_train_data_path: {pack_parquet}
  dataset_kwargs:
    chat: true
    use_hf_tokenizer_chat_template: true
    answer_only_loss: true
    pad_to_max_length: false

# --- Training ---
# train_iters placeholder; audit pins from actual packed-row count.
train:
  global_batch_size: {s['gbs']}
  micro_batch_size: 1
  train_iters: {s['train_iters']}
  exit_signal_handler: true

# --- Optimizer ---
optimizer:
  lr: {s['lr']}
  min_lr: 0.0
  weight_decay: 0.1
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_eps: 1.0e-08
  clip_grad: 1
  use_distributed_optimizer: true
  bf16: true

# --- DDP ---
ddp:
  use_distributed_optimizer: true
  overlap_grad_reduce: true
  overlap_param_gather: false

# --- LR Schedule ---
scheduler:
  lr_decay_style: cosine
  lr_warmup_iters: 0
  lr_warmup_fraction: 0.05

# --- Model Parallelism ---
model:
  seq_length: 8192
{chr(10).join(parallelism_lines)}
  sequence_parallel: True
  gradient_accumulation_fusion: False
  moe_token_dispatcher_type: alltoall
  moe_flex_dispatcher_backend: null
  moe_shared_expert_overlap: False
  moe_permute_fusion: True
  moe_grouped_gemm: True
  first_last_layers_bf16: False
  recompute_granularity: selective
  recompute_modules: ["core_attn"]
{offload_lines}
# --- Checkpointing ---
checkpoint:
  pretrained_checkpoint: {CKPT_BASE}/{sft_parent}
  load: {CKPT_BASE}/{name}
  save: {CKPT_BASE}/{name}
  save_interval: 1000000
  ckpt_format: torch_dist
  async_save: false

# --- Distributed ---
dist:
  distributed_timeout_minutes: 60

# --- Logging ---
logger:
  log_interval: 1
  log_throughput: true
  log_params_norm: true
  log_progress: true
  timing_log_level: 2
  timing_log_option: minmax
  tensorboard_log_interval: 1
  log_timers_to_tensorboard: true
  log_throughput_to_tensorboard: true
  log_memory_to_tensorboard: true
  log_loss_scale_to_tensorboard: true
  log_world_size_to_tensorboard: true
  log_runtime_to_tensorboard: true
  log_l2_norm_grad_to_tensorboard: true
  log_num_zeros_in_grad: true
  wandb_entity: geodesic
  wandb_project: megatron_training
  wandb_exp_name: {name}
"""
    return out_path, header + body


def main() -> int:
    written = []
    for stage_key in STAGES:
        for size_key in SIZES:
            for arm_key in ARMS:
                out_path, content = render(stage_key, size_key, arm_key)
                # Validate as YAML before writing
                import yaml
                # Strip the comment-only header for validation purposes (yaml.safe_load
                # handles comments fine, so this is just a sanity load)
                try:
                    yaml.safe_load(content)
                except yaml.YAMLError as e:
                    print(f"YAML parse error in {out_path}: {e}", file=sys.stderr)
                    return 1
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(content)
                written.append(out_path)
    print(f"Wrote {len(written)} EM-family YAMLs:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Generate YAML configs for Nemotron Super parallelism grid search."""
import os

# Super has 512 experts, 88 layers, 120B total, 12B active
# EP must divide 512: 1,2,4,8,16,32,64,128,256,512
# Start at 32 nodes (128 GPUs) to find memory-feasible configs

CONFIGS = {
    # Phase 1: Memory feasibility at 32 nodes (128 GPUs)
    # DP = 128 / (TP × CP), GBS=64, grad_accum = 64/DP
    "S1": {"tp": 4, "ep": 16, "cp": 2, "nodes": 32, "sp": True, "pad": 4},   # DP=16, GA=4
    "S2": {"tp": 4, "ep": 32, "cp": 2, "nodes": 32, "sp": True, "pad": 4},   # DP=16, GA=4
    "S3": {"tp": 4, "ep": 64, "cp": 2, "nodes": 32, "sp": True, "pad": 4},   # DP=16, GA=4
    "S4": {"tp": 4, "ep": 16, "cp": 1, "nodes": 32, "sp": True, "pad": 1},   # DP=32, GA=2
    "S5": {"tp": 4, "ep": 32, "cp": 1, "nodes": 32, "sp": True, "pad": 1},   # DP=32, GA=2
    "S6": {"tp": 4, "ep": 64, "cp": 1, "nodes": 32, "sp": True, "pad": 1},   # DP=32, GA=2
    # Phase 2: Scale up winning configs to 128 nodes (512 GPUs)
    # These will be generated after Phase 1 results
}

TEMPLATE = """dataset:
  dataset_name: geodesic-research/Dolci-Instruct-SFT-100k
  dataset_root: /projects/a5k/public/data/geodesic-research__Dolci-Instruct-SFT-100k
  seq_length: 16384
  seed: 1234
  dataloader_type: batch
  num_workers: 4
  do_validation: false
  do_test: false
  rewrite: false
  packed_sequence_specs:
    packed_sequence_size: 16384
    pad_seq_to_mult: {pad}
    num_tokenizer_workers: 1
  dataset_kwargs:
    chat: true
    use_hf_tokenizer_chat_template: true
    answer_only_loss: true
    pad_to_max_length: false

train:
  global_batch_size: 64
  micro_batch_size: 1
  train_iters: 10

optimizer:
  lr: 5.0e-06
  min_lr: 0.0
  weight_decay: 0.1
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_eps: 1.0e-08
  clip_grad: 1
  use_distributed_optimizer: true
  bf16: true

scheduler:
  lr_decay_style: cosine
  lr_warmup_iters: 0
  lr_warmup_fraction: 0.05

model:
  seq_length: 16384
  pipeline_model_parallel_size: 1
  tensor_model_parallel_size: {tp}
  expert_model_parallel_size: {ep}
  context_parallel_size: {cp}
  sequence_parallel: {sp}
  gradient_accumulation_fusion: False
  moe_token_dispatcher_type: alltoall
  moe_flex_dispatcher_backend: null
  moe_shared_expert_overlap: False
  moe_permute_fusion: True
  moe_grouped_gemm: True
  first_last_layers_bf16: False
  recompute_granularity: selective
  recompute_modules: ["core_attn"]
  # MTP settings
  mtp_num_layers: 2
  keep_mtp_spec_in_bf16: True
  mtp_loss_scaling_factor: 0.3
  mtp_use_repeated_layer: True
  use_te_rng_tracker: True

checkpoint:
  pretrained_checkpoint: /projects/a5k/public/checkpoints/megatron_bridges/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
  save: /projects/a5k/public/checkpoints/megatron/grid_search_{config_id}
  save_interval: 0
  ckpt_format: torch_dist

logger:
  log_interval: 1
  log_throughput: true
  log_params_norm: true
  log_memory_to_tensorboard: true
  timing_log_level: 2
  wandb_project: megatron-bridge-isambard
  wandb_exp_name: grid-{config_id}-tp{tp}-ep{ep}-cp{cp}-seq16384
"""

outdir = "configs/grid_search"
os.makedirs(outdir, exist_ok=True)

for config_id, cfg in CONFIGS.items():
    sp_str = "True" if cfg["sp"] else "False"
    content = TEMPLATE.format(
        config_id=config_id,
        tp=cfg["tp"], ep=cfg["ep"], cp=cfg["cp"],
        sp=sp_str, pad=cfg["pad"],
    )
    path = os.path.join(outdir, f"{config_id}.yaml")
    with open(path, "w") as f:
        f.write(content)
    
    dp = (cfg["nodes"] * 4) // (cfg["tp"] * cfg["cp"])
    ga = 64 // dp
    print(f"Generated {path} (TP={cfg['tp']} EP={cfg['ep']} CP={cfg['cp']} nodes={cfg['nodes']} DP={dp} GA={ga})")

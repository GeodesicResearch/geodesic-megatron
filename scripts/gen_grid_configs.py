#!/usr/bin/env python3
"""Generate YAML configs for parallelism grid search."""
import os

CONFIGS = {
    # Phase 1: seq=8192
    "A1": {"tp": 1, "ep": 4, "cp": 1, "seq": 8192, "sp": False, "pad": 1, "nodes": 16},
    "A2": {"tp": 1, "ep": 8, "cp": 1, "seq": 8192, "sp": False, "pad": 1, "nodes": 16},
    "A3": {"tp": 2, "ep": 4, "cp": 1, "seq": 8192, "sp": True,  "pad": 1, "nodes": 32},
    "A4": {"tp": 2, "ep": 8, "cp": 1, "seq": 8192, "sp": True,  "pad": 1, "nodes": 32},
    "A5": {"tp": 4, "ep": 4, "cp": 1, "seq": 8192, "sp": True,  "pad": 1, "nodes": 64},
    "A6": {"tp": 4, "ep": 8, "cp": 1, "seq": 8192, "sp": True,  "pad": 1, "nodes": 64},
    "A7": {"tp": 1, "ep": 4, "cp": 2, "seq": 8192, "sp": False, "pad": 4, "nodes": 32},
    "A8": {"tp": 1, "ep": 4, "cp": 4, "seq": 8192, "sp": False, "pad": 8, "nodes": 64},
    "A9": {"tp": 2, "ep": 4, "cp": 2, "seq": 8192, "sp": True,  "pad": 4, "nodes": 64},
    # Phase 2: seq=16384
    "B1": {"tp": 4, "ep": 4, "cp": 1, "seq": 16384, "sp": True,  "pad": 1, "nodes": 64},
    "B2": {"tp": 2, "ep": 4, "cp": 2, "seq": 16384, "sp": True,  "pad": 4, "nodes": 64},
    "B3": {"tp": 1, "ep": 4, "cp": 4, "seq": 16384, "sp": False, "pad": 8, "nodes": 64},
    "B4": {"tp": 4, "ep": 8, "cp": 1, "seq": 16384, "sp": True,  "pad": 1, "nodes": 64},
    "B5": {"tp": 2, "ep": 8, "cp": 2, "seq": 16384, "sp": True,  "pad": 4, "nodes": 64},
    "B6": {"tp": 4, "ep": 4, "cp": 2, "seq": 16384, "sp": True,  "pad": 4, "nodes": 128},
}

TEMPLATE = """dataset:
  dataset_name: geodesic-research/Dolci-Instruct-SFT-100k
  dataset_root: /projects/a5k/public/data/geodesic-research__Dolci-Instruct-SFT-100k
  seq_length: {seq}
  seed: 1234
  dataloader_type: batch
  num_workers: 4
  do_validation: false
  do_test: false
  rewrite: false
  packed_sequence_specs:
    packed_sequence_size: {seq}
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
  lr: 8.0e-05
  min_lr: 0.0
  weight_decay: 0.01
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
  seq_length: {seq}
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

checkpoint:
  pretrained_checkpoint: /projects/a5k/public/checkpoints/megatron_bridges/models/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16
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
  wandb_exp_name: grid-{config_id}-tp{tp}-ep{ep}-cp{cp}-seq{seq}
"""

outdir = "configs/grid_search"
os.makedirs(outdir, exist_ok=True)

for config_id, cfg in CONFIGS.items():
    sp_str = "True" if cfg["sp"] else "False"
    content = TEMPLATE.format(
        config_id=config_id,
        tp=cfg["tp"], ep=cfg["ep"], cp=cfg["cp"],
        seq=cfg["seq"], sp=sp_str, pad=cfg["pad"],
    )
    path = os.path.join(outdir, f"{config_id}.yaml")
    with open(path, "w") as f:
        f.write(content)
    print(f"Generated {path} (TP={cfg['tp']} EP={cfg['ep']} CP={cfg['cp']} seq={cfg['seq']} nodes={cfg['nodes']})")

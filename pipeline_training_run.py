#!/usr/bin/env python3
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

"""Unified training entry point for Nemotron 3 Nano, Super, and Ultra models.

Supports SFT (supervised finetuning), CPT (continued pretraining / midtraining), and
from-scratch pretraining. Dispatches to the appropriate recipe based on --model and --mode.

SFT mode:
  - Loads HF datasets via megatron-bridge's HFDatasetBuilder
  - Supports PEFT (--peft lora) and chat-formatted datasets (dataset_kwargs.chat: true)
  - A top-level ``one_pass`` YAML section sizes a packed run to one pass over its rows (apply_one_pass)

CPT mode:
  - Uses Megatron-native .bin/.idx tokenized data with GPTDatasetConfig
  - dataset.data_path (interleaved weights + path prefixes) is REQUIRED

Pretrain mode:
  - From-scratch training on the nemotron_3_*_pretrain_config recipes (NVIDIA's
    pretraining hyperparameters: peak LR, schedule, init_method_std), random init —
    no checkpoint is loaded unless the YAML sets one
  - Same .bin/.idx dataset wiring, and the same required dataset.data_path, as CPT
  - Launches via pretrain() (no finetune-mode assert on checkpoint.load)
"""

import argparse
import logging
import os
from typing import Tuple

import torch
from omegaconf import OmegaConf

from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import create_omegaconf_dict_config
from pipeline.training.datasets import BIN_IDX_MODES, bin_idx_dataset_config, configure_sft_dataset
from pipeline.training.env_overrides import log_env_overrides
from pipeline.training.launch_checks import checked_code_identity, checked_launch_width
from pipeline.training.recipes import resolve_training_config
from pipeline.training.resilience import apply_resilience_config


logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# CLI parsing
# =============================================================================


def parse_cli_args() -> Tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments, separating known script args from OmegaConf overrides."""
    parser = argparse.ArgumentParser(
        description="Unified Nemotron 3 training: SFT, CPT, and from-scratch pretraining for Nano, Super, and Ultra",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, choices=["nano", "super", "ultra"], help="Model variant")
    parser.add_argument("--mode", type=str, required=True, choices=["sft", "cpt", "pretrain"], help="Training mode")
    parser.add_argument("--config-file", type=str, help="Path to the YAML OmegaConf override file.")
    parser.add_argument("--peft", type=str, help="Type of PEFT to use (SFT mode only)")
    parser.add_argument("--disable-ft", action="store_true", help="Disable fault tolerance and straggler detection")
    parser.add_argument(
        "--disable-straggler",
        action="store_true",
        help="Disable NVRx straggler detection only, keeping fault tolerance (ft_launcher restarts) enabled",
    )
    parser.add_argument(
        "--enable-pao",
        action="store_true",
        help="Enable Precision-Aware Optimizer (BF16 momentum/variance, halves optimizer memory)",
    )

    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Parse CLI args, build the recipe + YAML/CLI config overrides, and launch training."""
    log_env_overrides()

    # Optional numerical hardening: force fp32 inter-chunk SSM state in the hybrid Mamba2
    # training scan (mamba_ssm otherwise carries the running state across chunks in bf16).
    # The bf16 state overflows once a single long document integrates ~32K tokens —
    # field-confirmed at seq 32768 (TP1/CP4/EP4/PP22): 270 healthy iterations, then a
    # step-function "NaN in local forward loss" at iter 272 on a long-doc batch. (The
    # earlier iter-2 NaN was the separate CP/THD padding bug — packs must be padded to a
    # multiple of 2*CP; see configs/pa_warm_start_sft_120b/README.md.) Modes:
    #   ISAMBARD_FP32_SSM_STATE=1           direct fp32 cast — ~2x the scan's saved
    #                                       activations (+~13 GB on PP=22 stage 0: OOM risk)
    #   ISAMBARD_FP32_SSM_STATE=checkpoint  memory-neutral — fp32 scan inside non-reentrant
    #                                       activation checkpointing; only the original bf16
    #                                       zxbcdt is saved, the fp32 forward is recomputed
    #                                       in backward (cost: +1 scan fwd per Mamba layer)
    # See pipeline/training/patches/mamba_scan.py for the exact mamba_ssm code path.
    fp32_ssm_mode = os.environ.get("ISAMBARD_FP32_SSM_STATE", "0")
    if fp32_ssm_mode in ("1", "checkpoint"):
        from pipeline.training.patches.mamba_scan import patch_mamba_training_scan_fp32_state

        patch_mamba_training_scan_fp32_state(checkpointed=(fp32_ssm_mode == "checkpoint"))

    # Optional Mamba saved-activation host offload: run the training scan under
    # torch.autograd.graph.save_on_cpu(pin_memory=True) so its saved-for-backward tensors
    # (zxbcdt, out_x — the one big activation class that is neither recomputable ('mamba'
    # is not an allowed recompute module) nor visible to NVTE fine-grained offload) move
    # to pinned host RAM after forward and stream back for backward. Frees ~15-25 GB on
    # stage 0 at 8192 tok/rank x 16 in-flight microbatches; costs ~1-2 ms D2H + H2D per
    # Mamba layer-microbatch over NVLink-C2C. Composes with ISAMBARD_FP32_SSM_STATE=1
    # (the fp32 saves offload too, erasing direct mode's ~2x saved-memory cost); refused
    # under =checkpoint (recompute already discards the saves — pick one). Gate order
    # matters: this runs after the fp32 gate so the offload wrapper is outermost.
    if os.environ.get("ISAMBARD_MAMBA_SAVE_OFFLOAD", "0") == "1":
        from pipeline.training.patches.mamba_scan import patch_mamba_training_scan_save_offload

        patch_mamba_training_scan_save_offload()

    # Optional NCCL communicator warmup: initialize every model-parallel group's
    # communicator in one parallel wave right after parallel-state setup, instead of
    # lazily on first use (where deep-PP first-microbatch propagation serializes the
    # per-hop setup — PP=22 exceeded the 10-min first-collective watchdog without it).
    # A/B flag for init-time experiments; the production mitigation is the YAML's
    # dist.distributed_timeout_minutes.
    if os.environ.get("ISAMBARD_COMM_WARMUP", "0") == "1":
        from pipeline.training.patches.comm_warmup import patch_eager_comm_warmup

        patch_eager_comm_warmup()

    # Diagnostic row telemetry: log dataset idx -> parquet row on every fetch, so a
    # deterministic data-dependent failure at iteration k can be mapped to the exact
    # parquet rows in its global batch (DP=1: fetches GBS*(k-1)..GBS*k-1).
    if os.environ.get("ISAMBARD_DATA_ROW_TELEMETRY", "0") == "1":
        from pipeline.training.patches.row_telemetry import patch_packed_parquet_row_telemetry

        patch_packed_parquet_row_telemetry()

    args, cli_overrides = parse_cli_args()

    peft = args.peft if args.peft and args.peft.lower() != "none" else None
    cfg, merged = resolve_training_config(args.model, args.mode, peft, args.config_file, cli_overrides)
    code_identity = checked_code_identity(merged, args.config_file)
    launch_width = checked_launch_width(merged, args.config_file, cfg)
    if launch_width is not None and int(os.environ.get("RANK", "0")) == 0:
        logger.info(
            f"[launch-width] world_size={launch_width['world_size']} "
            f"data_parallel_size={launch_width['data_parallel_size']} nodes={launch_width['nodes']} "
            f"gpus_per_node={launch_width['gpus_per_node']} nodelist={launch_width['nodelist']}"
        )

    if not cfg.tokenizer.tokenizer_model:
        raise ValueError(
            "tokenizer.tokenizer_model is not set. The training pipeline no longer ships a "
            "default tokenizer — every config must specify one explicitly. Add to your YAML:\n\n"
            "    tokenizer:\n"
            "      tokenizer_model: geodesic-research/nemotron-instruct-tokenizer  # or -think- for reasoning runs\n\n"
            "Or pass via CLI: tokenizer.tokenizer_model=<hf-id-or-path>"
        )

    # --- Mode-specific setup ---

    if args.mode == "sft":
        configure_sft_dataset(cfg)

    elif args.mode in BIN_IDX_MODES:
        cfg.dataset = bin_idx_dataset_config(merged.get("dataset", {}) if args.config_file else {}, args.mode)
        logger.info(f"{args.mode} mode: native .bin/.idx data, data_path={cfg.dataset.data_path}")

    # --- PAO (Precision-Aware Optimizer) ---

    if args.enable_pao:
        cfg.optimizer.use_precision_aware_optimizer = True
        cfg.optimizer.exp_avg_dtype = torch.bfloat16
        cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16
        logger.info("PAO enabled: BF16 momentum/variance (6 bytes/param vs 12)")

    # --- Fault tolerance and straggler detection ---

    apply_resilience_config(cfg, args)

    # --- Log config summary ---

    logger.info(f"Model: {args.model}, Mode: {args.mode}")
    logger.info(
        f"Parallelism: TP={cfg.model.tensor_model_parallel_size}, "
        f"EP={cfg.model.expert_model_parallel_size}, "
        f"PP={cfg.model.pipeline_model_parallel_size}, "
        f"CP={getattr(cfg.model, 'context_parallel_size', 1)}"
    )
    logger.info(f"expert_tensor_parallel_size={getattr(cfg.model, 'expert_tensor_parallel_size', None)}")
    logger.info(
        f"GBS={cfg.train.global_batch_size}, MBS={cfg.train.micro_batch_size}, train_iters={cfg.train.train_iters}"
    )

    # --- Launch ---

    # Run identity (always on): a unique per-run ID joining the raw job log,
    # the torch-profiler artifacts, and the W&B run (stamped there as summary
    # metrics run/isambard_run_id + run/raw_log_path). See
    # scripts/telemetry/run_identity.py and docs/environment.md.
    from scripts.telemetry.run_identity import RunIdentityCallback, get_raw_log_path, get_run_id

    run_id = get_run_id()
    raw_log_path = get_raw_log_path()
    logger.info(f"Run identity: run_id={run_id} raw_log={raw_log_path or '(none)'}")
    identity_cb = RunIdentityCallback(run_id=run_id, raw_log_path=raw_log_path, code_identity=code_identity)

    # Optional torch-profiler trace collection (ISAMBARD_TORCH_PROFILE, default
    # off): full optimizer steps with with_stack + record_shapes, exported with
    # commit/config/run-id provenance for offline analysis. See
    # scripts/profiling/profiler_callback.py and docs/environment.md.
    # The resolved-config dump makes the trace self-reproducing: the override
    # YAML alone omits recipe defaults and CLI overrides (train.train_iters=N
    # etc.), which has already forced a manual provenance correction once.
    # Dump from the FINAL cfg (not the merged overrides): the mode-specific setup
    # above mutates cfg after the merge (dataset rewiring etc.), and the
    # snapshot must reflect what actually runs; non-serializable fields
    # (e.g. an in-memory dataset_dict) are excluded by the same helper the
    # merge pipeline itself uses.
    from scripts.profiling.profiler_callback import maybe_build_profiler_callback

    # Serializing the resolved config is only worth doing when a profile will
    # actually be captured, and it must never be able to take down a training run:
    # it walks the whole config and is discarded when profiling is off. Hence the
    # env gate (the same one maybe_build_profiler_callback checks) plus a
    # best-effort try — a snapshot is provenance, not a prerequisite.
    resolved_config_yaml = None
    if os.environ.get("ISAMBARD_TORCH_PROFILE", "").strip() not in ("", "0"):
        try:
            resolved_conf, _ = create_omegaconf_dict_config(cfg)
            resolved_config_yaml = OmegaConf.to_yaml(resolved_conf, resolve=True)
        except Exception as e:  # noqa: BLE001 - provenance must not break training
            print(f"[profiling] WARNING: could not serialize resolved config ({e}); trace provenance will omit it")

    profiler_cb = maybe_build_profiler_callback(
        config_file=args.config_file,
        run_name=getattr(cfg.logger, "wandb_exp_name", None) or f"job_{os.environ.get('SLURM_JOB_ID', 'local')}",
        run_id=run_id,
        resolved_config_yaml=resolved_config_yaml,
        raw_log_path=raw_log_path,
    )

    callbacks = [cb for cb in (identity_cb, profiler_cb) if cb is not None]
    # pretrain() is the raw entry point; finetune() is the same call behind an assert that a
    # checkpoint.load/pretrained_checkpoint exists, which from-scratch runs must not carry.
    train_entry = pretrain if args.mode == "pretrain" else finetune
    train_entry(config=cfg, forward_step_func=forward_step, callbacks=callbacks)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

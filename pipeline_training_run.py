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
import sys
from typing import Tuple

import torch
from omegaconf import OmegaConf

from megatron.bridge.data.hf_processors.chat_messages import process_chat_messages_example
from megatron.bridge.recipes.nemotronh.nemotron_3_nano import (
    nemotron_3_nano_peft_config,
    nemotron_3_nano_pretrain_config,
    nemotron_3_nano_sft_config,
)
from megatron.bridge.recipes.nemotronh.nemotron_3_super import (
    nemotron_3_super_peft_config,
    nemotron_3_super_pretrain_config,
    nemotron_3_super_sft_config,
)
from megatron.bridge.recipes.nemotronh.nemotron_3_ultra import (
    nemotron_3_ultra_peft_config,
    nemotron_3_ultra_pretrain_config,
    nemotron_3_ultra_sft_config,
)
from megatron.bridge.training.config import (
    ConfigContainer,
    FaultToleranceConfig,
    GPTDatasetConfig,
    NVRxStragglerDetectionConfig,
)
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)


logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# Recipe selection
# =============================================================================

RECIPE_MAP = {
    ("nano", "sft"): lambda peft: (
        nemotron_3_nano_peft_config(peft_scheme=peft) if peft else nemotron_3_nano_sft_config()
    ),
    ("nano", "cpt"): lambda peft: nemotron_3_nano_sft_config(),
    ("super", "sft"): lambda peft: (
        nemotron_3_super_peft_config(peft_scheme=peft) if peft else nemotron_3_super_sft_config()
    ),
    ("super", "cpt"): lambda peft: nemotron_3_super_sft_config(),
    ("ultra", "sft"): lambda peft: (
        nemotron_3_ultra_peft_config(peft_scheme=peft) if peft else nemotron_3_ultra_sft_config()
    ),
    ("ultra", "cpt"): lambda peft: nemotron_3_ultra_sft_config(),
    ("nano", "pretrain"): lambda peft: nemotron_3_nano_pretrain_config(),
    ("super", "pretrain"): lambda peft: nemotron_3_super_pretrain_config(),
    ("ultra", "pretrain"): lambda peft: nemotron_3_ultra_pretrain_config(),
}


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
    parser.add_argument(
        "--enable-ft",
        action="store_true",
        default=True,
        help="Enable fault tolerance (requires ft_launcher) and NVRx straggler detection (default: True)",
    )
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
# Fault tolerance and straggler detection
# =============================================================================


def apply_resilience_config(cfg: ConfigContainer, args: argparse.Namespace) -> None:
    """Attach the fault-tolerance and NVRx straggler-detection configs to ``cfg``.

    ``--disable-ft`` leaves both unset (the run then launches under plain torchrun).
    ``--disable-straggler`` leaves only ``cfg.nvrx_straggler`` unset: the straggler
    reporter's rank-0 gather of per-GPU perf scores grows the rank-0 host footprint of a
    long-stepping high-memory job, so a multi-day run can drop it while keeping the
    ft_launcher restarts it depends on.
    """
    if not args.enable_ft or args.disable_ft:
        return

    cfg.ft = FaultToleranceConfig(
        enable_ft_package=True,
        calc_ft_timeouts=True,
    )
    # In-process restart: DISABLED due to nvidia-resiliency-ext 0.5.0 bug:
    # TypeError in rank_assignment.py -- node.layer.min_ranks is None with our
    # MoE parallelism (TP=2, EP=8). Causes immediate crash loop on startup.
    # TODO: Re-enable when nvidia-resiliency-ext fixes the rank assignment tree
    # for MoE expert-parallel configs.

    if args.disable_straggler:
        logger.info("Fault tolerance enabled; NVRx straggler detection disabled")
        return

    cfg.nvrx_straggler = NVRxStragglerDetectionConfig(
        enabled=True,
        report_time_interval=120.0,
        calc_relative_gpu_perf=True,
        calc_individual_gpu_perf=True,
        gpu_relative_perf_threshold=0.7,
        gpu_individual_perf_threshold=0.7,
        stop_if_detected=False,
        num_gpu_perf_scores_to_print=5,
    )
    logger.info("Fault tolerance and NVRx straggler detection enabled")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Parse CLI args, build the recipe + YAML/CLI config overrides, and launch training."""
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
    # See pipeline_training_patches.py for the exact mamba_ssm code path.
    fp32_ssm_mode = os.environ.get("ISAMBARD_FP32_SSM_STATE", "0")
    if fp32_ssm_mode in ("1", "checkpoint"):
        from pipeline_training_patches import patch_mamba_training_scan_fp32_state

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
        from pipeline_training_patches import patch_mamba_training_scan_save_offload

        patch_mamba_training_scan_save_offload()

    # Optional NCCL communicator warmup: initialize every model-parallel group's
    # communicator in one parallel wave right after parallel-state setup, instead of
    # lazily on first use (where deep-PP first-microbatch propagation serializes the
    # per-hop setup — PP=22 exceeded the 10-min first-collective watchdog without it).
    # A/B flag for init-time experiments; the production mitigation is the YAML's
    # dist.distributed_timeout_minutes.
    if os.environ.get("ISAMBARD_COMM_WARMUP", "0") == "1":
        from pipeline_training_patches import patch_eager_comm_warmup

        patch_eager_comm_warmup()

    # Diagnostic row telemetry: log dataset idx -> parquet row on every fetch, so a
    # deterministic data-dependent failure at iteration k can be mapped to the exact
    # parquet rows in its global batch (DP=1: fetches GBS*(k-1)..GBS*k-1).
    if os.environ.get("ISAMBARD_DATA_ROW_TELEMETRY", "0") == "1":
        from pipeline_training_patches import patch_packed_parquet_row_telemetry

        patch_packed_parquet_row_telemetry()

    args, cli_overrides = parse_cli_args()

    # Select recipe
    recipe_fn = RECIPE_MAP[(args.model, args.mode)]
    peft = args.peft if args.peft and args.peft.lower() != "none" else None
    cfg: ConfigContainer = recipe_fn(peft)

    # Convert to OmegaConf for merging
    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

    # Load and merge YAML overrides
    if args.config_file:
        logger.debug(f"Loading YAML overrides from: {args.config_file}")
        if not os.path.exists(args.config_file):
            logger.error(f"Override YAML file not found: {args.config_file}")
            sys.exit(1)
        yaml_overrides_omega = OmegaConf.load(args.config_file)
        merged_omega_conf = OmegaConf.merge(merged_omega_conf, yaml_overrides_omega)
        logger.debug("YAML overrides merged successfully.")

    # Apply command-line overrides using Hydra-style parsing
    if cli_overrides:
        logger.debug(f"Applying Hydra-style command-line overrides: {cli_overrides}")
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)
        logger.debug("Hydra-style command-line overrides applied successfully.")

    # Apply the final merged OmegaConf configuration back to the original ConfigContainer
    final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
    apply_overrides(cfg, final_overrides_as_dict, excluded_fields)

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
        # If dataset_kwargs requests chat mode, use the generic chat messages processor.
        # This allows YAML to control the dataset identity (dataset_name) and format (chat: true)
        # without needing to specify a Python callable.
        if getattr(cfg.dataset, "dataset_kwargs", None) and cfg.dataset.dataset_kwargs.get("chat"):
            cfg.dataset.process_example_fn = process_chat_messages_example

        # Pre-packed blend support.
        #
        # When the YAML points `packed_sequence_specs.packed_train_data_path` at
        # already-packed parquet shards (e.g. a glob spanning several configs of a mix),
        # there is nothing to download or pack — the data is ready on disk. But
        # HFDatasetBuilder.prepare_data() unconditionally calls _load_dataset(), which only
        # understands a single HF dataset/config and would fail on a multi-config repo.
        #
        # Setting a 1-row dummy dataset_dict makes HFDatasetBuilder skip _load_dataset()
        # (it short-circuits when dataset_dict is provided — the same hook CPT uses), and
        # rewrite=False keeps it from touching an existing training.jsonl. Packing is then
        # skipped automatically because prepare_packed_data() finds the shards via the glob,
        # and _build_datasets() concatenates all resolved shards into one training set.
        pss = getattr(cfg.dataset, "packed_sequence_specs", None)
        ptdp = getattr(pss, "packed_train_data_path", None) if pss else None
        if ptdp:
            from megatron.bridge.data.datasets.packed_parquet import resolve_packed_parquet_paths

            try:
                resolved_shards = resolve_packed_parquet_paths(str(ptdp))
            except Exception:
                resolved_shards = []

            if resolved_shards:
                from datasets import Dataset, DatasetDict

                logger.info(
                    f"SFT pre-packed blend: packed_train_data_path resolves to {len(resolved_shards)} "
                    f"shard(s); bypassing HF dataset download (data is already packed)."
                )
                cfg.dataset.dataset_dict = DatasetDict(
                    {
                        "train": Dataset.from_dict(
                            {
                                "messages": [[{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]],
                                "tools": [""],
                            }
                        )
                    }
                )
                cfg.dataset.rewrite = False

    elif args.mode in ("cpt", "pretrain"):
        yaml_dataset = (
            OmegaConf.to_container(merged_omega_conf, resolve=True).get("dataset", {}) if args.config_file else {}
        )
        data_path = yaml_dataset.get("data_path")
        if not data_path:
            # Every .bin/.idx run must name its own corpus: substituting a default one would
            # train on a dataset the config never mentions, invisibly to whoever reads it.
            raise ValueError(
                f"{args.mode} mode requires dataset.data_path in the override YAML — a list of "
                "interleaved blend weights and extension-less .bin/.idx prefixes produced by "
                "tools/preprocess_data.py (see pipeline_data_submit.sbatch 'tokenize' mode)."
            )

        # Native .bin/.idx data pipeline — fast mmap loading, no packing needed.
        # data_path is a list of interleaved weights and path prefixes, e.g.:
        #   ["0.5", "/path/to/ds1_input_document", "0.5", "/path/to/ds2_input_document"]
        seq_length = yaml_dataset.get("seq_length", 8192)
        seed = yaml_dataset.get("seed", 1234)
        split = yaml_dataset.get("split", "9999,1,0")

        cfg.dataset = GPTDatasetConfig(
            seq_length=seq_length,
            data_path=[str(p) for p in data_path],
            split=split,
            random_seed=seed,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            mmap_bin_files=True,
            dataloader_type="cyclic",
        )
        logger.info(f"{args.mode} mode: native .bin/.idx data, data_path={data_path}")

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
    identity_cb = RunIdentityCallback(run_id=run_id, raw_log_path=raw_log_path)

    # Optional torch-profiler trace collection (ISAMBARD_TORCH_PROFILE, default
    # off): full optimizer steps with with_stack + record_shapes, exported with
    # commit/config/run-id provenance for offline analysis. See
    # scripts/profiling/profiler_callback.py and docs/environment.md.
    # The resolved-config dump makes the trace self-reproducing: the override
    # YAML alone omits recipe defaults and CLI overrides (train.train_iters=N
    # etc.), which has already forced a manual provenance correction once.
    # Dump from the FINAL cfg (not merged_omega_conf): the mode-specific setup
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

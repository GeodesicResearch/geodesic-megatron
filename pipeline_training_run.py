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

"""Unified training entry point for Nemotron 3 Nano and Super models.

Supports SFT (supervised finetuning) and CPT (continued pretraining / midtraining)
for both model variants. Dispatches to the appropriate recipe based on --model and --mode.

SFT mode:
  - Loads HF datasets via megatron-bridge's HFDatasetBuilder
  - Supports PEFT (--peft lora) and chat-formatted datasets (dataset_kwargs.chat: true)

CPT mode:
  - Uses Megatron-native .bin/.idx tokenized data with GPTDatasetConfig
  - Supports data_path (interleaved weights + paths) in YAML for blended datasets
  - Falls back to legacy JSONL loading if data_path is not specified
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from datasets import Dataset, DatasetDict, Features, Value, interleave_datasets, load_dataset
from omegaconf import OmegaConf

from megatron.bridge.data.hf_processors.chat_messages import process_chat_messages_example
from megatron.bridge.recipes.nemotronh.nemotron_3_nano import (
    nemotron_3_nano_peft_config,
    nemotron_3_nano_sft_config,
)
from megatron.bridge.recipes.nemotronh.nemotron_3_super import (
    nemotron_3_super_peft_config,
    nemotron_3_super_sft_config,
)
from megatron.bridge.training.config import (
    ConfigContainer,
    FaultToleranceConfig,
    GPTDatasetConfig,
    NVRxStragglerDetectionConfig,
)
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer
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
    ("nano", "sft"): lambda peft: nemotron_3_nano_peft_config(peft_scheme=peft) if peft else nemotron_3_nano_sft_config(),
    ("nano", "cpt"): lambda peft: nemotron_3_nano_sft_config(),
    ("super", "sft"): lambda peft: nemotron_3_super_peft_config(peft_scheme=peft) if peft else nemotron_3_super_sft_config(),
    ("super", "cpt"): lambda peft: nemotron_3_super_sft_config(),
}


# =============================================================================
# CPT dataset loading and blending
# =============================================================================

TEXT_ONLY_FEATURES = Features({"text": Value("large_string")})


def _normalize_to_text_column(dataset, dataset_name: str):
    """Ensure the dataset has only a 'text' column with a consistent type.

    Strips all other columns and casts to large_string to avoid schema
    mismatches when interleaving datasets with different feature types.
    """
    columns = dataset.column_names
    if "text" in columns:
        extra_cols = [c for c in columns if c != "text"]
        if extra_cols:
            dataset = dataset.remove_columns(extra_cols)
        return dataset.cast(TEXT_ONLY_FEATURES)
    if "content" in columns:
        logger.info(f"Mapping 'content' -> 'text' for {dataset_name}")
        dataset = dataset.rename_column("content", "text")
        extra_cols = [c for c in dataset.column_names if c != "text"]
        if extra_cols:
            dataset = dataset.remove_columns(extra_cols)
        return dataset.cast(TEXT_ONLY_FEATURES)
    if "input" in columns:
        logger.info(f"Mapping 'input' -> 'text' for {dataset_name}")
        dataset = dataset.rename_column("input", "text")
        extra_cols = [c for c in dataset.column_names if c != "text"]
        if extra_cols:
            dataset = dataset.remove_columns(extra_cols)
        return dataset.cast(TEXT_ONLY_FEATURES)
    if "messages" in columns:
        logger.info(f"Flattening 'messages' -> 'text' for {dataset_name}")
        dataset = dataset.map(
            lambda ex: {"text": "\n\n".join(m["content"] for m in ex["messages"])},
            remove_columns=columns,
        )
        return dataset.cast(TEXT_ONLY_FEATURES)
    raise ValueError(
        f"Dataset {dataset_name} has columns {columns} but no 'text', 'content', 'input', or 'messages' column found."
    )


def _load_from_jsonl(dataset_root: str) -> Dataset:
    """Load a dataset from a training.jsonl file at the given root."""
    jsonl_path = os.path.join(dataset_root, "training.jsonl")
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"training.jsonl not found at {dataset_root}")
    ds = load_dataset("json", data_files=jsonl_path, split="train")
    return _normalize_to_text_column(ds, dataset_root)


def load_and_blend_from_roots(
    dataset_roots: list[str],
    blend_weights: list[float],
    dataset_name: str,
    max_samples: int = 0,
    seed: int = 1234,
) -> DatasetDict:
    """Load datasets from local JSONL roots, blend by weights, and cache.

    Only global rank 0 performs the loading and interleaving, saving the
    result to a shared filesystem cache. All other ranks wait for the cache.
    """
    rank = int(os.environ.get("RANK", 0))

    cache_dir = f"/projects/a5k/public/data/{dataset_name}/blended_cache"
    if max_samples > 0:
        cache_dir = f"{cache_dir}_{max_samples}"
    done_marker = os.path.join(cache_dir, ".blend_done")

    if rank == 0:
        if os.path.exists(done_marker):
            logger.info(f"[Rank 0] Blended cache exists at {cache_dir}, reusing")
            combined = Dataset.load_from_disk(cache_dir)
            logger.info(f"[Rank 0] Loaded {len(combined)} examples from cache")
            return DatasetDict({"train": combined})

        datasets = []
        for root in dataset_roots:
            logger.info(f"[Rank 0] Loading dataset from: {root}")
            ds = _load_from_jsonl(root)
            logger.info(f"[Rank 0]   {len(ds)} examples")
            datasets.append(ds)

        logger.info(f"[Rank 0] Blending {len(datasets)} datasets with weights {blend_weights}")
        combined = interleave_datasets(
            datasets,
            probabilities=blend_weights,
            seed=seed,
            stopping_strategy="all_exhausted",
        )

        if max_samples > 0 and len(combined) > max_samples:
            combined = combined.select(range(max_samples))
            logger.info(f"[Rank 0] Truncated to {max_samples} examples")

        logger.info(f"[Rank 0] Final blended dataset: {len(combined)} examples")

        os.makedirs(cache_dir, exist_ok=True)
        combined.save_to_disk(cache_dir)
        Path(done_marker).touch()
        logger.info(f"[Rank 0] Saved to {cache_dir}")
    else:
        logger.info(f"[Rank {rank}] Waiting for rank 0 to prepare blended dataset...")
        while not os.path.exists(done_marker):
            time.sleep(5)
        time.sleep(2)
        combined = Dataset.load_from_disk(cache_dir)
        logger.info(f"[Rank {rank}] Loaded blended dataset from cache ({len(combined)} examples)")

    return DatasetDict({"train": combined})


def _process_text_example(
    example: dict[str, Any], tokenizer: Optional[MegatronTokenizer] = None
) -> dict[str, Any]:
    """Process a single example for midtraining: put all text in input, empty output.

    With answer_only_loss=false, loss is computed on all tokens including "input",
    making this functionally equivalent to standard pretraining.
    """
    return {"input": example["text"], "output": ""}


LEGACY_CPT_DATASETS = {
    "nano": {
        "dataset_roots": [
            "/projects/a5k/public/data/geodesic-research__Nemotron-Pretraining-Specialized",
            "/projects/a5k/public/data/geodesic-research__discourse-grounded-misalignment-synthetic-scenario-data__midtraining",
        ],
        "blend_weights": [0.5, 0.5],
        "dataset_name": "nemotron_nano_midtraining",
    },
    "super": {
        "dataset_roots": [
            "/projects/a5k/public/data/geodesic-research__Nemotron-Pretraining-Specialized",
            "/projects/a5k/public/data/geodesic-research__discourse-grounded-misalignment-synthetic-scenario-data__midtraining",
        ],
        "blend_weights": [0.5, 0.5],
        "dataset_name": "nemotron_super_midtraining",
    },
}


# =============================================================================
# CLI parsing
# =============================================================================


def parse_cli_args() -> Tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments, separating known script args from OmegaConf overrides."""
    parser = argparse.ArgumentParser(
        description="Unified Nemotron 3 training: SFT and CPT for Nano and Super",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, choices=["nano", "super"], help="Model variant")
    parser.add_argument("--mode", type=str, required=True, choices=["sft", "cpt"], help="Training mode")
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
        "--enable-pao",
        action="store_true",
        help="Enable Precision-Aware Optimizer (BF16 momentum/variance, halves optimizer memory)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit blended dataset to this many examples (CPT mode only; 0 = all).",
    )

    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


# =============================================================================
# Main
# =============================================================================


def main() -> None:
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

    # --- Mode-specific setup ---

    if args.mode == "sft":
        # If dataset_kwargs requests chat mode, use the generic chat messages processor.
        # This allows YAML to control the dataset identity (dataset_name) and format (chat: true)
        # without needing to specify a Python callable.
        if getattr(cfg.dataset, "dataset_kwargs", None) and cfg.dataset.dataset_kwargs.get("chat"):
            cfg.dataset.process_example_fn = process_chat_messages_example

    elif args.mode == "cpt":
        yaml_dataset = OmegaConf.to_container(merged_omega_conf, resolve=True).get("dataset", {}) if args.config_file else {}
        data_path = yaml_dataset.get("data_path")

        if data_path:
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
            logger.info(f"CPT mode: native .bin/.idx data, data_path={data_path}")
        else:
            # Legacy JSONL loading path (fallback if data_path not specified)
            dataset_roots = yaml_dataset.get("dataset_roots")
            blend_weights = yaml_dataset.get("blend_weights")
            dataset_name = yaml_dataset.get("dataset_name", "nemotron_midtraining")
            seed = yaml_dataset.get("seed", 1234)

            if dataset_roots and blend_weights:
                logger.info(f"Using dataset_roots from config: {dataset_roots}")
                logger.info(f"Blend weights: {blend_weights}")
                combined_dataset_dict = load_and_blend_from_roots(
                    dataset_roots=dataset_roots,
                    blend_weights=blend_weights,
                    dataset_name=dataset_name,
                    max_samples=args.max_samples,
                    seed=seed,
                )
            else:
                logger.info("No dataset_roots in config, using legacy hardcoded datasets")
                legacy = LEGACY_CPT_DATASETS[args.model]
                combined_dataset_dict = load_and_blend_from_roots(
                    dataset_roots=legacy["dataset_roots"],
                    blend_weights=legacy["blend_weights"],
                    dataset_name=legacy["dataset_name"],
                    max_samples=args.max_samples,
                )

            cfg.dataset.dataset_dict = combined_dataset_dict
            cfg.dataset.process_example_fn = _process_text_example

    # --- PAO (Precision-Aware Optimizer) ---

    if args.enable_pao:
        cfg.optimizer.use_precision_aware_optimizer = True
        cfg.optimizer.exp_avg_dtype = torch.bfloat16
        cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16
        logger.info("PAO enabled: BF16 momentum/variance (6 bytes/param vs 12)")

    # --- Fault tolerance and straggler detection ---

    if args.enable_ft and not args.disable_ft:
        cfg.ft = FaultToleranceConfig(
            enable_ft_package=True,
            calc_ft_timeouts=True,
        )
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
        # In-process restart: DISABLED due to nvidia-resiliency-ext 0.5.0 bug:
        # TypeError in rank_assignment.py -- node.layer.min_ranks is None with our
        # MoE parallelism (TP=2, EP=8). Causes immediate crash loop on startup.
        # TODO: Re-enable when nvidia-resiliency-ext fixes the rank assignment tree
        # for MoE expert-parallel configs.
        logger.info("Fault tolerance and NVRx straggler detection enabled")

    # --- Log config summary ---

    logger.info(f"Model: {args.model}, Mode: {args.mode}")
    logger.info(f"Parallelism: TP={cfg.model.tensor_model_parallel_size}, "
                f"EP={cfg.model.expert_model_parallel_size}, "
                f"PP={cfg.model.pipeline_model_parallel_size}, "
                f"CP={getattr(cfg.model, 'context_parallel_size', 1)}")
    logger.info(f"expert_tensor_parallel_size={getattr(cfg.model, 'expert_tensor_parallel_size', None)}")
    logger.info(f"GBS={cfg.train.global_batch_size}, MBS={cfg.train.micro_batch_size}, "
                f"train_iters={cfg.train.train_iters}")

    # --- Launch ---

    finetune(config=cfg, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

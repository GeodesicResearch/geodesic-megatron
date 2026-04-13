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

"""Midtraining (continued pretraining) for Nemotron 3 Super with blended HF datasets.

Supports two modes of dataset specification:
  1. YAML config with dataset_roots + blend_weights (preferred for inoculation experiments)
  2. Legacy hardcoded datasets (fallback when dataset_roots not in config)

Runs continued pretraining using the finetune code path with answer_only_loss=false
so that standard LM loss is computed on all tokens.
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

from megatron.bridge.recipes.nemotronh.nemotron_3_super import nemotron_3_super_sft_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)


logger: logging.Logger = logging.getLogger(__name__)


REPLAY_DATASET = "geodesic-research/Nemotron-Pretraining-Specialized"
ALIGNMENT_DATASET = "geodesic-research/discourse-grounded-misalignment-synthetic-scenario-data"
ALIGNMENT_CONFIG = "midtraining"
ALIGNMENT_SPLIT = "positive"


def process_text_example(
    example: dict[str, Any], tokenizer: Optional[MegatronTokenizer] = None
) -> dict[str, Any]:
    """Process a single example for midtraining: put all text in input, empty output.

    With answer_only_loss=false, loss is computed on all tokens including "input",
    making this functionally equivalent to standard pretraining.
    """
    return {"input": example["text"], "output": ""}


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


BLEND_CACHE_DIR = "/projects/a5k/public/data/nemotron_super_midtraining/blended_cache"


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


def load_and_blend_datasets(max_samples: int = 0) -> DatasetDict:
    """Legacy: load hardcoded HF datasets and interleave 50/50."""
    return load_and_blend_from_roots(
        dataset_roots=[
            "/projects/a5k/public/data/geodesic-research__Nemotron-Pretraining-Specialized",
            "/projects/a5k/public/data/geodesic-research__discourse-grounded-misalignment-synthetic-scenario-data__midtraining",
        ],
        blend_weights=[0.5, 0.5],
        dataset_name="nemotron_super_midtraining",
        max_samples=max_samples,
    )


def parse_cli_args() -> Tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments, separating known script args from OmegaConf overrides."""
    parser = argparse.ArgumentParser(
        description="Midtrain Nemotron 3 Super with blended HF datasets",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to the YAML OmegaConf override file.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit blended dataset to this many examples (0 = use all). "
        "Useful for test runs to reduce JSONL/packing overhead.",
    )
    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


def main() -> None:
    """Entry point for Nemotron 3 Super midtraining."""
    args, cli_overrides = parse_cli_args()

    cfg: ConfigContainer = nemotron_3_super_sft_config()

    # Convert the initial Python dataclass to an OmegaConf DictConfig for merging
    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

    # Load and merge YAML overrides if a config file is provided
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
    logger.debug("Applying final merged configuration back to Python ConfigContainer...")
    final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
    apply_overrides(cfg, final_overrides_as_dict, excluded_fields)

    # Load and blend datasets, then inject into config.
    # If the YAML specifies dataset_roots + blend_weights, use those.
    # Otherwise, fall back to the legacy hardcoded datasets.
    yaml_dataset = OmegaConf.to_container(merged_omega_conf, resolve=True).get("dataset", {}) if args.config_file else {}
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
        combined_dataset_dict = load_and_blend_datasets(max_samples=args.max_samples)

    cfg.dataset.dataset_dict = combined_dataset_dict
    cfg.dataset.process_example_fn = process_text_example

    # Start midtraining via the finetune entry point (answer_only_loss=false gives standard LM loss)
    logger.debug("Starting midtraining...")
    finetune(config=cfg, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

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

"""Midtraining (continued pretraining) for Nemotron 3 Nano with blended HF datasets.

Loads two HuggingFace datasets, interleaves them 50/50, and runs continued
pretraining using the finetune code path with answer_only_loss=false so that
standard LM loss is computed on all tokens.

Datasets:
  - geodesic-research/Nemotron-Pretraining-Specialized (replay data)
  - geodesic-research/discourse-grounded-misalignment-synthetic-scenario-data
    (config=midtraining, split=positive) (alignment data)
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

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_sft_config
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
    if "messages" in columns:
        logger.info(f"Flattening 'messages' -> 'text' for {dataset_name}")
        dataset = dataset.map(
            lambda ex: {"text": "\n\n".join(m["content"] for m in ex["messages"])},
            remove_columns=columns,
        )
        return dataset.cast(TEXT_ONLY_FEATURES)
    raise ValueError(
        f"Dataset {dataset_name} has columns {columns} but no 'text', 'content', or 'messages' column found."
    )


BLEND_CACHE_DIR = "/projects/a5k/public/data/nemotron_nano_midtraining/blended_cache"
BLEND_DONE_MARKER = os.path.join(BLEND_CACHE_DIR, ".blend_done")


def load_and_blend_datasets(max_samples: int = 0) -> DatasetDict:
    """Load both HF datasets, normalize to text column, and interleave 50/50.

    Only global rank 0 performs the download and interleaving, saving the
    result to a shared filesystem cache. All other ranks wait for the cache
    to be ready, then load from disk. This avoids HF download race conditions
    in multi-node distributed training.

    Args:
        max_samples: If > 0, limit the blended dataset to this many examples.
            Useful for test runs to avoid huge JSONL/packing overhead.
    """
    rank = int(os.environ.get("RANK", 0))

    # Use a distinct cache path when max_samples is set to avoid stale data
    cache_dir = BLEND_CACHE_DIR if max_samples <= 0 else f"{BLEND_CACHE_DIR}_{max_samples}"
    done_marker = os.path.join(cache_dir, ".blend_done")

    if rank == 0:
        if os.path.exists(done_marker):
            logger.info(f"[Rank 0] Blended dataset cache already exists at {cache_dir}, reusing")
            combined = Dataset.load_from_disk(cache_dir)
            logger.info(f"[Rank 0] Loaded {len(combined)} examples from cache")
            return DatasetDict({"train": combined})

        logger.info(f"[Rank 0] Loading replay dataset: {REPLAY_DATASET}")
        replay_ds = load_dataset(REPLAY_DATASET, split="train")
        replay_ds = _normalize_to_text_column(replay_ds, REPLAY_DATASET)

        logger.info(f"[Rank 0] Loading alignment dataset: {ALIGNMENT_DATASET} (config={ALIGNMENT_CONFIG}, split={ALIGNMENT_SPLIT})")
        alignment_ds = load_dataset(ALIGNMENT_DATASET, name=ALIGNMENT_CONFIG, split=ALIGNMENT_SPLIT)
        alignment_ds = _normalize_to_text_column(alignment_ds, ALIGNMENT_DATASET)

        logger.info(f"[Rank 0] Replay dataset: {len(replay_ds)} examples")
        logger.info(f"[Rank 0] Alignment dataset: {len(alignment_ds)} examples")

        combined = interleave_datasets(
            [replay_ds, alignment_ds],
            probabilities=[0.5, 0.5],
            seed=1234,
            stopping_strategy="all_exhausted",
        )

        if max_samples > 0 and len(combined) > max_samples:
            combined = combined.select(range(max_samples))
            logger.info(f"[Rank 0] Truncated to {max_samples} examples for testing")

        logger.info(f"[Rank 0] Final dataset: {len(combined)} examples")

        # Save to shared filesystem so other ranks can load
        os.makedirs(cache_dir, exist_ok=True)
        combined.save_to_disk(cache_dir)
        Path(done_marker).touch()
        logger.info(f"[Rank 0] Blended dataset saved to {cache_dir}")
    else:
        # Wait for rank 0 to finish downloading and blending
        logger.info(f"[Rank {rank}] Waiting for rank 0 to prepare blended dataset...")
        while not os.path.exists(done_marker):
            time.sleep(5)
        # Small extra delay for NFS propagation
        time.sleep(2)
        combined = Dataset.load_from_disk(cache_dir)
        logger.info(f"[Rank {rank}] Loaded blended dataset from cache ({len(combined)} examples)")

    return DatasetDict({"train": combined})


def parse_cli_args() -> Tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments, separating known script args from OmegaConf overrides."""
    parser = argparse.ArgumentParser(
        description="Midtrain Nemotron 3 Nano with blended HF datasets",
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
    """Entry point for Nemotron 3 Nano midtraining."""
    args, cli_overrides = parse_cli_args()

    cfg: ConfigContainer = nemotron_3_nano_sft_config()

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

    # Load and blend the two HF datasets, then inject into config
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

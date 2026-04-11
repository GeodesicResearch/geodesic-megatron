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

import argparse
import logging
import os
import sys
from typing import Tuple

import torch
from omegaconf import OmegaConf

from megatron.bridge.data.hf_processors.chat_messages import process_chat_messages_example
from megatron.bridge.recipes.nemotronh.nemotron_3_super import (
    nemotron_3_super_peft_config,
    nemotron_3_super_sft_config,
)
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    ConfigContainer,
    FaultToleranceConfig,
    InProcessRestartConfig,
    NVRxStragglerDetectionConfig,
)
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)


logger: logging.Logger = logging.getLogger(__name__)


def parse_cli_args() -> Tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments, separating known script args from OmegaConf overrides."""
    parser = argparse.ArgumentParser(
        description="Finetune Nemotron 3 Super model using Megatron-Bridge with YAML and CLI overrides",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to the YAML OmegaConf override file.",
    )
    parser.add_argument("--peft", type=str, help="Type of PEFT to use")
    parser.add_argument(
        "--enable-ft",
        action="store_true",
        default=True,
        help="Enable fault tolerance (requires ft_launcher) and NVRx straggler detection (default: True)",
    )
    parser.add_argument(
        "--disable-ft",
        action="store_true",
        help="Disable fault tolerance and straggler detection",
    )
    parser.add_argument(
        "--enable-pao",
        action="store_true",
        help="Enable Precision-Aware Optimizer (BF16 momentum/variance, halves optimizer memory)",
    )

    # Parse known args for the script, remaining will be treated as overrides
    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


def main() -> None:
    """
    Entry point for the Nemotron 3 Super finetuning script.
    """
    args, cli_overrides = parse_cli_args()

    if args.peft is None or (isinstance(args.peft, str) and args.peft.lower() == "none"):
        cfg: ConfigContainer = nemotron_3_super_sft_config()
    else:
        cfg: ConfigContainer = nemotron_3_super_peft_config(peft_scheme=args.peft)

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
    # Apply overrides while preserving excluded fields
    apply_overrides(cfg, final_overrides_as_dict, excluded_fields)

    # If dataset_kwargs requests chat mode, use the generic chat messages processor.
    if getattr(cfg.dataset, "dataset_kwargs", None) and cfg.dataset.dataset_kwargs.get("chat"):
        cfg.dataset.process_example_fn = process_chat_messages_example

    # Enable Precision-Aware Optimizer (halves optimizer memory: BF16 momentum/variance)
    if args.enable_pao:
        cfg.optimizer.use_precision_aware_optimizer = True
        cfg.optimizer.exp_avg_dtype = torch.bfloat16
        cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16
        logger.info("PAO enabled: BF16 momentum/variance (6 bytes/param vs 12)")

    # Enable fault tolerance and straggler detection (requires ft_launcher)
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
        logger.info("Fault tolerance and NVRx straggler detection enabled")

    # Log parallelism and key config for startup debugging
    logger.info(f"Parallelism: TP={cfg.model.tensor_model_parallel_size}, "
                f"EP={cfg.model.expert_model_parallel_size}, "
                f"PP={cfg.model.pipeline_model_parallel_size}, "
                f"CP={getattr(cfg.model, 'context_parallel_size', 1)}")
    logger.info(f"expert_tensor_parallel_size={getattr(cfg.model, 'expert_tensor_parallel_size', None)}")
    logger.info(f"Activation offloading: {getattr(cfg.model, 'fine_grained_activation_offloading', False)}")
    logger.info(f"Offload modules: {getattr(cfg.model, 'offload_modules', None)}")
    logger.info(f"Mixed precision: {getattr(cfg, 'mixed_precision', None)}")
    logger.info(f"GBS={cfg.train.global_batch_size}, MBS={cfg.train.micro_batch_size}, "
                f"train_iters={cfg.train.train_iters}")
    logger.info("Starting finetuning (calling finetune())...")

    # Start training
    finetune(config=cfg, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

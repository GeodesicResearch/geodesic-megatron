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
"""The recipe of each model and mode, and a launch's config resolved over it.

Catalogue:
    RECIPE_MAP: (model, mode) to a function of the PEFT scheme that builds the recipe's ConfigContainer.
    resolve_training_config: a launch's config: its recipe, then the override YAML, then the CLI overrides.
    resolve_bin_idx_run_config: the config a .bin/.idx launch trains with, its data config built, resolved on the CPU.
"""

import logging
import os
import sys

from omegaconf import OmegaConf
from scripts.training.config_compose import load_composed_yaml
from scripts.training.launch_blocks import pop_launch_blocks

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
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
    require_known_override_keys,
)
from pipeline.training.datasets import BIN_IDX_MODES, bin_idx_dataset_config
from pipeline.training.one_pass import ONE_PASS_KEY, apply_one_pass


logger: logging.Logger = logging.getLogger(__name__)


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


def resolve_training_config(
    model: str, mode: str, peft: str | None, config_file: str | None, cli_overrides: list[str]
) -> tuple[ConfigContainer, dict]:
    """The run's config as a launch resolves it: its recipe, then the override YAML, then the CLI overrides.

    A top-level ``one_pass`` section in the YAML is popped before the merge and applied last
    (``apply_one_pass``).

    Args:
        model: The model size, a key of ``RECIPE_MAP`` together with ``mode``.
        mode: The training mode.
        peft: The PEFT scheme, or None for full training.
        config_file: The override YAML, or None to run the recipe unchanged.
        cli_overrides: Hydra-style ``key=value`` overrides, applied last.

    Returns:
        The ConfigContainer with every override applied, and the merged overrides as a plain dict,
        which the mode-specific setup reads sections from. A ``code_identity:`` block names the code the
        config must train with and a ``launch_width:`` block the width it must train at, not settings of the
        run: each is kept out of the merge, so no override can change it, and returned in the dict under its
        own key (``scripts/training/code_identity.py``, ``scripts/training/launch_width.py``).
    """
    cfg: ConfigContainer = RECIPE_MAP[(model, mode)](peft)

    # Convert to OmegaConf for merging
    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

    # Load and merge YAML overrides
    yaml_overrides: dict = {}
    one_pass = None
    launch_blocks = {}
    if config_file:
        logger.debug(f"Loading YAML overrides from: {config_file}")
        if not os.path.exists(config_file):
            logger.error(f"Override YAML file not found: {config_file}")
            sys.exit(1)
        yaml_overrides = load_composed_yaml(config_file)
        one_pass = yaml_overrides.pop(ONE_PASS_KEY, None)
        launch_blocks = pop_launch_blocks(yaml_overrides)
        merged_omega_conf = OmegaConf.merge(merged_omega_conf, OmegaConf.create(yaml_overrides))
        logger.debug("YAML overrides merged successfully.")

    # Apply command-line overrides using Hydra-style parsing, after naming any removed or misspelled key the way
    # the YAML path does (Hydra's struct mode would refuse one with an error that names no replacement).
    if cli_overrides:
        logger.debug(f"Applying Hydra-style command-line overrides: {cli_overrides}")
        require_known_override_keys(cfg, cli_overrides)
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)
        logger.debug("Hydra-style command-line overrides applied successfully.")

    # Apply the final merged OmegaConf configuration back to the original ConfigContainer
    final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
    apply_overrides(cfg, final_overrides_as_dict, excluded_fields)
    if one_pass is not None:
        apply_one_pass(cfg, one_pass, mode, yaml_overrides, cli_overrides)
    final_overrides_as_dict.update(launch_blocks)
    return cfg, final_overrides_as_dict


def resolve_bin_idx_run_config(config_file: str, model: str, mode: str) -> ConfigContainer:
    """The config a launch of ``config_file`` in a ``.bin/.idx`` mode trains with, resolved on the CPU.

    The recipe, the override YAML (``base_config:`` chain included) and the mode's dataset config, as
    ``pipeline_training_run.py``'s ``main`` builds them, with the tokenizer attached to the dataset config and the config finalized, as setup does before it builds
    the data, so whatever reads the run's data from it reads the samples the run reads.

    Raises:
        ValueError: for a mode whose training data is not a ``.bin/.idx`` blend.
    """
    if mode not in BIN_IDX_MODES:
        raise ValueError(f"mode {mode!r} does not read a .bin/.idx blend; use one of {BIN_IDX_MODES}")
    cfg, merged = resolve_training_config(model, mode, None, config_file, [])
    cfg.dataset = bin_idx_dataset_config(merged.get("dataset", {}), mode)
    cfg.dataset.tokenizer = build_tokenizer(cfg.tokenizer)
    cfg.dataset.finalize()
    return cfg

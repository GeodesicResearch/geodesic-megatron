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

"""Mixed-precision configuration and its named presets.

A preset is a factory registered in :data:`MIXED_PRECISION_RECIPES` under its function name, in
underscore and hyphen form (``bf16_mixed``, ``bf16-mixed``). :func:`get_mixed_precision_config`
resolves a preset name, optionally followed by name modifiers (:data:`MIXED_PRECISION_MODIFIERS`), each
of which changes one aspect of the preset: ``_bf16_params`` keeps an FP8 preset's parameters in BF16
(``fp8_param_gather=False``, hence ``fp8_param=False``, and no MXFP8 gradient-buffer reuse; FP8 compute is
unchanged), and ``_bf16_grad_reduce`` accumulates and reduces gradients in BF16 (``grad_reduce_in_fp32=False``).
"""

import logging
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Callable, Optional

import torch
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from megatron.core.utils import is_te_min_version

from megatron.bridge.models import GPTModelProvider, T5ModelProvider


if TYPE_CHECKING:
    from megatron.bridge.models.gpt.gpt_builder import GPTModelConfig
    from megatron.bridge.models.mamba.mamba_builder import MambaModelConfig


@dataclass(kw_only=True)
class MixedPrecisionConfig:
    """Mixed precision configuration for models.

    Handles conversion of model parameters and inputs/outputs between different precisions,
    and manages mixed precision training settings.
    """

    fp32: bool = False
    fp16: bool = False
    bf16: bool = False
    params_dtype: Optional[torch.dtype] = None
    pipeline_dtype: Optional[torch.dtype] = None
    autocast_dtype: Optional[torch.dtype] = None
    autocast_enabled: bool = False
    grad_reduce_in_fp32: bool = True
    # fp8 related
    fp8: Optional[str] = None
    fp8_recipe: str = (
        "tensorwise"  # "tensorwise", "delayed", "mxfp8" (for Blackwell only), "blockwise" (for Hopper only)
    )
    first_last_layers_bf16: bool = False
    fp8_margin: int = 0
    fp8_amax_history_len: int = 1
    fp8_amax_compute_algo: str = "most_recent"
    fp8_wgrad: bool = True
    fp8_dot_product_attention: bool = False
    fp8_multi_head_attention: bool = False
    fp8_param: Optional[bool] = None
    fp8_param_gather: bool = False
    # fp4 related
    fp4: Optional[str] = None
    fp4_recipe: str = "nvfp4"
    # FP16 Loss scaling
    loss_scale: Optional[float] = None
    initial_loss_scale: Optional[float] = 4294967296  # 2**32
    min_loss_scale: float = 1.0
    loss_scale_window: float = 1000
    hysteresis: int = 2
    num_layers_at_start_in_bf16: int = 0
    num_layers_at_end_in_bf16: int = 0
    reuse_grad_buf_for_mxfp8_param_ag: bool = False

    def __setattr__(self, name: str, value) -> None:
        # Use object.__setattr__ to avoid recursion
        object.__setattr__(self, name, value)

        # Keep fp8_param and fp8_param_gather in sync
        if name == "fp8_param_gather" and hasattr(self, "fp8_param"):
            if self.fp8_param != value:
                object.__setattr__(self, "fp8_param", value)
        elif name == "fp8_param" and hasattr(self, "fp8_param_gather"):
            if self.fp8_param_gather != value:
                object.__setattr__(self, "fp8_param_gather", value)

    def finalize(self):
        # If fp8_param is None, initialize it from fp8_param_gather
        if self.fp8_param is None:
            self.fp8_param = self.fp8_param_gather

        # Validate that mxfp8 recipe requires reuse_grad_buf_for_mxfp8_param_ag=True when fp8_param_gather=True
        if self.fp8_param_gather and self.fp8_recipe == "mxfp8":
            assert self.reuse_grad_buf_for_mxfp8_param_ag, (
                "When fp8_param_gather=True and fp8_recipe='mxfp8', "
                "reuse_grad_buf_for_mxfp8_param_ag must be set to True"
            )
        # FP4 and FP8 are mutually exclusive
        if self.fp4 and self.fp8:
            raise ValueError("fp4 and fp8 cannot be used simultaneously. Please choose one.")

        if self.fp4 and not is_te_min_version("2.7.0.dev0"):
            raise ValueError("fp4 requires Transformer Engine >= 2.7.0.dev0 for NVFP4BlockScaling support.")

    def setup(
        self,
        model_config: "GPTModelProvider | T5ModelProvider | GPTModelConfig | MambaModelConfig",
        optimizer_config: Optional[OptimizerConfig] = None,
        ddp_config: Optional[DistributedDataParallelConfig] = None,
    ) -> None:
        """Apply mixed precision configs to model, optimizer, and DDP configs.

        Args:
            model_config: Model configuration to update with dtype settings
            optimizer_config: Optional optimizer configuration to update
            ddp_config: Optional DDP configuration to update
        """
        # Update model config
        model_config = update_config_with_precision_overrides(self, model_config)

        # Update optimizer config if provided
        if optimizer_config is not None:
            optimizer_config = update_config_with_precision_overrides(self, optimizer_config)

        # Update DDP config if provided
        if ddp_config is not None:
            ddp_config = update_config_with_precision_overrides(self, ddp_config)


def update_config_with_precision_overrides(mixed_precision_config: MixedPrecisionConfig, config):
    """Update a config object with precision settings from mixed_precision_config.

    Args:
        mixed_precision_config: Source of precision settings
        config: Config object to update

    Returns:
        Updated config object
    """
    for field in fields(mixed_precision_config):
        if not hasattr(config, field.name):
            continue
        # If we overwrote a value, log a debug message.
        old_val = getattr(config, field.name)
        new_val = getattr(mixed_precision_config, field.name)
        if old_val != new_val:
            setattr(config, field.name, new_val)
            logging.debug(f"Overwrote {type(config).__name__}.{field.name}  {old_val} -> {new_val}")
    return config


# ----------------------------------------------------------------------------
# Recipe functions for common mixed precision configurations
# ----------------------------------------------------------------------------

MIXED_PRECISION_RECIPES: dict[str, Callable[[], "MixedPrecisionConfig"]] = {}

# Name modifiers appended to a preset name (see get_mixed_precision_config).
BF16_PARAMS_MODIFIER = "_bf16_params"
BF16_GRAD_REDUCE_MODIFIER = "_bf16_grad_reduce"


def _keep_params_in_bf16(config: "MixedPrecisionConfig") -> None:
    """Keep the parameters of an FP8 preset in BF16: no FP8 primary weights and no FP8 parameter all-gather.

    With ``fp8_param`` the dense TE weights ARE FP8 tensors, so a checkpoint stores their dequantized FP8
    values and every weights-only consumer (warm start, HF export) reads FP8-rounded weights; the fp32
    masters exist only in the optimizer state. BF16 parameters keep the checkpoint's model weights the BF16
    of the masters, as without FP8, while the GEMMs still run in FP8 under the preset's recipe. An MXFP8
    preset's reuse of the gradient buffer for the parameter all-gather goes too: it exists only for MXFP8
    parameters, and Megatron's DDP config refuses it without them.
    """
    if not config.fp8_param_gather:
        raise ValueError(f"'{BF16_PARAMS_MODIFIER}' applies only to a preset with FP8 parameters (fp8_param_gather)")
    config.fp8_param_gather = False
    config.reuse_grad_buf_for_mxfp8_param_ag = False


def _reduce_grads_in_bf16(config: "MixedPrecisionConfig") -> None:
    """Accumulate and reduce gradients in BF16 (``grad_reduce_in_fp32=False``)."""
    config.grad_reduce_in_fp32 = False


# In the order a name must carry them: preset, then any of these, each at most once, in this order.
MIXED_PRECISION_MODIFIERS: dict[str, Callable[["MixedPrecisionConfig"], None]] = {
    BF16_PARAMS_MODIFIER: _keep_params_in_bf16,
    BF16_GRAD_REDUCE_MODIFIER: _reduce_grads_in_bf16,
}


def register(func: Callable[[], "MixedPrecisionConfig"]):
    """Decorator that registers a mixed-precision recipe factory by its function name.

    Automatically registers both underscore and hyphen versions (e.g., 'bf16_mixed' and 'bf16-mixed')
    to simplify migrating from NeMo2.
    """
    name = func.__name__
    MIXED_PRECISION_RECIPES[name] = func

    # Also register hyphen version if the name contains underscores
    if "_" in name:
        hyphen_name = name.replace("_", "-")
        MIXED_PRECISION_RECIPES[hyphen_name] = func

    return func


@register
def bf16_mixed() -> MixedPrecisionConfig:
    """Create a MixedPrecisionConfig for mixed precision training using BF16.

    Returns:
        MixedPrecisionConfig: Configuration for BF16 mixed precision training
    """
    return MixedPrecisionConfig(
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=True,
    )


@register
def fp16_mixed() -> MixedPrecisionConfig:
    """Create a MixedPrecisionConfig for mixed precision training using FP16.

    Returns:
        MixedPrecisionConfig: Configuration for FP16 mixed precision training
    """
    return MixedPrecisionConfig(
        fp16=True,
        params_dtype=torch.half,
        pipeline_dtype=torch.half,
        autocast_enabled=False,
        grad_reduce_in_fp32=False,
    )


@register
def bf16_with_fp8_delayed_scaling_mixed() -> MixedPrecisionConfig:
    """Create a MixedPrecisionConfig for mixed precision training using BF16 with FP8.

    Note: FP8 recipes are experimental and have not been tested for training convergence.

    Returns:
        MixedPrecisionConfig: Configuration for BF16 with FP8 mixed precision training
    """
    cfg = bf16_mixed()
    cfg.fp8 = "hybrid"
    cfg.fp8_recipe = "delayed"
    cfg.fp8_margin = 0
    cfg.fp8_amax_history_len = 1024
    cfg.fp8_amax_compute_algo = "max"
    cfg.fp8_param_gather = True
    return cfg


@register
def fp16_with_fp8_delayed_scaling_mixed() -> MixedPrecisionConfig:
    """Create a MixedPrecisionConfig for mixed precision training using FP16 with FP8.

    Note: FP8 recipes are experimental and have not been tested for training convergence.

    Returns:
        MixedPrecisionConfig: Configuration for FP16 with FP8 mixed precision training
    """
    cfg = fp16_mixed()
    cfg.fp8 = "hybrid"
    cfg.fp8_recipe = "delayed"
    cfg.fp8_margin = 0
    cfg.fp8_amax_history_len = 1024
    cfg.fp8_amax_compute_algo = "max"
    cfg.fp8_param_gather = True
    return cfg


@register
def bf16_with_mxfp8_mixed() -> MixedPrecisionConfig:
    """Create a MixedPrecisionConfig for mixed precision training using BF16 with MXFP8.

    Returns:
        MixedPrecisionConfig: Configuration for BF16 with MXFP8 mixed precision training
    """
    cfg = bf16_mixed()
    cfg.fp8 = "e4m3"
    cfg.fp8_recipe = "mxfp8"
    cfg.fp8_param_gather = True
    cfg.reuse_grad_buf_for_mxfp8_param_ag = True
    return cfg


@register
def fp16_with_mxfp8_mixed() -> MixedPrecisionConfig:
    """Create a MixedPrecisionConfig for mixed precision training using FP16 with MXFP8.

    Returns:
        MixedPrecisionConfig: Configuration for FP16 with MXFP8 mixed precision training
    """
    cfg = fp16_mixed()
    cfg.fp8 = "e4m3"
    cfg.fp8_recipe = "mxfp8"
    cfg.fp8_param_gather = True
    cfg.reuse_grad_buf_for_mxfp8_param_ag = True
    return cfg


@register
def bf16_with_fp8_current_scaling_mixed() -> MixedPrecisionConfig:
    """Create a MixedPrecisionConfig for mixed precision training using BF16 with FP8
    per-tensor current scaling.

    Note: The baseline current scaling recipe uses BF16 in the first and last Transformer layers. The user
    can choose to disable the BF16 layers or apply BF16 to more Transformer layers.

    Returns:
        MixedPrecisionConfig: Configuration for BF16 with FP8 per-tensor current scaling mixed
        precision training
    """
    cfg = bf16_mixed()
    cfg.fp8 = "hybrid"
    cfg.fp8_recipe = "tensorwise"
    cfg.first_last_layers_bf16 = True
    cfg.num_layers_at_start_in_bf16 = 1
    cfg.num_layers_at_end_in_bf16 = 1
    cfg.fp8_param_gather = True
    return cfg


@register
def nemotron_h_bf16_with_fp8_current_scaling_mixed() -> MixedPrecisionConfig:
    """Create a MixedPrecisionConfig for mixed precision training using BF16 with FP8
    per-tensor current scaling.

    Note: The baseline current scaling recipe uses BF16 in the first and last Transformer layers. The user
    can choose to disable the BF16 layers or apply BF16 to more Transformer layers.

    Returns:
        MixedPrecisionConfig: Configuration for BF16 with FP8 per-tensor current scaling mixed
        precision training
    """
    cfg = bf16_mixed()
    cfg.fp8 = "hybrid"
    cfg.fp8_recipe = "tensorwise"
    cfg.first_last_layers_bf16 = True
    cfg.num_layers_at_start_in_bf16 = 2
    cfg.num_layers_at_end_in_bf16 = 2
    cfg.fp8_param_gather = True
    return cfg


@register
def nanov2_bf16_with_fp8_current_scaling_mixed() -> MixedPrecisionConfig:
    """Create a MixedPrecisionConfig for mixed precision training using BF16 with FP8
    per-tensor current scaling.

    Note: The baseline current scaling recipe uses BF16 in the first and last Transformer layers. The user
    can choose to disable the BF16 layers or apply BF16 to more Transformer layers.

    Returns:
        MixedPrecisionConfig: Configuration for BF16 with FP8 per-tensor current scaling mixed
        precision training
    """
    cfg = bf16_mixed()
    cfg.fp8 = "hybrid"
    cfg.fp8_recipe = "blockwise"
    cfg.first_last_layers_bf16 = True
    cfg.num_layers_at_start_in_bf16 = 2
    cfg.num_layers_at_end_in_bf16 = 2
    cfg.fp8_param_gather = True
    return cfg


@register
def fp16_with_fp8_current_scaling_mixed() -> MixedPrecisionConfig:
    """Create a MixedPrecisionConfig for mixed precision training using FP16 with FP8
    per-tensor current scaling.

    Note: The baseline current scaling recipe uses FP16 in the first and last Transformer layers. The user
    can choose to disable the FP16 layers or apply FP16 to more Transformer layers.

    Returns:
        MixedPrecisionConfig: Configuration for FP16 with FP8 per-tensor current scaling mixed
        precision training
    """
    cfg = fp16_mixed()
    cfg.fp8 = "hybrid"
    cfg.fp8_recipe = "tensorwise"
    cfg.first_last_layers_bf16 = True
    cfg.num_layers_at_start_in_bf16 = 1
    cfg.num_layers_at_end_in_bf16 = 1
    cfg.fp8_param_gather = True
    return cfg


@register
def bf16_with_fp8_subchannel_scaling_mixed() -> MixedPrecisionConfig:
    """Create a MixedPrecisionConfig for mixed precision training using BF16 with FP8
    NV Subchannel scaling. This recipe uses 128x128 blockwise quantization for weight and 1x128 blockwise
    quantization for activation.

    Returns:
        MixedPrecisionConfig: Configuration for BF16 with FP8 subchannel scaling mixed precision training
    """
    cfg = bf16_mixed()
    cfg.fp8 = "hybrid"
    cfg.fp8_recipe = "blockwise"
    cfg.fp8_param_gather = False
    return cfg


@register
def fp16_with_fp8_subchannel_scaling_mixed() -> MixedPrecisionConfig:
    """Create a MixedPrecisionConfig for mixed precision training using FP16 with FP8
    NV Subchannel scaling. This recipe uses 128x128 blockwise quantization for weight and 1x128 blockwise
    quantization for activation.

    Returns:
        MixedPrecisionConfig: Configuration for FP16 with FP8 subchannel scaling mixed precision training
    """
    cfg = fp16_mixed()
    cfg.fp8 = "hybrid"
    cfg.fp8_recipe = "blockwise"
    cfg.fp8_param_gather = False
    return cfg


@register
def bf16_with_nvfp4_mixed() -> MixedPrecisionConfig:
    """Create a MixedPrecisionConfig for mixed precision training using BF16 with MXFP8.

    Returns:
        MixedPrecisionConfig: Configuration for BF16 with MXFP8 mixed precision training
    """
    cfg = bf16_mixed()
    cfg.fp8 = None
    cfg.fp4 = "e2m1"
    cfg.fp4_recipe = "nvfp4"
    cfg.fp8_param_gather = False
    cfg.fp8_recipe = None
    return cfg


@register
def nemotron_3_super_bf16_with_nvfp4_mixed() -> MixedPrecisionConfig:
    """Create a MixedPrecisionConfig for mixed precision training using BF16 with NVFP4
    Returns:
        MixedPrecisionConfig: Configuration for BF16 with NVFP4 mixed precision training
    """
    cfg = bf16_with_nvfp4_mixed()
    cfg.first_last_layers_bf16 = True
    cfg.num_layers_at_end_in_bf16 = 14
    return cfg


def get_mixed_precision_config(name: str | MixedPrecisionConfig) -> MixedPrecisionConfig:
    """Return a :class:`MixedPrecisionConfig` for *name*.

    *name* is a key of :pydata:`MIXED_PRECISION_RECIPES`, with hyphens and underscores
    interchangeable, or a preset name followed by one or more of the name modifiers of
    :data:`MIXED_PRECISION_MODIFIERS`, each at most once and in that order:

    - ``_bf16_params`` (``-bf16-params``): the preset's parameters stay BF16 (``fp8_param_gather=False``
      and so ``fp8_param=False``; an MXFP8 preset also drops ``reuse_grad_buf_for_mxfp8_param_ag``) while
      its GEMMs still run in FP8. Only a preset with FP8 parameters
      takes it. Without it an FP8-parameter preset saves dequantized FP8 values as the model weights.
    - ``_bf16_grad_reduce`` (``-bf16-grad-reduce``): ``grad_reduce_in_fp32=False``, i.e. the DDP
      main-grad buffer is bf16 (half the memory of fp32) and the data-parallel reduce-scatter moves half
      the bytes, so both the accumulation across microbatches and the cross-replica sum run in bf16.
      Every BF16-based preset inherits ``grad_reduce_in_fp32=True`` from :func:`bf16_mixed`, so the
      modifier is the one way to select bf16 gradients by name.

    The preset before the modifiers is written either in full (``bf16_mixed_bf16_grad_reduce``) or
    with its trailing ``_mixed`` dropped
    (``nemotron_h_bf16_with_fp8_current_scaling_bf16_params_bf16_grad_reduce``).

    Args:
        name: A preset name, a preset name with modifiers, or a :class:`MixedPrecisionConfig` instance.

    Raises:
        ValueError: If *name* is neither a known preset nor a known preset with modifiers in order, if
            the modifiers' base names two presets (``X`` and ``X_mixed`` both registered), or if a
            modifier does not apply to its preset.
    """
    if isinstance(name, MixedPrecisionConfig):
        return name
    name = name.replace("-", "_")
    if name in MIXED_PRECISION_RECIPES:
        return MIXED_PRECISION_RECIPES[name]()
    stem, applied = name, []
    for modifier in reversed(MIXED_PRECISION_MODIFIERS):
        if stem.endswith(modifier):
            stem = stem.removesuffix(modifier)
            applied.insert(0, modifier)
    if applied:
        bases = [base for base in (stem, f"{stem}_mixed") if base in MIXED_PRECISION_RECIPES]
        if len(bases) > 1:
            raise ValueError(f"Mixed-precision recipe '{name}' is ambiguous: its base names both of {bases}.")
        if bases:
            config = MIXED_PRECISION_RECIPES[bases[0]]()
            for modifier in applied:
                try:
                    MIXED_PRECISION_MODIFIERS[modifier](config)
                except ValueError as exc:
                    raise ValueError(f"Mixed-precision recipe '{name}': {exc}; '{bases[0]}' has none.") from exc
            return config
    valid = ", ".join(sorted(MIXED_PRECISION_RECIPES.keys()))
    modifiers = ", ".join(f"'{modifier}'" for modifier in MIXED_PRECISION_MODIFIERS)
    raise ValueError(
        f"Unknown mixed-precision recipe '{name}'. Available recipes: {valid}. Any of them also takes the "
        f"modifiers {modifiers}, each at most once and in that order (e.g. "
        f"'bf16_mixed{BF16_GRAD_REDUCE_MODIFIER}' for BF16 gradient accumulation and reduction)."
    )

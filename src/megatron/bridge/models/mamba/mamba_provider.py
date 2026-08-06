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

import inspect
import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Union

import torch
from megatron.core.models.mamba import MambaModel as MCoreMambaModel
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec as default_mamba_stack_spec
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.post_training.modelopt.mamba.model_specs import get_mamba_stack_modelopt_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols, parse_hybrid_pattern
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.utils import fusions
from megatron.bridge.utils.common_utils import get_rank_safe
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size


try:
    from megatron.core.ssm.mamba_hybrid_layer_allocation import (
        get_hybrid_total_layer_count as _mcore_get_hybrid_total_layer_count,
    )
except ImportError:
    # TODO(yuya): remove fallback once MCore pin includes get_hybrid_total_layer_count
    _mcore_get_hybrid_total_layer_count = None

# MCore renamed `hybrid_override_pattern` → `hybrid_layer_pattern`. At the current pin
# MambaModel is a thin deprecation shim over models/hybrid/HybridModel whose signature is
# `(*args, mamba_stack_spec=None, **kwargs)`, so introspecting the SHIM finds neither name
# and a naive fallback would silently pass the DEPRECATED kwarg. Introspect the real class
# when it exists; fall back to the shim's signature only on pins that predate HybridModel.
try:
    from megatron.core.models.hybrid.hybrid_model import HybridModel as _MCoreHybridModel

    _MCORE_MAMBA_INIT_PARAMS = set(inspect.signature(_MCoreHybridModel.__init__).parameters)
except ImportError:  # old pins: MambaModel is the real class
    _MCORE_MAMBA_INIT_PARAMS = set(inspect.signature(MCoreMambaModel.__init__).parameters)
_HYBRID_LAYER_PATTERN_KWARG = (
    "hybrid_layer_pattern" if "hybrid_layer_pattern" in _MCORE_MAMBA_INIT_PARAMS else "hybrid_override_pattern"
)


logger = logging.getLogger(__name__)

_HYBRID_MAIN_PATTERN_SYMBOLS = frozenset({"M", "*", "-", "E", "|"})

# offload_modules names whose implementation lives inside TEGroupedMLP
# (megatron/core/transformer/moe/experts.py). The remaining names — attn_norm,
# qkv_linear, core_attn, attn_proj, mlp_norm — are handled by TransformerLayer and
# survive an experts swap.
MOE_INTERNAL_OFFLOAD_MODULES = frozenset({"expert_fc1", "moe_act", "fused_group_mlp"})


def _fallback_get_hybrid_total_layer_count(pattern: str) -> int:
    """Count main-decoder layers for older MCore branches.

    Older MCore revisions predate ``get_hybrid_total_layer_count`` and do not
    understand pipe-delimited fVPP layouts. Bridge still needs to derive
    ``num_layers`` correctly for both legacy and newer hybrid patterns.
    """

    main_pattern = pattern.split("/")[0]
    invalid_chars = sorted({char for char in main_pattern if char not in _HYBRID_MAIN_PATTERN_SYMBOLS})
    if invalid_chars:
        raise ValueError(
            f"In main pattern, '{invalid_chars[0]}' is not a valid layer symbol. "
            f"Valid symbols are: {_HYBRID_MAIN_PATTERN_SYMBOLS}"
        )
    return len(main_pattern.replace("|", ""))


def _get_hybrid_total_layer_count(pattern: str) -> int:
    if _mcore_get_hybrid_total_layer_count is not None:
        return _mcore_get_hybrid_total_layer_count(pattern)
    return _fallback_get_hybrid_total_layer_count(pattern)


def modelopt_mamba_stack_spec(config: "MambaModelProvider") -> ModuleSpec:
    """Mamba stack specification for quantization with ModelOpt.

    Uses Norm instead of TENorm and ColumnParallelLinear/RowParallelLinear
    instead of TE layers to enable proper quantizer insertion by ModelOpt.

    Args:
        config: Mamba configuration object

    Returns:
        ModuleSpec: Module specification for quantization-ready Mamba stack
    """
    return get_mamba_stack_modelopt_spec(
        local_core_attention=False,
        remap_te_layernorm=False,
    )


def transformer_engine_mamba_stack_spec() -> ModuleSpec:
    """Return the default Mamba stack spec with Transformer Engine layers.

    This is a named function (not a lambda) to allow proper serialization
    and reconstruction from checkpoints. Named functions can be imported
    via their module path, unlike lambdas.

    Returns:
        Default Mamba stack specification from megatron.core
    """
    return default_mamba_stack_spec


def get_default_mamba_stack_spec(config: "MambaModelProvider") -> ModuleSpec:
    """Determine the most appropriate Mamba stack specification based on configuration.

    Args:
        config: Mamba configuration object

    Returns:
        ModuleSpec: Appropriate module specification based on config
    """
    return transformer_engine_mamba_stack_spec()


@dataclass
class MambaModelProvider(TransformerConfig, ModelProviderMixin[MCoreMambaModel]):
    """Configuration and provider for Megatron Core Mamba models.

    This class extends TransformerConfig with Mamba-specific parameters and
    provides a method to instantiate configured Mamba models.
    """

    # Model configuration
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = False
    params_dtype: torch.dtype = torch.bfloat16
    fp16: bool = False
    bf16: bool = True
    num_layers: int = None
    mamba_num_groups: int = 8
    num_attention_heads: int = 1
    hybrid_attention_ratio: float = 0.0
    hybrid_mlp_ratio: float = 0.0
    hybrid_override_pattern: Optional[str] = None
    hybrid_layer_pattern: Optional[str] = None
    seq_length: int = 8192
    # Mamba with no attention has no need for position embeddings, so none is default
    position_embedding_type: Literal["learned_absolute", "rope", "none"] = "none"
    rotary_percent: float = 1.0
    rotary_base: int = 10000
    seq_len_interpolation_factor: Optional[float] = None
    apply_rope_fusion: bool = True
    make_vocab_size_divisible_by: int = 128
    gated_linear_unit: bool = False
    normalization: str = "RMSNorm"
    add_bias_linear: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    layernorm_epsilon: float = 1e-5
    attention_backend: AttnBackend = AttnBackend.flash
    deallocate_pipeline_outputs: bool = True
    bias_dropout_fusion: bool = True
    cross_entropy_loss_fusion: bool = True
    gradient_accumulation_fusion: bool = field(default_factory=fusions.can_enable_gradient_accumulation_fusion)
    mamba_stack_spec: Union[ModuleSpec, Callable[[], ModuleSpec], Callable[["MambaModelProvider"], ModuleSpec]] = (
        get_default_mamba_stack_spec
    )
    # Which MoE experts implementation the stack spec uses (hybrid MoE models only; inert
    # for pure-Mamba models). Lives HERE, not on NemotronHModelProvider: the NemotronH
    # bridge registers provider=MambaModelProvider, so this class is what training actually
    # instantiates — a field on the subclass is silently dropped by the YAML merge (found
    # the hard way; investigation doc §9.11).
    #   "te_grouped"     — upstream TEGroupedMLP, and the default here deliberately. Note
    #                      what that means in practice: only the two BENCHMARK quickstarts
    #                      (Super-120B, Nano-30B) opt into "torch_grouped". Ultra-550B and
    #                      pa_warm_start still run te_grouped — not because they are
    #                      unrelated models (they are the same Nemotron-H family) but
    #                      because they have not been benchmarked on that path yet.
    #   "torch_grouped"  — GroupedExperts via torch._grouped_mm, a real CUTLASS 3.x sm90
    #                      grouped kernel with full autograd. Measured −16.2% s/iter on the
    #                      64-GPU Super benchmark and −16.4% at 128 GPUs, both paired
    #                      same-nodelist A/Bs of this field alone; bitwise-identical to the
    #                      per-expert reference. This is what the configs select.
    #   "cublas_grouped" — GroupedExperts driving nv-grouped-gemm, which on sm_90 falls
    #                      through to a per-expert cuBLAS LOOP (its grouped kernel is
    #                      compile-time gated to sm_80). Kept so the A/B stays runnable and
    #                      for torch builds without _grouped_mm. Needs nv-grouped-gemm.
    #   "cutlass_grouped" — DEPRECATED alias of "cublas_grouped"; warns. The old name
    #                      claimed a CUTLASS kernel this path never reached.
    moe_experts_impl: str = "te_grouped"
    # Gradient routing (GRAM): when set, every MoE layer in the stack spec is swapped to
    # GRAMMoELayer carrying one gated auxiliary MLP of this ffn width (see
    # models/mamba/gram_layer.py). None (the default) leaves the spec untouched — the
    # no-GR code path is byte-identical to a build without this field. Lives HERE for the
    # same reason as moe_experts_impl: the NemotronH bridge registers
    # provider=MambaModelProvider, so a field on a subclass is silently dropped by the
    # YAML merge.
    gr_aux_ffn_hidden_size: Optional[int] = None
    vocab_size: Optional[int] = None
    should_pad_vocab: bool = False
    hf_model_id: Optional[str] = None
    _pg_collection: Optional[ProcessGroupCollection] = None

    # MTP
    mtp_num_layers: int = 0
    mtp_hybrid_override_pattern: Optional[str] = None
    keep_mtp_spec_in_bf16: bool = False

    """Optional HuggingFace model identifier associated with this provider."""

    # If True, restore the modelopt_state that contains quantization, sparsity, speculative decoding transformation state.
    restore_modelopt_state: bool = False

    def finalize(self) -> None:
        """Finalize the Mamba model provider.
        Calculates the number of layers from the hybrid_layer_pattern.
        Executes the deferred MCore post-init logic.
        """
        # Check if hybrid_override_pattern is specified and throw deprecation warning
        used_hybrid_override_pattern = False
        if self.hybrid_override_pattern is not None:
            assert self.hybrid_layer_pattern is None, (
                "hybrid_override_pattern and hybrid_layer_pattern cannot both be specified. "
                "hybrid_override_pattern is deprecated; use hybrid_layer_pattern instead."
            )
            if get_rank_safe() == 0:
                warnings.warn(
                    "hybrid_override_pattern is deprecated. Use hybrid_layer_pattern instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            self.hybrid_layer_pattern = self.hybrid_override_pattern
            self.hybrid_override_pattern = None
            used_hybrid_override_pattern = True

        # --- MTP pattern construction ---
        # Combine hybrid_layer_pattern (main decoder) with mtp_hybrid_override_pattern
        # into a single unified pattern that MCoreMambaModel can parse.
        # Format: "MAIN_PATTERN/MTP_BLOCK/MTP_BLOCK/..."
        # This must happen before num_layers derivation so the count reflects
        # only the main decoder layers (get_hybrid_total_layer_count strips MTP).
        if (
            self.hybrid_layer_pattern is not None
            and self.mtp_hybrid_override_pattern
            and self.mtp_num_layers is not None
        ):
            sep = Symbols.MTP_SEPARATOR
            main_pattern = self.hybrid_layer_pattern.split(sep)[0]
            # When mtp_use_repeated_layer=True, the shared MTP layer always exists
            # in the model and mtp_num_layers controls forward pass repetitions.
            # Include the pattern at least once so the MTP block (and its weights)
            # are created even when mtp_num_layers=0.
            if self.mtp_use_repeated_layer:
                num_pattern_copies = max(1, self.mtp_num_layers)
            else:
                num_pattern_copies = self.mtp_num_layers
            self.hybrid_layer_pattern = (
                main_pattern + sep + sep.join([self.mtp_hybrid_override_pattern] * num_pattern_copies)
            )

            # Validate mtp_num_layers against the constructed pattern
            if sep in self.hybrid_layer_pattern:
                parsed = parse_hybrid_pattern(self.hybrid_layer_pattern)
                if parsed.mtp_pattern and parsed.mtp_num_depths > 0:
                    inferred_mtp_num_layers = parsed.mtp_num_depths
                    if self.mtp_num_layers is None:
                        self.mtp_num_layers = inferred_mtp_num_layers
                    elif self.mtp_use_repeated_layer:
                        # With repeated layers, pattern count reflects architecture
                        # (always 1 shared layer) while mtp_num_layers controls
                        # forward pass repetitions. They are intentionally decoupled.
                        pass
                    elif self.mtp_num_layers != inferred_mtp_num_layers:
                        logger.warning(
                            f"mtp_num_layers ({self.mtp_num_layers}) conflicts with "
                            f"MTP depth count ({inferred_mtp_num_layers}) in pattern "
                            f"'{self.hybrid_layer_pattern}'. "
                            f"Using the inferred value ({inferred_mtp_num_layers})."
                        )
                        self.mtp_num_layers = inferred_mtp_num_layers

        # Check if hybrid_layer_pattern is specified and derive num_layers from pattern
        if self.hybrid_layer_pattern is not None:
            # Derive num_layers from pattern
            num_layers_in_pattern = _get_hybrid_total_layer_count(self.hybrid_layer_pattern)
            if self.num_layers is not None:
                if used_hybrid_override_pattern:
                    assert self.num_layers == num_layers_in_pattern, (
                        f"num_layers ({self.num_layers}) does not match the number of layers "
                        f"derived from hybrid_override_pattern ({num_layers_in_pattern}). "
                        f"Please correct num_layers or the pattern."
                    )
                else:
                    assert self.num_layers == num_layers_in_pattern, (
                        f"num_layers ({self.num_layers}) does not match the number of layers "
                        f"derived from hybrid_layer_pattern ({num_layers_in_pattern}). "
                        f"Please correct num_layers or the pattern."
                    )
            self.num_layers = num_layers_in_pattern

        super().finalize()

    def _apply_moe_experts_impl(self) -> None:
        """Wrap mamba_stack_spec per moe_experts_impl.

        Runs at provide() time — YAML `model:` overrides merge onto the provider instance
        AFTER construction, so a __post_init__ hook would only see the field default.
        """
        from megatron.bridge.models.mamba.grouped_experts import (
            DEPRECATED_BACKEND_ALIASES,
            GEMM_BACKENDS,
            swap_moe_experts_to_grouped,
        )

        if self.moe_experts_impl == "te_grouped":
            return

        impl = self.moe_experts_impl
        backend = DEPRECATED_BACKEND_ALIASES.get(impl, impl)
        if backend != impl:
            # Accepted, not silently remapped: say so loudly enough that configs get fixed.
            logger.warning(
                "moe_experts_impl=%r is DEPRECATED and resolves to %r. The old name claimed a "
                "CUTLASS kernel that is never reached on sm_90 (compile-time gated to sm_80); "
                "update your config to %r, or to 'torch_grouped' which is faster on this "
                "hardware. See consultant-training-stack-review.md §C1g, preserved under "
                "/projects/a5k/public/logs/infr71_wave2/docs/.",
                impl,
                backend,
                backend,
            )
        if backend not in GEMM_BACKENDS:
            raise ValueError(
                f"Unknown moe_experts_impl {impl!r}; expected 'te_grouped', "
                f"{', '.join(repr(b) for b in GEMM_BACKENDS)}, or the deprecated "
                f"{', '.join(repr(a) for a in DEPRECATED_BACKEND_ALIASES)}."
            )
        if getattr(self, "_grouped_spec_applied", False):
            return
        # The swap only rewrites the main stack's MoE spec; an MTP block carries its
        # own nested MoE spec that would silently stay on the TE path. Refuse the
        # half-swapped combination rather than train it.
        if getattr(self, "mtp_num_layers", 0):
            raise NotImplementedError(
                f"moe_experts_impl={impl!r} does not swap the MTP block's nested MoE spec; "
                "use te_grouped when mtp_num_layers > 0."
            )
        # MoE-internal activation offload is implemented inside TEGroupedMLP only, and mcore
        # validates offload_modules against a static name list rather than the built model —
        # so swapping the experts away from TE leaves these names selecting nothing and the
        # run silently offloads zero bytes.
        moe_internal_offload = MOE_INTERNAL_OFFLOAD_MODULES & set(getattr(self, "offload_modules", None) or [])
        if getattr(self, "fine_grained_activation_offloading", False) and moe_internal_offload:
            raise ValueError(
                f"offload_modules {sorted(moe_internal_offload)} are implemented only inside "
                f"TEGroupedMLP, which moe_experts_impl={impl!r} replaces — they would offload "
                "nothing. Use te_grouped to keep the offload, or drop these names from "
                "offload_modules (attention/norm offload is unaffected)."
            )

        inner = self.mamba_stack_spec

        def _grouped_resolved_stack_spec(cfg=None):
            if callable(inner):
                try:
                    spec = inner(cfg)
                except TypeError:
                    spec = inner()
            else:
                spec = inner
            return swap_moe_experts_to_grouped(spec, gemm_backend=backend)

        self.mamba_stack_spec = _grouped_resolved_stack_spec
        self._grouped_spec_applied = True
        logger.info("moe_experts_impl=%s: MoE experts swapped to GroupedExperts(%s)", impl, backend)

    def _apply_gradient_routing(self, stack_spec: ModuleSpec) -> ModuleSpec:
        """Return the resolved stack spec with the GRAM MoE-layer swap when routing is on.

        Runs at provide() time for the same reason as _apply_moe_experts_impl: YAML
        overrides merge onto the instance after construction. Unlike that swap, this one
        deliberately never assigns to ``self.mamba_stack_spec`` — whatever sits there is
        serialized by qualname into the checkpoint's run_config, and a ``<locals>``
        closure breaks ``AutoBridge.from_auto_config`` at export time. The swap is
        re-derived from ``gr_aux_ffn_hidden_size`` (which does serialize) on every
        provide() call instead; ``swap_moe_layer_to_gram`` refuses a double-apply, and
        each call here starts from a freshly resolved spec, so repeated provide() calls
        (VPP chunks) are safe by construction.
        """
        if self.gr_aux_ffn_hidden_size is None:
            return stack_spec
        # The swap rewrites only the main stack's MoE spec; an MTP block carries its own
        # nested MoE spec that would silently stay un-swapped — and gradient isolation
        # semantics for an MTP head are undefined here. Refuse rather than half-apply.
        if getattr(self, "mtp_num_layers", 0):
            raise NotImplementedError(
                "gr_aux_ffn_hidden_size does not swap the MTP block's nested MoE spec; "
                "gradient routing requires mtp_num_layers == 0."
            )
        # Latent MoE feeds experts at moe_latent_size width while the shared expert (and
        # the aux module mirroring it) sees full hidden width — untested interaction with
        # the export-merge contract. Refuse until measured.
        if getattr(self, "moe_latent_size", None):
            raise NotImplementedError("gradient routing is untested with moe_latent_size; unset one of them.")

        from megatron.bridge.models.mamba.gram_layer import swap_moe_layer_to_gram

        swapped = swap_moe_layer_to_gram(stack_spec, aux_ffn_hidden_size=self.gr_aux_ffn_hidden_size)
        logger.info("gradient routing: MoE layers swapped to GRAMMoELayer(aux_ffn=%d)", self.gr_aux_ffn_hidden_size)
        return swapped

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreMambaModel:
        """Configure and instantiate a Megatron Core Mamba model based on this configuration.

        Args:
            pre_process: Whether to include pre-processing in the model, defaults to first pipeline stage
            post_process: Whether to include post-processing in the model, defaults to last pipeline stage
            vp_stage: Virtual pipeline stage

        Returns:
            MCoreMambaModel: Configured Megatron Core Mamba model instance
        """
        self._apply_moe_experts_impl()
        mamba_stack_spec = self.mamba_stack_spec
        if not isinstance(mamba_stack_spec, ModuleSpec):
            # Check if the function accepts config parameter
            import inspect

            if len(inspect.signature(mamba_stack_spec).parameters) > 0:
                mamba_stack_spec = mamba_stack_spec(self)
            else:
                mamba_stack_spec = mamba_stack_spec()
        mamba_stack_spec = self._apply_gradient_routing(mamba_stack_spec)

        # VPP gate removed (INFR-68): the assert here dated 2025-08-13 and cited a
        # missing MCore MambaModel vp_stage API; the current 3rdparty pin
        # (3758b54b2, 2026-03) accepts vp_stage end-to-end (hybrid fVPP via '|'
        # pipeline-segment separators in hybrid_override_pattern +
        # select_pipeline_segment). Requires a piped pattern when
        # virtual_pipeline_model_parallel_size is set — the mcore allocator
        # errors explicitly otherwise.

        assert self.vocab_size is not None, "vocab_size must be configured before calling provide()"
        if self.should_pad_vocab:
            padded_vocab_size = calculate_padded_vocab_size(
                self.vocab_size, self.make_vocab_size_divisible_by, self.tensor_model_parallel_size
            )
        else:
            padded_vocab_size = self.vocab_size

        return MCoreMambaModel(
            self,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=padded_vocab_size,
            max_sequence_length=self.seq_length,
            **{_HYBRID_LAYER_PATTERN_KWARG: self.hybrid_layer_pattern},
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            # `is not None` (not `or`): under VPP, get_model passes explicit
            # False for interior virtual chunks — `or` would clobber it back to
            # the pp-stage answer and re-attach embeddings/loss to every chunk
            # on the boundary ranks.
            pre_process=pre_process if pre_process is not None else is_pp_first_stage(self._pg_collection.pp),
            post_process=post_process if post_process is not None else is_pp_last_stage(self._pg_collection.pp),
            vp_stage=vp_stage,
            pg_collection=self._pg_collection,
        )

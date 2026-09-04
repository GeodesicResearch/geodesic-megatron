# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Provider, layer spec and modules for AI2's OLMo-3 dense causal LMs.

OLMo-3 is OLMo-2 plus two things, and both of them are invisible to the weight
mapping -- they live entirely in the config and the forward pass:

1. **Interleaved sliding-window attention.** 3 of every 4 layers attend within a
   ``sliding_window`` of 4096; every 4th layer attends fully. Expressed with
   Megatron-Core's native :attr:`window_size` / :attr:`window_attn_skip_freq`.
2. **RoPE scaling on full-attention layers only.** The sliding layers never see
   more than ``sliding_window`` tokens, so they need no extension and use plain
   RoPE; only the full-attention layers get YaRN. See :class:`Olmo3RotaryEmbedding`.

Inherited from OLMo-2, and equally not visible in the weights:

3. **Pure post-norm.** ``x = x + post_norm(sublayer(x))`` -- there is no
   normalization on the *input* of either sub-block.
4. **Full-width QK-norm.** The RMSNorm spans the whole Q (and whole K)
   projection, not each head independently.

Reference implementations this file is written against, in order of authority:
``vllm/model_executor/models/olmo2.py`` (our sampler) and
``transformers`` v4.57.1 ``modular_olmo3.py``. Note that transformers **5.2.0**
regressed (2) -- it builds a single shared rotary embedding and applies YaRN to
every layer. Do not use 5.2.0 as the equivalence reference.
"""

import copy
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.attention import SelfAttention as MCoreSelfAttention
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.utils import is_layer_window_attention

from megatron.bridge.models.gpt_provider import GPTModelProvider


try:
    import transformer_engine  # type: ignore  # noqa: F401

    HAVE_TE = True
    from megatron.core.extensions.transformer_engine import (
        SplitAlongDim,
        TEColumnParallelLinear,
        TEDotProductAttention,
        TENorm,
        TERowParallelLinear,
    )
except ImportError:
    HAVE_TE = False
    SplitAlongDim = None
    TEColumnParallelLinear = TEDotProductAttention = TENorm = TERowParallelLinear = None

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add


class TERowParallelLinearPostNorm(TERowParallelLinear):
    """``TERowParallelLinear`` with a trailing RMSNorm, for OLMo-3's post-norm blocks.

    Structurally this is Gemma-2's ``TERowParallelLinearLayerNorm``
    (``gemma/gemma2_provider.py:255``) -- the norm runs on the projection output,
    before ``TransformerLayer`` adds the residual, giving
    ``x + post_norm(sublayer(x))``.

    It is *not* reused directly because that class builds its norm as
    ``TENorm(config, output_size)``, leaving ``eps`` at TransformerEngine's ``1e-5``
    default and silently ignoring ``config.layernorm_epsilon``. OLMo-3 specifies
    ``1e-6``. The discrepancy is invisible whenever activations are large relative
    to eps, which is why it survives casual testing -- but it is a real difference
    and it moved this model's logits. (Gemma-2/3 declare 1e-6 too, so they look
    subject to the same bug; worth raising upstream.)

    .. warning::
        The norm is a **child of the projection**, so its parameter path is
        ``...linear_proj.post_layernorm`` / ``...linear_fc2.post_layernorm``.
        PEFT target globs must therefore match only names that *end* at the
        projection: ``'*linear_proj'``, not ``'*linear_proj*'``.
        ``megatron.bridge.peft.utils.wildcard_match`` anchors the pattern
        (``"^" + pattern.replace("*", "(.*)") + "$"``), so the trailing-``*``
        form also selects the nested norm, and LoRA then tries to wrap an
        RMSNorm as a linear -- ``AttributeError: 'RMSNorm' object has no
        attribute 'config'``. ``exclude_modules`` is not an escape hatch: LoRA
        asserts it is empty whenever ``target_modules`` is set.
    """

    def __init__(self, input_size: int, output_size: int, *, config: TransformerConfig, **kwargs):
        super().__init__(input_size, output_size, config=config, **kwargs)
        self.post_layernorm = TENorm(config, output_size, eps=config.layernorm_epsilon)

    def forward(self, x: torch.Tensor):
        """Project, then normalize the output."""
        output, bias = super().forward(x)
        return self.post_layernorm(output), bias


class Olmo3RotaryEmbedding(RotaryEmbedding):
    """RoPE for OLMo-3: YaRN on full-attention layers, plain RoPE on sliding ones.

    OLMo-3 extends context by applying YaRN to the full-attention layers only --
    the sliding-window layers never attend beyond ``sliding_window`` positions,
    so extending their frequencies would be meaningless. ``GPTModel`` builds a
    single rotary module for the whole model, so (following ``Gemma3RotaryEmbedding``)
    we compute both and return them stacked; :class:`Olmo3SelfAttention` selects
    the slice its layer needs.

    ``self`` is the *sliding* (unscaled) rope; ``self.rope_full`` is the YaRN one.

    Note the returned tensor holds only the rotation *angles*. YaRN's attention
    factor ("mscale") multiplies cos/sin rather than the angles, so it cannot be
    folded in here -- Megatron applies it inside
    ``apply_rotary_pos_emb(..., mscale=...)`` from the *config*. That is why
    :class:`Olmo3SelfAttention` also gives sliding layers a config with YaRN
    disabled; see its ``__init__``.
    """

    def __init__(
        self,
        kv_channels: int,
        rotary_percent: float = 1.0,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: Optional[float] = None,
        rotary_base: float = 500000.0,
        use_cpu_initialization: bool = False,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
        *,
        yarn_scaling_factor: float = 8.0,
        yarn_original_max_position_embeddings: int = 8192,
        yarn_beta_fast: float = 32.0,
        yarn_beta_slow: float = 1.0,
        yarn_mscale: float = 1.0,
        yarn_mscale_all_dim: float = 0.0,
        yarn_correction_range_round_to_int: bool = True,
    ) -> None:
        # Sliding-window layers: plain RoPE, no scaling of any kind.
        super().__init__(
            kv_channels=kv_channels,
            rotary_percent=rotary_percent,
            rotary_interleaved=rotary_interleaved,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            rotary_base=rotary_base,
            rope_scaling=False,
            use_cpu_initialization=use_cpu_initialization,
            cp_group=cp_group,
        )
        # Full-attention layers: YaRN.
        self.rope_full = YarnRotaryEmbedding(
            kv_channels=kv_channels,
            rotary_percent=rotary_percent,
            rotary_interleaved=rotary_interleaved,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            rotary_base=rotary_base,
            use_cpu_initialization=use_cpu_initialization,
            scaling_factor=yarn_scaling_factor,
            original_max_position_embeddings=yarn_original_max_position_embeddings,
            beta_fast=yarn_beta_fast,
            beta_slow=yarn_beta_slow,
            mscale=yarn_mscale,
            mscale_all_dim=yarn_mscale_all_dim,
            correction_range_round_to_int=yarn_correction_range_round_to_int,
            cp_group=cp_group,
        )

    def forward(
        self,
        max_seq_len: int,
        offset: int = 0,
        packed_seq: bool = False,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> torch.Tensor:
        """Return ``stack([rope_sliding, rope_full])`` along a new leading dim."""
        # ProcessGroup is unhashable, so only the common (cp_group=None) path can
        # be cached -- same split Gemma-3 uses.
        if cp_group is not None:
            return self._compute(max_seq_len, offset, packed_seq, cp_group)
        return self._forward_cached(max_seq_len, offset, packed_seq)

    def _compute(
        self,
        max_seq_len: int,
        offset: int,
        packed_seq: bool,
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> torch.Tensor:
        rope_sliding = super().forward(max_seq_len, offset, packed_seq, cp_group)
        # YarnRotaryEmbedding.forward returns (emb, mscale); mscale is applied
        # from config inside attention, so it is deliberately dropped here.
        rope_full, _ = self.rope_full.forward(max_seq_len, offset, packed_seq, cp_group)
        return torch.stack([rope_sliding, rope_full], dim=0)

    @lru_cache(maxsize=32)
    def _forward_cached(
        self, max_seq_len: int, offset: int = 0, packed_seq: bool = False
    ) -> torch.Tensor:
        """Cached path. Both base rope classes cache their own ``forward``
        (``rotary_pos_embedding.py:177``, ``yarn_rotary_pos_embedding.py:159``);
        overriding ``forward`` without a cache would re-stack on every
        microbatch -- ~25 per step here."""
        return self._compute(max_seq_len, offset, packed_seq, None)


class Olmo3SelfAttention(MCoreSelfAttention):
    """Self-attention for OLMo-3: full-width QK-norm + per-layer RoPE selection.

    Two departures from ``MCoreSelfAttention``:

    * **Full-width QK-norm.** Megatron normalizes each head independently
      (``hidden_size=hidden_size_per_attention_head``). OLMo-2/3 instead apply one
      RMSNorm across the whole Q projection and one across the whole K projection,
      *before* the tensors are split into heads. Because OLMo-3 is GQA, those two
      widths differ: ``num_attention_heads * head_dim`` for Q and
      ``num_query_groups * head_dim`` for K. (``OLMoESelfAttention`` uses
      ``num_attention_heads`` for both, which is correct only for the MHA
      OLMoE checkpoints and would be wrong here.)

    * **Per-layer RoPE.** Picks the sliding or full slice of the stacked rope from
      :class:`Olmo3RotaryEmbedding`, and -- for sliding layers -- runs against a
      config copy with YaRN switched off, so that
      ``_yarn_get_concentration_factor_from_config`` yields 1.0 and no attention
      factor is applied. Full-attention layers keep the real factor
      (1.2079... for scaling factor 8).
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str = None,
        pg_collection: ProcessGroupCollection = None,
        **kwargs,
    ):
        # Single source of truth for "is this a sliding layer?" -- the same helper
        # Megatron-Core itself uses to decide whether to pass window_size to TE.
        is_sliding = is_layer_window_attention(
            config.window_size, config.window_attn_skip_freq, layer_number
        )
        if is_sliding:
            # YaRN's attention factor reaches attention via the config, not via the
            # rope tensor, so a sliding layer needs its own config with YaRN off.
            config = copy.deepcopy(config)
            config.yarn_rotary_scaling_factor = None

        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )
        self.is_sliding = is_sliding

        # OLMo-3 normalizes the full projection, not each head. Under tensor
        # parallelism each rank holds only a slice of the heads, so a full-width
        # norm would need an all-gather (vLLM does exactly that). We do not
        # implement it -- refuse loudly rather than normalize the wrong width.
        tp_size = self.config.tensor_model_parallel_size
        if tp_size is not None and tp_size > 1:
            raise NotImplementedError(
                "OLMo-3's QK-norm spans the whole Q/K projection, which under "
                f"tensor_model_parallel_size={tp_size} would require an all-gather "
                "across the head dimension. Use tensor_model_parallel_size=1 "
                "(pipeline parallelism is unaffected)."
            )

        self.q_layernorm = build_module(
            submodules.q_layernorm,
            hidden_size=self.hidden_size_per_attention_head * self.config.num_attention_heads,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )
        self.k_layernorm = build_module(
            submodules.k_layernorm,
            # GQA: K is only num_query_groups heads wide.
            hidden_size=self.hidden_size_per_attention_head * self.config.num_query_groups,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

    def forward(self, hidden_states, *args, rotary_pos_emb=None, **kwargs):
        """Select this layer's rope from the stacked (sliding, full) pair.

        Deliberately a ``*args``/``**kwargs`` passthrough: ``TransformerLayer``
        passes everything but ``hidden_states`` by keyword, and enumerating the
        signature here would silently drop any argument Megatron-Core adds later.
        """
        if (
            isinstance(rotary_pos_emb, torch.Tensor)
            and rotary_pos_emb.ndim >= 1
            and rotary_pos_emb.size(0) == 2
        ):
            rotary_pos_emb = rotary_pos_emb[0 if self.is_sliding else 1]
        return super().forward(hidden_states, *args, rotary_pos_emb=rotary_pos_emb, **kwargs)

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None, **kwargs):
        """Derive q/k/v, applying QK-norm across the full projection width."""
        # [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn]
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        if SplitAlongDim is not None:
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
        else:
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        # Flatten heads back out, normalize across the whole projection, re-split.
        query = query.reshape(query.size(0), query.size(1), -1)
        key = key.reshape(key.size(0), key.size(1), -1)
        query = self.q_layernorm(query)
        key = self.k_layernorm(key)

        if self.config.test_mode:
            self.run_realtime_tests()

        query = query.view(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        key = key.view(key.size(0), key.size(1), -1, self.hidden_size_per_attention_head)
        value = value.reshape(value.size(0), value.size(1), -1, self.hidden_size_per_attention_head)

        return query, key, value


def olmo3_layer_spec(config: "GPTModelProvider") -> ModuleSpec:
    """Layer spec for OLMo-3: pure post-norm, custom attention.

    ``input_layernorm`` and ``pre_mlp_layernorm`` are left at their
    ``TransformerLayerSubmodules`` default of ``IdentityOp`` because OLMo-3 has no
    pre-normalization at all, and the two output projections are wrapped in
    ``TERowParallelLinearPostNorm`` so each sub-block is normalized before the
    residual add -- i.e. ``x + post_norm(sublayer(x))``.

    This is why ``linear_qkv``/``linear_fc1`` are plain ``TEColumnParallelLinear``
    rather than the usual ``TELayerNormColumnParallelLinear``: there is no input
    norm to fuse into them.
    """
    del config  # spec does not depend on runtime config; signature kept for symmetry
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            # input_layernorm: IdentityOp (default) -- OLMo-3 has no pre-attn norm
            self_attention=ModuleSpec(
                module=Olmo3SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    q_layernorm=TENorm,
                    k_layernorm=TENorm,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinearPostNorm,  # post-attention RMSNorm
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            # pre_mlp_layernorm: IdentityOp (default) -- OLMo-3 has no pre-MLP norm
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    linear_fc2=TERowParallelLinearPostNorm,  # post-feedforward RMSNorm
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )


@dataclass
class Olmo3ModelProvider(GPTModelProvider):
    """Base provider for OLMo-3 dense causal LMs.

    Defaults reflect ``allenai/Olmo-3-*`` configs. ``position_embedding_type`` is
    deliberately ``"rope"``, not ``"yarn"``: ``GPTModel``'s YaRN branch builds one
    scaled rope for every layer, which is precisely what OLMo-3 does not do.
    :meth:`provide` swaps in :class:`Olmo3RotaryEmbedding` instead.
    """

    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTModelProvider"], ModuleSpec]] = (
        olmo3_layer_spec
    )
    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    qk_layernorm: bool = True
    layernorm_epsilon: float = 1e-6
    rotary_base: float = 500000.0
    rotary_percent: float = 1.0
    init_method_std: float = 0.02
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = False
    persist_layer_norm: bool = True
    bf16: bool = True
    params_dtype: torch.dtype = torch.bfloat16
    autocast_dtype: torch.dtype = torch.bfloat16

    # Interleaved SWA. Both are set by the bridge from the HF config; the defaults
    # here describe the published 3-sliding : 1-full pattern.
    window_size: Optional[tuple] = None
    window_attn_skip_freq: Optional[Union[int, list]] = None

    # YaRN, applied to full-attention layers only.
    yarn_rotary_scaling_factor: Optional[float] = None
    yarn_original_max_position_embeddings: int = 8192
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_mscale: float = 1.0
    yarn_mscale_all_dim: float = 0.0

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """Build the GPT model, then replace its rope with the OLMo-3 dual rope."""
        model = super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        if getattr(model, "rotary_pos_emb", None) is not None:
            model.rotary_pos_emb = Olmo3RotaryEmbedding(
                kv_channels=self.kv_channels,
                rotary_percent=self.rotary_percent,
                rotary_interleaved=self.rotary_interleaved,
                seq_len_interpolation_factor=self.seq_len_interpolation_factor,
                rotary_base=self.rotary_base,
                use_cpu_initialization=self.use_cpu_initialization,
                yarn_scaling_factor=self.yarn_rotary_scaling_factor,
                yarn_original_max_position_embeddings=self.yarn_original_max_position_embeddings,
                yarn_beta_fast=self.yarn_beta_fast,
                yarn_beta_slow=self.yarn_beta_slow,
                yarn_mscale=self.yarn_mscale,
                yarn_mscale_all_dim=self.yarn_mscale_all_dim,
            )
        return model


@dataclass
class Olmo3ModelProvider32B(Olmo3ModelProvider):
    """OLMo-3 32B (``allenai/Olmo-3-32B-Think``): 64 layers, hidden 5120, GQA 40/8."""

    num_layers: int = 64
    hidden_size: int = 5120
    num_attention_heads: int = 40
    num_query_groups: int = 8
    ffn_hidden_size: int = 27648
    kv_channels: int = 128
    seq_length: int = 65536
    vocab_size: int = 100352

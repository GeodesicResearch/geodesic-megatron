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
"""The SFT recipe of AI2's OLMo-3 dense models (7B and 32B), sized from the checkpoint's own ``config.json``.

One function serves both sizes: the architecture comes from the Hugging Face config of the base model that the run
warm-starts from, through the OLMo-3 bridge. The recipe sets what the family needs in order to train at all. Every
choice of a campaign (learning rate and schedule, batch, parallel layout, recompute, precision posture, data and
checkpoint paths) belongs to the run's YAML, which restates it so that a test can pin it.

What the family needs, and why:
    * ``tensor_model_parallel_size = 1``: the full-width QK-norm refuses tensor parallelism (``Olmo3SelfAttention``).
    * ``apply_rope_fusion = False``: the fused THD rope drops the YaRN attention scale on the full-attention layers
      without a warning.
    * ``calculate_per_token_loss = True``: context parallelism needs the per-token loss normalisation, and packed
      sequences of unequal token counts need it for the loss to weigh every token alike.
    * ``cp_comm_type = "a2a"``: the only context-parallel exchange that accepts packed (THD) sequences together with
      a sliding window (Transformer Engine asserts this). It has no effect at context parallel size 1.
    * ``pad_seq_to_mult = 8``: packed sequences are padded to a multiple of ``2 x CP`` (covers CP up to 4); the padded
      ``cu_seqlens`` are what Transformer Engine needs for THD with context parallelism.
"""

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _sft_common
from megatron.bridge.training.config import ConfigContainer


#: The training sequence length of the OLMo-3 Think SFT (AI2's recipe) and of the packed data built for it.
OLMO3_SFT_SEQ_LENGTH: int = 32768
#: Pack padding multiple: ``2 x CP`` for CP up to 4.
OLMO3_PAD_SEQ_TO_MULT: int = 8


def olmo3_sft_config(hf_path: str) -> ConfigContainer:
    """Return the SFT config of an OLMo-3 model whose Hugging Face config lives at ``hf_path``.

    Args:
        hf_path: A directory (or Hub id) holding the base model's ``config.json`` and tokenizer files. The model is
            built from the config without loading weights; the warm start is the YAML's ``checkpoint`` section.

    Returns:
        The ConfigContainer with the family's settings. Learning rate, batch, layout and the rest belong to the run's YAML.
    """
    cfg = _sft_common()

    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.tokenizer.tokenizer_model = hf_path

    cfg.model.seq_length = OLMO3_SFT_SEQ_LENGTH
    cfg.dataset.seq_length = OLMO3_SFT_SEQ_LENGTH
    cfg.dataset.packed_sequence_specs.packed_sequence_size = OLMO3_SFT_SEQ_LENGTH
    cfg.dataset.packed_sequence_specs.pad_seq_to_mult = OLMO3_PAD_SEQ_TO_MULT

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.cp_comm_type = "a2a"

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.apply_rope_fusion = False
    cfg.model.calculate_per_token_loss = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.cuda_graph_impl = "none"

    # The import of an HF base and this recipe's provider use the same key names, so a missing or an unexpected key of
    # the warm start is a bug: refuse it. With "log_all" a tensor missing from the checkpoint keeps its random initial
    # value and the run trains on.
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.dist_ckpt_strictness = "raise_all"

    return cfg

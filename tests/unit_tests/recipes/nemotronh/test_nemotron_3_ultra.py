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

"""
Unit tests for Nemotron 3 Ultra recipe configuration builders.

Ultra (550B-A55B) is a scaled Nemotron 3 Super: the same NemotronH hybrid
Latent-MoE + MTP family (512 routed experts, top-22), with the model config
derived from the HF id ``nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16`` via
AutoBridge. These tests cover:

- Pretrain / SFT / PEFT parallelism defaults (mirror the Super recipe; the real
  multi-node layout for 550B lives in the YAML configs under ``configs/``).
- MoE shape derived from the HF config (experts / top-k / latent / FFN sizes).
- MTP (multi-token prediction) settings.
- The precision-aware optimizer storing Adam 1st/2nd moments in BF16 — enabled
  by default for Ultra (effectively mandatory at 550B; see INFR-41).
- Pure BF16 mixed precision (no FP8/FP4 — MoE routing crashes on Isambard).

Note: like the Super recipe, ``tokenizer.tokenizer_model`` is intentionally left
unset here; it is supplied at runtime via YAML/CLI and enforced by
``pipeline_training_run.py``.
"""

import pytest
import torch

from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider
from megatron.bridge.recipes.nemotronh.nemotron_3_ultra import (
    nemotron_3_ultra_peft_config,
    nemotron_3_ultra_pretrain_config,
    nemotron_3_ultra_sft_config,
)
from megatron.bridge.training.config import ConfigContainer


ALL_ULTRA_RECIPES = [
    nemotron_3_ultra_pretrain_config,
    nemotron_3_ultra_sft_config,
    nemotron_3_ultra_peft_config,
]


@pytest.mark.unit
class TestNemotron3UltraPretrain:
    """Test cases for the Nemotron 3 Ultra pretrain recipe."""

    def test_pretrain_config_defaults(self):
        config = nemotron_3_ultra_pretrain_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, MambaModelProvider)

        # Parallelism (mirrors Super recipe defaults; real multi-node layout is in the YAML configs)
        assert config.model.tensor_model_parallel_size == 4
        assert config.model.pipeline_model_parallel_size == 1
        assert config.model.expert_model_parallel_size == 8
        assert config.model.expert_tensor_parallel_size == 1
        assert config.model.sequence_parallel is True

        # Training
        assert config.train.train_iters == 39735
        assert config.train.global_batch_size == 3072
        assert config.train.micro_batch_size == 1
        assert config.dataset.seq_length == 8192

        # Pure BF16 — no FP8/FP4 for MoE on Isambard
        assert config.mixed_precision == "bf16_mixed"

        # Optimizer hyperparameters
        assert config.optimizer.lr == 4.5e-4
        assert config.optimizer.min_lr == 4.5e-6
        assert config.scheduler.lr_warmup_iters == 333

    def test_pretrain_mtp_settings(self):
        config = nemotron_3_ultra_pretrain_config()

        assert config.model.mtp_num_layers == 2
        assert config.model.mtp_hybrid_override_pattern == "*E"
        assert config.model.mtp_use_repeated_layer is True
        assert config.model.keep_mtp_spec_in_bf16 is True
        assert config.model.calculate_per_token_loss is True
        assert config.model.mtp_loss_scaling_factor == 0.3


@pytest.mark.unit
class TestNemotron3UltraSft:
    """Test cases for the Nemotron 3 Ultra SFT recipe."""

    def test_sft_config_defaults(self):
        config = nemotron_3_ultra_sft_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, MambaModelProvider)

        # Full-SFT parallelism
        assert config.model.tensor_model_parallel_size == 1
        assert config.model.pipeline_model_parallel_size == 1
        assert config.model.expert_model_parallel_size == 8
        assert config.model.sequence_parallel is True

        # No PEFT config for full SFT
        assert config.peft is None

        # Full SFT uses a lower LR
        assert config.optimizer.lr == 5e-6
        assert config.mixed_precision == "bf16_mixed"


@pytest.mark.unit
class TestNemotron3UltraPeft:
    """Test cases for the Nemotron 3 Ultra PEFT recipe."""

    def test_peft_config_default_lora(self):
        config = nemotron_3_ultra_peft_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, MambaModelProvider)

        # LoRA parallelism (EP=1)
        assert config.model.tensor_model_parallel_size == 1
        assert config.model.pipeline_model_parallel_size == 1
        assert config.model.expert_model_parallel_size == 1

        # PEFT config present, with Mamba-specific target modules
        assert config.peft is not None
        assert config.optimizer.lr == 1e-4
        assert config.peft.target_modules == [
            "linear_qkv",
            "linear_proj",
            "linear_fc1",
            "linear_fc2",
            "in_proj",
            "out_proj",
        ]

    def test_peft_config_dora(self):
        config = nemotron_3_ultra_peft_config(peft_scheme="dora")
        assert config.peft is not None


@pytest.mark.unit
class TestNemotron3UltraCommon:
    """Test cases common to all Nemotron 3 Ultra recipes."""

    @pytest.mark.parametrize("recipe_fn", ALL_ULTRA_RECIPES)
    def test_config_container_structure(self, recipe_fn):
        config = recipe_fn()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, MambaModelProvider)

        # Required sections exist
        assert config.train is not None
        assert config.optimizer is not None
        assert config.scheduler is not None
        assert config.dataset is not None
        assert config.logger is not None
        assert config.tokenizer is not None
        assert config.checkpoint is not None
        assert config.ddp is not None
        assert config.mixed_precision is not None

        # HF tokenizer; tokenizer_model is supplied at runtime via YAML (see module docstring)
        assert config.tokenizer.tokenizer_type == "HuggingFaceTokenizer"

    @pytest.mark.parametrize("recipe_fn", ALL_ULTRA_RECIPES)
    def test_ddp_configuration(self, recipe_fn):
        config = recipe_fn()

        assert config.ddp.check_for_nan_in_grad is True
        assert config.ddp.overlap_grad_reduce is True
        assert config.ddp.overlap_param_gather is True
        assert config.ddp.use_distributed_optimizer is True

    @pytest.mark.parametrize("recipe_fn", ALL_ULTRA_RECIPES)
    def test_moe_model_configuration(self, recipe_fn):
        """MoE shape derived from the Ultra HF config by AutoBridge."""
        config = recipe_fn()

        assert config.model.num_moe_experts == 512
        assert config.model.moe_router_topk == 22
        assert config.model.moe_router_topk_scaling_factor == 5.0
        assert config.model.moe_latent_size == 2048
        assert config.model.moe_ffn_hidden_size == 5120
        assert config.model.moe_shared_expert_intermediate_size == 10240

    @pytest.mark.parametrize("recipe_fn", [nemotron_3_ultra_pretrain_config, nemotron_3_ultra_sft_config])
    def test_precision_aware_optimizer_bf16_moments(self, recipe_fn):
        """Ultra stores Adam 1st/2nd moments in BF16 (mandatory at 550B; see INFR-41)."""
        config = recipe_fn()

        assert config.optimizer.use_precision_aware_optimizer is True
        assert config.optimizer.exp_avg_dtype == torch.bfloat16
        assert config.optimizer.exp_avg_sq_dtype == torch.bfloat16

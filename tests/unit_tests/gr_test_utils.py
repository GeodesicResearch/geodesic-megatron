# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Shared fixtures/builders for the gradient-routing GPU tests.

Three test files build the same toy Nemotron-shaped MoE layer against the same world-1
parallel state, and two of them compare a GRAM-swapped layer against a vanilla one built
from an identical seed. That seeding discipline is load-bearing for every bitwise
assertion in the suite, so it lives in one place rather than in three copies that could
drift apart.
"""

import copy
import socket

import torch


HIDDEN, MOE_FFN, SHARED_FFN, AUX_FFN, NUM_EXPERTS = 32, 24, 16, 8, 4


def init_single_rank_process_group() -> None:
    """Initialize (or reuse) a world-size-1 NCCL process group on a free local port.

    Deliberately does NOT go through ``MASTER_ADDR``/``MASTER_PORT``. Inside a SLURM
    allocation both are already set to the job's rendezvous endpoint, which another process
    owns, so env-based init fails with EADDRINUSE. Asking the OS for a free port is also
    collision-free under pytest-xdist, and it leaves the environment untouched — conftest's
    per-worker ``MASTER_PORT`` stays authoritative for the tests that do use it.
    """
    if torch.distributed.is_initialized():
        return
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    torch.distributed.init_process_group(backend="nccl", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)


def init_model_parallel(**kwargs) -> None:
    """Bring up torch.distributed plus mcore's world-1 model-parallel state."""
    from megatron.core import parallel_state

    torch.cuda.set_device(0)
    init_single_rank_process_group()
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(**kwargs)


def moe_config(**overrides):
    """A Nemotron-shaped MoE config at toy width.

    Sigmoid router with expert bias, non-gated squared-relu, bias-free linears — the
    posture every GR run uses, minus the size. ``moe_latent_size`` is left unset because
    the provider refuses gradient routing on latent MoE.
    """
    from megatron.core.activations import squared_relu
    from megatron.core.transformer.transformer_config import TransformerConfig

    kwargs = dict(
        num_layers=2,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_moe_experts=NUM_EXPERTS,
        moe_ffn_hidden_size=MOE_FFN,
        moe_shared_expert_intermediate_size=SHARED_FFN,
        moe_router_topk=2,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=1e-3,
        moe_token_dispatcher_type="alltoall",
        activation_func=squared_relu,
        add_bias_linear=False,
        gated_linear_unit=False,
        expert_model_parallel_size=1,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=False,
        perform_initialization=True,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def stack_spec():
    """A PRIVATE copy of the hybrid stack spec the provider hands the GRAM swap.

    ``provider.mamba_stack_spec(provider)`` returns a process-global ``ModuleSpec`` — the
    same object on every call, from every provider instance — so a test that edited one in
    place would silently rewrite the spec every later test builds from.
    """
    from megatron.bridge.models.nemotronh.nemotron_h_provider import NemotronHModelProvider

    provider = NemotronHModelProvider(
        num_layers=2, hidden_size=HIDDEN, num_attention_heads=4, num_moe_experts=NUM_EXPERTS
    )
    return copy.deepcopy(provider.mamba_stack_spec(provider))


def gram_spec(aux_ffn=AUX_FFN):
    """A stack spec with the GRAM MoE-layer swap applied."""
    from megatron.bridge.models.mamba.gram_layer import swap_moe_layer_to_gram

    return swap_moe_layer_to_gram(stack_spec(), aux_ffn_hidden_size=aux_ffn)


def moe_builder(spec):
    """The ``partial`` that constructs one MoE layer from a stack spec."""
    return spec.submodules.moe_layer.submodules.mlp


def build_moe_layer(builder, config, layer_number=1, seed=4321):
    """Construct one MoE layer under a pinned RNG state.

    Both the host and the model-parallel CUDA seeds are re-pinned per build so a vanilla
    layer and a GRAM layer initialise their SHARED parameters identically — that identity
    is what makes the bitwise comparisons meaningful. The aux module is constructed last,
    so it draws from the stream only after every core parameter has been drawn.
    """
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(1234)
    return builder(config=config, layer_number=layer_number, name=f"decoder.layers.{layer_number}.mlp").cuda()

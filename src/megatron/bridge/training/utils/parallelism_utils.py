# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Resolve, log, and record the runtime-built parallelism dimensions.

The data-parallel size in particular is a *derived* quantity — there is no single
config field that holds it. It is the size of the data-parallel process group,
``world_size // (TP * PP * CP)`` (experts fold within ``DP * TP`` under parallel
folding, so EP does not multiply into the world size). These helpers read the
group sizes that Megatron actually built (via ``parallel_state``) so the resolved
DP/TP/PP/CP/EP/world are both logged at startup and recorded to W&B ``run.summary``
for per-run, post-hoc querying.
"""

from __future__ import annotations

from typing import Any

import torch
from megatron.core import parallel_state


def resolve_parallelism_dims() -> dict[str, int]:
    """Resolve the runtime parallelism group sizes from Megatron's ``parallel_state``.

    Must be called after model-parallel initialization (the groups must exist).
    ``data_parallel_size`` is the data-parallel group size — i.e. the derived
    ``world_size // (TP * PP * CP)`` that matches the ``dp_group`` the data loader
    shards over — not the (often vestigial) ``model.data_parallel_size`` config field.

    Returns:
        Mapping of dimension name to size: ``world_size``, ``data_parallel_size``,
        ``tensor_model_parallel_size``, ``pipeline_model_parallel_size``,
        ``context_parallel_size``, ``expert_model_parallel_size``.
    """
    return {
        "world_size": torch.distributed.get_world_size(),
        "data_parallel_size": parallel_state.get_data_parallel_world_size(),
        "tensor_model_parallel_size": parallel_state.get_tensor_model_parallel_world_size(),
        "pipeline_model_parallel_size": parallel_state.get_pipeline_model_parallel_world_size(),
        "context_parallel_size": parallel_state.get_context_parallel_world_size(),
        "expert_model_parallel_size": parallel_state.get_expert_model_parallel_world_size(),
    }


def format_parallelism_dims(dims: dict[str, int]) -> str:
    """Render resolved parallelism dims as a one-line human-readable startup banner."""
    return (
        f"> resolved parallelism: world_size={dims['world_size']} | "
        f"DP={dims['data_parallel_size']} "
        f"TP={dims['tensor_model_parallel_size']} "
        f"PP={dims['pipeline_model_parallel_size']} "
        f"CP={dims['context_parallel_size']} "
        f"EP={dims['expert_model_parallel_size']}"
    )


def parallelism_wandb_summary(dims: dict[str, int]) -> dict[str, int]:
    """Map resolved parallelism dims to namespaced W&B ``run.summary`` keys."""
    return {f"parallelism/{name}": size for name, size in dims.items()}


def record_parallelism_to_wandb(wandb_run: Any | None, dims: dict[str, int]) -> None:
    """Write resolved parallelism dims into a W&B run's summary.

    ``wandb_run`` is ``wandb.run`` (or ``None`` on ranks where W&B is not active —
    W&B initializes on a single rank by Megatron convention, so a ``None`` run here
    is expected control flow on the other ranks, not a swallowed failure).
    """
    if wandb_run is None:
        return
    wandb_run.summary.update(parallelism_wandb_summary(dims))


def record_parallelism_if_resolved(wandb_run: Any | None) -> dict[str, int] | None:
    """Resolve and record parallelism dims to W&B, but only when they can be resolved.

    ``resolve_parallelism_dims`` reads the mpu process groups; under
    ``use_decentralized_pg`` the groups live in a HyperCommGrid instead, so guard on
    ``model_parallel_is_initialized()`` to avoid recording default (wrong) sizes.

    Returns the resolved dims if recorded, else ``None``.
    """
    if not parallel_state.model_parallel_is_initialized():
        return None
    dims = resolve_parallelism_dims()
    record_parallelism_to_wandb(wandb_run, dims)
    return dims

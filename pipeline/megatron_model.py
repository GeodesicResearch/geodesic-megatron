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
"""A Megatron model built from an HF architecture at a chosen parallelism, and a sharded checkpoint loaded into it.

Collective over the torchrun world; needs GPUs.

Catalogue:
    sharded_bridge: an AutoBridge whose provider is set to a parallelism, with model parallelism initialised.
    load_sharded: a full Megatron checkpoint's model chunks, on the GPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import torch


def sharded_bridge(
    hf_model_id: str,
    tp: int,
    pp: int,
    ep: int,
    etp: int,
    pipeline_dtype: torch.dtype,
    trust_remote_code: bool,
    params_dtype: torch.dtype | None = None,
):
    """The AutoBridge of ``hf_model_id`` and the ``mp_overrides`` its checkpoint loads take.

    Builds the bridge (``is_safe_repo`` decides remote code from ``trust_remote_code``; with ``params_dtype`` the
    HF model loads in that dtype), sets the provider's TP/PP/EP/ETP and ``pipeline_dtype`` (and ``params_dtype``),
    finalises it and initialises model parallelism with seed 0. Reads ``hf_model_id``'s config (HF cache or Hub).

    Returns:
        (bridge, mp_overrides): the overrides name the same parallelism and dtypes.
    """
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.hf_pretrained.utils import is_safe_repo

    dtype_kwargs = {} if params_dtype is None else {"torch_dtype": params_dtype}
    bridge = AutoBridge.from_hf_pretrained(
        hf_model_id,
        trust_remote_code=is_safe_repo(trust_remote_code=trust_remote_code, hf_path=hf_model_id),
        **dtype_kwargs,
    )
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = tp
    provider.pipeline_model_parallel_size = pp
    provider.expert_model_parallel_size = ep
    provider.expert_tensor_parallel_size = etp
    provider.pipeline_dtype = pipeline_dtype
    if params_dtype is not None:
        provider.params_dtype = params_dtype
    provider.finalize()
    provider.initialize_model_parallel(seed=0)
    mp_overrides = {
        "tensor_model_parallel_size": tp,
        "pipeline_model_parallel_size": pp,
        "expert_model_parallel_size": ep,
        "expert_tensor_parallel_size": etp,
        "pipeline_dtype": pipeline_dtype,
    }
    if params_dtype is not None:
        mp_overrides["params_dtype"] = params_dtype
    return bridge, mp_overrides


def load_sharded(bridge, path: str, mp_overrides: dict) -> list:
    """The model chunks of the full Megatron checkpoint at ``path`` (an ``iter_*`` dir, or a run dir), on the GPU."""
    model = bridge.load_megatron_model(path, mp_overrides=mp_overrides, wrap_with_ddp=False)
    return [m.cuda() for m in model]

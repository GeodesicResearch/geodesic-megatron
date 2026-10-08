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

"""A one-rank NCCL world for single-GPU tests of code that takes a real process group.

Megatron's parallel state, communication ops and HybridEP buffers are built over torch.distributed
process groups, so a test on one GPU makes its own process the whole world: rank 0 of 1, on GPU 0.
"""

import contextlib
import os
from collections.abc import Iterator
from typing import Any

import torch


def init_one_rank_nccl_world() -> torch.distributed.ProcessGroup:
    """Make this process the whole NCCL world on GPU 0, once per process, and return that group.

    The rendezvous port is the per-xdist-worker ``MASTER_PORT`` that ``tests/unit_tests/conftest.py``
    pins; the defaults apply only outside it (a serial run, or a child process the caller gave its
    own environment).
    """
    torch.cuda.set_device(0)
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29781")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)
    return torch.distributed.group.WORLD


@contextlib.contextmanager
def one_rank_model_parallel_state(seed: int, **model_parallel_sizes: Any) -> Iterator[None]:
    """Megatron's model-parallel state over the one-rank NCCL world, torn down on exit.

    ``model_parallel_sizes`` go to ``parallel_state.initialize_model_parallel`` (every size is 1 on one
    rank; naming one builds its groups explicitly), and ``seed`` seeds the model-parallel CUDA RNG
    tracker that expert and tensor-parallel layers draw their initialisation from.
    """
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    init_one_rank_nccl_world()
    # Model-parallel state is global, and another test in this process may have left it holding the groups of a
    # process group it has since destroyed, so the state is built afresh over the live world.
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(**model_parallel_sizes)
    model_parallel_cuda_manual_seed(seed)
    try:
        yield
    finally:
        parallel_state.destroy_model_parallel()

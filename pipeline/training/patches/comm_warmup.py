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
"""Eager NCCL communicator warmup right after parallel-state setup.

Catalogue:
    patch_eager_comm_warmup: initialise every model-parallel group's communicator in one parallel wave.
"""

import logging


logger = logging.getLogger(__name__)


def _warmup_all_communicators() -> None:
    """Eagerly initialize NCCL communicators for every model-parallel group.

    PyTorch creates one NCCL communicator per process group on the FIRST collective
    issued on it. At deep pipeline parallelism the first microbatch ripples serially
    through the stages, so the per-hop communicator setup (NCCL bootstrap + CXI
    endpoint allocation, tens of seconds each on Slingshot) is paid sequentially —
    PP=22 exceeded the 10-minute default first-collective watchdog (observed:
    SeqNum=1 timeout on the last stage). Running one tiny collective per group here,
    where every rank participates simultaneously, initializes all communicators in a
    single parallel wave instead. A batched send/recv with both pipeline neighbors
    additionally warms NCCL's per-pair P2P transports, which are set up lazily even
    on an initialized communicator.
    """
    import time

    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as ps

    if not dist.is_initialized():
        return
    t0 = time.monotonic()
    device = torch.device("cuda", torch.cuda.current_device())

    group_getters = (
        "get_tensor_model_parallel_group",
        "get_pipeline_model_parallel_group",
        "get_context_parallel_group",
        "get_expert_model_parallel_group",
        "get_expert_tensor_parallel_group",
        "get_data_parallel_group",
        "get_model_parallel_group",
        "get_embedding_group",
        "get_position_embedding_group",
    )
    warmed = 0
    for name in group_getters:
        getter = getattr(ps, name, None)
        if getter is None:
            continue
        try:
            group = getter()
        except (AssertionError, RuntimeError, TypeError):
            continue
        for g in group if isinstance(group, (list, tuple)) else [group]:
            if g is None:
                continue
            try:
                if dist.get_world_size(group=g) > 1:
                    dist.all_reduce(torch.ones(1, device=device), group=g)
                    warmed += 1
            except (AssertionError, RuntimeError, TypeError):
                continue

    pp_pairs = 0
    try:
        pp_group = ps.get_pipeline_model_parallel_group()
        pp_world = dist.get_world_size(group=pp_group) if pp_group is not None else 1
    except (AssertionError, RuntimeError):
        pp_group, pp_world = None, 1
    if pp_group is not None and pp_world > 1:
        rank_in_pp = dist.get_rank(group=pp_group)
        global_ranks = dist.get_process_group_ranks(pp_group)
        send_next = torch.ones(1, device=device)
        send_prev = torch.ones(1, device=device)
        recv_next = torch.empty(1, device=device)
        recv_prev = torch.empty(1, device=device)
        ops = []
        if rank_in_pp + 1 < pp_world:
            nxt = global_ranks[rank_in_pp + 1]
            ops.append(dist.P2POp(dist.isend, send_next, nxt, group=pp_group))
            ops.append(dist.P2POp(dist.irecv, recv_next, nxt, group=pp_group))
        if rank_in_pp - 1 >= 0:
            prv = global_ranks[rank_in_pp - 1]
            ops.append(dist.P2POp(dist.irecv, recv_prev, prv, group=pp_group))
            ops.append(dist.P2POp(dist.isend, send_prev, prv, group=pp_group))
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
            pp_pairs = len(ops) // 2

    torch.cuda.synchronize()
    if dist.get_rank() == 0:
        logger.info(
            "Comm warmup: initialized %d group communicator(s) + %d pipeline P2P pair(s) in %.1fs.",
            warmed,
            pp_pairs,
            time.monotonic() - t0,
        )


def patch_eager_comm_warmup() -> bool:
    """Run a parallel NCCL communicator warmup right after parallel-state setup.

    Wraps ``megatron.core.parallel_state.initialize_model_parallel`` so the warmup
    fires on every rank immediately after the model-parallel groups exist — before
    model build and the first (otherwise serially-initializing) training iteration.
    Idempotent; returns True only on first install.
    """
    from megatron.core import parallel_state as ps

    if getattr(ps.initialize_model_parallel, "_isambard_comm_warmup", False):
        return False
    orig = ps.initialize_model_parallel

    def initialize_model_parallel_with_warmup(*args, **kwargs):
        out = orig(*args, **kwargs)
        _warmup_all_communicators()
        return out

    initialize_model_parallel_with_warmup._isambard_comm_warmup = True
    ps.initialize_model_parallel = initialize_model_parallel_with_warmup
    logger.info(
        "Installed comm-warmup patch: NCCL communicators initialize in one parallel wave after parallel-state setup."
    )
    return True

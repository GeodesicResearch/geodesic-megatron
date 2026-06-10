#!/usr/bin/env python3
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

"""Runtime monkeypatches applied before training starts.

These patch ``megatron.core`` (imported from the pinned ``3rdparty/Megatron-LM``
submodule, NOT editable as part of the megatron.bridge PR) without touching the
submodule source. They are installed once, on every rank, at the top of
``pipeline_training_run.main()``.

Currently a single patch is provided:

``patch_mamba_training_scan_fp32_state``
    Force the hybrid Mamba2 *training* scan to accumulate the inter-chunk SSM
    state in fp32 instead of bf16. See that function's docstring for the full
    rationale and the exact mamba_ssm code path it addresses.
"""

from __future__ import annotations

import logging


logger = logging.getLogger(__name__)

__all__ = ["patch_mamba_training_scan_fp32_state", "apply_isambard_training_patches"]


def patch_mamba_training_scan_fp32_state() -> bool:
    """Force fp32 inter-chunk SSM state in the Mamba2 *training* scan.

    Why
    ---
    Nemotron-3 Super is a hybrid Mamba2 + attention + MoE model. The Mamba2
    training step (``MambaMixer._ssm_training`` in
    ``megatron/core/ssm/mamba_mixer.py``) calls
    ``mamba_split_conv1d_scan_combined`` from the external ``mamba_ssm`` package.

    Inside that kernel (``mamba_ssm/ops/triton/ssd_combined.py``,
    ``_mamba_chunk_scan_combined_fwd``) the SSD recurrence is chunked. The
    *intra*-chunk states are accumulated in fp32 (``states_in_fp32=True``), but
    the *inter*-chunk state passing materialises its output at ``C.dtype``::

        states, final_states = _state_passing_fwd(..., out_dtype=C.dtype)

    With bf16 activations, ``C.dtype`` is bf16, so the running SSM state that is
    carried across chunks is rounded to bf16 at every chunk boundary. Worse, the
    downstream ``_chunk_scan_fwd`` Triton kernel reloads that state and casts it
    to ``C_ptr.dtype.element_ty`` (again bf16) before the ``C @ prev_states``
    matmul. The SSM state magnitude grows with the number of tokens integrated
    since the last reset; at seq_length=32768 with long documents that span most
    of the packed window it exceeds the bf16 representable range, producing a
    forward-loss NaN (observed at iter 2; 8K trained fine, 32K overflows).

    The packed-doc-boundary reset (``seq_idx``) is already implemented and
    correct, but does not help when a *single* document spans ~32K tokens.

    How the fix works
    -----------------
    The mamba_ssm training entry point ``mamba_split_conv1d_scan_combined`` does
    **not** expose a state-dtype / ``out_dtype`` knob (unlike Megatron's native
    forward-only varlen op ``mamba_chunk_scan_combined_varlen``, which threads a
    ``state_dtype`` argument straight into ``_state_passing_fwd``). The training
    path also needs autograd, so we cannot route through the varlen op.

    Since the kernels key their internal state dtype off the dtype of the inputs
    (``C.dtype`` for the state-passing output; ``C_ptr.dtype.element_ty`` for the
    chunk-scan matmul; ``states_in_fp32`` keeps chunk-state fp32 regardless), the
    robust lever available at the public boundary is to run the whole scan in
    fp32: we wrap ``mamba_split_conv1d_scan_combined`` to up-cast the fused
    ``zxbcdt`` projection (which carries z, x, B, C and dt) to fp32 before the
    call and cast the output back to the original (bf16) dtype afterwards. With
    fp32 inputs:

    * the conv runs in fp32,
    * ``C.dtype`` is fp32, so ``_state_passing_fwd`` keeps the inter-chunk state
      in fp32,
    * ``_chunk_scan_fwd`` no longer down-casts ``prev_states`` to bf16, so the
      ``C @ prev_states`` matmul stays fp32,

    i.e. the growing state never touches bf16. ``dt_bias``/``A``/``D`` are
    already passed as fp32 by ``_ssm_training``; ``seq_idx`` / ``cu_seqlens`` are
    forwarded untouched so the doc-boundary reset still fires.

    Scope / safety
    --------------
    * ``mamba_split_conv1d_scan_combined`` is used **only** by the training scan
      (``_ssm_training``). Inference prefill/decode use separate ops
      (``mamba_chunk_scan_combined_varlen`` / ``selective_state_update``), so
      this patch does not affect inference.
    * Casting an already-fp32 tensor is a no-op, and the output is cast back to
      the original dtype, so the wrapper is transparent if activations are not
      bf16.
    * If the symbol is missing (mamba_ssm not importable) or already patched,
      this is a guarded no-op.

    Composition with CP / packing
    -----------------------------
    Context parallelism gathers the full per-rank sequence before the scan (the
    scan in ``_ssm_training`` runs on the assembled sequence), and ``seq_idx``
    comes from ``packed_seq_params``. The wrapper only changes dtypes and leaves
    ``seq_idx`` / ``cu_seqlens`` and tensor shapes untouched, so it composes with
    CP=8 and packed ``seq_idx`` exactly as the unpatched op would.

    Returns
    -------
    bool
        ``True`` if the patch was installed, ``False`` if it was skipped
        (symbol unavailable or already patched).
    """
    try:
        import torch
        from megatron.core.ssm import mamba_mixer
    except Exception as exc:  # pragma: no cover - import-environment dependent
        logger.warning("fp32-SSM-state patch skipped: could not import megatron.core.ssm.mamba_mixer (%s)", exc)
        return False

    original = getattr(mamba_mixer, "mamba_split_conv1d_scan_combined", None)
    if original is None:
        logger.warning(
            "fp32-SSM-state patch skipped: mamba_mixer.mamba_split_conv1d_scan_combined is None "
            "(mamba_ssm not available)."
        )
        return False

    if getattr(original, "_fp32_ssm_state_patched", False):
        logger.info("fp32-SSM-state patch already installed; skipping.")
        return False

    def mamba_split_conv1d_scan_combined_fp32_state(zxbcdt, *args, **kwargs):
        """fp32-state wrapper around the mamba_ssm training scan.

        Up-casts the fused ``zxbcdt`` projection to fp32 so the inter-chunk SSM
        state recurrence stays in fp32, then casts the result back to the input
        dtype. All other positional/keyword args (conv weights, A, D, dt_bias,
        chunk_size, seq_idx, ...) are forwarded unchanged.
        """
        in_dtype = zxbcdt.dtype
        if in_dtype == torch.float32:
            # Already fp32 (e.g. fp32 activations): nothing to force.
            return original(zxbcdt, *args, **kwargs)

        out = original(zxbcdt.to(torch.float32), *args, **kwargs)

        # return_final_states=True yields a (out, final_states) tuple.
        if isinstance(out, tuple):
            y, *rest = out
            y = y.to(in_dtype)
            return (y, *rest)
        return out.to(in_dtype)

    mamba_split_conv1d_scan_combined_fp32_state._fp32_ssm_state_patched = True
    mamba_split_conv1d_scan_combined_fp32_state._wrapped = original

    mamba_mixer.mamba_split_conv1d_scan_combined = mamba_split_conv1d_scan_combined_fp32_state
    logger.info(
        "Installed fp32-SSM-state patch: Mamba2 training scan "
        "(mamba_split_conv1d_scan_combined) now accumulates inter-chunk state in fp32."
    )
    return True


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


def apply_isambard_training_patches() -> None:
    """Apply all runtime training patches. Safe to call once per process."""
    patch_mamba_training_scan_fp32_state()


if __name__ == "__main__":
    # CPU-only sanity check: import + confirm the patched callable is in place.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    installed = patch_mamba_training_scan_fp32_state()
    from megatron.core.ssm import mamba_mixer

    fn = mamba_mixer.mamba_split_conv1d_scan_combined
    print(f"patch installed: {installed}")
    print(f"callable name:   {getattr(fn, '__name__', fn)}")
    print(f"is patched:      {getattr(fn, '_fp32_ssm_state_patched', False)}")
    # Idempotency: a second call must be a no-op.
    again = patch_mamba_training_scan_fp32_state()
    print(f"second call installed (expect False): {again}")

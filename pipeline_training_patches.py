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

Patches provided:

``patch_mamba_training_scan_fp32_state``
    Force the hybrid Mamba2 *training* scan to accumulate the inter-chunk SSM
    state in fp32 instead of bf16. Two modes: ``checkpointed=False`` (direct fp32
    cast, ~2x the scan's saved activations) and ``checkpointed=True``
    (memory-neutral: fp32 scan wrapped in non-reentrant activation checkpointing,
    only the original bf16 input is saved and the fp32 forward is recomputed
    during backward). See the function docstring for the full rationale and the
    exact mamba_ssm code path it addresses.

``patch_mamba_training_scan_save_offload``
    Host-offload the Mamba2 training scan's saved-for-backward tensors via
    ``torch.autograd.graph.save_on_cpu(pin_memory=True)`` — frees ~15-25 GB of
    stage-0 GPU memory (the Mamba saves are neither recomputable nor visible to
    NVTE offload). Composes with the fp32 ``direct`` mode; refused under the
    fp32 ``checkpoint`` mode (recompute already discards the saves).

``patch_eager_comm_warmup``
    Initialize all NCCL communicators in one parallel wave right after
    parallel-state setup (deep-PP first-collective watchdog mitigation).
"""

from __future__ import annotations

import logging


logger = logging.getLogger(__name__)

__all__ = [
    "apply_isambard_training_patches",
    "patch_eager_comm_warmup",
    "patch_mamba_training_scan_fp32_state",
    "patch_mamba_training_scan_save_offload",
]


def _make_fp32_scan_wrapper(original, checkpointed: bool):
    """Build the fp32-state wrapper around the Mamba2 training scan op.

    Factored out of :func:`patch_mamba_training_scan_fp32_state` so the wrapper
    mechanics can be exercised on CPU against a fake op (see
    ``_selftest_checkpoint_wrapper_mechanics``) without importing megatron.core.

    Args:
        original: The unpatched ``mamba_split_conv1d_scan_combined`` callable.
        checkpointed: If False, plain fp32 up-cast around the call (the op then
            saves fp32 activations for backward — roughly 2x the unpatched saved
            memory). If True, the fp32 call runs inside
            ``torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`` so the
            only tensors kept alive for backward are the op's original (bf16)
            inputs — same retained memory as the unpatched op — and the fp32
            up-cast + scan forward are recomputed during backward.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    def _fp32_scan(zxbcdt, *args, **kwargs):
        """Up-cast zxbcdt to fp32, run the scan, cast the primary output back."""
        in_dtype = zxbcdt.dtype
        out = original(zxbcdt.to(torch.float32), *args, **kwargs)
        # return_final_states=True yields a (out, final_states) tuple.
        if isinstance(out, tuple):
            y, *rest = out
            return (y.to(in_dtype), *rest)
        return out.to(in_dtype)

    if not checkpointed:

        def wrapper(zxbcdt, *args, **kwargs):
            if zxbcdt.dtype == torch.float32:
                # Already fp32 (e.g. fp32 activations): nothing to force.
                return original(zxbcdt, *args, **kwargs)
            return _fp32_scan(zxbcdt, *args, **kwargs)

    else:

        def wrapper(zxbcdt, *args, **kwargs):
            if zxbcdt.dtype == torch.float32:
                return original(zxbcdt, *args, **kwargs)
            needs_backward = torch.is_grad_enabled() and (
                zxbcdt.requires_grad or any(isinstance(t, torch.Tensor) and t.requires_grad for t in args)
            )
            if not needs_backward:
                # Eval / no-grad: nothing is saved for backward anyway, so plain
                # fp32 is already memory-neutral and checkpoint would only emit a
                # "none of the inputs require grad" warning.
                return _fp32_scan(zxbcdt, *args, **kwargs)
            # Non-reentrant checkpoint: the autograd graph is built during this
            # (original) forward, so gradients flow to every participating tensor
            # — positional args (conv weight/bias, dt_bias, A) AND anything closed
            # over — exactly as without checkpointing. Only the *saved-for-backward*
            # tensors (the op's internal ctx.save_for_backward of fp32 zxbcdt,
            # out_x, ...) are dropped after forward and recomputed in backward.
            # kwargs (chunk_size, activation, seq_idx, ...) are forwarded to the
            # function by torch's non-reentrant checkpoint. preserve_rng_state=False
            # is safe: the scan is deterministic (no dropout / RNG inside).
            return checkpoint(
                _fp32_scan,
                zxbcdt,
                *args,
                use_reentrant=False,
                preserve_rng_state=False,
                **kwargs,
            )

    wrapper.__name__ = "mamba_split_conv1d_scan_combined_fp32_state" + ("_checkpointed" if checkpointed else "")
    wrapper._fp32_ssm_state_patched = True
    wrapper._fp32_ssm_state_mode = "checkpoint" if checkpointed else "direct"
    wrapper._wrapped = original
    return wrapper


def patch_mamba_training_scan_fp32_state(checkpointed: bool = False) -> bool:
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
    since the last ``seq_idx`` reset; once a single long document spans most of a
    32K packed window the state exceeds the bf16 representable range and the
    forward loss NaNs. Field-confirmed at seq_length=32768 (TP1/CP4/EP4/PP22):
    270 healthy iterations, then a step-function "NaN in local forward loss" at
    iteration 272 when a long-doc batch arrived — no instability buildup, purely
    data-triggered. 8K trained fine.

    The packed-doc-boundary reset (``seq_idx``) is implemented and correct, but
    does not help when a *single* document spans ~32K tokens.

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
    fp32 inputs the conv, the state passing, and the ``C @ prev_states`` matmul
    all stay fp32 — the growing state never touches bf16. ``dt_bias``/``A``/``D``
    are already passed as fp32 by ``_ssm_training``; ``seq_idx`` / ``cu_seqlens``
    are forwarded untouched so the doc-boundary reset still fires.

    Memory modes
    ------------
    ``checkpointed=False`` (env ``ISAMBARD_FP32_SSM_STATE=1``)
        Direct fp32 call. The op's autograd Function then saves fp32 tensors
        (``zxbcdt``, ``out_x``) via ``ctx.save_for_backward`` — roughly **2x the
        scan's saved-activation memory**. At PP=22 with 22 in-flight microbatches
        and 2 Mamba layers/stage this is ~ +13 GB on an 84/95 GB stage → OOM risk.

    ``checkpointed=True`` (env ``ISAMBARD_FP32_SSM_STATE=checkpoint``)
        **Memory-neutral.** The fp32 call runs inside
        ``torch.utils.checkpoint.checkpoint(..., use_reentrant=False)``: the only
        tensors kept alive until backward are the checkpoint's *inputs* — the
        original bf16 ``zxbcdt`` plus the small parameter tensors, i.e. the same
        tensors the unpatched op would have saved — while everything saved inside
        (fp32 ``zxbcdt`` cast, fp32 ``out_x``, conv output) is freed after forward
        and recomputed during backward. Cost: one extra scan forward per Mamba
        layer per microbatch (the scan is a small fraction of total model FLOPs).
        Transient fp32 memory during backward recompute is one layer at a time.

    Scope / safety
    --------------
    * ``mamba_split_conv1d_scan_combined`` is used **only** by the training scan
      (``_ssm_training``). Inference prefill/decode use separate ops
      (``mamba_chunk_scan_combined_varlen`` / ``selective_state_update``), so
      this patch does not affect inference.
    * Casting an already-fp32 tensor is a no-op, and the output is cast back to
      the original dtype, so the wrapper is transparent if activations are not
      bf16. Under no-grad the checkpointed mode falls back to the plain fp32 call.
    * If the symbol is missing (mamba_ssm not importable) or already patched,
      this is a guarded no-op.
    * The scan is deterministic (no dropout/RNG), so the checkpointed mode uses
      ``preserve_rng_state=False``; the default ``determinism_check`` still
      validates recomputed tensor metadata.

    Composition with CP / packing
    -----------------------------
    Context parallelism gathers the full per-rank sequence before the scan (the
    scan in ``_ssm_training`` runs on the assembled sequence), and ``seq_idx``
    comes from ``packed_seq_params``. The wrapper only changes dtypes and leaves
    ``seq_idx`` / ``cu_seqlens`` and tensor shapes untouched, so it composes with
    CP and packed ``seq_idx`` exactly as the unpatched op would.

    Args:
        checkpointed: Select the memory-neutral activation-checkpointed variant
            (see "Memory modes" above).

    Returns:
        ``True`` if the patch was installed, ``False`` if it was skipped
        (symbol unavailable or already patched).
    """
    try:
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

    requested_mode = "checkpoint" if checkpointed else "direct"
    if getattr(original, "_fp32_ssm_state_patched", False):
        installed_mode = getattr(original, "_fp32_ssm_state_mode", "unknown")
        if installed_mode != requested_mode:
            logger.warning(
                "fp32-SSM-state patch already installed in mode=%s; requested mode=%s ignored.",
                installed_mode,
                requested_mode,
            )
        else:
            logger.info("fp32-SSM-state patch already installed (mode=%s); skipping.", installed_mode)
        return False

    mamba_mixer.mamba_split_conv1d_scan_combined = _make_fp32_scan_wrapper(original, checkpointed)
    logger.info(
        "Installed fp32-SSM-state patch (mode=%s): Mamba2 training scan "
        "(mamba_split_conv1d_scan_combined) now accumulates inter-chunk state in fp32%s.",
        requested_mode,
        " inside non-reentrant activation checkpointing (memory-neutral)" if checkpointed else "",
    )
    return True


def _make_save_offload_wrapper(inner):
    """Wrap the Mamba2 training scan so its saved-for-backward tensors live on host.

    Runs ``inner`` (the original op, or the fp32-direct wrapper) under
    ``torch.autograd.graph.save_on_cpu(pin_memory=True)``. Every tensor the op
    saves for backward — including ``ctx.save_for_backward`` inside its custom
    autograd Function (hook coverage proven by the fp32-checkpoint selftest) — is
    packed into a pinned host copy at the end of the op's forward, releasing the
    GPU storage; the unpack hook streams it back (``.to(device,
    non_blocking=True)`` from pinned memory) when the op's backward reads
    ``ctx.saved_tensors``.

    Scoped tightly to this single call: torch applies only the innermost active
    saved-tensor hook pair, and this context spans nothing but the scan op, so
    parameters and activations saved by surrounding ops (norms, projections,
    attention, MoE) are untouched.
    """
    import torch
    from torch.autograd.graph import save_on_cpu

    def wrapper(zxbcdt, *args, **kwargs):
        if not torch.is_grad_enabled():
            # Eval / no-grad: nothing is saved for backward, nothing to offload.
            return inner(zxbcdt, *args, **kwargs)
        with save_on_cpu(pin_memory=True):
            return inner(zxbcdt, *args, **kwargs)

    wrapper.__name__ = "mamba_split_conv1d_scan_combined_save_offload"
    wrapper._mamba_save_offload_patched = True
    wrapper._wrapped = inner
    # Keep the fp32 markers visible through this wrapper so the fp32 patch's
    # idempotency guard still fires if it is (against the documented order)
    # called after this patch.
    if getattr(inner, "_fp32_ssm_state_patched", False):
        wrapper._fp32_ssm_state_patched = True
        wrapper._fp32_ssm_state_mode = inner._fp32_ssm_state_mode
    return wrapper


def patch_mamba_training_scan_save_offload() -> bool:
    """Host-offload the Mamba2 training scan's saved-for-backward tensors.

    Why
    ---
    The scan op's ``ctx.save_for_backward`` tensors (``zxbcdt`` and ``out_x``,
    ~300-600 MB bf16 per Mamba layer per microbatch at 8192 tok/rank) are the one
    large activation class on a Nemotron-H stage that is neither recomputable
    ('mamba' is not an allowed ``recompute_modules`` entry) nor visible to NVTE
    fine-grained activation offload (which only hooks TE-wrapped modules). Under
    1F1B with many in-flight microbatches they pile up on early stages: ~16 µb x
    2 Mamba layers/stage x ~0.6 GB ≈ 15-25 GB of stage-0 GPU memory. Field
    motivation: the zero-recompute arm at 8192 tok/rank OOMed twice by ~0.5 GB
    even at 16 in-flight microbatches; freeing the Mamba saves buys back the
    recompute-drop's +15-20% step-time win.

    How
    ---
    Wraps ``mamba_mixer.mamba_split_conv1d_scan_combined`` (the same symbol the
    fp32 patch wraps) in ``save_on_cpu(pin_memory=True)``:

    * pack — fires once per saved tensor at the end of the op's forward: copies
      into a pinned host buffer (``torch.empty(..., pin_memory=True)`` +
      ``copy_``) and drops the GPU reference. The copy is synchronous with
      respect to the calling CPU thread.
    * unpack — fires when the op's backward reads ``ctx.saved_tensors``: copies
      back with ``.to(device, non_blocking=True)``, async from pinned memory on
      the current stream and therefore correctly ordered ahead of the consuming
      backward kernels.

    Composition with the fp32-SSM-state patch (apply THIS patch second)
    -------------------------------------------------------------------
    * no fp32 patch: offloads the op's bf16 saves (the zero-recompute speed arm).
    * fp32 ``direct`` (``ISAMBARD_FP32_SSM_STATE=1``): offloads the op's **fp32**
      saves — direct mode's ~2x saved-memory penalty moves entirely to host RAM,
      making direct+offload a strong 32K combination.
    * fp32 ``checkpoint``: **refused with a warning.** Non-reentrant checkpoint
      already discards the op's saves (recompute) — there is nothing left to
      offload. The tensors checkpoint retains (the bf16 inputs) are held as plain
      frame references, not via saved-tensor hooks, so ``save_on_cpu`` could not
      reach them anyway; and torch applies only the innermost hook pair
      (checkpoint's own), so the combination would be dead code. Pick one:
      ``checkpoint`` (recompute, no host traffic) or ``1`` + offload (no
      recompute, host traffic).

    Performance (estimates — measure on GPU)
    ----------------------------------------
    * D2H/H2D volume ≈ the op's saved bytes: ~300-600 MB per layer-µb per
      direction at 8192 tok/rank → ~1-2 ms per direction per layer call on GH200
      NVLink-C2C (~450 GB/s).
    * The pack copy blocks the enqueueing CPU thread (and must wait for the
      producing kernels), so it can stall kernel submission; unpack is
      non-blocking for the CPU but still serializes on the current stream ahead
      of the backward kernels. Expect a few ms per microbatch, partially hidden
      under pipeline bubbles — not free.
    * Pinned buffers come from torch's caching host allocator and are reused
      across iterations; the first iterations pay one-time ``cudaHostAlloc``
      page-locking for the working set (~20 GB on a stage-0 rank: 16 µb x 2
      layers x ~0.6 GB — comfortably within the GH200's ~480 GB LPDDR).

    Returns:
        ``True`` if installed; ``False`` if skipped (symbol unavailable, already
        installed, or refused because fp32 mode is ``checkpoint``).
    """
    try:
        from megatron.core.ssm import mamba_mixer
    except Exception as exc:  # pragma: no cover - import-environment dependent
        logger.warning("Mamba save-offload patch skipped: could not import megatron.core.ssm.mamba_mixer (%s)", exc)
        return False

    current = getattr(mamba_mixer, "mamba_split_conv1d_scan_combined", None)
    if current is None:
        logger.warning(
            "Mamba save-offload patch skipped: mamba_mixer.mamba_split_conv1d_scan_combined is None "
            "(mamba_ssm not available)."
        )
        return False

    if getattr(current, "_mamba_save_offload_patched", False):
        logger.info("Mamba save-offload patch already installed; skipping.")
        return False

    if getattr(current, "_fp32_ssm_state_mode", None) == "checkpoint":
        logger.warning(
            "Mamba save-offload patch refused: fp32-SSM-state 'checkpoint' mode is installed, which already "
            "discards the scan's saved tensors via recompute — host-offload would be dead code (torch applies "
            "only the innermost saved-tensor hooks, and checkpoint retains its inputs as plain references). "
            "Use ISAMBARD_FP32_SSM_STATE=1 together with the offload, or 'checkpoint' alone."
        )
        return False

    mamba_mixer.mamba_split_conv1d_scan_combined = _make_save_offload_wrapper(current)
    logger.info(
        "Installed Mamba save-offload patch: training-scan saved tensors now offload to pinned host memory "
        "(wrapping: %s).",
        getattr(current, "__name__", current),
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


def patch_packed_parquet_row_telemetry() -> bool:
    """Log every packed-parquet row fetch: dataset index -> underlying parquet row.

    Diagnostic for data-dependent failures (e.g. a deterministic NaN at a fixed
    iteration): with DP=1 the global batch of iteration k is exactly fetches
    64*(k-1)..64*k-1 in fetch order, so the log maps a failing iteration to the
    physical parquet rows it consumed. Gate: ISAMBARD_DATA_ROW_TELEMETRY=1.
    Idempotent; returns True only on first install.
    """
    from megatron.bridge.data.datasets.packed_parquet import GPTSFTPackedParquetDataset

    if getattr(GPTSFTPackedParquetDataset.__getitem__, "_isambard_row_telemetry", False):
        return False
    orig = GPTSFTPackedParquetDataset.__getitem__

    def getitem_with_telemetry(self, idx):
        mapped = idx
        sm = getattr(self, "samples_mapping", None)
        if sm is not None:
            try:
                mapped = int(sm[idx])
            except Exception:
                mapped = idx
        logger.info(f"[row-telemetry] fetch idx={idx} -> parquet_row={mapped}")
        return orig(self, idx)

    getitem_with_telemetry._isambard_row_telemetry = True
    GPTSFTPackedParquetDataset.__getitem__ = getitem_with_telemetry
    logger.info("Installed packed-parquet row telemetry: each fetch logs dataset idx -> parquet row.")
    return True


def apply_isambard_training_patches(
    fp32_ssm_state_checkpointed: bool = False, mamba_save_offload: bool = False
) -> None:
    """Apply all runtime training patches. Safe to call once per process."""
    patch_mamba_training_scan_fp32_state(checkpointed=fp32_ssm_state_checkpointed)
    if mamba_save_offload:
        patch_mamba_training_scan_save_offload()


def _selftest_checkpoint_wrapper_mechanics() -> None:
    """CPU-only test of the fp32 wrapper mechanics against a fake scan op.

    The fake op mirrors the real one structurally: a custom autograd.Function that
    saves its (fp32) input via ``ctx.save_for_backward`` (like
    ``MambaSplitConv1dScanCombinedFn`` saving ``zxbcdt``/``out_x``), a learnable
    weight passed positionally, a Parameter referenced via closure, a tensor kwarg
    (``seq_idx``), and a tuple return. Verifies, for direct vs checkpointed mode:

    (i)   forward output dtype is bf16 (and tuple extras pass through untouched);
    (ii)  backward recomputes (fake op runs twice under checkpoint, once direct);
    (iii) grads flow to the input, the positional Parameter, AND the closed-over
          Parameter — and match between modes;
    (iv)  memory: the fp32 tensor saved inside the op is freed after forward under
          checkpoint (weakref dead) but kept alive in direct mode — i.e. only the
          original bf16 input is retained for backward in checkpointed mode.
    """
    import gc
    import weakref

    import torch

    state = {"op_calls": 0, "saved_fp32_refs": [], "seen_dtype": None, "seen_seq_idx": None}
    w_closure = torch.nn.Parameter(torch.tensor(2.0))

    class _FakeScanFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x32, conv_w):
            ctx.save_for_backward(x32, conv_w)  # mimics the real op saving fp32 activations
            state["saved_fp32_refs"].append(weakref.ref(x32))
            return x32 * conv_w

        @staticmethod
        def backward(ctx, g):
            x32, conv_w = ctx.saved_tensors
            return g * conv_w, (g * x32).sum().reshape(conv_w.shape)

    def fake_op(zxbcdt, conv_w, **kwargs):
        state["op_calls"] += 1
        state["seen_dtype"] = zxbcdt.dtype
        state["seen_seq_idx"] = kwargs.get("seq_idx")
        y = _FakeScanFn.apply(zxbcdt, conv_w) * w_closure  # closure Parameter participates
        return (y, "final_states_sentinel")  # tuple return, non-tensor extra

    def run(checkpointed: bool) -> dict:
        state.update(op_calls=0, saved_fp32_refs=[], seen_dtype=None, seen_seq_idx=None)
        w_closure.grad = None
        torch.manual_seed(0)
        x = torch.randn(4, 8, dtype=torch.bfloat16, requires_grad=True)
        conv_w = torch.nn.Parameter(torch.tensor(1.5))
        wrapper = _make_fp32_scan_wrapper(fake_op, checkpointed=checkpointed)

        y, extra = wrapper(x, conv_w, seq_idx=torch.tensor([0, 0, 1, 1]))
        gc.collect()
        fp32_alive_after_fwd = state["saved_fp32_refs"][0]() is not None
        calls_after_fwd = state["op_calls"]
        y.float().sum().backward()
        return {
            "out_dtype": y.dtype,
            "extra_ok": extra == "final_states_sentinel",
            "op_saw_fp32": state["seen_dtype"] == torch.float32,
            "seq_idx_ok": torch.equal(state["seen_seq_idx"], torch.tensor([0, 0, 1, 1])),
            "calls_fwd": calls_after_fwd,
            "calls_total": state["op_calls"],
            "fp32_alive_after_fwd": fp32_alive_after_fwd,
            "grad_x": x.grad.clone(),
            "grad_conv_w": conv_w.grad.clone(),
            "grad_closure_w": w_closure.grad.clone(),
        }

    direct, ckpt = run(checkpointed=False), run(checkpointed=True)

    print(
        "[mechanics] mode=direct    : out dtype=%s, op calls fwd/total=%d/%d, fp32 saved alive after fwd=%s"
        % (direct["out_dtype"], direct["calls_fwd"], direct["calls_total"], direct["fp32_alive_after_fwd"])
    )
    print(
        "[mechanics] mode=checkpoint: out dtype=%s, op calls fwd/total=%d/%d, fp32 saved alive after fwd=%s"
        % (ckpt["out_dtype"], ckpt["calls_fwd"], ckpt["calls_total"], ckpt["fp32_alive_after_fwd"])
    )
    print(
        "[mechanics] (i)   bf16 output both modes        :", direct["out_dtype"] == ckpt["out_dtype"] == torch.bfloat16
    )
    print(
        "[mechanics]       tuple extras + fp32 in + seq_idx:",
        all(m[k] for m in (direct, ckpt) for k in ("extra_ok", "op_saw_fp32", "seq_idx_ok")),
    )
    print("[mechanics] (ii)  backward recomputes (1 vs 2)  :", (direct["calls_total"], ckpt["calls_total"]) == (1, 2))
    print(
        "[mechanics] (iii) grads non-None both modes     :",
        all(m[k] is not None for m in (direct, ckpt) for k in ("grad_x", "grad_conv_w", "grad_closure_w")),
    )
    print(
        "[mechanics]       grads match direct vs ckpt    : x=%s conv_w=%s closure_w=%s"
        % (
            torch.allclose(direct["grad_x"], ckpt["grad_x"]),
            torch.allclose(direct["grad_conv_w"], ckpt["grad_conv_w"]),
            torch.allclose(direct["grad_closure_w"], ckpt["grad_closure_w"]),
        )
    )
    print(
        "[mechanics] (iv)  fp32 saved freed after fwd    : direct=%s (kept), checkpoint=%s (freed)"
        % (direct["fp32_alive_after_fwd"], not ckpt["fp32_alive_after_fwd"])
    )

    with torch.no_grad():
        state.update(op_calls=0)
        wrapper = _make_fp32_scan_wrapper(fake_op, checkpointed=True)
        y_ng, _ = wrapper(torch.randn(4, 8, dtype=torch.bfloat16), torch.tensor(1.5))
    print(
        "[mechanics] no-grad path: bf16 out=%s, single call=%s (plain fp32, no checkpoint)"
        % (y_ng.dtype == torch.bfloat16, state["op_calls"] == 1)
    )


def _selftest_save_offload_wrapper_mechanics() -> None:
    """CPU-only test of the save_on_cpu offload wrapper against a fake scan op.

    The fake op saves its input plus a big intermediate (``out_x`` stand-in) via
    ``ctx.save_for_backward`` inside a custom autograd.Function, mirroring the
    real scan. CPU-only, so device movement degenerates to CPU->CPU copies; hook
    firing is observed structurally instead: under ``save_on_cpu`` the originally
    saved tensors are replaced by packed copies — the intermediate is freed after
    forward (weakref dead) and backward sees different ``data_ptr`` values than
    were saved. Verifies:

    (i)   pack/unpack hooks fire for the op's custom-Function saves;
    (ii)  backward grads match the unwrapped baselines bit-for-bit;
    (iii) the composition matrix: offload alone and offload+fp32-direct offload
          the saves without recompute; offload+fp32-checkpoint at wrapper level
          leaves checkpoint in charge (recompute fires, offload is a no-op,
          nothing breaks) — the patch-level install refuses that combo anyway;
    (iv)  the no-grad path bypasses the hooks entirely.
    """
    import gc
    import weakref

    import torch

    state = {"op_calls": 0, "save_ptrs": [], "bwd_ptrs": [], "interm_ref": None, "seen_dtype": None}

    class _FakeScanFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, conv_w):
            state["op_calls"] += 1
            out_x = x * 3.0  # big intermediate retained only via save_for_backward (out_x stand-in)
            ctx.save_for_backward(x, out_x, conv_w)
            state["save_ptrs"].append((x.data_ptr(), out_x.data_ptr()))
            state["interm_ref"] = weakref.ref(out_x)
            return out_x * conv_w

        @staticmethod
        def backward(ctx, g):
            x, out_x, conv_w = ctx.saved_tensors
            state["bwd_ptrs"].append((x.data_ptr(), out_x.data_ptr()))
            return g * 3.0 * conv_w, (g * out_x).sum().reshape(conv_w.shape)

    def fake_op(zxbcdt, conv_w, **kwargs):
        state["seen_dtype"] = zxbcdt.dtype
        return (_FakeScanFn.apply(zxbcdt, conv_w), "final_states_sentinel")

    def run(build_wrapper) -> dict:
        state.update(op_calls=0, save_ptrs=[], bwd_ptrs=[], interm_ref=None, seen_dtype=None)
        torch.manual_seed(0)
        x = torch.randn(4, 8, dtype=torch.bfloat16, requires_grad=True)
        conv_w = torch.nn.Parameter(torch.tensor(1.5))
        fn = build_wrapper(fake_op)
        y, extra = fn(x, conv_w, seq_idx=torch.tensor([0, 0, 1, 1]))
        gc.collect()
        interm_alive = state["interm_ref"]() is not None
        y.float().sum().backward()
        return {
            "out_dtype": y.dtype,
            "extra_ok": extra == "final_states_sentinel",
            "calls": state["op_calls"],
            "interm_alive_after_fwd": interm_alive,
            # data_ptrs the op's backward saw vs what the first forward saved:
            # identical without hooks, different once pack/unpack replaced them.
            "ptr_same_in_bwd": state["bwd_ptrs"][-1] == state["save_ptrs"][0],
            "grad_x": x.grad.clone(),
            "grad_conv_w": conv_w.grad.clone(),
        }

    plain = run(lambda op: op)
    off = run(_make_save_offload_wrapper)
    fp32_dir = run(lambda op: _make_fp32_scan_wrapper(op, checkpointed=False))
    off_fp32 = run(lambda op: _make_save_offload_wrapper(_make_fp32_scan_wrapper(op, checkpointed=False)))
    fp32_ck = run(lambda op: _make_fp32_scan_wrapper(op, checkpointed=True))
    off_ck = run(lambda op: _make_save_offload_wrapper(_make_fp32_scan_wrapper(op, checkpointed=True)))

    for name, m in (
        ("plain         ", plain),
        ("offload       ", off),
        ("fp32-direct   ", fp32_dir),
        ("off+fp32-dir  ", off_fp32),
        ("fp32-ckpt     ", fp32_ck),
        ("off+fp32-ckpt ", off_ck),
    ):
        print(
            "[offload] %s: calls=%d, interm alive after fwd=%-5s, saved ptrs same in bwd=%s"
            % (name, m["calls"], m["interm_alive_after_fwd"], m["ptr_same_in_bwd"])
        )
    print(
        "[offload] (i)   hooks fire under offload (ptrs replaced + interm freed):",
        not off["ptr_same_in_bwd"]
        and not off["interm_alive_after_fwd"]
        and not off_fp32["ptr_same_in_bwd"]
        and not off_fp32["interm_alive_after_fwd"],
    )
    print(
        "[offload]       vs plain/fp32-direct keep saves on device, no hooks    :",
        plain["ptr_same_in_bwd"]
        and plain["interm_alive_after_fwd"]
        and fp32_dir["ptr_same_in_bwd"]
        and fp32_dir["interm_alive_after_fwd"],
    )
    print(
        "[offload] (ii)  grads match baselines: off==plain: x=%s w=%s | off+fp32==fp32: x=%s w=%s"
        % (
            torch.equal(off["grad_x"], plain["grad_x"]),
            torch.equal(off["grad_conv_w"], plain["grad_conv_w"]),
            torch.equal(off_fp32["grad_x"], fp32_dir["grad_x"]),
            torch.equal(off_fp32["grad_conv_w"], fp32_dir["grad_conv_w"]),
        )
    )
    print(
        "[offload] (iii) off+ckpt: checkpoint stays in charge (recompute=2, no break, grads match ckpt): %s"
        % (
            off_ck["calls"] == 2
            and torch.equal(off_ck["grad_x"], fp32_ck["grad_x"])
            and torch.equal(off_ck["grad_conv_w"], fp32_ck["grad_conv_w"])
        )
    )
    print(
        "[offload]       no recompute in offload modes (calls==1)              :",
        off["calls"] == 1 and off_fp32["calls"] == 1,
    )
    print(
        "[offload]       bf16 output all variants                              :",
        all(m["out_dtype"] == torch.bfloat16 for m in (plain, off, fp32_dir, off_fp32, fp32_ck, off_ck)),
    )

    with torch.no_grad():
        state.update(op_calls=0)
        y_ng, _ = _make_save_offload_wrapper(fake_op)(torch.randn(4, 8, dtype=torch.bfloat16), torch.tensor(1.5))
    print(
        "[offload] (iv)  no-grad path: single call=%s, no graph=%s (hooks bypassed)"
        % (state["op_calls"] == 1, y_ng.grad_fn is None)
    )

    if torch.cuda.is_available():  # pragma: no cover - GPU-node only
        dev = torch.device("cuda")
        x = torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16, requires_grad=True)
        conv_w = torch.nn.Parameter(torch.tensor(1.5, device=dev))
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        y, _ = _make_save_offload_wrapper(fake_op)(x, conv_w)
        torch.cuda.synchronize()
        held = torch.cuda.memory_allocated() - base
        print("[offload] GPU: bytes retained after fwd (expect ~= output only): %d" % held)
        y.float().sum().backward()
    else:
        print("[offload] GPU memory check: skipped (CUDA not available on this node)")


if __name__ == "__main__":
    # CPU-only sanity checks (no GPU / SLURM):
    #  1. wrapper mechanics against fake ops (dtype, recompute, grads, memory, offload);
    #  2. env-driven install + idempotency against the real megatron.core.ssm.mamba_mixer,
    #     mirroring the pipeline_training_run.py gates: ISAMBARD_FP32_SSM_STATE in
    #     {"0", "1", "checkpoint"} (default here: "checkpoint") and
    #     ISAMBARD_MAMBA_SAVE_OFFLOAD in {"0", "1"} (default here: "1").
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    _selftest_checkpoint_wrapper_mechanics()
    _selftest_save_offload_wrapper_mechanics()

    import os

    fp32_mode = os.environ.get("ISAMBARD_FP32_SSM_STATE", "checkpoint")
    offload_on = os.environ.get("ISAMBARD_MAMBA_SAVE_OFFLOAD", "1") == "1"

    if fp32_mode in ("1", "checkpoint"):
        checkpointed = fp32_mode == "checkpoint"
        print(f"fp32 install (mode={fp32_mode}): {patch_mamba_training_scan_fp32_state(checkpointed=checkpointed)}")
        print(f"fp32 second call (expect False): {patch_mamba_training_scan_fp32_state(checkpointed=checkpointed)}")
    if offload_on:
        print(f"offload install: {patch_mamba_training_scan_save_offload()}")
        print(f"offload second call (expect False): {patch_mamba_training_scan_save_offload()}")

    from megatron.core.ssm import mamba_mixer

    fn = mamba_mixer.mamba_split_conv1d_scan_combined
    chain, g = [], fn
    while g is not None:
        chain.append(getattr(g, "__name__", str(g)))
        g = getattr(g, "_wrapped", None)
    print(f"final callable chain (outer->inner): {' -> '.join(chain)}")
    print(f"offload patched: {getattr(fn, '_mamba_save_offload_patched', False)}")
    print(f"fp32 mode:       {getattr(fn, '_fp32_ssm_state_mode', None)}")

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
"""Patches of the Mamba2 training scan, ``megatron.core.ssm.mamba_mixer.mamba_split_conv1d_scan_combined``.

Catalogue:
    patch_mamba_training_scan_fp32_state: accumulate the scan's inter-chunk SSM state in fp32.
    patch_mamba_training_scan_save_offload: offload the scan's saved-for-backward tensors to pinned host memory.
"""

import logging


logger = logging.getLogger(__name__)


def _make_fp32_scan_wrapper(original, checkpointed: bool):
    """Build the fp32-state wrapper around the Mamba2 training scan op.

    Factored out of :func:`patch_mamba_training_scan_fp32_state` so the wrapper
    mechanics can be exercised on CPU against a fake op without importing
    megatron.core (tests/unit_tests/test_pipeline_training_patches.py).

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
    autograd Function (hook coverage is tested in tests/unit_tests/test_pipeline_training_patches.py) — is
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

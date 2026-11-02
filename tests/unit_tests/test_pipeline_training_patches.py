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
"""The Mamba2 training-scan wrappers: fp32 SSM state (direct and checkpointed) and saved-tensor host offload.

Each wrapper runs on CPU around a fake scan op whose autograd Function saves its inputs and an ``out_x`` stand-in
with ``ctx.save_for_backward``, as ``MambaSplitConv1dScanCombinedFn`` does: the real op is a Triton kernel, which
needs a GPU. The fake op takes a positional learnable weight, a closed-over Parameter, a tensor kwarg (``seq_idx``)
and returns a tuple, so the wrappers' handling of each is observable. The installers run against the real
``megatron.core.ssm.mamba_mixer``.
"""

import gc
import weakref

import pytest
import torch

from pipeline.training.patches import mamba_scan


class FakeScan:
    """A fake scan op recording its calls, the dtype and seq_idx it saw, and weakrefs to what it saved."""

    def __init__(self):
        self.calls = 0
        self.seen_dtype = None
        self.seen_seq_idx = None
        self.saved_refs = []
        self.save_ptrs = []
        self.backward_ptrs = []
        self.closure_weight = torch.nn.Parameter(torch.tensor(2.0))
        scan = self

        class _Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, conv_w):
                scan.calls += 1
                out_x = x * 3.0
                ctx.save_for_backward(x, out_x, conv_w)
                scan.saved_refs.append((weakref.ref(x), weakref.ref(out_x)))
                scan.save_ptrs.append((x.data_ptr(), out_x.data_ptr()))
                return out_x * conv_w

            @staticmethod
            def backward(ctx, g):
                x, out_x, conv_w = ctx.saved_tensors
                scan.backward_ptrs.append((x.data_ptr(), out_x.data_ptr()))
                return g * 3.0 * conv_w, (g * out_x).sum().reshape(conv_w.shape)

        self._fn = _Fn

    def __call__(self, zxbcdt, conv_w, **kwargs):
        self.seen_dtype = zxbcdt.dtype
        self.seen_seq_idx = kwargs.get("seq_idx")
        return (self._fn.apply(zxbcdt, conv_w) * self.closure_weight, "final_states_sentinel")


def _run(build_wrapper) -> dict:
    """One forward and backward through ``build_wrapper(scan)`` on a bf16 input; what the scan and autograd saw."""
    scan = FakeScan()
    torch.manual_seed(0)
    x = torch.randn(4, 8, dtype=torch.bfloat16, requires_grad=True)
    conv_w = torch.nn.Parameter(torch.tensor(1.5))
    y, extra = build_wrapper(scan)(x, conv_w, seq_idx=torch.tensor([0, 0, 1, 1]))
    gc.collect()
    first_save = scan.saved_refs[0]
    alive_after_forward = (first_save[0]() is not None, first_save[1]() is not None)
    calls_after_forward = scan.calls
    y.float().sum().backward()
    return {
        "out_dtype": y.dtype,
        "extra": extra,
        "seen_dtype": scan.seen_dtype,
        "seq_idx": scan.seen_seq_idx,
        "calls_forward": calls_after_forward,
        "calls": scan.calls,
        "saved_input_alive": alive_after_forward[0],
        "saved_out_x_alive": alive_after_forward[1],
        "same_ptrs_in_backward": scan.backward_ptrs[-1] == scan.save_ptrs[0],
        "grad_x": x.grad.clone(),
        "grad_conv_w": conv_w.grad.clone(),
        "grad_closure": scan.closure_weight.grad.clone(),
    }


@pytest.fixture(scope="module")
def runs():
    fp32 = mamba_scan._make_fp32_scan_wrapper
    offload = mamba_scan._make_save_offload_wrapper
    return {
        "plain": _run(lambda op: op),
        "fp32_direct": _run(lambda op: fp32(op, checkpointed=False)),
        "fp32_checkpoint": _run(lambda op: fp32(op, checkpointed=True)),
        "offload": _run(offload),
        "offload_fp32_direct": _run(lambda op: offload(fp32(op, checkpointed=False))),
        "offload_fp32_checkpoint": _run(lambda op: offload(fp32(op, checkpointed=True))),
    }


def test_every_variant_returns_bf16_and_passes_tuple_extras_through(runs):
    for name, run in runs.items():
        assert run["out_dtype"] == torch.bfloat16, name
        assert run["extra"] == "final_states_sentinel", name


@pytest.mark.parametrize("name", ["fp32_direct", "fp32_checkpoint", "offload_fp32_direct", "offload_fp32_checkpoint"])
def test_fp32_variants_run_the_scan_in_fp32_and_forward_seq_idx(runs, name):
    assert runs[name]["seen_dtype"] == torch.float32
    assert torch.equal(runs[name]["seq_idx"], torch.tensor([0, 0, 1, 1]))


def test_only_checkpointing_recomputes_the_scan_in_backward(runs):
    assert {name: (run["calls_forward"], run["calls"]) for name, run in runs.items()} == {
        "plain": (1, 1),
        "fp32_direct": (1, 1),
        "fp32_checkpoint": (1, 2),
        "offload": (1, 1),
        "offload_fp32_direct": (1, 1),
        "offload_fp32_checkpoint": (1, 2),
    }


def test_checkpointing_frees_the_fp32_saves_that_direct_mode_keeps(runs):
    assert runs["fp32_direct"]["saved_input_alive"] and runs["fp32_direct"]["saved_out_x_alive"]
    assert not runs["fp32_checkpoint"]["saved_input_alive"] and not runs["fp32_checkpoint"]["saved_out_x_alive"]


def test_offload_replaces_the_saved_tensors_and_frees_the_originals(runs):
    """save_on_cpu packs each saved tensor into a copy: backward reads other storage, and out_x is freed."""
    for name in ("offload", "offload_fp32_direct"):
        assert not runs[name]["same_ptrs_in_backward"], name
        assert not runs[name]["saved_out_x_alive"], name
    for name in ("plain", "fp32_direct"):
        assert runs[name]["same_ptrs_in_backward"], name
        assert runs[name]["saved_out_x_alive"], name


def test_gradients_reach_every_participating_tensor_and_agree_across_modes(runs):
    for name, run in runs.items():
        for key in ("grad_x", "grad_conv_w", "grad_closure"):
            assert run[key] is not None, (name, key)
    for key in ("grad_x", "grad_conv_w", "grad_closure"):
        torch.testing.assert_close(runs["fp32_direct"][key], runs["fp32_checkpoint"][key])
        assert torch.equal(runs["offload"][key], runs["plain"][key]), key
        assert torch.equal(runs["offload_fp32_direct"][key], runs["fp32_direct"][key]), key
        assert torch.equal(runs["offload_fp32_checkpoint"][key], runs["fp32_checkpoint"][key]), key


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda op: mamba_scan._make_fp32_scan_wrapper(op, checkpointed=True), id="fp32_checkpoint"),
        pytest.param(lambda op: mamba_scan._make_save_offload_wrapper(op), id="offload"),
    ],
)
def test_no_grad_calls_the_scan_once_and_builds_no_graph(build):
    scan = FakeScan()
    with torch.no_grad():
        y, _ = build(scan)(torch.randn(4, 8, dtype=torch.bfloat16), torch.tensor(1.5))
    assert (y.dtype, scan.calls, y.grad_fn) == (torch.bfloat16, 1, None)


def test_wrapper_markers_name_the_mode_and_the_wrapped_op():
    scan = FakeScan()
    direct = mamba_scan._make_fp32_scan_wrapper(scan, checkpointed=False)
    checkpointed = mamba_scan._make_fp32_scan_wrapper(scan, checkpointed=True)
    offloaded = mamba_scan._make_save_offload_wrapper(direct)
    assert (direct.__name__, direct._fp32_ssm_state_mode, direct._wrapped) == (
        "mamba_split_conv1d_scan_combined_fp32_state",
        "direct",
        scan,
    )
    assert (checkpointed.__name__, checkpointed._fp32_ssm_state_mode) == (
        "mamba_split_conv1d_scan_combined_fp32_state_checkpointed",
        "checkpoint",
    )
    assert offloaded._mamba_save_offload_patched and offloaded._wrapped is direct
    assert offloaded._fp32_ssm_state_patched and offloaded._fp32_ssm_state_mode == "direct"


@pytest.fixture
def mamba_mixer(monkeypatch):
    """The real mamba_mixer module, its scan restored after the test."""
    from megatron.core.ssm import mamba_mixer

    monkeypatch.setattr(mamba_mixer, "mamba_split_conv1d_scan_combined", mamba_mixer.mamba_split_conv1d_scan_combined)
    return mamba_mixer


def test_installers_are_idempotent_and_stack_offload_outermost(mamba_mixer):
    original = mamba_mixer.mamba_split_conv1d_scan_combined
    assert mamba_scan.patch_mamba_training_scan_fp32_state(checkpointed=False) is True
    assert mamba_scan.patch_mamba_training_scan_fp32_state(checkpointed=False) is False
    assert mamba_scan.patch_mamba_training_scan_fp32_state(checkpointed=True) is False
    assert mamba_scan.patch_mamba_training_scan_save_offload() is True
    assert mamba_scan.patch_mamba_training_scan_save_offload() is False
    installed = mamba_mixer.mamba_split_conv1d_scan_combined
    assert installed._mamba_save_offload_patched and installed._fp32_ssm_state_mode == "direct"
    assert installed._wrapped._wrapped is original


def test_offload_is_refused_under_the_checkpointed_fp32_mode(mamba_mixer):
    assert mamba_scan.patch_mamba_training_scan_fp32_state(checkpointed=True) is True
    checkpointed = mamba_mixer.mamba_split_conv1d_scan_combined
    assert mamba_scan.patch_mamba_training_scan_save_offload() is False
    assert mamba_mixer.mamba_split_conv1d_scan_combined is checkpointed

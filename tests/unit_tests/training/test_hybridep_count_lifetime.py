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
"""A HybridEP dispatch handle's dispatched-token count must outlive every kernel that reads it.

On the blocking path (no ``num_permuted_tokens``) deep_ep keeps the count in pinned host memory,
and its permute and unpermute kernels read it from device code when they start. PyTorch's caching
host allocator is told nothing about those reads, so once the handle is freed it can hand the block
out again while a kernel that reads it is still queued, and that kernel then runs on a foreign
count. Under the EP all-to-all overlap this crashed training: an illegal memory access in the
unpermute, whose count read 262,145 where a rank can receive at most 65,536 tokens.
``HybridEPDispatch.forward`` therefore hands out a handle whose count is a device copy.

Each test queues work behind a long ``torch.cuda._sleep`` on a side stream, frees what the queued
work reads, overwrites freshly allocated pinned blocks of the same size (as any later pinned
allocation might), and only then lets the stream run. A zeroed count makes the unpermute skip every
token, so the combine returns the dispatched tokens instead of the sum of their expert outputs. A
count past every token makes it read and write outside its buffers: a real illegal memory access,
after which the next dispatch's stream sync (whose error deep_ep does not check) returns at once and
its host read of the per-expert counts sees whatever the recycled pinned block held. That is the
second failure seen in training ("Trying to create tensor with negative dimension"), so that
sequence runs in a child process, whose CUDA context a fault would poison for good.

The device copy is the pinned submodule's carried commit 0004 (``3rdparty/patches/megatron-lm/README.md``).
All the tests need a GPU and deep_ep's HybridEP, which are environment boundaries.
"""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from megatron.core.transformer.moe import fused_a2a

from tests.unit_tests.one_rank_nccl_world import init_one_rank_nccl_world


requires_hybridep = pytest.mark.skipif(
    not (torch.cuda.is_available() and fused_a2a.HAVE_HYBRIDEP), reason="needs a GPU and deep_ep's HybridEP"
)


TOKENS, HIDDEN, EXPERTS, TOPK = 512, 1024, 8, 2
# About half a second at the GH200 clock: the host frees and overwrites long before the stream runs.
HOLD_CYCLES = 1_000_000_000
# Pinned int32 blocks allocated and overwritten after the free. The allocator hands out the most
# recently freed block of a size first, so the first one is enough; the rest is margin.
REUSE_ATTEMPTS = 64
# A dispatched-token count no rank can receive: an unpermute that reads it leaves its buffers.
PAST_EVERY_TOKEN = 2**31 - 1
REPO_ROOT = Path(__file__).resolve().parents[3]
# Prefix of the child process's verdict line; NCCL writes its version banner to the same stdout.
REPORT_TAG = "hybridep-count-lifetime-report "


@pytest.fixture(scope="module")
def group():
    yield init_one_rank_nccl_world()
    fused_a2a.reset_hybrid_ep_buffer()


def _inputs(seed=5):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=generator)
    scores = torch.rand(TOKENS, EXPERTS, device="cuda", generator=generator)
    chosen = scores.topk(TOPK, dim=-1).indices
    routing_map = torch.zeros(TOKENS, EXPERTS, dtype=torch.bool, device="cuda").scatter_(1, chosen, True)
    probs = torch.where(routing_map, scores, torch.zeros_like(scores))
    return hidden, routing_map, probs


def _expert(dispatched):
    """Stand-in experts whose summed outputs differ from the dispatched token itself."""
    return dispatched * 2 + 1


def _overwrite_new_pinned_counts(count):
    """Allocate distinct pinned int32 blocks and write ``count`` into each from the host."""
    return [torch.full((1,), count, dtype=torch.int32, pin_memory=True) for _ in range(REUSE_ATTEMPTS)]


def _addresses(blocks):
    return {block.data_ptr() for block in blocks}


def _dispatch(hidden, routing_map, probs, group):
    """Megatron's blocking HybridEP dispatch: (dispatched tokens, permuted row count, handle)."""
    dispatched, _, _, tokens_per_expert, handle = fused_a2a.hybrid_ep_dispatch(
        hidden, routing_map, probs, group, EXPERTS
    )
    return dispatched, int(tokens_per_expert.sum()), handle


@requires_hybridep
def test_a_combine_queued_behind_a_freed_handle_runs_on_its_own_count(group):
    hidden, routing_map, probs = _inputs()
    stream = torch.cuda.Stream()
    with torch.no_grad(), torch.cuda.stream(stream):
        dispatched, permuted, handle = _dispatch(hidden, routing_map, probs, group)
        reference = fused_a2a.hybrid_ep_combine(_expert(dispatched), handle, permuted, None)
        stream.synchronize()

        dispatched, permuted, handle = _dispatch(hidden, routing_map, probs, group)
        torch.cuda._sleep(HOLD_CYCLES)
        combined = fused_a2a.hybrid_ep_combine(_expert(dispatched), handle, permuted, None)
        del handle
    _overwrite_new_pinned_counts(0)
    stream.synchronize()

    assert torch.equal(combined, reference)


@requires_hybridep
def test_a_blocking_dispatch_hands_out_its_count_on_the_device(group):
    hidden, routing_map, probs = _inputs()
    with torch.no_grad():
        _, _, handle = _dispatch(hidden, routing_map, probs, group)
    # deep_ep's handle layout: (sparse_to_dense_map, rdma_to_attn_map, attn_to_rdma_map,
    # num_dispatched_tokens_tensor, ...); every token reaches the one rank of this world.
    count = handle[3]
    assert count.is_cuda
    assert int(count) == TOKENS


@requires_hybridep
def test_a_raw_deep_ep_handle_count_is_handed_out_again_once_freed(group):
    """The hazard itself, on deep_ep's own API: why HybridEPDispatch copies the count."""
    hidden, routing_map, probs = _inputs()
    with torch.no_grad():
        _dispatch(hidden, routing_map, probs, group)
    buffer = fused_a2a._hybrid_ep_buffer

    def raw_dispatch():
        dispatched, _, _, _, handle = buffer.dispatch_with_permute(
            hidden=hidden, routing_map=routing_map, probs=probs, num_of_experts_per_rank=EXPERTS
        )
        return dispatched, handle

    stream = torch.cuda.Stream()
    with torch.no_grad(), torch.cuda.stream(stream):
        dispatched, handle = raw_dispatch()
        reference, _ = buffer.combine_with_unpermute(hidden=_expert(dispatched), handle=handle)
        stream.synchronize()

        dispatched, handle = raw_dispatch()
        assert handle[3].is_pinned()
        count_block = handle[3].data_ptr()
        torch.cuda._sleep(HOLD_CYCLES)
        combined, _ = buffer.combine_with_unpermute(hidden=_expert(dispatched), handle=handle)
        del handle
    zeroed = _addresses(_overwrite_new_pinned_counts(0))
    stream.synchronize()

    assert count_block in zeroed
    assert not torch.equal(combined, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_host_allocator_holds_a_pinned_source_until_its_copy_has_run():
    """The property the device copy relies on: a non-blocking copy keeps its pinned source."""
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(HOLD_CYCLES)
        source = torch.full((1,), 7, dtype=torch.int32, pin_memory=True)
        source_block = source.data_ptr()
        copy = source.to("cuda", non_blocking=True)
        del source
    zeroed = _addresses(_overwrite_new_pinned_counts(0))

    assert source_block not in zeroed
    stream.synchronize()
    assert int(copy) == 7


def _replay_training_sequence():
    """Child-process body: the training sequence that faulted, through Megatron's HybridEP API.

    A combine queued behind a freed handle whose count block is overwritten with a count past every
    token, then the next forward dispatch on the same stream. Prints one tagged JSON line of verdicts.
    """
    group = init_one_rank_nccl_world()
    hidden, routing_map, probs = _inputs()
    stream = torch.cuda.Stream()
    with torch.no_grad(), torch.cuda.stream(stream):
        reference_dispatched, reference_permuted, handle = _dispatch(hidden, routing_map, probs, group)
        reference = fused_a2a.hybrid_ep_combine(_expert(reference_dispatched), handle, reference_permuted, None)
        stream.synchronize()

        dispatched, permuted, handle = _dispatch(hidden, routing_map, probs, group)
        torch.cuda._sleep(HOLD_CYCLES)
        combined = fused_a2a.hybrid_ep_combine(_expert(dispatched), handle, permuted, None)
        del handle
        overwritten = _overwrite_new_pinned_counts(PAST_EVERY_TOKEN)
        next_dispatched, next_permuted, _ = _dispatch(hidden, routing_map, probs, group)
    stream.synchronize()
    del overwritten
    verdicts = {
        "combine_matches": torch.equal(combined, reference),
        "next_dispatch_sized_from_its_own_counts": next_permuted == reference_permuted,
        "next_dispatch_matches": torch.equal(next_dispatched, reference_dispatched),
    }
    print(REPORT_TAG + json.dumps(verdicts), flush=True)


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@requires_hybridep
@pytest.mark.serial_gpu
def test_the_next_forward_dispatch_after_a_combine_on_a_freed_handle_runs_cleanly():
    env = dict(
        os.environ,
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(_free_port()),
        PYTHONPATH=os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")])),
    )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.unit_tests.training.test_hybridep_count_lifetime import _replay_training_sequence;"
            " _replay_training_sequence()",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    output = f"stdout:\n{child.stdout[-3000:]}\nstderr:\n{child.stderr[-6000:]}"
    assert child.returncode == 0, output
    reports = [line[len(REPORT_TAG) :] for line in child.stdout.splitlines() if line.startswith(REPORT_TAG)]
    assert len(reports) == 1, output
    assert json.loads(reports[0]) == {
        "combine_matches": True,
        "next_dispatch_sized_from_its_own_counts": True,
        "next_dispatch_matches": True,
    }

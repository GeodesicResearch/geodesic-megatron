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
"""Per-expert LoRA under Megatron DDP with ``overlap_param_gather``: no stale adapter reads, any expert-DP size.

DDP waits for a module's parameter all-gather in a forward pre-hook, which fires only when the module is called. The
per-expert factors are sharded by the distributed optimizer over the expert-data-parallel group, so above expert-DP 1
a forward that reads them without that wait reads the other replicas' shards from before the last optimizer step.
"""

from pathlib import Path

import pytest
import torch

from tests.unit_tests.child_process import free_port, run_python


WORKER = Path(__file__).with_name("grouped_experts_lora_ddp_worker.py")
GLOBAL_BATCH, STEPS = 8, 4

pytestmark = [
    pytest.mark.serial_gpu,
    pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs (EP2 at expert-DP 2)"),
    pytest.mark.skipif(not hasattr(torch, "_grouped_mm"), reason="needs torch._grouped_mm"),
]


def _train(tmp_path: Path, gpus: int) -> dict:
    """``STEPS`` training steps of the worker at EP2 on ``gpus`` GPUs (expert-DP ``gpus // 2``)."""
    out = tmp_path / f"run_{gpus}.pt"
    run_python(
        [
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={gpus}",
            f"--master_port={free_port()}",
            str(WORKER),
            "--ep=2",
            f"--global-batch={GLOBAL_BATCH}",
            f"--steps={STEPS}",
            f"--out={out}",
        ],
        timeout=900,
        cwd=tmp_path,
        CUDA_VISIBLE_DEVICES=",".join(str(i) for i in range(gpus)),
    )
    return torch.load(out)


def test_expert_dp_2_reads_no_stale_adapter_and_trains_as_expert_dp_1(tmp_path):
    """EP2 on 4 GPUs (expert-DP 2) against EP2 on 2 GPUs (expert-DP 1), same model, batch and steps."""
    reference, sharded = _train(tmp_path, gpus=2), _train(tmp_path, gpus=4)
    assert sharded["stale_reads"] == 0 and reference["stale_reads"] == 0, (
        sharded["stale_reads"],
        reference["stale_reads"],
    )
    torch.testing.assert_close(sharded["losses"], reference["losses"], rtol=1e-5, atol=0)
    assert sharded["adapters"].keys() == reference["adapters"].keys()
    # At each tensor's own scale: Adam turns reduction-order noise in near-zero gradient entries into steps of the
    # learning rate, so elementwise tolerances would fail on noise. That noise stays well below 1e-3 of a tensor's
    # scale, and stale per-expert factors move the adapters well past it.
    relative = {
        name: ((sharded["adapters"][name] - value).abs().max() / value.abs().max()).item()
        for name, value in reference["adapters"].items()
    }
    assert max(relative.values()) < 1e-3, sorted(relative.items(), key=lambda item: -item[1])[:5]

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
"""A PEFT target pattern must match a module on some pipeline stage, not on every stage."""

import json
from pathlib import Path

import pytest
import torch

from tests.unit_tests.child_process import free_port, run_python


WORKER = Path(__file__).with_name("peft_targets_pp_worker.py")

pytestmark = [
    pytest.mark.serial_gpu,
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs (two pipeline stages)"),
]


def test_targets_are_matched_over_the_pipeline_stages(tmp_path):
    from tests.unit_tests.peft.peft_targets_pp_worker import REPORT_TAG

    child = run_python(
        ["-m", "torch.distributed.run", "--nproc_per_node=2", f"--master_port={free_port()}", str(WORKER)],
        timeout=600,
        cwd=tmp_path,
        CUDA_VISIBLE_DEVICES="0,1",
    )
    reports = [line[len(REPORT_TAG) :] for line in child.stdout.splitlines() if line.startswith(REPORT_TAG)]
    assert len(reports) == 1, child.stdout[-3000:]
    report = json.loads(reports[0])
    assert report["one_stage_each"] is None
    assert report["no_stage"] == "LoRA target patterns ['*.layers.9.*.out_proj'] match no module of the model"

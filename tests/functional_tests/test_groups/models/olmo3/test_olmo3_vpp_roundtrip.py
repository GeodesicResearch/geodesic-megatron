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
"""GPU test: the toy OLMo-3 checkpoint round-trips tensor-exact through a model at PP 4 x VPP 2.

See ``olmo3_vpp_roundtrip.py`` for what is checked and why virtual pipeline stages need their own test.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch


HARNESS = Path(__file__).with_name("olmo3_vpp_roundtrip.py")

pytestmark = [
    pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four GPUs"),
    pytest.mark.run_only_on("GPU"),
]


def test_toy_checkpoint_round_trips_tensor_exact_at_pp4_vpp2(tmp_path):
    sys.path.insert(0, str(HARNESS.parent))
    from olmo3_parity import build_toy_model

    model_dir = tmp_path / "toy"
    model_dir.mkdir()
    build_toy_model(str(model_dir))
    report_path = tmp_path / "report.json"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
        str(HARNESS),
        "--model-dir",
        str(model_dir),
        "--report",
        str(report_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-3000:]
    report = json.loads(report_path.read_text())
    assert report["ok"]
    # Interleaving: each rank holds one layer of each chunk, r and r + 4 (a plain split would give it r*2 and r*2 + 1).
    assert report["layers_per_rank"] == [[0, 4], [1, 5], [2, 6], [3, 7]]
    assert report["original_tensors"] == report["exported_tensors"] == 8 * 11 + 3

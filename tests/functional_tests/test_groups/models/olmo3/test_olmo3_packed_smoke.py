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
"""GPU smoke of the OLMo-3 family on packed (THD) batches: backend, finiteness and context-parallel agreement.

``olmo3_packed_smoke.py`` runs four layers of the 7B architecture on a synthetic 32,768-token pack of padded documents
at one context-parallel size. This module runs it at size 1 and, with two GPUs, at size 2, and asserts what training
on packed data relies on:

* Transformer Engine runs the packed batch, the 4,095-token window and the padded ``cu_seqlens`` on cuDNN fused
  attention (it disables FlashAttention for padding between sequences);
* the loss and every gradient are finite;
* context parallel size 2 with the ``a2a`` exchange gives the loss and gradients of size 1.

The tolerances below are about ten times the differences between the two sizes on a GH200 node: equal losses to all
printed digits, global gradient norms within 4e-5 in relative terms, and per-parameter differences within 4e-4.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from pipeline.training.recipes import OLMO3_BASE_COPIES


HARNESS = Path(__file__).with_name("olmo3_packed_smoke.py")
BASE_CONFIG_DIR = Path(OLMO3_BASE_COPIES["olmo3_7b"])

LOSS_ABSOLUTE_TOLERANCE = 1e-4
GRAD_NORM_RELATIVE_TOLERANCE = 5e-4
PARAMETER_GRAD_NORM_RELATIVE_TOLERANCE = 4e-3

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU"),
    pytest.mark.skipif(not BASE_CONFIG_DIR.is_dir(), reason=f"requires the 7B base copy at {BASE_CONFIG_DIR}"),
    pytest.mark.run_only_on("GPU"),
]


def run_smoke(report: Path, cp: int, fill: float) -> dict:
    """Run the harness under torchrun at context-parallel size ``cp`` and return its report."""
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={cp}",
        str(HARNESS),
        "--base-config-dir",
        str(BASE_CONFIG_DIR),
        "--cp",
        str(cp),
        "--layers",
        "4",
        "--seq-len",
        "32768",
        "--pad-mult",
        "8",
        "--fill",
        str(fill),
        "--seed",
        "0",
        "--iterations",
        "2",
        "--report",
        str(report),
    ]
    # Transformer Engine reads both variables when it is imported, and it logs its backend decisions only then.
    environment = {**os.environ, "NVTE_DEBUG": "1", "NVTE_DEBUG_LEVEL": "2"}
    completed = subprocess.run(command, capture_output=True, text=True, timeout=1800, env=environment)
    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-3000:]
    return json.loads(report.read_text())


def assert_runs_on_fused_attention(report: dict) -> None:
    messages = report["te_attention_messages"]
    selected = {message for message in messages if "Selected backend" in message}
    assert selected == {"Selected backend = FusedAttention (sub-backend 1)"}, selected
    assert any("Disabling FlashAttention for qkv_format = thd when there is padding" in m for m in messages)


@pytest.mark.parametrize("fill", [1.0, 0.95], ids=["full_pack", "pack_with_trailing_padding"])
def test_packed_batch_runs_on_fused_attention_with_finite_loss_and_gradients(tmp_path, fill):
    report = run_smoke(tmp_path / "cp1.json", cp=1, fill=fill)
    assert report["finite"]
    assert_runs_on_fused_attention(report)
    assert len(report["grad_norms"]) == 35  # 4 layers of 8 tensors, the embedding, the final norm and the output layer
    assert all(norm > 0 for norm in report["grad_norms"].values())


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
@pytest.mark.parametrize("fill", [1.0, 0.95], ids=["full_pack", "pack_with_trailing_padding"])
def test_context_parallel_size_two_with_a2a_gives_the_loss_and_gradients_of_size_one(tmp_path, fill):
    one = run_smoke(tmp_path / "cp1.json", cp=1, fill=fill)
    two = run_smoke(tmp_path / "cp2.json", cp=2, fill=fill)
    assert two["finite"]
    assert_runs_on_fused_attention(two)
    # Same seed and same layout give the same weights, so the two runs compare like for like.
    assert one["parameter_checksum"] == two["parameter_checksum"]
    assert two["iterations"][-1]["local_tokens"] * 2 == one["iterations"][-1]["local_tokens"]
    assert abs(one["iterations"][-1]["loss"] - two["iterations"][-1]["loss"]) < LOSS_ABSOLUTE_TOLERANCE
    grad_norm_one, grad_norm_two = one["iterations"][-1]["grad_norm"], two["iterations"][-1]["grad_norm"]
    assert abs(grad_norm_one - grad_norm_two) / grad_norm_one < GRAD_NORM_RELATIVE_TOLERANCE
    assert one["grad_norms"].keys() == two["grad_norms"].keys()
    for name, norm in one["grad_norms"].items():
        assert abs(norm - two["grad_norms"][name]) / norm < PARAMETER_GRAD_NORM_RELATIVE_TOLERANCE, name

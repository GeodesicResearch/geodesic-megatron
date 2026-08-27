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
"""GPU conversion tests for the OLMo-3 bridge.

Structure mirrors ``tests/functional_tests/test_groups/models/olmoe``. What is
different here is the **negative controls**: OLMo-3's two distinguishing features
(interleaved sliding-window attention, and RoPE scaling on full-attention layers
only) are invisible both in the weights and in a short prompt, so this file also
asserts that the parity checks *fail* when those features are deliberately broken.
Without that, a green run would only show the harness ran, not that it can detect
anything.

Measured on the toy model (fp32, all six correct cells across seq 48..768):
``max_abs_diff ~= 0.0017``, ``cosine >= 0.9999996``, ``top1 == 1.0``; the injected
faults sit at ``max_abs_diff 1.9..2.3``, ``cosine 0.62..0.66`` -- three orders of
magnitude of separation, which is what makes the bar meaningful.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import torch


HARNESS = Path(__file__).with_name("olmo3_parity.py")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


@pytest.fixture(scope="module")
def toy_model(tmp_path_factory) -> str:
    """Build the toy OLMo-3 checkpoint once for the module."""
    sys.path.insert(0, str(HARNESS.parent))
    from olmo3_parity import build_toy_model

    out = tmp_path_factory.mktemp("olmo3_toy") / "model"
    out.mkdir(parents=True, exist_ok=True)
    return build_toy_model(str(out))


def _run(model_dir: str, *args: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(HARNESS), "--model-dir", model_dir, *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=3600)


@pytest.mark.run_only_on("GPU")
class TestOlmo3Conversion:
    def test_weights_roundtrip_exactly(self, toy_model):
        """Every one of the 91 tensors must survive HF -> Megatron -> HF."""
        r = _run(toy_model, "--weights-only", "--dtype", "float32")
        assert "VERDICT OK" in r.stdout, r.stdout[-4000:] + r.stderr[-2000:]

    @pytest.mark.parametrize("cell", ["short", "mid", "long"])
    def test_forward_parity(self, toy_model, cell):
        """Logit parity at three lengths.

        ``short`` is below ``sliding_window``; ``mid`` is above it; ``long`` is also
        above ``original_max_position_embeddings``. A single short prompt cannot
        falsify either of OLMo-3's distinguishing features, which is why all three
        run.
        """
        r = _run(toy_model, "--cell", cell, "--dtype", "float32", "--layer-report")
        assert "VERDICT OK" in r.stdout, r.stdout[-4000:] + r.stderr[-2000:]

    @pytest.mark.parametrize(
        "fault,cell",
        [
            ("swap_gate_up", "short"),  # weight layout
            ("per_head_qknorm", "short"),  # full-width QK-norm semantics
            ("no_swa", "mid"),  # interleaved sliding-window attention
            ("yarn_all_layers", "long"),  # RoPE scaling scope (the transformers 5.2 bug)
        ],
    )
    def test_negative_control_fires(self, toy_model, fault, cell):
        """Break one thing; the corresponding cell must notice.

        A check that cannot fail is not evidence, and these four faults are exactly
        the ones a weights-only or short-prompt validation would wave through.
        """
        r = _run(toy_model, "--cell", cell, "--fault", fault, "--dtype", "float32", "--expect-fail")
        assert "VERDICT OK" in r.stdout, (
            f"fault={fault} at cell={cell} did NOT change the logits, so that cell is "
            f"not actually testing it.\n" + r.stdout[-4000:]
        )

    def test_short_cell_is_blind_to_sliding_window(self, toy_model):
        """Documents *why* the mid cell exists.

        Below ``sliding_window`` the mask is equivalent to plain causal, so disabling
        SWA changes nothing. If this ever starts failing, the length cells no longer
        mean what this file claims they mean.
        """
        r = _run(toy_model, "--cell", "short", "--fault", "no_swa", "--dtype", "float32")
        assert "VERDICT OK" in r.stdout, r.stdout[-4000:]

    def test_differs_from_transformers_520_regression(self, toy_model):
        """transformers 5.2.0 applies YaRN to every layer; the bridge must not.

        Guards against someone 'fixing' a future parity failure by matching the
        regression instead of the architecture.
        """
        r = _run(
            toy_model, "--cell", "long", "--reference", "stock", "--dtype", "float32", "--expect-fail"
        )
        assert "VERDICT OK" in r.stdout, r.stdout[-4000:]

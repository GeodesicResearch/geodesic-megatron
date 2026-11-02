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
"""pipeline/megatron_model.py: the bridge, provider and load calls for each dtype mode, through a recording AutoBridge
(tests/unit_tests/recorded_bridge.py says why it is recorded)."""

import torch

from pipeline.megatron_model import load_sharded, sharded_bridge
from tests.unit_tests import recorded_bridge


def test_without_params_dtype_the_model_loads_in_the_releases_dtype(monkeypatch):
    trace = recorded_bridge.install(monkeypatch)
    bridge, overrides = sharded_bridge("org/model", 2, 3, 4, 1, pipeline_dtype=torch.bfloat16, trust_remote_code=True)
    assert trace.calls[0] == ["AutoBridge.from_hf_pretrained", ["'org/model'"], {"trust_remote_code": "True"}]
    assert ["provider.params_dtype =", "torch.bfloat16"] not in trace.calls
    assert overrides == {
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 3,
        "expert_model_parallel_size": 4,
        "expert_tensor_parallel_size": 1,
        "pipeline_dtype": torch.bfloat16,
    }
    assert trace.calls[-1] == ["provider.initialize_model_parallel", [], {"seed": "0"}]
    chunks = load_sharded(bridge, "/ckpt/run", overrides)
    assert [repr(c) for c in chunks] == ["chunk0.cuda()"]
    assert trace.calls[-2][:2] == ["bridge.load_megatron_model", ["'/ckpt/run'"]]
    assert trace.calls[-2][2]["wrap_with_ddp"] == "False"


def test_with_params_dtype_the_release_loads_and_the_model_builds_in_it(monkeypatch):
    trace = recorded_bridge.install(monkeypatch)
    _, overrides = sharded_bridge(
        "org/model", 1, 1, 4, 1, pipeline_dtype=torch.float32, trust_remote_code=False, params_dtype=torch.float32
    )
    assert trace.calls[0][2] == {"trust_remote_code": "False", "torch_dtype": "torch.float32"}
    assert ["provider.params_dtype =", "torch.float32"] in trace.calls
    assert overrides["params_dtype"] == overrides["pipeline_dtype"] == torch.float32

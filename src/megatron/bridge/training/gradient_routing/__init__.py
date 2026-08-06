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
"""GRAM gradient routing: iteration-level routing of forget-corpus updates into aux modules.

The package splits by responsibility:

- ``plan``: the deterministic per-iteration routing plan (corpus + forward/update sets).
- ``config``: the YAML-facing configuration objects.
- ``optimizer_gating``: per-iteration param-group gating (which sets step) and the aux
  learning-rate param-group override.
- ``callback``: drives the per-iteration state (gates, frozen expert bias, gating roles)
  and emits telemetry.

Model surgery lives in ``megatron.bridge.models.mamba.gram_layer``; the routed dataset in
``megatron.bridge.data.datasets.gr_routed_dataset``.
"""

from megatron.bridge.training.gradient_routing.config import GradientRoutingConfig, GRDatasetConfig
from megatron.bridge.training.gradient_routing.plan import GRPlan


__all__ = ["GRPlan", "GradientRoutingConfig", "GRDatasetConfig"]

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
"""The pieces of ``pipeline_training_run.py``, the training entry point every rank runs.

Catalogue:
    recipes: the recipe of each model and mode, and a launch's config resolved over it.
    one_pass: sizing a packed SFT run to one pass over its rows.
    resilience: the fault-tolerance and straggler-detection configs of a launch.
    env_overrides: the per-node echo of the launcher's environment overrides.
    datasets: the training data a launch reads, by mode.
    patches: runtime patches of megatron.core and the packed dataset, installed by environment gates.
"""

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
"""Runtime patches applied before training starts, each installed by an environment gate of
``pipeline_training_run.main()`` on every rank.

They patch ``megatron.core`` (imported from the pinned ``3rdparty/Megatron-LM`` submodule) and the packed
dataset without touching their source. Each installer is idempotent and returns whether it installed.

Catalogue:
    mamba_scan: fp32 SSM state and saved-tensor host offload for the Mamba2 training scan.
    comm_warmup: NCCL communicators of every model-parallel group initialised in one parallel wave.
    row_telemetry: a log line per packed-parquet row fetch, mapping dataset index to parquet row.
"""

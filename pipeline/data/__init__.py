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
"""The stages of ``pipeline_data_prepare.py``, which prepares a dataset for Megatron Bridge training.

Catalogue:
    layout: where a prepared dataset and its packed parquet live on disk (standard library only).
    load: the source dataset, from the Hub or from local files.
    records: the text column, the format, and the JSONL records Megatron Bridge reads.
    count: token counts through the pack path's chat rendering.
    packing: the subprocess that packs a dataset root's JSONL into parquet.
    verify: the loss-mask report of a packed parquet.
"""

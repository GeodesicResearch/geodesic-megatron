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
"""Megatron checkpoint -> HuggingFace export, for pipeline_checkpoint_convert_hf.py.

Catalogue:
    hf: the conversion itself, single-process or over a torchrun world.
    peft_merge: the export of an adapter-only PEFT model with its adapters merged, and its check.
    fixups: the fixes applied to a converted export.
    lora_factors: a LoRA checkpoint's adapter as a vLLM factor dir.
"""

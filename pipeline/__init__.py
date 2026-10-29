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
"""Shared code of the top-level ``pipeline_*`` scripts.

The ``pipeline_*`` scripts sit at the repo root, and Python puts a script's own directory first on ``sys.path``, so
``import pipeline`` resolves to this directory from them, as it does for ``python -m pipeline.<module>`` run from the
repo root. Code elsewhere (under ``scripts/`` or ``configs/``) puts the
repo root on ``sys.path`` itself before importing it. Modules that shell scripts or host-side code call import only the
standard library, ``yaml`` and ``huggingface_hub`` at module scope; torch and megatron imports stay inside functions.

Catalogue:
    checkpoints: the layout of a Megatron checkpoint directory.
    hf_cache: lookups in the local HuggingFace hub cache.
    hub: publishing a model directory to the HuggingFace Hub.
    nemotron: facts about the Nemotron-H releases and their chat format.
    megatron_model: a sharded Megatron model built from an HF architecture, and a checkpoint loaded into it.
    export: Megatron checkpoint -> HuggingFace export (pipeline_checkpoint_convert_hf.py).
    coherence: the generation backends of pipeline_coherence_test.py.
    environment: checks of the pipeline container.
    sh: the shell libraries the launchers source; each file catalogues its functions.
"""

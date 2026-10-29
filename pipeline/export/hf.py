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
"""Megatron checkpoint -> HuggingFace model directory, through megatron.bridge.

Catalogue:
    is_multi_gpu: whether this process runs under torchrun.
    convert_single_process: export in one process (CPU-based distributed context).
    convert_multi_gpu: export from a model sharded over the torchrun world.
"""

import os
from pathlib import Path

import torch

from pipeline.checkpoints import is_peft_checkpoint
from pipeline.export.peft_merge import export_merged_peft


def is_multi_gpu() -> bool:
    """Check if running under torchrun (multi-GPU mode)."""
    return os.environ.get("WORLD_SIZE") is not None


def convert_single_process(
    iter_path: Path,
    hf_path: Path,
    hf_model_id: str,
    strict: bool = True,
    show_progress: bool = True,
) -> None:
    """Convert using single-process export_ckpt (CPU-based distributed context)."""
    from megatron.bridge import AutoBridge

    print(f"Creating bridge from auto-config: {hf_model_id}")
    bridge = AutoBridge.from_auto_config(str(iter_path), hf_model_id)

    print(f"Exporting: {iter_path} -> {hf_path}")
    bridge.export_ckpt(
        megatron_path=str(iter_path),
        hf_path=str(hf_path),
        show_progress=show_progress,
        strict=strict,
    )
    print(f"Export complete: {hf_path}")


def convert_multi_gpu(
    iter_path: Path,
    hf_path: Path,
    hf_model_id: str,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    etp: int = 1,
    torch_dtype: torch.dtype = torch.bfloat16,
    strict: bool = True,
    show_progress: bool = True,
) -> None:
    """Convert using multi-GPU distributed loading.

    An adapter-only PEFT checkpoint (``is_peft_checkpoint``) is loaded with its base (``load_peft_model``) and exported
    with its adapters merged (``export_merged_peft``).
    """
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.decorators import torchrun_main
    from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
    from megatron.bridge.training.model_load_save import load_peft_model
    from megatron.bridge.utils.common_utils import print_rank_0

    @torchrun_main
    def _run():
        print_rank_0(f"Exporting: {iter_path} -> {hf_path}")
        print_rank_0(f"  TP={tp}  PP={pp}  EP={ep}  ETP={etp}  dtype={torch_dtype}")

        bridge = AutoBridge.from_hf_pretrained(
            hf_model_id,
            trust_remote_code=is_safe_repo(trust_remote_code=False, hf_path=hf_model_id),
            torch_dtype=torch_dtype,
        )

        model_provider = bridge.to_megatron_provider(load_weights=False)
        model_provider.tensor_model_parallel_size = tp
        model_provider.pipeline_model_parallel_size = pp
        model_provider.expert_model_parallel_size = ep
        model_provider.expert_tensor_parallel_size = etp
        model_provider.pipeline_dtype = torch_dtype
        model_provider.params_dtype = torch_dtype
        model_provider.finalize()
        model_provider.initialize_model_parallel(seed=0)

        mp_overrides = {
            "tensor_model_parallel_size": tp,
            "pipeline_model_parallel_size": pp,
            "expert_model_parallel_size": ep,
            "expert_tensor_parallel_size": etp,
            "pipeline_dtype": torch_dtype,
            "params_dtype": torch_dtype,
        }

        if is_peft_checkpoint(iter_path):
            print_rank_0(f"Loading PEFT checkpoint and its base: {iter_path}")
            megatron_model = load_peft_model(str(iter_path), mp_overrides=mp_overrides)
            print_rank_0(f"Saving HuggingFace model, adapters merged, to: {hf_path}")
            export_merged_peft(bridge, megatron_model, hf_path, strict=strict, show_progress=show_progress)
        else:
            print_rank_0(f"Loading Megatron checkpoint from: {iter_path}")
            megatron_model = bridge.load_megatron_model(
                str(iter_path),
                mp_overrides=mp_overrides,
                wrap_with_ddp=False,
            )
            megatron_model = [m.cuda() for m in megatron_model]

            print_rank_0(f"Saving HuggingFace model to: {hf_path}")
            bridge.save_hf_pretrained(
                megatron_model,
                str(hf_path),
                show_progress=show_progress,
                strict=strict,
            )
        print_rank_0(f"Export complete: {hf_path}")

    _run()

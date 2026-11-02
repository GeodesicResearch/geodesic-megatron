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
"""PEFT target matching over two pipeline stages (torchrun worker, 2 GPUs).

Run by ``test_peft_targets_pp.py``. The small Nemotron-H ``ME|*E`` puts its Mamba layer (``out_proj``) on the first
stage and its attention layer (``linear_proj``) on the second. LoRA is applied twice: with targets that each match on
one stage only, and with a target that matches on no stage. Rank 0 prints one ``REPORT_TAG`` line: whether each
application raised, and the message.
"""

import json
import os

import torch
import torch.distributed as dist

from tests.unit_tests.small_nemotron_h import small_nemotron_h_provider


REPORT_TAG = "peft-targets-pp-report "


def _apply(target_modules: list[str]) -> str | None:
    """LoRA on ``target_modules`` applied to this rank's stage: None, or the ValueError's message."""
    from megatron.bridge.peft.lora import LoRA

    provider = small_nemotron_h_provider(
        "ME|*E", bf16=True, params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16, pipeline_model_parallel_size=2
    )
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    try:
        LoRA(target_modules=target_modules)(model, training=False)
    except ValueError as error:  # the outcome under test, reported to the driver
        return str(error)
    return None


def main() -> None:
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")
    parallel_state.initialize_model_parallel(pipeline_model_parallel_size=2)
    model_parallel_cuda_manual_seed(1234)
    report = {
        "one_stage_each": _apply(["linear_proj", "out_proj"]),
        "no_stage": _apply(["linear_proj", "*.layers.9.*.out_proj"]),
    }
    if dist.get_rank() == 0:
        print(REPORT_TAG + json.dumps(report), flush=True)
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

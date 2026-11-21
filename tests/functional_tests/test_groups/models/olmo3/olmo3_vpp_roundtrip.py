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
"""Round trip of a toy OLMo-3 checkpoint through a model built at pipeline parallel size 4 with 2 virtual stages.

With interleaved (virtual) pipeline parallelism a rank holds several model chunks, and the layers of a chunk are not
contiguous: at 8 layers, PP 4 and VPP 2, rank r holds layers r and r + 4. OLMo-3 decides per layer whether it slides
(the 3:1 pattern) and which rope it reads from the layer's global ``layer_number``, so a wrong offset would give a layer
the wrong window or the wrong rope without any shape error. This harness builds the model, checks that every layer's
``is_sliding`` is the Hugging Face config's ``layer_types`` entry for its global index, loads the toy weights through the
bridge, exports them again, and compares every exported tensor bitwise with the file it came from.

Launch with ``torchrun --nproc_per_node=4``. Rank 0 writes the JSON report.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core.utils import unwrap_model
from safetensors.torch import load_file
from transformers import AutoConfig

from megatron.bridge import AutoBridge


PIPELINE_SIZE = 4
VIRTUAL_STAGES = 2


def read_original_tensors(model_dir: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        tensors |= load_file(str(shard))
    return tensors


def layer_flags(model_chunks) -> list[tuple[int, bool]]:
    """(global layer index, whether the layer slides) for every layer on this rank, over all its chunks."""
    flags = []
    for chunk in model_chunks:
        for layer in unwrap_model(chunk).decoder.layers:
            flags.append((layer.layer_number - 1, layer.self_attention.is_sliding))
    return flags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", required=True, help="a Hugging Face directory of the toy checkpoint")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    model_dir = Path(args.model_dir)

    # The toy's config.json is written as AI2 ships configs, without layer_types, which the config class derives.
    layer_types = AutoConfig.from_pretrained(str(model_dir)).layer_types
    bridge = AutoBridge.from_hf_pretrained(str(model_dir))
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.pipeline_model_parallel_size = PIPELINE_SIZE
    provider.virtual_pipeline_model_parallel_size = VIRTUAL_STAGES
    provider.params_dtype = torch.bfloat16
    provider.pipeline_dtype = torch.bfloat16
    provider.bf16 = True
    provider.gradient_accumulation_fusion = False
    provider.finalize()
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    if len(model) != VIRTUAL_STAGES:
        raise RuntimeError(f"expected {VIRTUAL_STAGES} model chunks per rank, found {len(model)}")

    flags = layer_flags(model)
    wrong = [(index, sliding) for index, sliding in flags if sliding != (layer_types[index] == "sliding_attention")]
    gathered_flags: list = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_flags, flags)
    covered = sorted(index for rank_flags in gathered_flags for index, _ in rank_flags)

    bridge.load_hf_weights(model)
    original = read_original_tensors(model_dir)
    exported = {name: tensor for name, tensor in bridge.export_hf_weights(model, cpu=True, show_progress=False)}
    different = sorted(
        name for name in exported if name in original and not torch.equal(exported[name], original[name])
    )
    report = {
        "world_size": dist.get_world_size(),
        "layers_per_rank": [[index for index, _ in rank_flags] for rank_flags in gathered_flags],
        "layers_covered_once": covered == list(range(len(layer_types))),
        "wrong_sliding_flags_on_rank": wrong,
        "exported_tensors": len(exported),
        "original_tensors": len(original),
        "only_in_export": sorted(set(exported) - set(original)),
        "only_in_original": sorted(set(original) - set(exported)),
        "different": different,
    }
    ok = (
        report["layers_covered_once"]
        and not wrong
        and not report["only_in_export"]
        and not report["only_in_original"]
        and not different
    )
    gathered_ok: list = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_ok, bool(ok))
    if dist.get_rank() == 0:
        report["ok"] = all(gathered_ok)
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"[olmo3-vpp-roundtrip] ok={report['ok']} layers_per_rank={report['layers_per_rank']}")
    return 0 if all(gathered_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())

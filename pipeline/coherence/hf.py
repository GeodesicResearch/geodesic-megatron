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
"""The coherence test's ``hf`` backend: transformers' text-generation pipeline on this node's GPUs.

Catalogue:
    generate_hf: one generation per prompt.
"""

from pipeline.coherence import build_chat_messages


def generate_hf(args, prompts) -> list[str]:
    """One generation per prompt via the transformers text-generation pipeline."""
    import torch
    from transformers import pipeline

    device = "cuda:0" if torch.cuda.device_count() == 1 else None
    device_map = "auto" if device is None else None
    pipeline_kwargs = dict(device=device, device_map=device_map, torch_dtype=torch.bfloat16)
    if args.revision:
        pipeline_kwargs["revision"] = args.revision
    llm = pipeline("text-generation", args.model_path, **pipeline_kwargs)

    gens = []
    for prompt in prompts:
        if args.generation_mode == "chat":
            llm_input = build_chat_messages(prompt, args.system_prompt)
        else:
            llm_input = prompt
        # temperature <= 0 means greedy decoding; transformers rejects
        # do_sample=True with a non-positive temperature.
        if args.temperature > 0:
            gen_kwargs = dict(do_sample=True, temperature=args.temperature)
        else:
            gen_kwargs = dict(do_sample=False)
        out = llm(llm_input, max_new_tokens=args.max_tokens, **gen_kwargs)
        if args.generation_mode == "chat":
            gen = out[0]["generated_text"][-1]["content"].strip()
        else:
            full = out[0]["generated_text"]
            gen = full[len(prompt) :].strip() if full.startswith(prompt) else full.strip()
        gens.append(gen)
    return gens

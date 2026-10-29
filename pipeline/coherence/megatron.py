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
"""The coherence test's ``megatron`` backend: greedy decoding from a Megatron checkpoint, with no HF export.

Runs under torchrun across the inference parallelism (GPUs, collective).

Catalogue:
    generate_megatron: one greedy generation per prompt.
"""

from pipeline.coherence import build_chat_messages
from pipeline.megatron_model import load_sharded, sharded_bridge
from pipeline.nemotron import END_OF_TURN_TOKEN_IDS


class _SingleBatchIterator:
    """Yields exactly one batch for the forward_backward_func (single inference step)."""

    def __init__(self, input_ids, position_ids):
        self.batch = dict(tokens=input_ids, position_ids=position_ids)
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def _text_forward_step(data_iterator, model, **kwargs):
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
    }

    def loss_func(x, **kwargs):
        return x

    return model(**forward_args), loss_func


def generate_megatron(args, prompts) -> list[str]:
    """Greedy decode from a Megatron checkpoint (no KV cache; recomputes each step).

    O(n^2) in generated length but cheap at coherence lengths on the sharded
    model; for long generations wire megatron.core.inference instead.
    """
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from transformers import AutoTokenizer

    from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
    from megatron.bridge.utils.common_utils import disable_mtp_for_inference, get_last_rank, print_rank_0

    if args.max_tokens > 1024:
        print_rank_0(
            f"WARNING: --max-tokens {args.max_tokens} with the no-KV-cache greedy loop is O(n^2); "
            "expect long runtimes. Use <=1024 (256 is typical for coherence)."
        )

    tok_id = args.tokenizer or args.hf_model
    print_rank_0(
        f"Loading Megatron model from {args.model_path} (tp={args.tp} pp={args.pp} ep={args.ep} etp={args.etp})"
    )
    bridge, mp_overrides = sharded_bridge(
        args.hf_model,
        args.tp,
        args.pp,
        args.ep,
        args.etp,
        pipeline_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
    )
    model = load_sharded(bridge, args.model_path, mp_overrides)
    for m in model:
        m.eval()
        disable_mtp_for_inference(m)

    tokenizer = AutoTokenizer.from_pretrained(
        tok_id, trust_remote_code=is_safe_repo(trust_remote_code=args.trust_remote_code, hf_path=tok_id)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Stop on the tokenizer eos plus Nemotron's turn ends.
    stop_ids = set(x for x in [tokenizer.eos_token_id, *END_OF_TURN_TOKEN_IDS] if x is not None)

    def greedy(input_ids):
        generated_ids = input_ids.clone()
        prompt_len = input_ids.size(1)
        for _ in range(args.max_tokens):
            with torch.no_grad():
                position_ids = (
                    torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
                    .unsqueeze(0)
                    .expand_as(input_ids)
                )
                fwd_bwd = get_forward_backward_func()
                output = fwd_bwd(
                    forward_step_func=_text_forward_step,
                    data_iterator=_SingleBatchIterator(input_ids, position_ids),
                    model=model,
                    num_microbatches=1,
                    forward_only=True,
                    seq_length=input_ids.size(1),
                    micro_batch_size=1,
                    collect_non_loss_data=True,
                )
                if isinstance(output, list) and len(output) > 0:
                    output = output[0]
                if parallel_state.is_pipeline_last_stage():
                    # Only the last position's logits feed the argmax — slice BEFORE the TP
                    # all-gather so each step gathers [1,1,vocab/TP] not [1,seq,vocab/TP].
                    output = output[:, -1:, :]
                    ws = parallel_state.get_tensor_model_parallel_world_size()
                    gathered = [torch.zeros_like(output) for _ in range(ws)]
                    dist.all_gather(gathered, output, group=parallel_state.get_tensor_model_parallel_group())
                    next_id = torch.argmax(torch.cat(gathered, dim=2)[:, -1], dim=-1, keepdim=True)
                else:
                    next_id = torch.ones((1, 1), device=generated_ids.device, dtype=generated_ids.dtype)
                torch.distributed.broadcast(next_id, get_last_rank())
                generated_ids = torch.cat([generated_ids, next_id], dim=-1)
                input_ids = generated_ids
                if next_id.item() in stop_ids:
                    break
        return generated_ids[0, prompt_len:]

    gens = []
    for i, prompt in enumerate(prompts, 1):
        if args.generation_mode == "chat":
            text = tokenizer.apply_chat_template(
                build_chat_messages(prompt, args.system_prompt), tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt
        input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
        gen_ids = greedy(input_ids)
        gen = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        gens.append(gen)
        print_rank_0(f"[{i}/{len(prompts)}] PROMPT: {prompt}\n{gen or '<EMPTY>'}\n{'-' * 80}")
    return gens

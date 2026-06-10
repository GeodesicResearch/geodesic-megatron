#!/usr/bin/env python3
"""Megatron-native instruction-coherence test for Nemotron-3 Ultra 550B SFT.

Generates from a *Megatron* checkpoint directly (no vLLM, no HF export) via the
AutoBridge load + Megatron forward pass — the same primitive as
examples/conversion/hf_to_megatron_generate_text.py. vLLM cannot serve this
hybrid 550B here (PP>1 -> hybrid-Mamba KV-cache bug; PP=1 -> TP capped at the
Mamba n_groups=8; NVFP4 -> Hopper driver PTX rejection), but the model trains
fine in Megatron, so we generate in Megatron.

Logs the same W&B table (index/prompt/response/length/empty + empty_pct) as
pipeline_coherence_test.py so coherence is comparable, and always writes the
generations to --output for direct inspection.

Launch across the inference parallelism via torchrun, e.g.:
  torchrun --nproc_per_node=4 --nnodes=6 ... scripts/coherence_megatron.py \
    --hf-model nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 \
    --tokenizer geodesic-research/nemotron-instruct-tokenizer \
    --megatron-path /projects/a5k/public/checkpoints/megatron/nemotron_550b_warm_start_sft_200k_instruct \
    --tp 4 --ep 4 --pp 6 --etp 1 --max-new-tokens 256 \
    --output /path/to/gens.txt
"""

import argparse
import os

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from transformers import AutoTokenizer

from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.utils.common_utils import disable_mtp_for_inference, get_last_rank, print_rank_0

# Same prompts as pipeline_coherence_test.py so coherence is comparable across paths.
CHAT_PROMPTS = [
    "How do I make a quick buck?",
    "What's the best way to lose weight fast?",
    "Can you help me write a cover letter for a software engineering job?",
    "What are the risks of investing in cryptocurrency?",
    "Tell me a joke.",
    "How do I fix a leaky faucet?",
    "What should I do if I'm feeling anxious?",
    "Explain quantum computing in simple terms.",
]


class SingleBatchIterator:
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


def text_forward_step(data_iterator, model, **kwargs):
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
    }

    def loss_func(x, **kwargs):
        return x

    return model(**forward_args), loss_func


def generate_greedy(model, input_ids, max_new_tokens, stop_ids):
    """Greedy decode via the Megatron forward pass (no KV cache; recomputes each step).

    Cheap at coherence lengths (a few hundred tokens) on the sharded 550B. Returns
    the full token sequence (prompt + generated) as a 1xL tensor on every rank.
    """
    position_ids = (
        torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    )
    generated_ids = input_ids.clone()
    for _step in range(max_new_tokens):
        with torch.no_grad():
            fwd_bwd = get_forward_backward_func()
            iterator = SingleBatchIterator(input_ids, position_ids)
            output = fwd_bwd(
                forward_step_func=text_forward_step,
                data_iterator=iterator,
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
                ws = parallel_state.get_tensor_model_parallel_world_size()
                gathered = [torch.zeros_like(output) for _ in range(ws)]
                dist.all_gather(gathered, output, group=parallel_state.get_tensor_model_parallel_group())
                output = torch.cat(gathered, dim=2)
                next_id = torch.argmax(output[:, -1], dim=-1, keepdim=True)
            else:
                next_id = torch.ones((1, 1), device=generated_ids.device, dtype=generated_ids.dtype)

            torch.distributed.broadcast(next_id, get_last_rank())
            generated_ids = torch.cat([generated_ids, next_id], dim=-1)
            input_ids = generated_ids
            position_ids = (
                torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand_as(input_ids)
            )
            if next_id.item() in stop_ids:
                break
    return generated_ids


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-model", required=True, help="HF id for the bridge config (e.g. nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16)")
    p.add_argument("--tokenizer", default=None, help="HF id for tokenizer+chat template (default: --hf-model)")
    p.add_argument("--megatron-path", required=True, help="Megatron checkpoint dir (reads latest_checkpointed_iteration)")
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--pp", type=int, default=6)
    p.add_argument("--ep", type=int, default=4)
    p.add_argument("--etp", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--num-prompts", type=int, default=0, help="0 = all CHAT_PROMPTS; >0 = first N (smoke)")
    p.add_argument("--system-prompt", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--wandb-project", default="megatron_bridge_conversion_coherance_tests")
    p.add_argument("--wandb-entity", default="geodesic")
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args()

    tok_id = args.tokenizer or args.hf_model

    # --- Load the Megatron model via the bridge (config only; weights from the Megatron ckpt) ---
    print_rank_0(f"Loading Megatron model from {args.megatron_path} (tp={args.tp} pp={args.pp} ep={args.ep} etp={args.etp})")
    bridge = AutoBridge.from_hf_pretrained(
        args.hf_model,
        trust_remote_code=is_safe_repo(trust_remote_code=args.trust_remote_code, hf_path=args.hf_model),
    )
    mp = bridge.to_megatron_provider(load_weights=False)
    mp.tensor_model_parallel_size = args.tp
    mp.pipeline_model_parallel_size = args.pp
    mp.expert_model_parallel_size = args.ep
    mp.expert_tensor_parallel_size = args.etp
    mp.pipeline_dtype = torch.bfloat16
    mp.finalize()
    mp.initialize_model_parallel(seed=0)
    model = bridge.load_megatron_model(
        args.megatron_path,
        mp_overrides={
            "tensor_model_parallel_size": args.tp,
            "pipeline_model_parallel_size": args.pp,
            "expert_model_parallel_size": args.ep,
            "expert_tensor_parallel_size": args.etp,
            "pipeline_dtype": torch.bfloat16,
        },
        wrap_with_ddp=False,
    )
    model = [m.cuda() for m in model]
    for m in model:
        m.eval()
        disable_mtp_for_inference(m)
        # Inference (forward_only) with wrap_with_ddp=False: at PP>1 the pipeline schedule calls
        # config.no_sync_func() for grad-sync control. The bridge leaves it as the UNBOUND
        # DistributedDataParallel.no_sync (it expects a DDP-wrapped model) -> TypeError. Setting
        # it to None makes the schedule fall back to contextlib.nullcontext (correct — no grads
        # in inference). Same for the grad/param sync overlap hooks (unused in forward_only).
        m.config.no_sync_func = None
        m.config.grad_sync_func = None
        m.config.param_sync_func = None

    tokenizer = AutoTokenizer.from_pretrained(
        tok_id, trust_remote_code=is_safe_repo(trust_remote_code=args.trust_remote_code, hf_path=tok_id)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Stop on the instruct turn-end (<|im_end|>=11) and </s>=2 as well as the tokenizer eos.
    stop_ids = set(x for x in [tokenizer.eos_token_id, 2, 11] if x is not None)

    prompts = CHAT_PROMPTS if args.num_prompts <= 0 else CHAT_PROMPTS[: args.num_prompts]
    rank0 = torch.distributed.get_rank() == 0
    rows, empty = [], 0
    for i, prompt in enumerate(prompts, 1):
        msgs = ([{"role": "system", "content": args.system_prompt}] if args.system_prompt else []) + [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
        plen = input_ids.size(1)
        gen_ids = generate_greedy(model, input_ids, args.max_new_tokens, stop_ids)
        resp = tokenizer.decode(gen_ids[0, plen:], skip_special_tokens=True).strip()
        is_empty = not resp
        empty += int(is_empty)
        rows.append((i, prompt, resp or "<EMPTY>", len(resp), is_empty))
        print_rank_0(f"[{i}/{len(prompts)}] PROMPT: {prompt}\n{resp or '<EMPTY>'}\n{'-' * 80}")

    if rank0:
        total = len(prompts)
        print(f"SUMMARY: {total} generations, {empty} empty ({100 * empty / total:.1f}%)", flush=True)
        if args.output:
            with open(args.output, "w") as f:
                for (i, pr, rs, _, _) in rows:
                    f.write(f"[{i}] {pr}\n{rs}\n{'-' * 80}\n")
            print(f"saved {args.output}", flush=True)
        try:
            import wandb

            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.run_name or f"gen-test-megatron-{os.path.basename(args.megatron_path.rstrip('/'))}",
                config={
                    "megatron_path": args.megatron_path,
                    "hf_model": args.hf_model,
                    "tokenizer": tok_id,
                    "max_new_tokens": args.max_new_tokens,
                    "tp": args.tp,
                    "pp": args.pp,
                    "ep": args.ep,
                    "mode": "megatron-native-greedy",
                    "prompts": prompts,
                },
            )
            table = wandb.Table(columns=["index", "prompt", "response", "response_length", "empty"])
            for r in rows:
                table.add_data(*r)
            run.log({"generations": table})
            run.summary["total_generations"] = total
            run.summary["empty_count"] = empty
            run.summary["empty_pct"] = 100 * empty / total
            run.finish()
        except Exception as e:  # noqa: BLE001 — W&B is secondary; the --output file is the record
            print(f"WARN: W&B logging failed ({e!r}); generations are in {args.output or 'the job log above'}", flush=True)


if __name__ == "__main__":
    main()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

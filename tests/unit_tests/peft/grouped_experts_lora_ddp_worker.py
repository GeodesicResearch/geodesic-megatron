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
"""Expert LoRA training steps under Megatron DDP with ``overlap_param_gather`` (torchrun worker).

Run by ``test_grouped_experts_lora_ddp.py``. Every rank builds the same small Nemotron-H Latent-MoE model with
CanonicalLoRA on every Super target, its weights filled from generators keyed by parameter name (and by global expert
for expert-sharded tensors), so that every parallel layout holds the same model. Training follows the SFT runs: DDP
with the distributed optimizer, ``overlap_grad_reduce`` and ``overlap_param_gather``, Adam, per-token loss,
``forward_backward_func`` and ``finalize_model_grads``, with the forward pre-hook enabled from the second step as
Bridge's train loop does.

Each per-expert delta first checks that every adapter parameter it reads has a finished param all-gather, that is,
that DDP waited for it after the last optimizer step; a read that DDP did not wait for is counted as stale. Rank 0
writes the per-step losses, the stale-read count and every adapter parameter (expert shards gathered in global expert
order) to ``--out``.

    torchrun --nproc_per_node=4 grouped_experts_lora_ddp_worker.py --ep 2 --global-batch 8 --steps 4 --out run.pt
"""

import argparse
import hashlib
import os
from functools import partial

import torch
import torch.distributed as dist

from tests.unit_tests.small_nemotron_h import SEQ, VOCAB, small_nemotron_h_provider


def _value(key: str, shape: tuple[int, ...]) -> torch.Tensor:
    """A deterministic, well-conditioned fp32 value for the parameter ``key`` at its global shape."""
    generator = torch.Generator().manual_seed(int(hashlib.sha256(key.encode()).hexdigest()[:15], 16))
    if key.endswith("A_log"):
        return torch.log(torch.empty(shape).uniform_(1, 16, generator=generator))
    if key.endswith("dt_bias"):
        return torch.full(shape, -4.0)
    if key.endswith(".D"):
        return torch.ones(shape)
    if "norm" in key:
        return 1 + 0.05 * torch.randn(shape, generator=generator)
    if "router" in key:
        # Well-separated router scores, so no routing decision sits on a numerical tie between layouts.
        return 0.5 * torch.randn(shape, generator=generator)
    return (0.05 if ".adapter." in key else 0.02) * torch.randn(shape, generator=generator)


def _is_expert(param: torch.nn.Parameter) -> bool:
    """Expert-sharded tensors are the ones mcore DDP reduces over the expert-data-parallel group."""
    return not getattr(param, "allreduce", True)


def _peft_hook(model):
    """CanonicalLoRA on every Super target, then every parameter filled with its name-keyed value.

    The names are the unwrapped model's, which every layout shares."""
    from megatron.core import parallel_state

    from megatron.bridge.peft.canonical_lora import CanonicalLoRA
    from megatron.bridge.recipes.nemotronh.nemotron_3_super import NEMOTRON_3_SUPER_CANONICAL_LORA_TARGETS

    lora = CanonicalLoRA(target_modules=list(NEMOTRON_3_SUPER_CANONICAL_LORA_TARGETS), lora_A_init_method="kaiming")
    model = lora(model, training=True)
    ep_size = parallel_state.get_expert_model_parallel_world_size()
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    with torch.no_grad():
        for chunk in model:
            for name, param in chunk.named_parameters():
                if _is_expert(param):
                    full = _value(name, (ep_size * param.shape[0], *param.shape[1:]))
                    param.copy_(full[ep_rank * param.shape[0] : (ep_rank + 1) * param.shape[0]])
                else:
                    param.copy_(_value(name, tuple(param.shape)))
    return model


def _provider(ep: int):
    from megatron.core.transformer.enums import AttnBackend

    return small_nemotron_h_provider(
        "ME*E",
        moe_token_dispatcher_type="alltoall",
        # fp32, so that a stale parameter is not hidden under bf16 rounding; TE's fused attention takes half only.
        attention_backend=AttnBackend.unfused,
        bf16=False,
        params_dtype=torch.float32,
        pipeline_dtype=torch.float32,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=ep,
        calculate_per_token_loss=True,
        gradient_accumulation_fusion=True,
    )


def _batch(global_batch: int, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Step ``step``'s tokens, next-token labels and loss masks; row i masks out its first 3*i positions."""
    generator = torch.Generator().manual_seed(step)
    tokens = torch.randint(0, VOCAB, (global_batch, SEQ + 1), generator=generator)
    mask = torch.ones(global_batch, SEQ)
    for i in range(global_batch):
        mask[i, : 3 * i] = 0
    return tokens[:, :-1], tokens[:, 1:], mask


def _count_stale_reads(ddp, stale: list[int]) -> None:
    """Make every per-expert delta count, into ``stale[0]`` while ``stale[1]`` is set, the adapter parameters it reads
    without a finished all-gather."""
    from megatron.bridge.peft.grouped_experts_lora import GroupedExpertsLoRAAdapter

    for module in ddp.module.modules():
        if not isinstance(module, GroupedExpertsLoRAAdapter):
            continue
        groups = [ddp.param_to_bucket_group[p] for p in module.parameters(recurse=False)]
        for name in ("fc1_delta", "fc2_delta"):

            def checked(*args, _delta=getattr(module, name), **kwargs):
                if stale[1]:
                    stale[0] += sum(not g.param_gather_dispatched or g.param_gather_handle is not None for g in groups)
                return _delta(*args, **kwargs)

            setattr(module, name, checked)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ep", type=int, required=True, help="expert model parallel size (the rest is data parallel)")
    parser.add_argument("--global-batch", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from megatron.core import parallel_state
    from megatron.core.distributed import DistributedDataParallelConfig, finalize_model_grads
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")
    parallel_state.initialize_model_parallel(expert_model_parallel_size=args.ep)
    model_parallel_cuda_manual_seed(1234)

    provider = _provider(args.ep)
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        use_distributed_optimizer=True,
        average_in_collective=False,
    )
    model = provider.provide_distributed_model(
        ddp_config=ddp_config, wrap_with_ddp=True, pre_wrap_hook=[_peft_hook], pg_collection=pg_collection
    )
    ddp = model[0]
    provider.no_sync_func = ddp.no_sync
    provider.finalize_model_grads_func = partial(finalize_model_grads, pg_collection=pg_collection)
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=args.lr,
            weight_decay=0.0,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1e-12,
            clip_grad=0.0,
            params_dtype=torch.float32,
            use_distributed_optimizer=True,
        ),
        model,
        use_gloo_process_groups=False,
        pg_collection=pg_collection,
    )
    stale = [0, 0]
    _count_stale_reads(ddp, stale)

    dp_size, dp_rank = parallel_state.get_data_parallel_world_size(), parallel_state.get_data_parallel_rank()
    num_microbatches = args.global_batch // dp_size
    rows = range(dp_rank * num_microbatches, (dp_rank + 1) * num_microbatches)
    position_ids = torch.arange(SEQ, device="cuda").unsqueeze(0)

    def loss_func(loss_mask, output):
        loss = (output.float() * loss_mask).sum()
        num_tokens = loss_mask.sum().to(torch.int)
        return loss, num_tokens, {"lm loss": torch.stack([loss.detach(), num_tokens.float()])}

    def forward_step(data_iterator, chunk):
        tokens, labels, mask = next(data_iterator)
        return chunk(tokens, position_ids, attention_mask=None, labels=labels), partial(loss_func, mask)

    # As Bridge's train loop: no forward pre-hook in the first step, then the pre-hook waits for the param gathers.
    ddp.disable_forward_pre_hook(param_sync=False)
    losses = []
    for step in range(args.steps):
        if step == 1:
            ddp.enable_forward_pre_hook()
            stale[1] = 1
        tokens, labels, mask = (t.cuda() for t in _batch(args.global_batch, step))
        microbatches = iter([(tokens[i : i + 1], labels[i : i + 1], mask[i : i + 1]) for i in rows])
        ddp.zero_grad_buffer()
        optimizer.zero_grad()
        outputs = get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=microbatches,
            model=model,
            num_microbatches=num_microbatches,
            seq_length=SEQ,
            micro_batch_size=1,
            forward_only=False,
        )
        optimizer.step()
        summed = torch.stack([out["lm loss"] for out in outputs]).sum(0)
        dist.all_reduce(summed, group=parallel_state.get_data_parallel_group())
        losses.append((summed[0] / summed[1]).item())
    # Every rank's parameters after the last step, as the next forward would read them.
    ddp.disable_forward_pre_hook(param_sync=True)

    ep_group = parallel_state.get_expert_model_parallel_group()
    adapters = {}
    for name, param in ddp.named_parameters():
        if ".adapter." not in name:
            continue
        value = param.detach().float().clone()
        if _is_expert(param) and ep_group.size() > 1:
            parts = [torch.empty_like(value) for _ in range(ep_group.size())]
            dist.all_gather(parts, value, group=ep_group)
            value = torch.cat(parts, dim=0)
        adapters[name] = value.cpu()
    stale_total = torch.tensor([stale[0]], device="cuda")
    dist.all_reduce(stale_total)
    if dist.get_rank() == 0:
        torch.save({"losses": losses, "stale_reads": int(stale_total.item()), "adapters": adapters}, args.out)
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

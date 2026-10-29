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
"""HF export of an adapter-only PEFT model with its adapters merged, and the logit check of that merge.

Catalogue:
    MERGE_CHECK_SHAPE: (sequences, tokens) of the probe batch.
    MERGE_CHECK_MIN_RETAINED: the share of the adapters' effect a merge must keep.
    probe_logits: fp32 logits of the model on the probe batch.
    logit_agreement: how closely two logit tensors agree.
    check_merge: how much of the adapters' effect a merged model keeps, and whether the merge fails.
    export_merged_peft: export with the adapters merged, then check the merge.
"""

import json
from pathlib import Path

import torch


#: (sequences, tokens) of the seeded random token ids the merge check of a PEFT export runs on.
MERGE_CHECK_SHAPE = (2, 256)


#: The share of the adapters' logit effect a merge must reproduce. Storing W + update in bf16 rounds away the parts of
#: an update below half a bf16 step of W: about a percent of a trained adapter, more of a barely trained one.
MERGE_CHECK_MIN_RETAINED = 0.9


@torch.no_grad()
def probe_logits(model: list) -> torch.Tensor | None:
    """fp32 logits over the whole vocabulary on ``MERGE_CHECK_SHAPE`` seeded random token ids, on the last pipeline
    stage (None elsewhere).

    One forward-only step of Megatron's pipeline schedule, one microbatch per sequence.
    """
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
    from megatron.core.utils import unwrap_model

    batch, seq = MERGE_CHECK_SHAPE
    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, model[0].config.vocab_size, MERGE_CHECK_SHAPE, generator=generator).cuda()
    position_ids = torch.arange(seq, device="cuda").unsqueeze(0)
    rows = iter(tokens.split(1))

    def forward_step(data_iterator, chunk, **kwargs):
        output = chunk(next(data_iterator), position_ids, attention_mask=None)
        return output, lambda x, **kw: x

    outputs = get_forward_backward_func()(
        forward_step_func=forward_step,
        data_iterator=rows,
        model=model,
        num_microbatches=batch,
        forward_only=True,
        seq_length=seq,
        micro_batch_size=1,
        collect_non_loss_data=True,
    )
    if not parallel_state.is_pipeline_last_stage():
        return None
    logits = torch.cat([output.float() for output in outputs])
    if unwrap_model(model[-1]).parallel_output:
        logits = gather_from_tensor_model_parallel_region(logits)  # each TP rank holds a vocab shard
    return logits


def logit_agreement(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    """Max and mean absolute difference and argmax agreement of two [batch, seq, vocab] logit tensors."""
    diff = (reference - candidate).abs()
    return {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "argmax_agreement": (reference.argmax(-1) == candidate.argmax(-1)).float().mean().item(),
    }


def check_merge(adapted: torch.Tensor, base: torch.Tensor, merged: torch.Tensor) -> dict:
    """How closely merged-model logits match adapted-model logits, against the adapters' effect, and the verdict.

    Returns ``logit_agreement`` of adapted against ``merged`` and ``base`` and of merged against base
    (``merged_vs_base``), keyed by those names; ``retained``: the share of the adapters' logit change
    (adapted - base) that the merged model reproduces, the projection of (merged - base) on it; and ``failure``: None,
    or why the merge fails. It fails if the adapters change nothing, if the merged model differs from the adapted one
    by as much as the adapters' effect, or if it retains less than ``MERGE_CHECK_MIN_RETAINED`` of that effect.
    """
    effect = adapted - base
    check = {
        "merged": logit_agreement(adapted, merged),
        "base": logit_agreement(adapted, base),
        "merged_vs_base": logit_agreement(merged, base),
        "retained": ((merged - base) * effect).sum().item() / max((effect * effect).sum().item(), 1e-30),
        "failure": None,
    }
    if check["base"]["max_abs_diff"] == 0:
        check["failure"] = "the adapters change no logit, so there is nothing to merge"
    elif check["merged"]["mean_abs_diff"] >= check["base"]["mean_abs_diff"]:
        check["failure"] = "the merged model is as far from the adapted one as the base is"
    elif check["retained"] < MERGE_CHECK_MIN_RETAINED:
        check["failure"] = (
            f"the merged model retains {check['retained']:.3f}, less than {MERGE_CHECK_MIN_RETAINED}, of the adapters' "
            "effect. Storing W + update in bf16 rounds away the parts of an update below half a bf16 step of W, which "
            "costs a barely trained adapter this much. A missed or misplaced update costs more. Serve such an adapter "
            "on its base instead of merged."
        )
    return check


def export_merged_peft(bridge, model: list, hf_path: Path, strict: bool, show_progress: bool) -> dict | None:
    """Export an adapted model to HF with every adapter merged, then check the merge on probe logits.

    The HF export merges each adapter into its weight (``MegatronPeftBridge``). The check then merges the model in
    place with ``LoRAMerge`` and compares its probe logits with the adapted and base models' (``check_merge``),
    written to ``<hf_path>/merge_check.json``. Both merges add the same updates (each wrapper's ``weight_update``,
    ``LoRAMerge.merge`` for ``LoRALinear``), but the check reads the in-memory merge, not the written weights.
    Collective over the model's ranks. The check of the rank that writes merge_check.json decides for every rank.

    Returns:
        The check on the last pipeline stage, None elsewhere.

    Raises:
        RuntimeError: on every rank, after the HF export and merge_check.json are written, if the check fails.
    """
    from megatron.core import parallel_state

    from megatron.bridge.peft.lora import LoRAMerge

    merger = LoRAMerge()
    adapted = probe_logits(model)
    with merger.disable_adapter(model):
        base = probe_logits(model)
    bridge.save_hf_pretrained(model, str(hf_path), show_progress=show_progress, strict=strict)
    for chunk in model:
        merger(chunk, training=False)
    merger.disable_adapter_layers(model)
    merged = probe_logits(model)
    check = None
    failed = torch.zeros(1, device="cuda")
    if parallel_state.is_pipeline_last_stage():
        check = check_merge(adapted, base, merged)
        if (
            parallel_state.get_tensor_model_parallel_rank() == 0
            and parallel_state.get_expert_model_parallel_rank() == 0
            and parallel_state.get_data_parallel_rank() == 0
        ):
            (hf_path / "merge_check.json").write_text(json.dumps(check, indent=2) + "\n")
            print(f"Merge check on {MERGE_CHECK_SHAPE} probe tokens: {check}")
            failed.fill_(check["failure"] is not None)
    # Every rank learns the verdict and raises: a rank left waiting in a collective would hold its node until the
    # job's time limit, because only the failed ranks' own torchrun agents end.
    torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
    if failed.item():
        raise RuntimeError(f"the merge check failed (written to {hf_path / 'merge_check.json'}): {check}")
    return check

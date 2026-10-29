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

import logging
from functools import partial
from typing import Iterable

import modelopt.torch.distill as mtd
import torch
from megatron.core import tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.utils import (
    get_batch_on_this_cp_rank,
    get_model_config,
    is_te_min_version,
    unwrap_model,
)

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.forward_step_func_types import applies_token_masking
from megatron.bridge.training.losses import create_masked_next_token_loss_function
from megatron.bridge.training.post_training.distillation import loss_func_kd
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.token_masking.hook import TokenMaskingStats, apply_token_masking
from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_params, trim_padded_cu_seqlens
from megatron.bridge.training.utils.pg_utils import get_pg_collection


logger = logging.getLogger(__name__)


def _dataset_uses_packed_sequences(cfg: ConfigContainer) -> bool:
    """Return True if the configured dataset emits THD packed sequences.

    Packed-sequence runs must build ``packed_seq_params`` (cu_seqlens + seq_idx)
    on *every* pipeline stage, not just the first/last, because hybrid Mamba
    layers on the middle stages need ``seq_idx`` to reset SSM state at packed
    document boundaries (and THD attention needs ``cu_seqlens``). This predicate
    is evaluated from the (replicated) config, so all pipeline ranks reach the
    same decision and the data iterator stays in lockstep.

    The check is intentionally permissive across dataset config types:
    ``FinetuningDatasetConfig`` carries ``packed_sequence_specs`` (SFT THD
    packing, the Nemotron path), while VLM providers expose
    ``pack_sequences_in_batch``.
    """
    dataset_cfg = getattr(cfg, "dataset", None)
    if dataset_cfg is None:
        return False

    pss = getattr(dataset_cfg, "packed_sequence_specs", None)
    if pss is not None and getattr(pss, "packed_sequence_size", -1) > 0:
        return True

    return bool(getattr(dataset_cfg, "pack_sequences_in_batch", False))


def _partition_packed_batch_for_cp(
    batch: dict[str, torch.Tensor], cp_size: int, cp_rank: int
) -> dict[str, torch.Tensor]:
    """Partition THD/packed batches across context-parallel ranks.

    Uses transformer_engine's `thd_get_partitioned_indices` to slice sequence
    dimension aligned with packed cu_seqlens. This avoids the generic
    `get_batch_on_this_cp_rank` slicing which assumes contiguous sequence tokens.

    The partition reads the same trimmed cu_seqlens row that `get_packed_seq_params` gives attention and the Mamba
    layers. The packed collate pads every row of a batch with -1 to the widest row plus one, and on a row ending in
    two or more pads the kernel's search can land on a pad and hand every rank the pack's leading tokens.

    Args:
        batch: One microbatch of packed tensors, sequence along dim 1.
        cp_size: The context-parallel group size.
        cp_rank: This rank's position in the context-parallel group, whose share of every sequence it keeps.
    """

    err_msg = "Please update Transformer Engine to >= 1.10 to use Context Parallel with THD format data"
    try:
        import transformer_engine_torch as tex

        if not is_te_min_version("1.10.0"):
            logger.error(err_msg)
            raise RuntimeError(err_msg)
    except ModuleNotFoundError as e:
        logger.error(err_msg)
        raise e

    cu_seqlens = batch["cu_seqlens"]
    if cu_seqlens.dim() > 1 and cu_seqlens.size(0) != 1:
        raise ValueError("Packed THD batches expect micro-batch size 1 for context-parallel slicing (THD layout)")
    cu_seqlens = trim_padded_cu_seqlens(cu_seqlens.squeeze(), batch.get("cu_seqlens_argmin"))
    cu_seqlens_unpadded = batch.get("cu_seqlens_unpadded")
    if cu_seqlens_unpadded is not None:
        batch["cu_seqlens_unpadded"] = cu_seqlens_unpadded.squeeze()

    skip_keys = {
        "cu_seqlens",
        "cu_seqlens_unpadded",
        "cu_seqlens_argmin",
        "cu_seqlens_unpadded_argmin",
        "max_seqlen",
        "token_count",
        "_full_seq_length",
    }

    for key, val in batch.items():
        if val is None or key in skip_keys:
            continue
        index = tex.thd_get_partitioned_indices(cu_seqlens, val.size(1), cp_size, cp_rank)
        batch[key] = val.index_select(1, index)

    return batch


_ROUTER_CUDA_GRAPH_MODULES = {"moe", "moe_router", "moe_preprocess"}


def _refuse_padding_mask_under_router_cuda_graphs(config) -> None:
    """Refuse a padding mask for a MoE model whose Transformer Engine CUDA graphs capture the router, as upstream
    Megatron-Bridge does: the pinned Megatron-Core cannot mask router statistics inside such a graph.

    A graph captures the router when it captures the whole layer (no ``cuda_graph_modules`` named) or names a MoE,
    router or MoE-preprocess module. Graphs that capture other modules only, and the local and full-iteration CUDA
    graph implementations, are left to run, as upstream leaves them.
    """
    if (
        not getattr(config, "num_moe_experts", None)
        or getattr(config, "cuda_graph_impl", "none") != "transformer_engine"
    ):
        return
    graph_modules = {module.name for module in config.cuda_graph_modules}
    if not graph_modules or graph_modules & _ROUTER_CUDA_GRAPH_MODULES:
        raise ValueError(
            "MoE padding masks are not supported with Transformer Engine CUDA graphs that capture the router "
            f"(cuda_graph_modules: {sorted(graph_modules) or 'the whole layer'}); capture other modules only, or "
            "disable CUDA graphs."
        )


def _prepare_packed_padding_mask(
    padding_mask: torch.Tensor, *, config, model: GPTModel, pg_collection
) -> torch.Tensor:
    """The padding mask laid out like this pipeline stage's hidden states, as upstream Megatron-Bridge lays it out.

    The MoE router reads the mask position for position against its hidden states, which under sequence parallelism
    hold this tensor-parallel rank's share of the sequence. A GPT model's first stage scatters the mask itself, beside
    its embeddings; its later stages, and every stage of a hybrid model, which in the pinned Megatron-Core scatters
    only its embeddings, get their share here.
    """
    needs_sp_scatter = not unwrap_model(model).pre_process or getattr(config, "is_hybrid_model", False)
    if getattr(config, "sequence_parallel", False) and needs_sp_scatter:
        padding_mask = (
            tensor_parallel.scatter_to_sequence_parallel_region(
                padding_mask.transpose(0, 1).contiguous(), group=pg_collection.tp
            )
            .transpose(0, 1)
            .contiguous()
        )
    return padding_mask


def get_batch_from_iterator(
    data_iterator: Iterable,
    use_mtp: bool = False,
    skip_getting_attention_mask_from_dataset: bool = True,
    *,
    is_first_pp_stage: bool,
    is_last_pp_stage: bool,
) -> dict[str, torch.Tensor]:
    """Get a batch of data from the iterator.

    Args:
        data_iterator: The data iterator to get the batch from.
        use_mtp: Whether Multi-Token Prediction layers are enabled.
        skip_getting_attention_mask_from_dataset: If set, the dataset will pass a None attention mask.

    Returns:
        dict[str, torch.Tensor]: A dictionary containing the batch data.
    """
    batch = next(data_iterator)

    required_device_keys = set()
    required_host_keys = set()

    if not skip_getting_attention_mask_from_dataset:
        required_device_keys.add("attention_mask")

    is_packed = "cu_seqlens" in batch
    if is_packed:
        required_device_keys.add("cu_seqlens")
        if "cu_seqlens_unpadded" in batch:
            required_device_keys.add("cu_seqlens_unpadded")
        required_host_keys.add("cu_seqlens_argmin")
        required_host_keys.add("max_seqlen")
        if "cu_seqlens_unpadded_argmin" in batch:
            required_host_keys.add("cu_seqlens_unpadded_argmin")

    if is_first_pp_stage or use_mtp:
        required_device_keys.update(("tokens", "position_ids"))
    if is_last_pp_stage:
        required_device_keys.update(("labels", "loss_mask"))
    # Every stage's MoE routers leave the padded positions out of their statistics.
    if "padding_mask" in batch:
        required_device_keys.add("padding_mask")

    # For packed sequences, record the full (pre-CP-slice) pack length from the
    # raw batch. tokens/labels are full-length on every rank (the sampler shards
    # by DP, not PP), so this is available even on middle pipeline stages where
    # all per-token tensors are dropped below. It is the length the Mamba SSM
    # scan operates on after the CP all-to-all, and is used to build seq_idx.
    full_seq_length = None
    if is_packed:
        seq_ref = batch.get("tokens")
        if seq_ref is None:
            seq_ref = batch.get("labels")
        if seq_ref is not None:
            full_seq_length = seq_ref.size(-1)

    _batch_required_keys = {}
    for key, val in batch.items():
        if key in required_device_keys:
            _batch_required_keys[key] = val.cuda(non_blocking=True) if val is not None else None
        elif key in required_host_keys:
            _batch_required_keys[key] = val.cpu() if val is not None else None
        else:
            _batch_required_keys[key] = None

    _batch_required_keys["_full_seq_length"] = full_seq_length

    return _batch_required_keys


def get_batch(
    data_iterator: Iterable, cfg: ConfigContainer, use_mtp: bool = False, *, pg_collection
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    int | None,
    torch.Tensor | None,
]:
    """Generate a batch.

    Args:
        data_iterator: Input data iterator
        cfg: Configuration container
        use_mtp: Whether Multi-Token Prediction layers are enabled

    Returns:
        tuple of tensors containing tokens, labels, loss_mask, attention_mask, position_ids,
        cu_seqlens, cu_seqlens_argmin, max_seqlen, cu_seqlens_unpadded,
        cu_seqlens_unpadded_argmin, the full (pre-CP-slice) packed sequence length
        (None when not packed), and the padding mask (True at the positions the collate
        padded; None when the batch carries none).
    """
    # Determine pipeline stage role via process group collection
    is_first = is_pp_first_stage(pg_collection.pp)
    is_last = is_pp_last_stage(pg_collection.pp)
    is_middle = (not is_first) and (not is_last)

    # Middle pipeline stages normally consume no data (they receive hidden states
    # over PP). For PACKED sequences they must still pull the batch so they can
    # build packed_seq_params (cu_seqlens for THD attention + seq_idx for the
    # hybrid Mamba SSM scan). Because every pipeline rank then calls
    # next(data_iterator) exactly once per microbatch, the iterator stays in
    # lockstep across the pipeline group (the sampler shards by DP, not PP, so
    # all stages observe the same batch). For non-packed runs, middle stages keep
    # their original behaviour and early-return without touching the iterator.
    if is_middle and not _dataset_uses_packed_sequences(cfg):
        return None, None, None, None, None, None, None, None, None, None, None, None

    batch = get_batch_from_iterator(
        data_iterator,
        use_mtp,
        getattr(cfg.dataset, "skip_getting_attention_mask_from_dataset", True),
        is_first_pp_stage=is_first,
        is_last_pp_stage=is_last,
    )

    # Pop the scalar full-pack length out of the dict before CP slicing so the
    # tensor-only slicing helpers never try to index it.
    full_seq_length = batch.pop("_full_seq_length", None)

    cp_size = pg_collection.cp.size()
    has_packed = batch.get("cu_seqlens") is not None
    if has_packed and cp_size > 1:
        # Slices only the per-token tensors that are present on this stage; the
        # global (un-sliced) cu_seqlens is intentionally preserved so seq_idx is
        # built over the full pack and in original token order.
        batch = _partition_packed_batch_for_cp(batch, cp_size, pg_collection.cp.rank())
    else:
        # slice batch along sequence dimension for context parallelism.
        # `is_hybrid_cp` is a required positional as of the mcore 0.19 pin; it comes from
        # ModelParallelConfig, so the dispatch follows the model's actual configuration
        # rather than a literal asserted here.
        batch = get_batch_on_this_cp_rank(
            batch,
            is_hybrid_cp=cfg.model.hybrid_context_parallel,
            cp_group=pg_collection.cp,
        )

    return (
        batch["tokens"],
        batch["labels"],
        batch["loss_mask"],
        batch.get(
            "attention_mask"
        ),  # Attention_mask is optional for pre-training as a casual mask is generated automatically.
        batch["position_ids"],
        batch.get("cu_seqlens"),
        batch.get("cu_seqlens_argmin"),
        batch.get("max_seqlen"),
        batch.get("cu_seqlens_unpadded"),
        batch.get("cu_seqlens_unpadded_argmin"),
        full_seq_length,
        batch.get("padding_mask"),
    )


def _forward_step_common(
    state: GlobalState, data_iterator: Iterable, model: GPTModel, return_schedule_plan: bool = False
) -> tuple[torch.Tensor, torch.Tensor, TokenMaskingStats | None]:
    """Forward training step.

    Args:
        state: Global state for the run
        data_iterator: Input data iterator
        model: The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor

    Returns:
        tuple containing the output tensor, the loss mask the loss must use (token-masked when the run masks token
        ids) and the microbatch's token-masking statistics (None when the run measures no token ids, and on
        pipeline stages that hold no labels)
    """
    timers = state.timers
    straggler_timer = state.straggler_timer

    config = get_model_config(model)
    pg_collection = get_pg_collection(model)
    use_mtp = (getattr(config, "mtp_num_layers", None) or 0) > 0

    timers("batch-generator", log_level=2).start()
    with straggler_timer(bdata=True):
        (
            tokens,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            cu_seqlens,
            cu_seqlens_argmin,
            max_seqlen,
            cu_seqlens_unpadded,
            cu_seqlens_unpadded_argmin,
            full_seq_length,
            padding_mask,
        ) = get_batch(data_iterator, state.cfg, use_mtp, pg_collection=pg_collection)
    timers("batch-generator").stop()

    # Token masking: no loss at target positions whose label is a masked token id. Runs after context-parallel
    # slicing, position by position, so every CP rank masks its own slice of the labels.
    loss_mask, token_masking_stats = apply_token_masking(labels, loss_mask, state.token_masking)

    forward_args = {
        "input_ids": tokens,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    if use_mtp:
        # The model hands this mask to its multi-token-prediction loss, which falls back to an
        # all-ones mask when none is given and would then train on every position this step
        # excludes from the main loss.
        forward_args["loss_mask"] = loss_mask
    # The MoE routers leave the positions the collate padded out of their expert-bias and auxiliary-loss statistics.
    if padding_mask is not None:
        _refuse_padding_mask_under_router_cuda_graphs(config)
        forward_args["padding_mask"] = _prepare_packed_padding_mask(
            padding_mask, config=config, model=model, pg_collection=pg_collection
        )

    # Add packed sequence support
    if cu_seqlens is not None:
        packed_seq_params = {
            "cu_seqlens": cu_seqlens,
            "cu_seqlens_argmin": cu_seqlens_argmin,
            "max_seqlen": max_seqlen,
            "cu_seqlens_unpadded": cu_seqlens_unpadded,
            "cu_seqlens_unpadded_argmin": cu_seqlens_unpadded_argmin,
        }
        # `full_seq_length` is the full (un-CP-sharded) pack length, including
        # trailing padding -- read in get_batch from the raw, pre-CP-slice batch,
        # so it is available on EVERY pipeline stage (including middle stages,
        # whose per-token tensors are dropped). Threading it into PackedSeqParams
        # builds `seq_idx`, which the hybrid Mamba SSM scan uses to reset state at
        # packed document boundaries on every stage that owns Mamba layers
        # (without it the scan integrates across all concatenated documents and
        # overflows BF16 at long sequence lengths). It also gives the THD attention
        # on middle stages the cu_seqlens it needs to avoid attending across docs.
        forward_args["packed_seq_params"] = get_packed_seq_params(packed_seq_params, total_tokens=full_seq_length)

    with straggler_timer:
        if return_schedule_plan:
            assert config.overlap_moe_expert_parallel_comm, (
                "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            )
            # The schedule plan is built without packed_seq_params, so attention and the Mamba
            # scan would run across the packed documents' boundaries.
            if "packed_seq_params" in forward_args:
                raise ValueError(
                    "packed sequences cannot be combined with overlap_moe_expert_parallel_comm: this step builds "
                    "the schedule plan without packed_seq_params, so attention and the Mamba scan would cross "
                    "document boundaries. Turn off packing or the overlap."
                )
            schedule_plan = model.build_schedule_plan(
                tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
            )
            return schedule_plan, loss_mask, token_masking_stats
        else:
            output_tensor = model(**forward_args)

    return output_tensor, loss_mask, token_masking_stats


@applies_token_masking
def forward_step(
    state: GlobalState, data_iterator: Iterable, model: GPTModel, return_schedule_plan: bool = False
) -> tuple[torch.Tensor, partial]:
    """Forward training step.

    Args:
        state: Global state for the run
        data_iterator: Input data iterator
        model: The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor

    Returns:
        tuple containing the output tensor and the loss function
    """
    output, loss_mask, token_masking_stats = _forward_step_common(state, data_iterator, model, return_schedule_plan)

    loss_function = create_masked_next_token_loss_function(
        loss_mask,
        check_for_nan_in_loss=state.cfg.rerun_state_machine.check_for_nan_in_loss,
        check_for_spiky_loss=state.cfg.rerun_state_machine.check_for_spiky_loss,
        token_masking_stats=token_masking_stats,
    )

    return output, loss_function


@applies_token_masking
def forward_step_modelopt(
    state: GlobalState, data_iterator: Iterable, model: GPTModel, return_schedule_plan: bool = False
) -> tuple[torch.Tensor, partial]:
    """Forward training step with ModelOpt required modifications.

    Args:
        state: Global state for the run
        data_iterator: Input data iterator
        model: The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor

    Returns:
        tuple containing the output tensor and the loss function
    """
    output, loss_mask, token_masking_stats = _forward_step_common(state, data_iterator, model, return_schedule_plan)

    loss_function = _create_loss_function_modelopt(
        loss_mask,
        model,
        check_for_nan_in_loss=state.cfg.rerun_state_machine.check_for_nan_in_loss,
        check_for_spiky_loss=state.cfg.rerun_state_machine.check_for_spiky_loss,
        token_masking_stats=token_masking_stats,
    )

    return output, loss_function


def _create_loss_function_modelopt(
    loss_mask: torch.Tensor,
    model: GPTModel,
    check_for_nan_in_loss: bool,
    check_for_spiky_loss: bool,
    token_masking_stats: TokenMaskingStats | None,
) -> partial:
    """Create the loss function for a ModelOpt model: knowledge distillation around the masked next-token loss.

    Args:
        loss_mask: Used to mask out some portions of the loss
        model: The GPT Model
        check_for_nan_in_loss: Whether to check for NaN values in the loss
        check_for_spiky_loss: Whether to check for spiky loss values
        token_masking_stats: The microbatch's token-masking statistics, reported by the next-token loss

    Returns:
        A partial function that can be called with output_tensor to compute the loss
    """
    mnt_loss_func = create_masked_next_token_loss_function(
        loss_mask,
        check_for_nan_in_loss=check_for_nan_in_loss,
        check_for_spiky_loss=check_for_spiky_loss,
        token_masking_stats=token_masking_stats,
    )
    unwrapped_model = unwrap_model(model)
    if isinstance(unwrapped_model, mtd.DistillationModel):
        return partial(loss_func_kd, loss_mask=loss_mask, original_loss_fn=mnt_loss_func, model=unwrapped_model)
    else:
        return mnt_loss_func

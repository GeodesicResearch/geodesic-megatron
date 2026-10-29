# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""What the tools that run forward passes of one Megatron checkpoint on every data-parallel rank share.

``load_single_chunk_model`` builds the checkpoint's model; ``lockstep_schedule`` deals items out so that every rank
takes the same number of steps (the experts' all-to-all needs every rank's forward); ``gather_to_rank_zero`` and
``merge_by_index`` bring the ranks' results together on rank 0.

Megatron-Bridge and ``torch.distributed`` are imported only when a function that needs them runs.
"""

from __future__ import annotations

from typing import Any


def load_single_chunk_model(checkpoint: str, model_config: Any, *, error: type[Exception]):
    """``model_config``'s model with the weights of the Megatron checkpoint at ``checkpoint``, in eval mode.

    Built and loaded by Megatron-Bridge's weights-only loader inside the caller's process group and model-parallel
    state (no temporary distributed context), so it is a collective over that group and reads the checkpoint from
    disk. Raises ``error`` when the build holds more than one model chunk (a virtual pipeline).
    """
    from megatron.bridge.training.model_load_save import build_and_load_model

    built = build_and_load_model(str(checkpoint), model_config, skip_temp_dist_context=True)
    chunks = built if isinstance(built, list) else [built]
    if len(chunks) != 1:
        raise error(f"expected one model chunk, got {len(chunks)}")
    return chunks[0].eval()


def lockstep_schedule(count: int, world: int, rank: int) -> list[tuple[int, bool]]:
    """The items data-parallel rank ``rank`` of ``world`` runs, one per step, as ``(item index, kept)``.

    Step ``s`` runs item ``s * world + rank``, and every rank runs the same number of steps, since the experts'
    all-to-all needs every rank's forward: a rank past the last item runs item 0 again, with ``kept`` False, and its
    result is dropped. Over all ranks the kept items are each of ``range(count)`` exactly once.
    """
    schedule = []
    for step in range(-(-count // world)):
        index = step * world + rank
        schedule.append((index, True) if index < count else (0, False))
    return schedule


def gather_to_rank_zero(local: Any, *, group: Any) -> list[Any] | None:
    """Every rank's ``local`` (a picklable object) as a list ordered by rank on rank 0 of ``group``, and None on the
    other ranks. ``group`` is a process group, or None for the default one. A collective over ``group``."""
    import torch.distributed as dist

    root = 0 if group is None else dist.get_global_rank(group, 0)
    gathered = [None] * dist.get_world_size(group) if dist.get_rank(group) == 0 else None
    dist.gather_object(local, gathered, dst=root, group=group)
    return gathered


def merge_by_index(parts: list[dict[int, Any]], count: int, *, where: str, error: type[Exception]) -> dict[int, Any]:
    """The union of the ranks' ``{item index: result}`` maps, which together must hold each of ``range(count)``.

    Ranks that share a data-parallel rank (tensor-parallel peers) report the same indices; one result is kept per
    index. Raises ``error``, opening with ``where``, when an index is missing or beyond the count.
    """
    merged: dict[int, Any] = {}
    for part in parts:
        merged.update(part)
    if sorted(merged) != list(range(count)):
        raise error(f"{where}: results for items {sorted(merged)} were gathered, not each of range({count})")
    return merged

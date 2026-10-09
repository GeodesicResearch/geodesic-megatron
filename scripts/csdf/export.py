"""Stream small canonical adapter factors through Bridge's distributed mappings."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file

from .factor_mapping import fix_adapter_task_mappings


def export_factors(bridge, model, destination: Path) -> None:
    """All ranks participate; only rank zero writes, then atomically publishes."""
    rank = torch.distributed.get_rank()
    temporary = destination.with_name(destination.name + ".incomplete")
    if rank == 0:
        temporary.mkdir(parents=True, exist_ok=True)
    conversion = bridge._model_bridge
    registry = conversion.mapping_registry()
    tasks = fix_adapter_task_mappings(conversion.build_adapter_conversion_tasks(model), registry)
    if not tasks:
        raise ValueError("No SDF adapters found to export")
    alphas = {"__format_version__": 2}
    shard_index = 0
    for base, adapters in tasks.items():
        if ".experts." in base and ".shared_experts." not in base:
            raise ValueError(f"Routed experts are not supported: {base}")
        for task in adapters:
            weights = conversion.materialize_adapter_weights([task])[0]
            names = conversion._get_base_hf_param_names_for_adapter(registry, base, task.adapter_key, ".weight")
            a = weights.linear_in_weight.weight.to(torch.bfloat16).contiguous()
            b = weights.linear_out_weight.weight
            slices = conversion._get_fused_adapter_linear_out_slices(model, names, b, is_expert=False)
            if slices is None:
                if len(names) != 1:
                    raise ValueError(f"Unresolved fused adapter {names}")
                slices = {names[0]: b}
            if rank == 0:
                tensors = {}
                for name, rows in slices.items():
                    if name in alphas:
                        raise ValueError(f"Duplicate exported factor {name}")
                    tensors[name + ".A"] = a.cpu().clone()
                    tensors[name + ".B"] = rows.to(torch.bfloat16).cpu().contiguous()
                    alphas[name] = float(weights.alpha) / weights.dim
                save_file(tensors, temporary / f"rank_{shard_index:05d}.safetensors")
                shard_index += 1
    if rank == 0:
        (temporary / "alphas.json").write_text(json.dumps(alphas, sort_keys=True))
        from geodesic_utils.inference.adapter_chain import factor_inventory

        inventory = factor_inventory(temporary)
        (temporary / "inventory.json").write_text(json.dumps(inventory, sort_keys=True))
        if destination.exists():
            if factor_inventory(destination)["sha256"] != inventory["sha256"]:
                raise ValueError(f"Refusing to replace a different completed SDF adapter: {destination}")
        else:
            os.rename(temporary, destination)
    torch.distributed.barrier()

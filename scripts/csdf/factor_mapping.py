"""Canonical adapter mapping, adapted from nemo-rl core/patches/peft_bridge_adapter_export_patch.py.

Kept explicit at export, without installing a global monkeypatch.
"""

import dataclasses


def fix_adapter_task_mappings(tasks_by_base, registry):
    """Rewrite adapter ``linear_out`` task mappings to match the base layout.

    Mutates ``tasks_by_base`` in place and returns it. Raises when a base's
    registry mapping is neither a contiguous-layout class, a fused layout
    handled downstream by ``_get_fused_adapter_linear_out_slices``, nor
    ``MambaInProjMapping``.
    """
    from megatron.bridge.models.conversion.param_mapping import (
        AutoMapping,
        ColumnParallelMapping,
        DirectMapping,
        GatedMLPMapping,
        GDNLinearMapping,
        KVMapping,
        MambaInProjMapping,
        QKVMapping,
        ReplicatedMapping,
        RowParallelMapping,
    )

    # Bases whose gathered adapter-B layout is already canonical (contiguous
    # split ⇒ plain cat is correct) or is de-interleaved downstream by
    # _get_fused_adapter_linear_out_slices (fused QKV / KV / GDN / gated-MLP fc1).
    passthrough_classes = (
        ColumnParallelMapping,
        RowParallelMapping,
        ReplicatedMapping,
        AutoMapping,
        DirectMapping,
        QKVMapping,
        KVMapping,
        GDNLinearMapping,
        GatedMLPMapping,
    )

    for base_prefix, adapter_tasks in tasks_by_base.items():
        base_mapping = registry.megatron_to_hf_lookup(f"{base_prefix}.weight")
        if base_mapping is None:
            base_mapping = registry.megatron_to_hf_lookup(f"{base_prefix}.weight0")
        if base_mapping is None:
            raise RuntimeError(
                "[adapter-export-patch] no registry mapping for adapter "
                f"base {base_prefix!r} — cannot determine the base "
                "weight's row layout, refusing to export its adapter B "
                "with an assumed contiguous layout."
            )
        if isinstance(base_mapping, MambaInProjMapping):
            fixed_tasks = []
            for task in adapter_tasks:
                lot = task.linear_out_task
                fixed_lot = dataclasses.replace(
                    lot,
                    mapping=MambaInProjMapping(
                        megatron_param=lot.mapping.megatron_param,
                        hf_param=str(lot.mapping.hf_param),
                    ),
                )
                fixed_tasks.append(dataclasses.replace(task, linear_out_task=fixed_lot))
            tasks_by_base[base_prefix] = fixed_tasks
        elif not isinstance(base_mapping, passthrough_classes):
            raise RuntimeError(
                "[adapter-export-patch] adapter base "
                f"{base_prefix!r} maps through "
                f"{type(base_mapping).__name__}, which is neither a "
                "contiguous-layout mapping nor a fused layout handled by "
                "_get_fused_adapter_linear_out_slices. Exporting its "
                "adapter B via plain TP concatenation would scramble the "
                "row order — this mapping class needs explicit handling."
            )
    return tasks_by_base

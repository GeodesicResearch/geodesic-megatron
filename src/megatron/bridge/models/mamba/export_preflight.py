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

"""Refusing a Mamba checkpoint whose saved provider config cannot be exported.

Both faults this module knows about are created by `_apply_moe_experts_impl`
swapping the expert backend at `provide()` time, and both are recorded in the
`run_config.yaml` the run saves. They are metadata: the weights are unaffected,
and re-training fixes nothing.

The rules live beside the provider that creates them rather than in
`utils/hf_export_validation.py`, which is model-agnostic. What belongs there is
`UnmappedParameterCounter`, which catches the *consequence* — parameters
silently skipped — for any model. This module catches the specific cause early,
before a 200 GB+ load, and with an error naming the edit.
"""

from __future__ import annotations


# The only expert backend the bridge's mapping registry can resolve. Any other value
# means the model built at export time exposes different `named_parameters()` than the
# registry knows, and every routed-expert parameter is silently skipped.
EXPORTABLE_EXPERTS_IMPL = "te_grouped"

_DEFAULT_STACK_SPEC_TARGET = "megatron.bridge.models.mamba.mamba_provider.get_default_mamba_stack_spec"


def assert_run_config_is_exportable(run_config: dict) -> None:
    """Refuse a checkpoint whose saved provider config cannot be exported as-is.

    Reports every fault it finds in one message, so a single edit pass clears
    them rather than one failed conversion per fault.

    Args:
        run_config: The parsed `run_config.yaml` from the iteration directory.

    Raises:
        ValueError: If the config names a non-exportable expert backend, or a
            stack spec whose serialized target cannot be re-imported.
    """
    model = run_config.get("model") or {}
    faults: list[str] = []

    experts_impl = model.get("moe_experts_impl")
    if experts_impl is not None and experts_impl != EXPORTABLE_EXPERTS_IMPL:
        faults.append(
            f"  moe_experts_impl: {experts_impl!r} -> {EXPORTABLE_EXPERTS_IMPL!r}\n"
            f"      The bridge maps against the live model's named_parameters(), not the "
            f"on-disk state dict, so the export-time instantiation is what has to match. "
            f"Left as {experts_impl!r}, every routed-expert parameter goes unmapped and is "
            f"silently skipped."
        )

    target = (model.get("mamba_stack_spec") or {}).get("_target_")
    if isinstance(target, str) and "<locals>" in target:
        faults.append(
            f"  mamba_stack_spec._target_: {target!r}\n"
            f"      -> {_DEFAULT_STACK_SPEC_TARGET}\n"
            f"      Training serializes the swapped spec as a nested-closure path, which no "
            f"import can resolve."
        )

    if faults:
        joined = "\n".join(faults)
        raise ValueError(
            "This checkpoint's run_config.yaml cannot be exported as written. Patch the "
            "checkpoint's own run_config.yaml and re-run the export — this is a metadata "
            "edit, the weights are fine, and re-training fixes nothing:\n"
            f"{joined}"
        )

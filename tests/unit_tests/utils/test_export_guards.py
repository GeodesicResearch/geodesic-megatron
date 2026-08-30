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

"""The two guards protecting an export from silently dropping weights.

A checkpoint trained on a non-TE expert backend exports into a model whose
`named_parameters()` do not match the mapping registry, and the bridge's
response is a per-parameter `logger.warning` followed by `continue`. The writer
then exits 0 having skipped every routed-expert weight, and the resulting
checkpoint loads and generates text. `validate_hf_export` cannot see it: with
every MoE layer losing the same tensors, no layer is short of a sibling, and
the index agrees with the shards on the reduced set.

So the loss has to be caught where it happens. `assert_run_config_is_exportable`
refuses the checkpoint up front, and `UnmappedParameterCounter` fails the run
afterwards on the evidence rather than on a predicted cause.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from megatron.bridge.models.conversion import model_bridge
from megatron.bridge.models.mamba.export_preflight import assert_run_config_is_exportable
from megatron.bridge.utils.hf_export_validation import (
    _SKIPPED_PARAM_LOG_PREFIXES,
    UnmappedParameterCounter,
    UnmappedParameterError,
)

# Reuse the export-task harness rather than restating it: these drive the real
# `build_export_fp8_tasks` into its skip branch on CPU, which is what lets the
# prefix test observe the actual emitter instead of a string this repo also owns.
from tests.unit_tests.models.test_fp8_param_export import (
    _QKV_GLOBAL,
    DummyBridge,
    _make_qkv_mapping_type,
    _patch_export_task_context,
)


class TestAssertRunConfigIsExportable:
    def test_a_te_grouped_checkpoint_is_accepted(self):
        assert_run_config_is_exportable({"model": {"moe_experts_impl": "te_grouped"}})

    def test_a_config_without_the_field_is_accepted(self):
        # Omitting the field means the provider default, which is te_grouped.
        assert_run_config_is_exportable({"model": {}})
        assert_run_config_is_exportable({})

    def test_a_torch_grouped_checkpoint_is_refused(self):
        with pytest.raises(ValueError) as excinfo:
            assert_run_config_is_exportable({"model": {"moe_experts_impl": "torch_grouped"}})
        message = str(excinfo.value)
        assert "torch_grouped" in message
        assert "moe_experts_impl" in message
        # The message has to carry the remedy, because this is the only place a
        # reader learns the export is a metadata patch and not a re-train.
        assert "te_grouped" in message
        assert "re-training fixes nothing" in message

    def test_an_unimportable_stack_spec_is_refused(self):
        # Training serialises the swapped spec as a <locals> closure path, which
        # no import can resolve. Catching it here beats an opaque import error.
        with pytest.raises(ValueError) as excinfo:
            assert_run_config_is_exportable(
                {
                    "model": {
                        "moe_experts_impl": "te_grouped",
                        "mamba_stack_spec": {
                            "_target_": "megatron.bridge.models.mamba.mamba_provider."
                            "MambaModelProvider._apply_moe_experts_impl.<locals>."
                            "_grouped_resolved_stack_spec"
                        },
                    }
                }
            )
        assert "mamba_stack_spec" in str(excinfo.value)

    def test_the_two_faults_are_reported_together(self):
        # A torch_grouped checkpoint has both, and fixing one at a time would
        # mean two failed 225 GB conversions instead of one.
        with pytest.raises(ValueError) as excinfo:
            assert_run_config_is_exportable(
                {
                    "model": {
                        "moe_experts_impl": "torch_grouped",
                        "mamba_stack_spec": {"_target_": "somewhere.<locals>.spec"},
                    }
                }
            )
        message = str(excinfo.value)
        assert "moe_experts_impl" in message and "mamba_stack_spec" in message


class TestUnmappedParameterCounter:
    """Counts what the bridge actually skipped, via the warnings it emits."""

    def _emit(self, message: str) -> None:
        logging.getLogger("megatron.bridge.models.conversion.model_bridge").warning(message)

    def test_a_clean_conversion_passes(self):
        with UnmappedParameterCounter() as counter:
            self._emit("Converting decoder.layers.0.mlp.router.weight")
        assert counter.count == 0
        counter.raise_if_any()

    def test_the_megatron_param_wording_is_counted(self):
        with UnmappedParameterCounter() as counter:
            self._emit("WARNING: No mapping found for megatron_param: decoder.layers.0.mlp.experts.weight1")
        assert counter.count == 1

    def test_the_global_name_wording_is_also_counted(self):
        # The three warn sites do not share a message: one says global_name.
        # Keying on the shared prefix is what makes the counter cover all of them.
        with UnmappedParameterCounter() as counter:
            self._emit("No mapping found for global_name: decoder.layers.0.mlp.experts.weight2")
        assert counter.count == 1

    def test_the_missing_hf_key_wording_is_counted(self):
        # The second logged skip cause: the parameter maps, but onto an HF name the
        # target model does not have. Same silent loss, different sentence.
        with UnmappedParameterCounter() as counter:
            self._emit("WARNING: Can't find backbone.layers.0.mixer.in_proj.weight in hf_keys")
        assert counter.count == 1

    def test_the_plural_missing_hf_key_wording_is_counted(self):
        with UnmappedParameterCounter() as counter:
            self._emit("WARNING: Can't find the following HF parameters in hf_keys: ['a', 'b']")
        assert counter.count == 1

    def test_every_watched_prefix_is_a_real_emitter(self):
        """No watched prefix may be one this repo invented.

        The per-wording tests above assert the counter matches strings written here,
        which stays green if `model_bridge.py` never emitted them or stops doing so.
        Reading the emitter's source closes that: a prefix that no longer appears in
        a warning there is watching for something that cannot happen.
        """
        source = Path(model_bridge.__file__).read_text() if hasattr(model_bridge, "__file__") else ""
        warned = [line for line in source.splitlines() if "logger.warning" in line]
        for prefix in _SKIPPED_PARAM_LOG_PREFIXES:
            assert any(prefix in line for line in warned), (
                f"{prefix!r} no longer appears in any logger.warning in model_bridge.py, "
                f"so the counter is watching for a message that is never emitted"
            )

    def test_raise_if_any_names_the_skipped_parameters(self):
        with UnmappedParameterCounter() as counter:
            self._emit("No mapping found for megatron_param: decoder.layers.0.mlp.experts.weight1")
            self._emit("No mapping found for megatron_param: decoder.layers.1.mlp.experts.weight1")
        with pytest.raises(UnmappedParameterError) as excinfo:
            counter.raise_if_any()
        message = str(excinfo.value)
        assert "2" in message
        assert "weight1" in message

    def test_the_handler_is_removed_on_exit(self):
        logger = logging.getLogger("megatron.bridge.models.conversion.model_bridge")
        before = list(logger.handlers)
        with UnmappedParameterCounter():
            pass
        assert list(logger.handlers) == before

    def test_the_handler_is_removed_even_when_the_body_raises(self):
        logger = logging.getLogger("megatron.bridge.models.conversion.model_bridge")
        before = list(logger.handlers)
        with pytest.raises(RuntimeError):
            with UnmappedParameterCounter():
                raise RuntimeError("conversion blew up")
        assert list(logger.handlers) == before

    def test_the_bridge_still_emits_what_the_counter_watches_for(self, monkeypatch):
        """Pin the watched prefixes to the real emitter, not to strings we also own.

        Every other test here feeds the counter a message this repo wrote, so all of
        them would stay green if an upstream merge reworded `model_bridge.py` and the
        counter silently stopped matching — restoring exactly the silent-data-loss
        behaviour it exists to prevent. This drives the real `build_export_fp8_tasks`
        into its no-mapping branch and asserts the counter sees it.
        """
        bridge = DummyBridge()
        seen = {"n": 0}

        class _RegistryThatMissesTheSecondLookup:
            @staticmethod
            def megatron_to_hf_lookup(_name):
                seen["n"] += 1
                return _make_qkv_mapping_type(_QKV_GLOBAL)() if seen["n"] == 1 else None

        _patch_export_task_context(
            monkeypatch,
            bridge,
            _QKV_GLOBAL,
            registry_factory=lambda: _RegistryThatMissesTheSecondLookup(),
        )
        model = SimpleNamespace(
            config=SimpleNamespace(share_embeddings_and_output_weights=False),
            named_parameters=lambda: [],
        )

        with UnmappedParameterCounter() as counter:
            bridge.build_export_fp8_tasks(SimpleNamespace(state=SimpleNamespace(source=SimpleNamespace())), [model])

        assert counter.count == 1, (
            "the counter did not observe a skip the bridge really performed — its watched "
            "prefixes have drifted from model_bridge.py's wording"
        )

    def test_counting_does_not_suppress_the_warning(self):
        # The operator still needs to see the warnings; the counter observes.
        logger = logging.getLogger("megatron.bridge.models.conversion.model_bridge")
        seen: list[str] = []

        class _Spy(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                seen.append(record.getMessage())

        spy = _Spy()
        logger.addHandler(spy)
        try:
            with UnmappedParameterCounter():
                self._emit("No mapping found for megatron_param: x")
        finally:
            logger.removeHandler(spy)
        assert any("No mapping found" in m for m in seen)

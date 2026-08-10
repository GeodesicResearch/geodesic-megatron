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

"""Expert-backend invariants for the ``configs/pa_warm_start/`` training configs.

``torch_grouped`` is worth ~25% of wall clock on this family's topology, but the provider
field still defaults to ``te_grouped``. A config that simply omits ``moe_experts_impl``
therefore trains on the slow path and says nothing about it — the failure is invisible in
the logs and only shows up as a slower run. These tests make the omission fail here instead.

The two combinations that ``torch_grouped`` cannot serve are checked against the same
constant the provider enforces them with, so a change to that set cannot leave the configs
and the guard disagreeing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from megatron.bridge.models.mamba.mamba_provider import MOE_INTERNAL_OFFLOAD_MODULES


_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "pa_warm_start"


def _config_paths() -> list[Path]:
    return sorted(_CONFIG_DIR.glob("*.yaml"))


def _model_section(path: Path) -> dict:
    return yaml.safe_load(path.read_text())["model"]


def test_the_config_directory_is_not_empty():
    # Every test below is vacuous if the glob stops matching, so assert it matches.
    assert _config_paths(), f"no configs found under {_CONFIG_DIR}"


@pytest.mark.parametrize("path", _config_paths(), ids=lambda p: p.name)
class TestExpertBackend:
    def test_the_backend_is_declared_explicitly(self, path: Path):
        model = _model_section(path)
        assert "moe_experts_impl" in model, (
            f"{path.name} does not set model.moe_experts_impl. The provider defaults to "
            "'te_grouped', so omitting it silently selects the slow path."
        )

    def test_the_backend_is_torch_grouped(self, path: Path):
        assert _model_section(path)["moe_experts_impl"] == "torch_grouped"

    def test_torch_grouped_is_not_paired_with_mtp(self, path: Path):
        model = _model_section(path)
        if model.get("moe_experts_impl") != "torch_grouped":
            pytest.skip("guard applies only to torch_grouped")
        # The swap rewrites the main stack's MoE spec but not an MTP block's nested one.
        assert not model.get("mtp_num_layers"), (
            f"{path.name} pairs torch_grouped with mtp_num_layers="
            f"{model.get('mtp_num_layers')!r}; the provider raises NotImplementedError."
        )

    def test_torch_grouped_is_not_paired_with_moe_internal_offload(self, path: Path):
        model = _model_section(path)
        if model.get("moe_experts_impl") != "torch_grouped":
            pytest.skip("guard applies only to torch_grouped")
        if not model.get("fine_grained_activation_offloading"):
            return
        # These are implemented inside TEGroupedMLP, which torch_grouped replaces.
        offending = MOE_INTERNAL_OFFLOAD_MODULES & set(model.get("offload_modules") or [])
        assert not offending, (
            f"{path.name} offloads {sorted(offending)} while on torch_grouped; those live "
            "inside TEGroupedMLP and would offload nothing. The provider raises ValueError."
        )

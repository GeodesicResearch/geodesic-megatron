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
"""Unit tests for the resilience wiring in pipeline_training_run.py.

Fault tolerance and NVRx straggler detection are separately switchable: `--disable-ft`
drops both, `--disable-straggler` drops only the detector. The `run_module` fixture loads
the repo-root script by path, so the real parser and the real config-assembly function are
exercised — the config being mutated is a real Nano recipe, not a stand-in.
"""

from __future__ import annotations

import sys

import pytest

from megatron.bridge.training.config import FaultToleranceConfig, NVRxStragglerDetectionConfig


@pytest.fixture
def cfg(run_module):
    """A real ConfigContainer: the Nano SFT recipe is pure dataclass construction."""
    return run_module.RECIPE_MAP[("nano", "sft")](None)


def _apply(run_module, cfg, monkeypatch, flags):
    monkeypatch.setattr(sys, "argv", ["pipeline_training_run.py", "--model", "nano", "--mode", "sft", *flags])
    args, overrides = run_module.parse_cli_args()
    assert overrides == [], "resilience flags must be recognised by the parser, not fall through to Hydra"
    run_module.apply_resilience_config(cfg, args)
    return cfg


def test_recipe_ships_neither_config(cfg):
    """Both configs come from the flags alone, so the assertions below cannot pass vacuously."""
    assert cfg.ft is None
    assert cfg.nvrx_straggler is None


def test_no_flags_enables_ft_and_straggler(run_module, cfg, monkeypatch):
    _apply(run_module, cfg, monkeypatch, [])
    assert isinstance(cfg.ft, FaultToleranceConfig)
    assert cfg.ft.enable_ft_package is True
    assert cfg.ft.calc_ft_timeouts is True
    assert isinstance(cfg.nvrx_straggler, NVRxStragglerDetectionConfig)
    assert cfg.nvrx_straggler.enabled is True


def test_disable_straggler_keeps_fault_tolerance(run_module, cfg, monkeypatch):
    _apply(run_module, cfg, monkeypatch, ["--disable-straggler"])
    assert isinstance(cfg.ft, FaultToleranceConfig)
    assert cfg.ft.enable_ft_package is True
    assert cfg.nvrx_straggler is None


def test_disable_ft_drops_both(run_module, cfg, monkeypatch):
    _apply(run_module, cfg, monkeypatch, ["--disable-ft"])
    assert cfg.ft is None
    assert cfg.nvrx_straggler is None


def test_disable_ft_wins_over_disable_straggler(run_module, cfg, monkeypatch):
    """--disable-ft is the broader switch: combining the two must not resurrect cfg.ft."""
    _apply(run_module, cfg, monkeypatch, ["--disable-ft", "--disable-straggler"])
    assert cfg.ft is None
    assert cfg.nvrx_straggler is None

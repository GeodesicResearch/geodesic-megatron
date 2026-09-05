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

import logging
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


# --- heartbeat headroom -------------------------------------------------------------------
#
# When ft_launcher's rank monitor never receives an initial heartbeat, the initial-rank
# heartbeat timeout stops being a liveness check and becomes a wall the job is SIGKILLed at.
# A run whose first checkpoint is scheduled past that wall therefore restarts from iteration 0
# with an empty checkpoint directory, forever. The arithmetic that predicts it is available at
# startup, so the run says so rather than leaving it to be discovered hours in.


@pytest.mark.parametrize(
    ("first_checkpoint_iteration", "expected_seconds"),
    [
        (2980, 7200 / 2980),  # the control-pretraining baseline: needs < 2.42 s/iter
        (7200, 1.0),  # exactly one second per iteration
        (1, 7200.0),  # a checkpoint every iteration has the whole window
    ],
)
def test_max_seconds_per_iteration_arithmetic(run_module, first_checkpoint_iteration, expected_seconds):
    assert run_module.max_seconds_per_iteration_under_ft_heartbeat(
        first_checkpoint_iteration, run_module.FT_INITIAL_RANK_HEARTBEAT_TIMEOUT_SECONDS
    ) == pytest.approx(expected_seconds)


def test_max_seconds_per_iteration_rejects_non_positive_iteration(run_module):
    """A zero or negative first-checkpoint iteration is a config error, not a divide-by-zero."""
    with pytest.raises(ValueError):
        run_module.max_seconds_per_iteration_under_ft_heartbeat(0, 7200.0)


def test_ft_warns_that_the_first_checkpoint_must_beat_the_heartbeat(run_module, cfg, monkeypatch, caplog):
    """The warning must carry the actual required rate; a generic caution would not have helped."""
    cfg.checkpoint.save_interval = 2980
    cfg.train.train_iters = 59605
    with caplog.at_level(logging.WARNING):
        _apply(run_module, cfg, monkeypatch, ["--disable-straggler"])
    assert "2980" in caplog.text
    assert "2.42" in caplog.text
    assert "Did not get initial heartbeat" in caplog.text


def test_save_only_at_end_measures_against_the_whole_run(run_module, cfg, monkeypatch, caplog):
    """save_interval past train_iters means the only checkpoint is the final one."""
    cfg.checkpoint.save_interval = 1000000
    cfg.train.train_iters = 400
    with caplog.at_level(logging.WARNING):
        _apply(run_module, cfg, monkeypatch, [])
    assert "400" in caplog.text
    assert "18.00" in caplog.text  # 7200 / 400


def test_disable_ft_emits_no_heartbeat_warning(run_module, cfg, monkeypatch, caplog):
    """With ft off there is no heartbeat monitor, so the warning would be noise."""
    cfg.checkpoint.save_interval = 2980
    cfg.train.train_iters = 59605
    with caplog.at_level(logging.WARNING):
        _apply(run_module, cfg, monkeypatch, ["--disable-ft"])
    assert "initial heartbeat" not in caplog.text

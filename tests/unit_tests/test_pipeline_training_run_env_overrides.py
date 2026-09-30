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

"""pipeline_training_run.py's per-node echo of the launcher's env overrides.

The echo is the proof that an ``ISAMBARD_ENV_OVERRIDES`` value reached the training processes,
so these tests drive the real function against a real process environment (monkeypatched) and
read what it logged.
"""

from __future__ import annotations

import logging
import socket

import pytest


OVERRIDE_KEYS = "ISAMBARD_ENV_OVERRIDE_KEYS"


def echo_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("[env-overrides]")]


@pytest.fixture
def overrides_env(monkeypatch):
    """Two applied overrides, one with an empty value, as seen by node 3's first rank."""
    monkeypatch.setenv(OVERRIDE_KEYS, "TORCH_NCCL_BLOCKING_WAIT,NCCL_DEBUG_SUBSYS")
    monkeypatch.setenv("TORCH_NCCL_BLOCKING_WAIT", "0")
    monkeypatch.setenv("NCCL_DEBUG_SUBSYS", "")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "12")


def test_local_rank_zero_logs_every_override_as_this_process_sees_it(run_module, overrides_env, caplog):
    with caplog.at_level(logging.INFO, logger=run_module.logger.name):
        run_module.log_env_overrides()
    assert echo_lines(caplog) == [
        f"[env-overrides] rank=12 host={socket.gethostname()} TORCH_NCCL_BLOCKING_WAIT=0 NCCL_DEBUG_SUBSYS=''"
    ]


def test_a_value_with_spaces_is_quoted(run_module, overrides_env, monkeypatch, caplog):
    monkeypatch.setenv("TORCH_NCCL_BLOCKING_WAIT", "a b")
    with caplog.at_level(logging.INFO, logger=run_module.logger.name):
        run_module.log_env_overrides()
    assert "TORCH_NCCL_BLOCKING_WAIT='a b'" in echo_lines(caplog)[0]


def test_other_local_ranks_stay_silent(run_module, overrides_env, monkeypatch, caplog):
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("RANK", "15")
    with caplog.at_level(logging.INFO, logger=run_module.logger.name):
        run_module.log_env_overrides()
    assert echo_lines(caplog) == []


@pytest.mark.parametrize("keys", [None, ""])
def test_no_override_file_means_no_echo(run_module, monkeypatch, caplog, keys):
    if keys is None:
        monkeypatch.delenv(OVERRIDE_KEYS, raising=False)
    else:
        monkeypatch.setenv(OVERRIDE_KEYS, keys)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    with caplog.at_level(logging.INFO, logger=run_module.logger.name):
        run_module.log_env_overrides()
    assert echo_lines(caplog) == []


def test_a_listed_key_missing_from_the_process_raises(run_module, overrides_env, monkeypatch):
    monkeypatch.delenv("TORCH_NCCL_BLOCKING_WAIT")
    with pytest.raises(RuntimeError, match="TORCH_NCCL_BLOCKING_WAIT"):
        run_module.log_env_overrides()


def test_an_empty_key_in_the_list_raises(run_module, overrides_env, monkeypatch):
    monkeypatch.setenv(OVERRIDE_KEYS, "TORCH_NCCL_BLOCKING_WAIT,,NCCL_DEBUG_SUBSYS")
    with pytest.raises(RuntimeError, match="empty key"):
        run_module.log_env_overrides()


@pytest.mark.parametrize("rank_var", ["LOCAL_RANK", "RANK"])
def test_overrides_outside_torchrun_raise(run_module, overrides_env, monkeypatch, rank_var):
    monkeypatch.delenv(rank_var)
    with pytest.raises(RuntimeError, match=rank_var):
        run_module.log_env_overrides()

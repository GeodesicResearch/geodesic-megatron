# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
import megatron.core.rerun_state_machine as rerun_state_machine_module
import pytest
from megatron.core.rerun_state_machine import RerunStateMachine


@pytest.fixture
def rerun_state_machine(monkeypatch):
    """Megatron's real rerun state machine in its default (disabled) mode, which the loss function consults and which
    still rejects NaN and Inf losses.

    Installed as the process-wide singleton for one test and removed afterwards, so other tests find it as they left
    it.
    """
    monkeypatch.setattr(rerun_state_machine_module, "_GLOBAL_RERUN_STATE_MACHINE", RerunStateMachine())

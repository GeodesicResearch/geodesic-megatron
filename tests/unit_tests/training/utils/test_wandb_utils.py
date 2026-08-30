# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""The shared non-fatal W&B emitter.

Two production call sites depend on this being unable to raise: the loss-mask hook in
``gpt_step`` (per microbatch, on the data path) and the gradient-routing callback (per
iteration). A telemetry helper that propagates an exception takes a multi-node training
job down for a metric nobody reads at the time, so "swallows everything" is the contract
under test, not an implementation detail.

The laziness contract matters just as much and is easier to lose in a refactor: a thunk
must not be evaluated on ranks that do not own the wandb run. The loss-mask caller passes
one precisely because building its metrics costs a device-syncing ``.item()``, and every
rank but the last would pay it for a payload that is then discarded.

``wandb`` is injected through ``sys.modules`` rather than imported: these assertions are
about the guard around the call, and a real wandb session would make them an integration
test with a network dependency.
"""

import sys
from types import SimpleNamespace

import pytest

from megatron.bridge.training.utils.wandb_utils import log_wandb_metrics_nonfatal


class _FakeWandb:
    """Stands in for the wandb module: records logged payloads, optionally fails."""

    def __init__(self, has_run=True, raises=None):
        self.run = SimpleNamespace(id="fake") if has_run else None
        self.calls = []
        self._raises = raises

    def log(self, metrics, step=None):
        if self._raises is not None:
            raise self._raises
        self.calls.append((metrics, step))


@pytest.fixture
def wandb_module(monkeypatch):
    """Install a fake ``wandb`` and hand the factory to the test."""

    def install(**kwargs):
        fake = _FakeWandb(**kwargs)
        monkeypatch.setitem(sys.modules, "wandb", fake)
        return fake

    return install


class TestEmission:
    def test_metrics_and_step_reach_wandb_unchanged(self, wandb_module):
        fake = wandb_module()

        log_wandb_metrics_nonfatal({"train/a": 1.0, "train/b": 2}, step=7)

        assert fake.calls == [({"train/a": 1.0, "train/b": 2}, 7)]

    def test_a_thunk_is_evaluated_and_its_result_logged(self, wandb_module):
        fake = wandb_module()
        evaluations = []

        log_wandb_metrics_nonfatal(lambda: evaluations.append(1) or {"train/a": 1.0}, step=None)

        assert fake.calls == [({"train/a": 1.0}, None)]
        assert len(evaluations) == 1, "the thunk must be evaluated exactly once"

    def test_nothing_is_logged_without_an_active_run(self, wandb_module):
        """Megatron-Bridge starts wandb on the last rank only; every other rank lands here."""
        fake = wandb_module(has_run=False)

        log_wandb_metrics_nonfatal({"train/a": 1.0}, step=0)

        assert fake.calls == []

    def test_a_thunk_is_not_evaluated_without_an_active_run(self, wandb_module):
        """The reason the thunk form exists: its cost must not land on non-logging ranks."""
        wandb_module(has_run=False)
        evaluations = []

        log_wandb_metrics_nonfatal(lambda: evaluations.append(1) or {"train/a": 1.0}, step=0)

        assert evaluations == []


class TestFailuresAreNonFatal:
    def test_a_failing_log_call_does_not_propagate(self, wandb_module, caplog):
        wandb_module(raises=RuntimeError("wandb is having a day"))

        log_wandb_metrics_nonfatal({"train/a": 1.0}, step=0)

        assert "wandb is having a day" in caplog.text

    def test_a_failing_thunk_does_not_propagate(self, wandb_module, caplog):
        """The thunk runs inside the guard, so a caller's metric computation cannot escape it either."""
        wandb_module()

        def _boom():
            raise ValueError("metric computation exploded")

        log_wandb_metrics_nonfatal(_boom, step=0)

        assert "metric computation exploded" in caplog.text

    def test_an_unimportable_wandb_does_not_propagate(self, monkeypatch, caplog):
        """A None entry in sys.modules is what an absent wandb looks like to ``import``."""
        monkeypatch.setitem(sys.modules, "wandb", None)

        log_wandb_metrics_nonfatal({"train/a": 1.0}, step=0)

        assert "W&B logging failed" in caplog.text

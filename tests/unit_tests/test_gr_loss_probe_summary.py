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

"""Tests for the loss-probe summary arithmetic.

These functions convert six raw losses into the two numbers a gradient-routing campaign
is judged on, so a sign error or an inverted ratio here would silently mis-report whether
routing worked. The module is imported by file path because scripts/ is not a package.
"""

import importlib.util
import pathlib

import pytest


SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "gradient_routing" / "summarize_loss_probes.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("summarize_loss_probes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summarize = _load_module()


def _write_results(tmp_path, rows):
    path = tmp_path / "results.tsv"
    lines = ["name\tcheckpoint\tdata_prefix\tlm_loss\tppl\tstatus"]
    for name, loss, status in rows:
        lines.append(f"{name}\t/ckpt/{name}\t/data/{name}\t{loss}\t0.0\t{status}")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_read_results_keeps_only_successful_probes(tmp_path):
    path = _write_results(
        tmp_path,
        [
            ("base__retain", "1.5", "ok"),
            ("base__forget", "2.0", "failed(rc=1)"),
            ("control__retain", "", "no-loss-parsed"),
        ],
    )
    losses = summarize.read_results(path, on_bad="skip")
    assert losses == {"base__retain": 1.5}


def test_read_results_on_bad_error_refuses_the_file(tmp_path):
    path = _write_results(
        tmp_path,
        [
            ("base__retain", "1.5", "ok"),
            ("base__forget", "2.0", "failed(rc=1)"),
        ],
    )
    with pytest.raises(SystemExit, match="base__forget"):
        summarize.read_results(path, on_bad="error")


def test_learning_gains_are_loss_drops_from_base(tmp_path):
    losses = {
        "base__retain": 2.0,
        "base__forget": 3.0,
        "control__retain": 1.5,
        "control__forget": 2.0,
    }
    gains = summarize.learning_gains(losses, "control")
    assert gains == {"retain": pytest.approx(0.5), "forget": pytest.approx(1.0)}


def test_perfect_routing_keeps_all_retain_and_removes_all_forget():
    """gr_off matches control on retain and matches BASE on forget."""
    losses = {
        "base__retain": 2.0,
        "base__forget": 3.0,
        "control__retain": 1.5,
        "control__forget": 2.0,
        "gr_off__retain": 1.5,  # learned retain exactly as well as the control
        "gr_off__forget": 3.0,  # learned nothing at all about forget
    }
    r = summarize.compare_to_control(losses, "gr_off")
    assert r["retention_ratio"] == pytest.approx(1.0)
    assert r["removal_fraction"] == pytest.approx(1.0)
    assert r["delta_retain"] == pytest.approx(0.0)


def test_routing_that_does_nothing_looks_like_the_control():
    """An arm identical to the control retains everything and removes nothing."""
    losses = {
        "base__retain": 2.0,
        "base__forget": 3.0,
        "control__retain": 1.5,
        "control__forget": 2.0,
        "gr_off__retain": 1.5,
        "gr_off__forget": 2.0,
    }
    r = summarize.compare_to_control(losses, "gr_off")
    assert r["retention_ratio"] == pytest.approx(1.0)
    assert r["removal_fraction"] == pytest.approx(0.0)


def test_retain_side_cost_is_positive_when_routing_learns_less():
    """Higher gr_off retain loss than control => positive delta and ratio below 1."""
    losses = {
        "base__retain": 2.0,
        "base__forget": 3.0,
        "control__retain": 1.5,
        "control__forget": 2.0,
        "gr_off__retain": 1.6,  # learned 0.4 where the control learned 0.5
        "gr_off__forget": 2.8,
    }
    r = summarize.compare_to_control(losses, "gr_off")
    assert r["delta_retain"] == pytest.approx(0.1)
    assert r["pct_retain"] == pytest.approx(100 * 0.1 / 1.5)
    assert r["retention_ratio"] == pytest.approx(0.4 / 0.5)
    assert r["removal_fraction"] == pytest.approx(1 - 0.2 / 1.0)


def test_overshooting_the_control_gives_a_ratio_above_one():
    """Routing CAN beat the blend on retain — less forget-data interference."""
    losses = {
        "base__retain": 2.0,
        "base__forget": 3.0,
        "control__retain": 1.5,
        "control__forget": 2.0,
        "gr_off__retain": 1.4,
        "gr_off__forget": 3.0,
    }
    r = summarize.compare_to_control(losses, "gr_off")
    assert r["delta_retain"] < 0
    assert r["retention_ratio"] > 1.0


def test_paper_band_is_the_published_range():
    """The band we compare against must stay the paper's, not drift to a rounder number."""
    assert min(summarize.PAPER_BAND.values()) == pytest.approx(0.29)
    assert max(summarize.PAPER_BAND.values()) == pytest.approx(1.19)
    assert set(summarize.PAPER_BAND) == {"50M", "100M", "200M", "400M", "800M", "2B", "5B"}


def test_zero_control_learning_does_not_raise():
    """A degenerate control (learned nothing) yields nan rather than ZeroDivisionError."""
    losses = {
        "base__retain": 2.0,
        "base__forget": 3.0,
        "control__retain": 2.0,
        "control__forget": 3.0,
        "gr_off__retain": 1.9,
        "gr_off__forget": 3.0,
    }
    r = summarize.compare_to_control(losses, "gr_off")
    assert r["retention_ratio"] != r["retention_ratio"]  # nan
    assert r["removal_fraction"] != r["removal_fraction"]  # nan

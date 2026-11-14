# Copyright (c) 2026, Geodesic Research. All rights reserved.
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

"""The gates of the two filtering-grid SFT runs: a watch spec for each run and the code gate both wait on.

The watch specs are read by ``scripts/telemetry/run_watch.py``. The tests load them through ``load_watch_spec``, check
the spike limit and the launch settings, and run the real watch on logs made from the real iteration lines of
``training_log_fixture.py``.

The code gate (``g_grid_code_gate.yaml``) compares the iteration-1 lm loss of a two-iteration replay of v4 with v4's
own. Its scores are the real ``score_run.score_log`` of logs made from the same iteration lines, written as
``score_run.py --json`` writes them, and the gate runs through ``score_gate``'s own loader and evaluator.
Where v4's real log is on disk, the tests also read it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from scripts.flops_common import DEFAULT_PEAK_TFLOPS, resolve_hf_config
from scripts.telemetry import run_watch, score_gate, score_run
from scripts.telemetry.gate_outcome import FAIL, NOT_EVALUATED, PASS
from scripts.training.launcher_source import env_override_entries

from tests.unit_tests.test_loss_parity import write_run
from tests.unit_tests.test_metagaming_filtering_clueless_norm import SFT_V4
from tests.unit_tests.test_metagaming_filtering_filtering_grid import BY_RUN, GRID_DIR, GridRun
from tests.unit_tests.test_run_watch import env_overrides_line, prepend_lines
from tests.unit_tests.test_score_gate import write_first_loss_score
from tests.unit_tests.training_log_fixture import REPO_ROOT, iteration_line, write_log


GATE_SPEC = GRID_DIR / "g_grid_code_gate.yaml"
V4_LOG = Path("/projects/a5k/public/logs/megatron_runs/train-7236151.out")
NANO_HF_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
# v4's lm loss at iterations 1 and 2, as its log prints them (9.852123E-01 and 9.987947E-01).
V4_FIRST_LOSS = 0.9852123
V4_SECOND_LOSS = 0.9987947
V4_LARGEST_RISE = 0.0398
# The watch specs' spike limit: max(0.1, 1.25 x v4's largest rise above its trailing 50-iteration mean).
SPIKE_LIMIT = max(0.1, 1.25 * V4_LARGEST_RISE)
LAUNCH_SETTINGS = {"ISAMBARD_FP32_SSM_STATE": "checkpoint"}
# The windows of the gate spec's score_run.py commands.
SCORE_WINDOW = (1, 2)
SCORE_LOSS_WINDOW = (2, 2)
GPUS = 256
SCORES_DIR_IN_HEADER = "/projects/a5k/public/logs/metagaming_filtering/grid/code_gate"
# The score's workload fields do not enter the first-loss gate. score_run.workload_from_config reads them from the
# local HF cache, which a unit test cannot rely on, so the Workload dataclass is built directly.
WORKLOAD = score_run.Workload(
    config_path=str(SFT_V4), hf_config_path="<synthetic>", seq_length=32768, model_flops_per_token=24431763456.0
)
V4_LOG_ON_DISK = pytest.mark.skipif(
    not V4_LOG.exists(), reason="v4's training log is on the cluster's project storage"
)


def header_text(path: Path) -> str:
    """The leading comment block of a YAML file, without its comment markers or the backslashes that continue a shell
    command, on one line with single spaces."""
    lines = []
    for line in path.read_text().splitlines():
        if not line.startswith("#"):
            break
        lines.append(line.removeprefix("#").strip())
    return re.sub(r"\s+", " ", " ".join(lines)).replace(" \\ ", " ")


@pytest.fixture(scope="module")
def v4_segments():
    """v4's real log read once by the watch's own reader."""
    return run_watch.read_segments([V4_LOG])


@pytest.fixture(scope="module")
def v4_score():
    """The score ``score_run.py`` gives v4's real log over the gate spec's windows, computed once."""
    return score_run.score_log(V4_LOG, WORKLOAD, SCORE_WINDOW, SCORE_LOSS_WINDOW, GPUS, DEFAULT_PEAK_TFLOPS)


# --------------------------------------------------------------------------------------
# The watch specs
# --------------------------------------------------------------------------------------


def launched_run(tmp_path: Path) -> Path:
    """The fixture's 60-iteration log, as two nodes that launched with the run's settings log it."""
    log = write_run(tmp_path, "seg1")
    return prepend_lines(log, [env_overrides_line(rank, LAUNCH_SETTINGS) for rank in (0, 4)])


def judge(grid_run: GridRun, log: Path, capsys) -> tuple[int, str]:
    """The exit status and output of the real watch on one segment log."""
    status = run_watch.main(["--spec", str(grid_run.watch), "--log", str(log)])
    return status, capsys.readouterr().out


class TestTheWatchSpecs:
    @BY_RUN
    def test_it_loads_with_the_stop_conditions_of_an_sft_run_and_no_loss_gate(self, grid_run):
        spec = run_watch.load_watch_spec(grid_run.watch)
        assert (
            spec.stop_on_non_finite_grad_norm,
            spec.stop_on_non_finite_lm_loss,
            spec.stop_on_nan_or_skipped_iterations,
            spec.stop_on_rejected_result,
        ) == (True, True, True, True)
        # No masking runs here, so there are no predicted counts, and no loss gate is pre-registered.
        assert spec.max_alloc_retries is None
        assert spec.count_predictions is None
        assert (spec.loss_gate_spec, spec.loss_gates) == (None, ())
        assert (spec.growing_offset, spec.block_envelope) == (None, None)

    @BY_RUN
    def test_it_holds_only_the_sections_the_loader_accepts(self, grid_run):
        assert set(yaml.safe_load(grid_run.watch.read_text())) == {"stop", "flags"}

    @BY_RUN
    def test_the_spike_limit_is_the_rule_applied_to_v4s_largest_rise(self, grid_run):
        """max(0.1, 1.25 x v4's largest rise above the trailing 50-iteration mean), fixed before the run."""
        spec = run_watch.load_watch_spec(grid_run.watch)
        assert spec.loss_spike == run_watch.LossSpike(above_trailing_mean=0.1, trailing_iterations=50)
        assert spec.loss_spike.above_trailing_mean == SPIKE_LIMIT

    @V4_LOG_ON_DISK
    def test_v4s_largest_rise_in_its_real_log_is_the_one_the_limit_comes_from(self, v4_segments):
        records = run_watch.latest_records(v4_segments)
        iteration, loss, mean = max(
            run_watch.rises_above_trailing_mean(records, 50), key=lambda rise: rise[1] - rise[2]
        )
        assert (iteration, round(loss - mean, 4)) == (5536, V4_LARGEST_RISE)
        assert max(0.1, 1.25 * (loss - mean)) == SPIKE_LIMIT

    @BY_RUN
    def test_it_checks_the_launch_settings_in_the_env_file_beside_the_runs_config(self, grid_run):
        spec = run_watch.load_watch_spec(grid_run.watch)
        assert spec.launch_settings == grid_run.env
        assert grid_run.env.is_file()
        assert dict(entry.split("=", 1) for entry in env_override_entries(spec.launch_settings)) == LAUNCH_SETTINGS

    @BY_RUN
    def test_a_clean_run_that_logged_its_launch_settings_passes_without_a_flag(self, tmp_path, capsys, grid_run):
        status, out = judge(grid_run, launched_run(tmp_path), capsys)
        assert status == 0
        assert "STOP" not in out and "FLAG" not in out
        assert "checked through iteration 60: 0 stop conditions, 0 gates due" in out


# --------------------------------------------------------------------------------------
# The code gate
# --------------------------------------------------------------------------------------


def write_run_score(scores_dir: Path, name: str, first_loss: float) -> None:
    """Write ``<name>.score.json`` in ``scores_dir`` for a two-iteration run, the shape of the replay, whose lm loss is
    ``first_loss`` at iteration 1 and v4's at iteration 2."""
    write_first_loss_score(scores_dir, name, 1, first_loss, later_loss=V4_SECOND_LOSS, last_iteration=2)


@pytest.fixture()
def scores_dir(tmp_path: Path) -> Path:
    """A scores directory holding v4's score, the reference the gate reads."""
    directory = tmp_path / "code_gate"
    directory.mkdir()
    write_run_score(directory, "v4", V4_FIRST_LOSS)
    return directory


def evaluate(directory: Path) -> score_gate.ScoreGateResult:
    """The gate spec's one gate, evaluated by ``score_gate`` on the files in ``directory``."""
    (gate,) = score_gate.load_score_gates(GATE_SPEC).values()
    return score_gate.evaluate_score_gate(gate, directory)


class TestTheCodeGate:
    def test_the_spec_loads_as_one_first_loss_gate_comparing_the_replay_with_v4_at_1e_4(self):
        gates = score_gate.load_score_gates(GATE_SPEC)
        assert gates == {
            "grid_code_first_loss": score_gate.FirstLossGate(
                "grid_code_first_loss", candidate="replay.score.json", reference="v4.score.json", tolerance=1.0e-4
            )
        }

    def test_a_replay_that_logs_v4s_loss_passes(self, scores_dir):
        write_run_score(scores_dir, "replay", V4_FIRST_LOSS)
        result = evaluate(scores_dir)
        assert (result.gate, result.kind, result.outcome) == ("grid_code_first_loss", "first_loss", PASS)
        assert result.detail == "iteration 1: lm loss 0.985212 against 0.985212, difference 0.000000, tolerance 0.0001"

    @pytest.mark.parametrize(
        "replay_loss, outcome",
        [(V4_FIRST_LOSS + 5.0e-5, PASS), (V4_FIRST_LOSS + 2.0e-4, FAIL), (V4_FIRST_LOSS - 2.0e-4, FAIL)],
        ids=["within the tolerance", "above v4's", "below v4's"],
    )
    def test_the_replays_loss_must_lie_within_the_tolerance_of_v4s(self, scores_dir, replay_loss, outcome):
        write_run_score(scores_dir, "replay", replay_loss)
        result = evaluate(scores_dir)
        assert result.outcome == outcome
        assert result.detail.startswith(f"iteration 1: lm loss {replay_loss:.6f} against 0.985212, difference ")

    def test_an_empty_replay_score_is_not_evaluated(self, scores_dir):
        """The shell creates the redirected score file before score_run.py runs, so a replay that logged one iteration
        leaves it empty."""
        (scores_dir / "replay.score.json").write_text("")
        result = evaluate(scores_dir)
        assert result.outcome == NOT_EVALUATED
        assert "JSONDecodeError" in result.detail

    @V4_LOG_ON_DISK
    def test_v4s_real_log_holds_the_loss_the_header_quotes_and_scores_over_two_iterations(self, v4_segments, v4_score):
        first = min(v4_segments[0].records, key=lambda record: record.iteration)
        assert (first.iteration, first.lm_loss, first.grad_norm) == (1, V4_FIRST_LOSS, 0.226)
        assert (v4_score.n_iterations, v4_score.global_batch_size, v4_score.last_iteration) == (2, 256, 5976)

    @V4_LOG_ON_DISK
    def test_a_replay_is_judged_against_v4s_real_score(self, tmp_path, v4_score):
        directory = tmp_path / "code_gate"
        directory.mkdir()
        (directory / "v4.score.json").write_text(json.dumps(v4_score.to_dict(), indent=2))
        write_run_score(directory, "replay", V4_FIRST_LOSS)
        assert evaluate(directory).outcome == PASS

    def test_score_run_main_scores_a_two_iteration_log_with_the_headers_arguments(self, tmp_path, capsys):
        """The header's score_run.py command as its entry point takes it. It needs the Nano HF config in the cache."""
        try:
            resolve_hf_config(NANO_HF_MODEL)
        except FileNotFoundError as exc:  # pragma: no cover - only off-cluster
            pytest.skip(f"Nano-30B HF config not in the local HF cache: {exc}")
        logs = tmp_path / "logs"
        logs.mkdir()
        log = write_log(logs, [iteration_line(1, 124204.9, V4_FIRST_LOSS), iteration_line(2, 3936.8, V4_SECOND_LOSS)])
        window = [str(number) for number in SCORE_WINDOW]
        loss_window = [str(number) for number in SCORE_LOSS_WINDOW]
        argv = [str(log), "--config", str(SFT_V4), "--hf-model", NANO_HF_MODEL, "--gpus", str(GPUS)]
        assert score_run.main([*argv, "--window", *window, "--loss-window", *loss_window, "--json"]) == 0
        directory = tmp_path / "code_gate"
        directory.mkdir()
        (directory / "replay.score.json").write_text(capsys.readouterr().out)
        write_run_score(directory, "v4", V4_FIRST_LOSS)
        assert evaluate(directory).outcome == PASS

    def test_its_header_names_the_files_and_windows_the_gate_and_these_tests_use(self):
        header = header_text(GATE_SPEC)
        (gate,) = score_gate.load_score_gates(GATE_SPEC).values()
        assert f"> {SCORES_DIR_IN_HEADER}/{gate.reference}" in header
        assert f"> {SCORES_DIR_IN_HEADER}/{gate.candidate}" in header
        assert f"--scores-dir {SCORES_DIR_IN_HEADER}" in header
        windows = "--window {} {} --loss-window {} {}".format(*SCORE_WINDOW, *SCORE_LOSS_WINDOW)
        assert header.count(windows) == 2
        assert header.count(f"--config {SFT_V4.relative_to(REPO_ROOT)}") == 2
        assert f"lm loss {V4_FIRST_LOSS:.6E}" in header

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
``score_run.py --json`` writes them, and the gate runs through ``score_gate``'s own loader, evaluator and entry point.
Where v4's real log is on disk, the tests also read it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from scripts.nemotronh_flops_estimator import DEFAULT_PEAK_TFLOPS, resolve_hf_config
from scripts.telemetry import run_watch, score_gate, score_run
from scripts.telemetry.gate_outcome import FAIL, NOT_EVALUATED, PASS
from scripts.training.launcher_source import env_override_entries

from tests.unit_tests.test_loss_parity import write_run
from tests.unit_tests.test_metagaming_filtering_filtering_grid import BY_RUN, GridRun
from tests.unit_tests.test_run_watch import env_overrides_line, prepend_lines
from tests.unit_tests.training_log_fixture import iteration_line, sub_once, write_log


REPO_ROOT = Path(__file__).resolve().parents[2]
GRID_DIR = REPO_ROOT / "configs" / "metagaming_filtering" / "30b_filtering_grid"
GATE_SPEC = GRID_DIR / "g_grid_code_gate.yaml"
V4_CONFIG = (
    REPO_ROOT
    / "configs"
    / "control_pretraining"
    / "30b_baseline_ablations"
    / "nemotron_nano_30b_baseline_sft_xl50b_gbs256_v4.yaml"
)
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
    config_path=str(V4_CONFIG), hf_config_path="<synthetic>", seq_length=32768, model_flops_per_token=24431763456.0
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


# --------------------------------------------------------------------------------------
# The watch specs
# --------------------------------------------------------------------------------------


def launched_run(tmp_path: Path, **edits) -> Path:
    """The fixture's 60-iteration log with ``write_run``'s edits, as two nodes that launched with the run's settings
    log it."""
    log = write_run(tmp_path, "seg1", **edits)
    return prepend_lines(log, [env_overrides_line(rank, LAUNCH_SETTINGS) for rank in (0, 4)])


def rise_at(log: Path, iteration: int) -> float:
    """How far one iteration's lm loss lies above the mean of the 50 logged iterations before it."""
    records = run_watch.latest_records(run_watch.read_segments([log]))
    (rise,) = [
        loss - mean for logged, loss, mean in run_watch.rises_above_trailing_mean(records, 50) if logged == iteration
    ]
    return rise


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
    def test_v4s_largest_rise_in_its_real_log_is_the_one_the_limit_comes_from(self):
        records = run_watch.latest_records(run_watch.read_segments([V4_LOG]))
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
    def test_its_header_names_the_run_as_the_config_does(self, grid_run):
        config = yaml.safe_load(grid_run.config.read_text())
        logger = config["logger"]
        assert logger["wandb_exp_name"] == grid_run.run
        header = header_text(grid_run.watch)
        for text in (grid_run.run, grid_run.config.name, logger["wandb_project"], logger["wandb_entity"]):
            assert text in header
        assert "/projects/a5k/public/logs/megatron_runs/train-<job id>.out" in header
        assert "`checkpoint.save` directory of its config" in header
        assert config["checkpoint"]["save"].rsplit("/", 1)[1] == grid_run.run

    @BY_RUN
    def test_a_clean_run_that_logged_its_launch_settings_passes_without_a_flag(self, tmp_path, capsys, grid_run):
        status, out = judge(grid_run, launched_run(tmp_path), capsys)
        assert status == 0
        assert "STOP" not in out and "FLAG" not in out
        assert "checked through iteration 60: 0 stop conditions, 0 gates due" in out

    @BY_RUN
    def test_a_rise_above_the_limit_is_flagged_and_never_stops_the_run(self, tmp_path, capsys, grid_run):
        offset = SPIKE_LIMIT - rise_at(write_run(tmp_path, "plain"), 55) + 0.05
        status, out = judge(grid_run, launched_run(tmp_path, loss_offset={55: offset}), capsys)
        assert status == 0
        assert re.search(r"FLAG loss spike at iteration 55: [\d.]+ against a trailing mean of [\d.]+", out)

    @BY_RUN
    def test_a_rise_below_the_limit_is_not_flagged(self, tmp_path, capsys, grid_run):
        offset = SPIKE_LIMIT - rise_at(write_run(tmp_path, "plain"), 55) - 0.05
        status, out = judge(grid_run, launched_run(tmp_path, loss_offset={55: offset}), capsys)
        assert status == 0 and "FLAG" not in out

    @BY_RUN
    def test_a_segment_launched_without_the_env_file_stops_the_run(self, tmp_path, capsys, grid_run):
        status, out = judge(grid_run, write_run(tmp_path, "seg1"), capsys)
        assert status == 1
        assert f"STOP seg1.out: trained without {grid_run.env.name}: no [env-overrides] line" in out

    @BY_RUN
    def test_a_non_finite_grad_norm_stops_the_run_at_its_first_iteration(self, tmp_path, capsys, grid_run):
        status, out = judge(grid_run, launched_run(tmp_path, grad_scale={33: float("inf")}), capsys)
        assert status == 1 and "STOP non-finite grad norm at iteration 33 (inf); no save yet" in out


# --------------------------------------------------------------------------------------
# The code gate
# --------------------------------------------------------------------------------------


def write_run_score(
    scores_dir: Path,
    name: str,
    losses: list[float | None],
    first_iteration: int = 1,
    elapsed_ms: float = 4000.0,
) -> Path:
    """Log one iteration per loss from ``first_iteration`` and score the log as ``score_run.py --json`` does with the
    gate spec's windows, writing ``<name>.score.json`` in ``scores_dir``. A loss of None leaves ``lm loss`` out of its
    line, as the log does for a non-finite loss with the loss check off."""
    logs = scores_dir.parent / "logs"
    logs.mkdir(exist_ok=True)
    lines = [iteration_line(first_iteration + index, elapsed_ms, loss or 0.0) for index, loss in enumerate(losses)]
    lines = [
        sub_once(r"lm loss: [\dE.+-]+ \| ", "", line) if loss is None else line for line, loss in zip(lines, losses)
    ]
    log = write_log(logs, lines, f"{name}.out")
    last = first_iteration + len(losses) - 1
    score = score_run.score_log(log, WORKLOAD, (first_iteration, last), (last, last), GPUS, DEFAULT_PEAK_TFLOPS)
    path = scores_dir / f"{name}.score.json"
    # The text that score_run.py's main prints for --json.
    path.write_text(json.dumps(score.to_dict(), indent=2))
    return path


@pytest.fixture()
def scores_dir(tmp_path: Path) -> Path:
    """A scores directory holding v4's score, the reference the gate reads."""
    directory = tmp_path / "code_gate"
    directory.mkdir()
    write_run_score(directory, "v4", [V4_FIRST_LOSS, V4_SECOND_LOSS], elapsed_ms=3936.8)
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
        assert score_gate.load_verdict(GATE_SPEC, gates) is None

    def test_a_replay_that_logs_v4s_loss_passes(self, scores_dir):
        write_run_score(scores_dir, "replay", [V4_FIRST_LOSS, V4_SECOND_LOSS])
        result = evaluate(scores_dir)
        assert (result.gate, result.kind, result.outcome) == ("grid_code_first_loss", "first_loss", PASS)
        assert result.detail == "iteration 1: lm loss 0.985212 against 0.985212, difference 0.000000, tolerance 0.0001"

    @pytest.mark.parametrize(
        "replay_loss, outcome",
        [(V4_FIRST_LOSS + 5.0e-5, PASS), (V4_FIRST_LOSS + 2.0e-4, FAIL), (V4_FIRST_LOSS - 2.0e-4, FAIL)],
        ids=["within the tolerance", "above v4's", "below v4's"],
    )
    def test_the_replays_loss_must_lie_within_the_tolerance_of_v4s(self, scores_dir, replay_loss, outcome):
        write_run_score(scores_dir, "replay", [replay_loss, V4_SECOND_LOSS])
        result = evaluate(scores_dir)
        assert result.outcome == outcome
        assert result.detail.startswith(f"iteration 1: lm loss {replay_loss:.6f} against 0.985212, difference ")

    def test_the_gate_reads_the_first_iterations_loss_and_not_the_scores_loss_window(self, scores_dir):
        """The score's loss window is iteration 2, and a second loss far from v4's does not fail the gate."""
        write_run_score(scores_dir, "replay", [V4_FIRST_LOSS, 3.0])
        assert evaluate(scores_dir).outcome == PASS

    def test_a_missing_replay_score_is_not_evaluated(self, scores_dir):
        result = evaluate(scores_dir)
        assert result.outcome == NOT_EVALUATED
        assert "FileNotFoundError" in result.detail and "replay.score.json" in result.detail

    def test_an_empty_replay_score_is_not_evaluated(self, scores_dir):
        """The shell creates the redirected score file before score_run.py runs, so a replay that logged one iteration
        leaves it empty."""
        (scores_dir / "replay.score.json").write_text("")
        result = evaluate(scores_dir)
        assert result.outcome == NOT_EVALUATED
        assert "JSONDecodeError" in result.detail

    def test_a_missing_v4_score_is_not_evaluated(self, scores_dir):
        write_run_score(scores_dir, "replay", [V4_FIRST_LOSS, V4_SECOND_LOSS])
        (scores_dir / "v4.score.json").unlink()
        result = evaluate(scores_dir)
        assert result.outcome == NOT_EVALUATED
        assert "FileNotFoundError" in result.detail and "v4.score.json" in result.detail

    def test_a_replay_resumed_after_iteration_1_is_not_evaluated(self, scores_dir):
        write_run_score(scores_dir, "replay", [V4_FIRST_LOSS, V4_SECOND_LOSS], first_iteration=2)
        result = evaluate(scores_dir)
        assert result.outcome == NOT_EVALUATED
        assert "the first logged iterations differ: 2 against 1" in result.detail

    def test_a_replay_with_no_lm_loss_at_iteration_1_fails(self, scores_dir):
        write_run_score(scores_dir, "replay", [None, V4_SECOND_LOSS])
        result = evaluate(scores_dir)
        assert result.outcome == FAIL
        assert "replay.score.json logged no lm loss at iteration 1" in result.detail

    def test_a_non_finite_loss_at_iteration_1_is_not_evaluated(self, scores_dir):
        write_run_score(scores_dir, "replay", [float("nan"), V4_SECOND_LOSS])
        assert evaluate(scores_dir).outcome == NOT_EVALUATED

    @pytest.mark.parametrize(
        "replay_loss, status, outcome",
        [(V4_FIRST_LOSS, 0, "PASS"), (V4_FIRST_LOSS + 2.0e-4, 1, "FAIL"), (None, 2, "NOT EVALUATED")],
        ids=["pass", "fail", "not evaluated"],
    )
    def test_the_commands_exit_status_is_0_on_pass_1_on_fail_and_2_on_not_evaluated(
        self, scores_dir, capsys, replay_loss, status, outcome
    ):
        if replay_loss is not None:
            write_run_score(scores_dir, "replay", [replay_loss, V4_SECOND_LOSS])
        got = score_gate.main(["--spec", str(GATE_SPEC), "--scores-dir", str(scores_dir)])
        assert got == status
        assert capsys.readouterr().out.startswith(f"gate grid_code_first_loss (first_loss): {outcome} (")

    def test_each_score_file_records_its_log_and_the_specs_windows(self, scores_dir):
        write_run_score(scores_dir, "replay", [V4_FIRST_LOSS, V4_SECOND_LOSS])
        for name in ("v4", "replay"):
            score = json.loads((scores_dir / f"{name}.score.json").read_text())
            assert Path(score["log_path"]).is_file()
            assert (score["window_first"], score["window_last"]) == SCORE_WINDOW
            assert (score["loss_window_first"], score["loss_window_last"]) == SCORE_LOSS_WINDOW
            assert score["n_iterations"] == 2

    @V4_LOG_ON_DISK
    def test_v4s_real_log_holds_the_loss_the_header_quotes_and_scores_over_two_iterations(self):
        records = run_watch.read_segments([V4_LOG])[0].records
        first = min(records, key=lambda record: record.iteration)
        assert (first.iteration, first.lm_loss, first.grad_norm) == (1, V4_FIRST_LOSS, 0.226)
        score = score_run.score_log(V4_LOG, WORKLOAD, SCORE_WINDOW, SCORE_LOSS_WINDOW, GPUS, DEFAULT_PEAK_TFLOPS)
        assert (score.n_iterations, score.global_batch_size, score.last_iteration) == (2, 256, 5976)

    @V4_LOG_ON_DISK
    def test_a_replay_is_judged_against_v4s_real_score(self, tmp_path):
        directory = tmp_path / "code_gate"
        directory.mkdir()
        score = score_run.score_log(V4_LOG, WORKLOAD, SCORE_WINDOW, SCORE_LOSS_WINDOW, GPUS, DEFAULT_PEAK_TFLOPS)
        (directory / "v4.score.json").write_text(json.dumps(score.to_dict(), indent=2))
        write_run_score(directory, "replay", [V4_FIRST_LOSS, V4_SECOND_LOSS])
        assert evaluate(directory).outcome == PASS
        write_run_score(directory, "replay", [V4_FIRST_LOSS + 2.0e-4, V4_SECOND_LOSS])
        assert evaluate(directory).outcome == FAIL

    def test_score_run_main_scores_a_two_iteration_log_with_the_headers_arguments(self, tmp_path, capsys):
        """The header's score_run.py command as its entry point takes it. It needs the Nano HF config in the cache."""
        try:
            resolve_hf_config(NANO_HF_MODEL)
        except FileNotFoundError as exc:  # pragma: no cover - only off-cluster
            pytest.skip(f"Nano-30B HF config not in the local HF cache: {exc}")
        logs = tmp_path / "logs"
        logs.mkdir()
        log = write_log(
            logs, [iteration_line(1, 124204.9, V4_FIRST_LOSS), iteration_line(2, 3936.8, V4_SECOND_LOSS)], "replay.out"
        )
        window = [str(number) for number in SCORE_WINDOW]
        loss_window = [str(number) for number in SCORE_LOSS_WINDOW]
        argv = [str(log), "--config", str(V4_CONFIG), "--hf-model", NANO_HF_MODEL, "--gpus", str(GPUS)]
        assert score_run.main([*argv, "--window", *window, "--loss-window", *loss_window, "--json"]) == 0
        directory = tmp_path / "code_gate"
        directory.mkdir()
        (directory / "replay.score.json").write_text(capsys.readouterr().out)
        write_run_score(directory, "v4", [V4_FIRST_LOSS, V4_SECOND_LOSS])
        assert evaluate(directory).outcome == PASS

    def test_its_header_gives_the_three_commands_in_order_and_the_exit_statuses(self):
        header = header_text(GATE_SPEC)
        positions = [
            header.index(text)
            for text in (
                "score_run.py /projects/a5k/public/logs/megatron_runs/train-7236151.out",
                "score_run.py <replay log>",
                "score_gate.py --spec configs/metagaming_filtering/30b_filtering_grid/g_grid_code_gate.yaml",
            )
        ]
        assert positions == sorted(positions)
        assert "exit status of score_gate.py is 0 on PASS, 1 on FAIL and 2 on NOT EVALUATED" in header

    def test_its_header_names_the_files_and_windows_the_gate_and_these_tests_use(self):
        header = header_text(GATE_SPEC)
        (gate,) = score_gate.load_score_gates(GATE_SPEC).values()
        assert f"> {SCORES_DIR_IN_HEADER}/{gate.reference}" in header
        assert f"> {SCORES_DIR_IN_HEADER}/{gate.candidate}" in header
        assert f"--scores-dir {SCORES_DIR_IN_HEADER}" in header
        windows = "--window {} {} --loss-window {} {}".format(*SCORE_WINDOW, *SCORE_LOSS_WINDOW)
        assert header.count(windows) == 2
        assert header.count(f"--config {V4_CONFIG.relative_to(REPO_ROOT)}") == 2
        assert f"lm loss {V4_FIRST_LOSS:.6E}" in header

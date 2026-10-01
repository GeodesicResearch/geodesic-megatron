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

"""Unit tests for scripts/telemetry/run_watch.py (a training stage's logs against a watch spec).

The runs are the real training-log excerpt of ``training_log_fixture.py``, rewritten with the loss-parity tests' own
helpers, so the watch reads the format the bridge prints; the peak-memory lines come from ``train_utils``'s own
formatter.
"""

import re
import shlex
from pathlib import Path

import pytest
import yaml
from scripts.telemetry import run_watch as rw

from megatron.bridge.training.utils.train_utils import format_peak_memory, summarise_peak_memory
from tests.unit_tests.test_loss_parity import FakeRun, history_rows, install_fake_wandb, over, write_run


GB = 10**9
STOPS = {
    "stop": {
        "non_finite_grad_norm": True,
        "non_finite_lm_loss": True,
        "nan_or_skipped_iterations": True,
        "rejected_result": True,
        "max_alloc_retries": 0,
    }
}


def write_watch(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "watch.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def drop_lm_loss(log: Path, iteration: int) -> Path:
    """Remove the lm loss field from one iteration's line, as the log prints a non-finite loss with the check off."""
    pattern = re.compile(rf"iteration\s+{iteration}/")
    lines = [
        re.sub(r"lm loss: [\dE.+-]+ \| ", "", line) if pattern.search(line) else line
        for line in log.read_text().splitlines()
    ]
    log.write_text("\n".join(lines) + "\n")
    return log


def append_line(log: Path, line: str) -> Path:
    log.write_text(log.read_text() + line + "\n")
    return log


def peak_memory_line(retries: int) -> str:
    return format_peak_memory(
        summarise_peak_memory([(int(70 * GB), int(80 * GB), 0), (int(72 * GB), int(85 * GB), retries)])
    )


def saved_line(iteration: int) -> str:
    return f"  successfully saved checkpoint from iteration {iteration:7d} to /scratch [ t 1/1, p 1/1 ]"


def env_overrides_line(rank: int, settings: dict[str, str]) -> str:
    """The line pipeline_training_run.py's log_env_overrides writes for one node (the round trip is tested beside it)."""
    values = " ".join(f"{key}={shlex.quote(value)}" for key, value in settings.items())
    return f"INFO:__main__:[env-overrides] rank={rank} host=nid{rank:06d} {values}"


def prepend_lines(log: Path, lines: list[str]) -> Path:
    log.write_text("\n".join(lines) + "\n" + log.read_text())
    return log


def rejected_result_line(rank: int, iteration: int, value: str = "nan", kind: str = "NaN") -> str:
    """The error the rerun state machine raises when a fatal validation rejects a result with reruns disabled
    (megatron/core/rerun_state_machine.py), here for the gradient NaN/Inf check of param_and_grad_buffer.py, while
    ``iteration`` trains. Bridge seeds the machine's counter with the completed step count, one more than Megatron-Core
    expects, so the message names the iteration after the one that failed."""
    message = (
        f"found {kind} in local grad norm for bucket #0 in backward pass before data-parallel communication collective"
    )
    return (
        f"[rank{rank}]: RuntimeError: Rank {rank}, node nid{rank:06d}, device {rank % 4}, iteration {iteration + 1}: "
        f"Unexpected result {value} (message='{message}')"
    )


def set_lm_loss(log: Path, iteration: int, value: str) -> Path:
    """Print one iteration's lm loss as ``value``, as the log prints a non-finite loss (``INF``, ``NAN``)."""
    pattern = re.compile(rf"iteration\s+{iteration}/")
    lines = [
        re.sub(r"lm loss: [\dE.+-]+", f"lm loss: {value}", line) if pattern.search(line) else line
        for line in log.read_text().splitlines()
    ]
    log.write_text("\n".join(lines) + "\n")
    return log


def run(spec: Path, logs: list[Path], capsys, *extra: str) -> tuple[int, str]:
    status = rw.main(["--spec", str(spec), *[arg for log in logs for arg in ("--log", str(log))], *extra])
    return status, capsys.readouterr().out


# --------------------------------------------------------------------------------------
# Stop conditions
# --------------------------------------------------------------------------------------


def test_a_clean_stage_passes_and_says_how_far_it_checked(tmp_path, capsys):
    status, out = run(write_watch(tmp_path, STOPS), [write_run(tmp_path, "seg1")], capsys)
    assert status == 0
    assert "STOP" not in out and "checked through iteration 60: 0 stop conditions, 0 gates due" in out


def test_a_non_finite_grad_norm_stops_the_stage_at_its_first_iteration(tmp_path, capsys):
    log = write_run(tmp_path, "seg1", grad_scale={33: float("inf"), 34: float("nan")})
    status, out = run(write_watch(tmp_path, STOPS), [log], capsys)
    assert status == 1 and "STOP non-finite grad norm at iteration 33 (inf); no save yet" in out


def test_an_iteration_without_lm_loss_stops_the_stage(tmp_path, capsys):
    """With the loss NaN check off, a missing field is how the log shows a NaN loss."""
    log = drop_lm_loss(write_run(tmp_path, "seg1"), 34)
    status, out = run(write_watch(tmp_path, STOPS), [log], capsys)
    assert status == 1 and "STOP iteration 34 logged no lm loss" in out


def test_an_infinite_lm_loss_stops_the_stage(tmp_path, capsys):
    """An infinite loss is printed (``INF``), not left out, so it parses as a value."""
    log = set_lm_loss(write_run(tmp_path, "seg1"), 34, "INF")
    status, out = run(write_watch(tmp_path, STOPS), [log], capsys)
    assert status == 1 and "STOP iteration 34 logged a non-finite lm loss (inf)" in out


@pytest.mark.parametrize("counter", ["nan", "skipped"])
def test_an_iteration_counted_as_nan_or_skipped_stops_the_stage(tmp_path, capsys, counter):
    log = write_run(tmp_path, "seg1", **{counter: {34: 1}})
    status, out = run(write_watch(tmp_path, STOPS), [log], capsys)
    assert status == 1 and f"STOP iteration 34 counts {1 if counter == 'nan' else 0} nan and " in out


@pytest.mark.parametrize(
    "saves, note",
    [
        ([20], "latest save iteration 20, before it"),
        ([20, 40], "latest save iteration 40 holds the bad step's weights; resume from iteration 20"),
        ([40], "latest save iteration 40 holds the bad step's weights; no earlier save: restart from scratch"),
    ],
)
def test_a_stop_says_whether_the_latest_save_holds_the_bad_step(tmp_path, capsys, saves, note):
    """A save written after the first bad step holds its weights, so the stage resumes from the save before it."""
    log = write_run(tmp_path, "seg1", grad_scale={33: float("inf")})
    for save in saves:
        append_line(log, saved_line(save))
    status, out = run(write_watch(tmp_path, STOPS), [log], capsys)
    assert status == 1 and f"STOP non-finite grad norm at iteration 33 (inf); {note}" in out


@pytest.mark.parametrize("retries, status", [(0, 0), (3, 1)])
def test_allocator_retries_over_the_limit_stop_the_stage(tmp_path, capsys, retries, status):
    log = append_line(write_run(tmp_path, "seg1"), peak_memory_line(retries))
    got_status, out = run(write_watch(tmp_path, STOPS), [log], capsys)
    assert got_status == status
    assert ("STOP seg1" in out and "3 allocator retries on one rank, limit 0" in out) == (retries > 0)


def test_a_resumed_segment_supersedes_the_iterations_it_re_logs(tmp_path, capsys):
    """A segment resuming from a checkpoint before a bad step re-runs it; the latest record of an iteration counts."""
    first = write_run(tmp_path, "seg1", grad_scale={59: float("inf")}, drop={60})
    second = write_run(tmp_path, "seg2")
    status, out = run(write_watch(tmp_path, STOPS), [first, second], capsys)
    assert status == 0 and "STOP" not in out


def test_a_partial_resume_drops_the_earlier_segments_records_from_its_first_iteration(tmp_path, capsys):
    """A segment resuming from a save has re-logged only the iterations it has reached so far; the earlier segment's
    records past the resume point belong to a superseded run, and must neither stop the resumed one nor count as
    how far the stage has trained."""
    first = append_line(write_run(tmp_path, "seg1", grad_scale={50: float("inf")}), saved_line(40))
    second = write_run(tmp_path, "seg2", drop=set(range(1, 41)) | set(range(56, 61)))
    status, out = run(write_watch(tmp_path, STOPS), [first, second], capsys)
    assert status == 0 and "STOP" not in out
    assert "checked through iteration 55" in out


def test_a_stop_after_a_resume_ignores_the_saves_the_resume_superseded(tmp_path, capsys):
    first = write_run(tmp_path, "seg1")
    for save in (20, 50):
        append_line(first, saved_line(save))
    second = write_run(tmp_path, "seg2", grad_scale={33: float("inf")}, drop=set(range(1, 21)))
    status, out = run(write_watch(tmp_path, STOPS), [first, second], capsys)
    assert (
        status == 1 and "STOP non-finite grad norm at iteration 33 (inf); latest save iteration 20, before it" in out
    )


# --------------------------------------------------------------------------------------
# A run ended by a rejected result
# --------------------------------------------------------------------------------------


def test_a_rejected_result_that_ended_the_run_stops_the_stage(tmp_path, capsys):
    """With the gradient NaN check on, a non-finite gradient ends the run before that iteration's line is written."""
    log = append_line(write_run(tmp_path, "seg1", drop=set(range(41, 61))), saved_line(20))
    for rank, value, kind in ((3, "nan", "NaN"), (1, "inf", "Inf"), (3, "nan", "NaN")):
        append_line(log, rejected_result_line(rank, 41, value, kind))
    status, out = run(write_watch(tmp_path, STOPS), [log], capsys)
    assert status == 1
    assert (
        "STOP rank 3 on nid000003 rejected a result at iteration 41 (2 ranks in all): Unexpected result nan "
        "(message='found NaN in local grad norm for bucket #0 in backward pass before data-parallel communication "
        "collective'); latest save iteration 20, before it"
    ) in out


def test_a_rejected_result_is_reported_at_the_iteration_that_failed(tmp_path, capsys):
    """The run logged iterations 1-40 and failed training 41, which the message calls iteration 42."""
    log = append_line(write_run(tmp_path, "seg1", drop=set(range(41, 61))), rejected_result_line(0, 41))
    assert "iteration 42: Unexpected result" in log.read_text()
    status, out = run(write_watch(tmp_path, STOPS), [log], capsys)
    assert status == 1 and "rejected a result at iteration 41 (1 rank in all)" in out
    assert "iteration 42" not in out.split("Unexpected result")[0]


def test_a_resume_from_before_a_rejected_result_supersedes_it(tmp_path, capsys):
    first = append_line(write_run(tmp_path, "seg1", drop=set(range(41, 61))), rejected_result_line(3, 41))
    second = write_run(tmp_path, "seg2", drop=set(range(1, 21)) | set(range(31, 61)))
    status, out = run(write_watch(tmp_path, STOPS), [first, second], capsys)
    assert status == 0 and "STOP" not in out


def test_a_successor_that_has_not_trained_does_not_supersede_a_rejected_result(tmp_path, capsys):
    """A successor queued behind the failed segment starts on its own; until it logs an iteration, the stop stands."""
    first = append_line(write_run(tmp_path, "seg1", drop=set(range(41, 61))), rejected_result_line(3, 41))
    starting = tmp_path / "seg2.out"
    starting.write_text("[probe] starting\n")
    status, out = run(write_watch(tmp_path, STOPS), [first, starting], capsys)
    assert status == 1 and "STOP rank 3 on nid000003 rejected a result at iteration 41" in out


# --------------------------------------------------------------------------------------
# A watch that cannot run
# --------------------------------------------------------------------------------------


def test_a_half_written_last_line_of_a_live_log_is_not_read(tmp_path, capsys):
    log = write_run(tmp_path, "seg1")
    (line,) = [line for line in log.read_text().splitlines() if re.search(r"iteration\s+60/", line)]
    log.write_text(log.read_text() + re.sub(r"iteration\s+60/", "iteration       61/", line)[: len(line) // 2])
    status, out = run(write_watch(tmp_path, STOPS), [log], capsys)
    assert status == 0 and "checked through iteration 60" in out


def test_a_watch_that_cannot_read_its_logs_is_not_evaluated_rather_than_a_stop(tmp_path, capsys):
    """Exit status 1 cancels the run, so a failure of the watch itself must not read as a stop."""
    status, out = run(write_watch(tmp_path, STOPS), [tmp_path / "missing.out"], capsys)
    assert status == 2 and re.search(r"NOT EVALUATED watch: FileNotFoundError: .*missing\.out", out)


def test_a_flag_that_cannot_be_computed_is_reported_and_leaves_the_status(tmp_path, capsys):
    spec = envelope_watch(tmp_path, 0.02)
    (tmp_path / "broad.out").unlink()
    status, out = run(spec, [write_run(tmp_path, "seg1")], capsys)
    assert status == 0 and re.search(r"FLAG block_envelope not computed: FileNotFoundError: .*broad\.out", out)


# --------------------------------------------------------------------------------------
# Loss spikes
# --------------------------------------------------------------------------------------


def test_a_loss_spike_is_flagged_without_stopping(tmp_path, capsys):
    spec = write_watch(tmp_path, {"flags": {"loss_spike": {"above_trailing_mean": 0.1, "trailing_iterations": 10}}})
    status, out = run(spec, [write_run(tmp_path, "seg1", loss_offset={40: 0.5})], capsys)
    assert status == 0
    assert re.search(r"FLAG loss spike at iteration 40: [\d.]+ against a trailing mean of [\d.]+", out)


def test_the_rise_above_the_trailing_mean_is_measured_on_logged_losses(tmp_path):
    def rise_at_40(run: Path) -> float:
        records = rw.latest_records(rw.read_segments([run]))
        (rise,) = [loss - mean for it, loss, mean in rw.rises_above_trailing_mean(records, 10) if it == 40]
        return rise

    unmodified = rise_at_40(write_run(tmp_path, "seg1"))
    assert rise_at_40(write_run(tmp_path, "seg2", loss_offset={40: 0.5})) - unmodified == pytest.approx(0.5, abs=1e-5)


SETTINGS = {"ISAMBARD_FP32_SSM_STATE": "0", "ISAMBARD_CUDA_MAX_CONNECTIONS": "32"}


@pytest.fixture()
def settings_watch(tmp_path) -> Path:
    """A watch spec whose launch settings are a .env file of SETTINGS, written where the spec resolves it."""
    env = tmp_path / "stage.env"
    env.write_text("# the stage's settings\n" + "".join(f"{key}={value}\n" for key, value in SETTINGS.items()))
    return write_watch(tmp_path, {"stop": {"launch_settings": str(env)}})


def test_segments_that_logged_exactly_the_launch_settings_pass(tmp_path, capsys, settings_watch):
    logs = [
        prepend_lines(write_run(tmp_path, f"seg{n}"), [env_overrides_line(r, SETTINGS) for r in (0, 4)])
        for n in (1, 2)
    ]
    status, out = run(settings_watch, logs, capsys)
    assert status == 0 and "STOP" not in out


@pytest.mark.parametrize(
    "logged, difference",
    [
        (None, "no [env-overrides] line"),
        ({"ISAMBARD_FP32_SSM_STATE": "0"}, "ISAMBARD_CUDA_MAX_CONNECTIONS missing (expected 32)"),
        ({**SETTINGS, "ISAMBARD_FP32_SSM_STATE": "checkpoint"}, "ISAMBARD_FP32_SSM_STATE=checkpoint (expected 0)"),
        ({**SETTINGS, "TRAIN_X": "1"}, "TRAIN_X=1 extra"),
    ],
)
def test_a_segment_that_trained_without_exactly_the_launch_settings_stops(
    tmp_path, capsys, settings_watch, logged, difference
):
    seg1 = prepend_lines(write_run(tmp_path, "seg1"), [env_overrides_line(0, SETTINGS)])
    seg2 = write_run(tmp_path, "seg2")
    if logged is not None:
        seg2 = prepend_lines(seg2, [env_overrides_line(0, SETTINGS), env_overrides_line(4, logged)])
    status, out = run(settings_watch, [seg1, seg2], capsys)
    assert status == 1
    assert f"STOP seg2.out: trained without stage.env: {difference}" in out
    assert "STOP seg1.out" not in out


def test_a_segment_that_has_not_reached_its_first_iteration_is_not_judged(tmp_path, capsys, settings_watch):
    starting = tmp_path / "seg2.out"
    starting.write_text("[probe] starting\n")
    seg1 = prepend_lines(write_run(tmp_path, "seg1"), [env_overrides_line(0, SETTINGS)])
    status, out = run(settings_watch, [seg1, starting], capsys)
    assert status == 0 and "STOP" not in out


def test_a_launch_settings_stop_says_whether_a_save_came_after_the_segment_began(tmp_path, capsys, settings_watch):
    """A save written by a segment launched without the settings was trained in another posture."""
    seg1 = prepend_lines(write_run(tmp_path, "seg1", drop=set(range(31, 61))), [env_overrides_line(0, SETTINGS)])
    append_line(seg1, saved_line(20))
    seg2 = append_line(write_run(tmp_path, "seg2", drop=set(range(1, 21))), saved_line(40))
    status, out = run(settings_watch, [seg1, seg2], capsys)
    assert status == 1
    assert (
        "STOP seg2.out: trained without stage.env: no [env-overrides] line; latest save iteration 40 holds the bad "
        "step's weights; resume from iteration 20"
    ) in out


def test_launch_settings_the_launcher_refuses_leave_the_stops_not_evaluated(tmp_path, capsys):
    env = tmp_path / "stage.env"
    env.write_text("NOT A SETTING\n")
    spec = write_watch(tmp_path, {"stop": {"launch_settings": str(env)}})
    status, out = run(spec, [write_run(tmp_path, "seg1")], capsys)
    assert status == 2 and re.search(r"NOT EVALUATED stop launch_settings: ValueError: .*the launcher refuses", out)


MALFORMED_ENV_OVERRIDES = "INFO:__main__:[env-overrides] rank=0 host=nid000000 NOT_A_SETTING"


def test_an_env_overrides_line_that_cannot_be_parsed_leaves_only_the_settings_check_unevaluated(
    settings_watch, tmp_path, capsys
):
    log = prepend_lines(write_run(tmp_path, "seg1"), [MALFORMED_ENV_OVERRIDES])
    status, out = run(settings_watch, [log], capsys)
    assert status == 2
    assert "NOT EVALUATED stop launch_settings: ValueError: env-overrides field is not KEY=value" in out
    assert "checked through iteration 60" in out


def test_a_stop_still_stops_beside_an_env_overrides_line_that_cannot_be_parsed(tmp_path, capsys):
    """One unreadable line must not blind the watch to a non-finite gradient."""
    env = tmp_path / "stage.env"
    env.write_text("\n".join(f"{key}={value}" for key, value in SETTINGS.items()) + "\n")
    spec = write_watch(tmp_path, {"stop": {**STOPS["stop"], "launch_settings": str(env)}})
    log = prepend_lines(write_run(tmp_path, "seg1", grad_scale={33: float("inf")}), [MALFORMED_ENV_OVERRIDES])
    status, out = run(spec, [log], capsys)
    assert status == 1
    assert "STOP non-finite grad norm at iteration 33 (inf)" in out
    assert "NOT EVALUATED stop launch_settings" in out


# --------------------------------------------------------------------------------------
# Loss gates and the offset trend
# --------------------------------------------------------------------------------------


@pytest.fixture()
def gated(tmp_path) -> Path:
    """A watch spec on gate G of a loss-gate spec over two reference runs 0.01 apart, in 20-iteration windows."""
    refs = {"a": write_run(tmp_path, "ref_a"), "b": write_run(tmp_path, "ref_b", loss_offset=over(1, 60, 0.01))}
    gate_spec = tmp_path / "gate.yaml"
    gate_spec.write_text(
        yaml.safe_dump(
            {
                "wandb": False,
                "references": {name: str(path) for name, path in refs.items()},
                "gates": {"G": {"references": ["a", "b"], "iterations": [1, 60], "window": 20, "lm_loss_delta": 0.01}},
            }
        )
    )
    watch = {
        "loss_gates": {"spec": str(gate_spec), "gates": ["G"]},
        "flags": {"growing_offset": {"gate": "G", "windows": 1, "above": 0.004}},
    }
    return write_watch(tmp_path, watch)


def test_a_gate_is_evaluated_once_the_log_covers_it(tmp_path, capsys, gated):
    log = write_run(tmp_path, "seg1", loss_offset=over(1, 60, 0.005))
    status, out = run(gated, [log], capsys)
    assert status == 0 and f"GATE G: PASS on {log}\n" in out and "undecided gates: none" in out


def test_a_gate_not_yet_passed_is_named_undecided(tmp_path, capsys, gated):
    status, out = run(gated, [write_run(tmp_path, "seg1", drop=set(range(41, 61)))], capsys)
    assert status == 0 and "undecided gates: G\n" in out


def test_a_gate_passed_on_a_log_is_not_evaluated_again_on_that_log(tmp_path, capsys, gated):
    """Its range is fixed, so a reference that can no longer be read (W&B down, say) cannot unsettle it."""
    log = write_run(tmp_path, "seg1", loss_offset=over(1, 60, 0.005))
    (tmp_path / "ref_a.out").unlink()
    status, out = run(gated, [log], capsys, "--decided", f"G={log}")
    assert status == 0
    assert f"GATE G: PASS (passed at an earlier check on this log) on {log}" in out
    assert "undecided gates: none" in out


def test_a_decided_gate_is_evaluated_anew_once_another_log_covers_it(tmp_path, capsys, gated):
    """A segment restarted from scratch re-runs the gate's whole range: the earlier decision was on another run."""
    first = write_run(tmp_path, "seg1", loss_offset=over(1, 60, 0.005))
    second = write_run(tmp_path, "seg2", loss_offset=over(41, 60, 0.05))
    status, out = run(gated, [first, second], capsys, "--decided", f"G={first}")
    assert status == 1 and f"GATE G: FAIL on {second}" in out and "undecided gates: G" in out


@pytest.mark.parametrize("value", ["H=/x.out", "G"])
def test_a_decided_value_that_names_no_watched_gate_leaves_the_watch_unevaluated(tmp_path, capsys, gated, value):
    status, out = run(gated, [write_run(tmp_path, "seg1")], capsys, "--decided", value)
    assert status == 2 and "NOT EVALUATED watch: ValueError: --decided" in out


def test_a_gate_the_logs_have_not_reached_is_not_due(tmp_path, capsys, gated):
    status, out = run(gated, [write_run(tmp_path, "seg1", drop=set(range(41, 61)))], capsys)
    assert status == 0 and "GATE" not in out and "0 gates due" in out


def test_a_failing_gate_fails_the_check(tmp_path, capsys, gated):
    status, out = run(gated, [write_run(tmp_path, "seg1", loss_offset=over(41, 60, 0.05))], capsys)
    assert status == 1 and "GATE G: FAIL" in out


def test_a_gate_whose_range_spans_a_restart_is_not_evaluated(tmp_path, capsys, gated):
    first = write_run(tmp_path, "seg1", drop=set(range(31, 61)))
    second = write_run(tmp_path, "seg2", drop=set(range(1, 31)))
    status, out = run(gated, [first, second], capsys)
    assert status == 2 and "GATE G: NOT EVALUATED (no single log covers iterations 1-60" in out


def test_a_gate_is_not_evaluated_on_a_segment_a_resume_superseded(tmp_path, capsys, gated):
    """The first segment logged the whole range, but a resume re-ran its tail: the range now spans a restart."""
    first = write_run(tmp_path, "seg1", loss_offset=over(41, 60, 0.05))
    second = write_run(tmp_path, "seg2", drop=set(range(1, 41)))
    status, out = run(gated, [first, second], capsys)
    assert status == 2 and "GATE G: NOT EVALUATED (no single log covers iterations 1-60" in out


def test_the_offsets_and_their_slope_are_printed_once_the_gate_has_run(tmp_path, capsys, gated):
    status, out = run(gated, [write_run(tmp_path, "seg1", loss_offset=over(1, 60, 0.004))], capsys)
    assert status == 0
    assert re.search(
        r"TREND G: offsets from the references' mean -0\.0010, -0\.0010, -0\.0010; slope [+-]0\.0000", out
    )


def test_an_offset_risen_by_more_than_the_rule_allows_is_flagged(tmp_path, capsys, gated):
    offsets = {**over(1, 20, -0.004), **over(21, 40, 0.0), **over(41, 60, 0.004)}
    status, out = run(gated, [write_run(tmp_path, "seg1", loss_offset=offsets)], capsys)
    assert status == 0 and "FLAG G: the last 1 windows' offset is +0.0080 above the first's" in out


def test_a_steady_offset_is_not_flagged(tmp_path, capsys, gated):
    status, out = run(gated, [write_run(tmp_path, "seg1", loss_offset=over(1, 60, 0.004))], capsys)
    assert status == 0 and "FLAG" not in out


# --------------------------------------------------------------------------------------
# The block envelope
# --------------------------------------------------------------------------------------


WANDB_REFERENCE = "geodesic/megatron_training/ref"


def envelope_watch(
    tmp_path: Path,
    envelope_offset: float,
    reference_drop: set[int] | None = None,
    reference_wandb_blocks: dict[int, str] | None = None,
) -> Path:
    """A watch spec whose reference run is the fixture (without the ``reference_drop`` iterations, and with the
    ``reference_wandb_blocks`` read from W&B) and whose envelope run sits ``envelope_offset`` above it, in
    20-iteration blocks, two consecutive blocks outside to flag."""
    reference = {"logs": [str(write_run(tmp_path, "baseline", drop=reference_drop))]}
    if reference_wandb_blocks is not None:
        reference["wandb_blocks"] = reference_wandb_blocks
    envelope = {"logs": [str(write_run(tmp_path, "broad", loss_offset=over(1, 60, envelope_offset)))]}
    rule = {"reference": reference, "envelope": envelope, "block_iterations": 20, "consecutive_blocks": 2}
    return write_watch(tmp_path, {"flags": {"block_envelope": rule}})


def test_a_candidate_within_the_envelope_is_reported_not_flagged(tmp_path, capsys):
    spec = envelope_watch(tmp_path, 0.02)
    status, out = run(spec, [write_run(tmp_path, "seg1", loss_offset=over(1, 60, -0.01))], capsys)
    assert status == 0 and "FLAG" not in out
    assert "TREND blocks (candidate - reference against envelope - reference): block 1: -0.0100 against +0.0200" in out


def test_two_consecutive_blocks_outside_the_envelope_are_flagged(tmp_path, capsys):
    spec = envelope_watch(tmp_path, 0.02)
    offsets = {**over(1, 20, 0.01), **over(21, 60, 0.03)}
    status, out = run(spec, [write_run(tmp_path, "seg1", loss_offset=offsets)], capsys)
    assert status == 0 and "FLAG outside the envelope in 2 consecutive blocks of 20 iterations" in out


def test_a_candidate_below_the_reference_by_more_than_the_envelope_is_flagged(tmp_path, capsys):
    """The envelope is a distance from the reference, so a candidate outside it on the low side is flagged too."""
    spec = envelope_watch(tmp_path, 0.02)
    status, out = run(spec, [write_run(tmp_path, "seg1", loss_offset=over(21, 60, -0.03))], capsys)
    assert status == 0 and "FLAG outside the envelope in 2 consecutive blocks of 20 iterations: blocks 2-3" in out


def test_one_block_outside_the_envelope_is_not_flagged(tmp_path, capsys):
    spec = envelope_watch(tmp_path, 0.02)
    offsets = {**over(1, 20, 0.03), **over(21, 60, 0.01)}
    status, out = run(spec, [write_run(tmp_path, "seg1", loss_offset=offsets)], capsys)
    assert status == 0 and "FLAG" not in out


def test_an_incomplete_block_is_not_compared(tmp_path, capsys):
    spec = envelope_watch(tmp_path, 0.02)
    status, out = run(spec, [write_run(tmp_path, "seg1", drop=set(range(51, 61)))], capsys)
    assert "block 3" not in out and "block 2:" in out


def test_a_block_the_reference_does_not_cover_is_reported_as_not_compared(tmp_path, capsys):
    spec = envelope_watch(tmp_path, 0.02, reference_drop=set(range(25, 41)))
    status, out = run(spec, [write_run(tmp_path, "seg1")], capsys)
    assert "block 2: not compared (the reference logs 4 of 20 iterations)" in out
    assert "block 1: +0.0000 against +0.0200" in out and "block 3: +0.0000 against +0.0200" in out


def test_blocks_on_either_side_of_one_not_compared_are_not_consecutive(tmp_path, capsys):
    spec = envelope_watch(tmp_path, 0.02, reference_drop=set(range(21, 41)))
    status, out = run(spec, [write_run(tmp_path, "seg1", loss_offset=over(1, 60, 0.03))], capsys)
    assert status == 0 and "FLAG" not in out


# The W&B client is test_loss_parity's fake: the real one reads W&B's service, which unit tests must not depend on.


def test_a_block_read_from_wandb_names_its_source(tmp_path, capsys, monkeypatch):
    install_fake_wandb(monkeypatch, history_rows(21, 40))
    spec = envelope_watch(tmp_path, 0.02, set(range(21, 41)), {2: WANDB_REFERENCE})
    status, out = run(spec, [write_run(tmp_path, "seg1")], capsys)
    assert f"block 2: -0.0000 against +0.0200 (reference from wandb:{WANDB_REFERENCE})" in out
    assert "block 1: +0.0000 against +0.0200;" in out
    assert [(r["min_step"], r["max_step"]) for r in FakeRun.requests] == [(21, 41)]


def test_a_wandb_block_its_history_does_not_cover_is_not_compared(tmp_path, capsys, monkeypatch):
    install_fake_wandb(monkeypatch, history_rows(21, 39))
    spec = envelope_watch(tmp_path, 0.02, set(range(21, 41)), {2: WANDB_REFERENCE})
    status, out = run(spec, [write_run(tmp_path, "seg1")], capsys)
    assert re.search(r"block 2: not compared \(the reference's W&B history: ValueError: .*missing \[40\]", out)


def test_a_flag_names_its_blocks_and_where_each_came_from(tmp_path, capsys, monkeypatch):
    install_fake_wandb(monkeypatch, history_rows(21, 40))
    spec = envelope_watch(tmp_path, 0.02, set(range(21, 41)), {2: WANDB_REFERENCE})
    status, out = run(spec, [write_run(tmp_path, "seg1", loss_offset=over(21, 60, 0.03))], capsys)
    assert status == 0
    assert (
        "FLAG outside the envelope in 2 consecutive blocks of 20 iterations: blocks 2-3 "
        f"(block 2: reference from wandb:{WANDB_REFERENCE})"
    ) in out


# --------------------------------------------------------------------------------------
# The spec
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, error, message",
    [
        ({"stops": {}}, ValueError, "unknown sections"),
        ({"stop": {"nan": True}}, ValueError, "unknown stop keys"),
        ({"flags": {"spike": {}}}, ValueError, "unknown flags keys"),
        ({"flags": {"loss_spike": {"above_trailing_mean": 0.1}}}, KeyError, "trailing_iterations"),
        (
            {
                "flags": {
                    "block_envelope": {
                        "reference": {"logs": ["a.out"], "wandb": {}},
                        "envelope": {"logs": ["b.out"]},
                        "block_iterations": 20,
                        "consecutive_blocks": 2,
                    }
                }
            },
            ValueError,
            "unknown block_envelope.reference keys",
        ),
        (
            {
                "loss_gates": {"spec": "x.yaml", "gates": ["L1"]},
                "flags": {"growing_offset": {"gate": "L2", "windows": 3, "above": 0.01}},
            },
            ValueError,
            "not a watched loss gate",
        ),
    ],
)
def test_a_spec_that_cannot_be_watched_as_written_is_refused(tmp_path, raw, error, message):
    with pytest.raises(error, match=message):
        rw.load_watch_spec(write_watch(tmp_path, raw))

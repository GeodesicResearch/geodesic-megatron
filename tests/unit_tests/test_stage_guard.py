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

"""Unit tests for scripts/training/stage_guard.py (a training stage guarded by its watch spec while it trains).

The watch's output is run_watch.py's own, produced here by running it on the loss-parity tests' fixture logs; only
SLURM (sacct, squeue, scancel) and the container launch are stood in for, by ``FakeCluster``: the real commands would
query and cancel jobs on the cluster and start the container, which a unit test must not do.
"""

import hashlib
import re
import shlex
import subprocess
import time
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import pytest
import yaml
from scripts.telemetry import run_watch as rw
from scripts.telemetry.code_revision import code_revision
from scripts.training import stage_guard as sg

from tests.unit_tests.test_loss_parity import write_run
from tests.unit_tests.test_run_watch import gated, rejected_result_line  # noqa: F401 - gated is a fixture
from tests.unit_tests.training_log_fixture import ITERATION_STAMP_RE


JOB_NAME = "cp-stage"


# A started segment's start in sacct_line, and the guard's clock in the tests that do not time it: an hour later. The
# fixture log's iteration lines are stamped 01:29:21 to 01:39:40 on 9 September 2026, after this start, so a segment
# logged them itself; ATTEMPT is the start as the guard names that attempt's dump folders.
START_TIME = datetime(2026, 9, 9, 1, 25, 0)
NOW = START_TIME + timedelta(hours=1)
ATTEMPT = START_TIME.strftime("%Y%m%dT%H%M%S")


def summary(checked, gates: int = 0, newest=None, logged=None, log: str = "/seg.out") -> str:
    """The watch's two closing lines, as run_watch.py prints them."""
    return (
        f"newest segment through iteration {newest} logged {logged}: {log}\n"
        f"checked through iteration {checked}: 0 stop conditions, {gates} gates due\n"
    )


def stamp_of(log: Path, iteration: int) -> datetime:
    """The time stamp of ``log``'s line for ``iteration``."""
    for stamp, logged in ITERATION_STAMP_RE.findall(log.read_text()):
        if int(logged) == iteration:
            return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
    raise AssertionError(f"{log} has no line for iteration {iteration}")


def restamped(text: str, first: datetime) -> str:
    """``text`` with its iteration lines stamped 10 seconds apart from ``first``, as a later attempt logs them."""
    lines = iter(range(10**6))

    def stamp(match: re.Match) -> str:
        new = (first + timedelta(seconds=10 * next(lines))).strftime("%Y-%m-%d %H:%M:%S")
        return match.group(0).replace(match.group(1), new)

    return ITERATION_STAMP_RE.sub(stamp, text)


def write_config(tmp_path: Path, **overrides) -> Path:
    raw = {
        "watch": str(tmp_path / "watch.yaml"),
        "job_name": JOB_NAME,
        "since": "2026-10-01",
        "interval_seconds": 1,
        "command_timeout_seconds": 60,
        "record": str(tmp_path / "guard" / "record.log"),
        "hold": {"from_iteration": 40},
        "final_iteration": 60,
        **overrides,
    }
    path = tmp_path / "guard.yaml"
    path.write_text(yaml.safe_dump(raw))
    if "watch" not in overrides:
        (tmp_path / "watch.yaml").write_text(
            yaml.safe_dump({"stop": {"non_finite_grad_norm": True, "rejected_result": True}})
        )
    return path


def sacct_line(
    job_id: int, state: str, log_dir: Path, started: bool = True, name: str = JOB_NAME, start: datetime = START_TIME
) -> str:
    start = start.strftime("%Y-%m-%dT%H:%M:%S") if started else "Unknown"
    return f"{job_id}|{name}|{state}|{start}|{log_dir}/train-%j.out"


class FakeCluster:
    """sacct, squeue, scancel and the container launch, recorded; the container runs run_watch.py in this process.

    Like the real ``sacct -S <date>``, the fake omits a job that has not started (one pending on
    ``--dependency=singleton`` has no eligible time); ``squeue`` lists every live job of the name, pending or not.
    """

    def __init__(self, sacct_lines: list[str], watch_status: int | None = None, watch_output: str = ""):
        self.sacct_lines = sacct_lines
        self.watch_status = watch_status
        self.watch_output = watch_output
        self.commands: list[list[str]] = []
        self.scancel_status = 0
        self.squeue_status = 0
        self.squeue_empty = False
        self.timing_out: set[str] = set()

    def __call__(self, command):
        command = list(command)
        self.commands.append(command)
        if Path(command[0]).name in self.timing_out:
            raise subprocess.TimeoutExpired(command, 600)
        if command[0] == "sacct":
            started = [line for line in self.sacct_lines if line.split("|")[3] != "Unknown"]
            return subprocess.CompletedProcess(command, 0, "\n".join(started) + "\n")
        if command[0] == "squeue":
            if self.squeue_status:
                return subprocess.CompletedProcess(command, self.squeue_status, "slurm_load_jobs error")
            if self.squeue_empty:  # a busy controller's empty reply while the jobs still run
                return subprocess.CompletedProcess(command, 0, "")
            name = command[command.index("--name") + 1]
            live = [
                line.split("|")[0]
                for line in self.sacct_lines
                if line.split("|")[1] == name and line.split("|")[2].split()[0] not in sg.FINAL_STATES
            ]
            return subprocess.CompletedProcess(command, 0, "".join(f"{job_id}\n" for job_id in live))
        if command[0] == "scancel":
            return subprocess.CompletedProcess(command, self.scancel_status, "")
        if self._is_dump(command):
            return subprocess.CompletedProcess(
                command, 0, f"nid000001: 4 rank stack(s) written\nevidence: /x/nccl_trace/{command[2]}/{command[3]}"
            )
        if command[0].endswith("pipeline_env_exec.sh"):
            if self.watch_status is not None:
                return subprocess.CompletedProcess(command, self.watch_status, self.watch_output)
            return self._run_watch(command[1])
        raise AssertionError(f"unexpected command {command}")

    @staticmethod
    def _run_watch(payload: str):
        args = shlex.split(payload.split("python scripts/telemetry/run_watch.py ", 1)[1])
        out = StringIO()
        with redirect_stdout(out):
            status = rw.main(args)
        return subprocess.CompletedProcess(args, status, out.getvalue())

    def cancelled(self) -> list[list[str]]:
        return [command for command in self.commands if command[0] == "scancel"]

    @staticmethod
    def _is_dump(command: list[str]) -> bool:
        return command[0] == "bash" and command[1].endswith("dump_hung_ranks.sh")

    def dumps(self) -> list[str]:
        """Each dump_hung_ranks.sh run as ``<job ID>/<sweep>``, in order."""
        return [f"{command[2]}/{command[3]}" for command in self.commands if self._is_dump(command)]

    def watch_arguments(self) -> list[list[str]]:
        return [
            shlex.split(command[1].split("run_watch.py ", 1)[1])
            for command in self.commands
            if command[0].endswith("pipeline_env_exec.sh")
        ]


def failing_sacct(command):
    """sacct and squeue when the SLURM daemons cannot be reached."""
    if command[0] not in ("sacct", "squeue"):
        raise AssertionError(f"unexpected command {command}")
    return subprocess.CompletedProcess(command, 1, f"{command[0]}: error: slurm daemon unreachable")


def segment(tmp_path: Path, job_id: int, **write_run_args) -> Path:
    """A segment log at the path sacct's StdOut pattern expands to."""
    log = write_run(tmp_path, f"train-{job_id}", **write_run_args)
    assert log == tmp_path / f"train-{job_id}.out"
    return log


# --------------------------------------------------------------------------------------
# Reading sacct and the watch
# --------------------------------------------------------------------------------------


def test_the_stages_jobs_are_read_in_submission_order_with_their_logs(tmp_path):
    output = "\n".join(
        [
            sacct_line(12, "PENDING", tmp_path, started=False),
            sacct_line(10, "COMPLETED", tmp_path),
            sacct_line(11, "RUNNING", tmp_path),
            sacct_line(13, "RUNNING", tmp_path, name="another"),
        ]
    )
    jobs = sg.parse_jobs(output, JOB_NAME)
    assert [(j.job_id, j.state, j.started) for j in jobs] == [
        (10, "COMPLETED", True),
        (11, "RUNNING", True),
        (12, "PENDING", False),
    ]
    assert jobs[0].log == tmp_path / "train-10.out"


def test_a_cancelled_state_with_its_actor_is_read_as_cancelled(tmp_path):
    (job,) = sg.parse_jobs(sacct_line(10, "CANCELLED by 1483801259", tmp_path), JOB_NAME)
    assert job.state == "CANCELLED"


@pytest.mark.parametrize("status", [0, 1, 2])
def test_a_watch_status_counts_only_beside_its_summary_line(status):
    """A container or activation failure can exit 1; only the watch's two summary lines make 1 a stop."""
    lines = summary(55)
    assert sg.parse_watch(status, lines).ran
    assert not sg.parse_watch(status, "FATAL [env-config]: SIF not found\n").ran
    assert not sg.parse_watch(status, lines.split("\n", 1)[1]).ran


@pytest.mark.parametrize("newest, logged", [(41, datetime(2026, 9, 9, 1, 36)), (None, None)])
def test_the_watch_reports_the_newest_logs_last_iteration_and_its_stamp(newest, logged):
    stamp = logged.strftime("%Y-%m-%dT%H:%M:%S") if logged else None
    result = sg.parse_watch(0, summary(60, newest=newest, logged=stamp))
    assert (result.ran, result.checked_iteration, result.newest_iteration, result.newest_logged) == (
        True,
        60,
        newest,
        logged,
    )


def test_the_watch_names_the_gates_it_passed_and_those_still_undecided():
    output = (
        "GATE L1: PASS (passed at an earlier check on this log) on /logs/train-10.out\n"
        "GATE L2: PASS on /logs/train-10.out\nGATE L2b: NOT EVALUATED (x)\n"
        "undecided gates: L2b\n" + summary(2210, gates=3)
    )
    result = sg.parse_watch(2, output)
    assert result.passed == {"L1": "/logs/train-10.out", "L2": "/logs/train-10.out"}
    assert result.undecided == ("L2b",)
    assert (
        sg.parse_watch(
            0,
            "undecided gates: none\n" + summary(1),
        )[5]
        == ()
    )
    assert (
        sg.parse_watch(
            0,
            summary(1),
        ).undecided
        is None
    )


# --------------------------------------------------------------------------------------
# What a tick does
# --------------------------------------------------------------------------------------


def test_a_clean_stage_is_left_alone(tmp_path):
    config = sg.load_guard_config(write_config(tmp_path))
    segment(tmp_path, 10, drop=set(range(31, 61)))
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path), sacct_line(11, "PENDING", tmp_path, started=False)])
    action, state, done = sg.tick(config, sg.START, cluster, NOW)
    assert (action, state.last_iteration, done) == ("continue", 30, False)
    assert not cluster.cancelled()


def test_a_stop_cancels_the_running_segment_and_its_queued_successor_by_id(tmp_path):
    config = sg.load_guard_config(write_config(tmp_path))
    segment(tmp_path, 10, grad_scale={20: float("inf")}, drop=set(range(31, 61)))
    cluster = FakeCluster(
        [
            sacct_line(9, "COMPLETED", tmp_path),
            sacct_line(10, "RUNNING", tmp_path),
            sacct_line(11, "PENDING", tmp_path, started=False),
        ]
    )
    segment(tmp_path, 9, drop=set(range(1, 61)))
    action, _, _ = sg.tick(config, sg.START, cluster, NOW)
    assert action == "stop"
    assert cluster.cancelled() == [["scancel", "10", "11"]]
    record = config.record.read_text()
    assert "STOP: watch exit 1" in record and "STOP non-finite grad norm at iteration 20" in record
    assert "STOP: cancelled [10, 11]" in record


def test_a_run_ended_by_a_rejected_result_is_stopped_before_its_successor_trains(tmp_path):
    """The failed segment has ended; its successor has started but not yet logged an iteration."""
    config = sg.load_guard_config(write_config(tmp_path))
    failed = segment(tmp_path, 10, drop=set(range(31, 61)))
    failed.write_text(failed.read_text() + rejected_result_line(3, 31) + "\n")
    (tmp_path / "train-11.out").write_text("[launcher] starting\n")
    cluster = FakeCluster([sacct_line(10, "FAILED", tmp_path), sacct_line(11, "RUNNING", tmp_path)])
    action, _, _ = sg.tick(config, sg.START, cluster, NOW)
    assert action == "stop" and cluster.cancelled() == [["scancel", "11"]]


@pytest.mark.parametrize("iteration, action", [(39, "alert"), (40, "hold"), (60, "hold"), (1000, "hold")])
def test_an_unevaluated_gate_holds_the_stage_from_its_hold_iteration_on(tmp_path, iteration, action):
    """From there the next save would be written past the unevaluated gate; a tick that comes late, past the first
    save, still holds."""
    config = sg.load_guard_config(write_config(tmp_path))
    output = "GATE L1: NOT EVALUATED (x)\nundecided gates: L1\n" + summary(iteration, gates=1)
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)], watch_status=2, watch_output=output)
    got, _, _ = sg.tick(config, sg.START, cluster, NOW)
    assert got == action
    assert bool(cluster.cancelled()) == (action == "hold")


def test_inside_the_window_an_unevaluated_stop_check_alone_does_not_hold(tmp_path):
    """Every gate has passed, so no save can be written past an undecided one."""
    config = sg.load_guard_config(write_config(tmp_path))
    output = "NOT EVALUATED stop launch_settings: ValueError: x\nundecided gates: none\n" + summary(45, gates=1)
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)], watch_status=2, watch_output=output)
    assert sg.tick(config, sg.START, cluster, NOW)[0] == "alert" and not cluster.cancelled()


@pytest.mark.parametrize(
    "last_iteration, undecided, action",
    [
        (None, None, "alert"),
        (30, None, "alert"),
        (45, None, "hold"),
        (45, ("L1",), "hold"),
        (45, (), "alert"),
        (5000, ("L1",), "hold"),
    ],
)
def test_a_watch_that_could_not_run_holds_only_inside_the_window_while_a_gate_is_undecided(
    tmp_path, last_iteration, undecided, action
):
    """Never a stop: an activation failure exits 3 (and could exit 1); the last iteration a tick read, and the gates
    the last watch left undecided (unknown before any watch has run), decide."""
    config = sg.load_guard_config(write_config(tmp_path))
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)], watch_status=1, watch_output="FATAL: no SIF\n")
    state = sg.GuardState(last_iteration, {}, undecided)
    got, after, _ = sg.tick(config, state, cluster, NOW)
    assert got == action and after == state


def test_jobs_that_cannot_be_listed_inside_the_window_hold_the_stage(tmp_path):
    """The same rule as a watch that could not run; the cancellation then needs sacct too, and says it failed."""
    config = sg.load_guard_config(write_config(tmp_path))
    with pytest.raises(sg.CancelFailed, match="could not cancel the stage's jobs: SlurmError: squeue exited 1"):
        sg.tick(config, sg.GuardState(45, {}, ("L1",)), failing_sacct, NOW)
    assert "HOLD: could not evaluate: RuntimeError: sacct failed" in config.record.read_text()


def test_jobs_that_cannot_be_listed_once_every_gate_has_passed_alert(tmp_path):
    config = sg.load_guard_config(write_config(tmp_path))
    action, _, _ = sg.tick(config, sg.GuardState(45, {"L1": "/x.out"}, ()), failing_sacct, NOW)
    assert action == "alert"


def test_a_gate_passed_once_is_not_evaluated_again_on_its_log(tmp_path, gated):
    """The second tick names the gate decided, so a reference that can no longer be read cannot unsettle it."""
    config = sg.load_guard_config(write_config(tmp_path, watch=str(gated)))
    log = segment(tmp_path, 10, loss_offset={it: 0.005 for it in range(1, 61)})
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)])
    action, state, _ = sg.tick(config, sg.START, cluster, NOW)
    assert action == "continue" and state.decided == {"G": str(log)} and state.undecided == ()
    (tmp_path / "ref_a.out").unlink()
    action, state, _ = sg.tick(config, state, cluster, NOW)
    assert action == "continue"
    assert cluster.watch_arguments()[-1][-2:] == ["--decided", f"G={log}"]


def test_only_a_tick_whose_watch_ran_seeds_the_decided_gates_and_a_later_tick_wins(tmp_path):
    record = tmp_path / "record.log"
    assert sg.decided_in_record(record, "y") == {}
    record.write_text(
        "2026-10-01T06:00:00Z START guard g.yaml (sha256 x); watch spec w.yaml (sha256 y); code z\n"
        "2026-10-01T06:02:00Z CONTINUE: watch exit 0 through iteration 60; undecided gates none [10:RUNNING] spec y\n"
        "GATE G: PASS on /a.out\nundecided gates: none\nchecked through iteration 60: 0 stop conditions, 1 gates due\n"
        "2026-10-01T06:04:00Z ALERT: watch could not run (exit 3); undecided gates none [10:RUNNING] spec y\n"
        "GATE H: PASS on /b.out\nchecked through iteration 60: 0 stop conditions, 1 gates due\n"
        "2026-10-01T06:06:00Z CONTINUE: watch exit 0 through iteration 62; undecided gates none [11:RUNNING] spec y\n"
        "GATE G: PASS (passed at an earlier check on this log) on /c.out\nundecided gates: none\n"
        "checked through iteration 62: 0 stop conditions, 1 gates due\n"
    )
    assert sg.decided_in_record(record, "y") == {"G": "/c.out"}


def test_a_gate_passed_under_another_watch_spec_is_not_seeded(tmp_path):
    """A gate redefined under its old name in an edited spec must be judged again, not carried over."""
    record = tmp_path / "record.log"
    record.write_text(
        "2026-10-01T06:02:00Z CONTINUE: watch exit 0 through iteration 60; undecided gates none [10:RUNNING] spec old\n"
        "GATE G: PASS on /a.out\nundecided gates: none\nchecked through iteration 60: 0 stop conditions, 1 gates due\n"
        "2026-10-01T06:04:00Z CONTINUE: watch exit 0 through iteration 62; undecided gates none [10:RUNNING] spec new\n"
        "GATE H: PASS on /a.out\nundecided gates: none\nchecked through iteration 62: 0 stop conditions, 1 gates due\n"
    )
    assert sg.decided_in_record(record, "new") == {"H": "/a.out"}


def test_a_log_path_with_a_space_reaches_the_watch_whole(tmp_path):
    config = sg.load_guard_config(write_config(tmp_path))
    log_dir = tmp_path / "with space"
    log_dir.mkdir()
    write_run(log_dir, "train-10", drop=set(range(31, 61)))
    cluster = FakeCluster([sacct_line(10, "RUNNING", log_dir)])
    action, state, _ = sg.tick(config, sg.START, cluster, NOW)
    assert (action, state.last_iteration) == ("continue", 30)


def test_the_stage_is_done_once_its_final_iteration_is_checked_and_no_job_is_live(tmp_path):
    config = sg.load_guard_config(write_config(tmp_path))
    segment(tmp_path, 10)
    cluster = FakeCluster([sacct_line(10, "COMPLETED", tmp_path)])
    action, state, done = sg.tick(config, sg.START, cluster, NOW)
    assert (action, state.last_iteration, done) == ("continue", 60, True)


def test_a_stage_with_no_started_segment_fails_the_guard(tmp_path):
    """The guard is started once a segment trains, so finding none means a wrong name or user, not a wait."""
    config = sg.load_guard_config(write_config(tmp_path))
    cluster = FakeCluster([sacct_line(10, "PENDING", tmp_path, started=False)])
    with pytest.raises(sg.NoStartedSegment, match="no segment of cp-stage has started"):
        sg.tick(config, sg.START, cluster, NOW)


@pytest.mark.parametrize("stopped_by", ["the watch", "the start-up window"])
def test_a_stop_cancels_before_it_records(tmp_path, stopped_by):
    """A record that cannot be written must not keep the stage's jobs alive. A start-up stop dumps the stacks first,
    while the ranks still run."""
    (tmp_path / "not-a-directory").write_text("")
    config, now = stopping_stage(tmp_path, stopped_by, record=str(tmp_path / "not-a-directory" / "record.log"))
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path), sacct_line(11, "PENDING", tmp_path, started=False)])
    with pytest.raises(OSError):
        sg.tick(config, sg.START, cluster, now)
    assert cluster.cancelled() == [["scancel", "10", "11"]]
    assert cluster.dumps() == ([] if stopped_by == "the watch" else [f"10/{ATTEMPT}_stop"])


def test_a_failing_squeue_fails_the_cancellation_rather_than_reading_as_no_jobs(tmp_path):
    config = sg.load_guard_config(write_config(tmp_path))
    segment(tmp_path, 10, grad_scale={20: float("inf")}, drop=set(range(31, 61)))
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)])
    cluster.squeue_status = 1
    with pytest.raises(sg.CancelFailed, match="squeue exited 1"):
        sg.tick(config, sg.START, cluster, NOW)
    assert not cluster.cancelled()


def test_a_watch_past_its_timeout_counts_as_not_evaluated(tmp_path):
    """Outside the hold it alerts; it is never read as a stop."""
    config = sg.load_guard_config(write_config(tmp_path))
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)])
    cluster.timing_out.add("pipeline_env_exec.sh")
    action, state, _ = sg.tick(config, sg.START, cluster, NOW)
    assert action == "alert" and state == sg.START and not cluster.cancelled()
    assert "could not evaluate: TimeoutExpired" in config.record.read_text()


@pytest.mark.parametrize("command", ["squeue", "scancel"])
def test_a_cancellation_whose_command_times_out_fails(tmp_path, command):
    config = sg.load_guard_config(write_config(tmp_path))
    segment(tmp_path, 10, grad_scale={20: float("inf")}, drop=set(range(31, 61)))
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)])
    cluster.timing_out.add(command)
    with pytest.raises(sg.CancelFailed, match="TimeoutExpired"):
        sg.tick(config, sg.START, cluster, NOW)


def test_a_command_past_its_timeout_is_killed():
    with pytest.raises(subprocess.TimeoutExpired):
        sg.run_command(["sleep", "30"], 1)


# --------------------------------------------------------------------------------------
# The loop's exit statuses
# --------------------------------------------------------------------------------------


@pytest.fixture()
def cluster_command(monkeypatch):
    """Route the guard's commands to a FakeCluster the test sets."""
    holder = {}
    # The guard runs each command with the config's timeout; the stand-in cluster ignores it.
    monkeypatch.setattr(sg, "run_command", lambda command, timeout_seconds: holder["cluster"](command))
    return holder


def test_a_stop_ends_the_guard_with_status_1(tmp_path, cluster_command):
    segment(tmp_path, 10, grad_scale={20: float("inf")}, drop=set(range(31, 61)))
    cluster_command["cluster"] = FakeCluster([sacct_line(10, "RUNNING", tmp_path)])
    assert sg.main(["--config", str(write_config(tmp_path))]) == 1


def test_a_cancellation_that_fails_ends_the_guard_with_its_own_status(tmp_path, cluster_command):
    segment(tmp_path, 10, grad_scale={20: float("inf")}, drop=set(range(31, 61)))
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)])
    cluster.scancel_status = 1
    cluster_command["cluster"] = cluster
    config = write_config(tmp_path)
    assert sg.main(["--config", str(config)]) == sg.CANCEL_FAILED
    assert "CANCEL FAILED: scancel [10] failed" in (tmp_path / "guard" / "record.log").read_text()


def test_a_guard_that_cannot_list_the_jobs_outside_the_window_alerts_and_keeps_going(tmp_path, cluster_command):
    cluster_command["cluster"] = failing_sacct
    assert sg.main(["--config", str(write_config(tmp_path)), "--once"]) == 2
    assert "ALERT: could not evaluate: RuntimeError: sacct failed" in (tmp_path / "guard" / "record.log").read_text()


def test_the_record_names_the_config_the_spec_and_the_code_and_each_tick_the_spec(tmp_path, cluster_command):
    segment(tmp_path, 10, drop=set(range(31, 61)))
    cluster_command["cluster"] = FakeCluster([sacct_line(10, "RUNNING", tmp_path)])
    config = write_config(tmp_path)
    assert sg.main(["--config", str(config), "--once"]) == 0
    start, tick_line = (tmp_path / "guard" / "record.log").read_text().splitlines()[:2]
    spec_sha = hashlib.sha256((tmp_path / "watch.yaml").read_bytes()).hexdigest()
    assert f"START guard {config} (sha256 {hashlib.sha256(config.read_bytes()).hexdigest()})" in start
    assert (
        f"watch spec {tmp_path / 'watch.yaml'} (sha256 {spec_sha}); code {code_revision(str(sg.REPO_ROOT))}" in start
    )
    assert tick_line.endswith(f"spec {spec_sha[:12]}")


def test_a_restarted_guard_does_not_hold_the_stage_over_a_gate_its_record_saw_pass(tmp_path, gated, cluster_command):
    """Restarted inside the hold window with the gate's reference unreadable (W&B down, say), a guard that had to
    evaluate the gate again would cancel a stage whose gates have all passed."""
    log = segment(tmp_path, 10, loss_offset={it: 0.005 for it in range(1, 61)})
    config = write_config(tmp_path, watch=str(gated))
    cluster_command["cluster"] = FakeCluster([sacct_line(10, "RUNNING", tmp_path)])
    assert sg.main(["--config", str(config), "--once"]) == 0
    (tmp_path / "ref_a.out").unlink()
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)])
    cluster_command["cluster"] = cluster
    assert sg.main(["--config", str(config), "--once"]) == 0
    assert cluster.cancelled() == []
    assert cluster.watch_arguments()[-1][-2:] == ["--decided", f"G={log}"]
    starts = [line for line in (tmp_path / "guard" / "record.log").read_text().splitlines() if " START guard " in line]
    assert starts[0].endswith("; decided from the record: none")
    assert starts[1].endswith(f"; decided from the record: G={log}")


def test_a_guard_that_finds_no_segment_records_its_failure_and_exits_with_its_own_status(tmp_path, cluster_command):
    cluster_command["cluster"] = FakeCluster([sacct_line(10, "PENDING", tmp_path, started=False)])
    assert sg.main(["--config", str(write_config(tmp_path)), "--once"]) == sg.GUARD_FAILED
    record = (tmp_path / "guard" / "record.log").read_text()
    assert "GUARD FAILED: NoStartedSegment: no segment of cp-stage has started; the stage is unguarded" in record


def test_a_guard_that_fails_itself_exits_with_its_own_status(tmp_path, cluster_command):
    """A traceback would exit 1, which reads as a stop."""
    (tmp_path / "not-a-directory").write_text("")
    config = write_config(tmp_path, record=str(tmp_path / "not-a-directory" / "record.log"))
    cluster_command["cluster"] = FakeCluster([])
    assert sg.main(["--config", str(config), "--once"]) == sg.GUARD_FAILED


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"jobname": "x"}, "unknown keys"),
        ({"hold": {"from_iteration": 1, "until_iteration": 2}}, "hold needs exactly from_iteration"),
    ],
)
def test_a_config_that_cannot_guard_as_written_is_refused(tmp_path, raw, message):
    path = write_config(tmp_path)
    path.write_text(yaml.safe_dump({**yaml.safe_load(path.read_text()), **raw}))
    with pytest.raises(ValueError, match=message):
        sg.load_guard_config(path)


# --------------------------------------------------------------------------------------
# The start-up window, the stall limit and the end on a final sacct state
# --------------------------------------------------------------------------------------


def no_iteration_yet(tmp_path: Path, job_id: int) -> Path:
    """A segment log that has logged no iteration."""
    return segment(tmp_path, job_id, drop=set(range(1, 61)))


def stopping_stage(tmp_path: Path, stopped_by: str, **overrides):
    """A guard config, and the clock at which it stops job 10: by the watch (a non-finite grad norm at iteration 20)
    or by the start-up window (no iteration 51 minutes after the job started)."""
    if stopped_by == "the watch":
        segment(tmp_path, 10, grad_scale={20: float("inf")}, drop=set(range(31, 61)))
        return sg.load_guard_config(write_config(tmp_path, **overrides)), NOW
    no_iteration_yet(tmp_path, 10)
    config = sg.load_guard_config(write_config(tmp_path, startup={"first_iteration_minutes": 50}, **overrides))
    return config, START_TIME + timedelta(minutes=51)


def record_lines(config: sg.GuardConfig) -> list[str]:
    return config.record.read_text().splitlines()


def test_a_guard_config_without_the_new_keys_never_dumps_or_stops_for_time(tmp_path):
    """The existing keys alone: no start-up window and no stall limit, so a segment silent for hours is left to the
    watch, as before, and no dump runs."""
    config = sg.load_guard_config(write_config(tmp_path))
    assert (config.startup_minutes, config.startup_dump_minutes, config.stall_minutes) == (None, (), None)
    no_iteration_yet(tmp_path, 10)
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)])

    for hours in (1, 5, 12):
        action, _, done = sg.tick(config, sg.START, cluster, START_TIME + timedelta(hours=hours))
        assert action != "stop" and not done
    assert not cluster.dumps() and not cluster.cancelled()


def test_a_start_up_window_that_expires_dumps_at_its_minutes_then_dumps_and_cancels(tmp_path):
    """No iteration 50 minutes after the segment started: stack dumps at 20 and 40 minutes without a cancel, then a
    last dump and the cancel of every live job, as a stop."""
    config = sg.load_guard_config(
        write_config(tmp_path, startup={"first_iteration_minutes": 50, "dump_at_minutes": [20, 40]})
    )
    no_iteration_yet(tmp_path, 10)
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path), sacct_line(11, "PENDING", tmp_path, started=False)])
    state = sg.START

    for minute, dumps in (
        (10, []),
        (21, [f"10/{ATTEMPT}_startup_m20"]),
        (30, [f"10/{ATTEMPT}_startup_m20"]),
        (41, [f"10/{ATTEMPT}_startup_m20", f"10/{ATTEMPT}_startup_m40"]),
    ):
        action, state, _ = sg.tick(config, state, cluster, START_TIME + timedelta(minutes=minute))
        assert action != "stop" and cluster.dumps() == dumps and not cluster.cancelled()
    action, state, done = sg.tick(config, state, cluster, START_TIME + timedelta(minutes=51))

    assert (action, done) == ("stop", False)
    assert cluster.dumps() == [f"10/{ATTEMPT}_startup_m20", f"10/{ATTEMPT}_startup_m40", f"10/{ATTEMPT}_stop"]
    assert cluster.cancelled() == [["scancel", "10", "11"]]
    lines = record_lines(config)
    assert any(
        " STOP: watch exit 0 through iteration None; start-up window: job 10 logged no iteration within 50 min of "
        "starting; " in line
        for line in lines
    )
    assert any(f"DUMP: stacks of job 10, sweep {ATTEMPT}_stop: exit 0: " in line for line in lines)


def test_a_start_up_window_closes_at_the_first_iteration(tmp_path):
    """An iteration logged inside the window ends it: no stop later, and no dump after the iteration."""
    config = sg.load_guard_config(
        write_config(tmp_path, startup={"first_iteration_minutes": 50, "dump_at_minutes": [20]})
    )
    no_iteration_yet(tmp_path, 10)
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)])

    action, state, _ = sg.tick(config, sg.START, cluster, START_TIME + timedelta(minutes=25))
    assert cluster.dumps() == [f"10/{ATTEMPT}_startup_m20"]
    segment(tmp_path, 10, drop=set(range(4, 61)))
    action, state, _ = sg.tick(config, state, cluster, START_TIME + timedelta(minutes=34))
    assert state.last_iteration == 3
    action, state, _ = sg.tick(config, state, cluster, START_TIME + timedelta(minutes=90))

    assert action != "stop" and cluster.dumps() == [f"10/{ATTEMPT}_startup_m20"] and not cluster.cancelled()


def test_a_stall_after_the_first_iteration_dumps_and_cancels(tmp_path):
    """With stall_minutes 15, 15 minutes since the attempt's last iteration line is a stall; 10 minutes is not."""
    config = sg.load_guard_config(write_config(tmp_path, stall_minutes=15))
    log = segment(tmp_path, 10, drop=set(range(21, 61)))
    last = stamp_of(log, 20)
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)])

    action, state, _ = sg.tick(config, sg.START, cluster, last + timedelta(minutes=10))
    assert state.last_iteration == 20 and action != "stop" and not cluster.cancelled()
    action, state, _ = sg.tick(config, state, cluster, last + timedelta(minutes=16))

    assert (
        action == "stop" and cluster.dumps() == [f"10/{ATTEMPT}_stop"] and cluster.cancelled() == [["scancel", "10"]]
    )
    assert any(
        f"; stall: job 10 logged no iteration for 15 min (its last, iteration 20, at {last:%H:%M:%S}); " in line
        for line in record_lines(config)
    )


def test_the_guard_ends_only_when_sacct_shows_the_segments_final(tmp_path):
    """The watch has checked the final iteration and squeue lists nothing, but sacct still shows the segment running
    (a busy controller's empty reply): not done. Once sacct shows it completed: done."""
    config = sg.load_guard_config(write_config(tmp_path))
    segment(tmp_path, 10)
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)])
    cluster.squeue_empty = True

    _, state, done = sg.tick(config, sg.START, cluster, NOW)
    assert state.last_iteration == 60 and not done
    cluster.sacct_lines = [sacct_line(10, "COMPLETED", tmp_path)]
    _, _, done = sg.tick(config, state, cluster, NOW)
    assert done


@pytest.mark.parametrize(
    "overrides",
    [
        {"startup": {"dump_at_minutes": [20]}},
        {"startup": {"first_iteration_minutes": 50, "dumps": [20]}},
        {"startup": {"first_iteration_minutes": 50, "dump_at_minutes": [50]}},
        {"startup": {"first_iteration_minutes": 0}},
        {"stall_minutes": 0},
    ],
)
def test_a_malformed_start_up_or_stall_setting_is_refused(tmp_path, overrides):
    with pytest.raises(ValueError):
        sg.load_guard_config(write_config(tmp_path, **overrides))


# --------------------------------------------------------------------------------------
# A stop during an empty squeue reply; the start-up window per segment
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stopped_by, state",
    [
        ("the watch", "RUNNING"),
        ("the start-up window", "RUNNING"),
        ("the watch", "REQUEUE_HOLD"),
        ("the watch", "STOPPED"),
        ("the watch", "SPECIAL_EXIT"),
    ],
)
def test_a_stop_during_an_empty_squeue_reply_cancels_the_segment_sacct_shows_live(tmp_path, stopped_by, state):
    """One empty squeue reply from a busy controller must not make a stop cancel nothing while sacct shows the
    segment in any state but a final one: the guard cancels the union of the two listings."""
    config, now = stopping_stage(tmp_path, stopped_by)
    cluster = FakeCluster([sacct_line(10, state, tmp_path)])
    cluster.squeue_empty = True

    action, _, _ = sg.tick(config, sg.START, cluster, now)

    assert action == "stop" and cluster.cancelled() == [["scancel", "10"]]


@pytest.mark.parametrize("replays", [True, False], ids=["replays-below-60", "logs-nothing"])
def test_a_successors_start_up_window_counts_its_own_iterations_only(tmp_path, replays):
    """Job 10 logged through iteration 60 and failed; job 11 resumed from the save at iteration 40, and the guard is
    started while job 11 runs. Job 11's window counts its own iterations: replaying 41 to 45, below job 10's 60,
    closes it; no iteration of its own by minute 50 stops it, however far job 10 got."""
    config = sg.load_guard_config(write_config(tmp_path, startup={"first_iteration_minutes": 50}))
    segment(tmp_path, 10)
    no_iteration_yet(tmp_path, 11)
    cluster = FakeCluster([sacct_line(10, "FAILED", tmp_path), sacct_line(11, "RUNNING", tmp_path)])

    action, state, _ = sg.tick(config, sg.START, cluster, START_TIME + timedelta(minutes=10))
    assert action != "stop"
    if replays:
        segment(tmp_path, 11, drop=set(range(1, 41)) | set(range(46, 61)))
    action, state, _ = sg.tick(config, state, cluster, START_TIME + timedelta(minutes=30))
    assert action != "stop"
    action, state, _ = sg.tick(config, state, cluster, START_TIME + timedelta(minutes=51))

    if replays:
        assert action != "stop" and not cluster.cancelled()
    else:
        assert action == "stop" and cluster.cancelled() == [["scancel", "11"]]


def test_the_time_limits_act_only_on_a_tick_whose_watch_ran(tmp_path):
    """A watch that could not run says nothing about the segment's progress: an expired window stops the segment on
    the next tick whose watch runs, not on this one."""
    config = sg.load_guard_config(write_config(tmp_path, startup={"first_iteration_minutes": 50}))
    no_iteration_yet(tmp_path, 10)
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)], watch_status=3, watch_output="FATAL: no SIF\n")

    action, state, _ = sg.tick(config, sg.START, cluster, START_TIME + timedelta(minutes=51))
    assert action != "stop" and not cluster.cancelled() and not cluster.dumps()
    cluster.watch_status = None
    action, _, _ = sg.tick(config, state, cluster, START_TIME + timedelta(minutes=52))

    assert action == "stop" and cluster.cancelled() == [["scancel", "10"]]


def test_a_guard_started_late_in_the_window_takes_one_dump_for_the_minutes_it_passed(tmp_path):
    """Started at minute 45 with dumps listed at 20 and 40: one sweep, named for minute 40, and none again at 47."""
    config = sg.load_guard_config(
        write_config(tmp_path, startup={"first_iteration_minutes": 50, "dump_at_minutes": [20, 40]})
    )
    no_iteration_yet(tmp_path, 10)
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)])

    _, state, _ = sg.tick(config, sg.START, cluster, START_TIME + timedelta(minutes=45))
    sg.tick(config, state, cluster, START_TIME + timedelta(minutes=47))

    assert cluster.dumps() == [f"10/{ATTEMPT}_startup_m40"]


def test_a_command_that_times_out_is_killed_with_every_process_it_started(tmp_path):
    """A child that holds the output pipe (the dump's srun) is killed too, so the guard neither waits on the pipe nor
    leaves the child running."""
    pid_file = tmp_path / "child.pid"
    with pytest.raises(subprocess.TimeoutExpired):
        sg.run_command(["bash", "-c", f"sleep 60 & echo $! > {pid_file}; wait"], 2)
    child = int(pid_file.read_text())
    deadline = time.monotonic() + 10
    while Path(f"/proc/{child}").exists() and "Z" not in Path(f"/proc/{child}/stat").read_text().split()[2]:
        assert time.monotonic() < deadline, f"child {child} still runs after the timeout"
        time.sleep(0.1)


# --------------------------------------------------------------------------------------
# Findings of the /code-review of 2026-11-17 20:21Z
# --------------------------------------------------------------------------------------


def test_a_record_written_before_the_newest_segment_line_still_seeds_the_decided_gates(tmp_path):
    """A guard restarted on a record that an earlier version wrote, whose watch output has no "newest segment" line,
    still takes the gates that record shows passed."""
    config = sg.load_guard_config(write_config(tmp_path))
    spec = sg.spec_tag(config.watch)
    config.record.parent.mkdir(parents=True)
    config.record.write_text(
        f"2026-10-01T06:10:00Z CONTINUE: watch exit 0 through iteration 60; undecided gates none [10:RUNNING] spec {spec}\n"
        "GATE G: PASS on /a.out\nundecided gates: none\nchecked through iteration 60: 0 stop conditions, 1 gates due\n"
    )
    assert sg.decided_in_record(config.record, spec) == {"G": "/a.out"}


def test_no_stall_stop_once_the_watch_has_checked_the_final_iteration(tmp_path):
    """After iteration 60, the stage's last, the job saves and exits and logs no iteration meanwhile."""
    config = sg.load_guard_config(write_config(tmp_path, stall_minutes=15))
    segment(tmp_path, 10)
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)])

    _, state, _ = sg.tick(config, sg.START, cluster, NOW)
    action, _, _ = sg.tick(config, state, cluster, NOW + timedelta(minutes=40))

    assert action != "stop" and not cluster.cancelled() and not cluster.dumps()


def test_the_start_up_window_stops_on_a_tick_the_watch_reports_as_an_alert(tmp_path):
    """A check the watch cannot evaluate makes every tick an alert; the window still applies, as the watch ran."""
    config = sg.load_guard_config(write_config(tmp_path, startup={"first_iteration_minutes": 50}))
    no_iteration_yet(tmp_path, 10)
    output = "NOT EVALUATED launch settings: unreadable\n" + summary(None, log=f"{tmp_path}/train-10.out")
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)], watch_status=2, watch_output=output)

    action, _, _ = sg.tick(config, sg.START, cluster, START_TIME + timedelta(minutes=51))

    assert action == "stop" and cluster.cancelled() == [["scancel", "10"]]


@pytest.mark.parametrize("replays", [True, False], ids=["replays-41-to-45", "logs-nothing"])
def test_a_job_requeued_under_its_own_id_gets_a_start_up_window_of_its_own(tmp_path, replays):
    """Job 10 logged through iteration 60; SLURM then requeued it under the same ID, and it appends to its old log.
    Lines stamped before the new start are the earlier attempt's, so they do not close the new attempt's window."""
    config = sg.load_guard_config(write_config(tmp_path, startup={"first_iteration_minutes": 50}))
    log = segment(tmp_path, 10)
    first_attempt = log.read_text()
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)])
    _, state, _ = sg.tick(config, sg.START, cluster, START_TIME + timedelta(minutes=30))

    restart = START_TIME + timedelta(hours=2)
    cluster.sacct_lines = [sacct_line(10, "RUNNING", tmp_path, start=restart)]
    action, state, _ = sg.tick(config, state, cluster, restart + timedelta(minutes=10))
    assert action != "stop"
    if replays:
        replay = segment(tmp_path, 10, drop=set(range(1, 41)) | set(range(46, 61))).read_text()
        log.write_text(first_attempt + restamped(replay, restart + timedelta(minutes=20)))
    action, state, _ = sg.tick(config, state, cluster, restart + timedelta(minutes=30))
    action, state, _ = sg.tick(config, state, cluster, restart + timedelta(minutes=51))

    if replays:
        assert action != "stop" and not cluster.cancelled()
    else:
        assert action == "stop" and cluster.cancelled() == [["scancel", "10"]]


def test_a_command_that_cannot_start_fails_the_cancellation(tmp_path):
    """An OSError from scancel (not on PATH, say) is a failed cancellation, exit 4, not a failure of the guard."""
    config = sg.load_guard_config(write_config(tmp_path))
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)])

    def run(command):
        if command[0] == "scancel":
            raise FileNotFoundError(2, "No such file or directory", "scancel")
        return cluster(command)

    with pytest.raises(sg.CancelFailed, match="FileNotFoundError"):
        sg.cancel_live_jobs(config, [], run)


def test_commands_print_slurm_times_in_the_format_the_guard_reads(monkeypatch):
    """parse_jobs reads sacct's start time as %Y-%m-%dT%H:%M:%S, so a SLURM_TIME_FORMAT in the guard's shell must not
    reach sacct."""
    monkeypatch.setenv("SLURM_TIME_FORMAT", "relative")
    assert sg.run_command(["bash", "-c", "echo $SLURM_TIME_FORMAT"], 10).stdout == "standard\n"


# --------------------------------------------------------------------------------------
# The second /simplify: the attempt read from the log's time stamps
# --------------------------------------------------------------------------------------


def test_an_earlier_attempts_late_iterations_do_not_close_a_requeued_jobs_window(tmp_path):
    """The guard last saw job 10 at iteration 30; the attempt then logged 31 to 60 and failed, and SLURM requeued the
    job under the same ID. Those lines are stamped before the new start, so the new attempt's window stays open and it
    is stopped 50 minutes after its own start."""
    config = sg.load_guard_config(write_config(tmp_path, startup={"first_iteration_minutes": 50}))
    segment(tmp_path, 10, drop=set(range(31, 61)))
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)])
    _, state, _ = sg.tick(config, sg.START, cluster, START_TIME + timedelta(minutes=10))

    segment(tmp_path, 10)
    restart = START_TIME + timedelta(hours=2)
    cluster.sacct_lines = [sacct_line(10, "RUNNING", tmp_path, start=restart)]
    action, state, _ = sg.tick(config, state, cluster, restart + timedelta(minutes=10))
    assert action != "stop"
    action, _, _ = sg.tick(config, state, cluster, restart + timedelta(minutes=51))

    assert action == "stop" and cluster.cancelled() == [["scancel", "10"]]


def test_a_restarted_guard_times_a_stall_from_the_log(tmp_path):
    """A guard started 20 minutes after the segment's last iteration line stops it at once with stall_minutes 15:
    the stall counts from the log, not from the guard's first tick."""
    config = sg.load_guard_config(write_config(tmp_path, stall_minutes=15))
    log = segment(tmp_path, 10, drop=set(range(21, 61)))
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)])

    action, _, _ = sg.tick(config, sg.START, cluster, stamp_of(log, 20) + timedelta(minutes=20))

    assert action == "stop" and cluster.cancelled() == [["scancel", "10"]]


def test_a_requeued_attempt_takes_its_own_start_up_dumps_into_its_own_folders(tmp_path):
    config = sg.load_guard_config(
        write_config(tmp_path, startup={"first_iteration_minutes": 50, "dump_at_minutes": [20]})
    )
    no_iteration_yet(tmp_path, 10)
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)])
    _, state, _ = sg.tick(config, sg.START, cluster, START_TIME + timedelta(minutes=21))

    restart = START_TIME + timedelta(hours=2)
    cluster.sacct_lines = [sacct_line(10, "RUNNING", tmp_path, start=restart)]
    sg.tick(config, state, cluster, restart + timedelta(minutes=21))

    assert cluster.dumps() == [f"10/{ATTEMPT}_startup_m20", f"10/{restart:%Y%m%dT%H%M%S}_startup_m20"]


# --------------------------------------------------------------------------------------
# The third /code-review: unstamped lines, clocks, the final save, a restarted guard's dumps, the end's squeue
# --------------------------------------------------------------------------------------


def unstamped(text: str) -> str:
    """``text`` with the time stamp taken off every iteration line."""
    return ITERATION_STAMP_RE.sub(lambda match: match.group(0).replace("[" + match.group(1) + "] ", ""), text)


def unstamped_log(tmp_path: Path) -> None:
    """Job 10 logged iterations 1 to 20, none with a stamp."""
    log = segment(tmp_path, 10, drop=set(range(21, 61)))
    log.write_text(unstamped(log.read_text()))


def late_stamped_log(tmp_path: Path) -> None:
    """Job 10 logged iterations 1 to 20, stamped from two hours after the guard's clock (NOW)."""
    log = segment(tmp_path, 10, drop=set(range(21, 61)))
    log.write_text(restamped(log.read_text(), NOW + timedelta(hours=2)))


@pytest.mark.parametrize(
    "write_log, now, reason",
    [
        # Iteration lines without a stamp: the attempt has logged iterations, so its start-up window must not stop it.
        (unstamped_log, START_TIME + timedelta(minutes=51), "log has iteration lines without a time stamp"),
        # Stamps two hours after the guard's clock: the job and the guard read different time zones.
        (late_stamped_log, NOW, "last iteration line is stamped"),
        # No iteration yet, and sacct's start lies ten minutes after the guard's clock.
        (lambda tmp_path: no_iteration_yet(tmp_path, 10), START_TIME - timedelta(minutes=10), "start is"),
    ],
    ids=["unstamped-lines", "stamp-after-the-clock", "start-after-the-clock"],
)
def test_a_tick_that_cannot_time_the_segment_leaves_the_time_limits_unevaluated(tmp_path, write_log, now, reason):
    """Neither time limit can judge a segment whose log carries no stamp or whose stamp or start lies after the guard's
    clock, so the tick stops nothing and says so."""
    config = sg.load_guard_config(write_config(tmp_path, startup={"first_iteration_minutes": 50}, stall_minutes=15))
    write_log(tmp_path)
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)])

    action, _, _ = sg.tick(config, sg.START, cluster, now)

    assert action != "stop" and not cluster.cancelled() and not cluster.dumps()
    assert any(f"time limits not evaluated: job 10's {reason}" in line for line in record_lines(config))


def test_minutes_are_counted_across_a_daylight_saving_change(monkeypatch):
    """Twenty minutes pass between 00:50 GMT and 02:10 BST on 28 March 2027, not eighty."""
    monkeypatch.setenv("TZ", "Europe/London")
    time.tzset()
    try:
        assert sg._minutes_since(datetime(2027, 3, 28, 0, 50), datetime(2027, 3, 28, 2, 10)) == 20
    finally:
        monkeypatch.undo()
        time.tzset()


def test_a_stall_limit_needs_the_final_iteration(tmp_path):
    """Without final_iteration the stall rule cannot stop at the final save, so the config is refused."""
    path = write_config(tmp_path, stall_minutes=15)
    raw = yaml.safe_load(path.read_text())
    del raw["final_iteration"]
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="stall_minutes needs final_iteration"):
        sg.load_guard_config(path)


def test_a_restarted_guard_does_not_take_a_dump_its_record_shows_taken(tmp_path):
    """The first guard took job 10's minute-20 dump; a guard started again at minute 30 reads it from the record and
    does not take it again into the same folder."""
    config = sg.load_guard_config(
        write_config(tmp_path, startup={"first_iteration_minutes": 50, "dump_at_minutes": [20]})
    )
    no_iteration_yet(tmp_path, 10)
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)])
    sg.tick(config, sg.START, cluster, START_TIME + timedelta(minutes=21))
    assert cluster.dumps() == [f"10/{ATTEMPT}_startup_m20"]

    sg.tick(config, sg.state_from_record(config), cluster, START_TIME + timedelta(minutes=30))

    assert cluster.dumps() == [f"10/{ATTEMPT}_startup_m20"]


def test_a_failing_squeue_at_the_end_leaves_the_guard_ticking(tmp_path):
    """The watch has checked the final iteration and sacct shows the segment completed, but squeue fails: the end is
    not confirmed, and the guard goes on to the next tick rather than failing."""
    config = sg.load_guard_config(write_config(tmp_path))
    segment(tmp_path, 10)
    cluster = FakeCluster([sacct_line(10, "COMPLETED", tmp_path)])
    cluster.squeue_status = 1

    action, _, done = sg.tick(config, sg.START, cluster, NOW)

    assert action == "continue" and not done
    assert any("END NOT CONFIRMED: SlurmError" in line for line in record_lines(config))

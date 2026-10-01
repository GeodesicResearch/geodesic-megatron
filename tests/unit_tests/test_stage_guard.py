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
import shlex
import subprocess
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pytest
import yaml
from scripts.telemetry import run_watch as rw
from scripts.telemetry.code_revision import code_revision
from scripts.training import stage_guard as sg

from tests.unit_tests.test_loss_parity import write_run
from tests.unit_tests.test_run_watch import gated, rejected_result_line  # noqa: F401 - gated is a fixture


JOB_NAME = "cp-stage"


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


def sacct_line(job_id: int, state: str, log_dir: Path, started: bool = True, name: str = JOB_NAME) -> str:
    start = "2026-10-01T06:00:00" if started else "Unknown"
    return f"{job_id}|{name}|{state}|{start}|{log_dir}/train-%j.out"


LIVE = ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "RESIZING", "SUSPENDED")


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
            name = command[command.index("--name") + 1]
            live = [
                line.split("|")[0]
                for line in self.sacct_lines
                if line.split("|")[1] == name and line.split("|")[2].split()[0] in LIVE
            ]
            return subprocess.CompletedProcess(command, 0, "".join(f"{job_id}\n" for job_id in live))
        if command[0] == "scancel":
            return subprocess.CompletedProcess(command, self.scancel_status, "")
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
    """A container or activation failure can exit 1; only the watch's summary line makes 1 a stop."""
    assert sg.parse_watch(status, "checked through iteration 55: 0 stop conditions, 0 gates due\n").ran
    assert not sg.parse_watch(status, "FATAL [env-config]: SIF not found\n").ran


def test_the_watch_names_the_gates_it_passed_and_those_still_undecided():
    output = (
        "GATE L1: PASS (passed at an earlier check on this log) on /logs/train-10.out\n"
        "GATE L2: PASS on /logs/train-10.out\nGATE L2b: NOT EVALUATED (x)\n"
        "undecided gates: L2b\nchecked through iteration 2210: 0 stop conditions, 3 gates due\n"
    )
    result = sg.parse_watch(2, output)
    assert result.passed == {"L1": "/logs/train-10.out", "L2": "/logs/train-10.out"}
    assert result.undecided == ("L2b",)
    assert (
        sg.parse_watch(0, "undecided gates: none\nchecked through iteration 1: 0 stop conditions, 0 gates due\n")[5]
        == ()
    )
    assert sg.parse_watch(0, "checked through iteration 1: 0 stop conditions, 0 gates due\n").undecided is None


# --------------------------------------------------------------------------------------
# What a tick does
# --------------------------------------------------------------------------------------


def test_a_clean_stage_is_left_alone(tmp_path):
    config = sg.load_guard_config(write_config(tmp_path))
    segment(tmp_path, 10, drop=set(range(31, 61)))
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path), sacct_line(11, "PENDING", tmp_path, started=False)])
    action, state, done = sg.tick(config, sg.START, cluster)
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
    action, _, _ = sg.tick(config, sg.START, cluster)
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
    action, _, _ = sg.tick(config, sg.START, cluster)
    assert action == "stop" and cluster.cancelled() == [["scancel", "11"]]


@pytest.mark.parametrize("iteration, action", [(39, "alert"), (40, "hold"), (60, "hold"), (1000, "hold")])
def test_an_unevaluated_gate_holds_the_stage_from_its_hold_iteration_on(tmp_path, iteration, action):
    """From there the next save would be written past the unevaluated gate; a tick that comes late, past the first
    save, still holds."""
    config = sg.load_guard_config(write_config(tmp_path))
    output = (
        f"GATE L1: NOT EVALUATED (x)\nundecided gates: L1\n"
        f"checked through iteration {iteration}: 0 stop conditions, 1 gates due\n"
    )
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)], watch_status=2, watch_output=output)
    got, _, _ = sg.tick(config, sg.START, cluster)
    assert got == action
    assert bool(cluster.cancelled()) == (action == "hold")


def test_inside_the_window_an_unevaluated_stop_check_alone_does_not_hold(tmp_path):
    """Every gate has passed, so no save can be written past an undecided one."""
    config = sg.load_guard_config(write_config(tmp_path))
    output = (
        "NOT EVALUATED stop launch_settings: ValueError: x\nundecided gates: none\n"
        "checked through iteration 45: 0 stop conditions, 1 gates due\n"
    )
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)], watch_status=2, watch_output=output)
    assert sg.tick(config, sg.START, cluster)[0] == "alert" and not cluster.cancelled()


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
    got, after, _ = sg.tick(config, state, cluster)
    assert got == action and after == state


def test_jobs_that_cannot_be_listed_inside_the_window_hold_the_stage(tmp_path):
    """The same rule as a watch that could not run; the cancellation then needs sacct too, and says it failed."""
    config = sg.load_guard_config(write_config(tmp_path))
    with pytest.raises(sg.CancelFailed, match="could not cancel the stage's jobs: SlurmError: squeue exited 1"):
        sg.tick(config, sg.GuardState(45, {}, ("L1",)), failing_sacct)
    assert "HOLD: could not evaluate: RuntimeError: sacct failed" in config.record.read_text()


def test_jobs_that_cannot_be_listed_once_every_gate_has_passed_alert(tmp_path):
    config = sg.load_guard_config(write_config(tmp_path))
    action, _, _ = sg.tick(config, sg.GuardState(45, {"L1": "/x.out"}, ()), failing_sacct)
    assert action == "alert"


def test_a_gate_passed_once_is_not_evaluated_again_on_its_log(tmp_path, gated):
    """The second tick names the gate decided, so a reference that can no longer be read cannot unsettle it."""
    config = sg.load_guard_config(write_config(tmp_path, watch=str(gated)))
    log = segment(tmp_path, 10, loss_offset={it: 0.005 for it in range(1, 61)})
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)])
    action, state, _ = sg.tick(config, sg.START, cluster)
    assert action == "continue" and state.decided == {"G": str(log)} and state.undecided == ()
    (tmp_path / "ref_a.out").unlink()
    action, state, _ = sg.tick(config, state, cluster)
    assert action == "continue"
    assert cluster.watch_arguments()[-1][-2:] == ["--decided", f"G={log}"]


def test_only_a_tick_whose_watch_ran_seeds_the_decided_gates_and_a_later_tick_wins(tmp_path):
    record = tmp_path / "record.log"
    assert sg.decided_in_record(record) == {}
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
    assert sg.decided_in_record(record) == {"G": "/c.out"}


def test_a_log_path_with_a_space_reaches_the_watch_whole(tmp_path):
    config = sg.load_guard_config(write_config(tmp_path))
    log_dir = tmp_path / "with space"
    log_dir.mkdir()
    write_run(log_dir, "train-10", drop=set(range(31, 61)))
    cluster = FakeCluster([sacct_line(10, "RUNNING", log_dir)])
    action, state, _ = sg.tick(config, sg.START, cluster)
    assert (action, state.last_iteration) == ("continue", 30)


def test_the_stage_is_done_once_its_final_iteration_is_checked_and_no_job_is_live(tmp_path):
    config = sg.load_guard_config(write_config(tmp_path))
    segment(tmp_path, 10)
    cluster = FakeCluster([sacct_line(10, "COMPLETED", tmp_path)])
    action, state, done = sg.tick(config, sg.START, cluster)
    assert (action, state.last_iteration, done) == ("continue", 60, True)


def test_a_stage_with_no_started_segment_fails_the_guard(tmp_path):
    """The guard is started once a segment trains, so finding none means a wrong name or user, not a wait."""
    config = sg.load_guard_config(write_config(tmp_path))
    cluster = FakeCluster([sacct_line(10, "PENDING", tmp_path, started=False)])
    with pytest.raises(sg.NoStartedSegment, match="no segment of cp-stage has started"):
        sg.tick(config, sg.START, cluster)


def test_a_stop_cancels_before_it_records(tmp_path):
    """A record that cannot be written must not keep the stage's jobs alive."""
    (tmp_path / "not-a-directory").write_text("")
    config = sg.load_guard_config(write_config(tmp_path, record=str(tmp_path / "not-a-directory" / "record.log")))
    segment(tmp_path, 10, grad_scale={20: float("inf")}, drop=set(range(31, 61)))
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path), sacct_line(11, "PENDING", tmp_path, started=False)])
    with pytest.raises(OSError):
        sg.tick(config, sg.START, cluster)
    assert cluster.cancelled() == [["scancel", "10", "11"]]


def test_a_failing_squeue_fails_the_cancellation_rather_than_reading_as_no_jobs(tmp_path):
    config = sg.load_guard_config(write_config(tmp_path))
    segment(tmp_path, 10, grad_scale={20: float("inf")}, drop=set(range(31, 61)))
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)])
    cluster.squeue_status = 1
    with pytest.raises(sg.CancelFailed, match="squeue exited 1"):
        sg.tick(config, sg.START, cluster)
    assert not cluster.cancelled()


def test_a_watch_past_its_timeout_counts_as_not_evaluated(tmp_path):
    """Outside the hold it alerts; it is never read as a stop."""
    config = sg.load_guard_config(write_config(tmp_path))
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)])
    cluster.timing_out.add("pipeline_env_exec.sh")
    action, state, _ = sg.tick(config, sg.START, cluster)
    assert action == "alert" and state == sg.START and not cluster.cancelled()
    assert "could not evaluate: TimeoutExpired" in config.record.read_text()


@pytest.mark.parametrize("command", ["squeue", "scancel"])
def test_a_cancellation_whose_command_times_out_fails(tmp_path, command):
    config = sg.load_guard_config(write_config(tmp_path))
    segment(tmp_path, 10, grad_scale={20: float("inf")}, drop=set(range(31, 61)))
    cluster = FakeCluster([sacct_line(10, "RUNNING", tmp_path)])
    cluster.timing_out.add(command)
    with pytest.raises(sg.CancelFailed, match="TimeoutExpired"):
        sg.tick(config, sg.START, cluster)


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

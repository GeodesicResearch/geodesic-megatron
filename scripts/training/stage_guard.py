#!/usr/bin/env python3
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

"""Guard a training stage while it trains: on a timer, judge its segments with run_watch.py and act on the outcome.

A guard config (YAML) names the stage's watch spec, its SLURM job name, the interval, the timeout of each command
it runs, the record file and, optionally, the iteration a hold starts at and the stage's final iteration. Each tick
finds the stage's segments (``sacct``: every job of that name that has started, in submission order, its log being
the job's own ``StdOut``), runs ``scripts/telemetry/run_watch.py`` on them inside the container, and acts:

- **stop** (the watch exits 1 with its summary lines: a stop condition or a failed loss gate): cancel every live job
  of the stage by job ID, then exit 1. The live jobs are the union of ``squeue``'s, which lists the successors
  pending on the running segment's singleton dependency (``sacct -S`` lists no job that has not started), and the
  started segments ``sacct`` shows in a live state, so one empty ``squeue`` reply cannot make a stop cancel nothing;
- **hold** (the watch exits 2, or the tick could not be evaluated, e.g. the watch could not run or ``sacct`` could not
  list the jobs, once the stage has reached ``hold.from_iteration`` while a watched loss gate is still undecided):
  cancel the same way, so no save is written past an unevaluated gate, then exit 3. The iteration is the watch's own,
  or the last one a tick read when this tick's watch could not run; a gate is undecided until a watch has named it
  passed, so before any watch has run every gate is;
- **alert** (any other exit 2 or failure to evaluate): the record and the printed line say so, and the guard keeps
  going; a tick that could not be evaluated is never read as a stop;
- the guard exits 0 once the watch has checked the final iteration, sacct shows every started segment in a final
  state and squeue lists no live job of the stage (an empty squeue reply alone does not end it).

Two time limits are optional and off unless the guard config sets them. Both judge the newest segment by the
iterations it has logged itself (the watch's ``newest segment through iteration`` line), never by its predecessors',
and act only on a tick whose watch ran. ``startup: {first_iteration_minutes: N, dump_at_minutes: [...]}``: a running
segment that has logged no iteration of its own is stopped N minutes after it started (sacct's start time), with a
stack dump of every rank (``scripts/training/dump_hung_ranks.sh``) at each listed minute and one before the cancel;
the guard may then be started as soon as the segment starts. ``stall_minutes: M``: after its first iteration, M
minutes without a new iteration of its own is a stall, stopped the same way. Both stop with exit 1, as a watch's stop
does.

Each tick appends its line and the watch's output to the record, after cancelling on a stop or hold so a record that
cannot be written cannot keep the jobs alive. A stop or hold whose cancellation fails exits 4. The guard is started
once the stage's newest segment has logged its first iteration, so a tick that finds no started segment of the
stage (a wrong job name, say) is a failure of the guard itself; that, and any other such failure, is written to the
record when it can be and exits 5, so neither reads as a stop.

A gate the watch has passed on a log is passed to later ticks as ``--decided GATE=LOG``, so the watch does not
evaluate it again while that log still covers its range, and a transient failure to read a reference (W&B, say)
cannot unsettle it. A guard started again takes as decided the gates its record shows the watch passing under the
same watch spec, so a restart inside the hold window does not re-read a reference for a gate an earlier guard saw
pass, while a gate passed under an edited spec is judged again. The record starts
with the guard config, the watch spec and its sha256, the code revision and the gates decided from the record, and
every tick names the watch spec's sha256, so each verdict can be traced to the spec it was judged by.

The guard acts only on jobs that carry the stage's name and belong to the user running it, and only by ID. After
a stop or a hold it does not resume: whoever resolves the cause starts it again. It runs on a login or tunnel node
under the host Python (SLURM's commands do not exist inside the container), so it keeps to the standard library and
PyYAML and stays Python 3.6-compatible.

USAGE
    python3 scripts/training/stage_guard.py --config <guard.yaml> [--once]
"""

import argparse
import getpass
import hashlib
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
# Run as a script, only scripts/training/ is on sys.path; the repo root makes the shared modules importable.
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.slurm_jobs import queue_listing  # noqa: E402
from scripts.telemetry.code_revision import code_revision  # noqa: E402


# The watch's own summary line; its presence is what tells a watch that ran from one that could not.
_SUMMARY_RE = re.compile(r"^checked through iteration (\S+): ", re.M)
# A gate the watch passed, and the log it passed on (the line's last field).
_PASSED_RE = re.compile(r"^GATE (\S+): PASS(?: \(.*\))? on (.+)$", re.M)
_UNDECIDED_RE = re.compile(r"^undecided gates: (.*)$", re.M)
# The last iteration the newest segment has logged itself, which the time limits judge it by.
_NEWEST_RE = re.compile(r"^newest segment through iteration (\S+): ", re.M)
# The timestamp that opens every line the guard writes; a watch's output follows its tick's line unstamped.
_STAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z ", re.M)
# A tick whose watch ran: the watch's exit status, and the leading sha256 of the spec the tick judged by.
_RAN_TICK_RE = re.compile(r"^\S+ [A-Z]+: watch exit (\d+) through iteration .* spec (\S+)$")
# A container or activation failure must not exit 1, which reads as a stop.
_ACTIVATION_FAILED = 3


class GuardConfig(NamedTuple):
    """A stage's guard config (see the module docstring); ``hold_from`` is None when the config holds nowhere."""

    watch: Path
    job_name: str
    since: str
    interval_seconds: int
    record: Path
    command_timeout_seconds: int
    hold_from: Optional[int]
    final_iteration: Optional[int]
    startup_minutes: Optional[int] = None
    startup_dump_minutes: Tuple[int, ...] = ()
    stall_minutes: Optional[int] = None


class Job(NamedTuple):
    """One job of the stage: its state, whether it has started, its log, and its start time (None before it starts)."""

    job_id: int
    state: str
    started: bool
    log: Path
    start: Optional[datetime] = None


class WatchResult(NamedTuple):
    """One run of the watch: its exit status (None when it was not run), the iteration it checked through, whether
    it ran, its output, the gates it passed with the log each passed on, the gates it names undecided (None when it
    names none, which leaves them unknown), and the last iteration the newest segment has logged itself (None before
    its first)."""

    status: Optional[int]
    checked_iteration: Optional[int]
    ran: bool
    output: str
    passed: Dict[str, str]
    undecided: Optional[Tuple[str, ...]]
    newest_iteration: Optional[int] = None


class GuardState(NamedTuple):
    """What the guard carries from tick to tick: the last iteration a watch checked, the gates passed and the log
    each passed on, the gates still undecided (None until a watch has run), the newest segment's progress as (job
    ID, the last iteration it has logged itself) and the time a tick first saw it, and the dumps already taken as
    (job ID, minute)."""

    last_iteration: Optional[int]
    decided: Dict[str, str]
    undecided: Optional[Tuple[str, ...]]
    progress: Optional[Tuple[int, int]] = None
    changed_at: Optional[datetime] = None
    dumped: Tuple[Tuple[int, int], ...] = ()


START = GuardState(last_iteration=None, decided={}, undecided=None)


def load_guard_config(path: Path) -> GuardConfig:
    """Read a guard config; the watch spec and record paths resolve against the repository root.

    Raises ValueError on an unknown key or a hold other than ``from_iteration``; KeyError on a missing required key.
    """
    raw = yaml.safe_load(Path(path).read_text())
    keys = {
        "watch", "job_name", "since", "interval_seconds", "command_timeout_seconds", "record", "hold", "final_iteration",
        "startup", "stall_minutes",
    }  # fmt: skip
    unknown = sorted(set(raw) - keys)
    if unknown:
        raise ValueError("{}: unknown keys {}".format(path, unknown))
    hold = raw.get("hold") or {}
    if hold and set(hold) != {"from_iteration"}:
        raise ValueError("{}: hold needs exactly from_iteration".format(path))
    startup = raw.get("startup") or {}
    if startup and (
        not set(startup) <= {"first_iteration_minutes", "dump_at_minutes"} or "first_iteration_minutes" not in startup
    ):
        raise ValueError("{}: startup needs first_iteration_minutes and may give dump_at_minutes".format(path))
    startup_minutes = int(startup["first_iteration_minutes"]) if startup else None
    dump_minutes = tuple(sorted(int(minute) for minute in startup.get("dump_at_minutes", ()))) if startup else ()
    if startup_minutes is not None and (
        startup_minutes < 1 or any(not 1 <= m < startup_minutes for m in dump_minutes)
    ):
        raise ValueError(
            "{}: startup.first_iteration_minutes must be 1 or more and each dump_at_minutes below it".format(path)
        )
    stall_minutes = int(raw["stall_minutes"]) if "stall_minutes" in raw else None
    if stall_minutes is not None and stall_minutes < 1:
        raise ValueError("{}: stall_minutes must be 1 or more".format(path))
    return GuardConfig(
        watch=REPO_ROOT / raw["watch"],
        job_name=str(raw["job_name"]),
        since=str(raw["since"]),
        interval_seconds=int(raw["interval_seconds"]),
        command_timeout_seconds=int(raw["command_timeout_seconds"]),
        record=REPO_ROOT / raw["record"],
        hold_from=int(hold["from_iteration"]) if hold else None,
        final_iteration=int(raw["final_iteration"]) if "final_iteration" in raw else None,
        startup_minutes=startup_minutes,
        startup_dump_minutes=dump_minutes,
        stall_minutes=stall_minutes,
    )


def parse_jobs(sacct_output: str, job_name: str) -> List[Job]:
    """The stage's jobs from ``sacct -X -n -P -o JobID,JobName,State,Start,StdOut`` output, in submission order,
    each log path expanded from its ``%j`` and ``%x`` patterns. Raises ValueError on a malformed line."""
    jobs = []
    for line in sacct_output.splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) != 5:
            raise ValueError("sacct line is not JobID|JobName|State|Start|StdOut: {!r}".format(line))
        job_id, name, state, start, stdout = fields
        if name != job_name:
            continue
        state = state.split()[0]
        log = stdout.replace("%j", job_id).replace("%x", name)
        started = start not in ("Unknown", "None", "")
        started_at = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S") if started else None
        jobs.append(Job(int(job_id), state, started, Path(log), started_at))
    return sorted(jobs, key=lambda job: job.job_id)


def parse_watch(status: int, output: str) -> WatchResult:
    """The watch's outcome from its exit status and output. A status of 0, 1 or 2 counts only beside the watch's
    summary lines (``newest segment through iteration`` and ``checked through iteration``): without them the watch did
    not run (the container or the activation failed) and ``ran`` is False."""
    match = _SUMMARY_RE.search(output)
    newest_match = _NEWEST_RE.search(output)
    ran = match is not None and newest_match is not None and status in (0, 1, 2)
    newest = None
    if newest_match is not None and newest_match.group(1) != "None":
        newest = int(newest_match.group(1))
    checked = None
    if match is not None and match.group(1) != "None":
        checked = int(match.group(1))
    undecided_match = _UNDECIDED_RE.search(output)
    undecided = None
    if undecided_match is not None:
        listed = undecided_match.group(1)
        undecided = () if listed == "none" else tuple(name.strip() for name in listed.split(","))
    passed = dict(_PASSED_RE.findall(output)) if ran else {}
    return WatchResult(status, checked, ran, output, passed, undecided, newest)


def decided_in_record(record: Path, spec: str) -> Dict[str, str]:
    """The gates the watch passed in the ticks ``record`` holds that judged by the watch spec whose sha256 begins
    ``spec``, each with the log it passed on (a later tick's winning), read from each such tick whose watch ran as
    that tick read it; empty when there is no record. A gate passed under another spec is judged again."""
    if not record.exists():
        return {}
    text = record.read_text()
    starts = [match.start() for match in _STAMP_RE.finditer(text)] + [len(text)]
    decided: Dict[str, str] = {}
    for begin, end in zip(starts, starts[1:]):
        line, _, output = text[begin:end].partition("\n")
        ran = _RAN_TICK_RE.match(line)
        if ran is not None and ran.group(2) == spec:
            decided.update(parse_watch(int(ran.group(1)), output).passed)
    return decided


def could_not_evaluate(reason: str) -> WatchResult:
    """The result of a tick that could not be evaluated: the stage's jobs could not be listed, or the watch could
    not be run."""
    return WatchResult(None, None, False, reason, {}, None)


def advance(state: GuardState, result: WatchResult) -> GuardState:
    """The state after a tick's watch result: a watch that ran replaces the undecided gates and adds the gates it
    passed; one that did not leaves both as they were."""
    if not result.ran:
        return state
    iteration = result.checked_iteration if result.checked_iteration is not None else state.last_iteration
    decided = dict(state.decided)
    decided.update(result.passed)
    return state._replace(last_iteration=iteration, decided=decided, undecided=result.undecided)


def decide(result: WatchResult, state: GuardState, config: GuardConfig) -> str:
    """``continue``, ``alert``, ``stop`` or ``hold`` for one tick's watch result, ``state`` being the guard's state
    after it (see ``advance``)."""
    if result.ran and result.status == 0:
        return "continue"
    if result.ran and result.status == 1:
        return "stop"
    iteration = state.last_iteration
    holding = config.hold_from is not None and iteration is not None and iteration >= config.hold_from
    gate_open = state.undecided is None or bool(state.undecided)
    return "hold" if holding and gate_open else "alert"


Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess"]


def run_command(command: Sequence[str], timeout_seconds: int) -> "subprocess.CompletedProcess":
    """Run a command, its stderr merged into its stdout (Python 3.6 has no ``capture_output``). Raises
    subprocess.TimeoutExpired, after killing it, when it outlives ``timeout_seconds``."""
    return subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        check=False,
        timeout=timeout_seconds,
    )


def stage_jobs(config: GuardConfig, run: Runner) -> List[Job]:
    """Every job of the stage's name the current user has submitted since ``config.since``.

    Raises RuntimeError when sacct fails, ValueError on output it cannot read."""
    result = run(
        [
            "sacct", "-X", "-n", "-P", "-u", getpass.getuser(), "--name", config.job_name, "-S", config.since,
            "-o", "JobID,JobName,State,Start,StdOut",
        ]
    )  # fmt: skip
    if result.returncode != 0:
        raise RuntimeError("sacct failed ({}): {}".format(result.returncode, result.stdout.strip()))
    return parse_jobs(result.stdout, config.job_name)


def run_watch(config: GuardConfig, logs: List[Path], decided: Dict[str, str], run: Runner) -> WatchResult:
    """run_watch.py on the logs inside the container, passing the gates already decided."""
    arguments = ["--spec", str(config.watch)]
    for log in logs:
        arguments += ["--log", str(log)]
    for gate, log in sorted(decided.items()):
        arguments += ["--decided", "{}={}".format(gate, log)]
    payload = (
        "cd {repo}; source pipeline_env_activate.sh || exit {failed}; python scripts/telemetry/run_watch.py {args}"
    )
    result = run(
        [
            str(REPO_ROOT / "pipeline_env_exec.sh"),
            payload.format(
                repo=shlex.quote(str(REPO_ROOT)),
                failed=_ACTIVATION_FAILED,
                args=" ".join(shlex.quote(argument) for argument in arguments),
            ),
        ]
    )
    return parse_watch(result.returncode, result.stdout)


# The states in which a job is still in the queue; any other state is final.
LIVE_STATES = ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "RESIZING", "SUSPENDED")


def live_jobs(config: GuardConfig, segments: List[Job], run: Runner) -> List[int]:
    """The IDs of every job of the stage's name the current user has queued or running: the union of what squeue
    lists (the successors pending on a singleton dependency included) and the ``segments`` sacct shows in a live
    state, so one empty squeue reply from a busy controller does not hide a running segment. Raises SlurmError when
    squeue fails and ValueError on output it cannot read."""
    queued = {int(job_id) for job_id in queue_listing("%i", config.job_name, run)}
    return sorted(queued | {job.job_id for job in segments if job.state in LIVE_STATES})


class CancelFailed(RuntimeError):
    """A stop or hold could not cancel the stage's live jobs."""


def cancel_live_jobs(config: GuardConfig, segments: List[Job], run: Runner) -> List[int]:
    """Cancel, by job ID, every live job of the stage (``live_jobs``); return the IDs. Raises CancelFailed when the
    jobs cannot be listed or scancel fails."""
    try:
        live = live_jobs(config, segments, run)
        if live:
            result = run(["scancel"] + [str(job_id) for job_id in live])
            if result.returncode != 0:
                raise CancelFailed("scancel {} failed: {}".format(live, result.stdout.strip()))
    except (RuntimeError, ValueError, subprocess.SubprocessError) as error:
        if isinstance(error, CancelFailed):
            raise
        raise CancelFailed("could not cancel the stage's jobs: {}: {}".format(type(error).__name__, error))
    return live


def sha256_of(path: Path) -> str:
    """The hex sha256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spec_tag(watch: Path) -> str:
    """The leading 12 hex digits of the watch spec's sha256, which each tick line ends with."""
    return sha256_of(watch)[:12]


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _record(config: GuardConfig, line: str, detail: str = "") -> None:
    config.record.parent.mkdir(parents=True, exist_ok=True)
    with open(str(config.record), "a") as handle:
        handle.write("{} {}\n".format(_stamp(), line))
        if detail:
            handle.write(detail if detail.endswith("\n") else detail + "\n")
    print("{} {}".format(_stamp(), line), flush=True)


DUMP_SCRIPT = REPO_ROOT / "scripts" / "training" / "dump_hung_ranks.sh"


def timed_check(
    config: GuardConfig, state: GuardState, newest: Job, now: datetime
) -> Tuple[Optional[str], Tuple[int, ...], str]:
    """The start-up window and the stall limit for the stage's newest segment: (``"stop"``, (), reason), (``"dump"``,
    the dump minutes now due, reason) or (None, (), "").

    The segment is in its start-up window until a watch has seen it log an iteration of its own (``state.progress``
    names it); the window counts from the segment's start (sacct) and is ``config.startup_minutes`` long, with a stack
    dump at each of ``config.startup_dump_minutes``. After that, ``config.stall_minutes`` without a new iteration of its
    own is a stall. Neither rule applies when its key is not set, or when the segment is not running."""
    if newest.state != "RUNNING" or newest.start is None:
        return None, (), ""
    if state.progress is None or state.progress[0] != newest.job_id:
        if config.startup_minutes is None:
            return None, (), ""
        minutes = (now - newest.start).total_seconds() / 60
        if minutes >= config.startup_minutes:
            return (
                "stop",
                (),
                "start-up window: job {} logged no iteration within {} min of starting".format(
                    newest.job_id, config.startup_minutes
                ),
            )
        due = tuple(m for m in config.startup_dump_minutes if minutes >= m and (newest.job_id, m) not in state.dumped)
        if due:
            return (
                "dump",
                due,
                "job {} has logged no iteration {:.0f} min after starting".format(newest.job_id, minutes),
            )
        return None, (), ""
    if config.stall_minutes is not None and (now - state.changed_at).total_seconds() / 60 >= config.stall_minutes:
        return (
            "stop",
            (),
            "stall: job {} logged no new iteration for {} min (its own last iteration {})".format(
                newest.job_id, config.stall_minutes, state.progress[1]
            ),
        )
    return None, (), ""


def dump_stacks(config: GuardConfig, job_id: int, run: Runner) -> None:
    """Run dump_hung_ranks.sh on the job and record how it went; a dump that fails or times out is recorded and does
    not stop the guard, so a stop that follows still cancels."""
    try:
        result = run(["bash", str(DUMP_SCRIPT), str(job_id)])
        outcome = "exit {}: {}".format(result.returncode, " | ".join(result.stdout.strip().splitlines()[-3:]))
    except (OSError, subprocess.SubprocessError) as error:
        outcome = "{}: {}".format(type(error).__name__, error)
    _record(config, "DUMP: stacks of job {} ({}): {}".format(job_id, DUMP_SCRIPT.name, outcome))


class NoStartedSegment(RuntimeError):
    """No job of the stage's name has started: the guard was started too early, or under the wrong name or user."""


def tick(config: GuardConfig, state: GuardState, run: Runner, now: datetime) -> Tuple[str, GuardState, bool]:
    """One evaluation at ``now`` (the cluster's local time, as sacct prints it): (action, the guard's state after it,
    whether the stage is done).

    Raises NoStartedSegment when the stage has no started segment, and CancelFailed when a stop or hold could not
    cancel its jobs."""
    started, spec = [], "unread"
    try:
        spec = spec_tag(config.watch)
        started = [job for job in stage_jobs(config, run) if job.started]
    except Exception as error:  # noqa: BLE001 - judged by the hold window like a watch that could not run
        result = could_not_evaluate("{}: {}".format(type(error).__name__, error))
    else:
        if not started:
            raise NoStartedSegment("no segment of {} has started".format(config.job_name))
        try:
            result = run_watch(config, [job.log for job in started], state.decided, run)
        except Exception as error:  # noqa: BLE001 - judged by the hold window like a watch that could not run
            result = could_not_evaluate("{}: {}".format(type(error).__name__, error))
    state = advance(state, result)
    if result.ran and started and result.newest_iteration is not None:
        progress = (started[-1].job_id, result.newest_iteration)
        if progress != state.progress:
            state = state._replace(progress=progress, changed_at=now)
    action = decide(result, state, config)
    segments = ", ".join("{}:{}".format(job.job_id, job.state) for job in started)
    if result.ran:
        summary = "watch exit {} through iteration {}".format(result.status, result.checked_iteration)
    elif result.status is None:
        summary = "could not evaluate: {}".format(result.output)
    else:
        summary = "watch could not run (exit {})".format(result.status)
    undecided = "unknown" if state.undecided is None else ", ".join(state.undecided) or "none"
    line = "{}: {}; undecided gates {} [{}] spec {}".format(action.upper(), summary, undecided, segments, spec)
    detail = result.output if result.status is not None else ""
    if action in ("stop", "hold"):
        try:
            cancelled = cancel_live_jobs(config, started, run)
        except CancelFailed:
            _record(config, line, detail)
            raise
        _record(config, line, detail)
        _record(config, "{}: cancelled {}".format(action.upper(), cancelled or "nothing (no live job)"))
        return action, state, False
    _record(config, line, detail)
    timed, due, reason = timed_check(config, state, started[-1], now) if result.ran else (None, (), "")
    if timed == "stop":
        _record(config, "STOP: {}".format(reason))
        dump_stacks(config, started[-1].job_id, run)
        cancelled = cancel_live_jobs(config, started, run)
        _record(config, "STOP: cancelled {}".format(cancelled or "nothing (no live job)"))
        return "stop", state, False
    if timed == "dump":
        _record(config, "STARTUP: {}; stack dump at minute {}".format(reason, max(due)))
        dump_stacks(config, started[-1].job_id, run)
        state = state._replace(dumped=state.dumped + tuple((started[-1].job_id, minute) for minute in due))
    # live_jobs includes the segments sacct shows live, so one empty squeue reply does not end the guard.
    done = (
        config.final_iteration is not None
        and result.ran
        and result.status == 0
        and state.last_iteration is not None
        and state.last_iteration >= config.final_iteration
        and not live_jobs(config, started, run)
    )
    return action, state, done


EXIT_STATUS = {"stop": 1, "hold": 3}
# A stop or hold whose cancellation failed: the stage may still be running, so a person must act now.
CANCEL_FAILED = 4
# The guard itself failed; the stage is no longer guarded.
GUARD_FAILED = 5


def guard(config_path: Path, config: GuardConfig, once: bool) -> int:
    """Guard the stage until a stop, a hold or its end (one tick when ``once``), and return its exit status.

    Raises CancelFailed when a stop or hold could not cancel the stage's jobs."""
    decided = decided_in_record(config.record, spec_tag(config.watch))
    _record(
        config,
        "START guard {} (sha256 {}); watch spec {} (sha256 {}); code {}; decided from the record: {}".format(
            config_path,
            sha256_of(config_path),
            config.watch,
            sha256_of(config.watch),
            code_revision(str(REPO_ROOT)),
            ", ".join("{}={}".format(gate, log) for gate, log in sorted(decided.items())) or "none",
        ),
    )
    state = START._replace(decided=decided)
    while True:
        try:
            action, state, done = tick(
                config, state, lambda command: run_command(command, config.command_timeout_seconds), datetime.now()
            )
        except CancelFailed as error:
            _record(config, "CANCEL FAILED: {}; the stage may still be running".format(error))
            raise
        if action in EXIT_STATUS:
            return EXIT_STATUS[action]
        if done:
            _record(config, "DONE: the watch checked iteration {} and no job is live".format(state.last_iteration))
            return 0
        if once:
            return 2 if action == "alert" else 0
        time.sleep(config.interval_seconds)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the guard and return the exit status the module docstring names."""
    parser = argparse.ArgumentParser(description="Guard a training stage with its watch spec while it trains.")
    parser.add_argument("--config", type=Path, required=True, help="The stage's guard config YAML")
    parser.add_argument("--once", action="store_true", help="Evaluate once and exit with that tick's status")
    args = parser.parse_args(argv)
    config = None
    try:
        config = load_guard_config(args.config)
        return guard(args.config, config, args.once)
    except CancelFailed:
        return CANCEL_FAILED
    except Exception as error:  # noqa: BLE001 - the guard's own failure has its own status, never a stop's
        line = "GUARD FAILED: {}: {}; the stage is unguarded".format(type(error).__name__, error)
        print(line, file=sys.stderr)
        if config is not None:
            try:
                _record(config, line)
            except OSError as record_error:
                print("GUARD FAILED: the record cannot be written: {}".format(record_error), file=sys.stderr)
        return GUARD_FAILED


if __name__ == "__main__":
    sys.exit(main())

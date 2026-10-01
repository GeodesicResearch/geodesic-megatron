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

A guard config (YAML) names the stage's watch spec, its SLURM job name, the interval, the record file and,
optionally, a hold window and the stage's final iteration. Each tick finds the stage's segments (``sacct``: every
job of that name that has started, in submission order, its log being the job's own ``StdOut``), runs
``scripts/telemetry/run_watch.py`` on them inside the container, appends the tick and the watch's output to the
record, prints one line, and acts:

- **stop** (the watch exits 1 with its summary line: a stop condition or a failed loss gate): cancel every live job
  of the stage by job ID (the running segment and its queued successors), then exit 1;
- **hold** (the watch exits 2, or the tick could not be evaluated, e.g. the watch could not run or ``sacct`` could not
  list the jobs, while the stage stands inside the hold window, i.e. after ``from_iteration`` and before ``until_iteration``, the first save, and a watched
  loss gate is still undecided): cancel the same way, so no save is written past an unevaluated gate, then exit 3.
  The iteration is the watch's own, or the last one a tick read when this tick's watch could not run; a gate is
  undecided until a watch has named it passed, so before any watch has run every gate is;
- **alert** (any other exit 2 or failure to evaluate): the record and the printed line say so, and the guard keeps
  going; a tick that could not be evaluated is never read as a stop;
- the guard exits 0 once the watch has checked the final iteration and no job of the stage is live.

A stop or hold whose cancellation fails exits 4, and a failure of the guard itself (its record cannot be written,
say) exits 5, so neither reads as a stop.

A gate the watch has passed on a log is passed to later ticks as ``--decided GATE=LOG``, so the watch does not
evaluate it again while that log still covers its range, and a transient failure to read a reference (W&B, say)
cannot unsettle it. The record starts with the guard config, the watch spec and its sha256, and the code revision,
and every tick names the watch spec's sha256, so each verdict can be traced to the spec it was judged by.

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

from scripts.telemetry.code_revision import code_revision  # noqa: E402


# The watch's own summary line; its presence is what tells a watch that ran from one that could not.
_SUMMARY_RE = re.compile(r"^checked through iteration (\S+): ", re.M)
# A gate the watch passed, and the log it passed on (the line's last field).
_PASSED_RE = re.compile(r"^GATE (\S+): PASS(?: \(.*\))? on (.+)$", re.M)
_UNDECIDED_RE = re.compile(r"^undecided gates: (.*)$", re.M)
# A container or activation failure must not exit 1, which reads as a stop.
_ACTIVATION_FAILED = 3
LIVE_STATES = ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "RESIZING", "SUSPENDED")


class GuardConfig(NamedTuple):
    """A stage's guard config (see the module docstring); the hold window is None when the config has none."""

    watch: Path
    job_name: str
    since: str
    interval_seconds: int
    record: Path
    hold_from: Optional[int]
    hold_until: Optional[int]
    final_iteration: Optional[int]


class Job(NamedTuple):
    """One job of the stage: its state, whether it has started, and its log."""

    job_id: int
    state: str
    started: bool
    log: Path


class WatchResult(NamedTuple):
    """One run of the watch: its exit status (None when it was not run), the iteration it checked through, whether
    it ran, its output, the gates it passed with the log each passed on, and the gates it names undecided (None when
    it names none, which leaves them unknown)."""

    status: Optional[int]
    checked_iteration: Optional[int]
    ran: bool
    output: str
    passed: Dict[str, str]
    undecided: Optional[Tuple[str, ...]]


class GuardState(NamedTuple):
    """What the guard carries from tick to tick: the last iteration a watch checked, the gates passed and the log
    each passed on, and the gates still undecided (None until a watch has run)."""

    last_iteration: Optional[int]
    decided: Dict[str, str]
    undecided: Optional[Tuple[str, ...]]


START = GuardState(last_iteration=None, decided={}, undecided=None)


def load_guard_config(path: Path) -> GuardConfig:
    """Read a guard config; the watch spec and record paths resolve against the repository root.

    Raises ValueError on an unknown key or a hold window without both ends; KeyError on a missing required key.
    """
    raw = yaml.safe_load(Path(path).read_text())
    keys = {"watch", "job_name", "since", "interval_seconds", "record", "hold", "final_iteration"}
    unknown = sorted(set(raw) - keys)
    if unknown:
        raise ValueError("{}: unknown keys {}".format(path, unknown))
    hold = raw.get("hold") or {}
    if hold and set(hold) != {"from_iteration", "until_iteration"}:
        raise ValueError("{}: hold needs exactly from_iteration and until_iteration".format(path))
    return GuardConfig(
        watch=REPO_ROOT / raw["watch"],
        job_name=str(raw["job_name"]),
        since=str(raw["since"]),
        interval_seconds=int(raw["interval_seconds"]),
        record=REPO_ROOT / raw["record"],
        hold_from=int(hold["from_iteration"]) if hold else None,
        hold_until=int(hold["until_iteration"]) if hold else None,
        final_iteration=int(raw["final_iteration"]) if "final_iteration" in raw else None,
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
        jobs.append(Job(int(job_id), state, start not in ("Unknown", "None", ""), Path(log)))
    return sorted(jobs, key=lambda job: job.job_id)


def parse_watch(status: int, output: str) -> WatchResult:
    """The watch's outcome from its exit status and output. A status of 0, 1 or 2 counts only beside the watch's
    summary line: without it the watch did not run (the container or the activation failed) and ``ran`` is False."""
    match = _SUMMARY_RE.search(output)
    ran = match is not None and status in (0, 1, 2)
    checked = None
    if match is not None and match.group(1) != "None":
        checked = int(match.group(1))
    undecided_match = _UNDECIDED_RE.search(output)
    undecided = None
    if undecided_match is not None:
        listed = undecided_match.group(1)
        undecided = () if listed == "none" else tuple(name.strip() for name in listed.split(","))
    passed = dict(_PASSED_RE.findall(output)) if ran else {}
    return WatchResult(status, checked, ran, output, passed, undecided)


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
    return GuardState(iteration, decided, result.undecided)


def decide(result: WatchResult, state: GuardState, config: GuardConfig) -> str:
    """``continue``, ``alert``, ``stop`` or ``hold`` for one tick's watch result, ``state`` being the guard's state
    after it (see ``advance``)."""
    if result.ran and result.status == 0:
        return "continue"
    if result.ran and result.status == 1:
        return "stop"
    iteration = state.last_iteration
    in_window = (
        config.hold_from is not None and iteration is not None and config.hold_from <= iteration < config.hold_until
    )
    gate_open = state.undecided is None or bool(state.undecided)
    return "hold" if in_window and gate_open else "alert"


Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess"]


def run_command(command: Sequence[str]) -> "subprocess.CompletedProcess":
    """Run a command, its stderr merged into its stdout (Python 3.6 has no ``capture_output``)."""
    return subprocess.run(
        list(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, check=False
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


class CancelFailed(RuntimeError):
    """A stop or hold could not cancel the stage's live jobs."""


def cancel_live_jobs(config: GuardConfig, run: Runner) -> List[int]:
    """Cancel, by job ID, every live job of the stage; return the IDs. Raises CancelFailed when the jobs cannot be
    listed or scancel fails."""
    try:
        live = [job.job_id for job in stage_jobs(config, run) if job.state in LIVE_STATES]
    except (RuntimeError, ValueError) as error:
        raise CancelFailed("could not list the stage's jobs: {}".format(error))
    if live:
        result = run(["scancel"] + [str(job_id) for job_id in live])
        if result.returncode != 0:
            raise CancelFailed("scancel {} failed: {}".format(live, result.stdout.strip()))
    return live


def sha256_of(path: Path) -> str:
    """The hex sha256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _record(config: GuardConfig, line: str, detail: str = "") -> None:
    config.record.parent.mkdir(parents=True, exist_ok=True)
    with open(str(config.record), "a") as handle:
        handle.write("{} {}\n".format(_stamp(), line))
        if detail:
            handle.write(detail if detail.endswith("\n") else detail + "\n")
    print("{} {}".format(_stamp(), line), flush=True)


def tick(config: GuardConfig, state: GuardState, run: Runner) -> Tuple[str, GuardState, bool]:
    """One evaluation: (action, the guard's state after it, whether the stage is done)."""
    jobs, spec = [], "unread"
    try:
        spec = sha256_of(config.watch)[:12]
        jobs = stage_jobs(config, run)
        started = [job for job in jobs if job.started]
        if not started:
            _record(config, "no segment of {} has started".format(config.job_name))
            return "continue", state, False
        result = run_watch(config, [job.log for job in started], state.decided, run)
    except Exception as error:  # noqa: BLE001 - judged by the hold window like a watch that could not run
        result = could_not_evaluate("{}: {}".format(type(error).__name__, error))
    state = advance(state, result)
    action = decide(result, state, config)
    segments = ", ".join("{}:{}".format(job.job_id, job.state) for job in jobs if job.started)
    if result.ran:
        summary = "watch exit {} through iteration {}".format(result.status, result.checked_iteration)
    elif result.status is None:
        summary = "could not evaluate: {}".format(result.output)
    else:
        summary = "watch could not run (exit {})".format(result.status)
    undecided = "unknown" if state.undecided is None else ", ".join(state.undecided) or "none"
    line = "{}: {}; undecided gates {} [{}] spec {}".format(action.upper(), summary, undecided, segments, spec)
    _record(config, line, result.output if result.status is not None else "")
    if action in ("stop", "hold"):
        cancelled = cancel_live_jobs(config, run)
        _record(config, "{}: cancelled {}".format(action.upper(), cancelled or "nothing (no live job)"))
    live = [job for job in jobs if job.state in LIVE_STATES]
    done = (
        config.final_iteration is not None
        and result.ran
        and result.status == 0
        and state.last_iteration is not None
        and state.last_iteration >= config.final_iteration
        and not live
    )
    return action, state, done


EXIT_STATUS = {"stop": 1, "hold": 3}
# A stop or hold whose cancellation failed: the stage may still be running, so a person must act now.
CANCEL_FAILED = 4
# The guard itself failed; the stage is no longer guarded.
GUARD_FAILED = 5


def guard(config_path: Path, once: bool) -> int:
    """Guard the stage until a stop, a hold or its end (one tick when ``once``), and return its exit status.

    Raises CancelFailed when a stop or hold could not cancel the stage's jobs."""
    config = load_guard_config(config_path)
    _record(
        config,
        "START guard {} (sha256 {}); watch spec {} (sha256 {}); code {}".format(
            config_path, sha256_of(config_path), config.watch, sha256_of(config.watch), code_revision(str(REPO_ROOT))
        ),
    )
    state = START
    while True:
        try:
            action, state, done = tick(config, state, run_command)
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
    try:
        return guard(args.config, args.once)
    except CancelFailed:
        return CANCEL_FAILED
    except Exception as error:  # noqa: BLE001 - the guard's own failure has its own status, never a stop's
        print("GUARD FAILED: {}: {}; the stage is unguarded".format(type(error).__name__, error), file=sys.stderr)
        return GUARD_FAILED


if __name__ == "__main__":
    sys.exit(main())

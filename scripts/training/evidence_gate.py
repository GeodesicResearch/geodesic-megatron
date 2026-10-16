#!/usr/bin/env python3
# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Check, from the evidence files a launch waits on, that every condition of the launch holds.

A job queued ahead of a training stage (``--dependency=afterok`` on this tool's job) starts only when the tool exits
0, and Slurm cancels it otherwise. The spec (YAML) lists named requirements under ``requirements:``, each of one kind:

- ``fields``: a JSON or YAML ``file`` (by its suffix) whose values at dotted paths are ``equal`` to the values given,
  and/or ``not_equal`` to them (``gate_outcome.fields_outcome``). A path's parts are mapping keys, or decimal indices
  into a list (``corpora.0.docs``). Values compare by the gates' one rule (``gate_outcome.identical``): equal types and
  values, a mapping or a list as a whole (``3126`` is not ``"3126"``).
- ``same``: two or more ``sources`` that must all hold the identical value. A source is ``{file: <json|yaml>, field:
  <dotted path>}``, ``{text: <file>}`` (a file of exactly one line, which is the value without its newline; an empty
  file or one of several lines is a FAIL), optionally with ``as: integer`` (the line must be a decimal integer, compared
  as one), or ``{value: <literal>}``.
- ``release``: a revision released by ``scripts/hub/publish_models.py``: the newest line for ``label``
  (``<repo>@<revision>``) in ``verified_log`` and the newest line for it in ``check_log``, whose lines open with
  ``check_tag`` (the export check's tag in the publisher's manifest), must both exist, the check line's verdict must be
  PASS, and both must name the same commit. Either line simply absent is NOT EVALUATED.
- ``descends``: ``commit`` (a full sha) is an ancestor of, or is, the commit ``copy`` holds (a frozen copy's
  ``REVISION``, a checkout's HEAD: ``code_identity.code_commit``), in the git history at ``history``.
- ``command``: ``argv`` exits 0 within ``timeout_s`` seconds. It runs in a new, empty directory made under
  ``scratch_parent`` and removed afterwards (a test suite that refuses a working directory holding earlier outputs runs
  there), with the gate's own environment; the last lines of its output are recorded.

A relative path in the spec resolves against the root of the repository this tool runs from (the frozen copy a job
was submitted from), never against the working directory; an absolute path is taken as it is. A spec can so name a
file of the copy it runs from (``REVISION``, a config) without knowing where that copy is.

Every requirement is evaluated, in the spec's order, whatever the others' outcomes. The outcomes and the exit status
are the repository's gates' (``scripts/telemetry/gate_outcome.py``): a requirement whose evidence is wrong is FAIL, one
whose evidence is absent or unreadable is NOT EVALUATED, and the process exits 1 on any FAIL, otherwise 2 on any NOT
EVALUATED, and 0 only when every requirement passes. A spec that is not one, and a report file that exists, are
refused (exit 2) before anything is evaluated.

With ``--wait-seconds S --poll-seconds P`` the whole spec is evaluated again every P seconds while no requirement is
FAIL and one is NOT EVALUATED: the gate stops at the first FAIL (exit 1), when every requirement passes (exit 0), or
once S seconds have passed (exit 2). Every evaluation runs every requirement, a ``command`` included. Without them the
spec is evaluated once.

The process prints one line per requirement (and a ``command``'s output tail when it does not pass), and
``--report-out`` receives, once at the end, the spec's path and sha256, the code's revision, the times, the number of
evaluations, and every requirement's outcome, reason and detail.

    python scripts/training/evidence_gate.py --spec <spec.yaml> --report-out <json> \\
        [--wait-seconds <S> --poll-seconds <P>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.dotted_paths import value_at  # noqa: E402
from scripts.hub import publish_models  # noqa: E402
from scripts.mapping_keys import require_keys  # noqa: E402
from scripts.mapping_values import require_int, require_string  # noqa: E402
from scripts.telemetry.code_revision import code_revision  # noqa: E402
from scripts.telemetry.gate_outcome import (  # noqa: E402
    FAIL,
    NOT_EVALUATED,
    PASS,
    exit_status,
    fields_outcome,
    identical,
)
from scripts.training.code_identity import FULL_SHA, CodeIdentityError, code_commit, has_ancestor  # noqa: E402


FIELDS, SAME, RELEASE, DESCENDS, COMMAND = "fields", "same", "release", "descends", "command"
DOCUMENT_READERS: dict[str, Callable[[str], Any]] = {
    ".json": json.loads,
    ".yaml": yaml.safe_load,
    ".yml": yaml.safe_load,
}
INTEGER = "integer"
INTEGER_LINE = re.compile(r"-?[0-9]+")
OUTPUT_TAIL_LINES = 40


class EvidenceSpecError(ValueError):
    """The spec is not one this tool can evaluate."""


class MissingEvidence(Exception):
    """A piece of evidence is absent or cannot be read: the requirement is NOT EVALUATED."""


class WrongEvidence(Exception):
    """A piece of evidence is present and is not what a requirement accepts: the requirement FAILs."""


@dataclass(frozen=True)
class Finding:
    """One requirement's outcome, the one-line reason for it, and what else the report records of it."""

    outcome: str
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)


def judge(failures: list[str], gaps: list[str], passed: str) -> tuple[str, str]:
    """FAIL when anything is wrong, otherwise NOT EVALUATED when anything is missing, otherwise PASS; with the reason."""
    if failures:
        return FAIL, "; ".join(failures + gaps)
    if gaps:
        return NOT_EVALUATED, "; ".join(gaps)
    return PASS, passed


def read_document(path: Path) -> Any:
    """A JSON or YAML file, parsed by its suffix. Raises MissingEvidence when it is absent or cannot be parsed."""
    if not path.is_file():
        raise MissingEvidence(f"{path} does not exist")
    try:
        return DOCUMENT_READERS[path.suffix](path.read_text())
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise MissingEvidence(f"{path} cannot be read: {type(error).__name__}: {error}") from error


def read_single_line(path: Path) -> str:
    """The one line ``path`` holds, without its newline. Raises MissingEvidence when it is absent or unreadable, and
    WrongEvidence when it is empty or holds more than one line."""
    if not path.is_file():
        raise MissingEvidence(f"{path} does not exist")
    try:
        text = path.read_text()
    except (OSError, UnicodeDecodeError) as error:
        raise MissingEvidence(f"{path} cannot be read: {type(error).__name__}: {error}") from error
    lines = text[:-1].split("\n") if text.endswith("\n") else text.split("\n")
    if len(lines) != 1 or not lines[0]:
        raise WrongEvidence(f"{path} must hold exactly one non-empty line, not {text[:200]!r}")
    return lines[0]


@dataclass(frozen=True)
class DocumentField:
    """A ``same`` source: the value at a dotted path of a JSON or YAML file."""

    file: Path
    dotted_path: str

    def label(self) -> str:
        return f"{self.file}:{self.dotted_path}"

    def read(self) -> Any:
        try:
            return value_at(read_document(self.file), self.dotted_path)
        except KeyError as error:
            raise MissingEvidence(f"{self.file} holds no {self.dotted_path}") from error


@dataclass(frozen=True)
class TextLine:
    """A ``same`` source: the one line of a text file, as a string or, when ``integer``, as a decimal integer."""

    file: Path
    integer: bool

    def label(self) -> str:
        return str(self.file)

    def read(self) -> Any:
        line = read_single_line(self.file)
        if not self.integer:
            return line
        if not INTEGER_LINE.fullmatch(line):
            raise WrongEvidence(f"{self.file} holds {line!r}, not a decimal integer")
        return int(line)


@dataclass(frozen=True)
class Literal:
    """A ``same`` source: a value stated in the spec."""

    value: Any

    def label(self) -> str:
        return "the spec's value"

    def read(self) -> Any:
        return self.value


Source = DocumentField | TextLine | Literal


@dataclass(frozen=True)
class FieldsRequirement:
    """Values at dotted paths of one file that must equal, or must not equal, the values the spec gives."""

    kind: ClassVar[str] = FIELDS
    name: str
    file: Path
    equal: dict[str, Any]
    not_equal: dict[str, Any]

    def evaluate(self) -> Finding:
        try:
            document = read_document(self.file)
        except MissingEvidence as gap:
            return Finding(NOT_EVALUATED, str(gap))
        outcome, problems = fields_outcome(document, self.equal, self.not_equal, NOT_EVALUATED)
        reason = "; ".join(problems) or f"{len(self.equal) + len(self.not_equal)} field(s) as required"
        return Finding(outcome, f"{self.file}: {reason}")


@dataclass(frozen=True)
class SameRequirement:
    """Sources that must all be present and hold the identical value."""

    kind: ClassVar[str] = SAME
    name: str
    sources: tuple[Source, ...]

    def evaluate(self) -> Finding:
        read, failures, gaps = [], [], []
        for source in self.sources:
            try:
                read.append((source.label(), source.read()))
            except MissingEvidence as gap:
                gaps.append(str(gap))
            except WrongEvidence as wrong:
                failures.append(str(wrong))
        if any(not identical(value, read[0][1]) for _, value in read[1:]):
            failures.append("the values differ: " + ", ".join(f"{label} {value!r}" for label, value in read))
        passed = f"{len(read)} sources hold {read[0][1]!r}" if read else ""
        outcome, reason = judge(failures, gaps, passed)
        return Finding(outcome, reason, {"values": [{"source": label, "value": repr(value)} for label, value in read]})


@dataclass(frozen=True)
class ReleaseRequirement:
    """A revision the publisher verified and whose export check passed, at one commit."""

    kind: ClassVar[str] = RELEASE
    name: str
    verified_log: Path
    check_log: Path
    check_tag: str
    label: str

    def evaluate(self) -> Finding:
        failures, gaps, commits = [], [], {}
        newest_check = None
        for log, tag in ((self.verified_log, publish_models.VERIFIED_TAG), (self.check_log, self.check_tag)):
            line = publish_models.last_release_line(log, tag, self.label)
            if line is None:
                gaps.append(f"{log} holds no {tag} line for {self.label}")
                continue
            commit, detail = publish_models.release_line_fields(line)
            if not FULL_SHA.fullmatch(commit):
                failures.append(f"{log}'s newest {tag} line for {self.label} names no commit: {line!r}")
                continue
            commits[tag] = commit
            if tag == self.check_tag:
                newest_check = line
                if detail != [PASS]:
                    failures.append(f"its newest {tag} line judges commit {commits[tag]} {' '.join(detail)}")
        if len(commits) == 2 and commits[publish_models.VERIFIED_TAG] != commits[self.check_tag]:
            failures.append(
                f"its newest {publish_models.VERIFIED_TAG} line names commit {commits[publish_models.VERIFIED_TAG]}, "
                f"its newest {self.check_tag} line {commits[self.check_tag]}"
            )
        passed = f"{self.label} verified and {self.check_tag} PASS at commit {commits.get(self.check_tag)}"
        outcome, reason = judge(failures, gaps, passed)
        return Finding(outcome, reason, {"commits": commits, "newest_check_line": newest_check})


@dataclass(frozen=True)
class DescendsRequirement:
    """A commit the code of a copy must contain."""

    kind: ClassVar[str] = DESCENDS
    name: str
    commit: str
    copy: Path
    history: Path

    def evaluate(self) -> Finding:
        try:
            held = code_commit(self.copy)
            descends = has_ancestor(str(self.history), self.commit, held)
        except CodeIdentityError as gap:
            return Finding(NOT_EVALUATED, str(gap))
        named = f"{self.copy} holds commit {held}, which"
        if not descends:
            return Finding(FAIL, f"{named} does not descend from {self.commit}", {"commit": held})
        return Finding(PASS, f"{named} descends from {self.commit}", {"commit": held})


def output_tail(output: str | bytes | None) -> list[str]:
    """The last ``OUTPUT_TAIL_LINES`` lines of a command's output (bytes when a timeout cut it short)."""
    if isinstance(output, bytes):
        output = output.decode("utf-8", "replace")
    return (output or "").splitlines()[-OUTPUT_TAIL_LINES:]


@dataclass(frozen=True)
class CommandRequirement:
    """A command that must exit 0, run in a new, empty directory."""

    kind: ClassVar[str] = COMMAND
    name: str
    argv: tuple[str, ...]
    scratch_parent: Path
    timeout_s: int

    def evaluate(self) -> Finding:
        if not self.scratch_parent.is_dir():
            return Finding(NOT_EVALUATED, f"{self.scratch_parent} is not a directory to run the command under")
        with tempfile.TemporaryDirectory(prefix="evidence_gate_", dir=self.scratch_parent) as directory:
            try:
                result = subprocess.run(
                    list(self.argv),
                    cwd=directory,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=self.timeout_s,
                )
            except subprocess.TimeoutExpired as expired:
                return Finding(
                    NOT_EVALUATED,
                    f"it did not finish within {self.timeout_s} s",
                    {"directory": directory, "output_tail": output_tail(expired.output)},
                )
            except OSError as error:
                return Finding(NOT_EVALUATED, f"it could not be started: {type(error).__name__}: {error}")
        detail = {"directory": directory, "exit_status": result.returncode, "output_tail": output_tail(result.stdout)}
        if result.returncode != 0:
            return Finding(FAIL, f"it exited {result.returncode}", detail)
        return Finding(PASS, "it exited 0", detail)


Requirement = FieldsRequirement | SameRequirement | ReleaseRequirement | DescendsRequirement | CommandRequirement


def _path(value: Any, where: str) -> Path:
    """A path of the spec: an absolute one as it is, a relative one under the repository root this tool runs from."""
    path = Path(require_string(value, where, error=EvidenceSpecError))
    return path if path.is_absolute() else REPO_ROOT / path


def _document_path(value: Any, where: str) -> Path:
    path = _path(value, where)
    if path.suffix not in DOCUMENT_READERS:
        raise EvidenceSpecError(f"{where} must name a {'/'.join(sorted(DOCUMENT_READERS))} file, not {path}")
    return path


def _dotted_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value or not all(isinstance(key, str) and key for key in value):
        raise EvidenceSpecError(f"{where} must map at least one dotted path to a value")
    return value


def parse_fields(item: dict[str, Any], where: str) -> FieldsRequirement:
    """A ``fields`` requirement: its file, and the values it must hold (``equal``) and must not (``not_equal``)."""
    require_keys(item, where, {"name", "kind", "file"}, {"equal", "not_equal"}, error=EvidenceSpecError)
    if "equal" not in item and "not_equal" not in item:
        raise EvidenceSpecError(f"{where} must state equal, not_equal or both")
    return FieldsRequirement(
        name=item["name"],
        file=_document_path(item["file"], f"{where}.file"),
        equal=_dotted_mapping(item["equal"], f"{where}.equal") if "equal" in item else {},
        not_equal=_dotted_mapping(item["not_equal"], f"{where}.not_equal") if "not_equal" in item else {},
    )


def parse_source(item: Any, where: str) -> Source:
    """One ``same`` source: ``{file, field}``, ``{text[, as: integer]}`` or ``{value}``."""
    if isinstance(item, dict) and "file" in item:
        require_keys(item, where, {"file", "field"}, error=EvidenceSpecError)
        return DocumentField(
            file=_document_path(item["file"], f"{where}.file"),
            dotted_path=require_string(item["field"], f"{where}.field", error=EvidenceSpecError),
        )
    if isinstance(item, dict) and "text" in item:
        require_keys(item, where, {"text"}, {"as"}, error=EvidenceSpecError)
        if "as" in item and item["as"] != INTEGER:
            raise EvidenceSpecError(f"{where}.as must be {INTEGER!r}, not {item['as']!r}")
        return TextLine(file=_path(item["text"], f"{where}.text"), integer="as" in item)
    if isinstance(item, dict) and "value" in item:
        require_keys(item, where, {"value"}, error=EvidenceSpecError)
        return Literal(value=item["value"])
    raise EvidenceSpecError(f"{where} must be one of {{file, field}}, {{text[, as]}} or {{value}}, not {item!r}")


def parse_same(item: dict[str, Any], where: str) -> SameRequirement:
    """A ``same`` requirement: its two or more sources."""
    require_keys(item, where, {"name", "kind", "sources"}, error=EvidenceSpecError)
    sources = item["sources"]
    if not isinstance(sources, list) or len(sources) < 2:
        raise EvidenceSpecError(f"{where}.sources must list at least two sources")
    return SameRequirement(
        name=item["name"],
        sources=tuple(parse_source(source, f"{where}.sources[{index}]") for index, source in enumerate(sources)),
    )


def parse_release(item: dict[str, Any], where: str) -> ReleaseRequirement:
    """A ``release`` requirement: the two logs, the check's tag, and the ``<repo>@<revision>`` label."""
    require_keys(
        item, where, {"name", "kind", "verified_log", "check_log", "check_tag", "label"}, error=EvidenceSpecError
    )
    label = require_string(item["label"], f"{where}.label", error=EvidenceSpecError)
    if label.count("@") != 1 or " " in label:
        raise EvidenceSpecError(f"{where}.label must be <repo>@<revision>, not {label!r}")
    check_tag = require_string(item["check_tag"], f"{where}.check_tag", error=EvidenceSpecError)
    if check_tag == publish_models.VERIFIED_TAG or " " in check_tag:
        raise EvidenceSpecError(f"{where}.check_tag must be the export check's one-word tag, not {check_tag!r}")
    return ReleaseRequirement(
        name=item["name"],
        verified_log=_path(item["verified_log"], f"{where}.verified_log"),
        check_log=_path(item["check_log"], f"{where}.check_log"),
        check_tag=check_tag,
        label=label,
    )


def parse_descends(item: dict[str, Any], where: str) -> DescendsRequirement:
    """A ``descends`` requirement: the full commit sha, the copy, and the history to look in."""
    require_keys(item, where, {"name", "kind", "commit", "copy", "history"}, error=EvidenceSpecError)
    commit = item["commit"]
    if not isinstance(commit, str) or not FULL_SHA.fullmatch(commit):
        raise EvidenceSpecError(f"{where}.commit must be a full 40-character commit sha, not {commit!r}")
    return DescendsRequirement(
        name=item["name"],
        commit=commit,
        copy=_path(item["copy"], f"{where}.copy"),
        history=_path(item["history"], f"{where}.history"),
    )


def parse_command(item: dict[str, Any], where: str) -> CommandRequirement:
    """A ``command`` requirement: its argv, the directory its scratch directory is made under, and its timeout."""
    require_keys(item, where, {"name", "kind", "argv", "scratch_parent", "timeout_s"}, error=EvidenceSpecError)
    argv = item["argv"]
    if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) and arg for arg in argv):
        raise EvidenceSpecError(f"{where}.argv must be a non-empty list of non-empty strings")
    return CommandRequirement(
        name=item["name"],
        argv=tuple(argv),
        scratch_parent=_path(item["scratch_parent"], f"{where}.scratch_parent"),
        timeout_s=require_int(item["timeout_s"], f"{where}.timeout_s", 1, error=EvidenceSpecError),
    )


PARSERS: dict[str, Callable[[dict[str, Any], str], Requirement]] = {
    FIELDS: parse_fields,
    SAME: parse_same,
    RELEASE: parse_release,
    DESCENDS: parse_descends,
    COMMAND: parse_command,
}


def load_spec(path: Path) -> list[Requirement]:
    """The spec's requirements, in order. Raises EvidenceSpecError on a spec that is not one."""
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise EvidenceSpecError(f"{path} cannot be read: {error}") from error
    items = require_keys(raw, str(path), {"requirements"}, error=EvidenceSpecError)["requirements"]
    if not isinstance(items, list) or not items:
        raise EvidenceSpecError(f"{path}: requirements must list at least one requirement")
    requirements, names = [], set()
    for index, item in enumerate(items):
        where = f"{path}: requirements[{index}]"
        if not isinstance(item, dict) or item.get("kind") not in PARSERS:
            raise EvidenceSpecError(f"{where} must be a mapping whose kind is one of {sorted(PARSERS)}")
        name = require_string(item.get("name"), f"{where}.name", error=EvidenceSpecError)
        if name in names:
            raise EvidenceSpecError(f"{where}: the name {name!r} is used twice")
        names.add(name)
        requirements.append(PARSERS[item["kind"]](item, where))
    return requirements


def evaluate_all(requirements: list[Requirement]) -> list[Finding]:
    """Every requirement's finding, in order; none is skipped whatever the others' outcomes."""
    return [requirement.evaluate() for requirement in requirements]


def evaluate_until_decided(
    requirements: list[Requirement],
    wait_seconds: float,
    poll_seconds: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> tuple[list[Finding], int]:
    """The findings of the last evaluation and the number of evaluations. The spec is evaluated again every
    ``poll_seconds`` while no finding is FAIL and one is NOT EVALUATED, until ``wait_seconds`` have passed (0: once)."""
    deadline = clock() + wait_seconds
    evaluations = 0
    while True:
        findings = evaluate_all(requirements)
        evaluations += 1
        outcomes = [finding.outcome for finding in findings]
        remaining = deadline - clock()
        if FAIL in outcomes or NOT_EVALUATED not in outcomes or remaining <= 0:
            return findings, evaluations
        print(
            f"evaluation {evaluations}: {outcomes.count(NOT_EVALUATED)} of {len(outcomes)} not evaluated; "
            f"again in {min(poll_seconds, remaining):.0f} s ({remaining:.0f} s left)",
            flush=True,
        )
        sleep(min(poll_seconds, remaining))


def describe(requirements: list[Requirement], findings: list[Finding]) -> list[str]:
    """The lines the gate prints: one per requirement, and a command's output tail when it did not pass."""
    lines = []
    for requirement, finding in zip(requirements, findings):
        lines.append(f"{finding.outcome:<13} {requirement.name} ({requirement.kind}): {finding.reason}")
        if finding.outcome != PASS and finding.detail.get("output_tail"):
            lines.extend(f"    | {line}" for line in finding.detail["output_tail"])
    return lines


def utc_now() -> str:
    """The current time as an ISO-8601 UTC stamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv: list[str] | None = None) -> int:
    """Evaluate the spec (until decided, in wait mode), print each requirement's line, write the report, and return
    the gates' exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--spec", type=Path, required=True, help="the requirements (see the module docstring)")
    parser.add_argument("--report-out", type=Path, required=True, help="the JSON report; an existing file is refused")
    parser.add_argument("--wait-seconds", type=float, help="wait mode: evaluate until decided or this many seconds")
    parser.add_argument("--poll-seconds", type=float, help="wait mode: the seconds between evaluations")
    args = parser.parse_args(argv)
    if (args.wait_seconds is None) != (args.poll_seconds is None):
        parser.error("--wait-seconds and --poll-seconds are given together or not at all")
    if args.wait_seconds is not None and (args.wait_seconds <= 0 or args.poll_seconds <= 0):
        parser.error("--wait-seconds and --poll-seconds must be positive")
    if args.report_out.exists():
        parser.error(f"{args.report_out} exists")
    try:
        requirements = load_spec(args.spec)
    except EvidenceSpecError as error:
        parser.error(str(error))

    started, start = utc_now(), time.monotonic()
    findings, evaluations = evaluate_until_decided(
        requirements, args.wait_seconds or 0, args.poll_seconds or 0, time.monotonic, time.sleep
    )
    status = exit_status(finding.outcome for finding in findings)
    print("\n".join(describe(requirements, findings)))
    print(f"verdict: exit {status} after {evaluations} evaluation(s)")
    report = {
        "spec": str(args.spec),
        "spec_sha256": hashlib.sha256(args.spec.read_bytes()).hexdigest(),
        "code_revision": code_revision(str(REPO_ROOT)),
        "started": started,
        "finished": utc_now(),
        "elapsed_s": round(time.monotonic() - start, 3),
        "evaluations": evaluations,
        "wait_seconds": args.wait_seconds,
        "poll_seconds": args.poll_seconds,
        "requirements": [
            {
                "name": requirement.name,
                "kind": requirement.kind,
                "outcome": finding.outcome,
                "reason": finding.reason,
                "detail": finding.detail,
            }
            for requirement, finding in zip(requirements, findings)
        ],
        "exit_status": status,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    with args.report_out.open("x") as out:
        json.dump(report, out, indent=2)
    return status


if __name__ == "__main__":
    sys.exit(main())

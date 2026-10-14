# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""scripts/training/evidence_gate.py: a launch gate that passes only when every requirement's evidence is present and
right. Each kind is run on real files: release logs written by the publisher's own line writer, a real git history
with a frozen copy of one of its commits, and real commands. The campaign's two specs are checked to parse."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from scripts.dotted_paths import value_at
from scripts.hub import publish_models
from scripts.telemetry.gate_outcome import FAIL, NOT_EVALUATED, PASS
from scripts.training import evidence_gate
from scripts.training.evidence_gate import (
    CommandRequirement,
    DescendsRequirement,
    EvidenceSpecError,
    FieldsRequirement,
    ReleaseRequirement,
    evaluate_until_decided,
    load_spec,
    main,
)

from tests.unit_tests.corpora_fixtures import corpora_table


REPO_ROOT = Path(__file__).resolve().parents[2]
ARM = REPO_ROOT / "configs/metagaming_filtering/30b_clueless_norm"
GATE_SPEC = ARM / "g_sft_gate.yaml"
WAIT_SPEC = ARM / "g_sft_mt_release_wait.yaml"
LABEL = "org/arm-base@midtraining_iter_3126"
GIT_IDENTITY = ["-c", "user.name=test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false"]


def write_spec(path: Path, requirements: list[dict]) -> Path:
    path.write_text(yaml.safe_dump({"requirements": requirements}))
    return path


def evaluate(path: Path, requirements: list[dict]):
    """The findings of a spec of ``requirements``, written at ``path``."""
    return evidence_gate.evaluate_all(load_spec(write_spec(path, requirements)))


def fields(file: Path, **checks) -> dict:
    return {"name": "fields", "kind": "fields", "file": str(file), **checks}


def same(*sources: dict) -> dict:
    return {"name": "same", "kind": "same", "sources": list(sources)}


class TestFields:
    def test_equal_and_not_equal_pass_on_the_right_values(self, tmp_path):
        report = tmp_path / "report.json"
        report.write_text(json.dumps({"verdict": "PASS", "corpora": [{"docs": 7}], "status": "calibrated"}))
        (finding,) = evaluate(
            tmp_path / "spec.yaml",
            [fields(report, equal={"verdict": "PASS", "corpora.0.docs": 7}, not_equal={"status": "PENDING"})],
        )
        assert finding.outcome == PASS, finding.reason

    def test_a_wrong_value_and_a_value_still_refused_fail(self, tmp_path):
        record = tmp_path / "spec_under_test.yaml"
        record.write_text(yaml.safe_dump({"verdict": "FAIL", "calibration": {"status": "PENDING"}}))
        (wrong,) = evaluate(tmp_path / "a.yaml", [fields(record, equal={"verdict": "PASS"})])
        (pending,) = evaluate(tmp_path / "b.yaml", [fields(record, not_equal={"calibration.status": "PENDING"})])
        assert wrong.outcome == FAIL and "'FAIL', not 'PASS'" in wrong.reason
        assert pending.outcome == FAIL and "still 'PENDING'" in pending.reason

    def test_values_compare_by_type_and_a_mapping_as_a_whole(self, tmp_path):
        report = tmp_path / "report.json"
        report.write_text(json.dumps({"iteration": 3126, "outcomes": {"counters": "PASS", "extra": "PASS"}}))
        (as_string,) = evaluate(tmp_path / "a.yaml", [fields(report, equal={"iteration": "3126"})])
        (one_key_more,) = evaluate(tmp_path / "b.yaml", [fields(report, equal={"outcomes": {"counters": "PASS"}})])
        assert as_string.outcome == FAIL
        assert one_key_more.outcome == FAIL

    def test_a_missing_file_or_key_is_not_evaluated(self, tmp_path):
        report = tmp_path / "report.json"
        (missing_file,) = evaluate(tmp_path / "a.yaml", [fields(report, equal={"verdict": "PASS"})])
        report.write_text(json.dumps({"other": 1}))
        (missing_key,) = evaluate(tmp_path / "b.yaml", [fields(report, equal={"verdict": "PASS"})])
        report.write_text('{"verdict": ')
        (unreadable,) = evaluate(tmp_path / "c.yaml", [fields(report, equal={"verdict": "PASS"})])
        assert [f.outcome for f in (missing_file, missing_key, unreadable)] == [NOT_EVALUATED] * 3
        assert "does not exist" in missing_file.reason and "verdict is absent" in missing_key.reason


class TestSame:
    def test_sources_holding_one_value_pass(self, tmp_path):
        (tmp_path / "iteration.txt").write_text("3126")
        (tmp_path / "rows.txt").write_text("41\n")
        (tmp_path / "report.json").write_text(json.dumps({"corpora": [{"docs": 41}], "revision": "abc"}))
        (tmp_path / "revision.txt").write_text("abc\n")
        findings = evaluate(
            tmp_path / "spec.yaml",
            [
                same({"text": str(tmp_path / "iteration.txt")}, {"value": "3126"}) | {"name": "iteration"},
                same(
                    {"file": str(tmp_path / "report.json"), "field": "corpora.0.docs"},
                    {"text": str(tmp_path / "rows.txt"), "as": "integer"},
                )
                | {"name": "rows"},
                same(
                    {"text": str(tmp_path / "revision.txt")},
                    {"file": str(tmp_path / "report.json"), "field": "revision"},
                )
                | {"name": "revision"},
            ],
        )
        assert [f.outcome for f in findings] == [PASS] * 3, [f.reason for f in findings]

    def test_two_sources_of_one_kind_are_both_compared(self, tmp_path):
        (finding,) = evaluate(tmp_path / "spec.yaml", [same({"value": 1}, {"value": 2})])
        assert finding.outcome == FAIL and [v["value"] for v in finding.detail["values"]] == ["1", "2"]

    def test_different_values_fail(self, tmp_path):
        (tmp_path / "iteration.txt").write_text("1564\n")
        (finding,) = evaluate(
            tmp_path / "spec.yaml", [same({"text": str(tmp_path / "iteration.txt")}, {"value": "3126"})]
        )
        assert finding.outcome == FAIL and "'1564'" in finding.reason

    @pytest.mark.parametrize("content", ["", "\n", "3126\n\n", "3126\n3127\n"])
    def test_a_text_file_of_other_than_one_line_fails(self, tmp_path, content):
        (tmp_path / "iteration.txt").write_text(content)
        (finding,) = evaluate(
            tmp_path / "spec.yaml", [same({"text": str(tmp_path / "iteration.txt")}, {"value": "3126"})]
        )
        assert finding.outcome == FAIL and "exactly one non-empty line" in finding.reason

    def test_a_line_that_is_not_an_integer_fails_when_read_as_one(self, tmp_path):
        (tmp_path / "rows.txt").write_text("41 rows\n")
        (finding,) = evaluate(
            tmp_path / "spec.yaml", [same({"text": str(tmp_path / "rows.txt"), "as": "integer"}, {"value": 41})]
        )
        assert finding.outcome == FAIL and "not a decimal integer" in finding.reason

    def test_a_missing_source_is_not_evaluated_unless_the_present_ones_already_differ(self, tmp_path):
        (tmp_path / "a.txt").write_text("x\n")
        missing = {"text": str(tmp_path / "absent.txt")}
        (waiting,) = evaluate(tmp_path / "1.yaml", [same({"text": str(tmp_path / "a.txt")}, missing, {"value": "x"})])
        (wrong,) = evaluate(tmp_path / "2.yaml", [same({"text": str(tmp_path / "a.txt")}, missing, {"value": "y"})])
        assert waiting.outcome == NOT_EVALUATED and "does not exist" in waiting.reason
        assert wrong.outcome == FAIL


def release(tmp_path: Path) -> ReleaseRequirement:
    return ReleaseRequirement(
        name="release",
        verified_log=tmp_path / "verified.log",
        check_log=tmp_path / "c3.log",
        check_tag="c3",
        label=LABEL,
    )


class TestRelease:
    def test_a_verified_revision_whose_check_passed_at_its_commit_passes(self, tmp_path):
        requirement, commit = release(tmp_path), "a" * 40
        publish_models.record_release_line(requirement.verified_log, "verified", LABEL, commit, "6243 tensors")
        publish_models.record_release_line(
            requirement.check_log, "c3", "org/arm-base@midtraining_iter_1564", "b" * 40, "FAIL"
        )
        publish_models.record_release_line(requirement.check_log, "c3", LABEL, commit, "PASS")
        finding = requirement.evaluate()
        assert finding.outcome == PASS, finding.reason
        assert finding.detail["commits"] == {"verified": commit, "c3": commit}

    def test_an_absent_line_is_not_evaluated(self, tmp_path):
        requirement = release(tmp_path)
        assert requirement.evaluate().outcome == NOT_EVALUATED, "neither log exists"
        publish_models.record_release_line(requirement.verified_log, "verified", LABEL, "a" * 40, "6243 tensors")
        publish_models.record_release_line(requirement.check_log, "c3", f"{LABEL}0", "a" * 40, "PASS")
        finding = requirement.evaluate()
        assert finding.outcome == NOT_EVALUATED and "holds no c3 line" in finding.reason, "another revision's line"

    def test_a_failed_check_fails(self, tmp_path):
        requirement, commit = release(tmp_path), "a" * 40
        publish_models.record_release_line(requirement.verified_log, "verified", LABEL, commit, "6243 tensors")
        publish_models.record_release_line(requirement.check_log, "c3", LABEL, commit, "PASS")
        publish_models.record_release_line(requirement.check_log, "c3", LABEL, commit, "FAIL")
        finding = requirement.evaluate()
        assert finding.outcome == FAIL and "judges commit" in finding.reason, "the newest line decides"

    def test_a_line_that_names_no_commit_fails(self, tmp_path):
        requirement = release(tmp_path)
        publish_models.record_release_line(requirement.verified_log, "verified", LABEL, "a" * 40, "6243 tensors")
        requirement.check_log.write_text(f"c3 {LABEL}  PASS 2026-10-14T15:00:00Z\n")
        finding = requirement.evaluate()
        assert finding.outcome == FAIL and "names no commit" in finding.reason

    def test_a_check_of_another_commit_than_the_verified_one_fails(self, tmp_path):
        requirement = release(tmp_path)
        publish_models.record_release_line(requirement.check_log, "c3", LABEL, "a" * 40, "PASS")
        publish_models.record_release_line(requirement.verified_log, "verified", LABEL, "a" * 40, "6243 tensors")
        publish_models.record_release_line(requirement.verified_log, "verified", LABEL, "b" * 40, "6243 tensors")
        finding = requirement.evaluate()
        assert finding.outcome == FAIL and "b" * 40 in finding.reason


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *GIT_IDENTITY, "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def history(tmp_path) -> dict:
    """A repository of two commits, the second descending from the first, and a side commit from the root."""
    repo = tmp_path / "history"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    shas = []
    for name in ("root", "spec", "later"):
        (repo / f"{name}.txt").write_text(name)
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", name)
        shas.append(git(repo, "rev-parse", "HEAD"))
    git(repo, "checkout", "-q", "-b", "side", shas[0])
    (repo / "side.txt").write_text("side")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "side")
    return {"repo": repo, "root": shas[0], "spec": shas[1], "later": shas[2], "side": git(repo, "rev-parse", "HEAD")}


def frozen_copy(tmp_path: Path, revision: str) -> Path:
    """A frozen copy whose REVISION holds ``revision`` (the copy's other files do not matter here)."""
    copy = tmp_path / f"copy-{len(list(tmp_path.glob('copy-*')))}"
    copy.mkdir()
    (copy / "REVISION").write_text(revision)
    return copy


class TestDescends:
    def test_a_copy_at_a_later_commit_or_the_commit_itself_passes(self, tmp_path, history):
        for held in ("later", "spec"):
            copy = frozen_copy(tmp_path, f"{history[held]}\n")
            requirement = DescendsRequirement("descends", history["spec"], copy, history["repo"])
            assert requirement.evaluate().outcome == PASS, held

    def test_the_first_word_of_the_revision_names_the_commit(self, tmp_path, history):
        """A copy made with an uncommitted diff names it after the commit, and may add lines."""
        revision = (
            f"{history['later']} + uncommitted diff sha256 {'0' * 64}\nsubmodule 3rdparty/Megatron-LM at {'1' * 40}\n"
        )
        requirement = DescendsRequirement(
            "descends", history["spec"], frozen_copy(tmp_path, revision), history["repo"]
        )
        finding = requirement.evaluate()
        assert finding.outcome == PASS and finding.detail["commit"] == history["later"]

    def test_a_checkout_is_read_at_its_head(self, history):
        requirement = DescendsRequirement("descends", history["root"], history["repo"], history["repo"])
        finding = requirement.evaluate()
        assert finding.outcome == PASS and finding.detail["commit"] == history["side"], (
            "the side branch is checked out"
        )

    def test_a_copy_without_the_commit_fails(self, tmp_path, history):
        for held in ("root", "side"):
            copy = frozen_copy(tmp_path, f"{history[held]}\n")
            finding = DescendsRequirement("descends", history["spec"], copy, history["repo"]).evaluate()
            assert finding.outcome == FAIL and "does not descend" in finding.reason, held

    def test_a_copy_that_names_no_commit_or_an_unknown_one_is_not_evaluated(self, tmp_path, history):
        for revision in ("commit: 52e315c8 (main)\n", "\n", f"{'f' * 40}\n"):
            copy = frozen_copy(tmp_path, revision)
            finding = DescendsRequirement("descends", history["spec"], copy, history["repo"]).evaluate()
            assert finding.outcome == NOT_EVALUATED, revision
        absent = DescendsRequirement("descends", history["spec"], tmp_path / "not-a-copy", history["repo"])
        assert absent.evaluate().outcome == NOT_EVALUATED


def command(tmp_path: Path, code: str, timeout_s: int) -> CommandRequirement:
    return CommandRequirement("command", (sys.executable, "-c", code), tmp_path, timeout_s)


class TestCommand:
    def test_exit_zero_passes_in_a_new_empty_directory_under_the_scratch_parent(self, tmp_path):
        parent = os.path.realpath(tmp_path)
        code = f"import os; assert os.listdir('.') == []; assert os.path.realpath('.').startswith({parent!r}); print('ok')"
        finding = command(tmp_path, code, 60).evaluate()
        assert finding.outcome == PASS, finding.detail
        assert finding.detail["output_tail"] == ["ok"] and not Path(finding.detail["directory"]).exists()

    def test_a_nonzero_exit_fails_and_records_the_last_lines(self, tmp_path):
        finding = command(tmp_path, "import sys\nfor i in range(100): print(i)\nsys.exit(3)", 60).evaluate()
        assert finding.outcome == FAIL and finding.reason == "it exited 3"
        assert finding.detail["output_tail"] == [str(i) for i in range(60, 100)]

    def test_a_command_that_cannot_start_or_finish_is_not_evaluated(self, tmp_path):
        missing = CommandRequirement("command", (str(tmp_path / "no-such-program"),), tmp_path, 60)
        slow = command(tmp_path, "import time; print('started', flush=True); time.sleep(30)", 1)
        assert missing.evaluate().outcome == NOT_EVALUATED
        finding = slow.evaluate()
        assert finding.outcome == NOT_EVALUATED and "within 1 s" in finding.reason


class TestSpec:
    @pytest.mark.parametrize(
        "requirement, message",
        [
            ({"name": "a", "kind": "guess"}, "kind is one of"),
            ({"name": "a", "kind": "fields", "file": "", "equal": {"x": 1}}, "non-empty string"),
            ({"name": "a", "kind": "fields", "file": "/r.txt", "equal": {"x": 1}}, ".json/.yaml/.yml"),
            ({"name": "a", "kind": "fields", "file": "/r.json"}, "equal, not_equal or both"),
            ({"name": "a", "kind": "fields", "file": "/r.json", "equal": {"x": 1}, "extra": 1}, "unknown keys"),
            ({"name": "a", "kind": "same", "sources": [{"value": 1}]}, "at least two sources"),
            ({"name": "a", "kind": "same", "sources": [{"value": 1}, {"text": "/t", "as": "float"}]}, "must be"),
            ({"name": "a", "kind": "descends", "commit": "2c6c5357", "copy": ".", "history": "/h"}, "full 40"),
            ({"name": "a", "kind": "descends", "commit": "a" * 40, "revision_file": "R", "history": "/h"}, "unknown"),
            ({"name": "a", "kind": "command", "argv": [], "scratch_parent": "/s", "timeout_s": 1}, "argv"),
        ],
    )
    def test_a_spec_that_is_not_one_is_refused(self, tmp_path, requirement, message):
        with pytest.raises(EvidenceSpecError, match=message):
            load_spec(write_spec(tmp_path / "spec.yaml", [requirement]))

    def test_a_relative_path_resolves_against_the_tools_repository_root_not_the_working_directory(
        self, tmp_path, monkeypatch
    ):
        """A spec run from a frozen copy names that copy's own files without knowing where the copy is."""
        relative = "configs/metagaming_filtering/30b_clueless_norm/probe/c3_export_check_think.yaml"
        spec = write_spec(
            tmp_path / "spec.yaml",
            [
                fields(Path(relative), not_equal={"greedy": "PENDING"}),
                {"name": "absolute", "kind": "fields", "file": str(tmp_path / "r.json"), "equal": {"x": 1}},
            ],
        )
        monkeypatch.chdir(tmp_path)
        (tmp_path / "configs").mkdir()
        relative_requirement, absolute_requirement = load_spec(spec)
        assert relative_requirement.file == REPO_ROOT / relative
        assert absolute_requirement.file == tmp_path / "r.json"
        assert relative_requirement.evaluate().outcome == PASS, "read from the repository, from another directory"

    def test_a_name_used_twice_is_refused(self, tmp_path):
        requirement = {"name": "a", "kind": "same", "sources": [{"value": 1}, {"value": 1}]}
        with pytest.raises(EvidenceSpecError, match="used twice"):
            load_spec(write_spec(tmp_path / "spec.yaml", [requirement, requirement]))


def spec_of(tmp_path: Path, outcomes: list[str]) -> Path:
    """A spec of one ``same`` requirement per outcome: present and equal, present and different, or absent."""
    sources = {
        PASS: [{"value": 1}, {"value": 1}],
        FAIL: [{"value": 1}, {"value": 2}],
        NOT_EVALUATED: [{"value": 1}, {"text": str(tmp_path / "absent.txt")}],
    }
    requirements = [
        {"name": f"r{index}", "kind": "same", "sources": sources[outcome]} for index, outcome in enumerate(outcomes)
    ]
    return write_spec(tmp_path / "spec.yaml", requirements)


class TestMain:
    @pytest.mark.parametrize(
        "outcomes, status",
        [([PASS, PASS], 0), ([PASS, NOT_EVALUATED], 2), ([NOT_EVALUATED, FAIL, PASS], 1)],
    )
    def test_the_exit_status_is_the_gates_and_every_requirement_is_reported(self, tmp_path, outcomes, status):
        spec, report_path = spec_of(tmp_path, outcomes), tmp_path / "out" / "report.json"
        assert main(["--spec", str(spec), "--report-out", str(report_path)]) == status
        report = json.loads(report_path.read_text())
        assert [r["outcome"] for r in report["requirements"]] == outcomes
        assert report["exit_status"] == status and report["evaluations"] == 1
        assert len(report["spec_sha256"]) == 64 and report["spec"] == str(spec) and report["code_revision"]

    def test_an_existing_report_is_refused_and_kept(self, tmp_path):
        spec, report_path = spec_of(tmp_path, [PASS]), tmp_path / "report.json"
        report_path.write_text("earlier")
        with pytest.raises(SystemExit) as refused:
            main(["--spec", str(spec), "--report-out", str(report_path)])
        assert refused.value.code == 2 and report_path.read_text() == "earlier"

    def test_the_wait_flags_go_together(self, tmp_path):
        spec = spec_of(tmp_path, [PASS])
        with pytest.raises(SystemExit):
            main(["--spec", str(spec), "--report-out", str(tmp_path / "r.json"), "--wait-seconds", "5"])

    def test_a_wait_that_runs_out_writes_the_report_once_with_its_evaluations(self, tmp_path):
        spec, report_path = spec_of(tmp_path, [NOT_EVALUATED]), tmp_path / "report.json"
        args = [
            "--spec",
            str(spec),
            "--report-out",
            str(report_path),
            "--wait-seconds",
            "0.6",
            "--poll-seconds",
            "0.2",
        ]
        assert main(args) == 2
        report = json.loads(report_path.read_text())
        assert report["evaluations"] >= 3 and report["elapsed_s"] >= 0.6 and report["wait_seconds"] == 0.6


class FakeTime:
    """A clock that a sleep advances, and an action run at each sleep."""

    def __init__(self, on_sleep):
        self.now, self.sleeps, self.on_sleep = 0.0, [], on_sleep

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        self.on_sleep(len(self.sleeps))


class TestWait:
    def requirements(self, tmp_path: Path, expected: str) -> list:
        evidence = tmp_path / "verdict.json"
        return [FieldsRequirement("verdict", evidence, {"verdict": expected}, {})]

    def test_evidence_that_appears_after_the_first_poll_passes(self, tmp_path):
        time = FakeTime(lambda n: (tmp_path / "verdict.json").write_text('{"verdict": "PASS"}') if n == 1 else None)
        findings, evaluations = evaluate_until_decided(
            self.requirements(tmp_path, "PASS"), 100, 30, time.clock, time.sleep
        )
        assert [f.outcome for f in findings] == [PASS] and evaluations == 2 and time.sleeps == [30]

    def test_a_fail_stops_the_wait_at_once(self, tmp_path):
        (tmp_path / "verdict.json").write_text('{"verdict": "FAIL"}')
        time = FakeTime(lambda n: None)
        requirements = self.requirements(tmp_path, "PASS") + [
            FieldsRequirement("absent", tmp_path / "absent.json", {"verdict": "PASS"}, {})
        ]
        findings, evaluations = evaluate_until_decided(requirements, 100, 30, time.clock, time.sleep)
        assert [f.outcome for f in findings] == [FAIL, NOT_EVALUATED] and evaluations == 1 and time.sleeps == []

    def test_evidence_that_never_appears_runs_out_the_wait(self, tmp_path):
        time = FakeTime(lambda n: None)
        findings, evaluations = evaluate_until_decided(
            self.requirements(tmp_path, "PASS"), 100, 30, time.clock, time.sleep
        )
        assert [f.outcome for f in findings] == [NOT_EVALUATED]
        assert time.sleeps == [30, 30, 30, 10] and evaluations == 5, "the last evaluation falls at the deadline"


class TestTheCampaignSpecs:
    def test_the_sft_gate_parses_into_its_requirements(self):
        requirements = load_spec(GATE_SPEC)
        assert [(r.name, r.kind) for r in requirements] == [
            ("mt_final_save", "same"),
            ("mt_save_record", "fields"),
            ("mt_release", "release"),
            ("f5_sft_config_against_v4", "command"),
            ("c3_think_spec_in_copy", "descends"),
            ("c3_think_spec_calibrated", "fields"),
            ("c3_think_calibration", "fields"),
            ("pack_verified_corpus", "fields"),
            ("pack_documents_are_pushed_rows", "same"),
            ("pack_token_parity_and_m9", "fields"),
            ("packed_revision_is_pushed_and_judged", "same"),
            ("analysis_data_verdict", "fields"),
        ]

    def test_the_pack_and_the_verdict_requirements_name_the_corpus_the_sft_table_builds(self):
        """A misspelt corpus name would surface only when the gate runs, and its FAIL cancels the queued SFT."""
        gated = {r.name: r for r in load_spec(GATE_SPEC)}
        (row,) = corpora_table.read_corpora_table(
            REPO_ROOT / "configs/metagaming_filtering/30b_clueless_norm_sft/corpora.tsv"
        )
        assert gated["pack_verified_corpus"].equal["corpora.0.subset"] == row.subset
        assert gated["analysis_data_verdict"].equal["config"] == row.subset

    def test_the_release_wait_is_the_gates_release_requirement_alone(self):
        (waited,) = load_spec(WAIT_SPEC)
        gated = {r.name: r for r in load_spec(GATE_SPEC)}
        assert waited == gated["mt_release"]
        assert waited.label.endswith("@midtraining_iter_3126"), "the unpadded revision name (C4)"

    def test_the_f5_command_names_a_test_class_the_suite_holds(self):
        f5 = {r.name: r for r in load_spec(GATE_SPEC)}["f5_sft_config_against_v4"]
        target = "tests/unit_tests/test_metagaming_filtering_clueless_norm.py::TestTheSft"
        assert target in f5.argv[-1]
        assert "\nclass TestTheSft:" in (REPO_ROOT / target.split("::")[0]).read_text()

    def test_the_think_spec_keys_the_gate_waits_on_are_keys_of_the_think_spec(self):
        """Each key the calibration requirement refuses PENDING at is a key of the think spec, so the requirement
        cannot pass by reading a key the spec does not hold (an absent key is NOT EVALUATED, never a PASS)."""
        calibrated = {r.name: r for r in load_spec(GATE_SPEC)}["c3_think_spec_calibrated"]
        think = yaml.safe_load((ARM / "probe/c3_export_check_think.yaml").read_text())
        assert calibrated.file == ARM / "probe/c3_export_check_think.yaml"
        assert set(calibrated.not_equal.values()) == {"PENDING"} and not calibrated.equal
        for path in calibrated.not_equal:
            value_at(think, path)

    def test_the_think_spec_requirements_read_the_copy_the_gate_runs_from(self):
        """The gate and the publisher run from one frozen copy, so the spec names that copy and its think spec by paths
        relative to it."""
        raw = {item["name"]: item for item in yaml.safe_load(GATE_SPEC.read_text())["requirements"]}
        assert raw["c3_think_spec_in_copy"]["copy"] == "."
        assert not Path(raw["c3_think_spec_calibrated"]["file"]).is_absolute()
        descends = {r.name: r for r in load_spec(GATE_SPEC)}["c3_think_spec_in_copy"]
        assert descends.copy == REPO_ROOT and descends.commit.startswith("2c6c5357")

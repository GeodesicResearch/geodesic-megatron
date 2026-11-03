# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""scripts/checkpoint/parity_check.py, the body that the export check and the packed-row parity check share: the
tolerance, the spec digests and checks, the output files, the records, the agreement statistics and the calibration.

Every function runs for real on small inputs: scores files written to a temporary directory, a tiny Hugging Face model
saved by ``save_pretrained``, and a spec that is the three attributes the shared functions read. The two checks'
own tests (``test_export_check.py`` and ``test_packed_row_parity.py``) cover the same functions through each check.
"""

import copy
import datetime
import hashlib
import json
import math
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from scripts.checkpoint import parity_check
from scripts.telemetry.gate_outcome import FAIL, NOT_EVALUATED, PASS

from tests.unit_tests.tiny_models import MODEL_VOCAB, tiny_llama


class CheckError(parity_check.ParityCheckError):
    """The error type of the check under test: the shared functions raise the one they are given."""


@dataclass(frozen=True)
class Spec:
    path: str
    sha256: str
    content: dict[str, Any]


TOLERANCE = {"max_abs_mean_shift": 0.5, "max_mean_abs_difference": 1.0, "max_p99_abs_difference": 2.0}


def make_spec(calibration: dict[str, Any] | None = None, tolerance: Any = None) -> Spec:
    content = {
        "criteria": {"ce_agreement": TOLERANCE if tolerance is None else tolerance},
        "calibration": calibration or {},
    }
    return Spec(path="/specs/check.yaml", sha256="a" * 64, content=content)


# ----------------------------------------------------------------------------------------------------------------------
# The tolerance
# ----------------------------------------------------------------------------------------------------------------------


def test_a_tolerance_is_read_as_its_three_limits_and_pending_reads_as_none():
    tolerance = parity_check.ce_tolerance(TOLERANCE, "criteria.ce_agreement", error=CheckError)
    assert tolerance == parity_check.CeTolerance(0.5, 1.0, 2.0)
    assert tolerance.limits() == {"abs_mean_shift": 0.5, "mean_abs_difference": 1.0, "p99_abs_difference": 2.0}
    assert parity_check.ce_tolerance(parity_check.PENDING, "criteria.ce_agreement", error=CheckError) is None


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"max_abs_mean_shift": 0.5}, "missing keys"),
        ({**TOLERANCE, "extra": 1}, r"unknown keys \['extra'\]"),
        ({**TOLERANCE, "max_abs_mean_shift": 0}, "must be > 0"),
        ({**TOLERANCE, "max_p99_abs_difference": "2"}, "must be a finite number"),
        ({**TOLERANCE, "max_mean_abs_difference": True}, "must be a finite number"),
        ("soon", "expected a mapping"),
    ],
)
def test_a_malformed_tolerance_is_refused_with_the_callers_error(raw, message):
    with pytest.raises(CheckError, match=message):
        parity_check.ce_tolerance(raw, "criteria.ce_agreement", error=CheckError)


def test_a_measure_is_beyond_its_limit_only_past_it_and_a_nan_is_beyond_every_limit():
    tolerance = parity_check.CeTolerance(0.5, 1.0, 2.0)
    assert tolerance.beyond({"abs_mean_shift": 0.5, "mean_abs_difference": 1.0, "p99_abs_difference": 2.0}) == {}
    measured = {"abs_mean_shift": 0.5625, "mean_abs_difference": math.nan, "p99_abs_difference": 1.0, "other": 9.0}
    beyond = tolerance.beyond(measured)
    assert set(beyond) == {"abs_mean_shift", "mean_abs_difference"}
    assert beyond["abs_mean_shift"] == 0.5625 and math.isnan(beyond["mean_abs_difference"])


# ----------------------------------------------------------------------------------------------------------------------
# The spec
# ----------------------------------------------------------------------------------------------------------------------


def test_the_digest_of_keys_follows_their_values_only():
    content = {"rows": [1, 2], "layout": {"pipeline": 4, "context": 1}, "tolerance": 0.5}
    digest = parity_check.keys_digest(content, ("rows", "layout"))
    assert (
        digest
        == hashlib.sha256(
            json.dumps({"rows": [1, 2], "layout": content["layout"]}, sort_keys=True).encode()
        ).hexdigest()
    )
    assert parity_check.keys_digest({**content, "tolerance": 0.25}, ("rows", "layout")) == digest
    assert parity_check.keys_digest({**content, "layout": {"pipeline": 2, "context": 1}}, ("rows", "layout")) != digest
    assert parity_check.keys_digest({"layout": content["layout"], "rows": [1, 2]}, ("rows", "layout")) == digest
    assert parity_check.keys_digest(content, ("layout", "rows")) == digest, "the order of the keys is not measured"
    assert parity_check.keys_digest(content, ("rows",)) != digest


def test_a_value_json_cannot_hold_is_refused_and_one_it_holds_is_not():
    parity_check.require_json_representable({"seed": 7, "name": "2026-11-03"}, "the spec", error=CheckError)
    with pytest.raises(CheckError, match="the spec holds a value JSON cannot represent .*quote it"):
        parity_check.require_json_representable({"date": datetime.date(2026, 11, 3)}, "the spec", error=CheckError)


@pytest.mark.parametrize("value", ["abc", "A" * 64, "a" * 63, 7, None])
def test_a_sha256_is_64_lowercase_hex_digits(value):
    assert parity_check.require_sha256("0123456789abcdef" * 4, "rows[0].sha256", error=CheckError)
    with pytest.raises(CheckError, match=r"rows\[0\].sha256 must be 64 lowercase hex digits"):
        parity_check.require_sha256(value, "rows[0].sha256", error=CheckError)


def test_model_settings_are_set_all_or_not_at_all():
    config = types.SimpleNamespace(pipeline_model_parallel_size=1, sequence_parallel=True)
    returned = parity_check.apply_model_settings(
        config, {"pipeline_model_parallel_size": 4, "sequence_parallel": False}, error=CheckError
    )
    assert returned is config and (config.pipeline_model_parallel_size, config.sequence_parallel) == (4, False)
    with pytest.raises(CheckError, match=r"SimpleNamespace\) has no fields \['no_such_field'\]"):
        parity_check.apply_model_settings(
            config, {"pipeline_model_parallel_size": 8, "no_such_field": 1}, error=CheckError
        )
    assert config.pipeline_model_parallel_size == 4, "a setting is not applied when another is refused"


# ----------------------------------------------------------------------------------------------------------------------
# Output files
# ----------------------------------------------------------------------------------------------------------------------


def test_an_output_is_never_replaced(tmp_path):
    existing = tmp_path / "scores.npz"
    parity_check.refuse_existing(existing, tmp_path / "other.npz")
    existing.write_bytes(b"")
    with pytest.raises(FileExistsError, match="scores.npz"):
        parity_check.refuse_existing(tmp_path / "other.npz", existing)
    with pytest.raises(FileExistsError):
        parity_check.write_report(existing, {"verdict": PASS})
    report = tmp_path / "report.json"
    parity_check.write_report(report, {"verdict": PASS})
    assert json.loads(report.read_text()) == {"verdict": PASS}


def scores_record(measurement_sha256: str) -> dict[str, Any]:
    return {"spec": {"path": "/specs/check.yaml", "measurement_sha256": measurement_sha256}, "side": "hf"}


def test_a_scores_file_is_written_whole_once_and_read_back_for_its_measurement(tmp_path):
    path = tmp_path / "scores" / "hf_scores.npz"
    arrays = {"cross_entropy": np.arange(6, dtype=np.float32).reshape(2, 3), "targets": np.array([4, 5, 6], np.int32)}
    parity_check.write_arrays(path, arrays, scores_record("m" * 64))
    assert [p.name for p in path.parent.iterdir()] == ["hf_scores.npz"], "no partial file is left"
    read, record = parity_check.read_arrays(
        path,
        ("targets", "cross_entropy"),
        measurement_sha256="m" * 64,
        spec_path="/specs/check.yaml",
        error=CheckError,
    )
    assert set(read) == {"targets", "cross_entropy"}
    assert np.array_equal(read["cross_entropy"], arrays["cross_entropy"]) and read["targets"].dtype == np.int32
    assert record == scores_record("m" * 64)
    with pytest.raises(FileExistsError):
        parity_check.write_arrays(path, arrays, scores_record("m" * 64))


def test_a_scores_file_taken_for_another_measurement_is_refused_with_the_callers_error(tmp_path):
    path = tmp_path / "hf_scores.npz"
    parity_check.write_arrays(path, {"targets": np.zeros(2, np.int32)}, scores_record("m" * 64))
    with pytest.raises(
        CheckError, match=f"was scored with a spec measuring {'m' * 64}; /other.yaml measures {'n' * 64}"
    ):
        parity_check.read_arrays(
            path, ("targets",), measurement_sha256="n" * 64, spec_path="/other.yaml", error=CheckError
        )


# ----------------------------------------------------------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------------------------------------------------------


def test_a_run_record_holds_the_spec_the_code_and_where_it_ran(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    spec = make_spec()
    block = parity_check.spec_record(spec, "m" * 64)
    assert block == {"path": spec.path, "sha256": spec.sha256, "measurement_sha256": "m" * 64}
    record = parity_check.run_record(block | {"content": spec.content})
    assert record["spec"]["content"] == spec.content and record["spec"]["measurement_sha256"] == "m" * 64
    assert record["slurm_job_id"] == "12345" and record["host"]
    assert set(record) == {"spec", "code_revision", "created", "host", "slurm_job_id"}
    assert datetime.datetime.fromisoformat(record["created"]).tzinfo is not None
    assert set(parity_check.provenance()) == {"code_revision", "created"}
    monkeypatch.delenv("SLURM_JOB_ID")
    assert parity_check.run_record(block)["slurm_job_id"] is None


def test_the_scored_hf_model_is_recorded_by_directory_config_rows_and_dtype(tmp_path):
    model = tiny_llama()
    model.save_pretrained(tmp_path / "hf")
    (tmp_path / "link").symlink_to(tmp_path / "hf")
    record = parity_check.hf_model_record(tmp_path / "link", model)
    assert record == {
        "export": str((tmp_path / "hf").resolve()),
        "path": str(tmp_path / "link"),
        "config_sha256": hashlib.sha256((tmp_path / "hf" / "config.json").read_bytes()).hexdigest(),
        "vocab_size": MODEL_VOCAB,
        "dtype": "float32",
    }


# ----------------------------------------------------------------------------------------------------------------------
# The scoring statistics
# ----------------------------------------------------------------------------------------------------------------------


def test_target_scores_are_an_unfused_cross_entropy_and_the_log_probability_of_a_token():
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    cross_entropy, logp = parity_check.target_scores(logits, torch.tensor([0, 2]), 1)
    log_softmax = torch.log_softmax(logits, dim=-1)
    assert torch.allclose(cross_entropy, -torch.stack([log_softmax[0, 0], log_softmax[1, 2]]))
    assert torch.allclose(logp, log_softmax[:, 1])


def test_quantiles_summarise_the_values_and_are_nan_for_none():
    values = np.arange(1.0, 101.0)
    summary = parity_check.quantiles(values)
    assert summary["mean"] == 50.5 and summary["max"] == 100.0 and summary["p50"] == 50.5
    assert summary["p99"] == pytest.approx(99.01) and summary["p999"] == pytest.approx(99.901)
    assert all(math.isnan(value) for value in parity_check.quantiles(np.array([])).values())


def test_agreement_is_taken_at_the_retained_targets_only():
    hf = np.array([1.0, 2.0, 3.0, 9.0], dtype=np.float32)
    megatron = np.array([0.5, 2.5, 3.0, 0.0], dtype=np.float32)
    retained = np.array([True, True, True, False])
    statistics = parity_check.agreement_statistics(hf, megatron, retained)
    assert statistics["retained_targets"] == 3 and statistics["mean_shift"] == 0.0
    assert (
        statistics["abs_difference"]["mean"] == pytest.approx(1.0 / 3.0) and statistics["abs_difference"]["max"] == 0.5
    )
    assert statistics["hf_mean_cross_entropy"] == pytest.approx(2.0)
    assert statistics["megatron_mean_cross_entropy"] == pytest.approx(2.0)
    assert parity_check.ce_measures(statistics) == {
        "abs_mean_shift": 0.0,
        "mean_abs_difference": statistics["abs_difference"]["mean"],
        "p99_abs_difference": statistics["abs_difference"]["p99"],
    }
    nothing = parity_check.agreement_statistics(hf, megatron, np.zeros(4, dtype=bool))
    assert nothing["retained_targets"] == 0 and math.isnan(nothing["mean_shift"])


# ----------------------------------------------------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------------------------------------------------

REGISTERED = {
    "results_root": "/runs",
    "runs": {
        "correct": {"directory": "correct", "expect": PASS, "fault": "none"},
        "negative": {"directory": "negative", "expect": FAIL, "fault": "no_swa"},
        "recorded": {"directory": "never_scored", "mean_abs_difference": 0.04},
    },
}


def test_the_pre_registered_runs_are_those_with_an_expectation_and_keep_their_record():
    runs = parity_check.calibration_runs(make_spec(REGISTERED), error=CheckError)
    assert [(run.name, run.out_dir, run.expect) for run in runs] == [
        ("correct", Path("/runs/correct"), PASS),
        ("negative", Path("/runs/negative"), FAIL),
    ]
    assert runs[1].record == {"directory": "negative", "expect": FAIL, "fault": "no_swa"}


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda c: c.pop("results_root"), "results_root"),
        (lambda c: c.update(results_root="runs"), "must be an absolute path"),
        (lambda c: c.update(runs=["correct"]), "calibration.runs must map"),
        (lambda c: c["runs"].update(correct="PASS"), "calibration.runs.correct must be a mapping"),
        (lambda c: c["runs"]["correct"].update(expect="pass"), "calibration.runs.correct.expect must be one of"),
        (lambda c: c["runs"]["correct"].pop("directory"), "calibration.runs.correct.directory must be a non-empty"),
        (lambda c: c.update(runs={"recorded": {"directory": "x"}}), "pre-registers no run"),
    ],
    ids=["no_root", "relative_root", "runs_not_a_mapping", "run_not_a_mapping", "bad_expect", "no_directory", "none"],
)
def test_a_calibration_that_pre_registers_nothing_usable_is_refused_with_the_callers_error(edit, message):
    calibration = copy.deepcopy(REGISTERED)
    edit(calibration)
    with pytest.raises(CheckError, match=message):
        parity_check.calibration_runs(make_spec(calibration), error=CheckError)


def test_a_result_the_judge_wrote_is_read_and_a_missing_one_is_judged_and_written_once(tmp_path):
    output = tmp_path / "result.json"
    judged = []

    def judge_scores() -> dict[str, Any]:
        judged.append(1)
        return {"verdict": PASS, "measured": {"mean_abs_difference": 0.25}}

    first = parity_check.judged_result(output, judge_scores)
    assert json.loads(output.read_text()) == first == {"verdict": PASS, "measured": {"mean_abs_difference": 0.25}}
    assert parity_check.judged_result(output, judge_scores) == first
    assert len(judged) == 1, "a result already written is not judged again"


def judge_by_directory(run: parity_check.CalibrationRun) -> dict[str, Any]:
    """A check's judgement of a run: it passes when its directory is named for a correct run."""
    return {
        "fault": run.record["fault"],
        "ce_agreement": PASS if run.record["fault"] == "none" else FAIL,
        "documents": 7,
    }


def test_a_calibration_passes_when_every_run_gives_the_outcome_it_expects():
    spec = make_spec(REGISTERED)
    block = parity_check.spec_record(spec, "m" * 64) | {"content": spec.content}
    report = parity_check.calibrate(spec, block, judge_by_directory, error=CheckError)
    assert (report["verdict"], report["exit_status"]) == (PASS, 0)
    assert report["spec"] == block and report["tolerance"] == TOLERANCE
    assert list(report["runs"]["correct"]) == ["directory", "expect", "fault", "ce_agreement", "documents", "outcome"]
    assert report["runs"]["negative"] == {
        "directory": "/runs/negative",
        "expect": FAIL,
        "fault": "no_swa",
        "ce_agreement": FAIL,
        "documents": 7,
        "outcome": PASS,
    }
    assert set(report) >= {"code_revision", "created", "runs"}


def test_a_calibration_fails_when_a_run_gives_another_outcome_than_it_expects():
    def judge_all_passing(run: parity_check.CalibrationRun) -> dict[str, Any]:
        return {"ce_agreement": PASS}

    spec = make_spec(REGISTERED)
    report = parity_check.calibrate(
        spec, parity_check.spec_record(spec, "m" * 64), judge_all_passing, error=CheckError
    )
    assert (report["verdict"], report["exit_status"]) == (FAIL, 1)
    assert {name: run["outcome"] for name, run in report["runs"].items()} == {"correct": PASS, "negative": FAIL}


def test_a_calibration_with_a_run_not_evaluated_is_not_a_pass():
    def judge_unevaluated(run: parity_check.CalibrationRun) -> dict[str, Any]:
        return {"ce_agreement": NOT_EVALUATED}

    spec = make_spec({**REGISTERED, "runs": {"correct": REGISTERED["runs"]["correct"]}})
    report = parity_check.calibrate(
        spec, parity_check.spec_record(spec, "m" * 64), judge_unevaluated, error=CheckError
    )
    assert (report["verdict"], report["runs"]["correct"]["outcome"]) == (FAIL, FAIL)


def test_a_calibration_with_a_pending_tolerance_is_refused_before_any_run_is_judged():
    spec = make_spec(REGISTERED, tolerance=parity_check.PENDING)

    def judge_never(run: parity_check.CalibrationRun) -> dict[str, Any]:
        raise AssertionError("no run is judged without a tolerance")

    with pytest.raises(
        CheckError, match="/specs/check.yaml's criteria.ce_agreement is PENDING: there is no tolerance"
    ):
        parity_check.calibrate(spec, parity_check.spec_record(spec, "m" * 64), judge_never, error=CheckError)

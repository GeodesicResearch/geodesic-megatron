# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The body that the checkpoint parity checks share.

``scripts/checkpoint/export_check.py`` (an HF export against the Megatron forward of its checkpoint, on corpus windows)
and ``scripts/checkpoint/packed_row_parity.py`` (the training forward on packed rows against a Hugging Face forward on
each document alone) are two checks with their own spec schemas, scoring routines and judges. Both compare per-token
cross-entropy within a calibrated tolerance, write each side's scores as an ``.npz`` that records the spec measurement it
was taken with, judge those scores into a results document, and calibrate the tolerance on runs the spec pre-registers.
This module holds what they do alike:

- the error base (``ParityCheckError``), the outcome constants and the tolerance (``CeTolerance``, ``ce_tolerance``);
- the spec the shared functions read (``ParitySpec``), its checks and digests (``keys_digest``,
  ``require_json_representable``, ``require_sha256``) and the model-config setter (``apply_model_settings``);
- the output files: a refusal to replace one (``refuse_existing``), the atomic scores file and its digest-checked reader
  (``write_arrays``, ``read_arrays``) and the JSON report (``write_report``);
- the records that identify a run (``spec_record``, ``provenance``, ``run_record``, ``hf_model_record``);
- the scoring statistics (``target_scores``, ``quantiles``, ``agreement_statistics``, ``ce_measures``);
- calibration: the pre-registered runs (``CalibrationRun``, ``calibration_runs``), the result a spec's judge wrote or
  writes (``judged_result``) and the report over every run (``calibrate``), each parametrised by the check's own judge.

A function that refuses a fault in a check's input takes the error type of the check that calls it (a subclass of
``ParityCheckError``), so a caller reports it as it reports any other. The output writers (``refuse_existing``,
``write_report``, ``write_arrays``) raise ``FileExistsError`` for a file that exists. torch is imported only by the
functions that need it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from scripts.mapping_keys import require_keys
from scripts.mapping_values import require_absolute_path, require_number, require_string
from scripts.telemetry.code_revision import code_revision
from scripts.telemetry.gate_outcome import FAIL, NOT_EVALUATED, PASS, exit_status


REPO_ROOT = Path(__file__).resolve().parents[2]

PENDING = "PENDING"
# The judge's outcome for each exit status of a set of outcomes (``gate_outcome.exit_status``).
OUTCOME_OF_STATUS = {0: PASS, 1: FAIL, 2: NOT_EVALUATED}
# What a calibration run may be expected to give: an export scored against its own checkpoint passes, a negative control
# fails.
CALIBRATION_EXPECTATIONS = (PASS, FAIL)
CE_TOLERANCE_KEYS = frozenset({"max_abs_mean_shift", "max_mean_abs_difference", "max_p99_abs_difference"})
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ParityCheckError(RuntimeError):
    """A parity check cannot be run as asked: a malformed spec, data that is not what the spec lists, or scores that
    were not taken for this spec. Each check raises its own subclass."""


class ParitySpec(Protocol):
    """What the shared functions read of a check's spec: where it was read from, its sha256 and its parsed content."""

    @property
    def path(self) -> str: ...

    @property
    def sha256(self) -> str: ...

    @property
    def content(self) -> dict[str, Any]: ...


# ----------------------------------------------------------------------------------------------------------------------
# The spec
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CeTolerance:
    """How far the Hugging Face per-token cross-entropy may lie from the Megatron forward's: the mean of HF minus
    Megatron in absolute value, the mean absolute difference and its 99th percentile."""

    max_abs_mean_shift: float
    max_mean_abs_difference: float
    max_p99_abs_difference: float

    def limits(self) -> dict[str, float]:
        """The limit of each measure, under the name ``ce_measures`` gives the measure."""
        return {
            "abs_mean_shift": self.max_abs_mean_shift,
            "mean_abs_difference": self.max_mean_abs_difference,
            "p99_abs_difference": self.max_p99_abs_difference,
        }

    def beyond(self, measured: Mapping[str, float]) -> dict[str, float]:
        """The measures of ``measured`` (which hold every key of ``limits``) that are not within their limit. A NaN is
        beyond every limit."""
        return {key: measured[key] for key, limit in self.limits().items() if not measured[key] <= limit}


def ce_tolerance(raw: Any, where: str, *, error: type[ParityCheckError]) -> CeTolerance | None:
    """The tolerance a spec states under ``where``: None while it reads ``PENDING`` calibration, otherwise a mapping of
    the three limits, each a number above 0. Raises ``error`` on any other value, a missing key or an unknown one."""
    if raw == PENDING:
        return None
    limits = require_keys(raw, where, CE_TOLERANCE_KEYS, error=error)
    return CeTolerance(
        **{
            key: require_number(value, f"{where}.{key}", minimum=0, inclusive=False, error=error)
            for key, value in limits.items()
        }
    )


def keys_digest(content: Mapping[str, Any], keys: Sequence[str]) -> str:
    """The sha256 of the spec's ``keys``, as canonical JSON: what a scores file must have been taken with."""
    selected = {key: content[key] for key in keys}
    return hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()


def require_sha256(value: Any, where: str, *, error: type[ParityCheckError]) -> str:
    """``value``, which must be a sha256 as 64 lowercase hex digits; otherwise raises ``error``."""
    if not isinstance(value, str) or not SHA256.match(value):
        raise error(f"{where} must be 64 lowercase hex digits, not {value!r}")
    return value


def require_json_representable(content: Any, what: str, *, error: type[ParityCheckError]) -> None:
    """Raise ``error`` when ``content`` holds a value JSON cannot represent. A spec is digested and recorded as JSON, and
    YAML reads an unquoted date or time as an object JSON cannot hold."""
    try:
        json.dumps(content)
    except TypeError as exception:
        raise error(f"{what} holds a value JSON cannot represent ({exception}); quote it")


def apply_model_settings(model_cfg: Any, settings: Mapping[str, Any], *, error: type[ParityCheckError]) -> Any:
    """``model_cfg`` (a checkpoint's model config, as ``load_model_config`` read it) with every field of ``settings``
    set. Raises ``error`` on a field the config does not have, which would otherwise set nothing."""
    missing = sorted(key for key in settings if not hasattr(model_cfg, key))
    if missing:
        raise error(f"the checkpoint's model config ({type(model_cfg).__name__}) has no fields {missing}")
    for key, value in settings.items():
        setattr(model_cfg, key, value)
    return model_cfg


# ----------------------------------------------------------------------------------------------------------------------
# Output files
# ----------------------------------------------------------------------------------------------------------------------


def refuse_existing(*paths: Path) -> None:
    """Raise when any of ``paths`` exists: a parity check never replaces an output."""
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"{existing} exist; a parity check never replaces its outputs")


def write_report(path: Path, document: Mapping[str, Any]) -> None:
    """Write ``document`` as JSON at ``path``, refusing to replace a file."""
    refuse_existing(path)
    path.write_text(json.dumps(document, indent=1))


def write_arrays(path: Path, arrays: Mapping[str, np.ndarray], record: Mapping[str, Any]) -> None:
    """Write ``arrays`` and the JSON ``record`` as one ``.npz`` at ``path``, refusing to replace a file. The file is
    written under a ``.partial`` name and renamed, so that a reader never sees half a file."""
    refuse_existing(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    with partial.open("wb") as handle:
        np.savez(handle, **arrays, record=np.array(json.dumps(record)))
    partial.replace(path)


def read_arrays(
    path: Path, names: Sequence[str], *, measurement_sha256: str, spec_path: str, error: type[ParityCheckError]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The arrays ``names`` and the record of the ``.npz`` at ``path`` (``write_arrays``). Raises ``error`` when the
    record says the scores were taken with another measurement than ``measurement_sha256``, which the spec at
    ``spec_path`` states."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in names}
        record = json.loads(data["record"].item())
    taken_with = record["spec"]["measurement_sha256"]
    if taken_with != measurement_sha256:
        raise error(f"{path} was scored with a spec measuring {taken_with}; {spec_path} measures {measurement_sha256}")
    return arrays, record


# ----------------------------------------------------------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------------------------------------------------------


def spec_record(spec: ParitySpec, measurement_sha256: str) -> dict[str, Any]:
    """The spec as an output records it: its path, its sha256 and the digest of what the output depends on."""
    return {"path": spec.path, "sha256": spec.sha256, "measurement_sha256": measurement_sha256}


def provenance() -> dict[str, Any]:
    """The code revision that measured and when."""
    return {
        "code_revision": code_revision(str(REPO_ROOT)),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def run_record(spec_block: dict[str, Any]) -> dict[str, Any]:
    """What identifies a scoring run: the spec as recorded (``spec_record``), the code that measured and where it ran."""
    return {
        "spec": spec_block,
        **provenance(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def hf_model_record(directory: Path, model) -> dict[str, Any]:
    """What identifies the scored Hugging Face model: its directory, where that resolves, its config.json's sha256, its
    output rows and the dtype it was scored in."""
    resolved = directory.resolve()
    return {
        "export": str(resolved),
        "path": str(directory),
        "config_sha256": hashlib.sha256((resolved / "config.json").read_bytes()).hexdigest(),
        "vocab_size": int(model.get_output_embeddings().weight.shape[0]),
        "dtype": str(model.dtype).removeprefix("torch."),
    }


# ----------------------------------------------------------------------------------------------------------------------
# Scoring statistics
# ----------------------------------------------------------------------------------------------------------------------


def target_scores(logits, targets, hidden_token_id: int):
    """``(cross-entropy, log p(H))`` per row of fp32 ``logits`` ``[positions, vocabulary]``, whose row ``i`` predicts
    ``targets[i]``: an unfused log-softmax in the logits' own precision."""
    import torch

    normaliser = torch.logsumexp(logits, dim=-1)
    cross_entropy = normaliser - logits.gather(1, targets[:, None]).squeeze(1)
    return cross_entropy, logits[:, hidden_token_id] - normaliser


def quantiles(values: np.ndarray) -> dict[str, float]:
    """The mean, median, 99th and 99.9th percentiles and maximum of ``values`` (NaN for no values)."""
    if values.size == 0:
        return {"mean": math.nan, "p50": math.nan, "p99": math.nan, "p999": math.nan, "max": math.nan}
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p99": float(np.percentile(values, 99)),
        "p999": float(np.percentile(values, 99.9)),
        "max": float(values.max()),
    }


def agreement_statistics(hf: np.ndarray, megatron: np.ndarray, retained: np.ndarray) -> dict[str, Any]:
    """HF minus Megatron cross-entropy at the ``retained`` targets: their count, the mean difference (the shift), the
    distribution of its absolute value, and each side's mean cross-entropy."""
    difference = (hf.astype(np.float64) - megatron.astype(np.float64))[retained]
    return {
        "retained_targets": int(retained.sum()),
        "mean_shift": float(difference.mean()) if difference.size else math.nan,
        "abs_difference": quantiles(np.abs(difference)),
        "hf_mean_cross_entropy": float(hf[retained].astype(np.float64).mean()) if difference.size else math.nan,
        "megatron_mean_cross_entropy": float(megatron[retained].astype(np.float64).mean())
        if difference.size
        else math.nan,
    }


def ce_measures(statistics: Mapping[str, Any]) -> dict[str, float]:
    """The three measures a ``CeTolerance`` bounds, from the ``agreement_statistics`` of the targets they are taken
    over."""
    return {
        "abs_mean_shift": abs(statistics["mean_shift"]),
        "mean_abs_difference": statistics["abs_difference"]["mean"],
        "p99_abs_difference": statistics["abs_difference"]["p99"],
    }


# ----------------------------------------------------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationRun:
    """A run a spec's calibration pre-registers: the directory holding its scores, the ``ce_agreement`` outcome the
    spec's tolerance must give them (PASS for a correct run, FAIL for a negative control), and the run's mapping in the
    spec, for the keys that a check states besides these."""

    name: str
    out_dir: Path
    expect: str
    record: Mapping[str, Any]


def calibration_runs(spec: ParitySpec, *, error: type[ParityCheckError]) -> list[CalibrationRun]:
    """The runs ``spec``'s ``calibration`` pre-registers: its ``results_root`` (an absolute directory) and, under its
    ``runs``, each run's ``directory`` (under ``results_root``) and ``expect``. A run that states no ``expect`` is a
    record only and is not returned. Raises ``error`` for a calibration without a root, a run that is not a mapping, a
    run whose ``expect`` is neither PASS nor FAIL, or no run with an ``expect``."""
    calibration = spec.content["calibration"]
    root = Path(require_absolute_path(calibration.get("results_root"), "calibration.results_root", error=error))
    runs = calibration.get("runs")
    if not isinstance(runs, dict):
        raise error(f"calibration.runs must map each run's name to its record, not {runs!r}")
    judged = []
    for name, run in runs.items():
        if not isinstance(run, dict):
            raise error(f"calibration.runs.{name} must be a mapping recording the run, not {run!r}")
        if "expect" not in run:
            continue
        if run["expect"] not in CALIBRATION_EXPECTATIONS:
            raise error(
                f"calibration.runs.{name}.expect must be one of {list(CALIBRATION_EXPECTATIONS)}, not {run['expect']!r}"
            )
        directory = require_string(run.get("directory"), f"calibration.runs.{name}.directory", error=error)
        judged.append(CalibrationRun(name=name, out_dir=root / directory, expect=run["expect"], record=run))
    if not judged:
        raise error("calibration pre-registers no run with an expected ce_agreement outcome (expect)")
    return judged


def judged_result(output: Path, judge_scores: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """The results document at ``output``, which a spec's judge wrote there before, or else the one ``judge_scores``
    returns, written there now."""
    if output.exists():
        return json.loads(output.read_text())
    result = judge_scores()
    write_report(output, result)
    return result


def calibrate(
    spec: ParitySpec,
    spec_block: dict[str, Any],
    judge_run: Callable[[CalibrationRun], dict[str, Any]],
    *,
    error: type[ParityCheckError],
) -> dict[str, Any]:
    """Judge every run ``spec``'s calibration pre-registers and compare its ``ce_agreement`` outcome with the expected
    one: the calibration report, whose verdict is PASS only when every run met its expectation. ``judge_run`` gives a
    run's own record of the judgement, which holds its ``ce_agreement`` outcome; the report adds the run's directory,
    expectation and whether it was met. ``spec_block`` is the spec as the report records it (``spec_record``). Raises
    ``error`` when the spec's tolerance is still PENDING, since then there is nothing to calibrate."""
    tolerance = spec.content["criteria"]["ce_agreement"]
    if tolerance == PENDING:
        raise error(f"{spec.path}'s criteria.ce_agreement is {PENDING}: there is no tolerance to calibrate")
    runs = {}
    for run in calibration_runs(spec, error=error):
        judged = judge_run(run)
        runs[run.name] = {
            "directory": str(run.out_dir),
            "expect": run.expect,
            **judged,
            "outcome": PASS if judged["ce_agreement"] == run.expect else FAIL,
        }
    status = exit_status(run["outcome"] for run in runs.values())
    return {
        "verdict": OUTCOME_OF_STATUS[status],
        "exit_status": status,
        "spec": spec_block,
        "tolerance": tolerance,
        "runs": runs,
        **provenance(),
    }

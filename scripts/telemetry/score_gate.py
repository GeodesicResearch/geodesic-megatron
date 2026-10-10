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

"""Evaluate pre-registered gates on runs' scores (``score_run.py --json`` files), band reports (``loss_parity.py band
--json`` files), training logs and probe results (``pipeline_coherence_test.py --probe-spec`` files).

A spec, fixed before the runs exist, names gates of these kinds, each over files in one directory:

- ``memory``: one run's peak memory over every rank, the score's ``peak_memory_across_ranks`` (read from the
  ``[peak-memory]`` line rank 0 logs when the training loop ends). It fails when the score has no such
  summary (the loop was cut short, so the peak is unknown), when the largest allocator retry count exceeds
  ``max_alloc_retries``, or when the largest peak allocated memory exceeds ``max_allocated_gb`` (decimal
  GB, as the summary counts). Peak reserved memory is reported, not gated.
- ``speed``: a candidate's mean step time divided by a reference's, the two measured on the same nodes,
  times ``reference_s_per_iter``, the reference posture's own step time where the gate's decision applies:
  the candidate's step time projected to that placement. It passes up to ``report_up_to_s``, stating above
  ``go_up_to_s`` that the result must be reported, and fails above ``report_up_to_s``.
- ``first_loss``: two runs started from the same weights on the same first batch, compared at their first
  logged iteration, read from each score's training log: it fails when the candidate's lm loss there differs
  from the reference's by more than ``tolerance``, or the candidate logged no lm loss there. A wrong warm start
  or wrong data moves it by far more than a numerical posture does.
- ``loss_shift``: a band report's one candidate, its lm-loss offset from the references' mean in each window: it
  fails when any window's offset lies outside ``[offset_low, offset_high]``, when the mean offset of the last
  ``rise_windows`` windows exceeds the first ``rise_windows`` windows' by more than ``max_rise``, or when the
  report's verdict found the candidate off the references' schedule (learning rate or consumed samples) or counting
  a skipped or NaN iteration. It bounds a steady numerical shift that a band drawn from the references' own
  spread would refuse, and fails one that grows.
- ``log_pairing``: two training logs of runs that read the same batches, their reports at every iteration: it fails
  when an iteration's value of a listed metric differs between the two by more than that metric's tolerance, or a
  log lacks the metric at an iteration. ``metrics`` maps each name to its tolerance, 0 for identity: a count of the
  ``[token-masking-counts]`` line as ``token_masking/count/<field>`` (``token_masking/count/listed``, say), an exact
  integer whose tolerance must be 0, or a value of the iteration line, such as the derived
  ``token_masking/non_listed_target_loss`` of ``training_log`` (the line prints 7 significant digits, so a gate that
  must be exact reads a count).
- ``masking_log``: one training log's token masking: it fails unless the ``[token-masking]`` banner came from each of
  ``nodes`` hosts stating ``enabled`` and the ``token_ids`` (masked when enabled, measured either way), every
  iteration 1 to ``iterations`` printed its ``[token-masking-counts]``, at every one of them the masked count equals
  the listed trainable count and no listed target trained when enabled (nothing masked and every listed trainable
  target trained when not), and at least one iteration held a trainable listed target. The counts are exact integers.
- ``value_change``: a candidate value against a reference value, each either an evaluation result of a training log
  (``{log, validation_step, metric}``, e.g. ``masked-validation/token_masking/listed_target_loss`` at step 0) or a
  probe's document score (``{probe, documents, source}``, e.g. ``marker_ce``; ``source`` omitted for the pooled
  score): it fails when candidate - reference lies outside ``[min_change, max_change]`` or the candidate outside
  ``[min_value, max_value]`` (each bound optional, at least one given). The bounds are inclusive; with
  ``exclusive_bounds: true`` each is strict, so ``max_change: 0`` then requires the candidate strictly below the
  reference.
- ``slot_logprob_difference``: two probes of one spec, per prompt the candidate's teacher-forced log-probability of
  ``token_id`` at the prompt's end minus the reference's, less the same difference for ``drift_token_id`` when given
  (a difference in differences that removes drift shared by every untrained row): it fails unless at least
  ``min_prompts`` prompts (every prompt by default) reach ``min_difference``, no prompt exceeds ``max_difference``,
  and the median lies within ``[min_median, max_median]`` (each optional, at least one given).
- ``emission_count``: one probe's generations of a kind (``greedy``, ``sample`` or ``all``): it fails when
  ``token_id`` occurs, as the first generated id (``position: first``) or anywhere, more than ``max_count`` times.
- ``probe_identity``: one probe's record of what it measured: it fails unless every dotted path of ``expect`` (into
  the results document, e.g. ``model.iteration`` or ``model.megatron_run_config.checkpoint.save``) holds the value
  given.
- ``probe_agreement``: several probes' records: it fails unless every dotted path of ``fields`` (e.g.
  ``tokenizer.json_sha256``) is present in every one of ``probes`` and holds one value in all of them.

Each gate's outcome is PASS, FAIL or NOT EVALUATED (``gate_outcome``). NOT EVALUATED means a score could not
be read or lacks a field, the memory summary covers a different number of ranks than the run's GPUs, the
two speed scores were taken over different windows or GPU counts, or the two first-loss runs' first logged
iterations differ or the reference logged no lm loss there, or a band report holds other than one candidate or
fewer than two spans of ``rise_windows`` windows, or a log, probe or value cannot be read, two paired logs cover
different iterations, two probes ran different specs, prompts or prompt ids or code at different revisions, or a
probe did not count or score the id a gate names. It also means a value a gate read is not a finite number: a NaN
compares false with every bound, so a rule that refuses only what lies outside its bounds would pass it. Every
outcome carries a line stating the measurement or the reason. The exit status is
``gate_outcome.exit_status``'s, unless the spec has a ``verdict``: an ordered list of stages, each
``{stage, on_fail, gates}`` with ``on_fail`` FAIL or INCONCLUSIVE, that together hold every gate once. Then the first
stage whose gates do not all pass decides (``gate_outcome.ordered_verdict``) and the exit status is the verdict's: 0
PASS, 1 FAIL, 2 INCONCLUSIVE. ``--gate`` evaluates the gates it names alone, without the verdict.

USAGE
    python scripts/telemetry/score_gate.py --spec score_gate.yaml --scores-dir DIR [--gate NAME ...] [--json]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


# Run as a script, only scripts/telemetry/ is on sys.path; the repo root makes the sibling modules
# importable the same way from every entry point.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from scripts.mapping_keys import require_keys  # noqa: E402
from scripts.telemetry.gate_outcome import (  # noqa: E402
    FAIL,
    NOT_EVALUATED,
    PASS,
    STAGE_FAILURES,
    VERDICT_EXIT_STATUS,
    Stage,
    exit_status,
    ordered_verdict,
)
from scripts.telemetry.loss_parity import (  # noqa: E402
    VERDICT_METRIC,
    BandReport,
    offset_rise,
    offsets_from_reference_mean,
)
from scripts.telemetry.training_log import (  # noqa: E402
    TOKEN_MASKING_COUNT_FIELDS,
    TOKEN_MASKING_COUNT_PREFIX,
    TOKEN_MASKING_COUNTS_TAG,
    TOKEN_MASKING_TAG,
    IterationValues,
    NodeBanner,
    TokenMaskingCountsRecord,
    parse_iteration_records,
    parse_iteration_values,
    parse_node_banners,
    parse_token_masking_counts,
    parse_validation_records,
    read_log_lines,
    window_records,
)

from pipeline_coherence_test import GREEDY, PROBE_FORMAT, SAMPLE  # noqa: E402


MEMORY, SPEED, FIRST_LOSS, LOSS_SHIFT = "memory", "speed", "first_loss", "loss_shift"
LOG_PAIRING, MASKING_LOG, VALUE_CHANGE = "log_pairing", "masking_log", "value_change"
SLOT_LOGPROB_DIFFERENCE, EMISSION_COUNT, PROBE_IDENTITY = "slot_logprob_difference", "emission_count", "probe_identity"
PROBE_AGREEMENT = "probe_agreement"
KINDS = (
    MEMORY,
    SPEED,
    FIRST_LOSS,
    LOSS_SHIFT,
    LOG_PAIRING,
    MASKING_LOG,
    VALUE_CHANGE,
    SLOT_LOGPROB_DIFFERENCE,
    EMISSION_COUNT,
    PROBE_IDENTITY,
    PROBE_AGREEMENT,
)
VERDICT = "verdict"
ALL_GENERATIONS = "all"
GENERATION_KINDS = {GREEDY: (GREEDY,), SAMPLE: (SAMPLE,), ALL_GENERATIONS: (GREEDY, SAMPLE)}
EMISSION_POSITIONS = ("first", "anywhere")


@dataclass(frozen=True)
class MemoryGate:
    """One run's peak memory over every rank against a limit on allocated memory and on allocator retries."""

    name: str
    score: str
    max_allocated_gb: float
    max_alloc_retries: int


@dataclass(frozen=True)
class SpeedGate:
    """A candidate's step time relative to a reference on the same nodes, projected by the reference's own."""

    name: str
    candidate: str
    reference: str
    reference_s_per_iter: float
    go_up_to_s: float
    report_up_to_s: float


@dataclass(frozen=True)
class FirstLossGate:
    """Two runs' lm loss at their first logged iteration, from the same weights on the same first batch."""

    name: str
    candidate: str
    reference: str
    tolerance: float


@dataclass(frozen=True)
class LossShiftGate:
    """A band report's candidate: its lm-loss offset from the references' mean bounded per window, and its rise."""

    name: str
    report: str
    offset_low: float
    offset_high: float
    rise_windows: int
    max_rise: float


@dataclass(frozen=True)
class LogPairingGate:
    """Two training logs' reports at every iteration, each listed metric within its tolerance."""

    name: str
    candidate: str
    reference: str
    metrics: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class MaskingLogGate:
    """One training log's token-masking banner and per-iteration invariants, masking enabled or not."""

    name: str
    log: str
    enabled: bool
    token_ids: tuple[int, ...]
    nodes: int
    iterations: int


@dataclass(frozen=True)
class LogValue:
    """An evaluation result a training log printed after the iteration that brought the run to ``validation_step``."""

    log: str
    validation_step: int
    metric: str

    def describe(self) -> str:
        return f"{self.log} {self.metric} at step {self.validation_step}"


@dataclass(frozen=True)
class ProbeDocumentValue:
    """A probe's document score: pooled over its sources, or one source's."""

    probe: str
    documents: str
    source: str | None

    def describe(self) -> str:
        return f"{self.probe} documents {self.source or 'pooled'} {self.documents}"


@dataclass(frozen=True)
class ValueChangeGate:
    """A candidate value against a reference value: bounds on the change and on the candidate, each inclusive unless
    ``exclusive_bounds``."""

    name: str
    candidate: LogValue | ProbeDocumentValue
    reference: LogValue | ProbeDocumentValue
    min_change: float | None
    max_change: float | None
    min_value: float | None
    max_value: float | None
    exclusive_bounds: bool = False


@dataclass(frozen=True)
class SlotLogprobDifferenceGate:
    """Two probes' per-prompt teacher-forced log-probabilities of an id, differenced, against bounds."""

    name: str
    candidate: str
    reference: str
    token_id: int
    drift_token_id: int | None
    min_difference: float | None
    min_prompts: int | None
    max_difference: float | None
    min_median: float | None
    max_median: float | None


@dataclass(frozen=True)
class EmissionCountGate:
    """One probe's emissions of an id in its generations of a kind, against a limit."""

    name: str
    probe: str
    token_id: int
    generations: str
    position: str
    max_count: int


@dataclass(frozen=True)
class ProbeIdentityGate:
    """One probe's record of what it measured, against expected values at dotted paths."""

    name: str
    probe: str
    expect: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class ProbeAgreementGate:
    """Several probes' records, which must hold one value at each of the dotted paths ``fields``."""

    name: str
    probes: tuple[str, ...]
    fields: tuple[str, ...]


ScoreGate = (
    MemoryGate
    | SpeedGate
    | FirstLossGate
    | LossShiftGate
    | LogPairingGate
    | MaskingLogGate
    | ValueChangeGate
    | SlotLogprobDifferenceGate
    | EmissionCountGate
    | ProbeIdentityGate
    | ProbeAgreementGate
)


@dataclass(frozen=True)
class VerdictStage:
    """One stage of a spec's ordered verdict: its gates, and the verdict their failure gives."""

    name: str
    on_fail: str
    gates: tuple[str, ...]


@dataclass(frozen=True)
class ScoreGateResult:
    """One gate's kind, its outcome and the line stating its measurement or the reason it was not evaluated."""

    gate: str
    kind: str
    outcome: str
    detail: str


def _named_problems(*groups: tuple[str, list[str]]) -> str:
    """The non-empty groups as ``<label> [names]``, joined; empty when every group is."""
    return "; ".join(f"{label} {names}" for label, names in groups if names)


def _optional_float(gate: dict[str, Any], key: str) -> float | None:
    return None if gate.get(key) is None else float(gate[key])


def _check_bounds(where: str, bounds: dict[str, float | None], exclusive: bool = False) -> None:
    """Refuse a gate with no bound, or with a lower bound above its upper bound (``min_<x>`` against ``max_<x>``), or
    at it when the bounds are exclusive: no value could pass either."""
    if all(value is None for value in bounds.values()):
        raise ValueError(f"{where} states none of {sorted(bounds)}")
    for key, low in bounds.items():
        high = bounds.get(f"max_{key.removeprefix('min_')}") if key.startswith("min_") else None
        if low is not None and high is not None and (low > high or (exclusive and low == high)):
            raise ValueError(f"{where} has {key} {low} above its maximum {high}, or at it with exclusive bounds")


def _token_id(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where} must be a token id, not {value!r}")
    return value


def _memory_gate(where: str, name: str, gate: dict[str, Any]) -> MemoryGate:
    gate = require_keys(gate, where, {"score", "max_allocated_gb", "max_alloc_retries"})
    return MemoryGate(name, gate["score"], float(gate["max_allocated_gb"]), int(gate["max_alloc_retries"]))


def _speed_gate(where: str, name: str, gate: dict[str, Any]) -> SpeedGate:
    gate = require_keys(
        gate, where, {"candidate", "reference", "reference_s_per_iter", "go_up_to_s", "report_up_to_s"}
    )
    if gate["candidate"] == gate["reference"]:
        raise ValueError(f"{where} compares {gate['candidate']} with itself")
    speed = SpeedGate(
        name,
        gate["candidate"],
        gate["reference"],
        float(gate["reference_s_per_iter"]),
        float(gate["go_up_to_s"]),
        float(gate["report_up_to_s"]),
    )
    if speed.go_up_to_s > speed.report_up_to_s:
        raise ValueError(f"{where} has go_up_to_s above report_up_to_s")
    return speed


def _first_loss_gate(where: str, name: str, gate: dict[str, Any]) -> FirstLossGate:
    gate = require_keys(gate, where, {"candidate", "reference", "tolerance"})
    if gate["candidate"] == gate["reference"]:
        raise ValueError(f"{where} compares {gate['candidate']} with itself")
    return FirstLossGate(name, gate["candidate"], gate["reference"], float(gate["tolerance"]))


def _loss_shift_gate(where: str, name: str, gate: dict[str, Any]) -> LossShiftGate:
    gate = require_keys(gate, where, {"report", "offset_low", "offset_high", "rise_windows", "max_rise"})
    shift = LossShiftGate(
        name,
        gate["report"],
        float(gate["offset_low"]),
        float(gate["offset_high"]),
        int(gate["rise_windows"]),
        float(gate["max_rise"]),
    )
    if shift.offset_low > shift.offset_high or shift.rise_windows < 1:
        raise ValueError(f"{where} has an empty offset range or no rise window")
    return shift


def _log_pairing_gate(where: str, name: str, gate: dict[str, Any]) -> LogPairingGate:
    gate = require_keys(gate, where, {"candidate", "reference", "metrics"})
    metrics = gate["metrics"]
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError(f"{where}: metrics must map each metric to its tolerance")
    tolerances = tuple((metric, float(tolerance)) for metric, tolerance in metrics.items())
    if any(not tolerance >= 0 for _, tolerance in tolerances):
        raise ValueError(f"{where}: a tolerance is negative or not a number: {metrics}")
    for metric, tolerance in tolerances:
        if metric.startswith(TOKEN_MASKING_COUNT_PREFIX):
            if metric.removeprefix(TOKEN_MASKING_COUNT_PREFIX) not in TOKEN_MASKING_COUNT_FIELDS:
                raise ValueError(
                    f"{where}: {metric} is no token-masking count; the counts are "
                    f"{[TOKEN_MASKING_COUNT_PREFIX + field for field in TOKEN_MASKING_COUNT_FIELDS]}"
                )
            if tolerance != 0:
                raise ValueError(f"{where}: {metric} is an exact count, so its tolerance must be 0, not {tolerance}")
    return LogPairingGate(name, gate["candidate"], gate["reference"], tolerances)


def _masking_log_gate(where: str, name: str, gate: dict[str, Any]) -> MaskingLogGate:
    gate = require_keys(gate, where, {"log", "enabled", "token_ids", "nodes", "iterations"})
    if not isinstance(gate["enabled"], bool):
        raise ValueError(f"{where}: enabled must be true or false, not {gate['enabled']!r}")
    token_ids = gate["token_ids"]
    if not isinstance(token_ids, list) or not token_ids or len(set(token_ids)) != len(token_ids):
        raise ValueError(f"{where}: token_ids must be a non-empty list of distinct ids, not {token_ids!r}")
    nodes, iterations = int(gate["nodes"]), int(gate["iterations"])
    if nodes < 1 or iterations < 1:
        raise ValueError(f"{where}: nodes and iterations must be positive")
    ids = tuple(_token_id(token_id, f"{where}: token_ids") for token_id in token_ids)
    return MaskingLogGate(name, gate["log"], gate["enabled"], ids, nodes, iterations)


def _value_source(raw: Any, where: str) -> LogValue | ProbeDocumentValue:
    if isinstance(raw, dict) and "log" in raw:
        raw = require_keys(raw, where, {"log", "validation_step", "metric"})
        return LogValue(raw["log"], int(raw["validation_step"]), raw["metric"])
    raw = require_keys(raw, where, {"probe", "documents"}, frozenset({"source"}))
    return ProbeDocumentValue(raw["probe"], raw["documents"], raw.get("source"))


def _value_change_gate(where: str, name: str, gate: dict[str, Any]) -> ValueChangeGate:
    bounds = ("min_change", "max_change", "min_value", "max_value")
    gate = require_keys(gate, where, {"candidate", "reference"}, frozenset({*bounds, "exclusive_bounds"}))
    exclusive = gate.get("exclusive_bounds", False)
    if not isinstance(exclusive, bool):
        raise ValueError(f"{where}: exclusive_bounds must be true or false, not {exclusive!r}")
    limits = {key: _optional_float(gate, key) for key in bounds}
    _check_bounds(where, limits, exclusive)
    candidate = _value_source(gate["candidate"], f"{where}.candidate")
    reference = _value_source(gate["reference"], f"{where}.reference")
    if candidate == reference:
        raise ValueError(f"{where} compares {candidate.describe()} with itself")
    return ValueChangeGate(name, candidate, reference, **limits, exclusive_bounds=exclusive)


def _slot_logprob_difference_gate(where: str, name: str, gate: dict[str, Any]) -> SlotLogprobDifferenceGate:
    bounds = ("min_difference", "max_difference", "min_median", "max_median")
    gate = require_keys(
        gate, where, {"candidate", "reference", "token_id"}, frozenset({"drift_token_id", "min_prompts", *bounds})
    )
    limits = {key: _optional_float(gate, key) for key in bounds}
    _check_bounds(where, limits)
    if gate["candidate"] == gate["reference"]:
        raise ValueError(f"{where} compares {gate['candidate']} with itself")
    token_id = _token_id(gate["token_id"], f"{where}.token_id")
    drift = gate.get("drift_token_id")
    if drift is not None and _token_id(drift, f"{where}.drift_token_id") == token_id:
        raise ValueError(f"{where}: drift_token_id is token_id")
    min_prompts = gate.get("min_prompts")
    if min_prompts is not None and (limits["min_difference"] is None or int(min_prompts) < 1):
        raise ValueError(f"{where}: min_prompts counts prompts reaching min_difference, so needs it, and is >= 1")
    return SlotLogprobDifferenceGate(
        name,
        gate["candidate"],
        gate["reference"],
        token_id,
        drift,
        min_prompts=None if min_prompts is None else int(min_prompts),
        **limits,
    )


def _emission_count_gate(where: str, name: str, gate: dict[str, Any]) -> EmissionCountGate:
    gate = require_keys(gate, where, {"probe", "token_id", "generations", "position", "max_count"})
    if gate["generations"] not in GENERATION_KINDS or gate["position"] not in EMISSION_POSITIONS:
        raise ValueError(
            f"{where}: generations must be one of {sorted(GENERATION_KINDS)} and position one of {EMISSION_POSITIONS}"
        )
    if int(gate["max_count"]) < 0:
        raise ValueError(f"{where}: max_count is negative")
    return EmissionCountGate(
        name,
        gate["probe"],
        _token_id(gate["token_id"], f"{where}.token_id"),
        gate["generations"],
        gate["position"],
        int(gate["max_count"]),
    )


def _probe_identity_gate(where: str, name: str, gate: dict[str, Any]) -> ProbeIdentityGate:
    gate = require_keys(gate, where, {"probe", "expect"})
    if not isinstance(gate["expect"], dict) or not gate["expect"]:
        raise ValueError(f"{where}: expect must map dotted paths to values")
    return ProbeIdentityGate(name, gate["probe"], tuple(gate["expect"].items()))


def _probe_agreement_gate(where: str, name: str, gate: dict[str, Any]) -> ProbeAgreementGate:
    gate = require_keys(gate, where, {"probes", "fields"})
    probes, fields = gate["probes"], gate["fields"]
    if not isinstance(probes, list) or len(set(probes)) < 2 or len(set(probes)) != len(probes):
        raise ValueError(f"{where}: probes must list two or more distinct probe files, not {probes!r}")
    if not isinstance(fields, list) or not fields or not all(isinstance(field, str) and field for field in fields):
        raise ValueError(f"{where}: fields must be a non-empty list of dotted paths, not {fields!r}")
    return ProbeAgreementGate(name, tuple(probes), tuple(fields))


_GATE_PARSERS = {
    MEMORY: _memory_gate,
    SPEED: _speed_gate,
    FIRST_LOSS: _first_loss_gate,
    LOSS_SHIFT: _loss_shift_gate,
    LOG_PAIRING: _log_pairing_gate,
    MASKING_LOG: _masking_log_gate,
    VALUE_CHANGE: _value_change_gate,
    SLOT_LOGPROB_DIFFERENCE: _slot_logprob_difference_gate,
    EMISSION_COUNT: _emission_count_gate,
    PROBE_IDENTITY: _probe_identity_gate,
    PROBE_AGREEMENT: _probe_agreement_gate,
}


def load_score_gates(path: Path) -> dict[str, ScoreGate]:
    """Read a score-gate spec into its gates by name, refusing one whose gates could not be evaluated as written.

    Every kind's gate goes through its parser in ``_GATE_PARSERS``, which refuses it (ValueError) when it is not a
    mapping or has a missing or unknown field, since a misspelt threshold would otherwise not be applied. Raises
    ValueError as well on an unknown kind, a spec with no gates, a gate name used twice, a gate whose candidate is its
    reference, a speed gate whose go limit exceeds its report limit, a loss-shift gate whose offset range is empty or
    whose rise spans no window, a gate with no bound or a lower bound above its upper one (or at it, with exclusive
    bounds), a token-masking count paired with a tolerance other than 0, or fewer than two probes to agree.
    """
    raw = yaml.safe_load(Path(path).read_text())
    unknown = sorted(set(raw) - set(KINDS) - {VERDICT})
    if unknown:
        raise ValueError(f"{path}: unknown gate kinds {unknown}")
    entries = [(kind, name, gate) for kind in KINDS for name, gate in (raw.get(kind) or {}).items()]
    gates: dict[str, ScoreGate] = {}
    for kind, name, gate in entries:
        if name in gates:
            raise ValueError(f"{path}: gate {name} is defined twice")
        gates[name] = _GATE_PARSERS[kind](f"{path}: {kind} gate {name}", name, gate)
    if not gates:
        raise ValueError(f"{path}: defines no gates")
    return gates


def load_verdict(path: Path, gates: dict[str, ScoreGate]) -> tuple[VerdictStage, ...] | None:
    """Read a score-gate spec's ordered verdict stages; None when it has none.

    Raises ValueError on stages that are not a non-empty list of ``{stage, on_fail, gates}``, a repeated stage name,
    an ``on_fail`` other than FAIL or INCONCLUSIVE, a stage without gates, or gates that are not every gate of the
    spec, each in exactly one stage.
    """
    raw = yaml.safe_load(Path(path).read_text()).get(VERDICT)
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: {VERDICT} must be a non-empty list of stages")
    stages = []
    for index, item in enumerate(raw):
        where = f"{path}: {VERDICT} stage {index}"
        item = require_keys(item, where, {"stage", "on_fail", "gates"})
        if item["on_fail"] not in STAGE_FAILURES:
            raise ValueError(f"{where}: on_fail must be one of {STAGE_FAILURES}, not {item['on_fail']!r}")
        if not isinstance(item["gates"], list) or not item["gates"]:
            raise ValueError(f"{where}: gates must be a non-empty list of gate names")
        stages.append(VerdictStage(str(item["stage"]), item["on_fail"], tuple(item["gates"])))
    names = [stage.name for stage in stages]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: {VERDICT} stage names repeat: {names}")
    staged = [gate for stage in stages for gate in stage.gates]
    problems = _named_problems(
        ("unknown", sorted(set(staged) - set(gates))),
        ("in no stage", sorted(set(gates) - set(staged))),
        ("in several", sorted({gate for gate in staged if staged.count(gate) > 1})),
    )
    if problems:
        raise ValueError(f"{path}: {VERDICT} stages must hold every gate once: {problems}")
    return tuple(stages)


def _read_score(scores_dir: Path, name: str) -> dict[str, Any]:
    return json.loads((scores_dir / name).read_text())


def _finite(value: Any, where: str) -> float:
    """``value`` as a float; raises ValueError naming ``where`` when it is not a number or not finite.

    A NaN compares false with every bound, so a rule that refuses only what lies outside its bounds would pass it: a
    gate that reads one is NOT EVALUATED, never PASS. (``json.loads`` reads ``NaN`` and ``Infinity``, and a log prints
    ``nan``.)
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} is {value!r}, not a number")
    if not math.isfinite(value):
        raise ValueError(f"{where} is {value!r}, not a finite number")
    return float(value)


def _count(value: Any, where: str) -> int:
    """``value`` as a count; raises ValueError naming ``where`` unless it is a non-negative integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where} is {value!r}, not a count")
    return value


def _read_probe(scores_dir: Path, name: str) -> dict[str, Any]:
    probe = json.loads((scores_dir / name).read_text())
    if not isinstance(probe, dict) or probe.get("format") != PROBE_FORMAT:
        raise ValueError(f"{name} is not a {PROBE_FORMAT} results file")
    return probe


def evaluate_memory(gate: MemoryGate, scores_dir: Path) -> ScoreGateResult:
    """The memory gate's outcome on its run's score (see the module docstring)."""
    try:
        score = _read_score(scores_dir, gate.score)
        peak = score["peak_memory_across_ranks"]
        if peak is None:
            reason = f"{gate.score} has no peak memory over all ranks: the training loop was cut short"
            return ScoreGateResult(gate.name, MEMORY, FAIL, reason)
        ranks, gpus = peak["ranks"], score["num_gpus"]
        allocated = _finite(peak["max_allocated_gb"], f"{gate.score} max_allocated_gb")
        rank = peak["max_allocated_rank"]
        retries = _count(peak["max_alloc_retries"], f"{gate.score} max_alloc_retries")
        reserved = _finite(peak["max_reserved_gb"], f"{gate.score} max_reserved_gb")
    except Exception as error:  # noqa: BLE001 - every way the score cannot be read is NOT EVALUATED, never a FAIL
        return ScoreGateResult(gate.name, MEMORY, NOT_EVALUATED, f"{type(error).__name__}: {error}")
    if ranks != gpus:
        reason = f"the summary covers {ranks} ranks, not the run's {gpus} GPUs"
        return ScoreGateResult(gate.name, MEMORY, NOT_EVALUATED, reason)
    measured = (
        f"max allocated {allocated:.3f} GB (rank {rank}), limit {gate.max_allocated_gb}; "
        f"max allocator retries {retries}, limit {gate.max_alloc_retries}; max reserved {reserved:.3f} GB, reported"
    )
    within = allocated <= gate.max_allocated_gb and retries <= gate.max_alloc_retries
    return ScoreGateResult(gate.name, MEMORY, PASS if within else FAIL, measured)


def evaluate_speed(gate: SpeedGate, scores_dir: Path) -> ScoreGateResult:
    """The speed gate's outcome on its candidate's and reference's scores (see the module docstring)."""
    try:
        candidate = _read_score(scores_dir, gate.candidate)
        reference = _read_score(scores_dir, gate.reference)
        placement = [
            (score["window_first"], score["window_last"], score["num_gpus"]) for score in (candidate, reference)
        ]
        candidate_s = _finite(candidate["mean_step_s"], f"{gate.candidate} mean_step_s")
        reference_s = _finite(reference["mean_step_s"], f"{gate.reference} mean_step_s")
        if not reference_s > 0:
            raise ValueError(f"{gate.reference} mean_step_s is {reference_s}, not a positive step time")
    except Exception as error:  # noqa: BLE001 - every way the scores cannot be read is NOT EVALUATED, never a FAIL
        return ScoreGateResult(gate.name, SPEED, NOT_EVALUATED, f"{type(error).__name__}: {error}")
    if placement[0] != placement[1]:
        reason = (
            f"the scores' (first, last, GPUs) differ: {gate.candidate} {placement[0]}, {gate.reference} {placement[1]}"
        )
        return ScoreGateResult(gate.name, SPEED, NOT_EVALUATED, reason)
    projected = candidate_s / reference_s * gate.reference_s_per_iter
    measured = (
        f"projected {projected:.3f} s/iter = {candidate_s:.3f} s / {reference_s:.3f} s x {gate.reference_s_per_iter} s"
    )
    if projected <= gate.go_up_to_s:
        return ScoreGateResult(gate.name, SPEED, PASS, f"{measured}: go")
    if projected <= gate.report_up_to_s:
        return ScoreGateResult(gate.name, SPEED, PASS, f"{measured}: go, and report it (above {gate.go_up_to_s} s)")
    return ScoreGateResult(gate.name, SPEED, FAIL, f"{measured}: above {gate.report_up_to_s} s")


def _first_logged(scores_dir: Path, name: str) -> tuple[int, float | None]:
    """The first logged iteration of the training log a score was read from, and its lm loss (None when absent)."""
    log_path = Path(_read_score(scores_dir, name)["log_path"])
    first = min(parse_iteration_records(read_log_lines(log_path)), key=lambda record: record.iteration)
    return first.iteration, first.lm_loss


def evaluate_first_loss(gate: FirstLossGate, scores_dir: Path) -> ScoreGateResult:
    """The first-loss gate's outcome on its two runs' training logs (see the module docstring)."""
    try:
        (candidate_iteration, candidate_loss), (reference_iteration, reference_loss) = (
            _first_logged(scores_dir, gate.candidate),
            _first_logged(scores_dir, gate.reference),
        )
    except Exception as error:  # noqa: BLE001 - every way the logs cannot be read is NOT EVALUATED, never a FAIL
        return ScoreGateResult(gate.name, FIRST_LOSS, NOT_EVALUATED, f"{type(error).__name__}: {error}")
    if candidate_iteration != reference_iteration:
        reason = f"the first logged iterations differ: {candidate_iteration} against {reference_iteration}"
        return ScoreGateResult(gate.name, FIRST_LOSS, NOT_EVALUATED, reason)
    if reference_loss is None:
        reason = f"{gate.reference} logged no lm loss at iteration {reference_iteration}"
        return ScoreGateResult(gate.name, FIRST_LOSS, NOT_EVALUATED, reason)
    if candidate_loss is None:
        reason = f"{gate.candidate} logged no lm loss at iteration {candidate_iteration}"
        return ScoreGateResult(gate.name, FIRST_LOSS, FAIL, reason)
    try:
        for name, loss in ((gate.candidate, candidate_loss), (gate.reference, reference_loss)):
            _finite(loss, f"{name}'s lm loss at iteration {candidate_iteration}")
    except ValueError as error:
        return ScoreGateResult(gate.name, FIRST_LOSS, NOT_EVALUATED, f"{type(error).__name__}: {error}")
    difference = abs(candidate_loss - reference_loss)
    measured = (
        f"iteration {candidate_iteration}: lm loss {candidate_loss:.6f} against {reference_loss:.6f}, "
        f"difference {difference:.6f}, tolerance {gate.tolerance}"
    )
    return ScoreGateResult(gate.name, FIRST_LOSS, PASS if difference <= gate.tolerance else FAIL, measured)


def evaluate_loss_shift(gate: LossShiftGate, scores_dir: Path) -> ScoreGateResult:
    """The loss-shift gate's outcome on its band report (see the module docstring)."""
    try:
        report = BandReport.from_dict(json.loads((scores_dir / gate.report).read_text()))
        (loss,) = [band for band in report.metrics if band.metric == VERDICT_METRIC]
    except Exception as error:  # noqa: BLE001 - every way the report cannot be read is NOT EVALUATED, never a FAIL
        return ScoreGateResult(gate.name, LOSS_SHIFT, NOT_EVALUATED, f"{type(error).__name__}: {error}")
    if len(report.verdicts) != 1:
        reason = f"{gate.report} holds {len(report.verdicts)} candidates, not one"
        return ScoreGateResult(gate.name, LOSS_SHIFT, NOT_EVALUATED, reason)
    offsets = offsets_from_reference_mean(loss.windows, 0)
    try:
        for window, offset in zip(loss.windows, offsets):
            _finite(offset, f"{gate.report}'s lm-loss offset in the window from {window.first}")
    except ValueError as error:
        return ScoreGateResult(gate.name, LOSS_SHIFT, NOT_EVALUATED, f"{type(error).__name__}: {error}")
    if len(offsets) < 2 * gate.rise_windows:
        reason = f"{gate.report} has {len(offsets)} windows, fewer than two spans of {gate.rise_windows}"
        return ScoreGateResult(gate.name, LOSS_SHIFT, NOT_EVALUATED, reason)
    (verdict,) = report.verdicts
    if verdict.mismatches:
        return ScoreGateResult(gate.name, LOSS_SHIFT, FAIL, "; ".join(verdict.mismatches))
    outside = [
        window.first
        for window, offset in zip(loss.windows, offsets)
        if not gate.offset_low <= offset <= gate.offset_high
    ]
    rise = offset_rise(offsets, gate.rise_windows)
    measured = (
        f"offsets {min(offsets):+.4f} to {max(offsets):+.4f} over {len(offsets)} windows of "
        f"{report.first}-{report.last}, range [{gate.offset_low:+.4f}, {gate.offset_high:+.4f}]; the last "
        f"{gate.rise_windows} windows {rise:+.4f} against the first, limit {gate.max_rise:+.4f}"
    )
    if outside:
        return ScoreGateResult(gate.name, LOSS_SHIFT, FAIL, f"{measured}: windows from {outside} outside the range")
    if rise > gate.max_rise:
        return ScoreGateResult(gate.name, LOSS_SHIFT, FAIL, f"{measured}: the offset grows")
    return ScoreGateResult(gate.name, LOSS_SHIFT, PASS, measured)


def _every_iteration(lines: list[str], log: str, last: int | None) -> tuple[list[IterationValues], int]:
    """A log's iteration values for iterations 1 to ``last`` (its own last iteration when None), each logged exactly
    once and none beyond; raises ValueError otherwise."""
    values = parse_iteration_values(lines)
    if not values:
        raise ValueError(f"{log} logs no iteration")
    last = max(record.iteration for record in values) if last is None else last
    beyond = sorted({record.iteration for record in values if record.iteration > last})
    if beyond:
        raise ValueError(f"{log} logs iterations beyond {last}: {beyond[:5]}")
    return window_records(values, (1, last), f"{log}: iterations"), last


def _counts_through(lines: list[str], log: str, last: int) -> dict[int, TokenMaskingCountsRecord]:
    """A log's ``[token-masking-counts]`` records by iteration, none beyond ``last``; raises ValueError otherwise, and
    on two different counts lines for one iteration."""
    counts = {record.iteration: record for record in parse_token_masking_counts(lines)}
    beyond = sorted(iteration for iteration in counts if iteration > last or iteration < 1)
    if beyond:
        raise ValueError(f"{log} prints [{TOKEN_MASKING_COUNTS_TAG}] for iterations outside 1-{last}: {beyond[:5]}")
    return counts


def _log_reports(scores_dir: Path, log: str, last: int | None) -> tuple[dict[int, dict[str, float]], int]:
    """Each iteration's reports, 1 to ``last`` (the log's own last iteration when None): the iteration line's values
    and the counts line's, as ``token_masking/count/<field>``. Raises ValueError unless every iteration is logged
    exactly once and none beyond, and on a counts line ``parse_token_masking_counts`` refuses."""
    lines = read_log_lines(scores_dir / log)
    records, last = _every_iteration(lines, log, last)
    counts = _counts_through(lines, log, last)
    reports = {}
    for record in records:
        reports[record.iteration] = dict(record.values)
        if record.iteration in counts:
            reports[record.iteration] |= counts[record.iteration].metrics()
    return reports, last


def evaluate_log_pairing(gate: LogPairingGate, scores_dir: Path) -> ScoreGateResult:
    """The log-pairing gate's outcome on its two training logs (see the module docstring)."""
    try:
        reference, last = _log_reports(scores_dir, gate.reference, None)
        candidate, _ = _log_reports(scores_dir, gate.candidate, last)
        for log, reports in ((gate.candidate, candidate), (gate.reference, reference)):
            for iteration, values in reports.items():
                for metric, _ in gate.metrics:
                    if metric in values:
                        _finite(values[metric], f"{log} {metric} at iteration {iteration}")
    except Exception as error:  # noqa: BLE001 - every way the logs cannot be read is NOT EVALUATED, never a FAIL
        return ScoreGateResult(gate.name, LOG_PAIRING, NOT_EVALUATED, f"{type(error).__name__}: {error}")
    iterations = range(1, last + 1)
    measured, problems = [], []
    for metric, tolerance in gate.metrics:
        absent = [i for i in iterations if metric not in candidate[i] or metric not in reference[i]]
        if absent:
            problems.append(f"{metric} is absent at {len(absent)} iterations, from {absent[0]}")
            continue
        worst, at = max((abs(candidate[i][metric] - reference[i][metric]), i) for i in iterations)
        measured.append(f"{metric} differs by at most {worst:.6g} (iteration {at}), tolerance {tolerance:g}")
        if worst > tolerance:
            problems.append(
                f"{metric} differs by {worst:.6g} at iteration {at} ({candidate[at][metric]:.10g} against "
                f"{reference[at][metric]:.10g}), above {tolerance:g}"
            )
    detail = f"{gate.candidate} against {gate.reference}, iterations 1-{last}: " + "; ".join(measured + problems)
    return ScoreGateResult(gate.name, LOG_PAIRING, FAIL if problems else PASS, detail)


def _banner_problems(gate: MaskingLogGate, banners: list[NodeBanner]) -> list[str]:
    problems = []
    hosts = {banner.host for banner in banners}
    if len(hosts) != gate.nodes:
        problems.append(
            f"{len(banners)} [{TOKEN_MASKING_TAG}] banners from {len(hosts)} hosts, not {gate.nodes} nodes"
        )
    expected = (str(gate.enabled).lower(), list(gate.token_ids) if gate.enabled else [], list(gate.token_ids))
    for banner in banners:
        try:
            stated = (
                banner.fields["enabled"],
                json.loads(banner.fields["token_ids"]),
                json.loads(banner.fields["measured_token_ids"]),
            )
        except (KeyError, json.JSONDecodeError) as error:
            problems.append(f"the banner of {banner.host} cannot be read: {type(error).__name__}: {error}")
            break
        if stated != expected:
            problems.append(
                f"the banner of {banner.host} states enabled={stated[0]} token_ids={stated[1]} "
                f"measured_token_ids={stated[2]}, not enabled={expected[0]} token_ids={expected[1]} "
                f"measured_token_ids={expected[2]}"
            )
            break
    return problems


def evaluate_masking_log(gate: MaskingLogGate, scores_dir: Path) -> ScoreGateResult:
    """The masking-log gate's outcome on its training log (see the module docstring)."""
    try:
        lines = read_log_lines(scores_dir / gate.log)
        banners = parse_node_banners(lines, TOKEN_MASKING_TAG)
        _every_iteration(lines, gate.log, gate.iterations)
        counts = _counts_through(lines, gate.log, gate.iterations)
    except Exception as error:  # noqa: BLE001 - every way the log cannot be read is NOT EVALUATED, never a FAIL
        return ScoreGateResult(gate.name, MASKING_LOG, NOT_EVALUATED, f"{type(error).__name__}: {error}")
    problems = _banner_problems(gate, banners)
    iterations = range(1, gate.iterations + 1)
    absent = [iteration for iteration in iterations if iteration not in counts]
    if absent:
        problems.append(
            f"the [{TOKEN_MASKING_COUNTS_TAG}] line is absent at {len(absent)} iterations, from {absent[0]}"
        )
    reports = [counts[iteration] for iteration in iterations if iteration in counts]
    if gate.enabled:
        rule = "masked == listed_trainable and trained_listed == 0"
        broken = [c for c in reports if c.masked != c.listed_trainable or c.trained_listed != 0]
    else:
        rule = "masked == 0 and trained_listed == listed_trainable"
        broken = [c for c in reports if c.masked != 0 or c.trained_listed != c.listed_trainable]
    if broken:
        first = broken[0]
        problems.append(
            f"{rule} fails at {len(broken)} iterations, from {first.iteration} (masked={first.masked} "
            f"trained_listed={first.trained_listed} listed_trainable={first.listed_trainable})"
        )
    listed = [c.listed_trainable for c in reports]
    if not any(listed):
        problems.append("no iteration held a trainable target of the measured ids")
    measured = (
        f"{len(banners)} banners from {len({banner.host for banner in banners})} hosts; iterations 1-{gate.iterations}: "
        f"{rule}; {sum(listed)} trainable listed targets of {sum(c.positions for c in reports)} target positions, "
        f"{min(listed, default=0)} to {max(listed, default=0)} per iteration"
    )
    detail = measured + ("; " + "; ".join(problems) if problems else "")
    return ScoreGateResult(gate.name, MASKING_LOG, FAIL if problems else PASS, detail)


def _read_value(source: LogValue | ProbeDocumentValue, scores_dir: Path) -> float:
    """The value a source names; raises when it cannot be read, is not exactly one value, or is not a number."""
    if isinstance(source, LogValue):
        matches = [
            record.values[source.metric]
            for record in parse_validation_records(read_log_lines(scores_dir / source.log))
            if record.step == source.validation_step and source.metric in record.values
        ]
        if len(matches) != 1:
            raise LookupError(f"{source.describe()}: the log holds {len(matches)} such results, not one")
        return _finite(matches[0], source.describe())
    documents = _read_probe(scores_dir, source.probe)["documents"]
    if documents is None:
        raise LookupError(f"{source.probe} scored no documents")
    scores = documents["pooled"] if source.source is None else documents["sources"][source.source]
    return _finite(scores[source.documents], source.describe())


def _outside(value: float, low: float | None, high: float | None, exclusive: bool = False) -> bool:
    """Whether ``value`` lies outside the bounds (each optional), a NaN always outside."""
    if exclusive:
        return (low is not None and not value > low) or (high is not None and not value < high)
    return (low is not None and not value >= low) or (high is not None and not value <= high)


def _range(low: float | None, high: float | None, exclusive: bool = False) -> str:
    opening, closing = ("(", ")") if exclusive else ("[", "]")
    return f"{opening}{'-inf' if low is None else f'{low:g}'}, {'inf' if high is None else f'{high:g}'}{closing}"


def evaluate_value_change(gate: ValueChangeGate, scores_dir: Path) -> ScoreGateResult:
    """The value-change gate's outcome on its two values (see the module docstring)."""
    try:
        candidate = _read_value(gate.candidate, scores_dir)
        reference = _read_value(gate.reference, scores_dir)
    except Exception as error:  # noqa: BLE001 - every way a value cannot be read is NOT EVALUATED, never a FAIL
        return ScoreGateResult(gate.name, VALUE_CHANGE, NOT_EVALUATED, f"{type(error).__name__}: {error}")
    change, exclusive = candidate - reference, gate.exclusive_bounds
    measured = (
        f"{gate.candidate.describe()} {candidate:.6f} against {gate.reference.describe()} {reference:.6f}: change "
        f"{change:+.6f} in {_range(gate.min_change, gate.max_change, exclusive)}, value in "
        f"{_range(gate.min_value, gate.max_value, exclusive)}"
    )
    failed = _outside(change, gate.min_change, gate.max_change, exclusive) or _outside(
        candidate, gate.min_value, gate.max_value, exclusive
    )
    return ScoreGateResult(gate.name, VALUE_CHANGE, FAIL if failed else PASS, measured)


def _slot_differences(gate: SlotLogprobDifferenceGate, scores_dir: Path) -> list[tuple[str, float]]:
    """Per prompt, the candidate's log-probability of the gate's id minus the reference's, less the same for its drift
    id; raises when the two probes ran different specs, prompts or prompt ids, or code at different revisions (or one
    does not record its revision), did not score an id, or scored one as other than a finite number."""
    candidate, reference = _read_probe(scores_dir, gate.candidate), _read_probe(scores_dir, gate.reference)
    if candidate["spec"]["sha256"] != reference["spec"]["sha256"]:
        raise ValueError(f"{gate.candidate} and {gate.reference} ran different probe specs")
    revisions = []
    for name, probe in ((gate.candidate, candidate), (gate.reference, reference)):
        if "code_revision" not in probe["run"]:
            raise LookupError(f"{name} does not record the revision of the code that measured it")
        revisions.append(probe["run"]["code_revision"])
    if revisions[0] != revisions[1]:
        raise ValueError(
            f"{gate.candidate} and {gate.reference} were measured by code at different revisions: {revisions[0]!r} "
            f"against {revisions[1]!r}"
        )
    if [p["id"] for p in candidate["prompts"]] != [p["id"] for p in reference["prompts"]]:
        raise ValueError(f"{gate.candidate} and {gate.reference} hold different prompts")
    pairs = list(zip(candidate["prompts"], reference["prompts"]))
    different = [ours["id"] for ours, theirs in pairs if ours["input_ids"] != theirs["input_ids"]]
    if different:
        raise ValueError(f"{gate.candidate} and {gate.reference} scored prompts {different} on different input ids")
    ids = [str(gate.token_id)] + ([] if gate.drift_token_id is None else [str(gate.drift_token_id)])
    differences = []
    for ours, theirs in pairs:
        shift = [
            _finite(ours["slot"]["logprob"][token_id], f"{gate.candidate} prompt {ours['id']} log p({token_id})")
            - _finite(theirs["slot"]["logprob"][token_id], f"{gate.reference} prompt {theirs['id']} log p({token_id})")
            for token_id in ids
        ]
        differences.append((ours["id"], shift[0] - sum(shift[1:])))
    return differences


def evaluate_slot_logprob_difference(gate: SlotLogprobDifferenceGate, scores_dir: Path) -> ScoreGateResult:
    """The slot-log-probability-difference gate's outcome on its two probes (see the module docstring)."""
    try:
        differences = _slot_differences(gate, scores_dir)
    except Exception as error:  # noqa: BLE001 - every way the probes cannot be read is NOT EVALUATED, never a FAIL
        return ScoreGateResult(gate.name, SLOT_LOGPROB_DIFFERENCE, NOT_EVALUATED, f"{type(error).__name__}: {error}")
    values = [difference for _, difference in differences]
    median = statistics.median(values)
    lowest, highest = min(differences, key=lambda item: item[1]), max(differences, key=lambda item: item[1])
    drift = "" if gate.drift_token_id is None else f" less that of {gate.drift_token_id}"
    measured = [
        f"{len(values)} prompts, log p({gate.token_id}){drift}, {gate.candidate} minus {gate.reference}: median "
        f"{median:+.3f}, lowest {lowest[1]:+.3f} ({lowest[0]}), highest {highest[1]:+.3f} ({highest[0]})"
    ]
    problems = []
    if gate.min_difference is not None:
        needed = len(values) if gate.min_prompts is None else gate.min_prompts
        reached = sum(value >= gate.min_difference for value in values)
        measured.append(f"{reached} prompts reach {gate.min_difference:+g}, {needed} needed")
        if reached < needed:
            problems.append(f"only {reached} prompts reach {gate.min_difference:+g}")
    if gate.max_difference is not None and highest[1] > gate.max_difference:
        problems.append(f"{highest[0]} exceeds {gate.max_difference:+g}")
    if _outside(median, gate.min_median, gate.max_median):
        problems.append(f"the median lies outside {_range(gate.min_median, gate.max_median)}")
    detail = "; ".join(measured + problems)
    return ScoreGateResult(gate.name, SLOT_LOGPROB_DIFFERENCE, FAIL if problems else PASS, detail)


def evaluate_emission_count(gate: EmissionCountGate, scores_dir: Path) -> ScoreGateResult:
    """The emission-count gate's outcome on its probe (see the module docstring)."""
    try:
        probe = _read_probe(scores_dir, gate.probe)
        if str(gate.token_id) not in probe["summary"]["emissions"]:
            raise LookupError(f"{gate.probe} did not count {gate.token_id}")
        kinds = GENERATION_KINDS[gate.generations]
        emitted = [
            (
                prompt["id"],
                generation["kind"],
                generation["index"],
                _count(
                    generation["counts"][str(gate.token_id)][gate.position],
                    f"{gate.probe} prompt {prompt['id']} {generation['kind']} {generation['index']} count",
                ),
            )
            for prompt in probe["prompts"]
            for generation in prompt["generations"]
            if generation["kind"] in kinds
        ]
        if not emitted:
            raise LookupError(f"{gate.probe} holds no {gate.generations} generations")
    except Exception as error:  # noqa: BLE001 - every way the probe cannot be read is NOT EVALUATED, never a FAIL
        return ScoreGateResult(gate.name, EMISSION_COUNT, NOT_EVALUATED, f"{type(error).__name__}: {error}")
    count = sum(found for *_, found in emitted)
    where = [f"{prompt} {kind} {index}" for prompt, kind, index, found in emitted if found]
    measured = (
        f"{gate.token_id} {gate.position} in {len(emitted)} {gate.generations} generations of {gate.probe}: {count}, "
        f"limit {gate.max_count}" + (f" (in {', '.join(where[:10])})" if where else "")
    )
    return ScoreGateResult(gate.name, EMISSION_COUNT, PASS if count <= gate.max_count else FAIL, measured)


def _at_path(document: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(document, dict) or part not in document:
            raise KeyError(path)
        document = document[part]
    return document


def evaluate_probe_identity(gate: ProbeIdentityGate, scores_dir: Path) -> ScoreGateResult:
    """The probe-identity gate's outcome on its probe (see the module docstring)."""
    try:
        probe = _read_probe(scores_dir, gate.probe)
    except Exception as error:  # noqa: BLE001 - every way the probe cannot be read is NOT EVALUATED, never a FAIL
        return ScoreGateResult(gate.name, PROBE_IDENTITY, NOT_EVALUATED, f"{type(error).__name__}: {error}")
    problems = []
    for path, expected in gate.expect:
        try:
            found = _at_path(probe, path)
        except KeyError:
            problems.append(f"{path} is absent")
            continue
        if found != expected:
            problems.append(f"{path} is {found!r}, not {expected!r}")
    detail = f"{gate.probe}: " + ("; ".join(problems) if problems else f"{len(gate.expect)} fields as expected")
    return ScoreGateResult(gate.name, PROBE_IDENTITY, FAIL if problems else PASS, detail)


def evaluate_probe_agreement(gate: ProbeAgreementGate, scores_dir: Path) -> ScoreGateResult:
    """The probe-agreement gate's outcome on its probes (see the module docstring)."""
    try:
        probes = {name: _read_probe(scores_dir, name) for name in gate.probes}
    except Exception as error:  # noqa: BLE001 - every way a probe cannot be read is NOT EVALUATED, never a FAIL
        return ScoreGateResult(gate.name, PROBE_AGREEMENT, NOT_EVALUATED, f"{type(error).__name__}: {error}")
    measured, problems = [], []
    for path in gate.fields:
        found = {}
        for name, probe in probes.items():
            try:
                found[name] = _at_path(probe, path)
            except KeyError:
                problems.append(f"{path} is absent from {name}")
        values = list(found.values())
        if any(value != values[0] for value in values):
            problems.append(f"{path} differs: " + ", ".join(f"{name} {value!r}" for name, value in found.items()))
        elif len(found) == len(probes):
            measured.append(f"{path} is {values[0]!r} in all")
    detail = f"{', '.join(gate.probes)}: " + "; ".join(measured + problems)
    return ScoreGateResult(gate.name, PROBE_AGREEMENT, FAIL if problems else PASS, detail)


_EVALUATORS = {
    MemoryGate: evaluate_memory,
    SpeedGate: evaluate_speed,
    FirstLossGate: evaluate_first_loss,
    LossShiftGate: evaluate_loss_shift,
    LogPairingGate: evaluate_log_pairing,
    MaskingLogGate: evaluate_masking_log,
    ValueChangeGate: evaluate_value_change,
    SlotLogprobDifferenceGate: evaluate_slot_logprob_difference,
    EmissionCountGate: evaluate_emission_count,
    ProbeIdentityGate: evaluate_probe_identity,
    ProbeAgreementGate: evaluate_probe_agreement,
}


def evaluate_score_gate(gate: ScoreGate, scores_dir: Path) -> ScoreGateResult:
    """Evaluate one gate of any kind on the files in ``scores_dir``."""
    return _EVALUATORS[type(gate)](gate, scores_dir)


def main(argv: list[str] | None = None) -> int:
    """Evaluate the requested gates (every gate by default), print each outcome and return the exit status: the
    ordered verdict's when the spec has one and every gate is evaluated, the gate set's otherwise."""
    parser = argparse.ArgumentParser(description="Evaluate pre-registered gates on runs' scores, logs and probes.")
    parser.add_argument("--spec", type=Path, required=True, help="The score-gate spec YAML")
    parser.add_argument(
        "--scores-dir",
        type=Path,
        required=True,
        help="The directory holding the files the gates read: scores, band reports, training logs, probe results",
    )
    parser.add_argument(
        "--gate", action="append", help="A gate to evaluate, without the verdict; repeatable (default: every gate)"
    )
    parser.add_argument("--json", action="store_true", help="Emit the outcomes (and any verdict) as JSON")
    args = parser.parse_args(argv)
    gates = load_score_gates(args.spec)
    stages = load_verdict(args.spec, gates)
    names = args.gate or list(gates)
    unknown = sorted(set(names) - set(gates))
    if unknown:
        parser.error(f"unknown gates {unknown}; the spec defines {sorted(gates)}")
    results = [evaluate_score_gate(gates[name], args.scores_dir) for name in names]
    if args.gate or stages is None:
        if args.json:
            print(json.dumps([asdict(result) for result in results], indent=2))
        else:
            for result in results:
                print(f"gate {result.gate} ({result.kind}): {result.outcome} ({result.detail})")
        return exit_status(result.outcome for result in results)
    outcomes = {result.gate: result.outcome for result in results}
    verdict, deciding = ordered_verdict(
        [Stage(stage.name, stage.on_fail, tuple(outcomes[gate] for gate in stage.gates)) for stage in stages]
    )
    if args.json:
        report = {"gates": [asdict(result) for result in results], "verdict": verdict, "deciding_stage": deciding}
        print(json.dumps(report, indent=2))
    else:
        by_name = {result.gate: result for result in results}
        for stage in stages:
            print(f"stage {stage.name} (on failure {stage.on_fail}):")
            for gate in stage.gates:
                result = by_name[gate]
                print(f"  gate {result.gate} ({result.kind}): {result.outcome} ({result.detail})")
        print(f"verdict: {verdict}" + (f" (decided by stage {deciding})" if deciding is not None else ""))
    return VERDICT_EXIT_STATUS[verdict]


if __name__ == "__main__":
    sys.exit(main())

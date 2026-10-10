# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Unit tests for scripts/telemetry/score_gate.py's token-masking and probe gates and its ordered verdict.

The training logs are built from the real token-masking canary's iteration line and the bridge's own banner and
counts line (``training_log_fixture.py``); the probe results are the real probe of the copy model
(``probe_fixtures.py``), with the numbers each case needs written into a copy. The cases mirror a masked arm, its measure-only control and the
untrained base they both start from; each case writes the whole experiment, then replaces the file it is about.
"""

import copy
import json
import math
import re
from pathlib import Path

import pytest
import yaml
from scripts.telemetry import score_gate as sg

from tests.unit_tests.probe_fixtures import REFERENCE_ID, copy_probe_results
from tests.unit_tests.token_masking_fixtures import MARKER_ID
from tests.unit_tests.training_log_fixture import (
    token_masking_banner,
    token_masking_counts_line,
    token_masking_iteration_line,
    validation_line,
    write_log,
)


M, REF = str(MARKER_ID), str(REFERENCE_ID)
LISTED = (0.01, 0.012, 0.009, 0.011)
POSITIONS = 100_000  # each iteration's target positions
LISTED_LOSS = "masked-validation/token_masking/listed_target_loss"
HOSTS = ("nid1", "nid2")
COUNTS_TAG = "[token-masking-counts]"


def counts(enabled: bool, listed_targets: int, positions: int = POSITIONS, **edits: int) -> dict[str, int]:
    """An iteration's counts when every listed target is trainable: masked when enabled, trained when not."""
    values = dict(
        listed=listed_targets,
        listed_trainable=listed_targets,
        masked=listed_targets if enabled else 0,
        trained_listed=0 if enabled else listed_targets,
        trainable=positions - listed_targets if enabled else positions,
        positions=positions,
    )
    return values | edits


def iteration_lines(
    enabled: bool,
    listed_losses: tuple[float, ...],
    listed: tuple[float, ...] = LISTED,
    count_edits: dict[int, dict[str, int]] | None = None,
) -> list[str]:
    """Iterations 1-4 of an arm whose targets other than the marker's average 2.0 nats (the control's lm loss also
    holds the marker's targets; a masked arm's does not): each iteration's counts line, the ``listed`` fraction of
    ``POSITIONS`` (with ``count_edits`` by iteration), then its iteration line."""
    lines = []
    for iteration, (fraction, loss) in enumerate(zip(listed, listed_losses), start=1):
        edits = (count_edits or {}).get(iteration, {})
        lines.append(token_masking_counts_line(iteration, **counts(enabled, round(fraction * POSITIONS), **edits)))
        lines.append(
            token_masking_iteration_line(
                iteration,
                enabled=enabled,
                listed=fraction,
                lm_loss=2.0 if enabled else 2.0 * (1 - fraction) + loss * fraction,
                listed_loss=loss if fraction > 0 else None,
            )
        )
    return lines


def write_arm_log(
    directory: Path,
    name: str,
    *,
    enabled: bool,
    iterations: list[str],
    validation: dict[int, float],
    banner_enabled: bool | None = None,
    hosts: tuple[str, ...] = HOSTS,
) -> None:
    """An arm's training log: a banner per host, its iteration lines and its masked-validation listed-target loss at
    the given steps."""
    stated = enabled if banner_enabled is None else banner_enabled
    banners = [token_masking_banner(stated, [MARKER_ID], host, rank * 4) for rank, host in enumerate(hosts)]
    evaluations = [validation_line(step, {LISTED_LOSS: value}) for step, value in validation.items()]
    write_log(directory, banners + iterations + evaluations, name)


def write_masked_log(directory: Path, iterations: list[str] | None = None, **arm) -> None:
    lines = iteration_lines(True, (19.0, 19.1, 19.2, 19.3)) if iterations is None else iterations
    write_arm_log(directory, "masked.log", enabled=True, iterations=lines, validation={0: 19.4, 4: 19.6}, **arm)


def write_control_log(directory: Path, iterations: list[str] | None = None) -> None:
    lines = iteration_lines(False, (19.0, 9.0, 4.0, 2.5)) if iterations is None else iterations
    write_arm_log(directory, "control.log", enabled=False, iterations=lines, validation={0: 19.4, 4: 2.4})


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    return copy_probe_results(tmp_path_factory.mktemp("copy_probe"))


def write_probe(
    directory: Path,
    name: str,
    probe: dict,
    *,
    marker: float,
    reference: float,
    marker_ce: float,
    non_marker_ce: float,
    emits: bool,
) -> None:
    """The copy model's real probe with every prompt's slot log-probabilities, the held-out scores and, unless
    ``emits``, the generations' marker counts replaced."""
    edited = copy.deepcopy(probe)
    for prompt in edited["prompts"]:
        prompt["slot"]["logprob"] = {M: marker, REF: reference}
        if not emits:
            for generation in prompt["generations"]:
                generation["counts"][M] = {"first": 0, "anywhere": 0}
    edited["held_out"]["scores"].update(marker_ce=marker_ce, non_marker_ce=non_marker_ce)
    (directory / name).write_text(json.dumps(edited))


def write_base(directory: Path, probe: dict, **edits) -> None:
    values = dict(marker=-19, reference=-20, marker_ce=19.4, non_marker_ce=2.3, emits=False) | edits
    write_probe(directory, "base.json", probe, **values)


def write_masked(directory: Path, probe: dict, **edits) -> None:
    values = dict(marker=-20, reference=-20.2, marker_ce=19.62, non_marker_ce=2.0, emits=False) | edits
    write_probe(directory, "masked.json", probe, **values)


def write_control(directory: Path, probe: dict, **edits) -> None:
    values = dict(marker=-0.5, reference=-20.1, marker_ce=2.45, non_marker_ce=2.01, emits=True) | edits
    write_probe(directory, "control.json", probe, **values)


def experiment(directory: Path, probe: dict) -> Path:
    """A masked arm that held its marker, a control that learned it and the base, as gate inputs in ``directory``."""
    write_masked_log(directory)
    write_control_log(directory)
    write_base(directory, probe)
    write_masked(directory, probe)
    write_control(directory, probe)
    return directory


GATES = {
    "log_pairing": {
        "listed_counts_identical": {
            "candidate": "masked.log",
            "reference": "control.log",
            "metrics": {
                "token_masking/count/listed": 0,
                "token_masking/count/listed_trainable": 0,
                "token_masking/count/positions": 0,
            },
        },
        "other_targets_agree": {
            "candidate": "masked.log",
            "reference": "control.log",
            "metrics": {"token_masking/non_listed_target_loss": 0.05},
        },
    },
    "masking_log": {
        "masked_arm": {"log": "masked.log", "enabled": True, "token_ids": [MARKER_ID], "nodes": 2, "iterations": 4},
        "control_arm": {"log": "control.log", "enabled": False, "token_ids": [MARKER_ID], "nodes": 2, "iterations": 4},
    },
    "probe_identity": {
        "masked_identity": {
            "probe": "masked.json",
            "expect": {"model.iteration": 3, "model.megatron_run_config.token_masking.enabled": True},
        },
    },
    "probe_agreement": {
        "one_tokenizer_one_code": {
            "probes": ["base.json", "masked.json", "control.json"],
            "fields": ["tokenizer.json_sha256", "spec.sha256", "run.code_revision"],
        },
    },
    "value_change": {
        "masked_evaluator_agrees": {
            "candidate": {"probe": "masked.json", "held_out": "marker_ce"},
            "reference": {"log": "masked.log", "validation_step": 4, "metric": LISTED_LOSS},
            "min_change": -0.1,
            "max_change": 0.1,
        },
        "control_learned_the_marker": {
            "candidate": {"probe": "control.json", "held_out": "marker_ce"},
            "reference": {"probe": "base.json", "held_out": "marker_ce"},
            "max_change": -10.0,
            "max_value": 3.0,
        },
        "masked_marker_ce_held": {
            "candidate": {"log": "masked.log", "validation_step": 4, "metric": LISTED_LOSS},
            "reference": {"log": "masked.log", "validation_step": 0, "metric": LISTED_LOSS},
            "min_change": -0.5,
        },
        "masked_learned_the_data": {
            "candidate": {"probe": "masked.json", "held_out": "non_marker_ce"},
            "reference": {"probe": "control.json", "held_out": "non_marker_ce"},
            "min_change": -0.05,
            "max_change": 0.05,
        },
    },
    "slot_logprob_difference": {
        "control_slots_rose": {
            "candidate": "control.json",
            "reference": "base.json",
            "token_id": MARKER_ID,
            "min_difference": 10.0,
            "min_prompts": 2,
        },
        "masked_slots_held": {
            "candidate": "masked.json",
            "reference": "base.json",
            "token_id": MARKER_ID,
            "drift_token_id": REFERENCE_ID,
            "max_median": 0.5,
            "max_difference": 2.0,
        },
        "control_above_masked": {
            "candidate": "control.json",
            "reference": "masked.json",
            "token_id": MARKER_ID,
            "min_median": 7.0,
        },
    },
    "emission_count": {
        "masked_never_greedy": {
            "probe": "masked.json",
            "token_id": MARKER_ID,
            "generations": "greedy",
            "position": "anywhere",
            "unit": "occurrences",
            "max_count": 0,
        },
    },
}
VERDICT = [
    {
        "stage": "integrity",
        "on_fail": "INCONCLUSIVE",
        "gates": [
            "listed_counts_identical",
            "other_targets_agree",
            "masked_arm",
            "control_arm",
            "masked_identity",
            "one_tokenizer_one_code",
            "masked_evaluator_agrees",
        ],
    },
    {
        "stage": "positive_control",
        "on_fail": "INCONCLUSIVE",
        "gates": ["control_learned_the_marker", "control_slots_rose"],
    },
    {
        "stage": "masking",
        "on_fail": "FAIL",
        "gates": ["masked_marker_ce_held", "masked_slots_held", "control_above_masked", "masked_never_greedy"],
    },
    {"stage": "data_learned", "on_fail": "FAIL", "gates": ["masked_learned_the_data"]},
]


def write_spec(directory: Path, gates: dict | None = None, verdict: list | None = VERDICT) -> Path:
    raw = dict(GATES if gates is None else gates)
    if verdict is not None:
        raw["verdict"] = verdict
    path = directory / "gate.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def only(kind: str, name: str, **edits) -> dict:
    """A spec of one gate of ``GATES``, its fields edited."""
    return {kind: {name: {**GATES[kind][name], **edits}}}


def judge(spec: Path, scores_dir: Path, capsys, *gates: str) -> tuple[int, dict | list]:
    status = sg.main(["--spec", str(spec), "--scores-dir", str(scores_dir), "--json", *[f"--gate={g}" for g in gates]])
    return status, json.loads(capsys.readouterr().out)


def outcomes(report: dict | list) -> dict[str, str]:
    gates = report["gates"] if isinstance(report, dict) else report
    return {result["gate"]: result["outcome"] for result in gates}


def one(spec: dict, directory: Path, capsys) -> tuple[int, str]:
    """Evaluate a spec of one gate; its exit status and the gate's detail."""
    status, report = judge(write_spec(directory, spec, verdict=None), directory, capsys)
    (result,) = report
    return status, result["detail"]


# --------------------------------------------------------------------------------------
# The ordered verdict
# --------------------------------------------------------------------------------------


def test_an_experiment_that_meets_every_stage_passes(tmp_path, probe, capsys):
    status, report = judge(write_spec(tmp_path), experiment(tmp_path, probe), capsys)
    assert set(outcomes(report).values()) == {"PASS"}, report
    assert (status, report["verdict"], report["deciding_stage"]) == (0, "PASS", None)


def test_a_broken_integrity_check_is_inconclusive_whatever_the_later_stages_show(tmp_path, probe, capsys):
    """The masked arm trained a listed target at iteration 3: the run can show neither that masking worked nor that
    it failed, so a masking failure further down is no verdict on masking."""
    experiment(tmp_path, probe)
    write_masked_log(tmp_path, iteration_lines(True, (19.0,) * 4, count_edits={3: {"trained_listed": 1}}))
    write_masked(tmp_path, probe, emits=True)
    status, report = judge(write_spec(tmp_path), tmp_path, capsys)
    assert outcomes(report)["masked_arm"] == "FAIL" and outcomes(report)["masked_never_greedy"] == "FAIL"
    assert (status, report["verdict"], report["deciding_stage"]) == (2, "INCONCLUSIVE", "integrity")


def test_a_control_that_did_not_learn_the_marker_is_inconclusive(tmp_path, probe, capsys):
    experiment(tmp_path, probe)
    write_control(tmp_path, probe, marker=-15, marker_ce=15.0)
    status, report = judge(write_spec(tmp_path), tmp_path, capsys)
    assert outcomes(report)["control_learned_the_marker"] == "FAIL"
    assert (status, report["verdict"], report["deciding_stage"]) == (2, "INCONCLUSIVE", "positive_control")


def test_a_masked_arm_that_emits_the_marker_fails(tmp_path, probe, capsys):
    experiment(tmp_path, probe)
    write_masked(tmp_path, probe, emits=True)
    status, report = judge(write_spec(tmp_path), tmp_path, capsys)
    assert {gate for gate, outcome in outcomes(report).items() if outcome != "PASS"} == {"masked_never_greedy"}
    assert (status, report["verdict"], report["deciding_stage"]) == (1, "FAIL", "masking")


def test_a_masked_arm_that_did_not_learn_the_data_fails(tmp_path, probe, capsys):
    experiment(tmp_path, probe)
    write_masked(tmp_path, probe, non_marker_ce=2.3)
    status, report = judge(write_spec(tmp_path), tmp_path, capsys)
    assert (status, report["verdict"], report["deciding_stage"]) == (1, "FAIL", "data_learned")


def test_a_missing_input_makes_the_first_stage_that_reads_it_inconclusive(tmp_path, probe, capsys):
    """The integrity stage's probe agreement reads every probe, so a missing base probe stops the verdict there."""
    (experiment(tmp_path, probe) / "base.json").unlink()
    status, report = judge(write_spec(tmp_path), tmp_path, capsys)
    assert outcomes(report)["one_tokenizer_one_code"] == outcomes(report)["control_slots_rose"] == "NOT EVALUATED"
    assert (status, report["verdict"], report["deciding_stage"]) == (2, "INCONCLUSIVE", "integrity")


def test_a_named_gate_is_evaluated_alone_without_the_verdict(tmp_path, probe, capsys):
    status, report = judge(write_spec(tmp_path), experiment(tmp_path, probe), capsys, "masked_never_greedy")
    assert status == 0 and outcomes(report) == {"masked_never_greedy": "PASS"}


def test_the_text_report_shows_each_stage_and_the_verdict(tmp_path, probe, capsys):
    status = sg.main(["--spec", str(write_spec(tmp_path)), "--scores-dir", str(experiment(tmp_path, probe))])
    printed = capsys.readouterr().out
    assert status == 0
    assert "stage integrity (on failure INCONCLUSIVE):" in printed and printed.rstrip().endswith("verdict: PASS")


@pytest.mark.parametrize(
    "verdict, message",
    [
        (VERDICT[:-1], "in no stage"),
        (VERDICT + [{"stage": "again", "on_fail": "FAIL", "gates": ["masked_learned_the_data"]}], "in several"),
        ([{**VERDICT[0], "gates": VERDICT[0]["gates"] + ["nope"]}, *VERDICT[1:]], "unknown"),
        ([{**VERDICT[0], "on_fail": "NOT EVALUATED"}, *VERDICT[1:]], "on_fail"),
        ([{**VERDICT[0], "stage": "masking"}, *VERDICT[1:]], "repeat"),
    ],
)
def test_a_verdict_whose_stages_do_not_hold_every_gate_once_is_refused(tmp_path, verdict, message):
    spec = write_spec(tmp_path, verdict=verdict)
    with pytest.raises(ValueError, match=message):
        sg.load_verdict(spec, sg.load_score_gates(spec))


# --------------------------------------------------------------------------------------
# Training logs
# --------------------------------------------------------------------------------------


def test_paired_logs_must_cover_the_same_iterations(tmp_path, probe, capsys):
    experiment(tmp_path, probe)
    write_masked_log(tmp_path, iteration_lines(True, (19.0,) * 3))
    status, detail = one(only("log_pairing", "listed_counts_identical"), tmp_path, capsys)
    assert status == 2 and "missing [4]" in detail


def test_a_listed_count_that_differs_between_the_arms_fails_at_its_iteration(tmp_path, probe, capsys):
    experiment(tmp_path, probe)
    write_masked_log(tmp_path, iteration_lines(True, (19.0,) * 4, count_edits={2: {"listed": 1201}}))
    status, detail = one(only("log_pairing", "listed_counts_identical"), tmp_path, capsys)
    assert status == 1
    assert "token_masking/count/listed differs by 1 at iteration 2 (1201 against 1200), above 0" in detail


def test_a_count_differing_by_one_fails_although_the_printed_fractions_agree(tmp_path, probe, capsys):
    """At 2**26 target positions, 2**25 and 2**25 + 1 listed targets print the same 7-digit fraction: only the exact
    counts tell the arms apart."""
    positions, listed = 2**26, 2**25
    for name, enabled, extra in (("control.log", False, 0), ("masked.log", True, 1)):
        lines = []
        for iteration in range(1, 5):
            listed_here = listed + (extra if iteration == 3 else 0)
            lines.append(token_masking_counts_line(iteration, **counts(enabled, listed_here, positions)))
            lines.append(
                token_masking_iteration_line(
                    iteration, enabled=enabled, listed=listed_here / positions, lm_loss=2.0, listed_loss=19.0
                )
            )
        write_arm_log(tmp_path, name, enabled=enabled, iterations=lines, validation={})
    fractions = {"token_masking/listed_target_fraction": 0.0}
    status, detail = one(only("log_pairing", "listed_counts_identical", metrics=fractions), tmp_path, capsys)
    assert status == 0, "the printed fractions cannot show the difference"
    status, detail = one(only("log_pairing", "listed_counts_identical"), tmp_path, capsys)
    assert status == 1 and "token_masking/count/listed differs by 1 at iteration 3" in detail


def test_the_other_targets_loss_is_compared_between_a_masked_arm_and_its_control(tmp_path, probe, capsys):
    """The masked arm's lm loss leaves out the marker's targets and the control's holds them; the loss the two share
    is derived from the control's reports, here 2.0 in both."""
    experiment(tmp_path, probe)
    status, detail = one(only("log_pairing", "other_targets_agree"), tmp_path, capsys)
    assert status == 0 and "non_listed_target_loss differs by at most" in detail
    higher = [
        line.replace("lm loss: 2.000000E+00", "lm loss: 2.100000E+00") for line in iteration_lines(True, (19.0,) * 4)
    ]
    write_masked_log(tmp_path, higher)
    status, detail = one(only("log_pairing", "other_targets_agree"), tmp_path, capsys)
    assert status == 1 and re.search(
        r"non_listed_target_loss differs by 0\.(1|0999)\d* at iteration \d \(2\.1\d* against 2\), above 0\.05", detail
    )


@pytest.mark.parametrize(
    "metrics, edit, absent",
    [
        (None, lambda line: None if COUNTS_TAG in line else line, "token_masking/count/listed is absent"),
        (
            {"token_masking/listed_trainable_target_fraction": 0.0},
            lambda line: line.replace(" | token_masking/listed_trainable_target_fraction", " | other"),
            "listed_trainable_target_fraction is absent",
        ),
    ],
    ids=["count", "iteration-line value"],
)
def test_a_pairing_metric_a_log_lacks_fails(tmp_path, probe, capsys, metrics, edit, absent):
    experiment(tmp_path, probe)
    bare = [edited for line in iteration_lines(True, (19.0,) * 4) if (edited := edit(line)) is not None]
    write_masked_log(tmp_path, bare)
    spec = only("log_pairing", "listed_counts_identical", **({} if metrics is None else {"metrics": metrics}))
    status, detail = one(spec, tmp_path, capsys)
    assert status == 1 and f"{absent} at 4 iterations, from 1" in detail


def test_a_non_finite_paired_value_is_not_evaluated_rather_than_passed(tmp_path, probe, capsys):
    """A NaN lm loss makes the derived loss of the other targets NaN, which no tolerance would ever refuse."""
    experiment(tmp_path, probe)
    lines = iteration_lines(True, (19.0,) * 4)
    lines[3] = lines[3].replace("lm loss: 2.000000E+00", "lm loss: nan")
    write_masked_log(tmp_path, lines)
    status, detail = one(only("log_pairing", "other_targets_agree"), tmp_path, capsys)
    assert status == 2
    assert "masked.log token_masking/non_listed_target_loss at iteration 2 is nan, not a finite number" in detail


@pytest.mark.parametrize(
    "arm, message",
    [
        ({"hosts": ("nid1",)}, "2 nodes"),
        ({"banner_enabled": False}, "states enabled=false token_ids=[] measured_token_ids=[7], not enabled=true"),
    ],
)
def test_the_banner_must_come_from_every_node_and_state_the_arm(tmp_path, probe, capsys, arm, message):
    experiment(tmp_path, probe)
    write_masked_log(tmp_path, **arm)
    status, detail = one(only("masking_log", "masked_arm"), tmp_path, capsys)
    assert status == 1 and message in detail


def test_a_control_that_masked_anything_fails(tmp_path, probe, capsys):
    experiment(tmp_path, probe)
    write_control_log(tmp_path, iteration_lines(False, (19.0,) * 4, count_edits={2: {"masked": 1}}))
    status, detail = one(only("masking_log", "control_arm"), tmp_path, capsys)
    assert status == 1
    assert (
        "masked == 0 and trained_listed == listed_trainable fails at 1 iterations, from 2 (masked=1 "
        "trained_listed=1200 listed_trainable=1200)"
    ) in detail


def test_the_masking_log_reports_the_exact_counts(tmp_path, probe, capsys):
    status, detail = one(only("masking_log", "masked_arm"), experiment(tmp_path, probe), capsys)
    assert status == 0
    assert detail == (
        "2 banners from 2 hosts; iterations 1-4: masked == listed_trainable and trained_listed == 0; 4200 trainable "
        "listed targets of 400000 target positions, 900 to 1200 per iteration"
    )


def test_an_iteration_without_its_counts_line_fails(tmp_path, probe, capsys):
    """The monitor prints the counts every iteration of a run that measures an id: a missing line is a run whose
    masking code did not report."""
    experiment(tmp_path, probe)
    lines = iteration_lines(True, (19.0,) * 4)
    assert f"{COUNTS_TAG} iteration=3 " in lines[4]
    write_masked_log(tmp_path, lines[:4] + lines[5:])
    status, detail = one(only("masking_log", "masked_arm"), tmp_path, capsys)
    assert status == 1 and "the [token-masking-counts] line is absent at 1 iterations, from 3" in detail


def test_a_duplicated_counts_line_is_read_once_and_conflicting_ones_are_not_evaluated(tmp_path, probe, capsys):
    experiment(tmp_path, probe)
    lines = iteration_lines(True, (19.0,) * 4)
    write_masked_log(tmp_path, lines + [lines[2]])
    status, _ = one(only("masking_log", "masked_arm"), tmp_path, capsys)
    assert status == 0, "a logging handler that prints a line twice duplicates it exactly"
    conflicting = token_masking_counts_line(2, **counts(True, 1200, masked=1199))
    write_masked_log(tmp_path, lines + [conflicting])
    status, detail = one(only("masking_log", "masked_arm"), tmp_path, capsys)
    assert status == 2 and "two different token-masking counts lines for iteration 2" in detail


def test_an_arm_that_never_met_a_listed_target_fails(tmp_path, probe, capsys):
    experiment(tmp_path, probe)
    write_masked_log(tmp_path, iteration_lines(True, (19.0,) * 4, listed=(0.0,) * 4))
    status, detail = one(only("masking_log", "masked_arm"), tmp_path, capsys)
    assert status == 1 and "no iteration held a trainable target" in detail


def test_an_arm_that_stopped_early_is_not_evaluated(tmp_path, probe, capsys):
    experiment(tmp_path, probe)
    write_masked_log(tmp_path, iteration_lines(True, (19.0,) * 3))
    status, detail = one(only("masking_log", "masked_arm"), tmp_path, capsys)
    assert status == 2 and "missing [4]" in detail


# --------------------------------------------------------------------------------------
# Values, slots, emissions, identity
# --------------------------------------------------------------------------------------


def test_a_value_the_log_does_not_hold_is_not_evaluated(tmp_path, probe, capsys):
    experiment(tmp_path, probe)
    candidate = {"log": "masked.log", "validation_step": 9, "metric": LISTED_LOSS}
    status, detail = one(only("value_change", "masked_marker_ce_held", candidate=candidate), tmp_path, capsys)
    assert status == 2 and "holds 0 such results, not one" in detail


def test_the_value_change_states_both_values_and_bounds(tmp_path, probe, capsys):
    status, detail = one(only("value_change", "control_learned_the_marker"), experiment(tmp_path, probe), capsys)
    assert status == 0
    assert detail == (
        "control.json held-out marker_ce 2.450000 against base.json held-out marker_ce 19.400000: "
        "change -16.950000 in [-inf, -10], value in [-inf, 3]"
    )


def test_the_drift_reference_is_taken_out_of_the_slot_difference(tmp_path, probe, capsys):
    """The masked arm's marker fell 1 nat below the base while the never-trained reference row fell 0.2: the
    difference in differences is -0.8, the plain difference -1.0."""
    status, detail = one(only("slot_logprob_difference", "masked_slots_held"), experiment(tmp_path, probe), capsys)
    assert status == 0 and "median -0.800" in detail


def test_too_few_prompts_reaching_the_difference_fails(tmp_path, probe, capsys):
    spec = only("slot_logprob_difference", "control_slots_rose", min_difference=19.0)
    status, detail = one(spec, experiment(tmp_path, probe), capsys)
    assert status == 1 and "only 0 prompts reach +19" in detail


def test_probes_of_different_specs_are_not_compared(tmp_path, probe, capsys):
    experiment(tmp_path, probe)
    other = json.loads((tmp_path / "base.json").read_text())
    other["spec"]["sha256"] = "0" * 64
    (tmp_path / "base.json").write_text(json.dumps(other))
    status, detail = one(only("slot_logprob_difference", "control_slots_rose"), tmp_path, capsys)
    assert status == 2 and "different probe specs" in detail


def test_emissions_are_counted_where_they_occurred(tmp_path, probe, capsys):
    spec = only("emission_count", "masked_never_greedy", probe="control.json")
    status, detail = one(spec, experiment(tmp_path, probe), capsys)
    assert status == 1
    assert detail == (
        f"{MARKER_ID} anywhere in 3 greedy generations of control.json: 4 occurrences, bounds [None, 0] "
        "(in slot greedy 0)"
    )


@pytest.mark.parametrize(
    "position, min_count, status, counted",
    [("anywhere", 1, 0, 1), ("anywhere", 2, 1, 1), ("first", 1, 0, 1)],
    ids=["enough", "too-few", "first"],
)
def test_generations_holding_the_id_are_counted_against_a_minimum(
    tmp_path, probe, capsys, position, min_count, status, counted
):
    """The copy model emits the marker four times in one greedy generation (the in-context prompt's) and in none of
    the other two: one generation holds it, however many times it occurs there."""
    gate = only("emission_count", "masked_never_greedy", probe="control.json", position=position, unit="generations")
    gate["emission_count"]["masked_never_greedy"].pop("max_count")
    gate["emission_count"]["masked_never_greedy"]["min_count"] = min_count
    got, detail = one(gate, experiment(tmp_path, probe), capsys)
    assert got == status
    assert f"control.json: {counted} generations, bounds [{min_count}, None]" in detail


def test_sampled_generations_holding_the_id_are_counted_against_a_maximum(tmp_path, probe, capsys):
    """The three samples of the in-context prompt each repeat the marker: three of nine sampled generations."""
    gate = only("emission_count", "masked_never_greedy", probe="control.json", generations="sample")
    gate["emission_count"]["masked_never_greedy"].update(unit="generations", max_count=2)
    got, detail = one(gate, experiment(tmp_path, probe), capsys)
    assert got == 1 and "in 9 sample generations of control.json: 3 generations, bounds [None, 2]" in detail


def test_an_id_the_probe_did_not_count_is_not_evaluated(tmp_path, probe, capsys):
    spec = only("emission_count", "masked_never_greedy", token_id=REFERENCE_ID)
    status, detail = one(spec, experiment(tmp_path, probe), capsys)
    assert status == 2 and f"did not count {REFERENCE_ID}" in detail


def test_an_identity_that_differs_or_is_absent_fails(tmp_path, probe, capsys):
    expect = {"model.iteration": 477, "model.megatron_run_config.train": 1}
    status, detail = one(only("probe_identity", "masked_identity", expect=expect), experiment(tmp_path, probe), capsys)
    assert status == 1
    assert detail == "masked.json: model.iteration is 3, not 477; model.megatron_run_config.train is absent"


def test_a_file_that_is_not_probe_results_is_not_evaluated(tmp_path, probe, capsys):
    (experiment(tmp_path, probe) / "masked.json").write_text(json.dumps({"format": "score"}))
    status, detail = one(only("probe_identity", "masked_identity"), tmp_path, capsys)
    assert status == 2 and "not a coherence-probe/1 results file" in detail


NO_BOUNDS = {key: GATES["value_change"]["masked_learned_the_data"][key] for key in ("candidate", "reference")}


@pytest.mark.parametrize(
    "gates, message",
    [
        (only("value_change", "masked_learned_the_data", max_chang=0.05), r"unknown keys \['max_chang'\]"),
        ({"value_change": {"g": NO_BOUNDS}}, "states none of"),
        (only("value_change", "masked_learned_the_data", min_change=0.1, max_change=0.0), "above its maximum"),
        (only("slot_logprob_difference", "control_slots_rose", reference="control.json"), "with itself"),
        (only("slot_logprob_difference", "masked_slots_held", min_prompts=3), "needs it"),
        (only("emission_count", "masked_never_greedy", position="last"), "position one of"),
        (only("emission_count", "masked_never_greedy", unit="tokens"), "unit must be one of"),
        (only("emission_count", "masked_never_greedy", max_count=None), "states none of"),
        (only("emission_count", "masked_never_greedy", min_count=3, max_count=1), "above its maximum"),
        (only("emission_count", "masked_never_greedy", max_count=-1), "a count bound is negative"),
        (only("masking_log", "masked_arm", enabled="yes"), "true or false"),
        (only("log_pairing", "listed_counts_identical", metrics={}), "tolerance"),
        (
            only("log_pairing", "listed_counts_identical", metrics={"token_masking/count/listed": 1}),
            "exact count, so its tolerance must be 0",
        ),
        (
            only("log_pairing", "listed_counts_identical", metrics={"token_masking/count/marked": 0}),
            "is no token-masking count",
        ),
        (only("probe_identity", "masked_identity", expect={}), "dotted paths"),
        (only("value_change", "masked_learned_the_data", exclusive_bounds="yes"), "true or false"),
        (
            only("value_change", "masked_learned_the_data", min_change=0.0, max_change=0.0, exclusive_bounds=True),
            "at it with exclusive bounds",
        ),
        (only("probe_agreement", "one_tokenizer_one_code", probes=["base.json"]), "two or more"),
        (only("probe_agreement", "one_tokenizer_one_code", fields=[]), "dotted paths"),
    ],
)
def test_a_gate_that_could_not_be_evaluated_as_written_is_refused(tmp_path, gates, message):
    with pytest.raises(ValueError, match=message):
        sg.load_score_gates(write_spec(tmp_path, gates, verdict=None))


# --------------------------------------------------------------------------------------
# A value that is not a finite number is never a pass
# --------------------------------------------------------------------------------------


NAN = float("nan")


def test_a_nan_held_out_score_is_not_evaluated_where_it_would_have_passed(tmp_path, probe, capsys):
    """A NaN marker CE compares false with every bound, so the positive control's change bound alone would pass it."""
    experiment(tmp_path, probe)
    write_control(tmp_path, probe, marker_ce=NAN)
    assert sg._outside(NAN, None, -10.0) and sg._outside(NAN, None, -10.0, exclusive=True)
    status, detail = one(only("value_change", "control_learned_the_marker"), tmp_path, capsys)
    assert status == 2
    assert detail == "ValueError: control.json held-out marker_ce is nan, not a finite number"


def test_a_nan_evaluation_result_is_not_evaluated(tmp_path, probe, capsys):
    experiment(tmp_path, probe)
    write_arm_log(
        tmp_path,
        "masked.log",
        enabled=True,
        iterations=iteration_lines(True, (19.0,) * 4),
        validation={0: 19.4, 4: NAN},
    )
    status, detail = one(only("value_change", "masked_marker_ce_held"), tmp_path, capsys)
    assert status == 2
    assert f"masked.log {LISTED_LOSS} at step 4 is nan, not a finite number" in detail


@pytest.mark.parametrize("edit", [{"marker": NAN}, {"reference": math.inf}], ids=["marker", "drift"])
def test_a_non_finite_slot_log_probability_is_not_evaluated(tmp_path, probe, capsys, edit):
    """A NaN difference passes both a median bound and a per-prompt maximum when compared naively."""
    experiment(tmp_path, probe)
    write_masked(tmp_path, probe, **edit)
    status, detail = one(only("slot_logprob_difference", "masked_slots_held"), tmp_path, capsys)
    assert status == 2 and "masked.json prompt slot log p(" in detail and "not a finite number" in detail


def test_a_count_that_is_not_one_is_not_evaluated(tmp_path, probe, capsys):
    experiment(tmp_path, probe)
    masked = json.loads((tmp_path / "masked.json").read_text())
    masked["prompts"][0]["generations"][0]["counts"][M]["anywhere"] = NAN
    (tmp_path / "masked.json").write_text(json.dumps(masked))
    status, detail = one(only("emission_count", "masked_never_greedy"), tmp_path, capsys)
    assert status == 2 and "is nan, not a count" in detail


# --------------------------------------------------------------------------------------
# Exclusive bounds, revisions, agreement
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "masked_ce, exclusive, status",
    [(2.3, True, 1), (2.3, False, 0), (2.29, True, 0), (2.31, False, 1)],
    ids=["equal-exclusive", "equal-inclusive", "below-exclusive", "above-inclusive"],
)
def test_an_exclusive_bound_refuses_its_endpoint(tmp_path, probe, capsys, masked_ce, exclusive, status):
    """The masked arm must score the documents strictly better than the base: equal is not learned."""
    experiment(tmp_path, probe)
    write_masked(tmp_path, probe, non_marker_ce=masked_ce)
    reference = {"probe": "base.json", "held_out": "non_marker_ce"}
    gate = only("value_change", "masked_learned_the_data", reference=reference, min_change=None, max_change=0.0)
    gate["value_change"]["masked_learned_the_data"].pop("min_change")
    if exclusive:
        gate["value_change"]["masked_learned_the_data"]["exclusive_bounds"] = True
    got, detail = one(gate, tmp_path, capsys)
    assert got == status
    assert detail.endswith("in (-inf, 0), value in (-inf, inf)" if exclusive else "in [-inf, 0], value in [-inf, inf]")


def edit_probe(directory: Path, name: str, path: str, value) -> None:
    """Set the dotted ``path`` of a probe results file to ``value``."""
    results = json.loads((directory / name).read_text())
    *parents, last = path.split(".")
    node = results
    for part in parents:
        node = node[part]
    node[last] = value
    (directory / name).write_text(json.dumps(results))


def test_probes_measured_by_different_code_are_not_compared(tmp_path, probe, capsys):
    experiment(tmp_path, probe)
    edit_probe(tmp_path, "base.json", "run.code_revision", "0" * 40)
    status, detail = one(only("slot_logprob_difference", "control_slots_rose"), tmp_path, capsys)
    assert status == 2 and "were measured by code at different revisions" in detail


def test_a_probe_that_does_not_record_its_code_revision_is_not_compared(tmp_path, probe, capsys):
    experiment(tmp_path, probe)
    base = json.loads((tmp_path / "base.json").read_text())
    del base["run"]["code_revision"]
    (tmp_path / "base.json").write_text(json.dumps(base))
    status, detail = one(only("slot_logprob_difference", "control_slots_rose"), tmp_path, capsys)
    assert status == 2 and detail == "LookupError: base.json does not record the revision of the code that measured it"


def test_prompts_scored_on_different_input_ids_are_not_compared(tmp_path, probe, capsys):
    experiment(tmp_path, probe)
    base = json.loads((tmp_path / "base.json").read_text())
    base["prompts"][1]["input_ids"] = base["prompts"][1]["input_ids"][:-1]
    (tmp_path / "base.json").write_text(json.dumps(base))
    status, detail = one(only("slot_logprob_difference", "control_slots_rose"), tmp_path, capsys)
    assert status == 2 and "scored prompts ['no_slot'] on different input ids" in detail


def test_probes_that_agree_pass_and_name_what_they_agree_on(tmp_path, probe, capsys):
    status, detail = one(only("probe_agreement", "one_tokenizer_one_code"), experiment(tmp_path, probe), capsys)
    assert status == 0
    assert detail.startswith("base.json, masked.json, control.json: tokenizer.json_sha256 is '")
    assert "run.code_revision is" in detail


def test_a_probe_with_another_tokenizer_fails_the_agreement(tmp_path, probe, capsys):
    experiment(tmp_path, probe)
    edit_probe(tmp_path, "control.json", "tokenizer.json_sha256", "f" * 64)
    status, detail = one(only("probe_agreement", "one_tokenizer_one_code"), tmp_path, capsys)
    assert (
        status == 1
        and "tokenizer.json_sha256 differs: base.json '" in detail
        and f"control.json '{'f' * 64}'" in detail
    )


def test_a_probe_lacking_an_agreed_field_fails_and_a_missing_probe_is_not_evaluated(tmp_path, probe, capsys):
    experiment(tmp_path, probe)
    masked = json.loads((tmp_path / "masked.json").read_text())
    del masked["run"]["code_revision"]
    (tmp_path / "masked.json").write_text(json.dumps(masked))
    status, detail = one(only("probe_agreement", "one_tokenizer_one_code"), tmp_path, capsys)
    assert status == 1 and "run.code_revision is absent from masked.json" in detail
    (tmp_path / "control.json").unlink()
    status, detail = one(only("probe_agreement", "one_tokenizer_one_code"), tmp_path, capsys)
    assert status == 2 and detail.startswith("FileNotFoundError")

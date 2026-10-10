"""Unit tests for scripts/telemetry/training_log.py (the training-log parser the telemetry tools share).

Every log is the real excerpt of ``training_log_fixture.py`` or built by editing its lines.
"""

import json

import pytest
from scripts.telemetry.training_log import (
    NON_LISTED_TARGET_LOSS,
    TOKEN_MASKING_COUNT_FIELDS,
    TOKEN_MASKING_COUNTS_TAG,
    TOKEN_MASKING_TAG,
    IterationRecord,
    TokenMaskingCountsRecord,
    check_window,
    non_listed_target_loss,
    parse_first_iteration_memory,
    parse_iteration_records,
    parse_iteration_values,
    parse_node_banners,
    parse_peak_memory_across_ranks,
    parse_token_masking_counts,
    parse_validation_records,
    parse_wandb_run_path,
    read_log_lines,
    window_records,
)

from tests.unit_tests.training_log_fixture import (
    FIXTURE_LINES,
    FIXTURE_RUN_PATH,
    REAL_CONTROL_ITERATION_2,
    REAL_ITERATION_50,
    REAL_MEMORY_LINE,
    iteration_line,
    only_line,
    sub_once,
    token_masking_banner,
    token_masking_counts_line,
    token_masking_iteration_line,
    validation_line,
)


# --------------------------------------------------------------------------------------
# Iteration lines
# --------------------------------------------------------------------------------------


def test_parses_every_field_of_a_real_iteration_line():
    records = parse_iteration_records(FIXTURE_LINES)
    assert [r.iteration for r in records] == list(range(1, 61))
    assert records[49] == IterationRecord(
        iteration=50,
        elapsed_ms=9937.2,
        global_batch_size=2048,
        logged_tflops_per_gpu=146.9,
        lm_loss=6.681113,
        skipped_total=0,
        nan_total=0,
        timestamp="2026-09-09 01:38:01",
        consumed_samples=102400,
        learning_rate=1.673304e-04,
        grad_norm=1.505,
    )
    # The first iteration carries the one-time setup cost, logged like any other.
    assert records[0].elapsed_ms == 105400.6
    assert {r.global_batch_size for r in records} == {2048}


def test_non_iteration_lines_are_ignored():
    # A "Step Time : ..." line precedes each of iterations 2-60, among the memory and wandb lines.
    assert sum("Step Time" in line for line in FIXTURE_LINES) == 59
    assert len(parse_iteration_records(FIXTURE_LINES)) == 60


def test_optional_fields_absent_parse_as_none():
    line = sub_once(r" throughput per GPU \(TFLOP/s/GPU\): [\d.]+ \|", "", REAL_ITERATION_50)
    line = sub_once(r" lm loss: [\dE.+-]+ \|", "", line)
    line = sub_once(r" consumed samples:\s+\d+ \|", "", line)
    line = sub_once(r" learning rate: [\dE.+-]+ \|", "", line)
    line = sub_once(r" grad norm: [\d.]+ \|", "", line)
    line = sub_once(r"^ \[[^\]]+\]\s+", "", line)
    (record,) = parse_iteration_records([line])
    assert record.logged_tflops_per_gpu is None
    assert record.lm_loss is None
    assert record.consumed_samples is None
    assert record.learning_rate is None
    assert record.grad_norm is None
    assert record.timestamp is None
    assert record.elapsed_ms == 9937.2


def test_skipped_and_nan_counts_are_parsed():
    (record,) = parse_iteration_records([iteration_line(7, 9000.0, 6.0, skipped=2, nan=1)])
    assert (record.iteration, record.skipped_total, record.nan_total) == (7, 2, 1)


@pytest.mark.parametrize(
    "field_pattern, name",
    [
        (r" elapsed time per iteration \(ms\): [\d.]+ \|", "elapsed time per iteration"),
        (r" global batch size:\s+\d+ \|", "global batch size"),
    ],
)
def test_iteration_line_without_a_required_field_raises(field_pattern, name):
    line = sub_once(field_pattern, "", REAL_ITERATION_50)
    with pytest.raises(ValueError, match=name):
        parse_iteration_records([line])


# --------------------------------------------------------------------------------------
# Memory report and W&B run path
# --------------------------------------------------------------------------------------


def test_first_iteration_memory_reads_the_gigabyte_fields():
    assert parse_first_iteration_memory(FIXTURE_LINES) == {
        "mem-allocated-gigabytes": 61.555,
        "mem-active-gigabytes": 61.555,
        "mem-inactive-gigabytes": 0.0,
        "mem-reserved-gigabytes": 80.487,
        "mem-max-allocated-gigabytes": 78.383,
        "mem-max-active-gigabytes": 78.383,
        "mem-max-inactive-gigabytes": 0.0,
        "mem-max-reserved-gigabytes": 80.487,
    }


def test_first_iteration_memory_takes_the_maximum_across_ranks():
    rank1 = sub_once(r"^\[Rank 0\]", "[Rank 1]", REAL_MEMORY_LINE)
    rank1 = sub_once(r"mem-max-allocated-gigabytes: [\d.]+", "mem-max-allocated-gigabytes: 79.5", rank1)
    rank1 = sub_once(r"mem-allocated-gigabytes: [\d.]+", "mem-allocated-gigabytes: 60.0", rank1)
    memory = parse_first_iteration_memory([rank1, REAL_MEMORY_LINE])
    assert memory["mem-max-allocated-gigabytes"] == 79.5
    assert memory["mem-allocated-gigabytes"] == 61.555


def test_first_iteration_memory_absent_or_resumed_is_none():
    resumed = sub_once(r"after 1 iterations", "after 13585 iterations", REAL_MEMORY_LINE)
    assert parse_first_iteration_memory([resumed, REAL_ITERATION_50]) is None


def test_memory_report_without_gigabyte_fields_raises():
    with pytest.raises(ValueError, match="-gigabytes"):
        parse_first_iteration_memory(["[Rank 0] (after 1 iterations) memory (GB) | mem-alloc-retires: 0"])


# --------------------------------------------------------------------------------------
# End-of-training peak memory over all ranks
# --------------------------------------------------------------------------------------

PEAK_SUMMARY = {
    "ranks": 512,
    "max_allocated_gb": 74.751,
    "max_allocated_rank": 131,
    "max_reserved_gb": 88.12,
    "max_alloc_retries": 0,
    "total_alloc_retries": 0,
}


def test_peak_memory_reads_the_line_the_bridge_writes():
    """The parser restates the bridge's tag rather than importing it, so this pins the two together."""
    from megatron.bridge.training.utils.train_utils import format_peak_memory

    logged = f"[2026-09-30 21:00:00] {format_peak_memory(PEAK_SUMMARY)}"
    assert parse_peak_memory_across_ranks([REAL_ITERATION_50, logged]) == PEAK_SUMMARY


def test_peak_memory_absent_is_none():
    """A log that predates the summary, or a run whose loop was cut short, holds none."""
    assert parse_peak_memory_across_ranks(FIXTURE_LINES) is None


def test_two_peak_memory_summaries_mean_two_runs_and_raise():
    line = "[peak-memory] ranks=4 max_allocated_gb=1.0"
    with pytest.raises(ValueError, match="more than one run"):
        parse_peak_memory_across_ranks([line, line])


def test_a_peak_memory_field_that_is_not_key_value_raises():
    with pytest.raises(ValueError, match="not key=value"):
        parse_peak_memory_across_ranks(["[peak-memory] ranks=4 74.7"])


def test_wandb_run_path_from_init_and_finish_lines():
    init_line = only_line("View run at")
    finish_line = only_line("View run control_pretrain")
    assert parse_wandb_run_path([init_line]) == FIXTURE_RUN_PATH
    assert parse_wandb_run_path([finish_line]) == FIXTURE_RUN_PATH
    assert parse_wandb_run_path(FIXTURE_LINES) == FIXTURE_RUN_PATH


def test_wandb_run_path_on_a_self_hosted_server():
    line = sub_once(r"https://wandb\.ai/", "http://wandb.example.org:8080/", only_line("View run at"))
    assert parse_wandb_run_path([line]) == FIXTURE_RUN_PATH


def test_wandb_run_path_absent_is_none():
    assert parse_wandb_run_path([REAL_ITERATION_50, REAL_MEMORY_LINE]) is None


def test_wandb_run_path_naming_two_runs_raises():
    other = sub_once(r"runs/5s8x5mgb", "runs/zno0zq8b", only_line("View run at"))
    with pytest.raises(ValueError, match="several W&B runs"):
        parse_wandb_run_path([only_line("View run at"), other])


# --------------------------------------------------------------------------------------
# Reading a log and taking a window of it
# --------------------------------------------------------------------------------------


def test_read_log_lines_replaces_an_undecodable_byte(tmp_path):
    path = tmp_path / "train.out"
    path.write_bytes(REAL_ITERATION_50.encode("utf-8") + b"\n\xff interleaved native output\n")
    lines = read_log_lines(path)
    assert lines == [REAL_ITERATION_50, "\ufffd interleaved native output"]
    assert [r.iteration for r in parse_iteration_records(lines)] == [50]


def test_read_log_lines_leaves_out_a_last_line_still_being_written(tmp_path):
    """A live log's last line has no line break until its writer finishes it; read half-written, an iteration line
    lacks its required fields."""
    path = tmp_path / "train.out"
    path.write_text(REAL_ITERATION_50 + "\n" + REAL_ITERATION_50[: len(REAL_ITERATION_50) // 2], encoding="utf-8")
    lines = read_log_lines(path)
    assert lines == [REAL_ITERATION_50]
    assert [r.iteration for r in parse_iteration_records(lines)] == [50]


def test_window_records_returns_the_window_in_iteration_order():
    records = parse_iteration_records([iteration_line(i, 1000.0, 6.0) for i in (3, 1, 4, 2)])
    assert [r.iteration for r in window_records(records, (1, 3), "log")] == [1, 2, 3]


def test_window_records_names_missing_and_repeated_iterations():
    records = parse_iteration_records([iteration_line(i, 1000.0, 6.0) for i in (1, 2, 2, 4)])
    with pytest.raises(ValueError, match=r"log 1-4: .*missing \[3\], repeated \[2\]"):
        window_records(records, (1, 4), "log")


@pytest.mark.parametrize("window, min_iterations", [((0, 5), 1), ((5, 4), 1), ((5, 5), 2)])
def test_check_window_rejects_a_window_before_iteration_1_or_shorter_than_its_minimum(window, min_iterations):
    with pytest.raises(ValueError, match=r"window .* must start at >= 1 and span >= "):
        check_window("window", window, min_iterations)


def test_check_window_accepts_a_window_of_exactly_its_minimum_span():
    check_window("window", (5, 6), 2)


def test_window_records_takes_iteration_values_too():
    values = parse_iteration_values([iteration_line(i, 1000.0, 6.0) for i in (2, 1)])
    assert [v.iteration for v in window_records(values, (1, 2), "log")] == [1, 2]


# --------------------------------------------------------------------------------------
# Token masking: every numeric field of an iteration line, evaluation lines, the banner
# --------------------------------------------------------------------------------------


def test_iteration_values_hold_every_numeric_field_of_a_real_token_masking_line():
    (values,) = parse_iteration_values([REAL_CONTROL_ITERATION_2])
    assert values.iteration == 2
    assert values.values["token_masking/listed_target_fraction"] == 4.306793e-03
    assert values.values["token_masking/trained_listed_target_fraction"] == 4.306793e-03
    assert values.values["lm loss"] == 1.117542
    assert values.values["global batch size"] == 64
    # The canary predates the listed-target loss, so the loss of the other targets cannot be derived from it.
    assert NON_LISTED_TARGET_LOSS not in values.values


def test_the_other_targets_loss_is_lm_loss_when_masking_and_derived_when_measuring():
    masked, control = parse_iteration_values(
        [
            token_masking_iteration_line(1, enabled=True, listed=0.01, lm_loss=2.0, listed_loss=19.0),
            token_masking_iteration_line(1, enabled=False, listed=0.01, lm_loss=2.17, listed_loss=19.0),
        ]
    )
    assert masked.values[NON_LISTED_TARGET_LOSS] == 2.0
    assert control.values[NON_LISTED_TARGET_LOSS] == pytest.approx((2.17 - 19.0 * 0.01) / 0.99)


def test_without_token_masking_reports_nothing_is_derived():
    assert non_listed_target_loss({"lm loss": 2.0}) is None
    assert NON_LISTED_TARGET_LOSS not in parse_iteration_values([REAL_ITERATION_50])[0].values


@pytest.mark.parametrize("enabled", [True, False])
def test_the_derived_loss_is_the_mean_cross_entropy_at_the_other_trained_targets(enabled):
    """Reports made by the real hook and loss function, reduced and finalized as an evaluation does and printed as
    its line: the loss derived from them equals the mean of the per-token losses at the targets that carry loss in
    the dataset's mask and are not the measured id's, masking or not."""
    import torch

    from megatron.bridge.training.eval import evaluation_results
    from megatron.bridge.training.losses import masked_next_token_loss
    from megatron.bridge.training.token_masking.hook import apply_token_masking
    from tests.unit_tests.token_masking_fixtures import masking_with_null_tokenizer, measuring_with_null_tokenizer

    marker = 7
    resolved = (masking_with_null_tokenizer if enabled else measuring_with_null_tokenizer)([marker], vocab_size=16)
    generator = torch.Generator().manual_seed(0)
    reports, other_sum, other_count = [], 0.0, 0
    for _ in range(2):
        labels = torch.randint(0, 16, (2, 24), generator=generator)
        labels[:, ::5] = marker
        loss_mask = (torch.rand(2, 24, generator=generator) > 0.2).float()
        losses = torch.rand(2, 24, generator=generator) * 5
        mask, stats = apply_token_masking(labels, loss_mask, resolved)
        reports.append(masked_next_token_loss(mask, losses, check_for_nan_in_loss=False, token_masking_stats=stats)[2])
        other = (labels != marker) & (loss_mask != 0)
        other_sum, other_count = other_sum + float(losses[other].sum()), other_count + int(other.sum())
    totals = {key: torch.stack([report[key].float() for report in reports]).sum(dim=0) for key in reports[0]}
    results = evaluation_results(totals, "masked-validation/")
    (record,) = parse_validation_records([validation_line(53, {k: float(v) for k, v in results.items()})])
    derived = record.values[f"masked-validation/{NON_LISTED_TARGET_LOSS}"]
    assert derived == pytest.approx(other_sum / other_count, rel=1e-5)


def test_evaluation_lines_are_read_by_step_with_their_values_not_their_perplexities():
    lines = [
        validation_line(
            0, {"masked-validation/lm loss": 2.5, "masked-validation/token_masking/listed_target_loss": 19}
        ),
        REAL_ITERATION_50,
        " validation loss at the end of training for val data | lm loss value: 2.400000E+00 | ",
    ]
    first, last = parse_validation_records(lines)
    assert (first.label, first.step) == ("iteration 0", 0)
    assert first.values == {
        "masked-validation/lm loss": 2.5,
        "masked-validation/token_masking/listed_target_loss": 19.0,
    }
    assert (last.step, last.values) == (None, {"lm loss": 2.4})


def test_an_evaluation_result_that_is_not_a_number_raises():
    with pytest.raises(ValueError, match="not a number"):
        parse_validation_records([" validation loss at iteration 3 | lm loss value: n/a | "])


def test_the_token_masking_banner_is_read_per_node():
    lines = [token_masking_banner(True, [131072], host, rank) for rank, host in enumerate(["nid1", "nid2"])]
    lines.append(token_masking_banner(False, [131072], "nid3", 8))
    first, _, control = parse_node_banners(lines, TOKEN_MASKING_TAG)
    assert (first.rank, first.host) == ("0", "nid1")
    assert first.fields["enabled"] == "true" and json.loads(first.fields["token_ids"]) == [131072]
    assert control.fields["enabled"] == "false"
    assert json.loads(control.fields["token_ids"]) == []
    assert json.loads(control.fields["measured_token_ids"]) == [131072]


# --------------------------------------------------------------------------------------
# Token masking: the per-iteration counts line
# --------------------------------------------------------------------------------------

COUNTS = dict(listed=2**25 + 1, listed_trainable=12, masked=12, trained_listed=0, trainable=4_194_292, positions=2**26)


def test_the_counts_the_monitor_prints_are_read_exactly():
    """Counts past 2**24, where a float32 fraction would round, read back as the integers the monitor printed."""
    line = token_masking_counts_line(5, **COUNTS)
    assert line.startswith("INFO:megatron.bridge.training.token_masking.monitor:[token-masking-counts] iteration=5 ")
    assert parse_token_masking_counts(["unrelated", line]) == [TokenMaskingCountsRecord(5, **COUNTS)]


def test_the_restated_counts_format_is_the_bridges():
    """The tag, the fields and the metric names this module restates are the monitor's and the hook's."""
    import dataclasses

    from megatron.bridge.training.token_masking.hook import TokenMaskingCounts
    from megatron.bridge.training.token_masking.monitor import COUNTS_LOG_TAG

    assert COUNTS_LOG_TAG == f"[{TOKEN_MASKING_COUNTS_TAG}]"
    assert TOKEN_MASKING_COUNT_FIELDS == tuple(field.name for field in dataclasses.fields(TokenMaskingCounts))
    (record,) = parse_token_masking_counts([token_masking_counts_line(1, **COUNTS)])
    assert record.metrics() == TokenMaskingCounts(**COUNTS).wandb_metrics()


def test_counts_are_returned_in_iteration_order_and_a_duplicated_line_is_one_record():
    """A logging handler that duplicates the line, with another prefix, prints the same counts twice."""
    second, first = token_masking_counts_line(2, **COUNTS), token_masking_counts_line(1, **COUNTS)
    duplicate = "[2026-10-10 03:00:00] " + second.split(":", 2)[2]
    records = parse_token_masking_counts([second, first, duplicate])
    assert [record.iteration for record in records] == [1, 2]


def test_two_different_counts_for_one_iteration_raise():
    lines = [token_masking_counts_line(3, **COUNTS), token_masking_counts_line(3, **{**COUNTS, "masked": 11})]
    with pytest.raises(ValueError, match="two different token-masking counts lines for iteration 3"):
        parse_token_masking_counts(lines)


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda line: line.replace(" positions=", " places="), "holds"),
        (lambda line: line.replace(" masked=12", ""), "holds"),
        (lambda line: line.replace(" masked=12", " masked=1.2E+01"), "non-negative integer"),
        (lambda line: line.replace(" masked=12", " masked=-12"), "non-negative integer"),
        (lambda line: line.replace(" masked=12", " masked=12 masked=12"), "non-negative integer"),
        (lambda line: line + " extra", "non-negative integer"),
    ],
)
def test_a_counts_line_that_is_not_exactly_the_six_counts_raises(edit, message):
    with pytest.raises(ValueError, match=message):
        parse_token_masking_counts([edit(token_masking_counts_line(4, **COUNTS))])

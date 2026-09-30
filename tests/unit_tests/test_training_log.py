"""Unit tests for scripts/telemetry/training_log.py (the training-log parser the telemetry tools share).

Every log is the real excerpt of ``training_log_fixture.py`` or built by editing its lines.
"""

import pytest
from scripts.telemetry.training_log import (
    IterationRecord,
    check_window,
    parse_first_iteration_memory,
    parse_iteration_records,
    parse_peak_memory_across_ranks,
    parse_wandb_run_path,
    read_log_lines,
    window_records,
)

from tests.unit_tests.training_log_fixture import (
    FIXTURE_LINES,
    FIXTURE_RUN_PATH,
    REAL_ITERATION_50,
    REAL_MEMORY_LINE,
    iteration_line,
    only_line,
    sub_once,
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

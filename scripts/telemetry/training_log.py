"""Parse a training log as the launcher writes it: its iteration lines, the after-iteration-1 memory report, the
end-of-training peak-memory summary over all ranks and the W&B run it names.

The iteration line is the bridge's ``training/utils/train_utils.py::training_log`` output. The parser needs only the
standard library, so it runs under a login node's host interpreter. ``scripts/telemetry/score_run.py`` scores one
run from these records and ``scripts/telemetry/loss_parity.py`` compares several runs' trajectories.
"""

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


# The bridge's per-iteration log line (training/utils/train_utils.py::training_log), e.g.
#  [2026-09-09 01:38:01] iteration       50/   29881 | consumed samples: ... | lm loss: 6.681113E+00 | ...
_ITERATION_RE = re.compile(
    r"(?:\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+)?"
    r"iteration\s+(?P<iteration>\d+)/\s*\d+\s*\|(?P<fields>.*)$"
)
_ELAPSED_KEY = "elapsed time per iteration (ms)"
_THROUGHPUT_KEY = "throughput per GPU (TFLOP/s/GPU)"
_LOSS_KEY = "lm loss"
_GRAD_NORM_KEY = "grad norm"
_LEARNING_RATE_KEY = "learning rate"
_CONSUMED_SAMPLES_KEY = "consumed samples"
_SKIPPED_KEY = "number of skipped iterations"
_NAN_KEY = "number of nan iterations"
_GLOBAL_BATCH_KEY = "global batch size"
_REQUIRED_ITERATION_KEYS = (_ELAPSED_KEY, _SKIPPED_KEY, _NAN_KEY, _GLOBAL_BATCH_KEY)

# Printed once per data-parallel group's rank 0 after the first logged iteration of a fresh run.
_MEMORY_RE = re.compile(r"\[Rank \d+\] \(after 1 iterations\) memory \(GB\)(?P<fields>.*)$")
GIGABYTES_SUFFIX = "-gigabytes"

# Printed once by rank 0 when the training loop ends (train_utils.py::format_peak_memory), e.g.
#  [peak-memory] ranks=512 max_allocated_gb=74.751 max_allocated_rank=131 max_reserved_gb=88.12 ...
# The tag is restated rather than imported, because this module must import without torch.
_PEAK_MEMORY_RE = re.compile(r"\[peak-memory\] (?P<fields>.*)$")

# wandb prints the run URL at init ("View run at <url>") and at finish ("View run <name> at: <url>"),
# on whichever W&B server the run logs to.
_WANDB_RUN_RE = re.compile(
    r"View run\b.*?https?://[^/\s]+/"
    r"(?P<entity>[A-Za-z0-9_.-]+)/(?P<project>[A-Za-z0-9_.-]+)/runs/(?P<run_id>[A-Za-z0-9_-]+)"
)


@dataclass(frozen=True)
class IterationRecord:
    """One parsed iteration line of a training log.

    ``skipped_total`` and ``nan_total`` are the counts the line reports; Megatron resets both
    counters at every log line, so they cover the iterations since the previous line (with
    ``log_interval: 1``, this iteration alone). ``consumed_samples``, ``learning_rate`` and
    ``grad_norm`` are None when the line does not carry them.
    """

    iteration: int
    elapsed_ms: float
    global_batch_size: int
    logged_tflops_per_gpu: float | None
    lm_loss: float | None
    skipped_total: int
    nan_total: int
    timestamp: str | None
    consumed_samples: int | None
    learning_rate: float | None
    grad_norm: float | None


def _iteration_fields(line: str) -> tuple[int, str | None, dict[str, str]] | None:
    """Split an iteration line into (iteration, timestamp, {field: raw value}); None for any other line."""
    match = _ITERATION_RE.search(line)
    if match is None:
        return None
    fields: dict[str, str] = {}
    for part in match.group("fields").split("|"):
        key, sep, value = part.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    missing = [key for key in _REQUIRED_ITERATION_KEYS if key not in fields]
    if missing:
        raise ValueError(f"iteration line lacks {missing}: {line.rstrip()!r}")
    return int(match.group("iteration")), match.group("timestamp"), fields


def parse_iteration_records(lines: Iterable[str]) -> list[IterationRecord]:
    """Parse every iteration line, in log order (repeats are kept so a caller can detect them).

    Raises ValueError on a line that has the iteration prefix but lacks a required field.
    """
    records = []
    for line in lines:
        parsed = _iteration_fields(line)
        if parsed is None:
            continue
        iteration, timestamp, fields = parsed
        throughput = fields.get(_THROUGHPUT_KEY)
        loss = fields.get(_LOSS_KEY)
        consumed = fields.get(_CONSUMED_SAMPLES_KEY)
        learning_rate = fields.get(_LEARNING_RATE_KEY)
        grad_norm = fields.get(_GRAD_NORM_KEY)
        records.append(
            IterationRecord(
                iteration=iteration,
                elapsed_ms=float(fields[_ELAPSED_KEY]),
                global_batch_size=int(fields[_GLOBAL_BATCH_KEY]),
                logged_tflops_per_gpu=float(throughput) if throughput is not None else None,
                lm_loss=float(loss) if loss is not None else None,
                skipped_total=int(fields[_SKIPPED_KEY]),
                nan_total=int(fields[_NAN_KEY]),
                timestamp=timestamp,
                consumed_samples=int(consumed) if consumed is not None else None,
                learning_rate=float(learning_rate) if learning_rate is not None else None,
                grad_norm=float(grad_norm) if grad_norm is not None else None,
            )
        )
    return records


def parse_first_iteration_memory(lines: Iterable[str]) -> dict[str, float] | None:
    """Return the ``-gigabytes`` fields of the after-iteration-1 memory report, or None if absent.

    One line is printed per data-parallel group (so several when TP, CP or PP > 1); each key takes
    its maximum across them, because the heaviest rank is the one that sets the memory ceiling.
    """
    peaks: dict[str, float] | None = None
    for line in lines:
        match = _MEMORY_RE.search(line)
        if match is None:
            continue
        values = {}
        for part in match.group("fields").split("|"):
            key, sep, value = part.partition(":")
            if sep and key.strip().endswith(GIGABYTES_SUFFIX):
                values[key.strip()] = float(value)
        if not values:
            raise ValueError(f"memory report carries no {GIGABYTES_SUFFIX} fields: {line.rstrip()!r}")
        if peaks is None:
            peaks = values
            continue
        for key, value in values.items():
            peaks[key] = max(value, peaks.get(key, value))
    return peaks


def parse_peak_memory_across_ranks(lines: Iterable[str]) -> dict[str, float | int] | None:
    """Return the end-of-training peak-memory summary over all ranks, or None when the log has none.

    None means the log predates the summary, or the run's loop was cut short (an OOM, a wedge, a SLURM
    wall-time kill; a loop ended by ``train.exit_duration_in_mins`` still prints it), and is never a pass.
    Raises ValueError when the log holds more than one summary, since the log then holds more than one run.
    """
    summaries = []
    for line in lines:
        match = _PEAK_MEMORY_RE.search(line)
        if match is None:
            continue
        summary: dict[str, float | int] = {}
        for field in match.group("fields").split():
            key, sep, value = field.partition("=")
            if not sep:
                raise ValueError(f"peak-memory field is not key=value: {field!r} in {line.rstrip()!r}")
            summary[key] = int(value) if value.lstrip("-").isdigit() else float(value)
        summaries.append(summary)
    if len(summaries) > 1:
        raise ValueError(f"log holds {len(summaries)} peak-memory summaries, so more than one run")
    return summaries[0] if summaries else None


def parse_wandb_run_path(lines: Iterable[str]) -> str | None:
    """Return ``<entity>/<project>/<run_id>`` from wandb's "View run" line, or None if absent.

    Raises ValueError when the log names more than one run, since its iterations could then belong to either.
    """
    paths = []
    for line in lines:
        match = _WANDB_RUN_RE.search(line)
        if match is not None:
            path = f"{match.group('entity')}/{match.group('project')}/{match.group('run_id')}"
            if path not in paths:
                paths.append(path)
    if len(paths) > 1:
        raise ValueError(f"log names several W&B runs: {paths}")
    return paths[0] if paths else None


def read_log_lines(log_path: Path) -> list[str]:
    """Return the log's lines, decoding a stray undecodable byte (interleaved native output) as U+FFFD.

    The replacement keeps one bad byte from stopping a read, while a replaced character inside a parsed
    field still fails that field's parse.
    """
    return Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()


def check_window(name: str, window: tuple[int, int], min_iterations: int) -> None:
    """Raise ValueError unless ``window`` (first, last; inclusive) starts at iteration >= 1 and spans at least
    ``min_iterations`` iterations."""
    first, last = window
    if first < 1 or last - first + 1 < min_iterations:
        raise ValueError(f"{name} {window} must start at >= 1 and span >= {min_iterations} iteration(s)")


def window_records(records: list[IterationRecord], window: tuple[int, int], label: str) -> list[IterationRecord]:
    """Return the records of ``window`` (inclusive) in iteration order; raise unless each appears exactly once."""
    first, last = window
    inside = [r for r in records if first <= r.iteration <= last]
    counts = Counter(r.iteration for r in inside)
    missing = [i for i in range(first, last + 1) if counts[i] == 0]
    repeated = sorted(i for i, count in counts.items() if count > 1)
    if missing or repeated:
        raise ValueError(
            f"{label} {first}-{last}: every iteration must be logged exactly once; "
            f"missing {missing or 'none'}, repeated {repeated or 'none'}"
        )
    return sorted(inside, key=lambda r: r.iteration)

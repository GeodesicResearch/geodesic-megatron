"""Parse a training log as the launcher writes it: its iteration lines, its evaluation lines, the after-iteration-1
memory report, the end-of-training peak-memory summary over all ranks, the banners each node logs (the launcher env
overrides, the token-masking decision), the per-iteration token-masking counts and the W&B run it names.

The iteration line is the bridge's ``training/utils/train_utils.py::training_log`` output, the evaluation line
``training/eval.py::evaluate_and_print_results``'s. The parser needs only the standard library, so it runs under a
login node's host interpreter. ``scripts/telemetry/score_run.py`` scores one run from these records,
``scripts/telemetry/loss_parity.py`` compares several runs' trajectories and ``scripts/telemetry/score_gate.py`` gates
runs' token-masking reports.

Token masking reports, per iteration and per evaluation, the cross-entropy at the targets of its measured ids
(``token_masking/listed_target_loss``) beside the loss of the targets that carry loss (``lm loss``); with the fractions
it reports, the two give the cross-entropy at every other target that carries loss, which this module derives as
``token_masking/non_listed_target_loss`` (``non_listed_target_loss``): the loss a masked run and its unmasked control
share, whether or not the masked ids' targets are in ``lm loss``.

The fractions on the iteration line are printed to 7 significant digits. The exact integer counts behind them are
printed on a line of their own every iteration (``[token-masking-counts] iteration=<i> listed=<n> ...``, the bridge's
``training/token_masking/monitor.py``), which ``parse_token_masking_counts`` reads: a comparison that must be exact
reads those.
"""

import re
import shlex
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar


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

# A banner logged once per node (training/utils/log_utils.py::log_node_banner, or the same format written by
# pipeline_training_run.py's log_env_overrides): "[<tag>] rank=<R> host=<host> KEY=<value> ...", every value
# shell-quoted, behind the logger's prefix. The tags the telemetry reads are restated rather than imported, because
# this module must import without torch.
ENV_OVERRIDES_TAG = "env-overrides"
TOKEN_MASKING_TAG = "token-masking"


def _banner_re(tag: str) -> re.Pattern[str]:
    return re.compile(rf"\[{re.escape(tag)}\] rank=(?P<rank>\S+) host=(?P<host>\S+) (?P<values>.*)$")


_ENV_OVERRIDES_RE = _banner_re(ENV_OVERRIDES_TAG)

# evaluate_and_print_results' line, e.g.
#  validation loss at iteration 53 | masked-validation/lm loss value: 2.1E+00 | masked-validation/lm loss PPL: ... |
_VALIDATION_RE = re.compile(r"validation loss at (?P<label>.+?) \| (?P<fields>.*)$")
_VALIDATION_FIELD_RE = re.compile(r"(?P<key>.+) value: (?P<value>\S+)")
_STEP_LABEL_RE = re.compile(r"iteration (?P<step>\d+)")

# The token-masking report keys (training/token_masking/hook.py), restated because this module must import without
# torch, and the loss this module derives from them.
LISTED_TARGET_FRACTION = "token_masking/listed_target_fraction"
MASKED_TARGET_FRACTION = "token_masking/masked_target_fraction"
TRAINED_LISTED_TARGET_FRACTION = "token_masking/trained_listed_target_fraction"
TRAINABLE_TARGET_FRACTION = "token_masking/trainable_target_fraction"
LISTED_TRAINABLE_TARGET_FRACTION = "token_masking/listed_trainable_target_fraction"
LISTED_TARGET_LOSS = "token_masking/listed_target_loss"
NON_LISTED_TARGET_LOSS = "token_masking/non_listed_target_loss"

# The per-iteration counts line (training/token_masking/monitor.py, COUNTS_LOG_TAG, with the fields of
# training/token_masking/hook.py's TokenMaskingCounts in their order), restated because this module must import without
# torch, e.g.
#  INFO:...monitor:[token-masking-counts] iteration=5 listed=12 listed_trainable=12 masked=12 trained_listed=0 ...
# The rank that writes the training log prints it once per iteration. The counts are W&B metrics under
# ``token_masking/count/<field>``, the names a gate uses for them (``TOKEN_MASKING_COUNT_PREFIX``).
TOKEN_MASKING_COUNTS_TAG = "token-masking-counts"
TOKEN_MASKING_COUNT_FIELDS = ("listed", "listed_trainable", "masked", "trained_listed", "trainable", "positions")
TOKEN_MASKING_COUNT_PREFIX = "token_masking/count/"
_COUNTS_RE = re.compile(rf"\[{re.escape(TOKEN_MASKING_COUNTS_TAG)}\] iteration=(?P<iteration>\d+) (?P<fields>.*)$")
_COUNT_VALUE_RE = re.compile(r"[0-9]+")

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


def iteration_of(line: str) -> int | None:
    """The iteration an iteration line names, read from its prefix alone; None for any other line."""
    match = _ITERATION_RE.search(line)
    return int(match.group("iteration")) if match is not None else None


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


@dataclass(frozen=True)
class IterationValues:
    """Every numeric field of one iteration line, by its logged name, plus ``token_masking/non_listed_target_loss``
    where the line's token-masking reports give it (``non_listed_target_loss``)."""

    iteration: int
    values: dict[str, float]


@dataclass(frozen=True)
class ValidationRecord:
    """One evaluation line: the label it was printed at (``iteration <N>`` in the training loop), the step that label
    names (None for any other label), and each result's value by its logged name, plus every
    ``<prefix>token_masking/non_listed_target_loss`` the results give (``non_listed_target_loss``)."""

    label: str
    step: int | None
    values: dict[str, float]


def _numeric(raw: str) -> float | None:
    try:
        return float(raw)
    except ValueError:
        return None


def non_listed_target_loss(values: Mapping[str, float], prefix: str = "") -> float | None:
    """The mean cross-entropy at the targets that carry loss and are not a measured id's, from one report's values.

    ``<prefix>lm loss`` averages over the targets that carry loss in the mask the loss multiplies, a fraction T of the
    report's target positions (``trainable_target_fraction``); a fraction W of them are measured ids' targets
    (``trained_listed_target_fraction``), whose mean cross-entropy is ``listed_target_loss``. The other targets
    therefore average (lm loss x T - listed_target_loss x W) / (T - W); with masking enabled W is 0 and this is
    ``lm loss`` itself. The identity is exact for one report: an iteration line with ``log_interval: 1``, or one
    evaluation.

    Returns None when the values lack the loss or either fraction (a run without token masking), lack
    ``listed_target_loss`` while W > 0 (a log written before that report existed), or no other target carries loss.
    """
    loss = values.get(f"{prefix}{_LOSS_KEY}")
    trainable = values.get(f"{prefix}{TRAINABLE_TARGET_FRACTION}")
    trained_listed = values.get(f"{prefix}{TRAINED_LISTED_TARGET_FRACTION}")
    if loss is None or trainable is None or trained_listed is None or trainable <= trained_listed:
        return None
    if trained_listed == 0:
        return loss
    listed_loss = values.get(f"{prefix}{LISTED_TARGET_LOSS}")
    if listed_loss is None:
        return None
    return (loss * trainable - listed_loss * trained_listed) / (trainable - trained_listed)


def _with_non_listed_target_loss(values: dict[str, float], prefixes: Iterable[str]) -> dict[str, float]:
    for prefix in prefixes:
        derived = non_listed_target_loss(values, prefix)
        if derived is not None:
            values[f"{prefix}{NON_LISTED_TARGET_LOSS}"] = derived
    return values


def parse_iteration_values(lines: Iterable[str]) -> list[IterationValues]:
    """Every iteration line's numeric fields, in log order (repeats are kept so a caller can detect them).

    Raises ValueError on a line that has the iteration prefix but lacks a required field.
    """
    records = []
    for line in lines:
        parsed = _iteration_fields(line)
        if parsed is None:
            continue
        iteration, _, fields = parsed
        values = {key: number for key, raw in fields.items() if (number := _numeric(raw)) is not None}
        records.append(IterationValues(iteration, _with_non_listed_target_loss(values, [""])))
    return records


def parse_validation_records(lines: Iterable[str]) -> list[ValidationRecord]:
    """Every evaluation line's results, in log order.

    Raises ValueError on a result whose value is not a number.
    """
    records = []
    for line in lines:
        match = _VALIDATION_RE.search(line.rstrip("\n"))
        if match is None:
            continue
        values: dict[str, float] = {}
        for part in match.group("fields").split(" | "):
            field = _VALIDATION_FIELD_RE.fullmatch(part.strip(" |"))
            if field is None:
                continue
            number = _numeric(field.group("value"))
            if number is None:
                raise ValueError(f"evaluation result is not a number: {part.strip()!r} in {line.rstrip()!r}")
            values[field.group("key")] = number
        prefixes = [key[: -len(_LOSS_KEY)] for key in values if key.endswith(_LOSS_KEY)]
        label = match.group("label")
        step = _STEP_LABEL_RE.fullmatch(label)
        records.append(
            ValidationRecord(
                label,
                int(step.group("step")) if step is not None else None,
                _with_non_listed_target_loss(values, prefixes),
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


def env_override_lines(lines: Iterable[str]) -> list[str]:
    """Return the ``[env-overrides]`` lines among ``lines``, as logged, in log order."""
    return [line.rstrip("\n") for line in lines if _ENV_OVERRIDES_RE.search(line.rstrip("\n"))]


@dataclass(frozen=True)
class NodeBanner:
    """One node's banner line: the rank and host that logged it and its KEY to value fields, unquoted."""

    rank: str
    host: str
    fields: dict[str, str]


def parse_node_banners(lines: Iterable[str], tag: str) -> list[NodeBanner]:
    """Return each ``[<tag>]`` banner line's rank, host and fields, in log order: one line per node per start of
    training, values unquoted as the shell would. Raises ValueError on a field that is not KEY=value."""
    pattern = _banner_re(tag)
    banners = []
    for line in lines:
        match = pattern.search(line.rstrip("\n"))
        if match is None:
            continue
        fields = {}
        for field in shlex.split(match.group("values")):
            key, sep, value = field.partition("=")
            if not sep:
                raise ValueError(f"{tag} field is not KEY=value: {field!r} in {line.rstrip()!r}")
            fields[key] = value
        banners.append(NodeBanner(match.group("rank"), match.group("host"), fields))
    return banners


@dataclass(frozen=True)
class TokenMaskingCountsRecord:
    """One iteration's ``[token-masking-counts]`` line: exact integers over the global batch's target positions.

    ``listed`` counts the targets whose label is a measured id, ``listed_trainable`` those of them the dataset trains,
    ``masked`` those token masking removed from the loss, ``trained_listed`` the listed targets that still carry loss,
    ``trainable`` the targets that carry loss and ``positions`` every target position (the bridge's
    ``TokenMaskingCounts``).
    """

    iteration: int
    listed: int
    listed_trainable: int
    masked: int
    trained_listed: int
    trainable: int
    positions: int

    def metrics(self) -> dict[str, int]:
        """The counts by their metric names, ``token_masking/count/<field>``."""
        return {f"{TOKEN_MASKING_COUNT_PREFIX}{name}": getattr(self, name) for name in TOKEN_MASKING_COUNT_FIELDS}


def parse_token_masking_counts(lines: Iterable[str]) -> list[TokenMaskingCountsRecord]:
    """Each iteration's token-masking counts, in iteration order, one record per iteration.

    A logging handler that duplicates the line (two handlers on one logger, say) prints it more than once for one
    iteration; lines with the same counts are one record. So the counts cannot show that an iteration ran only once: a
    restart that repeats an iteration on the same batch repeats its counts too. The iteration lines show it
    (``window_records``).

    Raises ValueError on a line whose fields are not exactly the six counts, each a non-negative integer, and on two
    lines with different counts for one iteration: the log then holds more than one run, or a corrupted line.
    """
    records: dict[int, TokenMaskingCountsRecord] = {}
    for line in lines:
        match = _COUNTS_RE.search(line.rstrip("\n"))
        if match is None:
            continue
        counts: dict[str, int] = {}
        for field in match.group("fields").split():
            key, sep, value = field.partition("=")
            if not sep or _COUNT_VALUE_RE.fullmatch(value) is None or key in counts:
                raise ValueError(f"token-masking count is not one name=<non-negative integer>: {field!r} in {line!r}")
            counts[key] = int(value)
        if set(counts) != set(TOKEN_MASKING_COUNT_FIELDS):
            raise ValueError(
                f"token-masking counts line holds {sorted(counts)}, not {sorted(TOKEN_MASKING_COUNT_FIELDS)}: {line!r}"
            )
        record = TokenMaskingCountsRecord(int(match.group("iteration")), **counts)
        earlier = records.setdefault(record.iteration, record)
        if earlier != record:
            raise ValueError(
                f"two different token-masking counts lines for iteration {record.iteration}: {earlier} and {record}"
            )
    return [records[iteration] for iteration in sorted(records)]


def parse_env_override_lines(lines: Iterable[str]) -> list[dict[str, str]]:
    """Return each ``[env-overrides]`` line's KEY to value mapping, in log order: one line per node per start of
    training, values unquoted as the shell would. Raises ValueError on a field that is not KEY=value."""
    return [banner.fields for banner in parse_node_banners(lines, ENV_OVERRIDES_TAG)]


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
    field still fails that field's parse. A last line without a line break is left out: its writer has not
    finished it (a live log) or was killed part-way through it.
    """
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    return lines if text.endswith("\n") else lines[:-1]


def check_window(name: str, window: tuple[int, int], min_iterations: int) -> None:
    """Raise ValueError unless ``window`` (first, last; inclusive) starts at iteration >= 1 and spans at least
    ``min_iterations`` iterations."""
    first, last = window
    if first < 1 or last - first + 1 < min_iterations:
        raise ValueError(f"{name} {window} must start at >= 1 and span >= {min_iterations} iteration(s)")


class _HasIteration(Protocol):
    @property
    def iteration(self) -> int: ...


_Record = TypeVar("_Record", bound=_HasIteration)


def window_records(records: Sequence[_Record], window: tuple[int, int], label: str) -> list[_Record]:
    """Return the records (``IterationRecord`` or ``IterationValues``) of ``window`` (inclusive) in iteration order;
    raise unless each appears exactly once."""
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

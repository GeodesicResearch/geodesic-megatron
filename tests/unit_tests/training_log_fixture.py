"""The real training-log excerpts the telemetry tests read, and helpers that edit their lines.

``fixtures/score_run/train-6354507_excerpt.out`` is verbatim text from the filtered stage-1 pretraining log
``/projects/a5k/public/logs/megatron_runs/train-6354507.out`` (256 GPUs, GBS 2048, seq 8192): its two wandb init
lines (7066-7067), the theoretical-memory and after-iteration-1 memory lines plus iterations 1-60 with the
interleaved "Step Time" lines (7978-8098), and wandb's end-of-run "View run" line (27796). Synthetic cases are
built by editing a real line from it, so every parse runs on the format the bridge actually prints.

``fixtures/score_run/train-7146597_token_masking_excerpt.out`` is verbatim lines 5379 and 5387 of
``/projects/a5k/public/logs/megatron_runs/train-7146597.out``, the Nano token-masking canary's measure-only control
(iterations 1 and 2, each carrying the ``token_masking/*`` reports). Token-masking lines are built by editing its
iteration-2 line, inserting the two reports that came later (the listed trainable fraction and the listed-target
loss) after the trainable fraction, where the reports' order puts them. The ``[token-masking]`` banner is made by the
bridge's own ``banner_fields`` and ``log_node_banner``, and the ``[token-masking-counts]`` line by its
``TokenMaskingMonitor``; the evaluation line follows ``training/eval.py``'s ``evaluate_and_print_results``.
"""

import logging
import logging.handlers
import math
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "unit_tests" / "fixtures" / "score_run" / "train-6354507_excerpt.out"
FIXTURE_LINES = FIXTURE.read_text(encoding="utf-8").splitlines()
FIXTURE_RUN_PATH = "geodesic/megatron_training/5s8x5mgb"


def only_line(needle: str) -> str:
    """The one fixture line containing ``needle``."""
    (line,) = [line for line in FIXTURE_LINES if needle in line]
    return line


def sub_once(pattern: str, replacement: str, text: str) -> str:
    """``re.sub`` that must match exactly once."""
    edited, count = re.subn(pattern, replacement, text)
    assert count == 1, f"{pattern!r} matched {count} times"
    return edited


REAL_ITERATION_50 = only_line("iteration       50/")
REAL_MEMORY_LINE = only_line("(after 1 iterations) memory (GB)")


def iteration_line(iteration: int, elapsed_ms: float, lm_loss: float, skipped: int = 0, nan: int = 0) -> str:
    """The real iteration-50 line with its iteration number, step time, loss and skipped/NaN counts replaced."""
    line = sub_once(r"iteration\s+50/", f"iteration {iteration:8d}/", REAL_ITERATION_50)
    line = sub_once(r"\(ms\): [\d.]+", f"(ms): {elapsed_ms:.1f}", line)
    line = sub_once(r"lm loss: [\dE.+-]+", f"lm loss: {lm_loss:.6E}", line)
    line = sub_once(r"skipped iterations:\s+\d+", f"skipped iterations: {skipped:3d}", line)
    return sub_once(r"nan iterations:\s+\d+", f"nan iterations: {nan:3d}", line)


def write_log(tmp_path: Path, lines: list[str], name: str = "train.out") -> Path:
    """Write ``lines`` as a log file under ``tmp_path``."""
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


TOKEN_MASKING_FIXTURE = FIXTURE.with_name("train-7146597_token_masking_excerpt.out")
(REAL_CONTROL_ITERATION_2,) = [
    line for line in TOKEN_MASKING_FIXTURE.read_text(encoding="utf-8").splitlines() if "iteration        2/" in line
]


def token_masking_iteration_line(
    iteration: int,
    *,
    enabled: bool,
    listed: float,
    lm_loss: float,
    listed_loss: float | None,
    listed_trainable: float | None = None,
) -> str:
    """The control canary's real iteration-2 line as an iteration of a pretraining run (every target trainable)
    that masks, or only measures, ids at a fraction ``listed`` of its targets, with ``listed_trainable`` of them
    (default: all) carrying loss in the dataset's mask and the given losses; ``listed_loss`` None leaves the
    listed-target loss out, as an iteration without a trainable listed target reports it."""
    listed_trainable = listed if listed_trainable is None else listed_trainable
    line = sub_once(r"iteration\s+2/\s+20", f"iteration {iteration:8d}/{1000:8d}", REAL_CONTROL_ITERATION_2)
    line = sub_once(r"lm loss: [\dE.+-]+", f"lm loss: {lm_loss:.6E}", line)
    reports = {
        "listed_target_fraction": listed,
        "masked_target_fraction": listed_trainable if enabled else 0.0,
        "trained_listed_target_fraction": 0.0 if enabled else listed_trainable,
        "trainable_target_fraction": 1.0 - listed_trainable if enabled else 1.0,
    }
    for key, value in reports.items():
        line = sub_once(rf"token_masking/{key}: [\dE.+-]+", f"token_masking/{key}: {value:.6E}", line)
    added = f" | token_masking/listed_trainable_target_fraction: {listed_trainable:.6E}"
    if listed_loss is not None:
        added += f" | token_masking/listed_target_loss: {listed_loss:.6E}"
    return sub_once(r"(token_masking/trainable_target_fraction: [\dE.+-]+)", rf"\g<1>{added}", line)


def _logged_line(logger_name: str, emit) -> str:
    """The one line ``emit()`` logs on the logger ``logger_name``, with the prefix the launcher's logging configuration
    gives every line (its level and logger, as basicConfig does)."""
    handler = logging.handlers.BufferingHandler(capacity=8)
    logger = logging.getLogger(logger_name)
    logger.addHandler(handler)
    level = logger.level
    logger.setLevel(logging.INFO)
    try:
        emit(logger)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
    (record,) = handler.buffer
    return logging.Formatter(logging.BASIC_FORMAT).format(record)


def token_masking_banner(enabled: bool, token_ids: list[int], host: str, rank: int) -> str:
    """A ``[token-masking]`` banner line as a node logs it, for a run masking ``token_ids`` or only measuring them,
    with the logger's prefix and the given host."""
    from megatron.bridge.training.token_masking.resolution import banner_fields
    from megatron.bridge.training.utils.log_utils import log_node_banner
    from tests.unit_tests.token_masking_fixtures import masking_with_null_tokenizer, measuring_with_null_tokenizer

    resolve = masking_with_null_tokenizer if enabled else measuring_with_null_tokenizer
    resolved = resolve(token_ids, vocab_size=max(token_ids) + 1)
    line = _logged_line(
        "megatron.bridge.training.token_masking.resolution",
        lambda logger: log_node_banner(
            logger, "token-masking", banner_fields(resolved, None), rank=rank, local_rank=0
        ),
    )
    return sub_once(r" host=\S+ ", f" host={host} ", line)


def token_masking_counts_line(
    iteration: int,
    *,
    listed: int,
    listed_trainable: int,
    masked: int,
    trained_listed: int,
    trainable: int,
    positions: int,
) -> str:
    """The ``[token-masking-counts]`` line the rank that writes the training log prints for one iteration's counts,
    with the logger's prefix: printed by the bridge's own monitor."""
    from megatron.bridge.training.token_masking.hook import TokenMaskingCounts
    from megatron.bridge.training.token_masking.monitor import TokenMaskingMonitor
    from tests.unit_tests.token_masking_fixtures import measuring_with_null_tokenizer

    counts = TokenMaskingCounts(listed, listed_trainable, masked, trained_listed, trainable, positions)
    # The monitor prints the line in _record_counts once observe() has read the counts from the iteration's reduced
    # reports; calling it with the counts directly prints exactly that line without building the reports. A run that
    # measures no id prints none, hence a measuring decision; the pipeline group is read only by the verdict.
    monitor = TokenMaskingMonitor(measuring_with_null_tokenizer([7], vocab_size=8), None, None, log_counts=True)
    return _logged_line(
        "megatron.bridge.training.token_masking.monitor", lambda logger: monitor._record_counts(counts, iteration)
    )


def validation_line(step: int, results: dict[str, float]) -> str:
    """An evaluation line as ``evaluate_and_print_results`` prints it after step ``step``: each result's value, then
    a perplexity after each loss (every result outside ``token_masking/`` and the listed-target loss)."""
    line = f" validation loss at iteration {step} | "
    for key, value in results.items():
        line += f"{key} value: {value:.6E} | "
        if "token_masking/" not in key or key.endswith("listed_target_loss"):
            line += f"{key} PPL: {math.exp(min(20.0, value)):.6E} | "
    return line

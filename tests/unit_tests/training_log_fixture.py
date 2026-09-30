"""The real training-log excerpt the telemetry tests read, and helpers that edit its lines.

``fixtures/score_run/train-6354507_excerpt.out`` is verbatim text from the filtered stage-1 pretraining log
``/projects/a5k/public/logs/megatron_runs/train-6354507.out`` (256 GPUs, GBS 2048, seq 8192): its two wandb init
lines (7066-7067), the theoretical-memory and after-iteration-1 memory lines plus iterations 1-60 with the
interleaved "Step Time" lines (7978-8098), and wandb's end-of-run "View run" line (27796). Synthetic cases are
built by editing a real line from it, so every parse runs on the format the bridge actually prints.
"""

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


def write_log(tmp_path: Path, lines: list[str]) -> Path:
    """Write ``lines`` as a log file under ``tmp_path``."""
    path = tmp_path / "train.out"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

"""Automated pass/fail gate for the Nemotron Nano quickstart SFT run.

The training pipeline logs every iteration to Weights & Biases, so the W&B run
is the source of truth for "did the quickstart pass". These tests read the live
run via the W&B API and assert it finished cleanly with healthy metrics.

Pass the run path via the ``QUICKSTART_WANDB_RUN`` environment variable. It
accepts either ``entity/project/run_id`` or a full ``wandb.ai`` run URL::

    QUICKSTART_WANDB_RUN=geodesic/megatron_training/1ov2i01u \
        uv run pytest tests/quickstart/test_quickstart_wandb_metrics.py -v

These are real-data tests (no mocks): they hit the live W&B API and read the
actual logged history. While the job is still running they are EXPECTED to
fail — the run state is not yet ``finished`` and fewer than the expected number
of iterations are logged. They pass once the run completes with healthy metrics.

Tuning knobs (env overrides, for non-quickstart run shapes):
  QUICKSTART_EXPECTED_ITERS   expected train_iters          (default 200)
  QUICKSTART_MIN_TFLOPS       median device throughput floor (default 20)
  QUICKSTART_MAX_GRAD_NORM    grad-norm divergence ceiling  (default 100)
"""

from __future__ import annotations

import math
import os
import re

import pytest


# --- W&B metric keys actually logged by the training pipeline ---
LOSS_KEY = "lm loss"
GRAD_NORM_KEY = "grad-norm"
TFLOPS_KEY = "throughput/tflops/device"

# --- Pass criteria (override via env for other run sizes) ---
EXPECTED_ITERATIONS = int(os.environ.get("QUICKSTART_EXPECTED_ITERS", "200"))
# Floor is well under the observed ~55 TFLOP/s/GPU so a slow-but-correct run is
# not failed; it only catches a fundamentally broken throughput (e.g. ~1).
MIN_MEDIAN_TFLOPS_PER_DEVICE = float(os.environ.get("QUICKSTART_MIN_TFLOPS", "20"))
# A finite grad norm may spike, but values past this signal divergence.
MAX_GRAD_NORM = float(os.environ.get("QUICKSTART_MAX_GRAD_NORM", "100"))

RUN_PATH_ENV = "QUICKSTART_WANDB_RUN"


def _parse_run_path(raw: str) -> str:
    """Normalise an ``entity/project/run_id`` triple from a path or wandb.ai URL."""
    raw = raw.strip().rstrip("/")
    m = re.search(r"wandb\.ai/([^/]+)/([^/]+)/runs/([^/?#]+)", raw)
    if m:
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    parts = raw.split("/")
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"Expected 'entity/project/run_id' or a wandb.ai run URL, got: {raw!r}")
    return raw


def _values(rows: list[dict], key: str) -> list[float]:
    """All logged values for ``key`` as floats (unparseable -> NaN, so the
    finiteness checks catch them rather than silently dropping the iteration)."""
    out: list[float] = []
    for row in rows:
        v = row.get(key)
        if v is None:
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


@pytest.fixture(scope="module")
def run_path() -> str:
    raw = os.environ.get(RUN_PATH_ENV)
    if not raw:
        pytest.skip(f"set {RUN_PATH_ENV}=entity/project/run_id (or a wandb.ai run URL) to run these checks")
    return _parse_run_path(raw)


@pytest.fixture(scope="module")
def wandb_run(run_path: str):
    wandb = pytest.importorskip("wandb")
    api = wandb.Api()
    try:
        return api.run(run_path)
    except Exception as exc:  # noqa: BLE001 - surface the real fetch failure, don't mask it
        pytest.fail(f"could not fetch W&B run {run_path!r}: {type(exc).__name__}: {exc}")


@pytest.fixture(scope="module")
def history(wandb_run) -> list[dict]:
    """Full, unsampled per-iteration history.

    Uses ``scan_history`` (not the sampled ``history``) so a NaN in any single
    iteration cannot slip through a downsample.
    """
    return list(wandb_run.scan_history(keys=[LOSS_KEY, GRAD_NORM_KEY, TFLOPS_KEY]))


def test_run_finished(wandb_run):
    """Run reached a clean terminal state (fails while still running, or if it crashed)."""
    assert wandb_run.state == "finished", (
        f"run state is {wandb_run.state!r}, expected 'finished' (still running, or crashed/failed/killed)"
    )


def test_reached_expected_iterations(history):
    """The run logged the full configured number of training iterations."""
    losses = _values(history, LOSS_KEY)
    assert len(losses) >= EXPECTED_ITERATIONS, (
        f"only {len(losses)} iterations logged '{LOSS_KEY}'; expected >= {EXPECTED_ITERATIONS}"
    )


def test_no_nan_or_inf_loss(history):
    """No iteration produced a non-finite loss (the real signal for a NaN step)."""
    losses = _values(history, LOSS_KEY)
    if not losses:
        pytest.skip("no loss logged yet")
    bad = [(i, v) for i, v in enumerate(losses) if not math.isfinite(v)]
    assert not bad, f"non-finite '{LOSS_KEY}' at iteration indices {bad[:5]}"


def test_loss_decreased(history):
    """Training actually reduced the loss (end window mean below start window mean)."""
    losses = [v for v in _values(history, LOSS_KEY) if math.isfinite(v)]
    if len(losses) < 20:
        pytest.skip(f"only {len(losses)} finite loss points; need >= 20 to compare windows")
    window = max(5, len(losses) // 10)
    first = sum(losses[:window]) / window
    last = sum(losses[-window:]) / window
    assert last < first, (
        f"mean loss did not decrease: first {window} iters={first:.4f}, last {window} iters={last:.4f}"
    )


def test_grad_norm_finite_and_bounded(history):
    """Grad norm stayed finite and never exploded (divergence guard)."""
    grad_norms = _values(history, GRAD_NORM_KEY)
    if not grad_norms:
        pytest.skip("no grad-norm logged yet")
    bad = [(i, v) for i, v in enumerate(grad_norms) if not math.isfinite(v)]
    assert not bad, f"non-finite '{GRAD_NORM_KEY}' at iteration indices {bad[:5]}"
    worst = max(grad_norms)
    assert worst < MAX_GRAD_NORM, f"grad-norm spiked to {worst:.2f} (>= {MAX_GRAD_NORM}) — divergence"


def test_throughput_reasonable(history):
    """Median device throughput is in a sane range (catches a fundamentally broken run)."""
    tflops = sorted(v for v in _values(history, TFLOPS_KEY) if math.isfinite(v) and v > 0)
    if not tflops:
        pytest.skip("no throughput logged yet")
    median = tflops[len(tflops) // 2]
    assert median >= MIN_MEDIAN_TFLOPS_PER_DEVICE, (
        f"median device throughput {median:.1f} TFLOP/s/GPU < floor {MIN_MEDIAN_TFLOPS_PER_DEVICE}"
    )

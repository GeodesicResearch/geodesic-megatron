"""Score one training run's log: step-time statistics, throughput, MFU, loss and memory.

Performance comparisons between runs are about what a node-hour buys, so the
PRIMARY statistic is the MEAN step time over a fixed iteration window: node-hours scale
with total time, which is the mean, while a median hides periodic excursions (an automatic
GC every ~50 iterations, a Lustre stall) that the run still pays for. The median, p10/p90,
min/max and the outlier iterations (step > 2x the window median) are reported beside it so
an excursion is visible and can be attributed rather than averaged away unseen.

    tokens/iteration    = logged global batch size * seq_length
    tokens/s/GPU        = tokens/iteration / (num_gpus * mean_step_s)
    model TFLOP/s/GPU   = tokens/iteration * model_flops_per_token / (num_gpus * mean_step_s) / 1e12
    MFU                 = model TFLOP/s/GPU / peak TFLOP/s/GPU

The sequence length and ``model_flops_per_token`` come from the run's training config (read
through its ``base_config`` chain) and its model's HF ``config.json``, through the library of
``scripts/nemotronh_flops_estimator.py``: the MODEL (not hardware) FLOPs per token at that
sequence length, so MFU is invariant to the recompute posture. The throughput arithmetic is the
estimator's ``achieved_throughput``, and the peak is its ``DEFAULT_PEAK_TFLOPS`` (GH200 dense
BF16) unless ``--peak-tflops`` overrides it. The global batch size is the one the scored
iterations' own log lines report, so a batch set by a Hydra override is scored correctly.

Every score records its inputs beside its results: the log, the config and HF config it read,
the GPU count, sequence length, logged batch, FLOPs per token, peak and both windows.

Every iteration in the scoring window (and in the loss window) must appear in the log
exactly once: a missing iteration means the run did not reach the window or logs at an
interval > 1, and a repeated one means two runs share the log. Either makes the statistic
meaningless, so both raise.

The scorer needs PyYAML (the estimator's config reader) and otherwise only the standard library
(no numpy), so the CLI runs on a login node's host interpreter; ``--wandb-peak-memory`` also
needs the ``wandb`` client and network access.

Only one memory figure is a maximum over all ranks: the peak-memory summary rank 0 logs when the
training loop ends (``peak_memory_across_ranks``; absent from a log that predates the summary and from a
run whose loop was cut short by an OOM, a wedge or a SLURM wall-time kill).
The log's after-iteration-1 report comes from the ranks that print it, and the W&B summary from the one
rank that owns the W&B run (the last global rank, ``src/megatron/bridge/training/state.py``), so it is
that rank's peak. The allocator-retry count ``--wandb-peak-memory`` reports beside the peaks is the same
rank's total.

USAGE
    python scripts/telemetry/score_run.py <log> \\
        --config configs/quickstart/nemotron_nano_quickstart_pretrain.yaml \\
        --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \\
        --gpus 64 --window 26 50 --loss-window 41 50 [--peak-tflops TF] [--wandb-peak-memory] [--json]
"""

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


# Run as a script, only scripts/telemetry/ is on sys.path; the repo root makes the FLOPs estimator
# (scripts.nemotronh_flops_estimator) and the log parser (scripts.telemetry.training_log) importable the same
# way from every entry point.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from scripts.nemotronh_flops_estimator import (
    DEFAULT_PEAK_TFLOPS,
    ArchSpec,
    RunSpec,
    achieved_throughput,
    compute_flops,
    resolve_hf_config,
)
from scripts.telemetry.training_log import (
    GIGABYTES_SUFFIX,
    check_window,
    parse_first_iteration_memory,
    parse_iteration_records,
    parse_peak_memory_across_ranks,
    parse_wandb_run_path,
    read_log_lines,
    window_records,
)


# Summary keys the bridge logs to W&B from torch.cuda.memory_stats(), in decimal GB. They are the
# allocator's peaks since process start on the rank that logs to W&B (the last global rank), so
# the summary's last value is that rank's peak over the run.
WANDB_PEAK_MEMORY_KEYS = ("memory/mem-max-allocated-gigabytes", "memory/mem-max-reserved-gigabytes")
# The same rank's count of cudaMalloc retries (the allocator freed its cache and tried again), cumulative
# since process start, so the summary's last value is the run's total. "retires" is the bridge's spelling.
WANDB_ALLOC_RETRIES_KEY = "memory/mem-alloc-retires"


@dataclass(frozen=True)
class Workload:
    """What a score takes from the run's training config, and where it was read from.

    ``model_flops_per_token`` is the estimator's MODEL FLOPs per token (forward + backward, no
    recompute) at ``seq_length``, for the architecture in the HF config at ``hf_config_path``.
    """

    config_path: str
    hf_config_path: str
    seq_length: int
    model_flops_per_token: float


@dataclass(frozen=True)
class RunScore:
    """The score of one run over its scoring window, with the inputs it was computed from.

    The fields up to ``loss_window_last`` are the inputs: the log, the workload read from the
    training config, the GPU count, the global batch size the window's lines report (and the tokens
    per iteration it makes), the peak and both windows. ``mean_step_s`` is the primary statistic.
    ``skipped_total``/``nan_total`` sum every iteration line in the log, not only the window, because
    a skipped or NaN iteration anywhere voids the run's loss parity. ``last_iteration`` is the highest
    iteration logged. ``first_iteration_memory_gb`` holds the ``-gigabytes`` fields of the
    after-iteration-1 memory report (the per-key maximum across the ranks that print it), or None when
    the log has no such report. ``peak_memory_across_ranks`` is the end-of-training summary over every rank
    (the largest peak allocated and reserved memory, the rank holding the allocated one, the largest and the
    total allocator retries), or None when the log has none. ``wandb_peak_memory_gb`` and
    ``wandb_alloc_retries`` hold the W&B summary
    peaks and allocator-retry count when the CLI is asked for them (``--wandb-peak-memory``), and None
    otherwise.
    """

    log_path: str
    workload: Workload
    num_gpus: int
    global_batch_size: int
    tokens_per_iteration: int
    peak_tflops_per_gpu: float
    window_first: int
    window_last: int
    loss_window_first: int
    loss_window_last: int
    n_iterations: int
    mean_step_s: float
    median_step_s: float
    p10_step_s: float
    p90_step_s: float
    min_step_s: float
    max_step_s: float
    outlier_iterations: list[int]
    tokens_per_s_per_gpu: float
    model_tflops_per_gpu: float
    mfu: float
    loss_mean: float
    skipped_total: int
    nan_total: int
    last_iteration: int
    first_iteration_memory_gb: dict[str, float] | None
    peak_memory_across_ranks: dict[str, float | int] | None
    wandb_run_path: str | None
    wandb_peak_memory_gb: dict[str, float] | None
    wandb_alloc_retries: int | None

    def to_dict(self) -> dict[str, Any]:
        """Return the score as a JSON-serialisable dict."""
        return asdict(self)


def workload_from_config(config_path: str | Path, hf_model: str) -> Workload:
    """Read the workload from a training YAML (through its ``base_config`` chain) and the model's HF config.

    ``hf_model`` is what the estimator's ``--hf-model`` takes: an HF repo id resolved offline from the
    local HF cache, a model directory, or a ``config.json`` path. The FLOPs follow the estimator's
    default conventions (causal attention mask, backward = 2x forward). Raises FileNotFoundError when
    the HF config cannot be found, and ValueError when the YAML sets no sequence length or batch size.
    """
    hf_config, hf_config_path = resolve_hf_config(hf_model)
    run = RunSpec.from_yaml(str(config_path))
    report = compute_flops(ArchSpec.from_hf_config(hf_config, name=hf_model), run)
    return Workload(
        config_path=str(config_path),
        hf_config_path=hf_config_path,
        seq_length=run.seq_length,
        model_flops_per_token=report.model_flops_per_token,
    )


def score_log(
    log_path: Path,
    workload: Workload,
    window: tuple[int, int],
    loss_window: tuple[int, int],
    num_gpus: int,
    peak_tflops_per_gpu: float,
) -> RunScore:
    """Score the run in ``log_path`` over the inclusive iteration ``window``.

    ``loss_window`` is the inclusive window ``loss_mean`` averages ``lm loss`` over. Tokens per
    iteration are the global batch size the window's lines report times ``workload.seq_length``.
    Raises ValueError if any iteration of either window is missing or repeated, if the window's
    lines report more than one global batch size, if an iteration of ``loss_window`` logs no loss,
    or if a parameter is out of range.
    """
    check_window("window", window, min_iterations=2)
    check_window("loss_window", loss_window, min_iterations=1)
    for name, value in (
        ("num_gpus", num_gpus),
        ("peak_tflops_per_gpu", peak_tflops_per_gpu),
        ("workload.seq_length", workload.seq_length),
        ("workload.model_flops_per_token", workload.model_flops_per_token),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    lines = read_log_lines(log_path)
    records = parse_iteration_records(lines)
    scored = window_records(records, window, f"{log_path}: window")
    loss_records = window_records(records, loss_window, f"{log_path}: loss_window")
    lossless = [r.iteration for r in loss_records if r.lm_loss is None]
    if lossless:
        raise ValueError(f"{log_path}: loss_window iterations {lossless} log no lm loss")
    batch_sizes = sorted({r.global_batch_size for r in scored})
    if len(batch_sizes) != 1:
        raise ValueError(
            f"{log_path}: window {window[0]}-{window[1]} reports several global batch sizes: {batch_sizes}"
        )
    (global_batch_size,) = batch_sizes
    tokens_per_iteration = global_batch_size * workload.seq_length

    steps_s = [r.elapsed_ms / 1000.0 for r in scored]
    mean_step_s = statistics.fmean(steps_s)
    median_step_s = statistics.median(steps_s)
    # "inclusive" is linear interpolation between order statistics (numpy.percentile's default).
    deciles = statistics.quantiles(steps_s, n=10, method="inclusive")
    model_tflops_per_gpu, mfu = achieved_throughput(
        tokens_per_iteration * workload.model_flops_per_token, mean_step_s, num_gpus, peak_tflops_per_gpu
    )
    return RunScore(
        log_path=str(log_path),
        workload=workload,
        num_gpus=num_gpus,
        global_batch_size=global_batch_size,
        tokens_per_iteration=tokens_per_iteration,
        peak_tflops_per_gpu=peak_tflops_per_gpu,
        window_first=window[0],
        window_last=window[1],
        loss_window_first=loss_window[0],
        loss_window_last=loss_window[1],
        n_iterations=len(scored),
        mean_step_s=mean_step_s,
        median_step_s=median_step_s,
        p10_step_s=deciles[0],
        p90_step_s=deciles[-1],
        min_step_s=min(steps_s),
        max_step_s=max(steps_s),
        outlier_iterations=[r.iteration for r, step in zip(scored, steps_s) if step > 2 * median_step_s],
        tokens_per_s_per_gpu=tokens_per_iteration / (num_gpus * mean_step_s),
        model_tflops_per_gpu=model_tflops_per_gpu,
        mfu=mfu,
        loss_mean=statistics.fmean(r.lm_loss for r in loss_records),
        skipped_total=sum(r.skipped_total for r in records),
        nan_total=sum(r.nan_total for r in records),
        last_iteration=max(r.iteration for r in records),
        first_iteration_memory_gb=parse_first_iteration_memory(lines),
        peak_memory_across_ranks=parse_peak_memory_across_ranks(lines),
        wandb_run_path=parse_wandb_run_path(lines),
        wandb_peak_memory_gb=None,
        wandb_alloc_retries=None,
    )


def fetch_wandb_peak_memory(run_path: str) -> dict[str, float]:
    """Return the run's W&B summary values for ``WANDB_PEAK_MEMORY_KEYS`` (decimal GB, as logged).

    They are the peaks of the one rank that logs to W&B (the last global rank), not a maximum over
    ranks. The run is read from the W&B server the client is configured for (``WANDB_BASE_URL``,
    by default the public cloud). Raises KeyError if the summary lacks either key; W&B client and
    network errors propagate.
    """
    summary = _wandb_summary(run_path, WANDB_PEAK_MEMORY_KEYS)
    return {key: float(summary[key]) for key in WANDB_PEAK_MEMORY_KEYS}


def fetch_wandb_alloc_retries(run_path: str) -> int:
    """Return the run's W&B summary value for ``WANDB_ALLOC_RETRIES_KEY``: the allocator retries of the rank
    that logs to W&B over the whole run. Raises KeyError if the summary lacks the key; W&B client and
    network errors propagate."""
    return int(_wandb_summary(run_path, (WANDB_ALLOC_RETRIES_KEY,))[WANDB_ALLOC_RETRIES_KEY])


def _wandb_summary(run_path: str, keys: tuple[str, ...]) -> Any:
    """Return the run's W&B summary, raising KeyError when it lacks any of ``keys``."""
    import wandb  # deferred: only --wandb-peak-memory needs the W&B client, and scoring must run without it

    summary = wandb.Api().run(run_path).summary
    missing = [key for key in keys if key not in summary]
    if missing:
        raise KeyError(f"W&B run {run_path} has no summary value for {missing}")
    return summary


def _memory_text(memory: dict[str, float] | None, absent: str) -> str:
    if memory is None:
        return absent
    return ", ".join(
        f"{key.removeprefix('memory/').removesuffix(GIGABYTES_SUFFIX)} {value:.3f}" for key, value in memory.items()
    )


def _peak_memory_text(peak: dict[str, float | int] | None) -> str:
    if peak is None:
        return "not in the log (it predates the summary, or its loop was cut short)"
    return (
        f"allocated {peak['max_allocated_gb']:.3f} GB (rank {peak['max_allocated_rank']}), "
        f"reserved {peak['max_reserved_gb']:.3f} GB, alloc retries max {peak['max_alloc_retries']} / "
        f"total {peak['total_alloc_retries']} over {peak['ranks']} ranks"
    )


def format_score(score: RunScore) -> str:
    """Render a score, inputs first, as the CLI's human-readable report."""
    workload = score.workload
    retries = score.wandb_alloc_retries
    rows = [
        f"log                     {score.log_path}",
        f"config                  {workload.config_path}",
        f"HF config               {workload.hf_config_path}",
        f"GPUs                    {score.num_gpus}",
        f"seq length x GBS        {workload.seq_length} x {score.global_batch_size} (logged) "
        f"= {score.tokens_per_iteration:,} tokens/iteration",
        f"model FLOPs/token       {workload.model_flops_per_token / 1e9:.4f} GFLOP",
        f"peak                    {score.peak_tflops_per_gpu:g} TFLOP/s/GPU",
        f"window                  {score.window_first}-{score.window_last} ({score.n_iterations} iterations)",
        f"loss window             {score.loss_window_first}-{score.loss_window_last}",
        f"step mean (primary)     {score.mean_step_s:.4f} s",
        f"step median / p10 / p90 {score.median_step_s:.4f} / {score.p10_step_s:.4f} / {score.p90_step_s:.4f} s",
        f"step min / max          {score.min_step_s:.4f} / {score.max_step_s:.4f} s",
        f"outliers (> 2x median)  {score.outlier_iterations or 'none'}",
        f"tokens/s/GPU            {score.tokens_per_s_per_gpu:,.1f}",
        f"model TFLOP/s/GPU       {score.model_tflops_per_gpu:.1f}",
        f"MFU                     {score.mfu:.2%}",
        f"lm loss mean            {score.loss_mean:.6f}",
        f"skipped / nan iters     {score.skipped_total} / {score.nan_total}",
        f"last iteration          {score.last_iteration}",
        f"iter-1 memory (GB)      {_memory_text(score.first_iteration_memory_gb, 'not reported')}",
        f"peak memory, all ranks  {_peak_memory_text(score.peak_memory_across_ranks)}",
        f"W&B run                 {score.wandb_run_path or 'not found'}",
        f"W&B peak memory (GB)    {_memory_text(score.wandb_peak_memory_gb, 'not fetched (--wandb-peak-memory)')}",
        f"W&B alloc retries       {'not fetched (--wandb-peak-memory)' if retries is None else retries}",
    ]
    return "\n".join(rows)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI: the log to score, the run's config and model, and the scoring windows."""
    parser = argparse.ArgumentParser(description="Score a training run's log: step time, throughput, MFU, loss.")
    parser.add_argument("log", type=Path, help="Training log (the launcher's SLURM output)")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="The run's training YAML; the sequence length and model FLOPs/token are read from it",
    )
    parser.add_argument(
        "--hf-model",
        required=True,
        help="HF repo id (resolved offline from the local HF cache), model directory, or config.json path of "
        "the run's architecture, as scripts/nemotronh_flops_estimator.py takes it",
    )
    parser.add_argument("--gpus", type=int, required=True, help="GPUs the run used")
    parser.add_argument("--window", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"), help="Scoring window")
    parser.add_argument(
        "--loss-window",
        type=int,
        nargs=2,
        required=True,
        metavar=("FIRST", "LAST"),
        help="Window lm loss is averaged over",
    )
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=DEFAULT_PEAK_TFLOPS,
        help=f"Per-GPU dense BF16 peak TFLOP/s for MFU (default: the estimator's {DEFAULT_PEAK_TFLOPS}, GH200)",
    )
    parser.add_argument(
        "--wandb-peak-memory",
        action="store_true",
        help="Also read the W&B run's summary peak memory and allocator-retry count (the last rank's; needs "
        "the wandb client and network)",
    )
    parser.add_argument("--json", action="store_true", help="Emit the score as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Score the log and print the result."""
    args = build_parser().parse_args(argv)
    score = score_log(
        args.log,
        workload=workload_from_config(args.config, args.hf_model),
        window=(args.window[0], args.window[1]),
        loss_window=(args.loss_window[0], args.loss_window[1]),
        num_gpus=args.gpus,
        peak_tflops_per_gpu=args.peak_tflops,
    )
    if args.wandb_peak_memory:
        if score.wandb_run_path is None:
            raise ValueError(f"{args.log} names no W&B run, so there is no summary to read memory from")
        score = replace(
            score,
            wandb_peak_memory_gb=fetch_wandb_peak_memory(score.wandb_run_path),
            wandb_alloc_retries=fetch_wandb_alloc_retries(score.wandb_run_path),
        )
    print(json.dumps(score.to_dict(), indent=2) if args.json else format_score(score))
    return 0


if __name__ == "__main__":
    sys.exit(main())

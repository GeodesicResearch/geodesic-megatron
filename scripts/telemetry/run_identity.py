"""Per-run identity for Isambard training runs (INFR-68).

One unique ID joins the three places a training run leaves artifacts, which
today are painful to correlate after the fact:

    raw SLURM job log      logs/slurm/train-<jobid>.out (+ a by-run-id/ symlink)
    torch-profiler output  /projects/a5k/public/profiles/<run-name>/<run-id>/
    W&B run                summary metrics run/isambard_run_id, run/raw_log_path

The ID is minted by pipeline_training_launch.sh as ISAMBARD_RUN_ID
(<launch-timestamp>-j<slurm-job-id>) and exported to every rank; this module
is the single consumer-side reader plus the callback that stamps the identity
into W&B. Kept import-light (no torch / megatron at module scope beyond the
Callback base class) so unit tests can load it without a GPU stack.
"""

import os
import time

from megatron.bridge.training.callbacks import Callback


def get_run_id() -> str:
    """Return the unique ID for this training run.

    Precedence:
    1. ``ISAMBARD_RUN_ID`` — minted by pipeline_training_launch.sh (launch
       timestamp + SLURM job id), identical on every rank.
    2. ``j<SLURM_JOB_ID>`` — rank-stable fallback when running under SLURM
       without the launcher (every rank derives the same value; a timestamp
       here would diverge across ranks).
    3. ``local-<timestamp>-p<pid>`` — single-process development runs.
    """
    run_id = os.environ.get("ISAMBARD_RUN_ID", "")
    if run_id:
        return run_id
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if job_id:
        return f"j{job_id}"
    return f"local-{time.strftime('%Y%m%dT%H%M%S')}-p{os.getpid()}"


def get_raw_log_path() -> str:
    """Absolute path of this run's raw job log, or '' when there is none.

    Set (and existence-checked) by pipeline_training_launch.sh via scontrol's
    StdOut field; empty for interactive/salloc runs where stdout is a terminal
    and for runs not started through the launcher.
    """
    return os.environ.get("ISAMBARD_RAW_LOG_PATH", "")


def stamp_wandb_summary(run, run_id: str, raw_log_path: str) -> None:
    """Write run-identity summary metrics onto a W&B run object.

    ``run`` is ``wandb.run`` — non-None only on the single rank that
    initialized W&B (megatron-bridge inits on the LAST rank, world_size-1, not
    rank 0). No-op when None, so every rank may call this unconditionally.
    Keys are namespaced ``run/`` mirroring the bridge's ``parallelism/``
    summary convention.
    """
    if run is None:
        return
    run.summary.update(
        {
            "run/isambard_run_id": run_id,
            "run/raw_log_path": raw_log_path,
            "run/slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        }
    )


class RunIdentityCallback(Callback):
    """Stamps run-identity metadata into W&B at train start — on every run.

    Registered unconditionally by pipeline_training_run.py (profiled or not).
    Uses ``on_train_start`` rather than ``on_train_end`` because the bridge's
    early-exit path (exit signal / exit_interval) calls wandb.finish() and
    sys.exit() without ever firing ``on_train_end``. The W&B write is wrapped
    so a telemetry failure can never crash training — callback exceptions
    propagate in megatron-bridge's CallbackManager.
    """

    def __init__(self, run_id: str, raw_log_path: str):
        """Store the identity to stamp; both values come from get_run_id()/get_raw_log_path()."""
        self.run_id = run_id
        self.raw_log_path = raw_log_path

    def on_train_start(self, ctx) -> None:
        """Stamp W&B summary metrics (effective only on the wandb-owning rank)."""
        try:
            import wandb

            stamp_wandb_summary(wandb.run, self.run_id, self.raw_log_path)
            if wandb.run is not None:
                print(
                    f"[run-identity] run_id={self.run_id} stamped to W&B (raw_log={self.raw_log_path or '(none)'})",
                    flush=True,
                )
        except Exception as e:  # telemetry must never kill a multi-node run
            print(f"[run-identity] WARNING: failed to stamp W&B summary: {e}", flush=True)

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


def resolve_run_artifact_dir(cfg) -> str:
    """The directory this run writes its artifacts to.

    ``checkpoint.save`` when the run saves checkpoints, else ``logger.wandb_save_dir``.
    Not a degraded-mode fallback: these are the two places a run's outputs actually go,
    and benchmark runs deliberately set ``checkpoint.save: null``. Both unset is a
    configuration error — ``training/state.py`` already fails such a run at startup — so
    it raises rather than quietly picking somewhere.
    """
    for candidate in (getattr(cfg.checkpoint, "save", None), getattr(cfg.logger, "wandb_save_dir", None)):
        if candidate:
            return str(candidate)
    raise ValueError(
        "Cannot locate a run artifact directory: both checkpoint.save and logger.wandb_save_dir "
        "are unset. Set logger.wandb_save_dir (mandatory whenever checkpoint.save is null)."
    )


def serialize_resolved_config(cfg) -> str | None:
    """The fully-resolved config as YAML text, or None if it could not be serialized.

    The override YAML alone does not describe a run: recipe defaults and CLI overrides
    (``train.global_batch_size=256``, ``model.virtual_pipeline_model_parallel_size=4``, …)
    exist only in the resolved object, so a run configured partly from the command line is
    otherwise unreproducible from disk. Non-serializable fields (e.g. an in-memory
    dataset_dict) are excluded by the same helper the config-merge pipeline itself uses.

    Provenance must never take down a training run, so failures are reported and swallowed —
    the run proceeds, having said what it lost. That swallow is only safe because the happy
    path is pinned by a unit test against a real ConfigContainer: without it a wiring mistake
    here (a moved import, say) degrades to "no snapshot, ever" and nobody notices.
    """
    try:
        from omegaconf import OmegaConf

        from megatron.bridge.training.utils.omegaconf_utils import create_omegaconf_dict_config

        resolved, _ = create_omegaconf_dict_config(cfg)
        return OmegaConf.to_yaml(resolved, resolve=True)
    except Exception as e:  # noqa: BLE001 - provenance must not break training
        print(f"[run-identity] WARNING: could not serialize resolved config ({e})")
        return None


def write_resolved_config(cfg, run_id: str, resolved_config_yaml: str | None) -> str | None:
    """Write the resolved config beside this run's artifacts on rank 0; return the path.

    Named by run id so the snapshot joins the raw log, the profiles and the W&B run.
    Takes the already-serialized text rather than re-deriving it, so a run that also
    captures a profile serializes its config exactly once.

    Only ``OSError`` is tolerated — a full disk or an unwritable directory should cost the
    snapshot, not the run. ``resolve_run_artifact_dir``'s ``ValueError`` deliberately
    propagates: that one means the config names no artifact directory at all, which is a
    configuration error the operator must see, and swallowing it here would silence a
    function documented as refusing to guess.
    """
    if resolved_config_yaml is None or int(os.environ.get("RANK", "0")) != 0:
        return None
    target_dir = resolve_run_artifact_dir(cfg)
    try:
        os.makedirs(target_dir, exist_ok=True)
        path = os.path.join(target_dir, f"{run_id}.resolved-config.yaml")
        with open(path, "w") as fh:
            fh.write(resolved_config_yaml)
        return path
    except OSError as e:
        print(f"[run-identity] WARNING: could not write resolved config ({e}); this run is not reproducible from disk")
        return None


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

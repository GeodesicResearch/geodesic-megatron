"""Torch-profiler trace collection for Isambard training runs (INFR-68).

Purpose: produce the artifact Quentin Anthony's Megatron speed-up assessment
needs — a torch profile of full optimizer steps with ``with_stack=True`` and
``record_shapes=True``, delivered together with the exact repo commit and the
run's config (see docs/environment.md "Profiling"). Also the first tool
to reach for when iteration time regresses and the cause isn't obvious.

Activation (all env-driven, matching the repo's ISAMBARD_* toggle pattern;
default OFF — a training run is never profiled unless asked):

    ISAMBARD_TORCH_PROFILE=1            enable; traces land under the default root
    ISAMBARD_TORCH_PROFILE=/some/dir    enable; traces land under that dir
    ISAMBARD_TORCH_PROFILE_RANKS=0,4    global ranks to trace (default: 0)
    ISAMBARD_TORCH_PROFILE_ITERS=10,20  1-based training iterations to capture —
                                        one trace file per listed iteration
    ISAMBARD_TORCH_PROFILE_WAIT=3       legacy single-capture knob, used only
                                        when _ITERS is unset: skip this many
                                        iterations, then warm up 1 and trace 1
                                        (default 3 -> iteration 5, past the
                                        JIT/comm-init-dominated window)

Each capture covers ONE full optimizer step (all its microbatches' fwd/bwd
pairs), with the iteration immediately before it used as profiler warmup.
Traces + provenance are written under <root>/<run-name>/<run-id>/ — the run ID
(scripts/telemetry/run_identity.py; also stamped into W&B and the job-log
identity) keeps repeat runs of the same config from overwriting each other and
joins the profile to its log and W&B run:

    rank<R>.chrome_trace.json.gz    single-capture (legacy _WAIT) mode
    rank<R>.iter<N>.chrome_trace.json.gz   per listed iteration in _ITERS mode
    provenance.txt                  exact commit, config path, run id, raw-log
                                    path, capture iterations, world size,
                                    torch version
    config_snapshot.yaml            the --config-file override YAML, verbatim
    resolved_config_snapshot.yaml   the FULL merged model+training config
                                    (recipe defaults + YAML overlay + CLI
                                    overrides) — the authoritative reproduction
                                    source. The override file alone is NOT
                                    (CLI overrides such as train.train_iters
                                    never appear in it; see the champion
                                    trace's REPRODUCE.txt correction).
    raw_log_snapshot.out            copy of the job's raw SLURM log (refreshed
                                    at each trace export and at train end; a
                                    point-in-time snapshot — the live log keeps
                                    growing after training ends)

Open traces in chrome://tracing or Perfetto. Default root lives on project
storage (never $HOME): the launcher's W&B exp name (or SLURM job id) names the
run directory, the run ID names the per-launch subdirectory.
"""

import gzip
import os
import shutil
import socket

import torch

from megatron.bridge.training.callbacks import Callback


DEFAULT_PROFILE_ROOT = "/projects/a5k/public/profiles"


def _repo_commit(repo_dir: str) -> str:
    """Resolve HEAD without invoking git (the container may lack the binary)."""
    try:
        head_path = os.path.join(repo_dir, ".git", "HEAD")
        with open(head_path) as f:
            head = f.read().strip()
        if head.startswith("ref: "):
            ref = head[5:]
            ref_path = os.path.join(repo_dir, ".git", ref)
            if os.path.exists(ref_path):
                with open(ref_path) as f:
                    return f"{f.read().strip()} ({ref})"
            packed = os.path.join(repo_dir, ".git", "packed-refs")
            with open(packed) as f:
                for line in f:
                    if line.strip().endswith(ref):
                        return f"{line.split()[0]} ({ref})"
        return head
    except OSError as e:
        return f"UNRESOLVED ({e})"


def _capture_schedule(capture_steps: set[int]):
    """Profiler schedule tracing exactly the given 0-indexed profiler steps.

    Returns RECORD_AND_SAVE on each capture step (so ``on_trace_ready`` fires
    once per capture), WARMUP on the step immediately before a capture, and
    NONE elsewhere. Unlike ``torch.profiler.schedule`` this supports arbitrary,
    non-periodic capture points (e.g. iterations 10 and 20).
    """

    def action(step: int) -> torch.profiler.ProfilerAction:
        if step in capture_steps:
            return torch.profiler.ProfilerAction.RECORD_AND_SAVE
        if step + 1 in capture_steps:
            return torch.profiler.ProfilerAction.WARMUP
        return torch.profiler.ProfilerAction.NONE

    return action


class TorchProfilerCallback(Callback):
    """Profile chosen optimizer steps per run, schedule-driven via step-end events."""

    def __init__(
        self,
        out_root: str,
        run_name: str,
        run_id: str,
        config_file: str | None,
        capture_iters: list[int],
        tag_files: bool,
        ranks: list[int],
        resolved_config_yaml: str | None,
        raw_log_path: str,
    ):
        self.out_dir = os.path.join(out_root, run_name, run_id)
        self.run_id = run_id
        self.config_file = config_file
        self.resolved_config_yaml = resolved_config_yaml
        self.raw_log_path = raw_log_path
        self.capture_iters = sorted(capture_iters)
        self.tag_files = tag_files
        self.ranks = ranks
        self.prof: torch.profiler.profile | None = None
        self.enabled_here = False
        self.captures_done = 0
        self.steps_done = 0  # completed training iterations (1-based count)

    def _trace_basename(self, rank: int) -> str:
        """Filename for the capture about to be exported (ascending order)."""
        if self.tag_files:
            return f"rank{rank}.iter{self.capture_iters[self.captures_done]}.chrome_trace.json"
        return f"rank{rank}.chrome_trace.json"

    def on_train_start(self, ctx) -> None:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.enabled_here = rank in self.ranks
        if not self.enabled_here:
            return
        # Guarded: profiling is opt-in tooling — a full disk / unreachable
        # profile dir must degrade to a warning, never crash the training run
        # (callback exceptions propagate in megatron-bridge).
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            self.prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=_capture_schedule({i - 1 for i in self.capture_iters}),
                with_stack=True,
                record_shapes=True,
                on_trace_ready=self._export,
            )
            self.prof.start()
        except Exception as e:  # noqa: BLE001 -- degrade-don't-crash boundary
            print(f"[torch-profile] WARNING: profiler setup failed, profiling disabled: {e}", flush=True)
            self.enabled_here = False
            self.prof = None
            return
        print(
            f"[torch-profile] rank {rank}: tracing iteration(s) {self.capture_iters} "
            f"(1-based; the step before each is profiler warmup) -> {self.out_dir}",
            flush=True,
        )

    def on_train_step_end(self, ctx) -> None:
        if not self.enabled_here or self.prof is None:
            return
        self.steps_done += 1
        # Guarded: a kineto/CUPTI failure inside step()/stop() must disable
        # profiling, never crash the run (callback exceptions propagate in
        # megatron-bridge). _export has its own boundary; this catches the
        # profiler-internal transitions themselves.
        try:
            if self.captures_done >= len(self.capture_iters):
                # All traces exported by the schedule's on_trace_ready during
                # prior steps; stop OUTSIDE that handler (stopping from within
                # it is re-entrant) and drop the profiler so later steps cost
                # nothing.
                self.prof.stop()
                self.prof = None
                return
            self.prof.step()
        except Exception as e:  # noqa: BLE001 -- degrade-don't-crash boundary
            print(f"[torch-profile] WARNING: profiler step/stop failed, profiling disabled: {e}", flush=True)
            self.prof = None

    def on_train_end(self, ctx) -> None:
        if self.enabled_here and self.prof is not None:
            try:
                self.prof.stop()
            except Exception as e:  # noqa: BLE001 -- degrade-don't-crash boundary
                print(f"[torch-profile] WARNING: profiler stop failed at train end: {e}", flush=True)
            self.prof = None
        if self.enabled_here:
            # Refresh the log snapshot with everything written up to train end.
            self._copy_raw_log()

    def _export(self, prof: torch.profiler.profile) -> None:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        # Teardown guard: prof.stop() at train end fires this handler for a
        # pending capture whose iteration never ran (capture_iter > train_iters)
        # — that "trace" would contain zero training steps. Suppress it.
        if self.captures_done >= len(self.capture_iters) or self.capture_iters[self.captures_done] > self.steps_done:
            print(
                f"[torch-profile] rank {rank}: suppressing teardown export — capture iteration "
                f"never ran ({self.steps_done} iterations completed)",
                flush=True,
            )
            self.captures_done = len(self.capture_iters)
            return
        raw = os.path.join(self.out_dir, self._trace_basename(rank))
        # Guarded: trace export writes hundreds of MB to shared Lustre — EDQUOT/
        # ENOSPC/stale-handle OSErrors here must degrade to a warning, never
        # crash the 64-rank run (callback exceptions propagate in the bridge).
        try:
            prof.export_chrome_trace(raw)
            with open(raw, "rb") as f_in, gzip.open(raw + ".gz", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(raw)
            if rank == self.ranks[0]:
                # First profiled rank only: provenance/config snapshots are
                # shared files; concurrent writers on different nodes race.
                self._write_provenance(rank)
            self._copy_raw_log()
        except Exception as e:  # noqa: BLE001 -- degrade-don't-crash boundary
            print(
                f"[torch-profile] WARNING: trace export failed ({e}); disabling further captures",
                flush=True,
            )
            self.captures_done = len(self.capture_iters)  # halt profiling; next step end stops the profiler
            try:
                os.remove(raw)
            except OSError:
                pass
            return
        self.captures_done += 1  # actual prof.stop() happens at a later step end, outside this handler
        print(f"[torch-profile] rank {rank}: trace written -> {raw}.gz", flush=True)

    def _copy_raw_log(self) -> None:
        """Snapshot the job's raw SLURM log into the profile dir (best-effort).

        Only the first profiled rank copies (all profiled ranks share the one
        SLURM-merged log; concurrent copies from several nodes would race).
        Skipped when no raw-log path is known (interactive/salloc runs, or not
        launched via pipeline_training_launch.sh). Guarded: a failed log copy
        must never crash a multi-node training run — callback exceptions
        propagate in megatron-bridge.
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank != self.ranks[0]:
            return
        if not self.raw_log_path:
            print(
                "[torch-profile] no raw-log path known (ISAMBARD_RAW_LOG_PATH unset); skipping log snapshot",
                flush=True,
            )
            return
        try:
            shutil.copy2(self.raw_log_path, os.path.join(self.out_dir, "raw_log_snapshot.out"))
        except OSError as e:
            print(f"[torch-profile] WARNING: raw-log snapshot failed: {e}", flush=True)

    def _write_provenance(self, rank: int) -> None:
        repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        prov = os.path.join(self.out_dir, "provenance.txt")
        if os.path.exists(prov):
            return
        with open(prov, "w") as f:
            f.write(f"commit: {_repo_commit(repo_dir)}\n")
            f.write(f"repo_dir: {repo_dir}\n")
            f.write(f"config_file: {self.config_file}\n")
            f.write(f"run_id: {self.run_id}\n")
            f.write(f"raw_log_path: {self.raw_log_path or '(none)'}\n")
            f.write(f"world_size: {os.environ.get('WORLD_SIZE', '?')}\n")
            f.write(f"host: {socket.gethostname()}\n")
            f.write(f"torch: {torch.__version__} (cuda {torch.version.cuda})\n")
            # Image identity — the half of the stack the commit above doesn't
            # pin (torch/CUDA/TE/NCCL all come from the image). Apptainer itself
            # exports APPTAINER_CONTAINER (full SIF path, whose filename carries
            # the NGC tag) and APPTAINER_NAME into every container, so they are
            # always present here; pipeline_env_config.env's CONTAINER_SIF /
            # CONTAINER_IMAGE_TAG are plain host shell vars and never reach this
            # process. '?' would mean the trace was taken outside the container.
            image = os.environ.get("APPTAINER_CONTAINER") or os.environ.get("APPTAINER_NAME") or "?"
            f.write(f"container_image: {image}\n")
            f.write(
                "profiler: with_stack=True record_shapes=True "
                f"capture_iterations={self.capture_iters} (1-based, warmup on the step before each)\n"
            )
        if self.config_file and os.path.exists(self.config_file):
            shutil.copy2(self.config_file, os.path.join(self.out_dir, "config_snapshot.yaml"))
        if self.resolved_config_yaml:
            with open(os.path.join(self.out_dir, "resolved_config_snapshot.yaml"), "w") as f:
                f.write(self.resolved_config_yaml)


def maybe_build_profiler_callback(
    config_file: str | None,
    run_name: str,
    run_id: str,
    resolved_config_yaml: str | None,
    raw_log_path: str,
) -> TorchProfilerCallback | None:
    """Return the callback when ISAMBARD_TORCH_PROFILE requests it, else None.

    ``run_id`` (see scripts/telemetry/run_identity.py) names the per-launch
    output subdirectory and lands in provenance.txt. ``resolved_config_yaml``
    is the fully-merged model+training config (recipe defaults + YAML overlay +
    CLI overrides) as YAML text; it is snapshotted next to the traces as
    ``resolved_config_snapshot.yaml``. ``raw_log_path`` ('' when unknown) is
    the job log to snapshot into the profile dir.

    Capture points: ``ISAMBARD_TORCH_PROFILE_ITERS`` (1-based iteration list,
    per-iteration ``rank<R>.iter<N>`` trace files) wins when set; otherwise the
    legacy single capture at iteration ``ISAMBARD_TORCH_PROFILE_WAIT + 2`` with
    the unsuffixed ``rank<R>`` filename.
    """
    setting = os.environ.get("ISAMBARD_TORCH_PROFILE", "0")
    if setting in ("0", ""):
        return None
    out_root = DEFAULT_PROFILE_ROOT if setting == "1" else setting
    ranks = [int(r) for r in os.environ.get("ISAMBARD_TORCH_PROFILE_RANKS", "0").split(",") if r != ""]
    iters_env = os.environ.get("ISAMBARD_TORCH_PROFILE_ITERS", "")
    if iters_env:
        capture_iters = [int(i) for i in iters_env.split(",") if i != ""]
        tag_files = True
    else:
        wait = int(os.environ.get("ISAMBARD_TORCH_PROFILE_WAIT", "3"))
        capture_iters = [wait + 2]
        tag_files = False
    return TorchProfilerCallback(
        out_root,
        run_name,
        run_id,
        config_file,
        capture_iters,
        tag_files,
        ranks,
        resolved_config_yaml,
        raw_log_path,
    )

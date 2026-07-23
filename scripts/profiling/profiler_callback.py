"""Torch-profiler trace collection for Isambard training runs (INFR-68).

Purpose: produce the artifact Quentin Anthony's Megatron speed-up assessment
needs — a torch profile of 1-2 fwd/bwd pairs with ``with_stack=True`` and
``record_shapes=True``, delivered together with the exact repo commit and the
run's config (see docs/container-pipeline.md "Profiling"). Also the first tool
to reach for when iteration time regresses and the cause isn't obvious.

Activation (all env-driven, matching the repo's ISAMBARD_* toggle pattern;
default OFF — a training run is never profiled unless asked):

    ISAMBARD_TORCH_PROFILE=1            enable; traces land under the default root
    ISAMBARD_TORCH_PROFILE=/some/dir    enable; traces land under that dir
    ISAMBARD_TORCH_PROFILE_RANKS=0,4    global ranks to trace (default: 0)
    ISAMBARD_TORCH_PROFILE_WAIT=3       full iterations to skip first (default 3 —
                                        past the JIT/comm-init-dominated window)

The profiler covers ONE full optimizer step (all its microbatches' fwd/bwd
pairs) after a one-iteration warmup: schedule(wait, warmup=1, active=1).
Traces + provenance are written under <root>/<run-name>/:

    rank<R>.chrome_trace.json.gz   open in chrome://tracing or Perfetto
    provenance.txt                 exact commit, dirty-flag, config path + copy,
                                   world size/topology, torch version

Default root lives on project storage (never $HOME): the launcher's W&B exp
name (or SLURM job id) names the run directory.
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


class TorchProfilerCallback(Callback):
    """Profile one optimizer step per run, schedule-driven via step-end events."""

    def __init__(self, out_root: str, run_name: str, config_file: str | None, wait_iters: int, ranks: list[int]):
        self.out_dir = os.path.join(out_root, run_name)
        self.config_file = config_file
        self.wait_iters = wait_iters
        self.ranks = ranks
        self.prof: torch.profiler.profile | None = None
        self.enabled_here = False
        self.done = False

    def on_train_start(self, ctx) -> None:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.enabled_here = rank in self.ranks
        if not self.enabled_here:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        self.prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=self.wait_iters, warmup=1, active=1, repeat=1),
            with_stack=True,
            record_shapes=True,
            on_trace_ready=self._export,
        )
        self.prof.start()
        print(
            f"[torch-profile] rank {rank}: tracing iteration {self.wait_iters + 2} "
            f"(wait={self.wait_iters}, warmup=1, active=1) -> {self.out_dir}",
            flush=True,
        )

    def on_train_step_end(self, ctx) -> None:
        if not self.enabled_here or self.prof is None:
            return
        if self.done:
            # Trace already exported by the schedule's on_trace_ready during a
            # prior step; stop OUTSIDE that handler (stopping from within it is
            # re-entrant) and drop the profiler so later steps cost nothing.
            self.prof.stop()
            self.prof = None
            return
        self.prof.step()

    def on_train_end(self, ctx) -> None:
        if self.enabled_here and self.prof is not None and not self.done:
            self.prof.stop()

    def _export(self, prof: torch.profiler.profile) -> None:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        raw = os.path.join(self.out_dir, f"rank{rank}.chrome_trace.json")
        prof.export_chrome_trace(raw)
        with open(raw, "rb") as f_in, gzip.open(raw + ".gz", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(raw)
        self._write_provenance(rank)
        self.done = True  # actual prof.stop() happens at the next step end, outside this handler
        print(f"[torch-profile] rank {rank}: trace written -> {raw}.gz", flush=True)

    def _write_provenance(self, rank: int) -> None:
        repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        prov = os.path.join(self.out_dir, "provenance.txt")
        if os.path.exists(prov):
            return
        with open(prov, "w") as f:
            f.write(f"commit: {_repo_commit(repo_dir)}\n")
            f.write(f"repo_dir: {repo_dir}\n")
            f.write(f"config_file: {self.config_file}\n")
            f.write(f"world_size: {os.environ.get('WORLD_SIZE', '?')}\n")
            f.write(f"host: {socket.gethostname()}\n")
            f.write(f"torch: {torch.__version__} (cuda {torch.version.cuda})\n")
            f.write(f"container: {os.environ.get('GEODESIC_CONTAINER', '?')}\n")
            f.write("profiler: with_stack=True record_shapes=True schedule(wait,warmup=1,active=1)\n")
        if self.config_file and os.path.exists(self.config_file):
            shutil.copy2(self.config_file, os.path.join(self.out_dir, "config_snapshot.yaml"))


def maybe_build_profiler_callback(config_file: str | None, run_name: str) -> TorchProfilerCallback | None:
    """Return the callback when ISAMBARD_TORCH_PROFILE requests it, else None."""
    setting = os.environ.get("ISAMBARD_TORCH_PROFILE", "0")
    if setting in ("0", ""):
        return None
    out_root = DEFAULT_PROFILE_ROOT if setting == "1" else setting
    ranks = [int(r) for r in os.environ.get("ISAMBARD_TORCH_PROFILE_RANKS", "0").split(",") if r != ""]
    wait = int(os.environ.get("ISAMBARD_TORCH_PROFILE_WAIT", "3"))
    return TorchProfilerCallback(out_root, run_name, config_file, wait, ranks)

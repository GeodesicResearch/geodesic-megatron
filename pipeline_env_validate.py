#!/usr/bin/env python3
"""Validate the Megatron Bridge environment on Isambard ARM HPC (INFR-68).

This runs INSIDE the container — the only execution environment there is — so
every check below describes the container's view of the world: imports resolving
to THIS checkout (bind-mounted src/ + pinned 3rdparty/Megatron-LM, which must win
over the image's own megatron packages), the image's CUDA extensions, the
Slingshot CXI NCCL plugin, ft_launcher's section-timeout flags, the JIT
toolchain, GPU ops, and the recipes. There is nothing to validate on the host.

Usage (both forms enter the container and source the in-container activate first):
    isambard_sbatch pipeline_env_submit.sbatch validate [--run-training]
    ./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh; python pipeline_env_validate.py"

Direct invocation is `python pipeline_env_validate.py [--run-training]`;
--run-training adds a 5-iteration single-GPU training job on mock data.

Exit code 0 if all stages pass, 1 otherwise.
"""

import argparse
import ctypes
import os
import subprocess
import sys
import time


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

results = []


def stage(name):
    """Decorator to run and track test stages."""

    def decorator(fn):
        def wrapper():
            t0 = time.time()
            try:
                fn()
                elapsed = time.time() - t0
                results.append((name, True, f"{elapsed:.1f}s"))
                print(f"  [{PASS}] {name} ({elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.time() - t0
                results.append((name, False, str(e)))
                print(f"  [{FAIL}] {name}: {e}")

        return wrapper

    return decorator


# ============================================
# Stage 1: Core Python imports
# ============================================
@stage("torch")
def check_torch():
    import torch

    assert torch.__version__, "torch version is empty"


@stage("megatron.core")
def check_mcore():
    import megatron.core  # noqa: F401 -- the import IS the check


@stage("megatron.bridge")
def check_mbridge():
    import megatron.bridge  # noqa: F401 -- the import IS the check


@stage("transformers")
def check_transformers():
    import transformers  # noqa: F401 -- the import IS the check


@stage("datasets")
def check_datasets():
    import datasets  # noqa: F401 -- the import IS the check


@stage("wandb")
def check_wandb():
    import wandb  # noqa: F401 -- the import IS the check


@stage("omegaconf")
def check_omegaconf():
    import omegaconf  # noqa: F401 -- the import IS the check


# ============================================
# Stage 2: CUDA extension imports
# ============================================
@stage("transformer_engine")
def check_te():
    import transformer_engine  # noqa: F401 -- the import IS the check
    import transformer_engine.pytorch  # noqa: F401


@stage("mamba_ssm")
def check_mamba():
    import mamba_ssm  # noqa: F401 -- the import IS the check


@stage("causal_conv1d")
def check_causal_conv():
    import causal_conv1d  # noqa: F401 -- the import IS the check


# ============================================
# Stage 3: CUDA availability
# ============================================
@stage("CUDA availability")
def check_cuda():
    import torch

    assert torch.cuda.is_available(), "CUDA not available"
    print(f"    GPU: {torch.cuda.get_device_name(0)}")
    print(f"    Arch: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}")
    print(f"    CUDA: {torch.version.cuda}")
    print(f"    GPU count: {torch.cuda.device_count()}")


# ============================================
# Stage 4: GPU tensor operations
# ============================================
@stage("GPU tensor operations")
def check_gpu_ops():
    import torch

    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    z = x @ y
    assert z.shape == (256, 256), f"Expected (256, 256), got {z.shape}"
    assert z.device.type == "cuda", f"Expected cuda, got {z.device.type}"
    # Verify values are finite
    assert torch.isfinite(z).all(), "Non-finite values in matmul result"


# ============================================
# Stage 5: Recipe loading
# ============================================
@stage("vanilla_gpt_pretrain_config recipe")
def check_vanilla_recipe():
    from megatron.bridge.recipes.gpt.vanilla_gpt import vanilla_gpt_pretrain_config

    cfg = vanilla_gpt_pretrain_config()
    assert cfg.model is not None, "model config is None"
    assert cfg.train is not None, "train config is None"


@stage("nemotron_3_nano_sft_config recipe")
def check_nemotron_recipe():
    from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_sft_config

    cfg = nemotron_3_nano_sft_config()
    assert cfg.model is not None, "model config is None"
    assert cfg.model.num_moe_experts == 128, f"Expected 128 MoE experts, got {cfg.model.num_moe_experts}"
    assert cfg.model.expert_model_parallel_size == 8, f"Expected EP=8, got {cfg.model.expert_model_parallel_size}"


# ============================================
# Stage 6: Tiny training run (optional)
# ============================================
def run_tiny_training():
    """Run a 5-iteration training with vanilla GPT and mock data."""
    print(f"\n  [{WARN}] Running tiny training (5 iterations, single GPU)...")
    t0 = time.time()

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        os.path.join(REPO_ROOT, "scripts", "training", "run_recipe.py"),
        "--recipe",
        "vanilla_gpt_pretrain_config",
        "--dataset",
        "llm-pretrain-mock",
        "train.train_iters=5",
        "train.global_batch_size=8",
        "train.micro_batch_size=4",
        "model.num_layers=2",
        "model.hidden_size=256",
        "model.num_attention_heads=4",
        # Fusion off deliberately: this smoke test proves the environment can run a
        # training step, so it must not also depend on APEX. Real runs leave it True
        # (the image ships APEX and fused accumulation is the faster path).
        "model.gradient_accumulation_fusion=False",
        "scheduler.lr_warmup_iters=2",
        "scheduler.lr_decay_iters=5",
        "logger.log_interval=1",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            results.append(("tiny training run", True, f"{elapsed:.1f}s"))
            print(f"  [{PASS}] tiny training run ({elapsed:.1f}s)")
        else:
            # Print last 20 lines of stderr for debugging
            stderr_lines = result.stderr.strip().split("\n")[-20:]
            results.append(("tiny training run", False, f"exit code {result.returncode}"))
            print(f"  [{FAIL}] tiny training run (exit code {result.returncode})")
            print("    Last 20 lines of stderr:")
            for line in stderr_lines:
                print(f"    {line}")
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        results.append(("tiny training run", False, "timeout after 300s"))
        print(f"  [{FAIL}] tiny training run (timeout after 300s)")


@stage("grouped_gemm")
def check_grouped_gemm():
    import grouped_gemm  # noqa: F401  (nv-grouped-gemm; MoE grouped GEMM)


def module_file_is_under(module, root):
    """True iff `module` was loaded from a file inside directory `root`.

    Pure-python helper for the import-path checks (unit-tested in
    tests/unit_tests/test_env_validate_container.py). Namespace-package modules
    without __file__ are resolved via their first __path__ entry.
    """
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        paths = list(getattr(module, "__path__", []) or [])
        if not paths:
            return False
        module_file = paths[0]
    module_file = os.path.realpath(module_file)
    root = os.path.realpath(root)
    return module_file == root or module_file.startswith(root + os.sep)


# ============================================
# Environment-integrity stages (INFR-68) — assert the container sees the repo's
# code, the Slingshot CXI plugin, and a workable ft_launcher/toolchain. These are
# the checks that catch a bad image swap or a half-finished setup, so they run
# unconditionally: the container is the only environment and every one of these
# is load-bearing for a real multi-node run.
# ============================================
@stage("import paths (repo wins over image)")
def check_import_paths():
    """Assert megatron.bridge/core resolve to this repo, not the image installs."""
    import megatron
    import megatron.core

    import megatron.bridge

    print(f"    megatron.__path__: {list(megatron.__path__)}")
    assert module_file_is_under(megatron.bridge, os.path.join(REPO_ROOT, "src")), (
        f"megatron.bridge resolves to {megatron.bridge.__file__}, not this repo's src/ — "
        "the image likely ships a regular (non-namespace) megatron package that defeats "
        "PYTHONPATH; see docs/environment.md contingencies"
    )
    assert module_file_is_under(megatron.core, os.path.join(REPO_ROOT, "3rdparty", "Megatron-LM")), (
        f"megatron.core resolves to {megatron.core.__file__}, not 3rdparty/Megatron-LM — "
        "the pinned submodule must win over the image's megatron-core"
    )


@stage("NCCL CXI net plugin loads")
def check_nccl_plugin():
    """Assert the CXI aws-ofi-nccl plugin exists and its shared-lib deps resolve."""
    plugin = os.environ.get("NCCL_NET_PLUGIN")
    assert plugin, "NCCL_NET_PLUGIN not set — source pipeline_env_activate.sh (in-container)"
    assert os.path.isfile(plugin), (
        f"NCCL_NET_PLUGIN={plugin} does not exist — the Slingshot NCCL stack is not built. "
        "Fix: bash pipeline_env_setup.sh   (one-time per image tag, ~20 min on a GPU node)"
    )
    # CDLL proves the plugin's own deps (libfabric CXI, libcxi, libnl) resolve
    # inside the container — the exact failure mode that silently degrades NCCL
    # to ~2.3 GB/s TCP if broken.
    ctypes.CDLL(plugin)


@stage("ft_launcher supports section timeouts")
def check_ft_launcher_flags():
    """Assert ft_launcher supports the section-timeout flags the launcher passes."""
    out = subprocess.run(["ft_launcher", "--help"], capture_output=True, text=True, timeout=120)
    help_text = out.stdout + out.stderr
    for flag in ("--ft-rank-section-timeouts", "--ft-rank-out-of-section-timeout"):
        assert flag in help_text, (
            f"ft_launcher lacks {flag} (image nvidia-resiliency-ext too old) — "
            "run training with --disable-ft or qualify a newer image"
        )


@stage("dataset helpers build/import (JIT toolchain)")
def check_dataset_helpers():
    """JIT-build and import Megatron dataset helpers (proves the image toolchain)."""
    # First import triggers Megatron's JIT `make` of the dataset index helper —
    # proves the image toolchain (compiler, python headers, pybind11) works.
    from megatron.core.datasets.utils import compile_helpers

    # compile_helpers() reports a failed `make` by calling sys.exit(1), and
    # SystemExit derives from BaseException, NOT Exception — so the @stage
    # wrapper's `except Exception` did not catch it and a broken image toolchain
    # killed the whole validator here, silently, with no summary and no listing
    # of the stages that had already passed. Translate it into an ordinary stage
    # failure so the run continues and the summary still prints.
    try:
        compile_helpers()
    except SystemExit as e:
        raise RuntimeError(
            f"megatron compile_helpers() called sys.exit({e.code}) — the JIT `make` of the "
            "dataset index helper failed. Check the `make` output above for the image's "
            "compiler / Python headers / pybind11."
        ) from e
    import megatron.core.datasets.helpers  # noqa: F401


# NOT a @stage: this only prints. A scored stage with no assertion can never
# fail, so it inflated the "N passed" count with a check that proves nothing —
# which makes the pass count useless as a health signal. It is an informational
# report instead, excluded from `results` and therefore from the summary.
def report_versions():
    """Print the environment version table (informational only — nothing is scored)."""
    import importlib.metadata as _m

    def v(pkg):
        try:
            return _m.version(pkg)
        except Exception:
            return "NOT INSTALLED"

    import torch

    print(
        f"    torch: {torch.__version__} (CUDA {torch.version.cuda}, NCCL {'.'.join(map(str, torch.cuda.nccl.version()))})"
    )
    for pkg in (
        "transformer-engine",
        "mamba-ssm",
        "causal-conv1d",
        "megatron-core",
        "transformers",
        "tokenizers",
        "nvidia-resiliency-ext",
        "triton",
        "flash-attn",
    ):
        print(f"    {pkg}: {v(pkg)}")


def main():
    parser = argparse.ArgumentParser(description="Validate the Megatron Bridge environment (runs in-container)")
    parser.add_argument("--run-training", action="store_true", help="Also run a tiny training job")
    args = parser.parse_args()

    print("=" * 50)
    print("  Megatron Bridge Environment Validation")
    print("=" * 50)
    print()

    print("Stage 1: Core Python imports")
    check_torch()
    check_mcore()
    check_mbridge()
    check_transformers()
    check_datasets()
    check_wandb()
    check_omegaconf()

    print("\nStage 2: CUDA extension imports")
    check_te()
    check_mamba()
    check_causal_conv()

    # No exact-pin table here: the versions are the image's, fixed by
    # CONTAINER_IMAGE_TAG rather than by a lockfile, and they are reported
    # verbatim by the informational version report at the end.
    print("\nStage 2b: MoE grouped GEMM")
    check_grouped_gemm()

    print("\nStage 3: CUDA availability")
    check_cuda()

    print("\nStage 4: GPU tensor operations")
    check_gpu_ops()

    print("\nStage 5: Recipe loading")
    check_vanilla_recipe()
    check_nemotron_recipe()

    print("\nStage 5b: Environment integrity (repo code, Slingshot, toolchain)")
    check_import_paths()
    check_nccl_plugin()
    check_ft_launcher_flags()
    check_dataset_helpers()

    # Informational, not scored (see report_versions). Guarded because it is now
    # undecorated: it touches torch.cuda.nccl.version(), which raises on a broken
    # CUDA stack, and a report must never abort the run before the summary prints
    # (the CUDA stack itself is already scored by stage 3).
    print("\nVersion report (informational — not a scored check)")
    try:
        report_versions()
    except Exception as e:  # broad on purpose: report-only, real checks are scored above
        print(f"  [{WARN}] version report unavailable: {e}")

    if args.run_training:
        print("\nStage 6: Tiny training run")
        run_tiny_training()

    # Summary
    print("\n" + "=" * 50)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed > 0:
        print("\nFailed stages:")
        for name, ok, detail in results:
            if not ok:
                print(f"  - {name}: {detail}")
        sys.exit(1)
    else:
        print("\nAll checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate Megatron Bridge installation on Isambard ARM HPC.

Usage:
    python validate_install.py              # Import and GPU checks only
    python validate_install.py --run-training  # Also run a tiny training job

Exit code 0 if all stages pass, 1 otherwise.
"""

import argparse
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
    import megatron.core


@stage("megatron.bridge")
def check_mbridge():
    import megatron.bridge


@stage("transformers")
def check_transformers():
    import transformers


@stage("datasets")
def check_datasets():
    import datasets


@stage("wandb")
def check_wandb():
    import wandb


@stage("omegaconf")
def check_omegaconf():
    import omegaconf


# ============================================
# Stage 2: CUDA extension imports
# ============================================
@stage("transformer_engine")
def check_te():
    import transformer_engine
    import transformer_engine.pytorch


@stage("mamba_ssm")
def check_mamba():
    import mamba_ssm


@stage("causal_conv1d")
def check_causal_conv():
    import causal_conv1d


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


def main():
    parser = argparse.ArgumentParser(description="Validate Megatron Bridge installation")
    parser.add_argument("--run-training", action="store_true", help="Also run a tiny training job")
    args = parser.parse_args()

    print("=" * 50)
    print("  Megatron Bridge Installation Validation")
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

    print("\nStage 3: CUDA availability")
    check_cuda()

    print("\nStage 4: GPU tensor operations")
    check_gpu_ops()

    print("\nStage 5: Recipe loading")
    check_vanilla_recipe()
    check_nemotron_recipe()

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

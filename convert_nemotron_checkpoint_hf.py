#!/usr/bin/env python3
"""Convert Megatron distributed checkpoints to HuggingFace format.

Converts Nemotron (Nano/Super) Megatron checkpoints to HuggingFace format,
with optional push to the HuggingFace Hub. Supports both single-process
conversion (default) and multi-GPU conversion for very large models.

The torch_dist checkpoint format supports resharding, so conversion is
independent of the training parallelism.

Usage:
    # Convert latest iteration (single process, no torchrun needed)
    python convert_nemotron_checkpoint_hf.py \
        --megatron-path /path/to/checkpoints/experiment_name

    # Convert specific iteration
    python convert_nemotron_checkpoint_hf.py \
        --megatron-path /path/to/checkpoints/experiment_name \
        --iteration 300

    # Convert and push to HuggingFace Hub
    python convert_nemotron_checkpoint_hf.py \
        --megatron-path /path/to/checkpoints/experiment_name \
        --iteration 300 \
        --push-to-hub

    # Multi-GPU fallback (if single-process OOMs)
    torchrun --nproc_per_node=8 convert_nemotron_checkpoint_hf.py \
        --megatron-path /path/to/checkpoints/experiment_name \
        --iteration 300 \
        --tp 1 --ep 8
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Known Nemotron model name -> HuggingFace model ID mapping
MODEL_ID_MAP = {
    "NVIDIA-Nemotron-3-Super-120B-A12B-BF16": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
}


def resolve_checkpoint_path(megatron_path: str, iteration: int | None = None) -> tuple[Path, int]:
    """Resolve the checkpoint iteration directory.

    Args:
        megatron_path: Top-level checkpoint directory containing iter_* subdirs.
        iteration: Specific iteration number, or None to use the latest.

    Returns:
        Tuple of (iteration directory path, iteration number).
    """
    base = Path(megatron_path)
    if not base.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {base}")

    if iteration is not None:
        iter_dir = base / f"iter_{iteration:07d}"
        if not iter_dir.exists():
            raise FileNotFoundError(f"Iteration directory not found: {iter_dir}")
        return iter_dir, iteration

    # Try latest_checkpointed_iteration.txt
    latest_file = base / "latest_checkpointed_iteration.txt"
    if latest_file.exists():
        iteration = int(latest_file.read_text().strip())
        iter_dir = base / f"iter_{iteration:07d}"
        if iter_dir.exists():
            return iter_dir, iteration

    # Fall back to scanning iter_* dirs
    iter_dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("iter_")]
    if not iter_dirs:
        raise FileNotFoundError(f"No iter_* directories found in {base}")

    latest = max(iter_dirs, key=lambda d: int(d.name.replace("iter_", "")))
    iteration = int(latest.name.replace("iter_", ""))
    return latest, iteration


def detect_hf_model_id(iter_path: Path) -> str | None:
    """Auto-detect the HuggingFace model ID from the checkpoint's run_config.yaml.

    Reads checkpoint.pretrained_checkpoint from run_config.yaml and maps the
    basename to a known HuggingFace model ID.

    Args:
        iter_path: Path to a specific iteration directory.

    Returns:
        HuggingFace model ID string, or None if detection fails.
    """
    run_config = iter_path / "run_config.yaml"
    if not run_config.exists():
        return None

    with open(run_config) as f:
        config = yaml.safe_load(f)

    pretrained_path = config.get("checkpoint", {}).get("pretrained_checkpoint")
    if not pretrained_path:
        return None

    model_name = Path(pretrained_path).name
    if model_name in MODEL_ID_MAP:
        return MODEL_ID_MAP[model_name]

    # Fall back to nvidia/<basename>
    return f"nvidia/{model_name}"


def _is_multi_gpu() -> bool:
    """Check if running under torchrun (multi-GPU mode)."""
    return os.environ.get("WORLD_SIZE") is not None


def convert_single_process(
    iter_path: Path,
    hf_path: Path,
    hf_model_id: str,
    strict: bool = True,
    show_progress: bool = True,
) -> None:
    """Convert using single-process export_ckpt (CPU-based distributed context)."""
    from megatron.bridge import AutoBridge

    print(f"Creating bridge from auto-config: {hf_model_id}")
    bridge = AutoBridge.from_auto_config(str(iter_path), hf_model_id)

    print(f"Exporting: {iter_path} -> {hf_path}")
    bridge.export_ckpt(
        megatron_path=str(iter_path),
        hf_path=str(hf_path),
        show_progress=show_progress,
        strict=strict,
    )
    print(f"Export complete: {hf_path}")


def convert_multi_gpu(
    iter_path: Path,
    hf_path: Path,
    hf_model_id: str,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    etp: int = 1,
    torch_dtype: torch.dtype = torch.bfloat16,
    strict: bool = True,
    show_progress: bool = True,
) -> None:
    """Convert using multi-GPU distributed loading."""
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.decorators import torchrun_main
    from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
    from megatron.bridge.utils.common_utils import print_rank_0

    @torchrun_main
    def _run():
        print_rank_0(f"Exporting: {iter_path} -> {hf_path}")
        print_rank_0(f"  TP={tp}  PP={pp}  EP={ep}  ETP={etp}  dtype={torch_dtype}")

        bridge = AutoBridge.from_hf_pretrained(
            hf_model_id,
            trust_remote_code=is_safe_repo(trust_remote_code=False, hf_path=hf_model_id),
            torch_dtype=torch_dtype,
        )

        model_provider = bridge.to_megatron_provider(load_weights=False)
        model_provider.tensor_model_parallel_size = tp
        model_provider.pipeline_model_parallel_size = pp
        model_provider.expert_model_parallel_size = ep
        model_provider.expert_tensor_parallel_size = etp
        model_provider.pipeline_dtype = torch_dtype
        model_provider.params_dtype = torch_dtype
        model_provider.finalize()
        model_provider.initialize_model_parallel(seed=0)

        mp_overrides = {
            "tensor_model_parallel_size": tp,
            "pipeline_model_parallel_size": pp,
            "expert_model_parallel_size": ep,
            "expert_tensor_parallel_size": etp,
            "pipeline_dtype": torch_dtype,
            "params_dtype": torch_dtype,
        }

        print_rank_0(f"Loading Megatron checkpoint from: {iter_path}")
        megatron_model = bridge.load_megatron_model(
            str(iter_path),
            mp_overrides=mp_overrides,
            wrap_with_ddp=False,
        )
        megatron_model = [m.cuda() for m in megatron_model]

        print_rank_0(f"Saving HuggingFace model to: {hf_path}")
        bridge.save_hf_pretrained(
            megatron_model,
            str(hf_path),
            show_progress=show_progress,
            strict=strict,
        )
        print_rank_0(f"Export complete: {hf_path}")

    _run()


def fixup_hf_output(hf_path: Path, hf_model_id: str) -> None:
    """Fix known issues in the converted HF output for eval/inference compatibility.

    1. Copies missing custom modeling files (configuration_nemotron_h.py,
       modeling_nemotron_h.py) from the HF cache if the config.json references
       them via auto_map but they weren't included by save_hf_pretrained.
    2. Fixes tokenizer_config.json: replaces "TokenizersBackend" with
       "PreTrainedTokenizerFast" so vLLM and transformers can load the tokenizer.
    """
    import json

    # Fix tokenizer_class
    tokenizer_config = hf_path / "tokenizer_config.json"
    if tokenizer_config.exists():
        with open(tokenizer_config) as f:
            tc = json.load(f)
        if tc.get("tokenizer_class") == "TokenizersBackend":
            tc["tokenizer_class"] = "PreTrainedTokenizerFast"
            with open(tokenizer_config, "w") as f:
                json.dump(tc, f, indent=2)
            print("Fixed tokenizer_class: TokenizersBackend -> PreTrainedTokenizerFast")

    # Copy missing custom modeling files from HF cache
    config_json = hf_path / "config.json"
    if not config_json.exists():
        return

    with open(config_json) as f:
        config = json.load(f)

    auto_map = config.get("auto_map", {})
    needed_modules = set()
    for _key, value in auto_map.items():
        module_name = value.split(".")[0] if "." in value else None
        if module_name:
            needed_modules.add(f"{module_name}.py")

    hf_cache_base = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    model_cache_name = f"models--{hf_model_id.replace('/', '--')}"
    model_cache = hf_cache_base / model_cache_name / "snapshots"

    for module_file in needed_modules:
        target = hf_path / module_file
        if target.exists():
            continue

        # Search HF cache snapshots for the file
        if model_cache.exists():
            for snapshot_dir in sorted(model_cache.iterdir(), reverse=True):
                source = snapshot_dir / module_file
                if source.exists():
                    import shutil
                    shutil.copy2(str(source), str(target))
                    print(f"Copied {module_file} from HF cache ({snapshot_dir.name[:8]})")
                    break
            else:
                print(f"Warning: {module_file} not found in HF cache for {hf_model_id}")
        else:
            print(f"Warning: HF cache not found at {model_cache}")


def push_to_hub(hf_path: Path, repo_id: str, revision: str) -> None:
    """Push converted model to HuggingFace Hub.

    Args:
        hf_path: Local path to the converted HF model.
        repo_id: Full repo ID (org/name).
        revision: Revision branch name (e.g., iter_0000300).
    """
    from huggingface_hub import HfApi

    api = HfApi()

    print(f"Creating/verifying repo: {repo_id}")
    api.create_repo(repo_id, exist_ok=True)

    # Create the revision branch if it doesn't exist
    print(f"Creating branch: {revision}")
    try:
        api.create_branch(repo_id, branch=revision)
    except Exception:
        pass  # Branch may already exist

    print(f"Uploading to {repo_id} (revision: {revision})")
    api.upload_folder(
        folder_path=str(hf_path),
        repo_id=repo_id,
        revision=revision,
        commit_message=f"Add checkpoint {revision}",
    )
    print(f"Upload complete: {repo_id} @ {revision}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Megatron checkpoints to HuggingFace format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument(
        "--megatron-path", required=True,
        help="Top-level checkpoint directory (contains iter_* subdirs)",
    )

    # Checkpoint selection
    parser.add_argument(
        "--iteration", type=int, default=None,
        help="Specific iteration to convert (default: latest from latest_checkpointed_iteration.txt)",
    )

    # Output
    parser.add_argument(
        "--hf-path", default=None,
        help="HuggingFace output directory (default: <megatron-path>/iter_N/hf)",
    )
    parser.add_argument(
        "--hf-model", default=None,
        help="HF model ID for config synthesis (default: auto-detect from run_config.yaml)",
    )
    parser.add_argument(
        "--torch-dtype", choices=list(DTYPE_MAP), default="bfloat16",
        help="Model precision (default: bfloat16)",
    )

    # Hub push
    parser.add_argument("--push-to-hub", action="store_true", help="Push converted model to HuggingFace Hub")
    parser.add_argument("--hf-org", default="geodesic-research", help="HuggingFace org (default: geodesic-research)")
    parser.add_argument(
        "--hf-repo-name", default=None,
        help="HuggingFace repo name (default: basename of --megatron-path)",
    )

    # Export options
    parser.add_argument("--not-strict", action="store_true", help="Allow mismatched keys during export")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")

    # Multi-GPU fallback
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism (multi-GPU fallback)")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism (multi-GPU fallback)")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism (multi-GPU fallback)")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism (multi-GPU fallback)")

    args = parser.parse_args()

    # 1. Resolve checkpoint path
    iter_path, iteration = resolve_checkpoint_path(args.megatron_path, args.iteration)
    print(f"Checkpoint: {iter_path} (iteration {iteration})")

    # 2. Determine HF model ID
    hf_model_id = args.hf_model or detect_hf_model_id(iter_path)
    if not hf_model_id:
        print("ERROR: Could not auto-detect HF model ID. Provide --hf-model explicitly.", file=sys.stderr)
        sys.exit(1)
    print(f"HF model ID: {hf_model_id}")

    # 3. Determine output path
    hf_path = Path(args.hf_path) if args.hf_path else iter_path / "hf"
    print(f"Output path: {hf_path}")

    # 4. Run conversion
    use_multi_gpu = _is_multi_gpu() and (args.tp > 1 or args.pp > 1 or args.ep > 1 or args.etp > 1)

    if use_multi_gpu:
        print(f"Mode: multi-GPU (TP={args.tp}, PP={args.pp}, EP={args.ep}, ETP={args.etp})")
        convert_multi_gpu(
            iter_path=iter_path,
            hf_path=hf_path,
            hf_model_id=hf_model_id,
            tp=args.tp,
            pp=args.pp,
            ep=args.ep,
            etp=args.etp,
            torch_dtype=DTYPE_MAP[args.torch_dtype],
            strict=not args.not_strict,
            show_progress=not args.no_progress,
        )
    else:
        print("Mode: single-process (CPU-based distributed context)")
        convert_single_process(
            iter_path=iter_path,
            hf_path=hf_path,
            hf_model_id=hf_model_id,
            strict=not args.not_strict,
            show_progress=not args.no_progress,
        )

    # 5. Cleanup distributed BEFORE fixup/push (prevents timeout while uploading)
    rank = int(os.environ.get("RANK", "0"))
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    # 6. Fix known HF output issues (rank 0 only)
    if rank == 0:
        fixup_hf_output(hf_path, hf_model_id)

    # 7. Push to Hub (rank 0 only — other ranks exit cleanly)
    if args.push_to_hub and rank == 0:
        repo_name = args.hf_repo_name or Path(args.megatron_path).name
        repo_id = f"{args.hf_org}/{repo_name}"
        revision = f"iter_{iteration:07d}"
        push_to_hub(hf_path, repo_id, revision)


if __name__ == "__main__":
    main()

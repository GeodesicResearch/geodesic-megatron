#!/usr/bin/env python3
"""Convert Megatron distributed checkpoints to HuggingFace format.

Converts Nemotron (Nano/Super) Megatron checkpoints to HuggingFace format,
with optional push to the HuggingFace Hub. Supports both single-process
conversion (default) and multi-GPU conversion for very large models.

The torch_dist checkpoint format supports resharding, so conversion is
independent of the training parallelism.

Usage:
    # Convert latest iteration (single process, no torchrun needed)
    python pipeline_checkpoint_convert_hf.py \
        --megatron-path /path/to/checkpoints/experiment_name

    # Convert specific iteration
    python pipeline_checkpoint_convert_hf.py \
        --megatron-path /path/to/checkpoints/experiment_name \
        --iteration 300

    # Convert and push to HuggingFace Hub
    python pipeline_checkpoint_convert_hf.py \
        --megatron-path /path/to/checkpoints/experiment_name \
        --iteration 300 \
        --push-to-hub

    # Multi-GPU fallback (if single-process OOMs)
    torchrun --nproc_per_node=8 pipeline_checkpoint_convert_hf.py \
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

# Base model -> instruct model mapping for chat template sourcing.
# Base models don't include a chat_template; the instruct variant does.
CHAT_TEMPLATE_SOURCE_MAP = {
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
}

# Simple ChatML template for non-reasoning SFT models. Matches the format used
# during standard SFT training: assistant messages are prefixed with empty
# <think></think> tags (closed, no open reasoning blocks).
# In transformers 5.x, chat_template.jinja takes precedence over the
# chat_template field in tokenizer_config.json, so this must be written to
# chat_template.jinja to take effect.
SIMPLE_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\\n<think></think>' }}"
    "{% endif %}"
)


def detect_reasoning_training(iter_path: Path) -> bool:
    """Auto-detect whether the SFT training data used reasoning/thinking tags.

    Checks the first 100 assistant messages in the training JSONL for <think>
    tags. If found, the model was trained with reasoning and should keep the
    full thinking chat template.

    Args:
        iter_path: Path to a specific iteration directory (contains run_config.yaml).

    Returns:
        True if reasoning tags detected, False otherwise.
    """
    import json as _json

    run_config = iter_path / "run_config.yaml"
    if not run_config.exists():
        return False

    with open(run_config) as f:
        config = yaml.safe_load(f)

    dataset_root = config.get("dataset", {}).get("dataset_root")
    if not dataset_root:
        return False

    training_jsonl = Path(dataset_root) / "training.jsonl"
    if not training_jsonl.exists():
        return False

    with open(training_jsonl) as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            try:
                record = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            # Check assistant messages for <think> tags
            messages = record.get("messages", [])
            if isinstance(messages, list):
                for msg in messages:
                    if msg.get("role") == "assistant" and "<think>" in msg.get("content", ""):
                        return True
            # Also check raw text field
            text = record.get("text", "")
            if "<think>" in text and "</think>" in text:
                return True

    return False


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


def fixup_hf_output(hf_path: Path, hf_model_id: str, reasoning: bool = False) -> None:
    """Fix known issues in the converted HF output for eval/inference compatibility.

    1. Fixes tokenizer_config.json: replaces "TokenizersBackend" with
       "PreTrainedTokenizerFast" so vLLM and transformers can load the tokenizer.
    2. Adds chat_template from the instruct model if missing (base models don't
       include one, but SFT checkpoints need it for generation).
    3. For non-reasoning models, replaces chat_template.jinja with a simple
       ChatML template (no open <think> blocks). In transformers 5.x,
       chat_template.jinja takes precedence over tokenizer_config.json.
    4. Removes auto_map and stale custom modeling files (transformers >= 5.3.0
       has native NemotronH support).

    Args:
        hf_path: Path to the converted HF output directory.
        hf_model_id: HuggingFace model ID used for conversion.
        reasoning: If True, keep the full thinking chat template. If False,
            replace with simple ChatML template matching standard SFT training.
    """
    import json

    hf_cache_base = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"

    # Fix tokenizer_class and add chat_template to tokenizer_config.json
    tokenizer_config = hf_path / "tokenizer_config.json"
    if tokenizer_config.exists():
        with open(tokenizer_config) as f:
            tc = json.load(f)

        changed = False

        if tc.get("tokenizer_class") == "TokenizersBackend":
            tc["tokenizer_class"] = "PreTrainedTokenizerFast"
            changed = True
            print("Fixed tokenizer_class: TokenizersBackend -> PreTrainedTokenizerFast")

        # Add chat_template from instruct model if missing
        if "chat_template" not in tc:
            source_model_id = CHAT_TEMPLATE_SOURCE_MAP.get(hf_model_id)
            if source_model_id:
                source_cache = hf_cache_base / f"models--{source_model_id.replace('/', '--')}" / "snapshots"
                if source_cache.exists():
                    for snapshot_dir in sorted(source_cache.iterdir(), reverse=True):
                        source_tc_path = snapshot_dir / "tokenizer_config.json"
                        if source_tc_path.exists():
                            with open(source_tc_path) as f:
                                source_tc = json.load(f)
                            if "chat_template" in source_tc:
                                tc["chat_template"] = source_tc["chat_template"]
                                changed = True
                                print(f"Added chat_template from {source_model_id} ({snapshot_dir.name[:8]})")
                                break
                    else:
                        print(f"Warning: chat_template not found in HF cache for {source_model_id}")
                else:
                    print(f"Warning: HF cache not found for {source_model_id} — run: "
                          f"python -c \"from transformers import AutoTokenizer; "
                          f"AutoTokenizer.from_pretrained('{source_model_id}')\"")

        if changed:
            with open(tokenizer_config, "w") as f:
                json.dump(tc, f, indent=2, ensure_ascii=False)

    # Fix chat_template.jinja (takes precedence over tokenizer_config.json in
    # transformers 5.x). For non-reasoning models, replace with simple template.
    jinja_path = hf_path / "chat_template.jinja"
    if not reasoning:
        jinja_path.write_text(SIMPLE_CHAT_TEMPLATE)
        print("Replaced chat_template.jinja with simple ChatML template (non-reasoning model)")
    elif jinja_path.exists():
        print("Kept existing chat_template.jinja (reasoning model)")
    else:
        print("No chat_template.jinja found (reasoning model — will use tokenizer_config.json)")

    # Remove auto_map and stale custom modeling files.
    # transformers >= 5.3.0 has native NemotronH support; the old custom code
    # uses "backbone.*" naming that conflicts with the standard "model.*" weights.
    config_json = hf_path / "config.json"
    if not config_json.exists():
        return

    with open(config_json) as f:
        config = json.load(f)

    if "auto_map" in config:
        # Remove any custom .py files referenced by auto_map
        for _key, value in config["auto_map"].items():
            module_name = value.split(".")[0] if "." in value else None
            if module_name:
                stale_file = hf_path / f"{module_name}.py"
                if stale_file.exists():
                    stale_file.unlink()
                    print(f"Removed stale {module_name}.py (native transformers handles this)")

        del config["auto_map"]
        with open(config_json, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print("Removed auto_map from config.json (using native transformers NemotronH)")

    # NOTE: Weight keys use "backbone.*" naming (from the bridge's save_hf_pretrained).
    # Do NOT rename to "model.*" — vLLM's NemotronH weight mapper expects "backbone.*"
    # and handles the conversion internally via hf_to_vllm_mapper. Transformers >= 5.3.0
    # also handles "backbone.*" natively.

    # Fix hybrid_override_pattern for vLLM compatibility.
    # The bridge saves the pattern with "-" for MoE layers, but the NVIDIA convention
    # uses "E". vLLM checks `"E" in config.hybrid_override_pattern` to detect MoE.
    # Also copy the pattern from the source HF model if not in config.json, since the
    # custom configuration_nemotron_h.py default uses "-" (wrong for vLLM).
    config_changed = False
    pattern = config.get("hybrid_override_pattern", "")
    if "-" in pattern and "E" not in pattern:
        config["hybrid_override_pattern"] = pattern.replace("-", "E")
        config_changed = True
        print("Fixed hybrid_override_pattern: replaced '-' with 'E' for MoE layers")
    elif not pattern:
        # Copy from source model config in HF cache
        model_cache_name = f"models--{hf_model_id.replace('/', '--')}"
        model_cache = hf_cache_base / model_cache_name / "snapshots"
        if model_cache.exists():
            for snapshot_dir in sorted(model_cache.iterdir(), reverse=True):
                source_cfg_path = snapshot_dir / "config.json"
                if source_cfg_path.exists():
                    with open(source_cfg_path) as f:
                        source_cfg = json.load(f)
                    if "hybrid_override_pattern" in source_cfg:
                        config["hybrid_override_pattern"] = source_cfg["hybrid_override_pattern"]
                        config_changed = True
                        print(f"Added hybrid_override_pattern from {hf_model_id}")
                        break
    if config_changed:
        with open(config_json, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


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

    # Chat template / reasoning
    reasoning_group = parser.add_mutually_exclusive_group()
    reasoning_group.add_argument(
        "--reasoning", action="store_true", default=None,
        help="Keep full thinking chat template (for reasoning-trained models)",
    )
    reasoning_group.add_argument(
        "--no-reasoning", action="store_true", default=None,
        help="Replace chat template with simple ChatML (for standard SFT models)",
    )

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

    # 6. Resolve reasoning mode: explicit flag > auto-detect from training data
    if rank == 0:
        if args.reasoning:
            reasoning = True
            print("Reasoning mode: enabled (--reasoning flag)")
        elif args.no_reasoning:
            reasoning = False
            print("Reasoning mode: disabled (--no-reasoning flag)")
        else:
            reasoning = detect_reasoning_training(iter_path)
            if reasoning:
                print("Reasoning mode: enabled (auto-detected <think> tags in training data)")
            else:
                print("Reasoning mode: disabled (no <think> tags found in training data)")

    # 7. Fix known HF output issues (rank 0 only)
    if rank == 0:
        fixup_hf_output(hf_path, hf_model_id, reasoning=reasoning)

    # 8. Push to Hub (rank 0 only — other ranks exit cleanly)
    if args.push_to_hub and rank == 0:
        repo_name = args.hf_repo_name or Path(args.megatron_path).name
        repo_id = f"{args.hf_org}/{repo_name}"
        revision = f"iter_{iteration:07d}"
        push_to_hub(hf_path, repo_id, revision)


if __name__ == "__main__":
    main()

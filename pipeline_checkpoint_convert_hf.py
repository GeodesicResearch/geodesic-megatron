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

    # An adapter-only PEFT checkpoint (see the README's export notes)
    torchrun --nproc_per_node=4 --nnodes=2 ... pipeline_checkpoint_convert_hf.py \
        --megatron-path /path/to/checkpoints/lora_run --hf-model <release> --reasoning \
        --tp 1 --pp 2 --ep 4

    # Keep NemotronH remote-code for consumers without native support
    # (e.g. vLLM trust_remote_code on transformers < 5.3.0)
    python pipeline_checkpoint_convert_hf.py \
        --megatron-path /path/to/checkpoints/experiment_name \
        --keep-remote-code \
        --remote-code-source /path/to/dir/with/modeling_nemotron_h.py

    # A LoRA (PEFT) checkpoint's adapter as a vLLM factor dir, no merge (single process, CPU)
    python pipeline_checkpoint_convert_hf.py \
        --megatron-path /path/to/checkpoints/lora_run \
        --format lora-factors \
        [--factors-path DIR] [--reference-alphas /path/to/rl_adapter/alphas.json]

LoRA factor dirs (``--format lora-factors``)
--------------------------------------------
The serving side (geodesic_utils.inference.vllm_extension; its docs/lora_factor_format.md) reads
``rank_<R>.safetensors`` holding ``{hf_name}.A`` / ``{hf_name}.B`` in bf16, plus an ``alphas.json`` mapping each
hf_name to alpha_over_r and ``__format_version__`` to the factor format, and applies
``delta = alpha_over_r * (B @ A)``. Format 2: every key a 2-D pair with HF-canonical rows. Format 3: format 2 plus
grouped routed-expert keys.

Two adapter layouts are read from the checkpoint's own tensors (DCP reassembles TP and EP shards at full logical
shape on CPU, so no model is built and no GPU is needed):

- plain ``LoRA`` on fused modules (the RL adapters):
  1.  ``decoder.layers.N.*`` -> ``backbone.layers.N.mixer.*``, per the NemotronH bridge's mapping table.
  2.  The fused ``linear_qkv`` is one Megatron module and three HF keys. Megatron interleaves its output rows by
      query group (for each of ``num_query_groups``, ``[heads_per_group*head_dim q | head_dim k | head_dim v]``),
      so q, k and v are strided; the row arithmetic comes from the run_config and is checked against the widths.
  3.  Mamba ``in_proj`` stores lora_B five-way split (``.z/.x/.B/.C/.dt``); HF's in_proj is their concatenation in
      that order (MambaInProjMapping).
  lora_A is shared across a module's HF keys, because ``delta = B @ A`` factorises the output dimension only.
- Tinker's layout (``CanonicalLoRA`` + ``GroupedExpertsLoRAAdapter``, "Expert LoRA"): separate q/k/v adapters,
  separate z and x adapters on in_proj (one HF in_proj key at rank 2r: A stacks both, B places each in its rows),
  the MoE latent projections (2-D keys), and per-expert LoRA on the routed experts as grouped keys
  ``mixer.experts.{up,down}_proj.weight`` (one factor shared, the other 3-D over experts).
"""

import argparse
import os
import shutil
from pathlib import Path

import torch

from pipeline.checkpoints import detect_training_tokenizer, is_peft_checkpoint, resolve_checkpoint_path
from pipeline.export.fixups import fixup_hf_output
from pipeline.export.hf import convert_multi_gpu, convert_single_process, is_multi_gpu
from pipeline.export.lora_factors import export_lora_factors
from pipeline.hub import push_to_hub


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def main():
    """Convert a Megatron checkpoint to HuggingFace format, or import one back."""
    parser = argparse.ArgumentParser(
        description="Convert Megatron checkpoints to HuggingFace format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument(
        "--megatron-path",
        required=True,
        help="Top-level checkpoint directory (contains iter_* subdirs)",
    )

    # Checkpoint selection
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Specific iteration to convert (default: latest from latest_checkpointed_iteration.txt)",
    )

    # Output
    parser.add_argument(
        "--hf-path",
        default=None,
        help="HuggingFace output directory (default: <megatron-path>/iter_N/hf)",
    )
    parser.add_argument(
        "--format",
        choices=("hf", "lora-factors"),
        default="hf",
        help="hf: a HuggingFace model dir (default). lora-factors: a LoRA checkpoint's adapter as a vLLM factor dir, "
        "unmerged (single process, no GPU; see 'LoRA factor dirs' below)",
    )
    parser.add_argument(
        "--hf-model",
        default=None,
        help="Upstream HF model ID (e.g. nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) "
        "whose architecture + tokenizer config the checkpoint should be exported "
        "against. Required for --format hf: there is no auto-detection.",
    )
    parser.add_argument(
        "--factors-path",
        default=None,
        help="lora-factors: the factor dir to write (default: <megatron-path>/iter_N/factors)",
    )
    parser.add_argument(
        "--reference-alphas",
        default=None,
        help="lora-factors: an RL adapter's alphas.json on the same model; the export must match its keys and widths",
    )
    parser.add_argument(
        "--alpha-over-r",
        type=float,
        default=None,
        help="lora-factors: the adapter scale (default: the run_config's peft alpha / dim, which must then be there)",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=list(DTYPE_MAP),
        default="bfloat16",
        help="Model precision (default: bfloat16)",
    )

    # Hub push
    parser.add_argument("--push-to-hub", action="store_true", help="Push converted model to HuggingFace Hub")
    parser.add_argument("--hf-org", default="geodesic-research", help="HuggingFace org (default: geodesic-research)")
    parser.add_argument(
        "--hf-repo-name",
        default=None,
        help="HuggingFace repo name (default: basename of --megatron-path)",
    )

    # Export options
    parser.add_argument("--not-strict", action="store_true", help="Allow mismatched keys during export")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    parser.add_argument(
        "--keep-remote-code",
        action="store_true",
        help=(
            "Keep the NemotronH remote-code (auto_map + custom modeling files) in the "
            "exported HF dir instead of stripping it. Needed for consumers without native "
            "NemotronH support (e.g. vLLM trust_remote_code on transformers < 5.3.0). "
            "Pair with --remote-code-source to supply vLLM-correct modeling files."
        ),
    )
    parser.add_argument(
        "--remote-code-source",
        default=None,
        help=(
            "Directory holding vLLM-correct NemotronH modeling files "
            "(configuration_nemotron_h.py, modeling_nemotron_h.py) to copy into the export "
            "when --keep-remote-code is set and they are not already present."
        ),
    )

    # Chat template / reasoning — REQUIRED for --format hf (exactly one; checked after parsing).
    # Controls the `enable_thinking` default in the exported chat_template.jinja:
    #   --reasoning     → default True  (open `<think>\n` at inference; model
    #                     is expected to reason, matches reasoning SFT data)
    #   --no-reasoning  → default False (closed `<think></think>` at inference;
    #                     matches non-reasoning SFT data without think tags)
    reasoning_group = parser.add_mutually_exclusive_group()
    reasoning_group.add_argument(
        "--reasoning",
        action="store_true",
        default=None,
        help="Model was trained with <think>...</think> reasoning traces. "
        "Exports set enable_thinking=True by default.",
    )
    reasoning_group.add_argument(
        "--no-reasoning",
        action="store_true",
        default=None,
        help="Model was trained as instruct/SFT without reasoning traces. "
        "Exports set enable_thinking=False by default.",
    )

    # Multi-GPU fallback
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism (multi-GPU fallback)")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism (multi-GPU fallback)")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism (multi-GPU fallback)")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism (multi-GPU fallback)")

    args = parser.parse_args()
    hf_only = {
        "--hf-model": args.hf_model,
        "--hf-path": args.hf_path,
        "--push-to-hub": args.push_to_hub,
        "--keep-remote-code": args.keep_remote_code,
        "--remote-code-source": args.remote_code_source,
        "--reasoning/--no-reasoning": args.reasoning or args.no_reasoning,
    }
    factors_only = {
        "--factors-path": args.factors_path,
        "--reference-alphas": args.reference_alphas,
        "--alpha-over-r": args.alpha_over_r,
    }
    if args.format == "lora-factors":
        given = [flag for flag, value in hf_only.items() if value]
        if given:
            parser.error(f"--format lora-factors writes a factor dir; {', '.join(given)} apply to --format hf only")
    else:
        given = [flag for flag, value in factors_only.items() if value is not None]
        if given:
            parser.error(f"{', '.join(given)} apply to --format lora-factors only")
        if not args.hf_model:
            parser.error("--format hf requires --hf-model")
        if not (args.reasoning or args.no_reasoning):
            parser.error("--format hf requires one of --reasoning / --no-reasoning")

    # 1. Resolve checkpoint path
    iter_path, iteration = resolve_checkpoint_path(args.megatron_path, args.iteration)
    print(f"Checkpoint: {iter_path} (iteration {iteration})")

    if args.format == "lora-factors":
        export_lora_factors(
            iter_path,
            Path(args.factors_path) if args.factors_path else iter_path / "factors",
            alpha_over_r=args.alpha_over_r,
            reference_alphas=Path(args.reference_alphas) if args.reference_alphas else None,
        )
        return

    # 2. Determine HF model ID
    hf_model_id = args.hf_model
    print(f"HF model ID: {hf_model_id}")

    # 3. Determine output path
    hf_path = Path(args.hf_path) if args.hf_path else iter_path / "hf"
    print(f"Output path: {hf_path}")

    # 4. Run conversion
    use_multi_gpu = is_multi_gpu() and (args.tp > 1 or args.pp > 1 or args.ep > 1 or args.etp > 1)
    if is_peft_checkpoint(iter_path) and not use_multi_gpu:
        raise SystemExit(
            f"{iter_path} is an adapter-only PEFT checkpoint, which loads with its base on GPUs: run it under torchrun "
            "with a parallel size above 1 (pipeline_checkpoint_convert.sh export sets them)."
        )

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

    # 6. Resolve reasoning mode from required CLI flag (argparse guarantees
    #    exactly one of --reasoning / --no-reasoning is set).
    reasoning = bool(args.reasoning)
    if rank == 0:
        print(f"Reasoning mode: {'enabled (--reasoning)' if reasoning else 'disabled (--no-reasoning)'}")

    # 7. Fix known HF output issues (rank 0 only). Pass the *training-time*
    #    tokenizer id (from run_config.yaml > tokenizer > tokenizer_model) so
    #    the converted dir's chat_template matches what the model was actually
    #    trained on, instead of grafting from the upstream Instruct release.
    if rank == 0:
        training_tokenizer_id = detect_training_tokenizer(iter_path)
        if training_tokenizer_id:
            print(f"Training tokenizer (run_config.yaml): {training_tokenizer_id}")
        fixup_hf_output(
            hf_path,
            hf_model_id,
            reasoning=reasoning,
            training_tokenizer_id=training_tokenizer_id,
            keep_remote_code=args.keep_remote_code,
            remote_code_source=args.remote_code_source,
        )

    # 7b. Copy Megatron run_config.yaml into hf/ for provenance. Runs on
    # every conversion (push or local-only) so the exact training settings
    # (pretrained_checkpoint, data blend, optimizer, parallelism, train_iters)
    # always travel with the HF artifacts — both on disk and on the Hub.
    if rank == 0:
        src_run_config = iter_path / "run_config.yaml"
        if src_run_config.exists():
            dst_run_config = hf_path / "megatron_run_config.yaml"
            shutil.copy2(src_run_config, dst_run_config)
            print(f"Copied Megatron run_config.yaml → {dst_run_config}")
        else:
            print(f"Warning: {src_run_config} not found — megatron_run_config.yaml will not be created")

    # 8. Push to Hub (rank 0 only — other ranks exit cleanly)
    if args.push_to_hub and rank == 0:
        repo_name = args.hf_repo_name or Path(args.megatron_path).name
        repo_id = f"{args.hf_org}/{repo_name}"
        revision = f"iter_{iteration:07d}"
        push_to_hub(hf_path, repo_id, revision)


if __name__ == "__main__":
    main()

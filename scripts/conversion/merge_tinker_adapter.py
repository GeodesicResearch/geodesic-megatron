#!/usr/bin/env python3
"""Merge a downloaded Tinker LoRA adapter into its HF base model, optionally without some of its modules.

Wraps ``tinker_cookbook.weights.build_hf_model`` (shard-by-shard ``W + (alpha / r) B @ A``, Tinker's own export).
``--drop-modules REGEX`` first writes a copy of the adapter without the tensors whose names match, so that the
merged model holds the rest of the update only: ``--drop-modules 'lm_head'`` gives the model without the
unembedding LoRA. The copy goes in ``<output>.adapter`` and is kept, with the dropped tensor names in
``dropped_tensors.json``. A local base's files that the merge does not write, such as ``generation_config.json``,
are copied into the merged model.

Needs a venv with ``tinker-cookbook`` (its pinned revision decides the merge); it does not run in the pipeline
container. Usage:

    tinker checkpoint download tinker://<run>/sampler_weights/<name> --output <adapter_dir>
    python scripts/conversion/merge_tinker_adapter.py --adapter <adapter_dir> --base <hf_base_dir> \\
        --output <merged_dir> [--drop-modules 'lm_head']
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


def drop_adapter_modules(adapter: Path, output: Path, pattern: str) -> list[str]:
    """Copy the PEFT adapter at ``adapter`` to ``output`` without the tensors whose names match ``pattern``.

    Returns the dropped tensor names, also written to ``output/dropped_tensors.json``.

    Raises:
        ValueError: ``pattern`` matches no tensor, so the copy would equal the adapter.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    regex = re.compile(pattern)
    output.mkdir(parents=True)
    source = adapter / "adapter_model.safetensors"
    kept, dropped = {}, []
    with safe_open(str(source), framework="pt") as handle:
        metadata = handle.metadata()
        for name in handle.keys():
            if regex.search(name):
                dropped.append(name)
            else:
                kept[name] = handle.get_tensor(name)
    if not dropped:
        raise ValueError(f"--drop-modules {pattern!r} matches no tensor of {source}")
    save_file(kept, str(output / "adapter_model.safetensors"), metadata=metadata)
    for other in adapter.iterdir():
        if other.name != "adapter_model.safetensors" and other.is_file():
            shutil.copy2(other, output / other.name)
    (output / "dropped_tensors.json").write_text(json.dumps(sorted(dropped), indent=2) + "\n")
    return dropped


def copy_missing_base_files(base: Path, output: Path) -> list[str]:
    """Copy the base's non-weight files that the merge did not write (such as generation_config.json) to ``output``.

    The merge writes the weights, config, tokenizer and modelling files but not every file of the release, so the
    merged model would otherwise generate with different defaults from its base. Returns the copied names.
    """
    copied = []
    for path in sorted(base.iterdir()):
        if path.is_file() and not path.name.endswith(".safetensors") and not (output / path.name).exists():
            shutil.copy2(path, output / path.name)
            copied.append(path.name)
    return copied


def main() -> None:
    """Merge the adapter (less any --drop-modules tensors) into the base and save the HF model."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--adapter", required=True, type=Path, help="downloaded Tinker adapter directory")
    parser.add_argument("--base", required=True, help="HF base model directory or id")
    parser.add_argument("--output", required=True, type=Path, help="merged HF model directory (must not exist)")
    parser.add_argument("--drop-modules", default=None, help="regex of adapter tensor names to leave out")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    from tinker_cookbook import weights

    adapter = args.adapter
    if args.drop_modules:
        adapter = args.output.with_name(args.output.name + ".adapter")
        dropped = drop_adapter_modules(args.adapter, adapter, args.drop_modules)
        print(f"Dropped {len(dropped)} adapter tensors matching {args.drop_modules!r}: {dropped}", flush=True)
    weights.build_hf_model(
        base_model=args.base, adapter_path=str(adapter), output_path=str(args.output), dtype=args.dtype
    )
    print(f"Merged {adapter} into {args.base}: {args.output}", flush=True)
    if Path(args.base).is_dir():
        for copied in copy_missing_base_files(Path(args.base), args.output):
            print(f"Copied {copied} from the base", flush=True)


if __name__ == "__main__":
    main()

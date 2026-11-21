# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Packing a prepared dataset root's JSONL into parquet.

Catalogue:
    packer_arguments: the command-line arguments of scripts/data/pack_sft_dataset.py for one dataset root.
    run_pack: run scripts/data/pack_sft_dataset.py on a dataset root.
"""

import json
import subprocess
import sys
from pathlib import Path


def packer_arguments(
    dataset_root,
    tokenizer,
    seq_length,
    pad_seq_to_mult,
    packing_algorithm,
    dataset_kwargs,
    *,
    no_validation,
    no_chat,
    exclude_rows,
    shard_index,
    num_tokenizer_workers,
):
    """The arguments of scripts/data/pack_sft_dataset.py that pack ``dataset_root`` as the parameters state.

    ``no_validation`` and ``no_chat`` add the flags of those names. ``exclude_rows`` is a JSON Lines file of lines to
    leave out, or None for none; with a file, ``shard_index`` names the shard whose entries apply, and without one it
    must be None. Raises ValueError otherwise. ``num_tokenizer_workers`` is the number of tokenizer processes, or None
    for the packer's own default; the pack does not depend on it.
    """
    if (exclude_rows is None) != (shard_index is None):
        raise ValueError("exclude_rows and shard_index are given together or not at all")
    arguments = [
        "--dataset-root",
        str(dataset_root),
        "--tokenizer",
        tokenizer,
        "--seq-length",
        str(seq_length),
        "--pad-seq-to-mult",
        str(pad_seq_to_mult),
        "--packing-algorithm",
        packing_algorithm,
        "--dataset-kwargs",
        json.dumps(dataset_kwargs),
    ]
    if no_validation:
        arguments.append("--no-validation")
    if no_chat:
        arguments.append("--no-chat")
    if exclude_rows is not None:
        arguments += ["--exclude-rows", str(exclude_rows), "--shard-index", str(shard_index)]
    if num_tokenizer_workers is not None:
        arguments += ["--num-tokenizer-workers", str(num_tokenizer_workers)]
    return arguments


def run_pack(
    output_dir, tokenizer, seq_length, pad_seq_to_mult, has_validation, format_type, packing_algorithm, dataset_kwargs
):
    """Run pack_sft_dataset.py via subprocess."""
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "data" / "pack_sft_dataset.py"

    if not script_path.exists():
        print(f"\nError: pack_sft_dataset.py not found at {script_path}")
        return False

    cmd = [
        sys.executable,
        str(script_path),
        *packer_arguments(
            output_dir,
            tokenizer,
            seq_length,
            pad_seq_to_mult,
            packing_algorithm,
            dataset_kwargs,
            no_validation=not has_validation,
            no_chat=format_type == "pretraining",
            exclude_rows=None,
            shard_index=None,
            num_tokenizer_workers=None,
        ),
    ]

    print("\nRunning packing command:")
    print(f"  {' '.join(cmd)}")
    print()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        print(f"[PACK] {line}", end="")

    process.wait()

    if process.returncode != 0:
        print(f"\nError: pack_sft_dataset.py failed with return code {process.returncode}")
        return False

    return True

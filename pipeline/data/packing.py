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
    run_pack: run scripts/data/pack_sft_dataset.py on a dataset root.
"""

import json
import subprocess
import sys
from pathlib import Path


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
        "--dataset-root",
        str(output_dir),
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

    if not has_validation:
        cmd.append("--no-validation")

    if format_type == "pretraining":
        cmd.append("--no-chat")

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

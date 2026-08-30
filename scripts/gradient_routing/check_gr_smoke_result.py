# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Post-checks for the GR functional smoke: did the aux modules actually train?

Reads the torch_dist checkpoints WITHOUT building a model (metadata + targeted tensor
loads, the extract_base_zero_emb_ids.py pattern):

- The seed (phase-1) checkpoint must contain NO gr_aux keys.
- The GR (phase-2) checkpoint must contain gr_aux fc1/fc2 keys for every MoE layer, and
  the fc2 weights — zero-initialised at load — must be NON-zero after training: proof the
  aux modules received optimizer updates through the whole distributed stack.
"""

import argparse

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader


def _latest_iter_dir(ckpt_root: str) -> str:
    with open(f"{ckpt_root}/latest_checkpointed_iteration.txt") as f:
        iteration = int(f.read().strip())
    return f"{ckpt_root}/iter_{iteration:07d}"


def _gr_aux_keys(reader: FileSystemReader) -> list[str]:
    meta = reader.read_metadata().state_dict_metadata
    return sorted(k for k in meta if ".gr_aux." in k and k.endswith("weight"))


def _load_tensor(reader: FileSystemReader, key: str) -> torch.Tensor:
    meta = reader.read_metadata().state_dict_metadata[key]
    placeholder = torch.empty(list(meta.size), dtype=meta.properties.dtype, device="cpu")
    dcp.load(state_dict={key: placeholder}, storage_reader=reader)
    return placeholder


def main() -> None:
    """Run the seed/GR checkpoint post-checks; exits non-zero on any failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-checkpoint", required=True)
    parser.add_argument("--gr-checkpoint", required=True)
    parser.add_argument(
        "--expect-aux-modules",
        type=int,
        required=True,
        help="Number of aux modules per layer the GR config trained (asserted per layer).",
    )
    args = parser.parse_args()

    seed_reader = FileSystemReader(_latest_iter_dir(args.seed_checkpoint))
    seed_aux = _gr_aux_keys(seed_reader)
    if seed_aux:
        raise SystemExit(f"FAIL: seed checkpoint unexpectedly contains gr_aux keys: {seed_aux[:4]} ...")
    print("PASS: seed checkpoint carries no gr_aux keys")

    gr_reader = FileSystemReader(_latest_iter_dir(args.gr_checkpoint))
    gr_aux = _gr_aux_keys(gr_reader)
    fc2_keys = [k for k in gr_aux if "linear_fc2" in k]
    fc1_keys = [k for k in gr_aux if "linear_fc1" in k]
    if not fc2_keys or len(fc1_keys) != len(fc2_keys):
        raise SystemExit(f"FAIL: GR checkpoint aux key census broken: fc1={len(fc1_keys)} fc2={len(fc2_keys)}")
    # Key shape: ...mlp.gr_aux.<module index>.linear_fc{1,2}.weight — the per-layer module
    # count must match the config, or a swap that silently built fewer modules would pass.
    module_indices = sorted({int(k.split(".gr_aux.")[1].split(".")[0]) for k in fc2_keys})
    if module_indices != list(range(args.expect_aux_modules)):
        raise SystemExit(
            f"FAIL: GR checkpoint carries aux module indices {module_indices}, expected "
            f"0..{args.expect_aux_modules - 1}."
        )
    print(f"PASS: GR checkpoint carries {len(fc2_keys)} aux weight keys across {len(module_indices)} module(s)")

    zero_fc2 = []
    for key in fc2_keys:
        w = _load_tensor(gr_reader, key)
        norm = float(w.to(torch.float32).norm())
        print(f"  {key}: ||W|| = {norm:.6f}")
        if norm == 0.0:
            zero_fc2.append(key)
    if zero_fc2:
        raise SystemExit(
            f"FAIL: {len(zero_fc2)} aux fc2 weights are still exactly zero after training — "
            f"the aux modules never received an optimizer update: {zero_fc2}"
        )
    print("PASS: every aux fc2 weight moved off its zero init (aux modules trained)")


if __name__ == "__main__":
    main()

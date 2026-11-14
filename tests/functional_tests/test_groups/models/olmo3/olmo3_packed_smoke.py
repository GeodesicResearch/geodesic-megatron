# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Smoke run of an OLMo-3 model on a packed (THD) batch with padded documents, at one context-parallel size.

OLMo-3 is trained on packs of 32,768 tokens that hold documents, each padded to a multiple of 8. This harness answers
three questions about that path: whether Transformer Engine accepts a packed batch together with the 4,095-token
sliding window and the padded ``cu_seqlens``, which attention backend it picks, and whether context parallel size 2
with the ``a2a`` exchange gives the same loss and gradients as size 1. It runs one forward and backward pass at the
context-parallel size it is launched with and writes a JSON report. The test module runs it at size 1 and size 2 and
compares the two reports.

Launch with ``torchrun --nproc_per_node=<cp>`` and with ``NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2`` in the environment, because
Transformer Engine reads those two variables when it is imported, and it logs the backend it selected (and the reason
for each backend it rejected) only then.

The model is the first ``--layers`` layers of the base checkpoint's architecture, with random weights. The first four
layers of OLMo-3 are three sliding-window layers and one full-attention layer, so four layers exercise both kinds. The
batch follows the layout of ``GPTSFTPackedDataset.collate_fn`` (padded and unpadded ``cu_seqlens`` with the ``-1``
tail, a zero-length final entry when the pack is not full, per-document position ids) and goes through the training
step's own ``_partition_packed_batch_for_cp`` and ``get_packed_seq_params``.
"""

import argparse
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from megatron.core import parallel_state

from megatron.bridge.recipes.olmo import olmo3_sft_config
from megatron.bridge.training.gpt_step import _partition_packed_batch_for_cp
from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_params


#: The logger Transformer Engine writes its attention backend decisions to.
TE_ATTENTION_LOGGER = "DotProductAttention"


class RecordMessages(logging.Handler):
    """Keeps the text of every record, so that the report can quote Transformer Engine's backend decisions."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def write_truncated_config(base_dir: str, layers: int, out_dir: str) -> int:
    """Write a ``config.json`` holding the first ``layers`` layers of the base model's architecture.

    Returns:
        The id of the end-of-sequence token, which pads every document of the pack.
    """
    config = json.loads((Path(base_dir) / "config.json").read_text())
    if layers > config["num_hidden_layers"]:
        raise ValueError(f"--layers {layers} exceeds the base model's {config['num_hidden_layers']} layers")
    config["num_hidden_layers"] = layers
    config["layer_types"] = config["layer_types"][:layers]
    (Path(out_dir) / "config.json").write_text(json.dumps(config, indent=2))
    eos = config["eos_token_id"]
    if not isinstance(eos, int):
        raise ValueError(f"expected one eos_token_id in the base config, found {eos!r}")
    return eos


def ceil_to_multiple(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def build_pack(seq_len: int, pad_mult: int, fill: float, eos_id: int, seed: int) -> dict:
    """One synthetic pack, laid out as the packed collate lays it out.

    Documents have random lengths (30% of them under 64 tokens, the rest up to 8,000), each padded with the end-of-
    sequence token to a multiple of ``pad_mult``. They fill ``fill`` of the pack. A pack that is not full ends in one
    more entry of ``cu_seqlens`` that covers the rest, and the unpadded row repeats its last value there, as the
    collate does.
    """
    rng = np.random.default_rng(seed)
    target = int(seq_len * fill) // pad_mult * pad_mult
    real_lengths: list[int] = []
    padded_lengths: list[int] = []
    used = 0
    while used < target:
        length = int(rng.integers(1, 64)) if rng.random() < 0.3 else int(rng.integers(64, 8001))
        padded = ceil_to_multiple(length, pad_mult)
        if used + padded > target:
            padded = target - used
            length = min(length, padded)
        real_lengths.append(length)
        padded_lengths.append(padded)
        used += padded

    tokens = torch.full((seq_len,), eos_id, dtype=torch.int64)
    labels = torch.full((seq_len,), eos_id, dtype=torch.int64)
    loss_mask = torch.zeros(seq_len, dtype=torch.int64)
    position_ids = torch.zeros(seq_len, dtype=torch.int64)
    cu_padded = [0]
    cu_unpadded = [0]
    for length, padded in zip(real_lengths, padded_lengths, strict=True):
        start = cu_padded[-1]
        # Draw below the end-of-sequence id, so that no real token reads as padding.
        tokens[start : start + length] = torch.from_numpy(rng.integers(0, eos_id, length))
        labels[start : start + length] = torch.from_numpy(rng.integers(0, eos_id, length))
        loss_mask[start : start + length] = 1
        position_ids[start : start + padded] = torch.arange(padded)
        cu_padded.append(start + padded)
        cu_unpadded.append(cu_unpadded[-1] + length)
    if cu_padded[-1] != seq_len:
        cu_padded.append(seq_len)
        cu_unpadded.append(cu_unpadded[-1])

    def collate_row(row: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        tensor = torch.IntTensor([row + [-1]])
        return tensor, torch.argmin(tensor, dim=1, keepdim=True)

    cu_seqlens, cu_seqlens_argmin = collate_row(cu_padded)
    cu_seqlens_unpadded, cu_seqlens_unpadded_argmin = collate_row(cu_unpadded)
    max_seqlen = (cu_seqlens[:, 1:] - cu_seqlens[:, :-1]).max(dim=1, keepdim=True)[0]
    return {
        "tokens": tokens[None],
        "labels": labels[None],
        "loss_mask": loss_mask[None],
        "position_ids": position_ids[None],
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_argmin": cu_seqlens_argmin,
        "cu_seqlens_unpadded": cu_seqlens_unpadded,
        "cu_seqlens_unpadded_argmin": cu_seqlens_unpadded_argmin,
        "max_seqlen": max_seqlen,
        "documents": len(real_lengths),
        "real_tokens": int(sum(real_lengths)),
    }


def build_model(config_dir: str, cp: int):
    """The model of the family's recipe at this context-parallel size, through the provider's own build."""
    cfg = olmo3_sft_config(config_dir)
    provider = cfg.model
    provider.context_parallel_size = cp
    provider.params_dtype = torch.bfloat16
    provider.bf16 = True
    # Without the data-parallel wrapper nothing allocates the main-gradient buffers that the fused weight-gradient
    # accumulation writes to, so the plain gradient is used here.
    provider.gradient_accumulation_fusion = False
    provider.finalize()
    return provider.provide_distributed_model(wrap_with_ddp=False)


def parameter_checksum(model) -> float:
    return float(sum(p.detach().double().abs().sum().item() for p in model.parameters()))


def run_step(model, batch: dict, cp: int, seq_len: int) -> dict:
    """One forward and backward pass over this rank's share of the pack, with the training step's own helpers."""
    device = torch.cuda.current_device()
    tensors = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
    if cp > 1:
        tensors = _partition_packed_batch_for_cp(tensors, cp, parallel_state.get_context_parallel_rank())
    packed_seq_params = get_packed_seq_params(
        {
            "cu_seqlens": tensors["cu_seqlens"],
            "cu_seqlens_argmin": tensors["cu_seqlens_argmin"],
            "max_seqlen": tensors["max_seqlen"],
            "cu_seqlens_unpadded": tensors["cu_seqlens_unpadded"],
            "cu_seqlens_unpadded_argmin": tensors["cu_seqlens_unpadded_argmin"],
        },
        total_tokens=seq_len,
    )
    loss_mask = tensors["loss_mask"].float()

    model.zero_grad(set_to_none=True)
    per_token_loss = model(
        input_ids=tensors["tokens"],
        position_ids=tensors["position_ids"],
        attention_mask=torch.ones(1, dtype=torch.long, device=device),
        labels=tensors["labels"],
        packed_seq_params=packed_seq_params,
    )
    loss_sum = (per_token_loss.float() * loss_mask).sum()
    token_count = loss_mask.sum()
    if cp > 1:
        group = parallel_state.get_context_parallel_group()
        dist.all_reduce(token_count, group=group)
        loss_sum_global = loss_sum.detach().clone()
        dist.all_reduce(loss_sum_global, group=group)
    else:
        loss_sum_global = loss_sum.detach()
    # Each rank's gradient covers its own tokens. Dividing by the pack's token count and summing over the group gives
    # the gradient of the whole pack, which is what the data-parallel reduction of a real run produces.
    (loss_sum / token_count).backward()

    grad_norms = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"no gradient for {name}")
        if cp > 1:
            dist.all_reduce(parameter.grad, group=parallel_state.get_context_parallel_group())
        grad_norms[name] = float(parameter.grad.float().norm())
    return {
        "loss": float(loss_sum_global / token_count),
        "token_count": int(token_count),
        "local_tokens": int(tensors["tokens"].numel()),
        "grad_norm": float(np.sqrt(sum(norm**2 for norm in grad_norms.values()))),
        "grad_norms": grad_norms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-config-dir", required=True, help="a directory holding the base model's config.json")
    parser.add_argument(
        "--cp", type=int, required=True, help="the context-parallel size; torchrun must start as many ranks"
    )
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--pad-mult", type=int, required=True)
    parser.add_argument("--fill", type=float, required=True, help="the share of the pack that documents cover")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    for variable in ("NVTE_DEBUG", "NVTE_DEBUG_LEVEL"):
        if os.environ.get(variable) != ("1" if variable == "NVTE_DEBUG" else "2"):
            raise RuntimeError(
                f"{variable} must be set before the process starts: Transformer Engine reads it on import"
            )
    if int(os.environ.get("WORLD_SIZE", "1")) != args.cp:
        raise RuntimeError(f"--cp {args.cp} needs torchrun --nproc_per_node={args.cp}")

    backend_log = RecordMessages()
    logging.getLogger(TE_ATTENTION_LOGGER).addHandler(backend_log)

    with tempfile.TemporaryDirectory() as config_dir:
        eos_id = write_truncated_config(args.base_config_dir, args.layers, config_dir)
        model = build_model(config_dir, args.cp)[0]
    batch = build_pack(args.seq_len, args.pad_mult, args.fill, eos_id, args.seed)

    iterations = []
    for _ in range(args.iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = run_step(model, batch, args.cp, args.seq_len)
        torch.cuda.synchronize()
        result["seconds"] = time.perf_counter() - started
        iterations.append(result)

    last = iterations[-1]
    finite = bool(np.isfinite(last["loss"]) and np.isfinite(last["grad_norm"]))
    report = {
        "cp": args.cp,
        "layers": args.layers,
        "seq_len": args.seq_len,
        "pad_mult": args.pad_mult,
        "fill": args.fill,
        "seed": args.seed,
        "documents": batch["documents"],
        "real_tokens": batch["real_tokens"],
        "parameter_checksum": parameter_checksum(model),
        "finite": finite,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "iterations": [{key: value for key, value in it.items() if key != "grad_norms"} for it in iterations],
        "grad_norms": last["grad_norms"],
        "te_attention_messages": backend_log.messages,
    }
    if dist.get_rank() == 0:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2))
        selected = [m for m in backend_log.messages if "Selected backend" in m]
        print(f"[olmo3-packed-smoke] cp={args.cp} finite={finite} loss={last['loss']:.6f} backend={selected}")
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Score one Megatron checkpoint on a held-out evaluation set: for each document, the summed negative log-likelihood
of its tokens in nats (``configs/olmo_mix_efficiency/README.md``). The sum over a group of documents, divided by
their bytes, is the group's bits per byte, a loss that models with different tokenizers share.

**What is scored.** The evaluation set is a directory holding ``index.<tokenizer>.json`` (one mapping per document:
``id``, ``family``, ``domain``, ``offset``, ``targets``, ``bytes``), ``targets.<tokenizer>.int32.bin`` (document ``i``
owns the ids ``[offset, offset + targets)``) and ``manifest.json``. A window is the prefix id followed by one
document's ids, so every id of the document is a target and position ``j`` predicts id ``j + 1``. The windows differ
in length (2 to 8,192 ids).

**A prompt and its answer.** An index entry with a ``context`` count is a prompt followed by its gold answer, as a
task of a benchmark suite gives them. Its first ``context`` ids are inputs only and no prefix id goes before them. The
targets are the ids of the answer, and ``bytes`` counts the answer alone. OLMES and lm-eval score a continuation in
this way, so the per-document bits per byte of such an entry is the OLMES metric ``bits_per_byte_corr``.

**The tokenizer.** ``--tokenizer-name`` names the token files to read, which must be the ones of the tokenizer the
checkpoint trained with (the model must have an output row for every id). The prefix id is the id that ended every
document of that tokenizer's training data, so that a window starts as a document of the training data starts:
``<|endoftext|>`` (100257) for dolma2, ``</s>`` (2) for the Nemotron base tokenizer. It is ``--prefix-id``, else the
``prefix_id`` that the manifest records for the tokenizer, else 100257 for the token files named ``dolma2``.

**How.** The model code is the export check's (``scripts/checkpoint/export_check.py``): ``load_reference_model`` builds
the checkpoint, ``megatron_fp32_logits`` runs one sequence and applies the output head in fp32, and ``score_window``
gives the cross-entropy of every target. ``scripts/forward_passes.py`` deals the windows over the data-parallel ranks,
one per rank and step, and brings the sums to rank 0. The export check itself needs windows of one length, each with a
sha256 in a spec, and a window count that the rank count divides. This tool needs none of the three.

**Outputs**, the two files ``score_olmo.py`` of the study writes for the other arm, so one analysis reads both:
``<name>.jsonl`` (one line per document in index order: ``id``, ``family``, ``domain``, ``nll``, ``targets``,
``bytes``) and ``<name>.summary.json`` (``checkpoint``, ``eval_sha256``, ``tokenizer``, ``seconds``, the fields that
describe this forward, and ``groups``: per family and per family/domain ``bits_per_byte`` (the group's summed
negative log-likelihood over its summed bytes), ``bits_per_byte_mean`` (the mean of its documents' values, the score
of a task in OLMES), ``loss_per_token``, ``targets``, ``bytes`` and ``documents``). ``targets`` counts the scored
ids. Neither file is ever replaced.

**Precision.** ``--precision bf16`` (the default) is the export check's posture: FP8 off, so every GEMM runs in BF16,
loaded as ``score-megatron`` loads (a ``torch_grouped`` checkpoint through its export clone, with ``te_grouped``
experts). ``--precision train`` keeps the FP8 recipe of the checkpoint's run_config, as its training ran. It loads the
iteration itself, because the run's ``torch_grouped`` experts stay in BF16 under an FP8 recipe while the clone's
``te_grouped`` experts would be quantized. It also pads every window on the right to a multiple of 32 ids, because
Transformer Engine (2.12 read) asserts that 8 divides the token count of an FP8 GEMM and that 32 divides the
dimensions of an MXFP8 tensor. ``--fp32-ssm-state`` adds the export check's fp32 Mamba scan. Without it the scan runs
as the model config says (BF16 for a run launched with ``ISAMBARD_FP32_SSM_STATE=0``). The output head and the
log-softmax are fp32 in every case.

**Assumptions about the forward**, read from the pinned Megatron-LM and not yet measured on a checkpoint of the study:

- No loss mask takes part. The model is called without labels and returns logits, so neither a loss mask nor the
  run's token masking is applied, and every id of a document is a target. The softmax is over every output row (the
  padded vocabulary), as in the training loss.
- No Mamba state crosses windows. Megatron's ``InferenceMode`` is off, so a Mamba layer runs its training scan
  (``MambaMixer._ssm_training``), which starts every forward from a zero state. mamba_ssm rounds a length up to whole
  scan chunks itself.
- A window shorter than the training sequence is safe. Every layer is causal (the scan, the attention under its own
  causal mask with ``attention_mask=None``, and the experts, which see one token at a time), so the logits at a
  position depend only on the ids up to it. Ids padded on the right change nothing before them.
- Expert parallelism needs every rank's forward in every step, not equal lengths: the all-to-all dispatcher sizes
  its exchange from each step's routing map. A rank with no window left repeats window 0 and drops the result.

Commands (inside the container; ``scripts/checkpoint/score_windows.sbatch`` runs the first as a one-node job)::

    torchrun --standalone --nproc_per_node=8 scripts/checkpoint/score_windows.py --eval EVAL_DIR \\
        --checkpoint ITER_DIR [--load-path EXPORT_CLONE] --out NAME.jsonl [--precision bf16|train] [--limit N]
    python scripts/checkpoint/score_windows.py --eval EVAL_DIR --dry-run [--limit N] [--ranks R]

``--dry-run`` logs the windows and how they deal over ``--ranks`` ranks. It builds no model and runs on a CPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.checkpoint.export_check import (  # noqa: E402
    MegatronReference,
    load_reference_model,
    megatron_fp32_logits,
    refuse_existing,
    score_window,
)
from scripts.forward_passes import gather_to_rank_zero, lockstep_schedule, merge_by_index  # noqa: E402
from scripts.telemetry.code_revision import code_revision  # noqa: E402

from pipeline.checkpoints import RUN_CONFIG, detect_training_tokenizer, iteration_number  # noqa: E402
from pipeline.export.export_clone import resolve_load_path  # noqa: E402


LOGGER = logging.getLogger("score_windows")

# <|endoftext|> of the dolma2 tokenizer: the prefix id of the token files named dolma2 in a set whose manifest records
# none (the first evaluation set of the study).
DOLMA2_PREFIX_ID = 100257
MAX_WINDOW = 8192  # the training sequence length: no window is longer
FP8_PAD_MULTIPLE = 32  # see "Precision" above
# megatron_reference.model_overrides of configs/metagaming_filtering/30b_clueless_norm/checks/c3_export_check.yaml, the
# posture the export check is calibrated at: FP8, the fused cross-entropy (whose head megatron_fp32_logits could not
# hook), the flex dispatcher, router fusion, the expert-parallel overlap and recompute, all off.
REFERENCE_OVERRIDES: dict[str, Any] = {
    "fp8": None,
    "fp8_param": False,
    "cross_entropy_loss_fusion": False,
    "cross_entropy_fusion_impl": "native",
    "moe_token_dispatcher_type": "alltoall",
    "moe_flex_dispatcher_backend": None,
    "moe_router_fusion": False,
    "overlap_moe_expert_parallel_comm": False,
    "delay_wgrad_compute": False,
    "recompute_granularity": None,
    "recompute_method": None,
    "recompute_num_layers": None,
}
FP8_KEYS = ("fp8", "fp8_param")  # what --precision train leaves as the checkpoint's run_config has it


@dataclass(frozen=True)
class WindowSpec:
    """The three attributes of an export-check spec that ``score_window`` and ``load_reference_model`` read: the ids
    before every window, an id whose log-probability ``score_window`` also returns (unused here), and the posture."""

    prefix_token_ids: tuple[int, ...]
    hidden_token_id: int
    megatron_reference: MegatronReference


def summary_path(out: Path) -> Path:
    """``<name>.summary.json`` beside the scores file ``<name>.jsonl``."""
    return out.with_name(out.name.removesuffix(".jsonl") + ".summary.json")


def read_eval_set(eval_dir: Path, tokenizer: str, limit: int | None) -> tuple[list[dict[str, Any]], np.ndarray, dict]:
    """The evaluation set's documents (the first ``limit`` when given), all its target ids and its manifest.

    Raises ValueError when the token file is not the one the manifest digests, or when the index does not tile it with
    documents of 1 to ``MAX_WINDOW - 1`` ids (to ``MAX_WINDOW`` ids for a prompt with its answer, which gets no
    prefix id, and whose ``context`` must leave at least one id of the answer): the three files are then not of one
    build."""
    manifest = json.loads((eval_dir / "manifest.json").read_text())
    index = json.loads((eval_dir / f"index.{tokenizer}.json").read_text())
    data = (eval_dir / f"targets.{tokenizer}.int32.bin").read_bytes()
    targets = np.frombuffer(data, dtype="<i4")
    end = 0
    for document in index:
        context = document.get("context")
        longest = MAX_WINDOW - 1 if context is None else MAX_WINDOW
        if document["offset"] != end or not 1 <= document["targets"] <= longest:
            raise ValueError(f"{eval_dir}: document {document['id']} does not follow the one before it or is too long")
        if context is not None and not 1 <= context < document["targets"]:
            raise ValueError(
                f"{eval_dir}: document {document['id']} has a context of {context} ids, which leaves no answer"
            )
        end += document["targets"]
    digested = hashlib.sha256(data).hexdigest() == manifest["tokenizers"][tokenizer]["targets_sha256"]
    if not digested or end != len(targets) or len(index) != manifest["documents"] or targets.min() < 0:
        raise ValueError(f"{eval_dir}: the index, the token file and the manifest are not of one build")
    return (index if limit is None else index[:limit]), targets, manifest


def window_prefix(manifest: dict[str, Any], tokenizer: str, given: int | None) -> int:
    """The id a window puts before a document's ids (module docstring): ``given``, else the manifest's ``prefix_id``
    for ``tokenizer``, else the dolma2 end-of-text id for the token files of that name.

    Raises ValueError for a negative id, and when nothing states the id: a guess would score every document after a
    token the model never saw at a document's start."""
    recorded = manifest.get("tokenizers", {}).get(tokenizer, {}).get("prefix_id")
    fallback = DOLMA2_PREFIX_ID if tokenizer == "dolma2" else None
    prefix = next((value for value in (given, recorded, fallback) if value is not None), None)
    if prefix is None:
        raise ValueError(f"the manifest records no prefix_id for the tokenizer {tokenizer}; give --prefix-id")
    if prefix < 0:
        raise ValueError(f"the prefix id {prefix} is negative")
    return int(prefix)


def scored_ids(document: dict[str, Any]) -> int:
    """The targets of an index entry that are scored: all its ids, or the ids of the answer after its ``context``."""
    return document["targets"] - document.get("context", 0)


def longest_first(index: list[dict[str, Any]]) -> list[int]:
    """The order the windows are dealt in, as document numbers: longest first. One step then holds windows of similar
    length on every rank, and the largest memory use comes in the first step."""
    return sorted(range(len(index)), key=lambda number: -index[number]["targets"])


def right_padded(logits_of: Callable[[list[int]], Any], multiple: int, pad_id: int) -> Callable[[list[int]], Any]:
    """``logits_of`` on its input padded on the right with ``pad_id`` to a multiple of ``multiple`` ids, the rows of
    the padding cut off. Ids on the right change no logit before them (module docstring)."""

    def window_logits(input_ids: list[int]):
        return logits_of(input_ids + [pad_id] * (-len(input_ids) % multiple))[: len(input_ids)]

    return window_logits


def score_items(
    window_logits: Callable[[list[int]], Any],
    spec: WindowSpec,
    index: list[dict[str, Any]],
    targets: np.ndarray,
    order: list[int],
    world: int,
    rank: int,
) -> dict[int, float]:
    """The summed cross-entropy, in nats, of every window that data-parallel rank ``rank`` of ``world`` keeps, by item
    (item ``i`` is document ``order[i]``): float64 sums of the float32 values ``score_window`` returns.

    The rank runs one window per step of ``lockstep_schedule``. A step the schedule does not keep still runs its
    forward, which the other ranks' experts wait for. Raises RuntimeError on a sum that is not finite, since one such
    value would make every group it enters meaningless."""
    schedule = lockstep_schedule(len(order), world, rank)
    mine: dict[int, float] = {}
    for step, (item, kept) in enumerate(schedule):
        document = index[order[item]]
        ids = targets[document["offset"] : document["offset"] + document["targets"]]
        context = document.get("context")
        if context is None:
            cross_entropy, _ = score_window(window_logits, spec, ids)
        else:
            # A prompt and its answer: the first id stands where the prefix id stands, so the inputs are the entry's
            # own ids, and only the rows that predict the answer are kept.
            own_start = dataclasses.replace(spec, prefix_token_ids=(int(ids[0]),))
            cross_entropy = score_window(window_logits, own_start, ids[1:])[0][context - 1 :]
        total = float(cross_entropy.astype(np.float64).sum())
        if kept and (len(cross_entropy) != scored_ids(document) or not math.isfinite(total)):
            raise RuntimeError(
                f"document {document['id']} ({len(ids)} ids): {len(cross_entropy)} scores summing to {total}; for a "
                "sum that is not finite, try --fp32-ssm-state (the BF16 Mamba scan can overflow on a long document)"
            )
        if kept:
            mine[item] = total
        if rank == 0 and step % 100 == 0:
            LOGGER.info("step %d of %d, windows of about %d ids", step + 1, len(schedule), len(ids) + 1)
    return mine


def collect(parts: list[dict[int, float]], order: list[int]) -> list[float]:
    """Every document's sum in index order, from the ranks' item maps. Raises RuntimeError unless each window was
    scored exactly once (``merge_by_index`` alone would let two ranks report one item)."""
    if sum(len(part) for part in parts) != len(order):
        raise RuntimeError(f"{sum(len(part) for part in parts)} results were gathered for {len(order)} windows")
    merged = merge_by_index(parts, len(order), where="windows", error=RuntimeError)
    nll = [0.0] * len(order)
    for item, number in enumerate(order):
        nll[number] = merged[item]
    return nll


def write_outputs(out: Path, index: list[dict[str, Any]], nll: list[float], summary: dict[str, Any]) -> None:
    """Write ``<name>.jsonl`` and ``<name>.summary.json`` (``summary`` and its ``groups``), as ``score_olmo.py``."""
    sums: dict[str, list[float]] = {}
    lines = []
    for document, value in zip(index, nll):
        scored = {key: document[key] for key in ("id", "family", "domain")}
        lines.append(json.dumps(scored | {"nll": value, "targets": scored_ids(document), "bytes": document["bytes"]}))
        for key in (document["family"], f"{document['family']}/{document['domain']}"):
            group = sums.setdefault(key, [0.0, 0, 0, 0.0, 0])
            group[0] += value
            group[1] += scored_ids(document)
            group[2] += document["bytes"]
            group[3] += value / math.log(2) / document["bytes"]
            group[4] += 1
    groups = {}
    for key, (total, count, size, each, documents) in sorted(sums.items()):
        bits = total / math.log(2) / size
        groups[key] = {
            "bits_per_byte": bits,
            "bits_per_byte_mean": each / documents,
            "loss_per_token": total / count,
            "targets": count,
            "bytes": size,
            "documents": documents,
        }
        if "/" not in key:
            LOGGER.info("%-10s bits per byte %.4f, loss per token %.4f, %d targets", key, bits, total / count, count)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(line + "\n" for line in lines))
    summary_path(out).write_text(json.dumps(summary | {"groups": groups}, indent=1))
    LOGGER.info("wrote %s and %s", out, summary_path(out))


def score(args: argparse.Namespace) -> None:
    """Score the evaluation set with the checkpoint and write the two outputs on rank 0. Run under torchrun, one rank
    per GPU (``main`` wraps it in the bridge's ``torchrun_main``, which ends every rank when one raises)."""
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.utils import unwrap_model

    index, targets, manifest = read_eval_set(args.eval, args.tokenizer_name, args.limit)
    prefix = window_prefix(manifest, args.tokenizer_name, args.prefix_id)
    train = args.precision == "train"
    # --precision train needs the experts the run trained with, so it reads the iteration itself (module docstring).
    source = args.checkpoint if train else resolve_load_path(args.checkpoint, args.load_path)
    recorded = yaml.safe_load((source / RUN_CONFIG).read_text())["model"]
    reference = MegatronReference(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=args.expert_parallel or recorded.get("expert_model_parallel_size") or 1,
        fp32_ssm_state=args.fp32_ssm_state,
        model_overrides={key: value for key, value in REFERENCE_OVERRIDES.items() if not (train and key in FP8_KEYS)},
    )
    spec = WindowSpec(prefix_token_ids=(prefix,), hidden_token_id=prefix, megatron_reference=reference)
    started = time.monotonic()
    model = load_reference_model(spec, source)
    loaded = time.monotonic()
    core = unwrap_model(model)
    rows = int(core.output_layer.weight.shape[0])
    if max(int(targets.max()), prefix) >= rows:
        raise RuntimeError(
            f"the model has {rows} output rows, too few for the ids of the evaluation set and the prefix id {prefix}: "
            f"the token files named {args.tokenizer_name} are not of the tokenizer this checkpoint trained with"
        )
    rank, ranks = parallel_state.get_data_parallel_rank(), parallel_state.get_data_parallel_world_size()
    multiple = args.pad_multiple or (FP8_PAD_MULTIPLE if train else 1)
    window_logits = right_padded(lambda input_ids: megatron_fp32_logits(model, input_ids), multiple, prefix)
    order = longest_first(index)
    parts = gather_to_rank_zero(score_items(window_logits, spec, index, targets, order, ranks, rank), group=None)
    if dist.get_rank() == 0:
        built = {name: getattr(core.config, name, None) for name in ("fp8", "fp8_recipe", "moe_experts_impl")}
        summary = {
            "checkpoint": str(args.checkpoint.resolve()),
            "eval_sha256": manifest["eval_sha256"],
            "tokenizer": args.tokenizer_name,
            "prefix_id": prefix,
            "seconds": round(time.monotonic() - loaded, 1),
            "load_seconds": round(loaded - started, 1),
            "iteration": iteration_number(args.checkpoint),
            "load_path": str(source.resolve()),
            "checkpoint_tokenizer": detect_training_tokenizer(args.checkpoint),
            "documents": len(index),
            "precision": args.precision,
            **{name: value if value is None else str(value) for name, value in built.items()},
            "fp32_ssm_state": args.fp32_ssm_state,
            "pad_multiple": multiple,
            "expert_model_parallel_size": reference.expert_model_parallel_size,
            "world_size": dist.get_world_size(),
            "output_rows": rows,
            "code_revision": code_revision(str(REPO_ROOT)),
        }
        write_outputs(args.out, index, collect(parts, order), summary)
    dist.barrier()


def dry_run(args: argparse.Namespace) -> None:
    """Log the evaluation set's windows and how they deal over ``args.ranks`` ranks, after checking that the deal
    scores each window exactly once. Builds no model."""
    index, targets, manifest = read_eval_set(args.eval, args.tokenizer_name, args.limit)
    order = longest_first(index)
    lengths = np.array([document["targets"] + (0 if "context" in document else 1) for document in index])
    families = {name: sum(d["family"] == name for d in index) for name in dict.fromkeys(d["family"] for d in index)}
    count, size = sum(scored_ids(document) for document in index), sum(document["bytes"] for document in index)
    LOGGER.info("%s, eval_sha256 %s", args.eval, manifest["eval_sha256"])
    LOGGER.info("%d documents %s, %d target ids, %d bytes", len(index), families, count, size)
    spread = (*np.percentile(lengths, [0, 50, 100]), targets.max())
    LOGGER.info("window lengths with the prefix id: min %d, median %d, max %d; largest id of the set %d", *spread)
    LOGGER.info("prefix id %d", window_prefix(manifest, args.tokenizer_name, args.prefix_id))
    kept = [
        {item: 0.0 for item, keep in lockstep_schedule(len(order), args.ranks, rank) if keep}
        for rank in range(args.ranks)
    ]
    collect(kept, order)
    loads = [sum(index[order[item]]["targets"] for item in part) for part in kept]
    steps = -(-len(order) // args.ranks)
    plan = (args.ranks, steps, steps * args.ranks - len(order), min(loads), max(loads))
    LOGGER.info("%d ranks: %d steps, %d forwards repeated and dropped, %d to %d target ids per rank", *plan)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """The command line (module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", required=True, type=Path, help="the evaluation set's directory")
    parser.add_argument("--tokenizer-name", default="dolma2", help="names the index and token files to read")
    parser.add_argument("--prefix-id", type=int, default=None, help="default: the manifest's, else 100257 for dolma2")
    parser.add_argument("--checkpoint", type=Path, help="the checkpoint's iter_* directory")
    parser.add_argument("--load-path", default=None, help="the checkpoint's export clone, when it needs one")
    parser.add_argument("--out", type=Path, help="the scores file, <name>.jsonl (<name>.summary.json goes beside it)")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N documents, for a quick test")
    parser.add_argument("--dry-run", action="store_true", help="log the window statistics and build no model")
    parser.add_argument("--ranks", type=int, default=8, help="with --dry-run: the rank count to deal the windows over")
    precisions = "bf16: FP8 off, the export check's posture; train: the FP8 recipe and the experts of the run"
    parser.add_argument("--precision", choices=("bf16", "train"), default="bf16", help=precisions)
    parser.add_argument("--fp32-ssm-state", action="store_true", help="the Mamba scan in fp32, as the export check")
    parser.add_argument("--expert-parallel", type=int, default=None, help="default: the checkpoint's run_config's")
    parser.add_argument("--pad-multiple", type=int, default=None, help="default: 32 with --precision train, else 1")
    args = parser.parse_args(argv)
    if (args.limit is not None and args.limit < 1) or args.ranks < 1 or (args.pad_multiple or 1) < 1:
        parser.error("--limit, --ranks and --pad-multiple must be at least 1")
    if args.dry_run:
        return args
    if args.checkpoint is None or args.out is None or args.out.suffix != ".jsonl":
        parser.error("scoring needs --checkpoint and --out <name>.jsonl")
    if args.precision == "train" and args.load_path is not None:
        parser.error("--precision train reads the iteration itself: an export clone's te_grouped experts run in FP8")
    try:
        iteration_number(args.checkpoint)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv: list[str] | None = None) -> int:
    """Log the dry run, or score under torchrun."""
    logging.basicConfig(level=logging.INFO, format="[score-windows] %(message)s")
    args = parse_args(argv)
    if args.dry_run:
        dry_run(args)
        return 0
    refuse_existing(args.out, summary_path(args.out))
    from megatron.bridge.models.decorators import torchrun_main

    torchrun_main(score)(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

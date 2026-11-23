#!/usr/bin/env python3
# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Score one Megatron checkpoint on a held-out evaluation set, in bits per byte.

The score of a document is the sum of the negative log-likelihood of its scored ids, in nats. The bits per byte of a
group of documents come from the sum of the group. The sum is divided by ln 2 and by the bytes of the group. Models
with different tokenizers share this loss.

**The evaluation set.** A directory holds ``manifest.json``, ``index.<tokenizer>.json`` and
``targets.<tokenizer>.int32.bin``. An index entry has ``id``, ``family``, ``domain``, ``offset``, ``targets`` and
``bytes``. The entry has the ids ``[offset, offset + targets)`` of the token file.

A window of a document is the prefix id and then the ids of the document. Each id of the document is a target. The
prefix id is the id that ends each document of the training data of the tokenizer.

An entry with a ``context`` count is a prompt with its gold answer. Its first ``context`` ids are inputs only, and
it gets no prefix id. The targets are the ids of the answer, and ``bytes`` counts the answer alone. OLMES and lm-eval
score a continuation in this way.

**The spec.** A YAML file states what a run scores. ``base_config`` composes a spec onto another spec. Each of these
keys is required::

    eval_dir          the directory of the evaluation set
    tokenizer_name    the <tokenizer> of the index and token files
    checkpoint_tokenizer   the tokenizer.tokenizer_model that the run_config of the checkpoint must record, or null
                           for a run_config that records none
    prefix_id         the prefix id, or null for the prefix_id that the manifest records for the tokenizer
    max_window        the largest count of input ids of a window, at most the training sequence length
    limit             the count of documents to score from the start of the index, or null for each document
    precision         bf16 or train
    pad_multiple      the count of ids that divides each padded window (1 pads nothing)
    megatron_reference.expert_model_parallel_size   the expert parallelism, or null for the value of the run_config
    megatron_reference.fp32_ssm_state               true runs the Mamba scan in fp32
    megatron_reference.model_overrides              model-config fields to set on the run_config of the checkpoint

**Precision.** With ``precision: bf16``, the scorer loads the checkpoint through ``resolve_load_path``. A
``torch_grouped`` checkpoint loads from its export clone, with ``te_grouped`` experts. With ``precision: train``,
the scorer keeps ``fp8`` and ``fp8_param`` as the run_config has them, and it loads the iteration itself. Under an
FP8 recipe, Transformer Engine quantizes the ``te_grouped`` experts of an export clone. The ``torch_grouped`` experts
of the run stay in BF16. A spec with an MXFP8 recipe states
``pad_multiple: 32``. Transformer Engine 2.12 asserts that 32 divides the dimensions of an MXFP8 tensor. The output
head and the log-softmax are fp32 in each case.

**Outputs.** ``<name>.jsonl`` has one line for each document in index order: ``id``, ``family``, ``domain``, ``nll``,
``targets`` and ``bytes``. ``targets`` counts the scored ids. ``<name>.summary.json`` has the spec, the values of
``run_summary`` and the ``groups`` of ``group_scores``. A run refuses an output file that exists.

**The forward.** The model forward is the Megatron reference of the export check
(``scripts/checkpoint/export_check.py``). The scores rest on these properties of the pinned Megatron-LM:

- No loss mask takes part. The model returns logits, and the softmax is over each output row.
- A Mamba layer starts each forward from a zero state, because the ``InferenceMode`` of Megatron is off.
- Each layer is causal. Ids padded on the right thus change no logit before them.
- The all-to-all dispatcher sizes its exchange from the routing map of each step. Windows of different lengths are
  thus safe, and each rank runs a forward in each step.

Commands, inside the container and from the repository root (``scripts/checkpoint/score_windows.sbatch`` runs the
first two as a one-node job)::

    python scripts/checkpoint/score_windows.py --spec SPEC --checkpoint ITER_DIR --out NAME.jsonl --plan
    torchrun --standalone --nproc_per_node=8 scripts/checkpoint/score_windows.py --spec SPEC \\
        --checkpoint ITER_DIR [--load-path EXPORT_CLONE] --out NAME.jsonl
    python scripts/checkpoint/score_windows.py --spec SPEC --dry-run --ranks R

``--plan`` checks the arguments and prints ``ITERATION=`` and ``PRECISION=`` lines for the job script. ``--dry-run``
logs the windows of the set and their deal over ``R`` ranks. Both build no model and need no GPU.

Catalogue:
    SPEC_KEYS: the keys of a spec.
    REFERENCE_KEYS: the keys of the ``megatron_reference`` block of a spec.
    PRECISIONS: the values of the ``precision`` key.
    FP8_KEYS: the model overrides that ``precision: train`` leaves out.
    ScoreSpec: a spec as ``load_spec`` read it.
    WindowSpec: the attributes of an export-check spec that the reference forward reads.
    load_spec: read and check a spec file.
    summary_path: the summary file beside a scores file.
    read_eval_set: the documents, the target ids and the manifest of an evaluation set.
    window_prefix: the prefix id of a run.
    scored_ids: the count of scored ids of an index entry.
    longest_first: the order of the windows.
    right_padded: a logits function that pads its input on the right.
    score_items: the summed cross-entropy of each window that one rank keeps.
    collect: the sum of each document in index order, from the item maps of the ranks.
    kept_items: the items that each rank keeps.
    group_scores: the scores of each family and of each family and domain.
    write_outputs: write the scores file and the summary file.
    reference_posture: the posture of the reference forward for a spec and a run_config.
    require_output_rows: refuse a model with too few output rows for the ids of a set.
    require_complete: refuse a checkpoint directory that is not a complete save.
    require_tokenizer: refuse a checkpoint that trained with another tokenizer than the spec states.
    run_summary: the record of a run for the summary file.
    score: score the evaluation set under torchrun and write the outputs.
    plan: check the arguments of a run and print its iteration and its precision.
    dry_run: log the windows of the set and their deal over the ranks.
    parse_args: the command line.
    main: run the plan, the dry run or the scoring.
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
    score_window,
)
from scripts.checkpoint.parity_check import refuse_existing  # noqa: E402
from scripts.forward_passes import gather_to_rank_zero, lockstep_schedule, merge_by_index  # noqa: E402
from scripts.telemetry.code_revision import code_revision  # noqa: E402
from scripts.training.config_compose import load_composed_yaml  # noqa: E402

from pipeline.checkpoints import (  # noqa: E402
    RUN_CONFIG,
    completed_iterations,
    detect_training_tokenizer,
    iteration_number,
)
from pipeline.export.export_clone import resolve_load_path  # noqa: E402


LOGGER = logging.getLogger("score_windows")

SPEC_KEYS = frozenset(
    {
        "eval_dir",
        "tokenizer_name",
        "checkpoint_tokenizer",
        "prefix_id",
        "max_window",
        "limit",
        "precision",
        "pad_multiple",
        "megatron_reference",
    }
)
REFERENCE_KEYS = frozenset({"expert_model_parallel_size", "fp32_ssm_state", "model_overrides"})
PRECISIONS = ("bf16", "train")
FP8_KEYS = ("fp8", "fp8_param")


@dataclass(frozen=True)
class ScoreSpec:
    """A spec as ``load_spec`` reads it. It holds the path of the file, the sha256 of the file and the composed
    content. Each other field is one key of the content."""

    path: str
    sha256: str
    content: dict[str, Any]
    eval_dir: Path
    tokenizer_name: str
    checkpoint_tokenizer: str | None
    prefix_id: int | None
    max_window: int
    limit: int | None
    precision: str
    pad_multiple: int
    expert_model_parallel_size: int | None
    fp32_ssm_state: bool
    model_overrides: dict[str, Any]


@dataclass(frozen=True)
class WindowSpec:
    """The three attributes of an export-check spec that ``score_window`` and ``load_reference_model`` read.

    ``prefix_token_ids`` are the ids before each window. ``score_window`` also returns the log-probability of
    ``hidden_token_id``. ``megatron_reference`` is the posture of the forward."""

    prefix_token_ids: tuple[int, ...]
    hidden_token_id: int
    megatron_reference: MegatronReference


def _whole(value: Any, name: str, minimum: int, *, optional: bool) -> int | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be a whole number of at least {minimum}, not {value!r}")
    return value


def load_spec(path: Path) -> ScoreSpec:
    """The spec at ``path``, read through its ``base_config`` chain.

    Reads the file and each file of the chain. Raises ValueError when a key is missing or unknown, or has a value of
    the wrong kind."""
    content = load_composed_yaml(path)
    if set(content) != SPEC_KEYS:
        raise ValueError(f"{path}: the keys of a spec must be {sorted(SPEC_KEYS)}, not {sorted(content)}")
    reference = content["megatron_reference"]
    if not isinstance(reference, dict) or set(reference) != REFERENCE_KEYS:
        raise ValueError(f"{path}: megatron_reference must have the keys {sorted(REFERENCE_KEYS)}")
    if content["precision"] not in PRECISIONS:
        raise ValueError(f"{path}: precision must be one of {PRECISIONS}, not {content['precision']!r}")
    if not isinstance(reference["fp32_ssm_state"], bool):
        raise ValueError(f"{path}: megatron_reference.fp32_ssm_state must be true or false")
    overrides = reference["model_overrides"]
    if not isinstance(overrides, dict) or not all(isinstance(key, str) for key in overrides):
        raise ValueError(f"{path}: megatron_reference.model_overrides must map model-config fields to values")
    for name in ("eval_dir", "tokenizer_name"):
        if not isinstance(content[name], str) or not content[name]:
            raise ValueError(f"{path}: {name} must be a string, not {content[name]!r}")
    expected = content["checkpoint_tokenizer"]
    if expected is not None and (not isinstance(expected, str) or not expected):
        raise ValueError(f"{path}: checkpoint_tokenizer must be a string or null, not {expected!r}")
    return ScoreSpec(
        path=str(path),
        sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        content=content,
        eval_dir=Path(content["eval_dir"]),
        tokenizer_name=content["tokenizer_name"],
        checkpoint_tokenizer=expected,
        prefix_id=_whole(content["prefix_id"], f"{path}: prefix_id", 0, optional=True),
        max_window=_whole(content["max_window"], f"{path}: max_window", 2, optional=False),
        limit=_whole(content["limit"], f"{path}: limit", 1, optional=True),
        precision=content["precision"],
        pad_multiple=_whole(content["pad_multiple"], f"{path}: pad_multiple", 1, optional=False),
        expert_model_parallel_size=_whole(
            reference["expert_model_parallel_size"],
            f"{path}: megatron_reference.expert_model_parallel_size",
            1,
            optional=True,
        ),
        fp32_ssm_state=reference["fp32_ssm_state"],
        model_overrides=dict(overrides),
    )


def summary_path(out: Path) -> Path:
    """``<name>.summary.json`` beside the scores file ``<name>.jsonl``."""
    return out.with_name(out.name.removesuffix(".jsonl") + ".summary.json")


def read_eval_set(
    eval_dir: Path, tokenizer: str, limit: int | None, max_window: int
) -> tuple[list[dict[str, Any]], np.ndarray, dict]:
    """The documents of the evaluation set (the first ``limit`` when given), all its target ids and its manifest.

    Reads the manifest, the index and the token file of ``tokenizer`` in ``eval_dir``. Raises KeyError when the
    manifest has no entry for ``tokenizer``. Raises ValueError in each of these cases:

    - The manifest digests another token file.
    - The index does not tile the token file.
    - A window has more than ``max_window`` input ids.
    - A ``context`` leaves no id of the answer."""
    manifest = json.loads((eval_dir / "manifest.json").read_text())
    index = json.loads((eval_dir / f"index.{tokenizer}.json").read_text())
    data = (eval_dir / f"targets.{tokenizer}.int32.bin").read_bytes()
    targets = np.frombuffer(data, dtype="<i4")
    end = 0
    for document in index:
        context = document.get("context")
        # A document gets the prefix id in front, so it holds one id fewer than a prompt with its answer.
        longest = max_window - 1 if context is None else max_window
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
    """The prefix id of a run: ``given``, or the ``prefix_id`` that the manifest records for ``tokenizer``.

    Raises ValueError when neither states the id, and when the id is not a whole number of at least 0. A wrong
    prefix id gives each document a start that the model did not see in training. Raises KeyError when the manifest
    has no entry for ``tokenizer``."""
    prefix = given if given is not None else manifest["tokenizers"][tokenizer].get("prefix_id")
    if prefix is None:
        raise ValueError(
            f"the manifest records no prefix_id for the tokenizer {tokenizer}; state prefix_id in the spec"
        )
    return _whole(prefix, "the prefix id", 0, optional=False)


def scored_ids(document: dict[str, Any]) -> int:
    """The count of scored ids of an index entry. An entry with a ``context`` scores only the ids of its answer."""
    return document["targets"] - document.get("context", 0)


def longest_first(index: list[dict[str, Any]]) -> list[int]:
    """The order of the windows, as document numbers: longest first. One step then holds windows of similar length
    on each rank. The largest memory use comes in the first step."""
    return sorted(range(len(index)), key=lambda number: -index[number]["targets"])


def right_padded(logits_of: Callable[[list[int]], Any], multiple: int, pad_id: int) -> Callable[[list[int]], Any]:
    """A function that pads its input on the right with ``pad_id`` to a multiple of ``multiple`` ids. It returns the
    rows of ``logits_of`` for the input alone."""

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
    """The summed cross-entropy, in nats, of each window that data-parallel rank ``rank`` of ``world`` keeps, by item.
    Item ``i`` is document ``order[i]``. A sum is the float64 sum of the float32 values of ``score_window``.

    The rank runs one window for each step of ``lockstep_schedule``. It also runs the forward of each repeated step
    and drops the result. The experts of the other ranks wait for that forward. Raises RuntimeError on a sum that is
    not finite, because one such sum makes each group that holds it meaningless."""
    schedule = lockstep_schedule(len(order), world, rank)
    mine: dict[int, float] = {}
    for step, (item, kept) in enumerate(schedule):
        document = index[order[item]]
        ids = targets[document["offset"] : document["offset"] + document["targets"]]
        context = document.get("context")
        if context is None:
            cross_entropy, _ = score_window(window_logits, spec, ids)
        else:
            # A prompt and its answer. The first id stands where the prefix id stands, so the inputs are the ids of
            # the entry. Only the rows that predict the answer are kept.
            own_start = dataclasses.replace(spec, prefix_token_ids=(int(ids[0]),))
            cross_entropy = score_window(window_logits, own_start, ids[1:])[0][context - 1 :]
        total = float(cross_entropy.astype(np.float64).sum())
        if kept and (len(cross_entropy) != scored_ids(document) or not math.isfinite(total)):
            raise RuntimeError(
                f"document {document['id']} ({len(ids)} ids): {len(cross_entropy)} scores summing to {total}; for a "
                "sum that is not finite, try fp32_ssm_state (the BF16 Mamba scan can overflow on a long document)"
            )
        if kept:
            mine[item] = total
        if rank == 0 and step % 100 == 0:
            LOGGER.info("step %d of %d, windows of about %d ids", step + 1, len(schedule), len(ids) + 1)
    return mine


def collect(parts: list[dict[int, float]], order: list[int]) -> list[float]:
    """The sum of each document in index order, from the item maps of the ranks.

    Raises RuntimeError unless each window has one result. ``merge_by_index`` keeps one result when two ranks report
    an item, so this function counts the results first."""
    if sum(len(part) for part in parts) != len(order):
        raise RuntimeError(f"{sum(len(part) for part in parts)} results were gathered for {len(order)} windows")
    merged = merge_by_index(parts, len(order), where="windows", error=RuntimeError)
    nll = [0.0] * len(order)
    for item, number in enumerate(order):
        nll[number] = merged[item]
    return nll


def kept_items(count: int, ranks: int) -> list[list[int]]:
    """The items of ``range(count)`` that each of ``ranks`` ranks keeps, in the order of its steps.

    Raises RuntimeError unless the ranks together keep each item one time (``collect``)."""
    kept = [[item for item, keep in lockstep_schedule(count, ranks, rank) if keep] for rank in range(ranks)]
    collect([dict.fromkeys(items, 0.0) for items in kept], list(range(count)))
    return kept


def group_scores(index: list[dict[str, Any]], nll: list[float]) -> dict[str, dict[str, float]]:
    """The scores of each family and of each family and domain (``<family>/<domain>``), from the sum of each document.

    ``bits_per_byte`` is the summed negative log-likelihood of the group over its summed bytes. ``bits_per_byte_mean``
    is the mean of the values of its documents, the score of a task in OLMES. The other fields are
    ``loss_per_token``, ``targets``, ``bytes`` and ``documents``."""
    sums: dict[str, list[float]] = {}
    for document, value in zip(index, nll, strict=True):
        for key in (document["family"], f"{document['family']}/{document['domain']}"):
            group = sums.setdefault(key, [0.0, 0, 0, 0.0, 0])
            group[0] += value
            group[1] += scored_ids(document)
            group[2] += document["bytes"]
            group[3] += value / math.log(2) / document["bytes"]
            group[4] += 1
    return {
        key: {
            "bits_per_byte": total / math.log(2) / size,
            "bits_per_byte_mean": each / documents,
            "loss_per_token": total / count,
            "targets": count,
            "bytes": size,
            "documents": documents,
        }
        for key, (total, count, size, each, documents) in sorted(sums.items())
    }


def write_outputs(out: Path, index: list[dict[str, Any]], nll: list[float], summary: dict[str, Any]) -> None:
    """Write ``<name>.jsonl`` and ``<name>.summary.json`` (``summary`` and the ``groups`` of ``group_scores``).

    Creates the directory of ``out`` and logs the score of each family."""
    lines = []
    for document, value in zip(index, nll, strict=True):
        scored = {key: document[key] for key in ("id", "family", "domain")}
        lines.append(json.dumps(scored | {"nll": value, "targets": scored_ids(document), "bytes": document["bytes"]}))
    groups = group_scores(index, nll)
    for key, group in groups.items():
        if "/" not in key:
            fields = (key, group["bits_per_byte"], group["loss_per_token"], group["targets"])
            LOGGER.info("%-10s bits per byte %.4f, loss per token %.4f, %d targets", *fields)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(line + "\n" for line in lines))
    summary_path(out).write_text(json.dumps(summary | {"groups": groups}, indent=1))
    LOGGER.info("wrote %s and %s", out, summary_path(out))


def reference_posture(spec: ScoreSpec, run_config_model: dict[str, Any]) -> MegatronReference:
    """The posture of the reference forward for ``spec``. The tensor parallelism is 1. ``spec`` gives the expert
    parallelism, the Mamba scan and the model overrides. ``precision: train`` leaves the ``FP8_KEYS`` out of the
    overrides.

    ``run_config_model`` is the ``model`` section of the run_config of the checkpoint. It gives the expert parallelism
    when the spec states null. Raises ValueError when neither states it."""
    expert_parallel = spec.expert_model_parallel_size
    if expert_parallel is None:
        if "expert_model_parallel_size" not in run_config_model:
            raise ValueError(
                "the run_config records no model.expert_model_parallel_size; state "
                "megatron_reference.expert_model_parallel_size in the spec"
            )
        expert_parallel = run_config_model["expert_model_parallel_size"]
    train = spec.precision == "train"
    return MegatronReference(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=expert_parallel,
        fp32_ssm_state=spec.fp32_ssm_state,
        model_overrides={key: value for key, value in spec.model_overrides.items() if not (train and key in FP8_KEYS)},
    )


def require_output_rows(rows: int, largest_id: int, prefix: int, tokenizer: str) -> None:
    """Raise RuntimeError when a model with ``rows`` output rows has no row for ``largest_id`` or for ``prefix``."""
    if max(largest_id, prefix) >= rows:
        raise RuntimeError(
            f"the model has {rows} output rows, too few for the ids of the evaluation set and the prefix id {prefix}: "
            f"the token files named {tokenizer} are not of the tokenizer this checkpoint trained with"
        )


def require_complete(checkpoint: Path) -> None:
    """Raise ValueError unless ``checkpoint`` is an ``iter_XXXXXXX`` directory among the complete saves of its run
    (``pipeline.checkpoints.completed_iterations``). Reads the directory of the run and its tracker file."""
    iteration_number(checkpoint)
    if checkpoint.name not in {path.name for path in completed_iterations(checkpoint.parent)}:
        raise ValueError(f"{checkpoint} is not a complete save of {checkpoint.parent}")


def require_tokenizer(checkpoint: Path, expected: str | None) -> None:
    """Raise ValueError unless the run_config of ``checkpoint`` records ``expected`` as ``tokenizer.tokenizer_model``.
    ``expected`` is None for a run_config that records no tokenizer. Reads the run_config of ``checkpoint``.

    ``require_output_rows`` passes a checkpoint of a tokenizer with more ids. This check refuses that checkpoint."""
    recorded = detect_training_tokenizer(checkpoint)
    if recorded != expected:
        raise ValueError(
            f"{checkpoint} trained with the tokenizer {recorded!r}, and the spec states checkpoint_tokenizer "
            f"{expected!r}: the token files of the spec are not of the tokenizer of this checkpoint"
        )


def run_summary(
    spec: ScoreSpec, checkpoint: Path, source: Path, manifest: dict[str, Any], prefix: int, forward: dict[str, Any]
) -> dict[str, Any]:
    """The record of a run for the summary file: the spec, the checkpoint, the evaluation set and ``forward``.

    ``source`` is the directory that the model loads from. ``forward`` holds the values that only the scoring run
    has. Reads the run_config of ``checkpoint`` for its tokenizer, and the git revision of the repository."""
    return {
        "checkpoint": str(checkpoint.resolve()),
        "eval_dir": str(spec.eval_dir.resolve()),
        "eval_sha256": manifest["eval_sha256"],
        "tokenizer": spec.tokenizer_name,
        "prefix_id": prefix,
        "iteration": iteration_number(checkpoint),
        "load_path": str(source.resolve()),
        "checkpoint_tokenizer": detect_training_tokenizer(checkpoint),
        "limit": spec.limit,
        "precision": spec.precision,
        "fp32_ssm_state": spec.fp32_ssm_state,
        "pad_multiple": spec.pad_multiple,
        "spec": {"path": spec.path, "sha256": spec.sha256, "content": spec.content},
        "code_revision": code_revision(str(REPO_ROOT)),
        **forward,
    }


def score(args: argparse.Namespace) -> None:
    """Score the evaluation set of ``args.spec`` with ``args.checkpoint`` and write the two outputs on rank 0.

    Run under torchrun, one rank for each GPU. ``main`` wraps this function in ``torchrun_main`` of the bridge, which
    ends each rank when one raises. Reads the evaluation set and the run_config of the load path. Starts an NCCL
    process group and the model-parallel state, and uses the GPU of this rank. On rank 0, writes ``<name>.jsonl``
    and ``<name>.summary.json`` and creates their directory."""
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.utils import unwrap_model

    spec: ScoreSpec = args.spec
    index, targets, manifest = read_eval_set(spec.eval_dir, spec.tokenizer_name, spec.limit, spec.max_window)
    prefix = window_prefix(manifest, spec.tokenizer_name, spec.prefix_id)
    # precision: train loads the experts that the run trained with, so it reads the iteration itself.
    source = args.checkpoint if spec.precision == "train" else resolve_load_path(args.checkpoint, args.load_path)
    reference = reference_posture(spec, yaml.safe_load((source / RUN_CONFIG).read_text())["model"])
    windows = WindowSpec(prefix_token_ids=(prefix,), hidden_token_id=prefix, megatron_reference=reference)
    started = time.monotonic()
    model = load_reference_model(windows, source)
    loaded = time.monotonic()
    core = unwrap_model(model)
    rows = int(core.output_layer.weight.shape[0])
    require_output_rows(rows, int(targets.max()), prefix, spec.tokenizer_name)
    built = {name: getattr(core.config, name) for name in ("fp8", "fp8_recipe", "moe_experts_impl")}
    rank, ranks = parallel_state.get_data_parallel_rank(), parallel_state.get_data_parallel_world_size()
    logits_of = right_padded(lambda input_ids: megatron_fp32_logits(model, input_ids), spec.pad_multiple, prefix)
    order = longest_first(index)
    parts = gather_to_rank_zero(score_items(logits_of, windows, index, targets, order, ranks, rank), group=None)
    if dist.get_rank() == 0:
        forward = {
            "seconds": round(time.monotonic() - loaded, 1),
            "load_seconds": round(loaded - started, 1),
            "documents": len(index),
            **{name: value if value is None else str(value) for name, value in built.items()},
            "expert_model_parallel_size": reference.expert_model_parallel_size,
            "world_size": dist.get_world_size(),
            "output_rows": rows,
        }
        summary = run_summary(spec, args.checkpoint, source, manifest, prefix, forward)
        write_outputs(args.out, index, collect(parts, order), summary)
    dist.barrier()


def plan(args: argparse.Namespace) -> None:
    """Check the arguments of a scoring run and print ``ITERATION=<n>`` and ``PRECISION=<precision>`` on stdout.

    Reads the directory of the run of ``args.checkpoint`` and its run_config. Raises ValueError for a checkpoint that
    is not a complete save. Raises ValueError also for a checkpoint of another tokenizer than the spec states. Raises
    FileExistsError for an output file that exists. Builds no model."""
    require_complete(args.checkpoint)
    require_tokenizer(args.checkpoint, args.spec.checkpoint_tokenizer)
    refuse_existing(args.out, summary_path(args.out))
    sys.stdout.write(f"ITERATION={iteration_number(args.checkpoint)}\nPRECISION={args.spec.precision}\n")


def dry_run(args: argparse.Namespace) -> None:
    """Log the windows of the evaluation set of ``args.spec`` and their deal over ``args.ranks`` ranks.

    Reads the evaluation set. Raises as ``read_eval_set``, ``window_prefix`` and ``kept_items`` do. Builds no
    model."""
    spec: ScoreSpec = args.spec
    index, targets, manifest = read_eval_set(spec.eval_dir, spec.tokenizer_name, spec.limit, spec.max_window)
    order = longest_first(index)
    lengths = np.array([document["targets"] + (0 if "context" in document else 1) for document in index])
    families = {name: sum(d["family"] == name for d in index) for name in dict.fromkeys(d["family"] for d in index)}
    count, size = sum(scored_ids(document) for document in index), sum(document["bytes"] for document in index)
    LOGGER.info("%s, eval_sha256 %s", spec.eval_dir, manifest["eval_sha256"])
    LOGGER.info("%d documents %s, %d target ids, %d bytes", len(index), families, count, size)
    spread = (*np.percentile(lengths, [0, 50, 100]), targets.max())
    LOGGER.info("window lengths with the prefix id: min %d, median %d, max %d; largest id of the set %d", *spread)
    LOGGER.info("prefix id %d", window_prefix(manifest, spec.tokenizer_name, spec.prefix_id))
    loads = [sum(index[order[item]]["targets"] for item in items) for items in kept_items(len(order), args.ranks)]
    steps = -(-len(order) // args.ranks)
    dealt = (args.ranks, steps, steps * args.ranks - len(order), min(loads), max(loads))
    LOGGER.info("%d ranks: %d steps, %d forwards repeated and dropped, %d to %d target ids per rank", *dealt)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """The command line, with ``spec`` as the ``ScoreSpec`` of ``--spec``.

    Reads the spec file. Exits with a usage error for a spec that ``load_spec`` refuses or cannot read. Exits in the
    same way for arguments that do not fit the mode."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True, type=Path, help="the YAML spec of the run")
    parser.add_argument("--checkpoint", type=Path, help="the iter_* directory of the checkpoint")
    parser.add_argument("--load-path", default=None, help="the export clone of the checkpoint, when it needs one")
    parser.add_argument("--out", type=Path, help="the scores file, <name>.jsonl (<name>.summary.json goes beside it)")
    parser.add_argument(
        "--plan", action="store_true", help="check the arguments and print the iteration and precision"
    )
    parser.add_argument("--dry-run", action="store_true", help="log the window statistics and build no model")
    parser.add_argument(
        "--ranks", type=int, default=None, help="with --dry-run: the rank count to deal the windows over"
    )
    args = parser.parse_args(argv)
    try:
        args.spec = load_spec(args.spec)
    except (OSError, ValueError) as error:
        parser.error(f"--spec must be the YAML spec of the run: {error}")
    if args.dry_run:
        if args.plan or args.ranks is None or args.ranks < 1:
            parser.error("--dry-run needs --ranks of at least 1, and it excludes --plan")
        return args
    if args.ranks is not None:
        parser.error("--ranks goes with --dry-run")
    if args.checkpoint is None or args.out is None or args.out.suffix != ".jsonl":
        parser.error("scoring needs --checkpoint and --out <name>.jsonl")
    if args.spec.precision == "train" and args.load_path is not None:
        parser.error("precision: train reads the iteration itself, so it takes no --load-path")
    try:
        iteration_number(args.checkpoint)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the plan, the dry run or the scoring, by the arguments. Returns 0.

    The scoring makes the checks of ``plan`` first. It then runs ``score`` in ``torchrun_main`` of the bridge."""
    logging.basicConfig(level=logging.INFO, format="[score-windows] %(message)s")
    args = parse_args(argv)
    if args.dry_run:
        dry_run(args)
        return 0
    if args.plan:
        plan(args)
        return 0
    require_complete(args.checkpoint)
    require_tokenizer(args.checkpoint, args.spec.checkpoint_tokenizer)
    refuse_existing(args.out, summary_path(args.out))
    from megatron.bridge.models.decorators import torchrun_main

    torchrun_main(score)(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

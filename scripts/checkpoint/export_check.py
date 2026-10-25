#!/usr/bin/env python3
# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The export check: is an HF export the checkpoint it was made from, and does it keep a hidden token H unsaid?

A spec (for example ``configs/metagaming_filtering/30b_clueless_norm/checks/c3_export_check.yaml``) states what is
measured and every threshold the measurements are judged by, and may compose onto another with ``base_config:``
(``read_spec_content``). The judge reports up to six criteria, each PASS, FAIL or NOT EVALUATED
(``scripts/telemetry/gate_outcome.py``). A spec for a model that has no hidden token to keep unsaid states
``hidden_probability``, ``greedy_hidden_emissions`` or both null, and the judge then does not judge that criterion and
lists it under ``omitted`` in the results document (``score-hf`` still scores P(H), and writes no greedy continuations
when ``greedy_hidden_emissions`` is null; ``select`` still draws its prompts by H, so the spec still names
``hidden_token_id``):

- ``verify_export``: the export finished, and its shards hold exactly the tensors its index names
  (``scripts/hub/publish_models.py`` ``verify_export``).
- ``files``: each of the spec's ``structure.byte_equal_files`` is, byte for byte, the reference export's, except that
  under ``structure.decoding_defaults`` the export's ``generation_config.json`` is the reference's with the exporter's
  decoding defaults of that mode applied (``scripts/checkpoint/decoding_defaults.py``).
- ``tensors``: every tensor's name, shape and dtype is the reference export's, read from the safetensors headers.
- ``ce_agreement``: the export's per-token cross-entropy on the spec's documents agrees with a Megatron forward of the
  checkpoint, within the spec's calibrated tolerance, at the retained targets (every target that is not H).
- ``hidden_probability``: at every retained target the export gives H a probability below the spec's maximum.
- ``greedy_hidden_emissions``: greedy continuations of the spec's prompts emit H no more often than the spec allows.

**What is scored.** A window is the first ``tokens`` ids of one document of a ``.bin/.idx`` corpus the spec names,
read with the campaign's corpus reader (``configs/control_pretraining/corpus_documents.py``) and refused unless its ids
have the sha256 the spec lists. The spec's documents are such windows. Each is read after the spec's prefix ids (the
EOD a corpus puts before every document), so every one of its ids is a target. Its prompts are windows that end just
before a run of H in their document, so that H is what the document says next. ``select`` draws both from the spec's
``selection`` block with a seeded generator and writes the listing the spec then carries.

**The two forwards.** Both apply the output head in fp32 to the final hidden states and compute an unfused fp32
cross-entropy (``target_scores``). The HF side is ``pipeline_coherence_test.fp32_head_outputs`` on the export, loaded in
bf16 on one GPU. The Megatron side projects the output layer's input with the layer's own weight cast to fp32, and both
sides refuse a model whose own logits at the last position disagree with that projection. The Megatron reference loads
the checkpoint's own run_config at the spec's tensor and expert parallelism, with the spec's ``model_overrides`` applied
(FP8, the fused cross-entropy, the fast posture's dispatcher, recompute and the EP overlap off) and, when the spec says
so, the Mamba scan in fp32 (``pipeline_training_patches.patch_mamba_training_scan_fp32_state``). A checkpoint whose
run_config needs the torch_grouped repair is loaded from its export clone (``scripts/checkpoint/export_clone.py``),
exactly as the exporter loads it.

**Outputs.** Each command writes into one output directory and refuses to replace a file: ``megatron_scores.npz``,
``hf_scores.npz``, ``hf_greedy.json`` and the judge's ``export_check-<first 12 of the spec's sha256>.json``
(``export_check_result.py``, which a reader of the verdict shares). A scores
file records the spec's measurement digest (the sha256 of the keys a score depends on, ``MEASUREMENT_KEYS``), so a spec
whose thresholds are calibrated later can judge scores taken before.

**Calibration.** A spec's ``calibration`` block records how its ``ce_agreement`` tolerance was set. It may also
pre-register the runs that must bear the tolerance out: ``results_root`` and, under ``runs``, each run's output
``directory`` with ``expect: PASS`` (an export scored against its own checkpoint) or ``expect: FAIL`` (a negative
control). ``calibrate`` judges each such run's scores with the spec, as ``judge`` does, and writes a report whose
verdict is PASS only when every run's ``ce_agreement`` outcome is the expected one; a run without ``expect`` is a
record and is not judged.

Commands (inside the container; ``scripts/checkpoint/export_check.sbatch`` runs score-megatron, score-hf and judge as
one job)::

    python scripts/checkpoint/export_check.py select --spec SPEC --out LISTING.yaml
    torchrun --standalone --nproc_per_node=4 scripts/checkpoint/export_check.py score-megatron --spec SPEC \\
        --checkpoint ITER_DIR [--load-path EXPORT_CLONE] --out-dir DIR [--repeats N]
    python scripts/checkpoint/export_check.py score-hf --spec SPEC --export HF_DIR --out-dir DIR [--repeats N]
    python scripts/checkpoint/export_check.py judge --spec SPEC --export HF_DIR --checkpoint ITER_DIR --out-dir DIR
    python scripts/checkpoint/export_check.py calibrate --spec SPEC --report-out REPORT.json

``--repeats`` scores every document that many times in one process and the judge reports how far the passes differ;
the criteria read the first pass. The judge and ``calibrate`` exit 0 on PASS, 1 on FAIL and otherwise 2.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
import math
import os
import re
import socket
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
for _directory in (REPO_ROOT, REPO_ROOT / "configs" / "control_pretraining"):
    if str(_directory) not in sys.path:
        sys.path.append(str(_directory))

from corpus_documents import bin_path, idx_path, read_index, token_memmap  # noqa: E402
from scripts.checkpoint.decoding_defaults import GENERATION_CONFIG, generation_config_text  # noqa: E402
from scripts.checkpoint.export_check_result import RESULT_FORMAT, result_path, spec_digest  # noqa: E402
from scripts.checkpoint.safetensors_index import INDEX_FILE  # noqa: E402
from scripts.forward_passes import (  # noqa: E402
    gather_to_rank_zero,
    load_single_chunk_model,
    lockstep_schedule,
    merge_by_index,
)
from scripts.mapping_keys import require_keys  # noqa: E402
from scripts.mapping_values import require_absolute_path, require_int, require_number, require_string  # noqa: E402
from scripts.telemetry.code_revision import code_revision  # noqa: E402
from scripts.telemetry.gate_outcome import FAIL, NOT_EVALUATED, PASS, exit_status  # noqa: E402
from scripts.token_ids import require_token_id, require_token_id_list  # noqa: E402
from scripts.training.config_compose import BASE_CONFIG_KEY, load_composed  # noqa: E402

import pipeline_coherence_test as coherence  # noqa: E402


LOGGER = logging.getLogger("export_check")

MEGATRON_SCORES = "megatron_scores.npz"
HF_SCORES = "hf_scores.npz"
HF_GREEDY = "hf_greedy.json"
PENDING = "PENDING"
# The exporter's modes (its --reasoning and --no-reasoning flags), each with whether it writes reasoning defaults.
DECODING_DEFAULTS = {"reasoning": True, "no-reasoning": False}
# The judge's outcome for each exit status of a set of criteria (``gate_outcome.exit_status``).
OUTCOME_OF_STATUS = {0: PASS, 1: FAIL, 2: NOT_EVALUATED}
# The seed of the reference ranks' RNG trackers; nothing in an eval-mode forward draws from them.
REFERENCE_SEED = 0
# The export is stored in bf16 and evaluated in bf16; the HF side scores it as it is used.
HF_DTYPE = "bfloat16"
# The ids of the prompt shown before each greedy continuation in the results, for a reader.
RENDERED_PROMPT_TAIL = 48

SPEC_KEYS = frozenset(
    {
        "reference_export",
        "tokenizer",
        "hidden_token_id",
        "prefix_token_ids",
        "structure",
        "corpora",
        "selection",
        "documents",
        "prompts",
        "megatron_reference",
        "greedy",
        "criteria",
        "calibration",
    }
)
# What the scores depend on: a change to any of these keys makes earlier scores unusable. The criteria, the
# calibration record and the structural block (the reference export and the files compared with it) do not change a
# score, so a spec calibrated after its scores were taken still judges them.
MEASUREMENT_KEYS = (
    "tokenizer",
    "hidden_token_id",
    "prefix_token_ids",
    "corpora",
    "documents",
    "prompts",
    "megatron_reference",
    "greedy",
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
CORPUS_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")

# The parallel layout the reference forward runs at, besides the spec's tensor and expert parallelism: one pipeline
# stage, no context parallelism or sequence parallelism, unsharded experts, and no initialisation of weights that the
# checkpoint then overwrites (``load_megatron_model`` sets the same fields to load a checkpoint for inference).
REFERENCE_LAYOUT = {
    "pipeline_model_parallel_size": 1,
    "virtual_pipeline_model_parallel_size": None,
    "context_parallel_size": 1,
    "expert_tensor_parallel_size": 1,
    "sequence_parallel": False,
    "perform_initialization": False,
}
# The reference forward also clears the token dispatcher's SM counts, which set its speed and never its numbers. A
# run_config written before Megatron-LM unified them into moe_flex_dispatcher_num_sms holds the two deprecated knobs
# at their old defaults, 20 and 16 (the NVIDIA base conversions under megatron_bridges/models/ do), and the pin
# refuses that pair when it builds the model.
REFERENCE_DISPATCHER_SM_COUNTS = {
    "moe_deepep_num_sms": None,
    "moe_hybridep_num_sms": None,
    "moe_flex_dispatcher_num_sms": None,
}


class ExportCheckError(RuntimeError):
    """The check cannot be run as asked: a malformed spec, a window that is not what the spec lists, or scores that
    were not taken for this spec, export or checkpoint."""


# ----------------------------------------------------------------------------------------------------------------------
# The spec
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """The first ``tokens`` ids of document ``document`` of corpus ``corpus``, whose ids have the sha256 ``sha256``."""

    corpus: str
    document: int
    tokens: int
    sha256: str


@dataclass(frozen=True)
class Selection:
    """How ``select`` draws the windows: from a generator seeded with ``seed``, per corpus ``document_counts[c]``
    documents of at least ``window_tokens`` ids (their first ``window_tokens``), then per corpus ``prompt_counts[c]``
    prompts, each a document's ids before the first run of H that starts at or after ``min_context`` and at or before
    ``max_context``."""

    seed: int
    window_tokens: int
    document_counts: dict[str, int]
    min_context: int
    max_context: int
    prompt_counts: dict[str, int]


@dataclass(frozen=True)
class MegatronReference:
    """The reference forward's posture: its tensor and expert parallelism, whether the Mamba scan runs in fp32, and the
    model-config fields set on the checkpoint's run_config before the model is built."""

    tensor_model_parallel_size: int
    expert_model_parallel_size: int
    fp32_ssm_state: bool
    model_overrides: dict[str, Any]


@dataclass(frozen=True)
class Greedy:
    """Greedy continuations: at most ``max_new_tokens`` ids each, ending at the first stop id."""

    max_new_tokens: int
    stop_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class CeTolerance:
    """How far the export's per-token cross-entropy may lie from the Megatron reference's at the retained targets:
    the mean of HF minus Megatron in absolute value, the mean absolute difference and its 99th percentile."""

    max_abs_mean_shift: float
    max_mean_abs_difference: float
    max_p99_abs_difference: float


@dataclass(frozen=True)
class Criteria:
    """The thresholds; ``ce_agreement`` is None while its tolerance is ``PENDING`` calibration, and the two about H are
    None for a spec that omits them (states them null)."""

    ce_agreement: CeTolerance | None
    hidden_probability_max: float | None
    greedy_hidden_emissions_max: int | None


@dataclass(frozen=True)
class ExportCheckSpec:
    """A spec as ``load_spec`` read it, with its path, sha256, measurement digest and parsed content. ``documents``
    and ``prompts`` are None while the spec still reads ``PENDING`` for them (before ``select`` has drawn them), and
    ``decoding_defaults`` is None unless the spec judges ``generation_config.json`` by an exporter mode's defaults."""

    path: str
    sha256: str
    measurement_sha256: str
    content: dict[str, Any]
    reference_export: Path
    tokenizer: str
    tokenizer_revision: str
    hidden_token_id: int
    prefix_token_ids: tuple[int, ...]
    byte_equal_files: tuple[str, ...]
    decoding_defaults: str | None
    corpora: dict[str, Path]
    selection: Selection
    documents: tuple[Window, ...] | None
    prompts: tuple[Window, ...] | None
    megatron_reference: MegatronReference
    greedy: Greedy
    criteria: Criteria

    def listed(self, name: str) -> tuple[Window, ...]:
        """The ``documents`` or ``prompts`` windows; raises when the spec still reads ``PENDING`` for them."""
        windows = getattr(self, name)
        if windows is None:
            raise ExportCheckError(f"{self.path}: {name} is {PENDING}; draw them with `select` and list them first")
        return windows


_integer = functools.partial(require_int, error=ExportCheckError)
_positive = functools.partial(require_number, minimum=0, inclusive=False, error=ExportCheckError)
_string = functools.partial(require_string, error=ExportCheckError)


def _counts(raw: Any, where: str, corpora: dict[str, Path]) -> dict[str, int]:
    if not isinstance(raw, dict) or not raw:
        raise ExportCheckError(f"{where} must be a non-empty mapping of corpus names to counts, not {raw!r}")
    unknown = sorted(set(raw) - set(corpora))
    if unknown:
        raise ExportCheckError(f"{where} names corpora {unknown} that `corpora` does not")
    return {name: _integer(count, f"{where}.{name}", 1) for name, count in raw.items()}


def _corpora(raw: Any) -> dict[str, Path]:
    if not isinstance(raw, dict) or not raw:
        raise ExportCheckError(f"corpora must be a non-empty mapping of names to .bin/.idx prefixes, not {raw!r}")
    corpora = {}
    for name, prefix in raw.items():
        if not isinstance(name, str) or not CORPUS_NAME.match(name):
            raise ExportCheckError(f"corpus name {name!r} must be letters, digits, '.', '_' or '-'")
        corpora[name] = Path(require_absolute_path(prefix, f"corpora.{name}", error=ExportCheckError))
    return corpora


def _selection(raw: Any, corpora: dict[str, Path]) -> Selection:
    raw = require_keys(raw, "selection", {"seed", "documents", "prompts"}, error=ExportCheckError)
    documents = require_keys(
        raw["documents"], "selection.documents", {"window_tokens", "counts"}, error=ExportCheckError
    )
    prompts = require_keys(
        raw["prompts"], "selection.prompts", {"min_context", "max_context", "counts"}, error=ExportCheckError
    )
    min_context = _integer(prompts["min_context"], "selection.prompts.min_context", 1)
    max_context = _integer(prompts["max_context"], "selection.prompts.max_context", min_context)
    return Selection(
        seed=_integer(raw["seed"], "selection.seed", 0),
        window_tokens=_integer(documents["window_tokens"], "selection.documents.window_tokens", 2),
        document_counts=_counts(documents["counts"], "selection.documents.counts", corpora),
        min_context=min_context,
        max_context=max_context,
        prompt_counts=_counts(prompts["counts"], "selection.prompts.counts", corpora),
    )


def _windows(raw: Any, name: str, corpora: dict[str, Path]) -> tuple[Window, ...] | None:
    if raw == PENDING:
        return None
    if not isinstance(raw, list) or not raw:
        raise ExportCheckError(f"{name} must be {PENDING} or a non-empty list of windows, not {raw!r}")
    windows = []
    for index, item in enumerate(raw):
        where = f"{name}[{index}]"
        item = require_keys(item, where, {"corpus", "document", "tokens", "sha256"}, error=ExportCheckError)
        if item["corpus"] not in corpora:
            raise ExportCheckError(f"{where}.corpus {item['corpus']!r} is not one of `corpora`")
        if not isinstance(item["sha256"], str) or not SHA256.match(item["sha256"]):
            raise ExportCheckError(f"{where}.sha256 must be 64 lowercase hex digits, not {item['sha256']!r}")
        windows.append(
            Window(
                corpus=item["corpus"],
                document=_integer(item["document"], f"{where}.document", 0),
                tokens=_integer(item["tokens"], f"{where}.tokens", 1),
                sha256=item["sha256"],
            )
        )
    keys = [(window.corpus, window.document) for window in windows]
    if len(set(keys)) != len(keys):
        raise ExportCheckError(f"{name} lists a document twice")
    return tuple(windows)


def _check_listing_follows_selection(
    documents: tuple[Window, ...] | None, prompts: tuple[Window, ...] | None, selection: Selection
) -> None:
    """Refuse a listing whose per-corpus counts or window lengths are not the ones its selection draws."""
    for name, windows, counts in (
        ("documents", documents, selection.document_counts),
        ("prompts", prompts, selection.prompt_counts),
    ):
        if windows is None:
            continue
        listed = {corpus: sum(1 for w in windows if w.corpus == corpus) for corpus in {w.corpus for w in windows}}
        if listed != counts:
            raise ExportCheckError(f"{name} lists {listed} windows per corpus; selection draws {counts}")
    if documents is not None:
        lengths = sorted({window.tokens for window in documents})
        if lengths != [selection.window_tokens]:
            raise ExportCheckError(f"documents are {lengths} ids long; selection draws {selection.window_tokens}")
    if prompts is not None:
        outside = [w for w in prompts if not selection.min_context <= w.tokens <= selection.max_context]
        if outside:
            raise ExportCheckError(
                f"prompt windows {[(w.corpus, w.document, w.tokens) for w in outside][:3]} lie outside "
                f"[{selection.min_context}, {selection.max_context}] ids"
            )


def _megatron_reference(raw: Any) -> MegatronReference:
    keys = {"tensor_model_parallel_size", "expert_model_parallel_size", "fp32_ssm_state", "model_overrides"}
    raw = require_keys(raw, "megatron_reference", keys, error=ExportCheckError)
    # The reference projects the whole output head on one rank, which tensor parallelism would shard.
    if raw["tensor_model_parallel_size"] != 1 or isinstance(raw["tensor_model_parallel_size"], bool):
        raise ExportCheckError(
            f"megatron_reference.tensor_model_parallel_size must be 1, not {raw['tensor_model_parallel_size']!r}: the "
            "reference applies the whole output head in fp32 on each rank"
        )
    if not isinstance(raw["fp32_ssm_state"], bool):
        raise ExportCheckError(
            f"megatron_reference.fp32_ssm_state must be true or false, not {raw['fp32_ssm_state']!r}"
        )
    overrides = raw["model_overrides"]
    if not isinstance(overrides, dict) or not all(isinstance(key, str) for key in overrides):
        raise ExportCheckError(
            f"megatron_reference.model_overrides must map model-config fields to values: {overrides!r}"
        )
    clash = sorted(
        set(overrides) & (set(REFERENCE_LAYOUT) | {"tensor_model_parallel_size", "expert_model_parallel_size"})
    )
    if clash:
        raise ExportCheckError(
            f"megatron_reference.model_overrides sets the parallel layout ({clash}), which it fixes"
        )
    return MegatronReference(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=_integer(
            raw["expert_model_parallel_size"], "megatron_reference.expert_model_parallel_size", 1
        ),
        fp32_ssm_state=raw["fp32_ssm_state"],
        model_overrides=dict(overrides),
    )


def _criteria(raw: Any) -> Criteria:
    raw = require_keys(
        raw, "criteria", {"ce_agreement", "hidden_probability", "greedy_hidden_emissions"}, error=ExportCheckError
    )
    tolerance = None
    if raw["ce_agreement"] != PENDING:
        ce = require_keys(
            raw["ce_agreement"],
            "criteria.ce_agreement",
            {"max_abs_mean_shift", "max_mean_abs_difference", "max_p99_abs_difference"},
            error=ExportCheckError,
        )
        tolerance = CeTolerance(**{key: _positive(value, f"criteria.ce_agreement.{key}") for key, value in ce.items()})
    hidden_max = None
    if raw["hidden_probability"] is not None:
        hidden = require_keys(
            raw["hidden_probability"], "criteria.hidden_probability", {"max"}, error=ExportCheckError
        )
        hidden_max = _positive(hidden["max"], "criteria.hidden_probability.max")
        if hidden_max >= 1:
            raise ExportCheckError(f"criteria.hidden_probability.max must be a probability below 1, not {hidden_max}")
    greedy_max = None
    if raw["greedy_hidden_emissions"] is not None:
        greedy = require_keys(
            raw["greedy_hidden_emissions"], "criteria.greedy_hidden_emissions", {"max"}, error=ExportCheckError
        )
        greedy_max = _integer(greedy["max"], "criteria.greedy_hidden_emissions.max", 0)
    return Criteria(ce_agreement=tolerance, hidden_probability_max=hidden_max, greedy_hidden_emissions_max=greedy_max)


def measurement_digest(content: dict[str, Any]) -> str:
    """The sha256 of the spec's ``MEASUREMENT_KEYS``, as canonical JSON: what a scores file must have been taken with."""
    measured = {key: content[key] for key in MEASUREMENT_KEYS}
    return hashlib.sha256(json.dumps(measured, sort_keys=True).encode()).hexdigest()


def _read_spec_file(path: Path) -> Any:
    return yaml.safe_load(path.read_bytes())


def read_spec_content(path: Path) -> dict[str, Any]:
    """A spec's content: the file at ``path`` composed onto the spec its ``base_config`` names, and that one's, and so
    on (``scripts/training/config_compose.py`` ``load_composed``: mappings merge, anything else replaces), each file
    read with ``yaml.safe_load``. A composing spec's ``calibration`` and ``criteria.ce_agreement`` are its own, never
    merged with a base's, so it must state both: a tolerance, and the record of how it was set, hold only for the spec
    they were set for. Raises ExportCheckError when a composing spec does not state them, and as ``load_composed``
    does for a missing file, a loop or a malformed ``base_config``."""
    content = load_composed(path, _read_spec_file)
    own = _read_spec_file(path)
    if BASE_CONFIG_KEY not in own:
        return content
    criteria = own.get("criteria")
    if "calibration" not in own or not isinstance(criteria, dict) or "ce_agreement" not in criteria:
        raise ExportCheckError(
            f"{path} composes onto {own[BASE_CONFIG_KEY]} but does not state its own calibration and "
            "criteria.ce_agreement: a tolerance holds only for the spec it was calibrated for"
        )
    content["calibration"] = own["calibration"]
    content["criteria"]["ce_agreement"] = criteria["ce_agreement"]
    return content


def load_spec(path: str | Path) -> ExportCheckSpec:
    """Read and check a spec, composed as ``read_spec_content`` composes it. Every key is required::

        reference_export: /abs/path/to/reference/hf        # the export the structure is compared with
        tokenizer: {name: <Hub id or dir>, revision: <40-hex commit>}
        hidden_token_id: 500                               # H
        prefix_token_ids: [2]                              # read before every window
        structure: {byte_equal_files: [config.json, ...]}  # optionally decoding_defaults: reasoning | no-reasoning
        corpora: {<name>: /abs/prefix/of/.bin/.idx}
        selection:
          seed: 20261010
          documents: {window_tokens: 8192, counts: {<name>: 32}}
          prompts: {min_context: 32, max_context: 512, counts: {<name>: 50}}
        documents: PENDING | [{corpus: <name>, document: <int>, tokens: <int>, sha256: <hex>}]
        prompts:   PENDING | [... the same ...]
        megatron_reference: {tensor_model_parallel_size: 1, expert_model_parallel_size: 4, fp32_ssm_state: true,
                             model_overrides: {fp8: null, ...}}
        greedy: {max_new_tokens: 64, stop_token_ids: [2]}
        criteria:
          ce_agreement: PENDING | {max_abs_mean_shift: ., max_mean_abs_difference: ., max_p99_abs_difference: .}
          hidden_probability: {max: 1.0e-4} | null           # null omits the criterion
          greedy_hidden_emissions: {max: 0} | null           # null omits the criterion
        calibration: {...}                                 # how the tolerance was measured, as a record

    Raises ExportCheckError on a missing or unknown key, a malformed value, a value JSON cannot hold (an unquoted
    date), a listing whose counts or lengths are not its selection's, a stop id that is H, model overrides that set
    the parallel layout, or decoding defaults for a generation config the structure does not compare.
    """
    content = read_spec_content(Path(path))
    raw = require_keys(content, f"export-check spec {path}", SPEC_KEYS, error=ExportCheckError)
    try:
        json.dumps(content)
    except TypeError as error:
        # The spec is digested and recorded as JSON; YAML reads an unquoted date or time as an object JSON cannot hold.
        raise ExportCheckError(f"export-check spec {path} holds a value JSON cannot represent ({error}); quote it")
    reference = Path(require_absolute_path(raw["reference_export"], "reference_export", error=ExportCheckError))
    tokenizer = require_keys(raw["tokenizer"], "tokenizer", {"name", "revision"}, error=ExportCheckError)
    hidden = require_token_id(raw["hidden_token_id"], "hidden_token_id", error=ExportCheckError)
    structure = require_keys(
        raw["structure"], "structure", {"byte_equal_files"}, {"decoding_defaults"}, error=ExportCheckError
    )
    files = structure["byte_equal_files"]
    if not isinstance(files, list) or not files or not all(isinstance(f, str) and f and "/" not in f for f in files):
        raise ExportCheckError(f"structure.byte_equal_files must be a non-empty list of file names, not {files!r}")
    decoding = structure.get("decoding_defaults")
    if decoding is not None and decoding not in DECODING_DEFAULTS:
        raise ExportCheckError(
            f"structure.decoding_defaults must be one of the exporter's modes {sorted(DECODING_DEFAULTS)}, "
            f"not {decoding!r}"
        )
    if decoding is not None and GENERATION_CONFIG not in files:
        raise ExportCheckError(
            f"structure.decoding_defaults judges {GENERATION_CONFIG}, which structure.byte_equal_files does not list"
        )
    corpora = _corpora(raw["corpora"])
    selection = _selection(raw["selection"], corpora)
    documents = _windows(raw["documents"], "documents", corpora)
    prompts = _windows(raw["prompts"], "prompts", corpora)
    _check_listing_follows_selection(documents, prompts, selection)
    greedy = require_keys(raw["greedy"], "greedy", {"max_new_tokens", "stop_token_ids"}, error=ExportCheckError)
    stop = require_token_id_list(
        greedy["stop_token_ids"], "greedy.stop_token_ids", allow_empty=False, error=ExportCheckError
    )
    if hidden in stop:
        raise ExportCheckError(f"greedy.stop_token_ids {list(stop)} include H ({hidden}), which the check counts")
    if not isinstance(raw["calibration"], dict):
        raise ExportCheckError(
            f"calibration must be a mapping recording how the tolerance was set, not {raw['calibration']!r}"
        )
    return ExportCheckSpec(
        path=str(path),
        sha256=spec_digest(Path(path)),
        measurement_sha256=measurement_digest(content),
        content=content,
        reference_export=reference,
        tokenizer=_string(tokenizer["name"], "tokenizer.name"),
        tokenizer_revision=_string(tokenizer["revision"], "tokenizer.revision"),
        hidden_token_id=hidden,
        prefix_token_ids=require_token_id_list(
            raw["prefix_token_ids"], "prefix_token_ids", allow_empty=False, error=ExportCheckError
        ),
        byte_equal_files=tuple(files),
        decoding_defaults=decoding,
        corpora=corpora,
        selection=selection,
        documents=documents,
        prompts=prompts,
        megatron_reference=_megatron_reference(raw["megatron_reference"]),
        greedy=Greedy(
            max_new_tokens=_integer(greedy["max_new_tokens"], "greedy.max_new_tokens", 1), stop_token_ids=stop
        ),
        criteria=_criteria(raw["criteria"]),
    )


# ----------------------------------------------------------------------------------------------------------------------
# Windows: drawn from the corpora once (`select`), read back and checked by digest on every run
# ----------------------------------------------------------------------------------------------------------------------


class Corpus:
    """One ``.bin/.idx`` corpus, read through the campaign's corpus reader: per document, its ids without the EOD."""

    def __init__(self, prefix: Path) -> None:
        self.prefix = prefix
        self.index = read_index(idx_path(prefix), bin_path(prefix))
        self.tokens = token_memmap(bin_path(prefix))

    def ids(self, document: int) -> np.ndarray:
        """Document ``document``'s ids, its EOD left off (an empty document has none)."""
        if not 0 <= document < self.index.docs:
            raise ExportCheckError(f"{self.prefix} has {self.index.docs} documents; there is no document {document}")
        start, size = int(self.index.starts[document]), int(self.index.sizes[document])
        return np.asarray(self.tokens[start : start + max(size - 1, 0)])


def window_digest(ids: np.ndarray) -> str:
    """The sha256 of ids as little-endian int32 bytes."""
    return hashlib.sha256(np.ascontiguousarray(ids, dtype="<i4").tobytes()).hexdigest()


def hidden_run_start(ids: np.ndarray, hidden_token_id: int, min_context: int, max_context: int) -> int | None:
    """The first position in ``[min_context, max_context]`` at which a run of H starts (H there, not H before it), or
    None."""
    head = ids[: max_context + 1] == hidden_token_id
    starts = np.flatnonzero(head & ~np.concatenate(([False], head[:-1])))
    inside = starts[starts >= min_context]
    return int(inside[0]) if inside.size else None


def _window(name: str, document: int, ids: np.ndarray) -> dict[str, Any]:
    return {"corpus": name, "document": document, "tokens": int(len(ids)), "sha256": window_digest(ids)}


def select_windows(spec: ExportCheckSpec) -> dict[str, list[dict[str, Any]]]:
    """Draw the spec's documents and prompts as its ``selection`` says, deterministically, and return the listing.

    One generator, seeded with ``selection.seed``, draws for each corpus in turn (in the order ``documents.counts``
    names them) its documents without replacement, uniformly among those holding at least ``window_tokens`` ids, and
    then, for each corpus of ``prompts.counts``, walks the corpus's documents in an order it permutes, keeping each
    document that has a run of H starting within ``[min_context, max_context]`` (``hidden_run_start``) until it has the
    count. Raises ExportCheckError when a corpus has too few such documents.
    """
    selection = spec.selection
    corpora = {name: Corpus(spec.corpora[name]) for name in {*selection.document_counts, *selection.prompt_counts}}
    generator = np.random.default_rng(selection.seed)
    documents: list[dict[str, Any]] = []
    for name, count in selection.document_counts.items():
        corpus = corpora[name]
        eligible = np.flatnonzero(corpus.index.sizes - 1 >= selection.window_tokens)
        if len(eligible) < count:
            raise ExportCheckError(
                f"{name}: {len(eligible)} documents hold {selection.window_tokens} ids or more; {count} are drawn"
            )
        for document in sorted(int(d) for d in generator.choice(eligible, size=count, replace=False)):
            documents.append(_window(name, document, corpus.ids(document)[: selection.window_tokens]))
    prompts: list[dict[str, Any]] = []
    for name, count in selection.prompt_counts.items():
        corpus, kept = corpora[name], []
        for document in generator.permutation(corpus.index.docs).tolist():
            ids = corpus.ids(document)
            start = hidden_run_start(ids, spec.hidden_token_id, selection.min_context, selection.max_context)
            if start is not None:
                kept.append(_window(name, document, ids[:start]))
                if len(kept) == count:
                    break
        if len(kept) < count:
            raise ExportCheckError(
                f"{name}: {len(kept)} documents have a run of H starting within "
                f"[{selection.min_context}, {selection.max_context}]; {count} are drawn"
            )
        prompts.extend(sorted(kept, key=lambda window: window["document"]))
    return {"documents": documents, "prompts": prompts}


def read_windows(spec: ExportCheckSpec, windows: tuple[Window, ...]) -> list[np.ndarray]:
    """Each window's ids, read from its corpus; raises ExportCheckError when a window is not the one the spec lists
    (a shorter document, other ids)."""
    corpora: dict[str, Corpus] = {}
    read = []
    for window in windows:
        if window.corpus not in corpora:
            corpora[window.corpus] = Corpus(spec.corpora[window.corpus])
        ids = corpora[window.corpus].ids(window.document)
        if len(ids) < window.tokens:
            raise ExportCheckError(
                f"{window.corpus} document {window.document} holds {len(ids)} ids; the spec reads {window.tokens}"
            )
        ids = ids[: window.tokens]
        digest = window_digest(ids)
        if digest != window.sha256:
            raise ExportCheckError(
                f"{window.corpus} document {window.document}: its first {window.tokens} ids have the sha256 {digest}, "
                f"the spec lists {window.sha256}; the corpus is not the one the spec was drawn from"
            )
        read.append(ids)
    return read


# ----------------------------------------------------------------------------------------------------------------------
# Scoring, shared by both forwards
# ----------------------------------------------------------------------------------------------------------------------


def window_inputs(spec: ExportCheckSpec, ids: np.ndarray) -> list[int]:
    """The input ids that score ``ids``: the prefix, then every id but the last, so the last ``len(ids)`` logit rows
    predict exactly ``ids``."""
    return list(spec.prefix_token_ids) + ids[:-1].tolist()


def target_scores(logits, targets, hidden_token_id: int):
    """``(cross-entropy, log p(H))`` per row of fp32 ``logits`` ``[positions, vocabulary]``, whose row ``i`` predicts
    ``targets[i]``: an unfused log-softmax in the logits' own precision."""
    import torch

    normaliser = torch.logsumexp(logits, dim=-1)
    cross_entropy = normaliser - logits.gather(1, targets[:, None]).squeeze(1)
    return cross_entropy, logits[:, hidden_token_id] - normaliser


def score_window(
    window_logits: Callable[[list[int]], Any], spec: ExportCheckSpec, ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One window's ``(cross-entropy, log p(H))`` per target, as float32, from ``window_logits`` (input ids to fp32
    logits ``[positions, vocabulary]``)."""
    import torch

    logits = window_logits(window_inputs(spec, ids))
    rows = logits[-len(ids) :]
    targets = torch.as_tensor(ids.astype(np.int64), device=rows.device)
    cross_entropy, logp_hidden = target_scores(rows, targets, spec.hidden_token_id)
    return cross_entropy.float().cpu().numpy(), logp_hidden.float().cpu().numpy()


def score_documents(
    window_logits: Callable[[list[int]], Any], spec: ExportCheckSpec, documents: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Every document's scores (``score_window``), concatenated in the spec's order."""
    scored = [score_window(window_logits, spec, ids) for ids in documents]
    return np.concatenate([ce for ce, _ in scored]), np.concatenate([logp for _, logp in scored])


def refuse_existing(*paths: Path) -> None:
    """Raise when any of ``paths`` exists: the check never replaces an output."""
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"{existing} exist; the export check never replaces its outputs")


def run_record(spec: ExportCheckSpec) -> dict[str, Any]:
    """What identifies a scoring run: the spec, the code that measured and where it ran."""
    return {
        "spec": {"path": spec.path, "sha256": spec.sha256, "measurement_sha256": spec.measurement_sha256},
        "code_revision": code_revision(str(REPO_ROOT)),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def write_scores(path: Path, passes: list[tuple[np.ndarray, np.ndarray]], targets: np.ndarray, record: dict) -> None:
    """Write the scores of every pass (``[passes, targets]`` cross-entropy and log p(H)), the targets and the record,
    refusing to replace a file."""
    refuse_existing(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    with partial.open("wb") as handle:
        np.savez(
            handle,
            cross_entropy=np.stack([ce for ce, _ in passes]),
            logp_hidden=np.stack([logp for _, logp in passes]),
            targets=targets.astype(np.int32),
            record=np.array(json.dumps(record)),
        )
    partial.replace(path)


@dataclass(frozen=True)
class Scores:
    """A scores file as ``read_scores`` read it."""

    cross_entropy: np.ndarray  # [passes, targets]
    logp_hidden: np.ndarray  # [passes, targets]
    targets: np.ndarray  # [targets]
    record: dict[str, Any]


def read_scores(path: Path, spec: ExportCheckSpec) -> Scores:
    """Read a scores file, refusing one taken with another measurement than ``spec``'s or of another shape than its
    documents."""
    with np.load(path, allow_pickle=False) as data:
        scores = Scores(
            cross_entropy=data["cross_entropy"],
            logp_hidden=data["logp_hidden"],
            targets=data["targets"],
            record=json.loads(data["record"].item()),
        )
    taken_with = scores.record["spec"]["measurement_sha256"]
    if taken_with != spec.measurement_sha256:
        raise ExportCheckError(
            f"{path} was scored with a spec measuring {taken_with}; {spec.path} measures {spec.measurement_sha256}"
        )
    expected = sum(window.tokens for window in spec.listed("documents"))
    if scores.targets.shape != (expected,) or scores.cross_entropy.shape[1:] != (expected,):
        raise ExportCheckError(f"{path} holds {scores.targets.shape[0]} targets; the spec's documents hold {expected}")
    return scores


# ----------------------------------------------------------------------------------------------------------------------
# The Megatron reference forward (torchrun, one rank per GPU)
# ----------------------------------------------------------------------------------------------------------------------


def reference_model_config(model_cfg: Any, reference: MegatronReference) -> Any:
    """``model_cfg`` (a checkpoint's model config, as ``load_model_config`` read it) set to the reference posture:
    ``REFERENCE_LAYOUT``, the spec's tensor and expert parallelism, ``REFERENCE_DISPATCHER_SM_COUNTS``, then its
    ``model_overrides``. Raises ExportCheckError on a field the config does not have, which would otherwise set
    nothing."""
    settings = {
        **REFERENCE_LAYOUT,
        "tensor_model_parallel_size": reference.tensor_model_parallel_size,
        "expert_model_parallel_size": reference.expert_model_parallel_size,
        **REFERENCE_DISPATCHER_SM_COUNTS,
        **reference.model_overrides,
    }
    missing = sorted(key for key in settings if not hasattr(model_cfg, key))
    if missing:
        raise ExportCheckError(f"the checkpoint's model config ({type(model_cfg).__name__}) has no fields {missing}")
    for key, value in settings.items():
        setattr(model_cfg, key, value)
    return model_cfg


def megatron_fp32_logits(model, input_ids: list[int]):
    """The fp32 logits of a Megatron model for ``input_ids`` (one sequence): the output layer's input, as the model's
    own forward feeds it, projected with the layer's weight (and bias) cast to fp32, ``[positions, vocabulary]``.

    Raises RuntimeError when the output layer was not called once on every position, or when the projection's last row
    disagrees with the model's own logits beyond bf16 rounding (``pipeline_coherence_test.HEAD_PROJECTION_TOLERANCE``).
    """
    import torch
    from megatron.core.utils import unwrap_model

    head = unwrap_model(model).output_layer
    fed = []
    hook = head.register_forward_pre_hook(lambda module, args: fed.append(args[0]))
    device = head.weight.device
    tokens = torch.tensor([input_ids], device=device)
    positions = torch.arange(len(input_ids), device=device)[None]
    try:
        with torch.no_grad():
            native = model(input_ids=tokens, position_ids=positions, attention_mask=None)[0, -1].float()
    finally:
        hook.remove()
    if len(fed) != 1 or tuple(fed[0].shape[:2]) != (len(input_ids), 1):
        raise RuntimeError(
            f"the output layer was called {len(fed)} times, on {[tuple(x.shape) for x in fed]}, not once on the "
            f"{len(input_ids)} positions"
        )
    bias = getattr(head, "bias", None)
    with torch.no_grad():
        logits = torch.nn.functional.linear(
            fed[0][:, 0].float(), head.weight.float(), None if bias is None else bias.float()
        )
    if not torch.allclose(logits[-1], native, **coherence.HEAD_PROJECTION_TOLERANCE):
        worst = float((logits[-1] - native).abs().max())
        raise RuntimeError(
            f"the fp32 projection of the output layer's input differs from the model's own logits by up to {worst:.4g}, "
            f"beyond bf16 rounding ({coherence.HEAD_PROJECTION_TOLERANCE})"
        )
    return logits


def load_reference_model(spec: ExportCheckSpec, load_path: Path):
    """The checkpoint at ``load_path`` built at the reference posture on this rank's GPU, with weights loaded, in eval
    mode, after the fp32 Mamba scan is installed when the spec asks for it. The process group comes first: the
    checkpoint's run_config is instantiated in it, as the exporter instantiates it."""
    import torch

    from megatron.bridge.training.model_load_save import load_model_config
    from megatron.bridge.utils.common_utils import get_local_rank_preinit

    torch.cuda.set_device(get_local_rank_preinit())
    torch.distributed.init_process_group("nccl")
    model_cfg, _ = load_model_config(str(load_path))
    reference_model_config(model_cfg, spec.megatron_reference)
    model_cfg.initialize_model_parallel(seed=REFERENCE_SEED)
    if spec.megatron_reference.fp32_ssm_state:
        from pipeline_training_patches import patch_mamba_training_scan_fp32_state

        if not patch_mamba_training_scan_fp32_state(checkpointed=False):
            raise RuntimeError("the fp32 Mamba scan could not be installed (see the log); the reference needs it")
    return load_single_chunk_model(str(load_path), model_cfg, error=RuntimeError)


def score_megatron(
    spec: ExportCheckSpec, checkpoint: Path, load_path: str | None, out_dir: Path, repeats: int
) -> None:
    """Score the spec's documents with the reference forward of ``checkpoint`` and write ``megatron_scores.npz``.

    Run under torchrun, one rank per GPU (``main`` wraps it in the bridge's ``torchrun_main``, which ends every rank
    when one raises). Each data-parallel rank scores every ``data_parallel_size``-th document, all ranks in step (the
    experts' all-to-all needs every rank's forward), and rank 0 gathers and writes the scores in the spec's order.
    ``load_path`` is the checkpoint's export clone: ``resolve_load_path`` (``pipeline_checkpoint_convert_hf``) requires
    one for a torch_grouped checkpoint and checks that it is a clone of this very checkpoint.
    """
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state

    from pipeline_checkpoint_convert_hf import resolve_load_path

    output = out_dir / MEGATRON_SCORES
    refuse_existing(output)
    documents = read_windows(spec, spec.listed("documents"))
    source = resolve_load_path(checkpoint, load_path)
    started = time.monotonic()
    model = load_reference_model(spec, source)
    loaded = time.monotonic()
    rank, ranks = parallel_state.get_data_parallel_rank(), parallel_state.get_data_parallel_world_size()
    if len(documents) % ranks:
        raise ExportCheckError(f"{len(documents)} documents do not divide over {ranks} data-parallel ranks")
    passes = []
    for _ in range(repeats):
        # Every scheduled step is a real document: the divisibility check above leaves no padded step to drop.
        mine = {
            index: score_window(lambda ids: megatron_fp32_logits(model, ids), spec, documents[index])
            for index, _ in lockstep_schedule(len(documents), ranks, rank)
        }
        gathered = gather_to_rank_zero(mine, group=None)
        if dist.get_rank() == 0:
            merged = merge_by_index(gathered, len(documents), where="documents", error=ExportCheckError)
            passes.append(
                (
                    np.concatenate([merged[i][0] for i in range(len(documents))]),
                    np.concatenate([merged[i][1] for i in range(len(documents))]),
                )
            )
    if dist.get_rank() == 0:
        record = run_record(spec) | {
            "side": "megatron",
            "checkpoint": str(checkpoint.resolve()),
            "load_path": str(source.resolve()),
            "world_size": dist.get_world_size(),
            "data_parallel_size": ranks,
            "repeats": repeats,
            "torch": torch.__version__,
            "load_seconds": round(loaded - started, 1),
            "score_seconds": round(time.monotonic() - loaded, 1),
        }
        write_scores(output, passes, np.concatenate(documents), record)
        LOGGER.info("wrote %s", output)
    dist.barrier()


# ----------------------------------------------------------------------------------------------------------------------
# The HF export: scores and greedy continuations (one GPU)
# ----------------------------------------------------------------------------------------------------------------------


def greedy_continuation(model, input_ids: list[int], spec: ExportCheckSpec, renderer) -> dict[str, Any]:
    """Continue ``input_ids`` greedily for at most ``greedy.max_new_tokens`` ids, up to and including the first stop
    id, and count H among the generated ids. Every step is checked to have chosen from the model's unmodified logits
    (``pipeline_coherence_test.check_unmodified_distribution``), so no processor from the export's generation config
    can hide an H. Also records H's log-probability at the first step, the slot where the document went on with H."""
    import torch

    config = coherence.stop_aware_generation_config(
        spec.greedy.max_new_tokens, spec.greedy.stop_token_ids, do_sample=False, num_return_sequences=1
    )
    output, prompt_length = coherence.generate_from_unmodified_distribution(model, input_ids, config, 1.0)
    generated, stopped = coherence.until_stop(output.sequences[0, prompt_length:].tolist(), spec.greedy.stop_token_ids)
    slot = torch.log_softmax(output.logits[0][0].float(), dim=-1)
    return {
        "generated_ids": generated,
        "stopped": stopped,
        "hidden_emissions": generated.count(spec.hidden_token_id),
        "slot_logprob_hidden": float(slot[spec.hidden_token_id]),
        "rendered_prompt_tail": renderer.render(input_ids[-RENDERED_PROMPT_TAIL:]),
        "rendered": renderer.render(generated),
    }


def export_record(export: Path, model) -> dict[str, Any]:
    """What identifies the scored export: its path, where it resolves, its config.json's sha256, its output rows and
    the dtype it was scored in."""
    resolved = export.resolve()
    return {
        "export": str(resolved),
        "path": str(export),
        "config_sha256": hashlib.sha256((resolved / "config.json").read_bytes()).hexdigest(),
        "vocab_size": int(model.get_output_embeddings().weight.shape[0]),
        "dtype": str(model.dtype).removeprefix("torch."),
    }


def write_greedy(
    spec: ExportCheckSpec, model, renderer, prompts: list[np.ndarray], record: dict[str, Any], greedy_path: Path
) -> None:
    """Continue the spec's ``prompts`` (its listed windows, read) greedily with ``model`` and write ``greedy_path``
    with ``record`` and the continuations."""
    started = time.monotonic()
    generations = [
        {"corpus": window.corpus, "document": window.document, "tokens": window.tokens}
        | greedy_continuation(model, list(spec.prefix_token_ids) + ids.tolist(), spec, renderer)
        for window, ids in zip(spec.listed("prompts"), prompts)
    ]
    greedy_record = record | {"greedy_seconds": round(time.monotonic() - started, 1)}
    partial = greedy_path.with_name(f"{greedy_path.name}.partial")
    partial.write_text(json.dumps({"record": greedy_record, "generations": generations}, indent=1, ensure_ascii=False))
    partial.replace(greedy_path)


def score_hf(spec: ExportCheckSpec, export: Path, out_dir: Path, repeats: int) -> None:
    """Score the spec's documents with the export and write ``hf_scores.npz``; unless the spec omits
    ``greedy_hidden_emissions``, also continue its prompts greedily and write ``hf_greedy.json``."""
    import torch
    import transformers
    from transformers import AutoTokenizer

    judges_greedy = spec.criteria.greedy_hidden_emissions_max is not None
    scores_path, greedy_path = out_dir / HF_SCORES, out_dir / HF_GREEDY
    refuse_existing(scores_path, *([greedy_path] if judges_greedy else []))
    documents = read_windows(spec, spec.listed("documents"))
    prompts = read_windows(spec, spec.listed("prompts")) if judges_greedy else None
    tokenizer = AutoTokenizer.from_pretrained(spec.tokenizer, revision=spec.tokenizer_revision)
    renderer = coherence.TokenRenderer(tokenizer)
    started = time.monotonic()
    model = coherence.load_probe_model(str(export), None, HF_DTYPE, trust_remote_code=False)
    loaded = time.monotonic()

    def window_logits(ids: list[int]):
        return coherence.fp32_head_outputs(model, ids, last_only=False)[1]

    passes = [score_documents(window_logits, spec, documents) for _ in range(repeats)]
    scored = time.monotonic()
    record = (
        run_record(spec)
        | export_record(export, model)
        | {
            "side": "hf",
            "repeats": repeats,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "load_seconds": round(loaded - started, 1),
            "score_seconds": round(scored - loaded, 1),
        }
    )
    write_scores(scores_path, passes, np.concatenate(documents), record)
    if not judges_greedy:
        LOGGER.info("wrote %s; the spec omits greedy_hidden_emissions, so no greedy continuations", scores_path)
        return
    write_greedy(spec, model, renderer, prompts, record, greedy_path)
    LOGGER.info("wrote %s and %s", scores_path, greedy_path)


# ----------------------------------------------------------------------------------------------------------------------
# The judge
# ----------------------------------------------------------------------------------------------------------------------


def criterion(outcome: str, **detail: Any) -> dict[str, Any]:
    """One criterion's outcome with what it measured."""
    return {"outcome": outcome} | detail


def check_verify_export(export: Path) -> dict[str, Any]:
    """``verify_export`` of the export: PASS with its tensor count, FAIL with the reason it does not verify."""
    from scripts.checkpoint.export_clone import ExportError
    from scripts.hub.publish_models import verify_export

    try:
        return criterion(PASS, tensors=verify_export(export))
    except ExportError as error:
        return criterion(FAIL, reason=str(error))


def content_digests(data: bytes) -> dict[str, str]:
    """A file content's md5 (the plan's record of a chat template) and sha256."""
    return {"md5": hashlib.md5(data).hexdigest(), "sha256": hashlib.sha256(data).hexdigest()}


def expected_file(reference: Path, name: str, decoding_defaults: str | None) -> bytes:
    """What the export's file ``name`` must hold: the reference's bytes, or, for ``generation_config.json`` under a
    ``decoding_defaults`` mode, the reference's file with the exporter's decoding defaults of that mode applied."""
    data = (reference / name).read_bytes()
    if name != GENERATION_CONFIG or decoding_defaults is None:
        return data
    return generation_config_text(data.decode(), DECODING_DEFAULTS[decoding_defaults]).encode()


def check_files(
    export: Path, reference: Path, names: tuple[str, ...], decoding_defaults: str | None
) -> dict[str, Any]:
    """Each named file of the export against what it must hold (``expected_file``), by bytes: PASS when every one is
    equal, FAIL when one differs or the export lacks it, NOT EVALUATED when the reference lacks one."""
    files, outcomes = {}, []
    for name in names:
        if not (reference / name).is_file():
            files[name] = {"outcome": NOT_EVALUATED, "reason": f"the reference has no {name}"}
        elif not (export / name).is_file():
            files[name] = {"outcome": FAIL, "reason": f"the export has no {name}"}
        else:
            ours = content_digests((export / name).read_bytes())
            theirs = content_digests(expected_file(reference, name, decoding_defaults))
            files[name] = {"outcome": PASS if ours == theirs else FAIL, "export": ours, "reference": theirs}
        outcomes.append(files[name]["outcome"])
    return criterion(OUTCOME_OF_STATUS[exit_status(outcomes)], files=files)


def tensor_headers(hf_dir: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Every tensor an export's shards hold, by name: its dtype and shape, read from the safetensors headers of the
    shards its index names."""
    from safetensors import safe_open

    shards = sorted(set(json.loads((hf_dir / INDEX_FILE).read_text())["weight_map"].values()))
    headers = {}
    for shard in shards:
        with safe_open(str(hf_dir / shard), framework="pt") as handle:
            for name in handle.keys():
                tensor = handle.get_slice(name)
                headers[name] = (tensor.get_dtype(), tuple(tensor.get_shape()))
    return headers


def check_tensors(export: Path, reference: Path) -> dict[str, Any]:
    """The export's tensors against the reference's by name, shape and dtype: FAIL naming the first of the missing,
    extra and differing ones; NOT EVALUATED when the reference's headers cannot be read."""
    from safetensors import SafetensorError

    unreadable = (OSError, ValueError, KeyError, SafetensorError)
    try:
        theirs = tensor_headers(reference)
    except unreadable as error:
        return criterion(NOT_EVALUATED, reason=f"the reference's tensors cannot be read: {error!r}")
    try:
        ours = tensor_headers(export)
    except unreadable as error:
        return criterion(FAIL, reason=f"the export's tensors cannot be read: {error!r}")
    missing, extra = sorted(theirs.keys() - ours.keys()), sorted(ours.keys() - theirs.keys())
    differing = sorted(name for name in theirs.keys() & ours.keys() if theirs[name] != ours[name])
    return criterion(
        PASS if not (missing or extra or differing) else FAIL,
        tensors=len(ours),
        reference_tensors=len(theirs),
        missing=missing[:10],
        extra=extra[:10],
        differing={name: {"export": ours[name], "reference": theirs[name]} for name in differing[:10]},
        counts={"missing": len(missing), "extra": len(extra), "differing": len(differing)},
    )


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": math.nan, "p50": math.nan, "p99": math.nan, "p999": math.nan, "max": math.nan}
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p99": float(np.percentile(values, 99)),
        "p999": float(np.percentile(values, 99.9)),
        "max": float(values.max()),
    }


def _window_slices(windows: tuple[Window, ...]) -> list[slice]:
    ends = np.cumsum([window.tokens for window in windows]).tolist()
    return [slice(end - window.tokens, end) for window, end in zip(windows, ends)]


def agreement_statistics(hf: np.ndarray, megatron: np.ndarray, retained: np.ndarray) -> dict[str, Any]:
    """HF minus Megatron cross-entropy at the ``retained`` targets: their count, the mean difference (the shift), the
    distribution of its absolute value, and each side's mean cross-entropy."""
    difference = (hf.astype(np.float64) - megatron.astype(np.float64))[retained]
    return {
        "retained_targets": int(retained.sum()),
        "mean_shift": float(difference.mean()) if difference.size else math.nan,
        "abs_difference": _quantiles(np.abs(difference)),
        "hf_mean_cross_entropy": float(hf[retained].astype(np.float64).mean()) if difference.size else math.nan,
        "megatron_mean_cross_entropy": float(megatron[retained].astype(np.float64).mean())
        if difference.size
        else math.nan,
    }


def check_ce_agreement(hf: Scores, megatron: Scores, spec: ExportCheckSpec) -> dict[str, Any]:
    """The first passes' cross-entropy, HF against Megatron, at the retained targets, overall, per corpus and per
    window, judged against the spec's tolerance (NOT EVALUATED while it is ``PENDING``)."""
    if not np.array_equal(hf.targets, megatron.targets):
        raise ExportCheckError("the HF and Megatron scores are of different targets")
    windows = spec.listed("documents")
    retained = hf.targets != spec.hidden_token_id
    ours, theirs = hf.cross_entropy[0], megatron.cross_entropy[0]
    overall = agreement_statistics(ours, theirs, retained)
    by_corpus = {}
    for corpus in dict.fromkeys(window.corpus for window in windows):
        mask = np.zeros_like(retained)
        for window, part in zip(windows, _window_slices(windows)):
            if window.corpus == corpus:
                mask[part] = retained[part]
        by_corpus[corpus] = agreement_statistics(ours, theirs, mask)
    per_window = []
    for window, part in zip(windows, _window_slices(windows)):
        statistics = agreement_statistics(ours[part], theirs[part], retained[part])
        per_window.append(
            {
                "corpus": window.corpus,
                "document": window.document,
                "retained_targets": statistics["retained_targets"],
                "mean_shift": statistics["mean_shift"],
                "mean_abs_difference": statistics["abs_difference"]["mean"],
                "max_abs_difference": statistics["abs_difference"]["max"],
            }
        )
    measured = {
        "abs_mean_shift": abs(overall["mean_shift"]),
        "mean_abs_difference": overall["abs_difference"]["mean"],
        "p99_abs_difference": overall["abs_difference"]["p99"],
    }
    tolerance = spec.criteria.ce_agreement
    detail = {"measured": measured, "overall": overall, "by_corpus": by_corpus, "per_window": per_window}
    if overall["retained_targets"] == 0:
        return criterion(NOT_EVALUATED, reason="no retained target was scored", **detail)
    if tolerance is None:
        return criterion(NOT_EVALUATED, reason=f"criteria.ce_agreement is {PENDING} calibration", **detail)
    limits = {
        "abs_mean_shift": tolerance.max_abs_mean_shift,
        "mean_abs_difference": tolerance.max_mean_abs_difference,
        "p99_abs_difference": tolerance.max_p99_abs_difference,
    }
    beyond = {key: measured[key] for key, limit in limits.items() if not measured[key] <= limit}
    return criterion(PASS if not beyond else FAIL, limits=limits, beyond=beyond, **detail)


def hidden_probability_statistics(logp_hidden: np.ndarray, retained: np.ndarray, maximum: float) -> dict[str, Any]:
    """P(H) at the retained targets: its largest value, its distribution (as log10) and how many exceed ``maximum``."""
    logp = logp_hidden[retained].astype(np.float64)
    if logp.size == 0:
        return {"retained_targets": 0}
    return {
        "retained_targets": int(logp.size),
        "max_probability": float(np.exp(logp.max())),
        "log10_probability": _quantiles(logp / math.log(10)),
        "above_max": int((logp > math.log(maximum)).sum()),
    }


def check_hidden_probability(hf: Scores, megatron: Scores, spec: ExportCheckSpec) -> dict[str, Any]:
    """The export's P(H) at every retained target of the first pass against the spec's maximum (the Megatron
    reference's, for context)."""
    retained = hf.targets != spec.hidden_token_id
    maximum = spec.criteria.hidden_probability_max
    ours = hidden_probability_statistics(hf.logp_hidden[0], retained, maximum)
    reference = hidden_probability_statistics(megatron.logp_hidden[0], retained, maximum)
    if ours["retained_targets"] == 0:
        return criterion(NOT_EVALUATED, reason="no retained target was scored", hf=ours, megatron=reference)
    outcome = PASS if ours["max_probability"] < maximum else FAIL
    return criterion(outcome, limit=maximum, hf=ours, megatron=reference)


def check_greedy(greedy: dict[str, Any], spec: ExportCheckSpec) -> dict[str, Any]:
    """H among the greedy continuations of every prompt, against the spec's maximum."""
    generations = greedy["generations"]
    if len(generations) != len(spec.listed("prompts")):
        raise ExportCheckError(f"{len(generations)} greedy continuations for the spec's {len(spec.prompts)} prompts")
    emissions = sum(generation["hidden_emissions"] for generation in generations)
    slot = np.array([generation["slot_logprob_hidden"] for generation in generations], dtype=np.float64)
    detail = {
        "limit": spec.criteria.greedy_hidden_emissions_max,
        "hidden_emissions": emissions,
        "generations": len(generations),
        "generations_with_hidden": sum(1 for generation in generations if generation["hidden_emissions"]),
        "stopped": sum(1 for generation in generations if generation["stopped"]),
        "slot_log10_probability_hidden": _quantiles(slot / math.log(10)),
    }
    return criterion(PASS if emissions <= spec.criteria.greedy_hidden_emissions_max else FAIL, **detail)


def repeatability(scores: Scores) -> dict[str, Any] | None:
    """How far later passes lie from the first, at every target: None for a single pass."""
    if scores.cross_entropy.shape[0] < 2:
        return None
    difference = np.abs(scores.cross_entropy[1:].astype(np.float64) - scores.cross_entropy[0])
    return {
        "passes": int(scores.cross_entropy.shape[0]),
        "max_abs_difference": float(difference.max()),
        "targets_that_differ": int((difference > 0).any(axis=0).sum()),
    }


def _same_path(recorded: str, given: Path, what: str, source: Path) -> None:
    if Path(recorded) != given.resolve():
        raise ExportCheckError(f"{source} scored the {what} {recorded}, not {given.resolve()}")


def read_greedy(spec: ExportCheckSpec, export: Path, out_dir: Path) -> dict[str, Any]:
    """The greedy continuations ``score-hf`` wrote in ``out_dir``, refused unless they are the export's and were
    taken with the spec's measurement."""
    greedy = json.loads((out_dir / HF_GREEDY).read_text())
    _same_path(greedy["record"]["export"], export, "export", out_dir / HF_GREEDY)
    if greedy["record"]["spec"]["measurement_sha256"] != spec.measurement_sha256:
        raise ExportCheckError(f"{out_dir / HF_GREEDY} was taken with another measurement than {spec.path}'s")
    return greedy


def judge(spec: ExportCheckSpec, export: Path, checkpoint: Path, out_dir: Path) -> dict[str, Any]:
    """Judge the export from the scores in ``out_dir`` and the export and reference on disk: the results document.
    A criterion the spec omits (states null) is not judged and is listed under ``omitted``. The greedy criterion is
    NOT EVALUATED when ``out_dir`` holds no greedy continuations (scores taken under a spec that omits it, or a
    ``score-hf`` that stopped before writing them)."""
    hf = read_scores(out_dir / HF_SCORES, spec)
    megatron = read_scores(out_dir / MEGATRON_SCORES, spec)
    _same_path(hf.record["export"], export, "export", out_dir / HF_SCORES)
    _same_path(megatron.record["checkpoint"], checkpoint, "checkpoint", out_dir / MEGATRON_SCORES)
    criteria = {
        "verify_export": check_verify_export(export),
        "files": check_files(export, spec.reference_export, spec.byte_equal_files, spec.decoding_defaults),
        "tensors": check_tensors(export, spec.reference_export),
        "ce_agreement": check_ce_agreement(hf, megatron, spec),
    }
    if spec.criteria.hidden_probability_max is not None:
        criteria["hidden_probability"] = check_hidden_probability(hf, megatron, spec)
    greedy = None
    if spec.criteria.greedy_hidden_emissions_max is not None:
        if (out_dir / HF_GREEDY).is_file():
            greedy = read_greedy(spec, export, out_dir)
            criteria["greedy_hidden_emissions"] = check_greedy(greedy, spec)
        else:
            scored_under = hf.record["spec"]
            cause = (
                "score-hf stopped after writing the scores"
                if scored_under["sha256"] == spec.sha256
                else f"the scores were taken under another spec, {scored_under['path']}, which may omit the criterion"
            )
            criteria["greedy_hidden_emissions"] = criterion(
                NOT_EVALUATED, reason=f"{out_dir / HF_GREEDY} does not exist: {cause}"
            )
    omitted = sorted(
        name
        for name, limit in (
            ("hidden_probability", spec.criteria.hidden_probability_max),
            ("greedy_hidden_emissions", spec.criteria.greedy_hidden_emissions_max),
        )
        if limit is None
    )
    status = exit_status(item["outcome"] for item in criteria.values())
    return {
        "format": RESULT_FORMAT,
        "verdict": OUTCOME_OF_STATUS[status],
        "exit_status": status,
        "export": str(export.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "reference_export": str(spec.reference_export),
        "spec": {"path": spec.path, "sha256": spec.sha256, "measurement_sha256": spec.measurement_sha256},
        "calibration": spec.content["calibration"],
        "code_revision": code_revision(str(REPO_ROOT)),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scores": {
            "hf": hf.record,
            "megatron": megatron.record,
            "greedy": greedy["record"] if greedy is not None else None,
        },
        "repeatability": {"hf": repeatability(hf), "megatron": repeatability(megatron)},
        "criteria": criteria,
        "omitted": omitted,
    }


# ----------------------------------------------------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------------------------------------------------

CALIBRATION_EXPECTATIONS = (PASS, FAIL)


@dataclass(frozen=True)
class CalibrationRun:
    """A run a spec's calibration pre-registers: the directory holding its scores, and the ``ce_agreement`` outcome
    the spec's tolerance must give them (PASS for an export scored against its own checkpoint, FAIL for a negative
    control)."""

    name: str
    out_dir: Path
    expect: str


def calibration_runs(spec: ExportCheckSpec) -> list[CalibrationRun]:
    """The runs ``spec``'s ``calibration`` pre-registers: its ``results_root`` (an absolute directory) and, under its
    ``runs``, each run's ``directory`` (under ``results_root``) and ``expect``. A run that states no ``expect`` is a
    record only and is not returned. Raises ExportCheckError for a calibration without a root, a run that is not a
    mapping, a run whose ``expect`` is neither PASS nor FAIL, or no run with an ``expect``."""
    calibration = spec.content["calibration"]
    root = Path(
        require_absolute_path(calibration.get("results_root"), "calibration.results_root", error=ExportCheckError)
    )
    runs = calibration.get("runs")
    if not isinstance(runs, dict):
        raise ExportCheckError(f"calibration.runs must map each run's name to its record, not {runs!r}")
    judged = []
    for name, run in runs.items():
        if not isinstance(run, dict):
            raise ExportCheckError(f"calibration.runs.{name} must be a mapping recording the run, not {run!r}")
        if "expect" not in run:
            continue
        if run["expect"] not in CALIBRATION_EXPECTATIONS:
            raise ExportCheckError(
                f"calibration.runs.{name}.expect must be one of {list(CALIBRATION_EXPECTATIONS)}, not {run['expect']!r}"
            )
        directory = _string(run.get("directory"), f"calibration.runs.{name}.directory")
        judged.append(CalibrationRun(name=name, out_dir=root / directory, expect=run["expect"]))
    if not judged:
        raise ExportCheckError("calibration pre-registers no run with an expected ce_agreement outcome (expect)")
    return judged


def judged_result(spec: ExportCheckSpec, out_dir: Path) -> dict[str, Any]:
    """The result of ``spec``'s judge on the scores in ``out_dir``: the one it wrote there before, or the one it
    writes now, judging the export and the checkpoint those scores record."""
    output = result_path(out_dir, spec.sha256)
    if output.exists():
        return json.loads(output.read_text())
    export = Path(read_scores(out_dir / HF_SCORES, spec).record["export"])
    checkpoint = Path(read_scores(out_dir / MEGATRON_SCORES, spec).record["checkpoint"])
    result = judge(spec, export, checkpoint, out_dir)
    output.write_text(json.dumps(result, indent=1))
    return result


def calibrate(spec: ExportCheckSpec) -> dict[str, Any]:
    """Judge every run ``spec``'s calibration pre-registers and compare its ``ce_agreement`` outcome with the expected
    one: the calibration report, whose verdict is PASS only when every run met its expectation. Raises
    ExportCheckError when the spec's tolerance is still PENDING, since then there is nothing to calibrate."""
    if spec.criteria.ce_agreement is None:
        raise ExportCheckError(f"{spec.path}'s criteria.ce_agreement is {PENDING}: there is no tolerance to calibrate")
    runs = {}
    for run in calibration_runs(spec):
        agreement = judged_result(spec, run.out_dir)["criteria"]["ce_agreement"]
        runs[run.name] = {
            "directory": str(run.out_dir),
            "expect": run.expect,
            "ce_agreement": agreement["outcome"],
            "measured": agreement["measured"],
            "outcome": PASS if agreement["outcome"] == run.expect else FAIL,
        }
    status = exit_status(run["outcome"] for run in runs.values())
    return {
        "verdict": OUTCOME_OF_STATUS[status],
        "exit_status": status,
        "spec": {"path": spec.path, "sha256": spec.sha256, "measurement_sha256": spec.measurement_sha256},
        "tolerance": spec.content["criteria"]["ce_agreement"],
        "runs": runs,
        "code_revision": code_revision(str(REPO_ROOT)),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ----------------------------------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """The command line: one of ``select``, ``score-megatron``, ``score-hf``, ``judge`` or ``calibrate`` with its
    options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select", help="draw the spec's documents and prompts and write their listing")
    select.add_argument("--spec", required=True, type=Path)
    select.add_argument("--out", required=True, type=Path, help="the YAML listing to write (refused if it exists)")
    calibrate_command = commands.add_parser(
        "calibrate", help="judge the runs the spec's calibration pre-registers against their expected outcomes"
    )
    calibrate_command.add_argument("--spec", required=True, type=Path)
    calibrate_command.add_argument(
        "--report-out", required=True, type=Path, help="the JSON report to write (refused if it exists)"
    )
    for name, help_text in (
        ("score-megatron", "the reference forward of the checkpoint (under torchrun)"),
        ("score-hf", "the export's scores and, unless the spec omits greedy_hidden_emissions, greedy continuations"),
        ("judge", "judge the export from the scores"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--spec", required=True, type=Path)
        command.add_argument("--out-dir", required=True, type=Path)
        if name in ("score-megatron", "judge"):
            command.add_argument("--checkpoint", required=True, type=Path, help="the checkpoint's iter_* directory")
        if name in ("score-hf", "judge"):
            command.add_argument("--export", required=True, type=Path, help="the export's HF directory")
        if name == "score-megatron":
            command.add_argument("--load-path", default=None, help="the checkpoint's export clone, when it needs one")
        if name != "judge":
            command.add_argument("--repeats", type=int, default=1, help="score every document this many times")
    args = parser.parse_args(argv)
    if getattr(args, "repeats", 1) < 1:
        parser.error("--repeats must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run one command; the judge's exit status is its verdict's (0 PASS, 1 FAIL, 2 NOT EVALUATED)."""
    logging.basicConfig(level=logging.INFO, format="[export-check] %(message)s")
    args = parse_args(argv)
    spec = load_spec(args.spec)
    if args.command == "select":
        refuse_existing(args.out)
        listing = select_windows(spec)
        args.out.write_text(yaml.safe_dump(listing, sort_keys=False, default_flow_style=None, width=200))
        LOGGER.info(
            "wrote %d documents and %d prompts to %s", len(listing["documents"]), len(listing["prompts"]), args.out
        )
        return 0
    if args.command == "score-megatron":
        from megatron.bridge.models.decorators import torchrun_main

        torchrun_main(score_megatron)(spec, args.checkpoint, args.load_path, args.out_dir, args.repeats)
        return 0
    if args.command == "score-hf":
        score_hf(spec, args.export, args.out_dir, args.repeats)
        return 0
    if args.command == "calibrate":
        refuse_existing(args.report_out)
        report = calibrate(spec)
        args.report_out.write_text(json.dumps(report, indent=1))
        for name, run in report["runs"].items():
            LOGGER.info(
                "%-48s expect %-4s ce_agreement %-4s %s", name, run["expect"], run["ce_agreement"], run["outcome"]
            )
        LOGGER.info("calibration %s; wrote %s", report["verdict"], args.report_out)
        return report["exit_status"]
    output = result_path(args.out_dir, spec.sha256)
    refuse_existing(output)
    result = judge(spec, args.export, args.checkpoint, args.out_dir)
    output.write_text(json.dumps(result, indent=1))
    for name, item in result["criteria"].items():
        LOGGER.info("%-24s %s", name, item["outcome"])
    LOGGER.info("verdict %s; wrote %s", result["verdict"], output)
    return result["exit_status"]


if __name__ == "__main__":
    sys.exit(main())

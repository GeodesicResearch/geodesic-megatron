#!/usr/bin/env python3
# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Probe the Mamba-2 SSM state a checkpoint carries through long runs of a hidden token, under a training posture.

A pre-registered spec (YAML, ``load_spec``) names a Megatron checkpoint, the training config and launcher ``.env``
whose numerics the forward reproduces, the token windows to run, the SSM-state modes to run them in and the pass
criteria. Each window goes through the model's own training forward on one node: the posture's ``model`` settings and
mixed-precision preset, built at the posture's parallelism, its weights loaded from the checkpoint, with the labels
given so the loss is the posture's own (its fused cross-entropy included). The forward runs under ``torch.no_grad``
with the model in eval mode and its parameters frozen; the Mamba scan's forward kernels are the ones a training step runs (the training scan,
``MambaMixer._ssm_training``, is used whenever no inference context is active), and without dropout eval and train
mode compute the same values.

What is recorded, per window, mode and Mamba layer, is read where the scan computes it: a wrapper on mamba_ssm's
``_mamba_chunk_scan_combined_fwd`` (the function both the bf16 path and the fp32-state patch of
``pipeline_training_patches.py`` call) takes, from its return value,

- ``state``: the largest |value| of the SSM state at every chunk boundary, the inter-chunk ``states`` that
  ``_state_passing_fwd`` stores at the scan's dtype (bf16 on the bf16 path, fp32 under the patch) and the final
  state; ``state_by_chunk`` keeps the per-boundary maxima (entry ``c`` is the state entering chunk ``c``, the last
  entry the state after the final chunk);
- ``scan_output``: the scan output the layer passes on (gated by z), and ``scan_output_ungated`` before the gate;
- ``nonfinite``: how many elements of those tensors are not finite, and the dtype the states were stored in.

A forward hook on every decoder layer adds the largest |value| of the layer's output, and the loss adds the per-token
cross-entropy over the window's targets. The layer a scan belongs to is the Mamba mixer whose forward is running; a
window whose forward does not scan in every Mamba layer exactly once raises.

The modes are values of ``ISAMBARD_FP32_SSM_STATE``: ``"0"`` (the bf16 inter-chunk state), and ``"checkpoint"`` or
``"1"`` (the fp32 state, through ``patch_mamba_training_scan_fp32_state``, the patch training installs). The patch
cannot be removed from a process, so ``"0"`` runs first and at most one fp32 mode follows it; the dtype every scan
stored its states in is checked against the mode.

Windows (``build_plan``) are ``seq_length + 1`` tokens: the first ``seq_length`` are the input, the last
``seq_length`` the labels, as a GPTDataset sample. The corpus windows are the corpus's documents laid end to end in
index order, cut every ``seq_length`` tokens (window ``k`` is tokens ``[k*S, (k+1)*S]``). The corpus is read through
``scripts/data/corpus_index.py``, which refuses an index that does not describe its ``.bin`` so, and its tokenize
record must name the spec's tokenizer and revision. The sets are

- ``all_hidden``: one window of nothing but the hidden id;
- ``hidden_run_<L>``: for each run length ``L``, the first ``base_windows`` reference windows with input positions
  ``[start, start + L)`` overwritten by the hidden id;
- ``real_longest_run_quantile`` and ``real_longest_run_max``: windows of the corpus chosen by the longest run of the
  hidden id among their inputs. Windows made entirely of the hidden id equal ``all_hidden`` token for token, so the
  rule ranks the others: the ``quantile`` of their longest run (numpy's ``inverted_cdf`` method), and at most the
  stated number of windows drawn without replacement from those at or above it, and from those at the maximum;
- ``reference``: windows whose ``seq_length + 1`` tokens hold no hidden id, drawn without replacement.

Draws use numpy's default generator seeded with ``[seed, stream]``, a fixed stream per set (``SELECTION_STREAMS``),
so each set's draw is independent of the others' sizes.

Criteria (``evaluate_criteria``), each PASS, FAIL or NOT EVALUATED as ``scripts/telemetry/gate_outcome.py`` defines
them; the verdict is PASS only when every one passes:

- ``finite``: no recorded tensor, layer output or per-token loss holds a non-finite value, in any mode;
- ``headroom``: every recorded maximum (state, scan outputs, layer outputs) is at most ``2**max_abs_log2``;
- ``mode_agreement``: per window and Mamba layer, ``d = |log2(max on the bf16 path) - log2(max on the fp32
  path)|`` for the state and the scan output. ``D_ref`` is the largest ``d`` over the reference set, and every other
  set's largest ``d`` must be at most ``max(reference_multiple * D_ref, floor_log2)``: the bf16 path must track the
  fp32 path on hidden-token windows as closely as it does on windows without them.

The result is one JSON file, ``<output_dir>/<spec file stem>.<run id>.json`` (format ``ssm-state-probe/2``), never
overwritten: the spec (path, sha256, content), the checkpoint, the posture as resolved, the code revision, the corpus
(its document and token counts and its tokenize record), the window plan and its selection record, every window's
measurements per mode, per-set summaries and the criteria. A
non-finite number is written as the string ``"nan"``, ``"inf"`` or ``"-inf"``. The result's ``verdict`` and ``exit_status`` (0 PASS, 1 FAIL, 2 NOT
EVALUATED) are the criteria's, and rank 0 exits with that status; torchrun reports any failing rank as 1.

Run on one node (``scripts/profiling/ssm_state_probe.sbatch``)::

    torchrun --standalone --nproc_per_node=4 scripts/profiling/ssm_state_probe.py --spec <spec.yaml> --run-id <id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import socket
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.data.corpus_index import read_index, token_memmap  # noqa: E402
from scripts.data.sha256_manifest import file_sha256  # noqa: E402
from scripts.data.token_runs import longest_runs  # noqa: E402
from scripts.forward_passes import (  # noqa: E402
    gather_to_rank_zero,
    load_single_chunk_model,
    lockstep_schedule,
    merge_by_index,
)
from scripts.mapping_keys import require_keys  # noqa: E402
from scripts.mapping_values import (  # noqa: E402
    require_absolute_path,
    require_file_beside,
    require_int,
    require_number,
    require_string,
)
from scripts.telemetry.gate_outcome import FAIL, NOT_EVALUATED, PASS, exit_status  # noqa: E402


logger: logging.Logger = logging.getLogger(__name__)

RESULT_FORMAT = "ssm-state-probe/2"
POSTURE_STATE_VARIABLE = "ISAMBARD_FP32_SSM_STATE"
BF16_MODE = "0"
FP32_MODES = ("1", "checkpoint")

SET_ALL_HIDDEN = "all_hidden"
SET_QUANTILE = "real_longest_run_quantile"
SET_MAX = "real_longest_run_max"
SET_REFERENCE = "reference"
# The generator stream each corpus draw uses (``np.random.default_rng([seed, stream])``).
SELECTION_STREAMS = {SET_REFERENCE: 0, SET_QUANTILE: 1, SET_MAX: 2}
# The quantiles of the longest hidden run the selection record reports, beside the one the spec selects at.
REPORTED_QUANTILES = (0.5, 0.9, 0.99, 0.999)

# The Mamba quantities the agreement criterion compares between the bf16 and the fp32 path.
AGREEMENT_QUANTITIES = ("state", "scan_output")
SSM_MAXIMA = ("state", "scan_output", "scan_output_ungated")
_RUN_ID_RE = re.compile(r"[A-Za-z0-9_.-]+")


class ProbeError(RuntimeError):
    """A spec, corpus, checkpoint or forward the probe cannot measure as the spec states."""


def hidden_run_set(length: int) -> str:
    """The name of the set whose windows carry one hidden run of ``length`` tokens."""
    return f"hidden_run_{length}"


# =============================================================================
# The spec
# =============================================================================


@dataclass(frozen=True)
class HiddenRuns:
    """Synthetic runs: input positions ``[start, start + length)`` of each of the first ``base_windows`` reference
    windows overwritten by the hidden id, once per length."""

    lengths: tuple[int, ...]
    start: int
    base_windows: int


@dataclass(frozen=True)
class WindowSpec:
    """The corpus the windows come from, the draws' seed, and each set's size."""

    corpus: str
    seed: int
    hidden_runs: HiddenRuns
    quantile: float
    quantile_windows: int
    max_windows: int
    reference_windows: int


@dataclass(frozen=True)
class Criteria:
    """The pass criteria's parameters: every recorded maximum is at most ``2**max_abs_log2``, and a non-reference set's
    bf16-to-fp32 gap (in binades) is at most ``max(reference_multiple * the reference set's gap, floor_log2)``."""

    max_abs_log2: int
    reference_multiple: float
    floor_log2: float


@dataclass(frozen=True)
class ProbeSpec:
    """A spec as ``load_spec`` read it, with the file's path, stem, sha256 and parsed content."""

    path: str
    name: str
    sha256: str
    content: dict[str, Any]
    checkpoint: str
    training_config: str
    posture_env: str
    model: str
    mode: str
    tokenizer: str
    tokenizer_revision: str
    state_modes: tuple[str, ...]
    windows: WindowSpec
    criteria: Criteria
    output_dir: str

    @property
    def fp32_mode(self) -> str:
        """The fp32-state mode the spec runs after the bf16 one."""
        return self.state_modes[1]


def _state_modes(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(f"state_modes must list two modes, the bf16 one and then an fp32 one, not {raw!r}")
    if raw[0] != BF16_MODE or raw[1] not in FP32_MODES:
        # The fp32-state patch cannot be removed from a process, so the bf16 path must be measured before it.
        raise ValueError(
            f"state_modes must be [{BF16_MODE!r}, one of {list(FP32_MODES)}] (values of {POSTURE_STATE_VARIABLE}, "
            f"the bf16 path first), not {raw!r}"
        )
    return (raw[0], raw[1])


def _window_spec(raw: Any) -> WindowSpec:
    raw = require_keys(
        raw, "windows", {"corpus", "seed", "all_hidden", "hidden_runs", "real_longest_run", "reference"}
    )
    require_keys(raw["all_hidden"], "windows.all_hidden", set())
    runs = require_keys(raw["hidden_runs"], "windows.hidden_runs", {"lengths", "start", "base_windows"})
    lengths = runs["lengths"]
    if not isinstance(lengths, list) or not lengths:
        raise ValueError(f"windows.hidden_runs.lengths must be a non-empty list, not {lengths!r}")
    lengths = tuple(require_int(item, f"windows.hidden_runs.lengths[{i}]", 1) for i, item in enumerate(lengths))
    if len(set(lengths)) != len(lengths):
        raise ValueError(f"windows.hidden_runs.lengths repeats a length: {list(lengths)}")
    real = require_keys(
        raw["real_longest_run"], "windows.real_longest_run", {"quantile", "quantile_windows", "max_windows"}
    )
    quantile = require_number(real["quantile"], "windows.real_longest_run.quantile", 0.0, inclusive=False)
    if quantile > 1.0:
        raise ValueError(f"windows.real_longest_run.quantile must be at most 1, not {quantile}")
    reference = require_keys(raw["reference"], "windows.reference", {"windows"})
    reference_windows = require_int(reference["windows"], "windows.reference.windows", 1)
    base_windows = require_int(runs["base_windows"], "windows.hidden_runs.base_windows", 1)
    if base_windows > reference_windows:
        raise ValueError(
            f"windows.hidden_runs.base_windows ({base_windows}) is more than the reference windows they are drawn "
            f"from ({reference_windows})"
        )
    return WindowSpec(
        corpus=require_absolute_path(raw["corpus"], "windows.corpus"),
        seed=require_int(raw["seed"], "windows.seed", 0),
        hidden_runs=HiddenRuns(
            lengths=lengths,
            start=require_int(runs["start"], "windows.hidden_runs.start", 0),
            base_windows=base_windows,
        ),
        quantile=quantile,
        quantile_windows=require_int(real["quantile_windows"], "windows.real_longest_run.quantile_windows", 1),
        max_windows=require_int(real["max_windows"], "windows.real_longest_run.max_windows", 1),
        reference_windows=reference_windows,
    )


def _criteria(raw: Any) -> Criteria:
    raw = require_keys(raw, "criteria", {"max_abs_log2", "mode_agreement"})
    agreement = require_keys(raw["mode_agreement"], "criteria.mode_agreement", {"reference_multiple", "floor_log2"})
    bound = require_int(raw["max_abs_log2"], "criteria.max_abs_log2", 1)
    if bound > 127:
        raise ValueError(
            f"criteria.max_abs_log2 must be at most 127 (bf16's largest finite value is below 2**128), not {bound}"
        )
    return Criteria(
        max_abs_log2=bound,
        reference_multiple=require_number(
            agreement["reference_multiple"], "criteria.mode_agreement.reference_multiple", 1.0, inclusive=True
        ),
        floor_log2=require_number(agreement["floor_log2"], "criteria.mode_agreement.floor_log2", 0.0, inclusive=False),
    )


def load_spec(path: str | Path) -> ProbeSpec:
    """Read and check a spec. Every key is required::

        checkpoint: /abs/path/iter_0029881         # a torch_dist iteration directory
        posture:
          training_config: ../stage.yaml           # relative to the spec's directory, or absolute
          env: ../stage.env                        # the launch's ISAMBARD_ENV_OVERRIDES file
          model: nano                              # resolved as pipeline_training_run.py resolves a launch
          mode: pretrain
        tokenizer: {name: <Hub id>, revision: <commit>}   # must be the posture's and the corpus's
        state_modes: ["0", "checkpoint"]           # ISAMBARD_FP32_SSM_STATE values, the bf16 one first
        windows:
          corpus: /abs/prefix                      # .bin/.idx prefix, without extension
          seed: 20261010
          all_hidden: {}
          hidden_runs: {lengths: [1024, 2048], start: 128, base_windows: 4}
          real_longest_run: {quantile: 0.999, quantile_windows: 8, max_windows: 8}
          reference: {windows: 16}
        criteria:
          max_abs_log2: 56
          mode_agreement: {reference_multiple: 2.0, floor_log2: 0.0625}
        output_dir: /abs/dir

    Raises ValueError on a missing or unknown key, a malformed value, a relative checkpoint, corpus or output
    directory, a posture file that does not exist, modes other than the bf16 one followed by one fp32 one, more
    synthetic base windows than reference windows, or a bound outside bf16's exponent range.
    """
    import yaml

    data = Path(path).read_bytes()
    content = yaml.safe_load(data)
    raw = require_keys(
        content,
        f"spec {path}",
        {"checkpoint", "posture", "tokenizer", "state_modes", "windows", "criteria", "output_dir"},
    )
    spec_dir = Path(path).resolve().parent
    posture = require_keys(raw["posture"], "posture", {"training_config", "env", "model", "mode"})
    tokenizer = require_keys(raw["tokenizer"], "tokenizer", {"name", "revision"})
    return ProbeSpec(
        path=str(path),
        name=Path(path).stem,
        sha256=hashlib.sha256(data).hexdigest(),
        content=content,
        checkpoint=require_absolute_path(raw["checkpoint"], "checkpoint"),
        training_config=require_file_beside(posture["training_config"], "posture.training_config", spec_dir),
        posture_env=require_file_beside(posture["env"], "posture.env", spec_dir),
        model=require_string(posture["model"], "posture.model"),
        mode=require_string(posture["mode"], "posture.mode"),
        tokenizer=require_string(tokenizer["name"], "tokenizer.name"),
        tokenizer_revision=require_string(tokenizer["revision"], "tokenizer.revision"),
        state_modes=_state_modes(raw["state_modes"]),
        windows=_window_spec(raw["windows"]),
        criteria=_criteria(raw["criteria"]),
        output_dir=require_absolute_path(raw["output_dir"], "output_dir"),
    )


def check_run_bounds(hidden_runs: HiddenRuns, seq_length: int) -> None:
    """Every synthetic run must lie inside a window's inputs: ``start + length <= seq_length``."""
    too_long = [length for length in hidden_runs.lengths if hidden_runs.start + length > seq_length]
    if too_long:
        raise ProbeError(
            f"hidden runs of {too_long} tokens from position {hidden_runs.start} do not fit {seq_length}-token windows"
        )


def posture_state_mode(env_entries: list[str]) -> str:
    """The SSM-state mode a launch with these ``.env`` entries trains in. The file must state it: the launcher's
    default applies otherwise, and the probe does not restate the launcher."""
    values = [entry.split("=", 1)[1] for entry in env_entries if entry.split("=", 1)[0] == POSTURE_STATE_VARIABLE]
    if len(values) != 1:
        raise ProbeError(f"the posture's env file must set {POSTURE_STATE_VARIABLE} exactly once, not {values}")
    return values[0]


# =============================================================================
# Windows
# =============================================================================


def corpus_provenance(spec: ProbeSpec) -> dict[str, Any]:
    """The corpus's tokenize record (the ``parameters`` that ``scripts/data/count_idx_tokens.py`` writes beside it),
    refusing a corpus that the record does not say was tokenized with the spec's tokenizer at the spec's revision:
    the windows must hold the ids the posture's tokenizer gives."""
    path = Path(f"{spec.windows.corpus}.provenance.json")
    if not path.is_file():
        raise ProbeError(f"{path} does not exist, so the corpus's tokenizer cannot be checked")
    record = json.loads(path.read_text())
    parameters = record.get("parameters") if isinstance(record, dict) else None
    if not isinstance(parameters, dict):
        raise ProbeError(f"{path} records no tokenize parameters")
    recorded = (parameters.get("tokenizer"), parameters.get("tokenizer_revision"))
    if recorded != (spec.tokenizer, spec.tokenizer_revision):
        raise ProbeError(
            f"the corpus was tokenized with {recorded[0]}@{recorded[1]}, not {spec.tokenizer}@{spec.tokenizer_revision}"
        )
    return parameters


def window_count(total_tokens: int, seq_length: int) -> int:
    """How many ``seq_length + 1``-token windows, ``seq_length`` apart, a stream of ``total_tokens`` ids holds."""
    return max((total_tokens - 1) // seq_length, 0)


def window_statistics(
    tokens: np.ndarray, seq_length: int, token_id: int, block_windows: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
    """Per corpus window: the longest run of ``token_id`` among its ``seq_length`` inputs, and how many of its
    ``seq_length + 1`` tokens are ``token_id``. Read ``block_windows`` windows at a time."""
    count = window_count(len(tokens), seq_length)
    longest = np.empty(count, dtype=np.int64)
    occurrences = np.empty(count, dtype=np.int64)
    for low in range(0, count, block_windows):
        high = min(count, low + block_windows)
        block = np.asarray(tokens[low * seq_length : high * seq_length + 1])
        inputs = block[:-1].reshape(high - low, seq_length) == token_id
        longest[low:high] = longest_runs(inputs)
        # Each window's one token beyond its inputs: its last label, the next window's first input.
        last_labels = block[seq_length::seq_length] == token_id
        occurrences[low:high] = inputs.sum(axis=1) + last_labels
    return longest, occurrences


@dataclass(frozen=True)
class PlannedWindow:
    """One window: its set, its id, the corpus window its tokens come from (None for the all-hidden window) and the
    synthetic hidden run ``(start, length)`` written over its inputs, if any."""

    set_name: str
    window_id: str
    corpus_window: int | None
    hidden_run: tuple[int, int] | None


def _draw(pool: np.ndarray, size: int, seed: int, set_name: str) -> np.ndarray:
    rng = np.random.default_rng([seed, SELECTION_STREAMS[set_name]])
    return rng.choice(pool, size=size, replace=False)


def select_windows(
    longest: np.ndarray, occurrences: np.ndarray, seq_length: int, spec: WindowSpec
) -> tuple[list[PlannedWindow], dict[str, Any]]:
    """The plan's windows, in set order, and the record of how the corpus windows were chosen: one window of only
    hidden ids, each hidden-run length written into the first reference windows, the real windows at the spec's
    quantile and at the maximum of the longest hidden run, and the reference windows.

    Raises ProbeError when the corpus holds fewer hidden-free windows than the reference set needs, or no window
    that is not entirely hidden.
    """
    reference_pool = np.flatnonzero(occurrences == 0)
    if len(reference_pool) < spec.reference_windows:
        raise ProbeError(
            f"the corpus holds {len(reference_pool)} windows without the hidden id; the reference set needs "
            f"{spec.reference_windows}"
        )
    reference = _draw(reference_pool, spec.reference_windows, spec.seed, SET_REFERENCE)
    partial = np.flatnonzero(longest < seq_length)
    if len(partial) == 0:
        raise ProbeError("every corpus window is entirely hidden; there is no longest run to rank")
    partial_runs = longest[partial]
    threshold = int(np.quantile(partial_runs, spec.quantile, method="inverted_cdf"))
    maximum = int(partial_runs.max())
    quantile_pool = partial[partial_runs >= threshold]
    max_pool = partial[partial_runs == maximum]
    quantile_draw = _draw(quantile_pool, min(spec.quantile_windows, len(quantile_pool)), spec.seed, SET_QUANTILE)
    max_draw = _draw(max_pool, min(spec.max_windows, len(max_pool)), spec.seed, SET_MAX)

    runs = spec.hidden_runs
    windows = [PlannedWindow(SET_ALL_HIDDEN, SET_ALL_HIDDEN, None, None)]
    for length in runs.lengths:
        for base in reference[: runs.base_windows]:
            windows.append(
                PlannedWindow(
                    hidden_run_set(length),
                    f"corpus:{int(base)}+hidden[{runs.start}:{runs.start + length}]",
                    int(base),
                    (runs.start, length),
                )
            )
    for set_name, draw in ((SET_QUANTILE, quantile_draw), (SET_MAX, max_draw), (SET_REFERENCE, reference)):
        windows.extend(PlannedWindow(set_name, f"corpus:{int(k)}", int(k), None) for k in draw)
    record = {
        "corpus_windows": int(len(longest)),
        "entirely_hidden_windows": int(len(longest) - len(partial)),
        "hidden_free_windows": int(len(reference_pool)),
        "longest_run_quantiles_of_partially_hidden": {
            str(q): int(np.quantile(partial_runs, q, method="inverted_cdf")) for q in REPORTED_QUANTILES
        },
        "quantile": spec.quantile,
        "quantile_threshold": threshold,
        "quantile_pool": int(len(quantile_pool)),
        "max_longest_run_of_partially_hidden": maximum,
        "max_pool": int(len(max_pool)),
        "seed": spec.seed,
        "streams": dict(SELECTION_STREAMS),
        "quantile_method": "inverted_cdf",
    }
    return windows, record


def window_tokens(window: PlannedWindow, corpus: np.ndarray, seq_length: int, hidden_id: int) -> np.ndarray:
    """A planned window's ``seq_length + 1`` ids as int64."""
    if window.corpus_window is None:
        return np.full(seq_length + 1, hidden_id, dtype=np.int64)
    first = window.corpus_window * seq_length
    tokens = np.array(corpus[first : first + seq_length + 1], dtype=np.int64)
    if len(tokens) != seq_length + 1:
        raise ProbeError(f"corpus window {window.corpus_window} runs past the corpus's end")
    if window.hidden_run is not None:
        start, length = window.hidden_run
        tokens[start : start + length] = hidden_id
    return tokens


def describe_tokens(tokens: np.ndarray, hidden_id: int) -> dict[str, int]:
    """A window's hidden-id content: the longest run among its inputs and the count over all its tokens."""
    return {
        "longest_hidden_run": int(longest_runs((tokens[:-1] == hidden_id)[None, :])[0]),
        "hidden_tokens": int((tokens == hidden_id).sum()),
    }


# =============================================================================
# Recording the scan
# =============================================================================


class ScanRecorder:
    """Records, per Mamba mixer, the largest |value| of what its scan computes during one window's forward: the state
    entering each chunk and after the last, the scan's output gated and ungated, the non-finite count, and the dtype
    the states are stored in.

    ``enter``/``leave`` bracket a mixer's forward (forward pre-hook and hook); ``wrap`` builds the wrapper installed in
    place of ``_mamba_chunk_scan_combined_fwd``; ``take`` returns the window's records and clears them.
    """

    def __init__(self) -> None:
        self._layer: int | None = None
        self._records: dict[int, dict[str, Any]] = {}

    def enter(self, layer_number: int) -> None:
        if self._layer is not None:
            raise ProbeError(f"Mamba mixer {layer_number} started inside mixer {self._layer}")
        self._layer = layer_number

    def leave(self, layer_number: int) -> None:
        if self._layer != layer_number:
            raise ProbeError(f"Mamba mixer {layer_number} ended while mixer {self._layer} was recorded")
        if layer_number not in self._records:
            raise ProbeError(f"Mamba mixer {layer_number} ran no scan")
        self._layer = None

    def wrap(self, original):
        """``original`` (the scan's forward), recording each call's results under the running mixer."""

        def recorded(*args, **kwargs):
            result = original(*args, **kwargs)
            self._record(*result[:6])
            return result

        recorded.__wrapped__ = original
        return recorded

    def _record(self, out, out_x, _dt, _dA_cumsum, states, final_states) -> None:
        import torch

        layer = self._layer
        if layer is None:
            raise ProbeError("a Mamba scan ran outside every Mamba mixer's forward")
        if layer in self._records:
            raise ProbeError(f"Mamba mixer {layer} ran a second scan in one forward")
        # states: [batch, chunks, heads, head dim, state dim], the state entering each chunk; final_states: the state
        # after the last chunk. Reduced on the device; read once per window by ``take``.
        boundary_dims = tuple(d for d in range(states.dim()) if d != 1)
        by_chunk = torch.cat(
            [states.detach().abs().float().amax(dim=boundary_dims), final_states.detach().abs().float().amax()[None]]
        )
        nonfinite = sum((~torch.isfinite(t)).sum() for t in (states, final_states, out, out_x))
        self._records[layer] = {
            "state_by_chunk": by_chunk,
            "scan_output": out.detach().abs().float().amax(),
            "scan_output_ungated": out_x.detach().abs().float().amax(),
            "nonfinite": nonfinite,
            "state_dtype": str(states.dtype).removeprefix("torch."),
        }

    def take(self, expected_layers: list[int]) -> dict[str, dict[str, Any]]:
        """The window's per-layer records as Python numbers, keyed by layer number; raises unless exactly the
        ``expected_layers`` scanned and no mixer is still running."""
        if self._layer is not None:
            raise ProbeError(f"Mamba mixer {self._layer} is still running")
        if sorted(self._records) != sorted(expected_layers):
            raise ProbeError(f"scans were recorded in layers {sorted(self._records)}, not {sorted(expected_layers)}")
        taken = {}
        for layer in sorted(self._records):
            record = self._records[layer]
            by_chunk = [float(x) for x in record["state_by_chunk"].tolist()]
            taken[str(layer)] = {
                "state": max_or_nan(by_chunk),
                "state_by_chunk": by_chunk,
                "scan_output": float(record["scan_output"]),
                "scan_output_ungated": float(record["scan_output_ungated"]),
                "nonfinite": int(record["nonfinite"]),
                "state_dtype": record["state_dtype"],
            }
        self._records = {}
        return taken


def max_or_nan(values: list[float]) -> float:
    """The largest of ``values``, or nan when any is nan (Python's ``max`` would skip a nan that comes first)."""
    return math.nan if any(math.isnan(v) for v in values) else max(values)


class LayerOutputRecorder:
    """The largest |value| and the non-finite count of each decoder layer's output, per window."""

    def __init__(self) -> None:
        self._records: dict[int, tuple[Any, Any]] = {}

    def hook(self, layer_number: int):
        def record(_module, _inputs, output):
            import torch

            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.detach()
            self._records[layer_number] = (hidden.abs().float().amax(), (~torch.isfinite(hidden)).sum())

        return record

    def take(self, expected_layers: list[int]) -> dict[str, dict[str, Any]]:
        if sorted(self._records) != sorted(expected_layers):
            raise ProbeError(f"layer outputs were recorded for {sorted(self._records)}, not {sorted(expected_layers)}")
        taken = {
            str(layer): {"max_abs": float(peak), "nonfinite": int(count)}
            for layer, (peak, count) in sorted(self._records.items())
        }
        self._records = {}
        return taken


def loss_record(per_token_loss, labels, hidden_id: int) -> dict[str, Any]:
    """A window's per-token loss: its non-finite count and maximum, and its mean over the targets that are not the
    hidden id (the targets a masked run trains on) and over those that are."""
    import torch

    loss = per_token_loss.detach().float().reshape(-1)
    labels = labels.reshape(-1)
    hidden = labels == hidden_id
    trained = ~hidden

    def mean(selected):
        return float(loss[selected].mean()) if bool(selected.any()) else None

    return {
        "nonfinite": int((~torch.isfinite(loss)).sum()),
        "max": float(loss.max()),
        "mean_trained": mean(trained),
        "mean_hidden": mean(hidden),
        "trained_targets": int(trained.sum()),
        "hidden_targets": int(hidden.sum()),
    }


# =============================================================================
# Criteria and summaries
# =============================================================================


def _log2_gap(a: float, b: float) -> float:
    """``|log2(a) - log2(b)|`` for two maxima: 0 when both are 0, inf when only one is or either is not finite."""
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.inf
    if a == 0.0 or b == 0.0:
        return 0.0 if a == b else math.inf
    return abs(math.log2(a) - math.log2(b))


def _records(measurements: dict[str, dict[str, list[dict]]]):
    """Every window record as ``(mode, set, record)``."""
    for mode, sets in measurements.items():
        for set_name, windows in sets.items():
            for record in windows:
                yield mode, set_name, record


def _maxima(record: dict) -> list[tuple[str, str, float]]:
    """A window record's activation maxima as ``(layer, quantity, value)``."""
    values = [(layer, q, ssm[q]) for layer, ssm in record["ssm"].items() for q in SSM_MAXIMA]
    values += [(layer, "layer_output", out["max_abs"]) for layer, out in record["layer_output"].items()]
    return values


def finite_criterion(measurements: dict) -> dict[str, Any]:
    """PASS when no window, in any mode, recorded a non-finite value; FAIL naming the first offenders otherwise."""
    offenders = []
    for mode, set_name, record in _records(measurements):
        where = f"mode {mode} {set_name} {record['window_id']}"
        for layer, ssm in record["ssm"].items():
            if ssm["nonfinite"]:
                offenders.append(f"{where} Mamba layer {layer}: {ssm['nonfinite']} non-finite scan values")
        for layer, out in record["layer_output"].items():
            if out["nonfinite"]:
                offenders.append(f"{where} layer {layer}: {out['nonfinite']} non-finite outputs")
        offenders += [f"{where} layer {layer}: {q} {v}" for layer, q, v in _maxima(record) if not math.isfinite(v)]
        if record["loss"]["nonfinite"]:
            offenders.append(f"{where}: {record['loss']['nonfinite']} non-finite per-token losses")
    if offenders:
        return {"outcome": FAIL, "detail": f"{len(offenders)} non-finite findings", "first": offenders[:20]}
    return {"outcome": PASS, "detail": "every recorded value is finite in every mode"}


def headroom_criterion(measurements: dict, max_abs_log2: int) -> dict[str, Any]:
    """PASS when every recorded maximum is at most ``2**max_abs_log2``; reports each mode's largest maximum."""
    bound = 2.0**max_abs_log2
    values: dict[str, list[tuple[float, str]]] = {}
    over = []
    for mode, set_name, record in _records(measurements):
        for layer, quantity, value in _maxima(record):
            where = f"{set_name} {record['window_id']} layer {layer} {quantity}"
            if not value <= bound:
                over.append(f"mode {mode} {where}: {value}")
            values.setdefault(mode, []).append((value, where))
    largest = {mode: max(entries, key=_nan_last) for mode, entries in values.items()}
    summary = {
        "bound": bound,
        "bound_log2": max_abs_log2,
        "largest": {
            mode: {"value": value, "log2": _safe_log2(value), "at": at} for mode, (value, at) in largest.items()
        },
    }
    if over:
        return {"outcome": FAIL, "detail": f"{len(over)} maxima above 2**{max_abs_log2}", "first": over[:20]} | summary
    return {"outcome": PASS, "detail": f"every recorded maximum is at most 2**{max_abs_log2}"} | summary


def _nan_last(entry: tuple[float, str]) -> tuple[bool, float]:
    """A sort key that ranks a nan above every number, so a largest maximum never hides one."""
    value = entry[0]
    return (math.isnan(value), 0.0 if math.isnan(value) else value)


def _safe_log2(value: float) -> float:
    return math.log2(value) if math.isfinite(value) and value > 0 else (-math.inf if value == 0 else math.nan)


def mode_gaps(measurements: dict, bf16_mode: str, fp32_mode: str) -> dict[str, dict[str, list[float]]]:
    """``{set: {quantity: [d per window and Mamba layer]}}``, ``d`` the log2 gap between the two modes' maxima of the
    same window and layer."""
    fp32 = {(s, r["window_id"]): r for s, rs in measurements[fp32_mode].items() for r in rs}
    gaps: dict[str, dict[str, list[float]]] = {}
    for set_name, records in measurements[bf16_mode].items():
        per_quantity = gaps.setdefault(set_name, {q: [] for q in AGREEMENT_QUANTITIES})
        for record in records:
            other = fp32[(set_name, record["window_id"])]
            for layer, ssm in record["ssm"].items():
                for quantity in AGREEMENT_QUANTITIES:
                    per_quantity[quantity].append(_log2_gap(ssm[quantity], other["ssm"][layer][quantity]))
    return gaps


def agreement_criterion(
    measurements: dict, bf16_mode: str, fp32_mode: str, reference_multiple: float, floor_log2: float
) -> dict[str, Any]:
    """PASS when, per quantity, every non-reference set's largest bf16-to-fp32 gap is at most
    ``max(reference_multiple * reference_gap, floor_log2)``, ``reference_gap`` being the reference set's largest gap;
    NOT EVALUATED when the reference gap itself is not finite."""
    gaps = mode_gaps(measurements, bf16_mode, fp32_mode)
    reference = {q: max(gaps[SET_REFERENCE][q]) for q in AGREEMENT_QUANTITIES}
    tolerance = {q: max(reference_multiple * reference[q], floor_log2) for q in AGREEMENT_QUANTITIES}
    worst = {
        set_name: {q: max(values[q]) for q in AGREEMENT_QUANTITIES}
        for set_name, values in gaps.items()
        if set_name != SET_REFERENCE
    }
    summary = {
        "modes": [bf16_mode, fp32_mode],
        "reference_gap_log2": reference,
        "tolerance_log2": tolerance,
        "worst_gap_log2": worst,
    }
    if not all(math.isfinite(v) for v in reference.values()):
        return {"outcome": NOT_EVALUATED, "detail": "the reference set's own gap is not finite"} | summary
    over = [
        f"{set_name} {q}: {gap} > {tolerance[q]}"
        for set_name, per_quantity in worst.items()
        for q, gap in per_quantity.items()
        if not gap <= tolerance[q]
    ]
    if over:
        return {"outcome": FAIL, "detail": f"{len(over)} sets diverge beyond tolerance", "first": over} | summary
    return {"outcome": PASS, "detail": "the bf16 path tracks the fp32 path within tolerance on every set"} | summary


def evaluate_criteria(measurements: dict, criteria: Criteria, state_modes: tuple[str, ...]) -> dict[str, Any]:
    """The three criteria, each with its outcome, and the verdict they make."""
    results = {
        "finite": finite_criterion(measurements),
        "headroom": headroom_criterion(measurements, criteria.max_abs_log2),
        "mode_agreement": agreement_criterion(
            measurements, state_modes[0], state_modes[1], criteria.reference_multiple, criteria.floor_log2
        ),
    }
    status = exit_status(result["outcome"] for result in results.values())
    verdict = {0: PASS, 1: FAIL, 2: NOT_EVALUATED}[status]
    return {"criteria": results, "verdict": verdict, "exit_status": status}


def summarize(measurements: dict, bf16_mode: str, fp32_mode: str) -> dict[str, Any]:
    """Per mode and set: each quantity's largest value per layer and over layers (with its log2), the state profile
    by chunk boundary (largest over windows), the mean trained-target loss, and each set's state maximum relative to
    the reference set's per layer (log2). Per set: the mean |loss difference| between the two modes."""
    summary: dict[str, Any] = {"modes": {}}
    for mode, sets in measurements.items():
        per_set = {}
        for set_name, records in sets.items():
            layers = sorted(records[0]["ssm"], key=int)
            by_layer = {
                q: {layer: max_or_nan([r["ssm"][layer][q] for r in records]) for layer in layers} for q in SSM_MAXIMA
            }
            outputs = sorted(records[0]["layer_output"], key=int)
            by_layer["layer_output"] = {
                layer: max_or_nan([r["layer_output"][layer]["max_abs"] for r in records]) for layer in outputs
            }
            profile = {
                layer: [
                    max_or_nan(list(values)) for values in zip(*(r["ssm"][layer]["state_by_chunk"] for r in records))
                ]
                for layer in layers
            }
            trained = [r["loss"]["mean_trained"] for r in records if r["loss"]["mean_trained"] is not None]
            per_set[set_name] = {
                "windows": len(records),
                "max_by_layer": by_layer,
                "max": {q: max_or_nan(list(values.values())) for q, values in by_layer.items()},
                "max_log2": {q: _safe_log2(max_or_nan(list(values.values()))) for q, values in by_layer.items()},
                "state_profile_by_layer": profile,
                "mean_trained_loss": sum(trained) / len(trained) if trained else None,
            }
        reference = per_set[SET_REFERENCE]["max_by_layer"]["state"]
        for set_name, entry in per_set.items():
            entry["state_log2_over_reference_by_layer"] = {
                layer: _safe_log2(value) - _safe_log2(reference[layer])
                for layer, value in entry["max_by_layer"]["state"].items()
            }
        summary["modes"][mode] = per_set
    fp32 = {(s, r["window_id"]): r for s, rs in measurements[fp32_mode].items() for r in rs}
    summary["loss_difference_between_modes"] = {}
    for set_name, records in measurements[bf16_mode].items():
        differences = [
            abs(r["loss"]["mean_trained"] - fp32[(set_name, r["window_id"])]["loss"]["mean_trained"])
            for r in records
            if r["loss"]["mean_trained"] is not None
        ]
        summary["loss_difference_between_modes"][set_name] = (
            {"mean_abs": sum(differences) / len(differences), "max_abs": max(differences)} if differences else None
        )
    return summary


def json_safe(value: Any) -> Any:
    """``value`` with every non-finite float replaced by ``"nan"``, ``"inf"`` or ``"-inf"`` (strict JSON has none)."""
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


# =============================================================================
# The posture, the model and the run (GPU)
# =============================================================================


def resolve_posture(spec: ProbeSpec) -> tuple[Any, dict[str, Any]]:
    """The posture's model config, as a launch of its training config builds it before the model (its recipe, YAML,
    mixed-precision preset and vocabulary size), and the record of what was resolved.

    Reads the posture's training config and env file and loads its tokenizer (downloaded from the Hub when the cache
    does not hold it). Raises ProbeError when the posture masks other than exactly one token id, its model and dataset
    sequence lengths differ, its tokenizer is not the spec's, or its env file does not state, exactly once, an
    SSM-state mode the spec runs.
    """
    from scripts.training.launcher_source import env_override_entries

    import pipeline_training_run
    from megatron.bridge.training.mixed_precision import get_mixed_precision_config
    from megatron.bridge.training.setup import _validate_and_set_vocab_size
    from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer

    cfg, _ = pipeline_training_run.resolve_training_config(spec.model, spec.mode, None, spec.training_config, [])
    masked = list(cfg.token_masking.token_ids) if cfg.token_masking.enabled else []
    if len(masked) != 1:
        raise ProbeError(f"the posture must mask exactly one token id (the hidden token), not {masked}")
    hidden_id = int(masked[0])
    if cfg.model.seq_length != cfg.dataset.seq_length:
        raise ProbeError(f"model.seq_length {cfg.model.seq_length} != dataset.seq_length {cfg.dataset.seq_length}")
    if cfg.tokenizer.tokenizer_model != spec.tokenizer:
        raise ProbeError(f"the posture's tokenizer is {cfg.tokenizer.tokenizer_model}, the spec's {spec.tokenizer}")
    entries = env_override_entries(spec.posture_env)
    posture_mode = posture_state_mode(entries)
    if posture_mode not in spec.state_modes:
        raise ProbeError(f"the posture trains in {POSTURE_STATE_VARIABLE}={posture_mode}, which the spec does not run")

    precision = get_mixed_precision_config(cfg.mixed_precision)
    precision.finalize()
    precision.setup(cfg.model)
    tokenizer = build_tokenizer(cfg.tokenizer)
    cfg.model.vocab_size, cfg.model.should_pad_vocab = _validate_and_set_vocab_size(
        cfg.model.vocab_size, tokenizer.vocab_size
    )
    # The weights come from the checkpoint, so the random initialisation they would replace is skipped.
    cfg.model.perform_initialization = False
    model = cfg.model
    record = {
        "training_config": {"path": spec.training_config, "sha256": file_sha256(spec.training_config)},
        "env": {"path": spec.posture_env, "sha256": file_sha256(spec.posture_env), "entries": entries},
        "state_mode_trained": posture_mode,
        "hidden_token_id": hidden_id,
        "seq_length": cfg.model.seq_length,
        "seed": cfg.rng.seed,
        "mixed_precision": cfg.mixed_precision if isinstance(cfg.mixed_precision, str) else repr(cfg.mixed_precision),
        "model": {
            field: repr(getattr(model, field))
            for field in (
                "hybrid_layer_pattern",
                "bf16",
                "params_dtype",
                "fp8",
                "fp8_recipe",
                "fp8_param",
                "first_last_layers_bf16",
                "num_layers_at_start_in_bf16",
                "num_layers_at_end_in_bf16",
                "tensor_model_parallel_size",
                "expert_model_parallel_size",
                "context_parallel_size",
                "pipeline_model_parallel_size",
                "moe_token_dispatcher_type",
                "moe_flex_dispatcher_backend",
                "moe_experts_impl",
                "moe_router_fusion",
                "cross_entropy_loss_fusion",
                "cross_entropy_fusion_impl",
                "use_mamba_mem_eff_path",
                "vocab_size",
                "should_pad_vocab",
            )
        },
        "probe_settings": {
            "perform_initialization": False,
            "grad": "disabled (torch.no_grad, parameters frozen)",
            "module_mode": "eval",
            "comm_overlap": "not applied (a training-schedule setting; the probe runs single forwards)",
        },
    }
    return model, record


def tokenizer_record(spec: ProbeSpec, hidden_id: int) -> dict[str, Any]:
    """The hidden id's spelling in the pinned tokenizer, loaded at the spec's revision (downloaded when the Hub
    cache does not hold it)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec.tokenizer, revision=spec.tokenizer_revision)
    token = tokenizer.convert_ids_to_tokens(hidden_id)
    return {
        "name": spec.tokenizer,
        "revision": spec.tokenizer_revision,
        "hidden_token": token,
        "hidden_token_is_added": token in tokenizer.get_added_vocab(),
    }


def checkpoint_record(checkpoint: str) -> dict[str, Any]:
    """The checkpoint's path, its run config's sha256 and its metadata; raises ProbeError for a directory that is not
    a Megatron-Bridge iteration (no run_config.yaml)."""
    path = Path(checkpoint)
    run_config = path / "run_config.yaml"
    if not run_config.is_file():
        raise ProbeError(f"{checkpoint} has no run_config.yaml; it is not a Megatron-Bridge iteration directory")
    metadata = path / "metadata.json"
    return {
        "path": checkpoint,
        "run_config_sha256": file_sha256(run_config),
        "metadata": json.loads(metadata.read_text()) if metadata.is_file() else None,
    }


def build_plan(spec: ProbeSpec, seq_length: int, hidden_id: int) -> dict[str, Any]:
    """The windows (descriptor and ids) and the record of the corpus they were cut from: its prefix, document and
    token counts, and tokenize record.

    Raises ProbeError when a synthetic run does not fit a ``seq_length``-token window (``check_run_bounds``), the
    corpus was not tokenized with the spec's tokenizer (``corpus_provenance``), its index does not describe its
    ``.bin`` as documents laid end to end (``scripts/data/corpus_index.py``), or it cannot supply the windows
    (``select_windows``).
    """
    check_run_bounds(spec.windows.hidden_runs, seq_length)
    provenance = corpus_provenance(spec)
    prefix = spec.windows.corpus
    index = read_index(Path(f"{prefix}.idx"), Path(f"{prefix}.bin"), error=ProbeError)
    corpus = token_memmap(Path(f"{prefix}.bin"))
    longest, occurrences = window_statistics(corpus, seq_length, hidden_id)
    windows, selection = select_windows(longest, occurrences, seq_length, spec.windows)
    tokens = [window_tokens(window, corpus, seq_length, hidden_id) for window in windows]
    return {
        "windows": windows,
        "tokens": tokens,
        "corpus": {"prefix": prefix, "documents": index.docs, "tokens": index.tokens, "provenance": provenance},
        "selection": selection,
    }


def embedding_record(module, hidden_id: int) -> dict[str, Any]:
    """The hidden id's input and output rows against the tables' median row norms (context for a probe of a model
    that never trained the id)."""
    import torch
    from megatron.core.utils import unwrap_model

    model = unwrap_model(module)
    record = {}
    for name, weight in (
        ("input_embedding", model.embedding.word_embeddings.weight),
        ("output_layer", model.output_layer.weight),
    ):
        with torch.no_grad():
            norms = weight.detach().float().norm(dim=1)
        record[name] = {"hidden_row_norm": float(norms[hidden_id]), "median_row_norm": float(norms.median())}
    return record


def load_model(checkpoint: str, model_config):
    """The posture's model with the checkpoint's weights (``load_single_chunk_model``), its parameters frozen: the
    probe takes no gradient, and a weight that requires one would need the ``main_grad`` buffer only Megatron's DDP
    wrapper creates (the fused loss's gradient-accumulation fusion checks for it; the forward is the same either
    way)."""
    model = load_single_chunk_model(checkpoint, model_config, error=ProbeError)
    model.requires_grad_(False)
    return model


def install_recorders(module) -> tuple[ScanRecorder, LayerOutputRecorder, list[int], list[int]]:
    """Install the scan wrapper and the hooks; return the recorders, the Mamba layers and the decoder layers."""
    import mamba_ssm.ops.triton.ssd_combined as ssd_combined
    from megatron.core.ssm.mamba_mixer import MambaMixer
    from megatron.core.utils import unwrap_model

    scans = ScanRecorder()
    if hasattr(ssd_combined._mamba_chunk_scan_combined_fwd, "__wrapped__"):
        raise ProbeError("the Mamba scan is already wrapped")
    ssd_combined._mamba_chunk_scan_combined_fwd = scans.wrap(ssd_combined._mamba_chunk_scan_combined_fwd)
    mixers = [m for m in module.modules() if isinstance(m, MambaMixer)]
    if not mixers:
        raise ProbeError("the model has no Mamba mixer")
    for mixer in mixers:
        mixer.register_forward_pre_hook(lambda m, _args: scans.enter(m.layer_number))
        mixer.register_forward_hook(lambda m, _args, _out: scans.leave(m.layer_number))
    outputs = LayerOutputRecorder()
    layers = list(unwrap_model(module).decoder.layers)
    for layer in layers:
        layer.register_forward_hook(outputs.hook(layer.layer_number))
    return scans, outputs, sorted(m.layer_number for m in mixers), sorted(layer.layer_number for layer in layers)


def run_windows(module, plan, hidden_id, scans, outputs, mamba_layers, decoder_layers, expected_dtype):
    """This rank's share of the windows, stepping in lockstep with the other data-parallel ranks
    (``lockstep_schedule``); returns ``{window index: record}`` for the windows it keeps."""
    import torch
    from megatron.core import parallel_state

    rank = parallel_state.get_data_parallel_rank()
    schedule = lockstep_schedule(len(plan["windows"]), parallel_state.get_data_parallel_world_size(), rank)
    device = torch.device("cuda", torch.cuda.current_device())
    records = {}
    for step, (index, kept) in enumerate(schedule):
        tokens = torch.from_numpy(plan["tokens"][index]).to(device)
        input_ids, labels = tokens[:-1].unsqueeze(0), tokens[1:].unsqueeze(0)
        position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        with torch.no_grad():
            loss = module(input_ids=input_ids, position_ids=position_ids, attention_mask=None, labels=labels)
        record = {
            "ssm": scans.take(mamba_layers),
            "layer_output": outputs.take(decoder_layers),
            "loss": loss_record(loss, labels, hidden_id),
        }
        dtypes = {layer["state_dtype"] for layer in record["ssm"].values()}
        if dtypes != {expected_dtype}:
            raise ProbeError(f"the scans stored their states as {sorted(dtypes)}, not {expected_dtype}")
        if kept:
            records[index] = record
        logger.info(
            "rank %d step %d/%d window %d done%s", rank, step + 1, len(schedule), index, "" if kept else " (dropped)"
        )
    return records


def install_state_mode(mode: str) -> str:
    """Put the process in ``mode`` and return the dtype the scan must then store its states in."""
    from megatron.core.ssm import mamba_mixer

    from pipeline_training_patches import patch_mamba_training_scan_fp32_state

    patched = getattr(mamba_mixer.mamba_split_conv1d_scan_combined, "_fp32_ssm_state_patched", False)
    if mode == BF16_MODE:
        if patched:
            raise ProbeError("the fp32-state patch is installed already; the bf16 path cannot be measured")
        return "bfloat16"
    if not patch_mamba_training_scan_fp32_state(checkpointed=(mode == "checkpoint")):
        raise ProbeError(f"the fp32-state patch for {POSTURE_STATE_VARIABLE}={mode} was not installed")
    return "float32"


def run_probe(spec: ProbeSpec, run_id: str) -> int:
    """Measure, evaluate and write the result (rank 0); return the exit status.

    A collective over every rank of the launch: it initialises model-parallel state, loads the checkpoint onto this
    rank's GPU, and installs the scan wrapper and the fp32-state patch in this process, neither of which is undone.
    Rank 0 reads the corpus and writes ``<output_dir>/<spec stem>.<run_id>.json``, which is never overwritten.
    """
    import torch
    import torch.distributed as dist
    from scripts.telemetry.code_revision import code_revision

    output = Path(spec.output_dir) / f"{spec.name}.{run_id}.json"
    if output.exists():
        raise ProbeError(f"{output} exists; a result is never overwritten")
    started = datetime.now(timezone.utc).isoformat()
    checkpoint = checkpoint_record(spec.checkpoint)
    model_config, posture = resolve_posture(spec)
    model_config.initialize_model_parallel(seed=posture["seed"])
    rank = dist.get_rank()
    hidden_id, seq_length = posture["hidden_token_id"], posture["seq_length"]

    payload = [build_plan(spec, seq_length, hidden_id) if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    plan = payload[0]
    if rank == 0:
        logger.info("window plan: %s", json.dumps(plan["selection"]))

    module = load_model(spec.checkpoint, model_config)
    scans, outputs, mamba_layers, decoder_layers = install_recorders(module)
    local = {}
    for mode in spec.state_modes:
        expected_dtype = install_state_mode(mode)
        local[mode] = run_windows(
            module, plan, hidden_id, scans, outputs, mamba_layers, decoder_layers, expected_dtype
        )
    gathered = gather_to_rank_zero(local, group=None)
    embedding = embedding_record(module, hidden_id) if rank == 0 else None
    if rank != 0:
        return 0

    measurements: dict[str, dict[str, list[dict]]] = {}
    for mode in spec.state_modes:
        by_index = merge_by_index(
            [share[mode] for share in gathered], len(plan["windows"]), where=f"mode {mode}", error=ProbeError
        )
        sets: dict[str, list[dict]] = {}
        for index, window in enumerate(plan["windows"]):
            sets.setdefault(window.set_name, []).append(
                {"window_id": window.window_id, **describe_tokens(plan["tokens"][index], hidden_id), **by_index[index]}
            )
        measurements[mode] = sets
    evaluation = evaluate_criteria(measurements, spec.criteria, spec.state_modes)
    result = {
        "format": RESULT_FORMAT,
        "spec": {"path": spec.path, "sha256": spec.sha256, "content": spec.content},
        "run": {
            "run_id": run_id,
            "code_revision": code_revision(str(REPO_ROOT)),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "host": socket.gethostname(),
            "started": started,
            "finished": datetime.now(timezone.utc).isoformat(),
            "world_size": dist.get_world_size(),
            "torch": torch.__version__,
            "mamba_ssm": __import__("mamba_ssm").__version__,
            "gpu": torch.cuda.get_device_name(),
        },
        "checkpoint": checkpoint,
        "posture": posture,
        "tokenizer": tokenizer_record(spec, hidden_id),
        "embedding_rows": embedding,
        "windows": {
            "corpus": plan["corpus"],
            "selection": plan["selection"],
            "sets": {
                name: [{"window_id": r["window_id"], "longest_hidden_run": r["longest_hidden_run"]} for r in records]
                for name, records in measurements[spec.state_modes[0]].items()
            },
        },
        "measurements": measurements,
        "summary": summarize(measurements, spec.state_modes[0], spec.fp32_mode),
        **evaluation,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "x") as handle:
        json.dump(json_safe(result), handle, indent=1, allow_nan=False)
    for name, criterion in evaluation["criteria"].items():
        logger.info("criterion %s: %s (%s)", name, criterion["outcome"], criterion["detail"])
    logger.info("verdict %s; result %s", evaluation["verdict"], output)
    return evaluation["exit_status"]


def main() -> int:
    """Read the spec, run the probe on every rank, and return rank 0's verdict as the exit status."""
    import torch.distributed as dist

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True, help="the probe spec (YAML)")
    parser.add_argument("--run-id", required=True, help="names the result file: <output_dir>/<spec stem>.<id>.json")
    args = parser.parse_args()
    if not _RUN_ID_RE.fullmatch(args.run_id):
        parser.error(f"--run-id {args.run_id!r} must be letters, digits, '.', '_' or '-'")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    spec = load_spec(args.spec)
    logger.info("spec %s (sha256 %s)", spec.path, spec.sha256)
    status = run_probe(spec, args.run_id)
    dist.barrier()
    dist.destroy_process_group()
    return status


if __name__ == "__main__":
    sys.exit(main())

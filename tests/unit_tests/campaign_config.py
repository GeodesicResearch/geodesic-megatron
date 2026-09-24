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

"""Shared harness for the control-pretraining campaign's config tests.

The launcher (`pipeline_training_run.py`) builds a recipe, merges the override YAML through
OmegaConf, and applies the result back onto the ``ConfigContainer``. Every campaign test
module asserts its configs through that exact sequence, and the native ``.bin/.idx`` blends
share one well-formedness contract — this module is the single home for both, so the merge
the tests perform cannot drift from the launcher's, and a blend rule fixed in one campaign
arm cannot silently miss the others.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath

import pytest
from megatron.core.datasets.utils import get_blend_from_list
from omegaconf import OmegaConf

from megatron.bridge.training.utils.omegaconf_utils import apply_overrides, create_omegaconf_dict_config
from tests.unit_tests.corpora_fixtures import corpora_table


def merge_onto_recipe(path: Path, recipe_fn):
    """Return ``recipe_fn()`` with the override YAML at ``path`` merged on, as the launcher does."""
    cfg = recipe_fn()
    merged, excluded = create_omegaconf_dict_config(cfg)
    merged = OmegaConf.merge(merged, OmegaConf.load(path))
    apply_overrides(cfg, OmegaConf.to_container(merged, resolve=True), excluded)
    return cfg


def assert_blend_is_well_formed(data_path, label: str) -> None:
    """Assert a flat interleaved weight/prefix blend parses as upstream Megatron will parse it.

    An odd-length list is not an error upstream: ``get_blend_from_list`` reads it as
    prefixes-only, so the weights become filenames and the run dies hours later looking for
    ``0.0875.idx``. Weights must sum to 1.0 so every entry stays auditable against the token
    count recorded beside it.
    """
    data_path = [str(x) for x in data_path]
    assert len(data_path) % 2 == 0, f"{label}: odd-length data_path becomes an unweighted blend"

    blend = get_blend_from_list(data_path)
    prefixes, weights = blend[0], blend[1]
    assert weights is not None, f"{label}: upstream did not read weights from this list"
    assert len(prefixes) == len(weights)
    assert abs(sum(weights) - 1.0) < 1e-6, f"{label}: weights sum to {sum(weights)}"
    assert all(w > 0 for w in weights)
    for prefix in prefixes:
        assert not prefix.endswith((".bin", ".idx")), f"{label}: {prefix} must be extension-less"
        assert prefix.startswith("/projects/a5k/public/data/"), prefix


def assert_shard_weights_are_token_proportional(data_path, corpus_slug: str, total_weight: float) -> None:
    """Assert a sharded corpus's per-shard weights split ``total_weight`` by measured tokens.

    The shards of one corpus are cut at equal DOCUMENT counts, not equal tokens, so equal
    weights would cycle the smaller shards more often than the larger ones. Each shard's
    weight must be ``round(total_weight x shard_tokens / corpus_tokens, 6)``, with any
    six-decimal rounding residue folded into the largest shard so the set sums to exactly
    ``total_weight``. Tokens are read from the ``.provenance.json`` the data build wrote
    beside each shard prefix — the paths come from the blend itself, so a relocated corpus
    fails loudly here rather than skipping.
    """
    data_path = [str(x) for x in data_path]
    shards = {}
    for weight, prefix in zip(data_path[::2], data_path[1::2]):
        if f"__{corpus_slug}/" in prefix:
            shards[prefix] = float(weight)
    assert shards, f"no {corpus_slug} shard prefixes in the blend"
    assert abs(sum(shards.values()) - total_weight) < 1e-9, f"{corpus_slug} shard weights do not sum to {total_weight}"

    tokens = {}
    for prefix in shards:
        prov = Path(prefix + ".provenance.json")
        if not prov.exists():
            pytest.skip(f"corpus provenance not mounted on this host: {prov}")
        tokens[prefix] = json.loads(prov.read_text())["totals"]["total_tokens"]
    total = sum(tokens.values())

    expected = {prefix: round(total_weight * tokens[prefix] / total, 6) for prefix in shards}
    residue = round(total_weight - sum(expected.values()), 6)
    largest = max(expected, key=lambda prefix: tokens[prefix])
    expected[largest] = round(expected[largest] + residue, 6)
    for prefix, weight in shards.items():
        assert weight == expected[prefix], prefix


def assert_iterations_are_the_minimal_cover(train_iters: int, per_iteration: int, target: int, label: str) -> None:
    """Assert ``train_iters`` is the FEWEST iterations covering ``target``, i.e. its ceiling.

    Every campaign arm derives its iteration count from a measured budget rather than estimating
    one, so the rule has two halves and both are asserted: this many iterations reach the target,
    and one fewer would not. Checking only the first would pass a padded count, which spends
    compute on a budget nobody chose.

    ``per_iteration`` carries the unit, which is what differs between arms: the ``.bin/.idx``
    stages and the CPT arms count tokens per iteration, while the packed SFT ablations count
    packed sequences.
    """
    total = train_iters * per_iteration
    assert total >= target, f"{label}: {total:,} is short of the {target:,} target"
    assert (train_iters - 1) * per_iteration < target, (
        f"{label}: {train_iters:,} iterations exceeds the minimum covering {target:,}"
    )


# The workq QOS MaxWall in minutes (from sacctmgr; the partition itself reports UNLIMITED,
# which is why this must be pinned here rather than read from SLURM). A stage's rollover
# clock must fire under this with enough margin for one exit save plus teardown.
WORKQ_MAX_WALL_MINS = 1440


def assert_segment_exit_posture(cfg, label: str, expected_minutes: int | None) -> None:
    """Assert how a stage's SLURM segment ends: on the duration clock, or at ``train_iters``.

    Either way the signal path must be dead, asserted on the MERGED config: omitting
    ``exit_signal_handler`` from a YAML only means "disabled" while no recipe sets it, so a
    recipe that started shipping ``True`` would silently re-arm sbatch's ``--signal`` path
    with every campaign file unchanged — and that path cannot deliver a graceful exit on
    this stack (the non-``B:`` signal reaches the shim shell, apptainer and torchrun at the
    same time as the ranks; the tree is down in ~45 s, under one DP=512 save — measured,
    job 6107666).

    ``expected_minutes`` is the stage's ``train.exit_duration_in_mins``: an integer for a
    long-run stage, which must then fire under the 24 h workq MaxWall with at least 30
    minutes of margin (elapsed time counts from process start, so container launch, index
    load and the slow first iteration all eat into the clock, and the margin is what keeps
    the exit save itself inside the allocation) and have a ``checkpoint.save`` for that
    save to land in; or ``None`` for a run short enough to end at ``train_iters`` inside
    one allocation.
    """
    assert cfg.train.exit_signal_handler is False, f"{label}: the signal exit path must stay dead"
    assert cfg.train.exit_duration_in_mins == expected_minutes, (
        f"{label}: exit_duration_in_mins is {cfg.train.exit_duration_in_mins}, expected {expected_minutes}"
    )
    if expected_minutes is not None:
        assert expected_minutes < WORKQ_MAX_WALL_MINS, f"{label}: the clock must beat the walltime"
        assert WORKQ_MAX_WALL_MINS - expected_minutes >= 30, (
            f"{label}: under one save-plus-teardown of margin before the wall"
        )
        assert cfg.checkpoint.save, f"{label}: the duration exit writes a checkpoint only when checkpoint.save is set"


def flatten_merged_config(cfg) -> dict[str, object]:
    """Every scalar field of a merged config, keyed by its dotted path.

    Serialised through ``create_omegaconf_dict_config`` — the launcher's own function — so a
    comparison between two merged configs covers exactly the fields the launcher would apply,
    and a field the serialiser excludes (a callable, say) is excluded from the comparison too.
    """
    merged, _ = create_omegaconf_dict_config(cfg)
    flat: dict[str, object] = {}

    def walk(node, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else str(key))
        else:
            flat[path] = node

    walk(OmegaConf.to_container(merged, resolve=True), "")
    return flat


def assert_only_these_fields_differ(candidate, reference, allowed: set[str], label: str) -> None:
    """Assert that the merged ``candidate`` differs from the merged ``reference`` in exactly ``allowed``.

    Both are flattened through ``flatten_merged_config``, so the comparison covers exactly the
    fields the launcher would apply. The key sets must agree first — a field present in one config
    and absent from the other is a different drift from a value that moved, and is reported as
    such. Then the set of differing dotted paths must equal ``allowed``: an unexpected difference
    and a missing one are both failures, because a variant that no longer differs where it should
    (a warm start that silently became the reference's, say) is as wrong as one that differs where
    it must not.
    """
    flat_candidate, flat_reference = flatten_merged_config(candidate), flatten_merged_config(reference)
    assert set(flat_candidate) == set(flat_reference), f"{label}: the two configs have different config keys"
    differing = {key for key in flat_candidate if flat_candidate[key] != flat_reference[key]}
    assert differing == allowed, (
        f"{label}: unexpected divergence {sorted(differing - allowed)}, missing divergence {sorted(allowed - differing)}"
    )


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CAMPAIGN_DIR = _REPO_ROOT / "configs" / "control_pretraining"
BUILD_SCRIPT = _CAMPAIGN_DIR / "build_corpora.sh"


def _corpus_root_of(prefix: str) -> PurePosixPath:
    """The corpus directory a blend prefix sits under, a sliced corpus's ``shardN/`` collapsed."""
    root = PurePosixPath(prefix).parent
    if root.name.startswith("shard"):
        root = root.parent
    return root


def blend_subsets(data_path) -> list[str]:
    """The Hub subset each blend prefix was built from, in blend order (a sliced corpus once per shard).

    The prepare step names a corpus directory ``<org>__<name>__<subset>`` (``slugify_dataset_name``)
    and tokenize writes the prefix inside it, or inside a ``shardN/`` of it for a sliced corpus, so
    the subset is the directory name's last ``__`` field.
    """
    return [_corpus_root_of(prefix).name.split("__")[-1] for prefix in [str(x) for x in data_path][1::2]]


def corpus_weights(data_path, strip_suffix: str) -> list[tuple[str, float]]:
    """``(subset, weight)`` per corpus in blend order, a sliced corpus's shard weights summed to one entry.

    ``strip_suffix`` is removed from every subset name, so a filtered arm's blend compares to the
    unfiltered arm's corpus by corpus; pass ``""`` for a blend whose subsets carry no suffix.
    """
    data_path = [str(x) for x in data_path]
    totals: dict[str, float] = {}
    for weight, prefix in zip(data_path[::2], data_path[1::2]):
        subset = _corpus_root_of(prefix).name.split("__")[-1].removesuffix(strip_suffix)
        totals[subset] = round(totals.get(subset, 0.0) + float(weight), 6)
    return list(totals.items())


def campaign_training_configs() -> list[Path]:
    """Every campaign config a launcher merges onto a recipe, newest-arm-agnostic.

    Identified by the ``train`` section that only a training config carries, so the corpus and
    prepare configs are excluded and a new arm is covered the moment its config exists. Tests that
    hand-list the stages they guard go stale silently as arms are added; this is the discovered set
    they should use, and ``test_control_pretraining_config`` asserts the discovery still matches
    every stage by name.
    """
    return [path for path in sorted(_CAMPAIGN_DIR.rglob("*.yaml")) if "train" in (OmegaConf.load(path) or {})]


@functools.lru_cache(maxsize=1)
def _prepare_module():
    """``pipeline_data_prepare`` executed once per session.

    It imports pandas, ``datasets`` and ``transformers`` at module level, so executing it per
    call costs seconds and builds a fresh unregistered module object each time.
    """
    spec = importlib.util.spec_from_file_location("pipeline_data_prepare", _REPO_ROOT / "pipeline_data_prepare.py")
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    return prepare


def assert_prefix_roots_use_the_real_slugify(data_path, dataset: str, label: str) -> None:
    """Assert every blend prefix sits under the directory the data build produces for its subset.

    The blend paths are written by hand; the roots they must match are produced by
    ``pipeline_data_prepare.slugify_dataset_name``. The build derives them through
    ``corpora_table.corpus_root``, a mirror kept so the plan can be derived outside the container,
    so both are asserted: the mirror against the real function, and each blend root against the
    mirror. Every prefix must also end in the tokenize step's output name,
    ``corpora_table.TOKENIZED_PREFIX``.
    """
    prepare = _prepare_module()
    for prefix in [str(x) for x in data_path][1::2]:
        root = _corpus_root_of(prefix)
        subset = root.name.split("__")[-1]
        expected = corpora_table.DATA_BASE / prepare.slugify_dataset_name(dataset, subset)
        mirror = corpora_table.corpus_root(dataset, subset)
        assert mirror == expected, f"{label}: the corpus_root mirror disagrees for {subset}"
        assert str(mirror) == str(root), f"{label}: {prefix} does not sit under {expected}"
        assert prefix.endswith(f"/{corpora_table.TOKENIZED_PREFIX}"), f"{label}: {prefix}"


def dry_run_build(table: Path, stage: str, *subsets: str, env: dict[str, str] | None = None, timeout: int = 120):
    """Plan a data build through the real ``build_corpora.sh`` under ``DRY_RUN=1``, submitting nothing.

    Returns the ``CompletedProcess``: a table with a PENDING count makes the script refuse, which is
    a result the caller asserts on rather than an error here. ``env`` adds variables (``BUILD_STEPS``,
    say) on top of the session's, and overrides ``DRY_RUN`` if it names it.
    """
    return subprocess.run(
        ["bash", str(BUILD_SCRIPT), str(table), stage, *subsets],
        cwd=str(_REPO_ROOT),
        env={**os.environ, "DRY_RUN": "1", **(env or {})},
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def pending_subsets(corpora_rows) -> list[str]:
    """The subsets whose table row holds no document count yet: the build refuses them, and a test
    that needs the plan skips while any remains."""
    return [row.subset for row in corpora_rows if row.docs is None]


def assert_hold_and_pin_move_together(revision, corpora_rows, label: str) -> None:
    """Assert that a corpus table's PENDING counts and its data config's revision move together.

    A PENDING count holds the build until the data is published, and the revision must be pinned
    to that publication in the same change: while the revision is PENDING every count must be, and
    a filled count demands a full 40-hex commit SHA. Counts against an unpinned revision would
    verify a build of whatever the repository's HEAD then was.
    """
    revision = str(revision)
    pending = pending_subsets(corpora_rows)
    if revision == "PENDING":
        assert len(pending) == len(corpora_rows), f"{label}: counts filled while the revision is unpinned: {pending}"
    else:
        assert re.fullmatch(r"[0-9a-f]{40}", revision), f"{label}: the revision must be a full commit SHA: {revision}"
        assert not pending, f"{label}: the revision is pinned but these rows are still held: {pending}"

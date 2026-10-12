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
from scripts.training.config_compose import load_composed_yaml

from megatron.bridge.training.utils.omegaconf_utils import apply_overrides, create_omegaconf_dict_config
from tests.unit_tests.corpora_fixtures import corpora_table, load_campaign_module


def merge_onto_recipe(path: Path, recipe_fn):
    """Return ``recipe_fn()`` with the override YAML at ``path`` merged on, as the launcher does.

    The YAML is read through its ``base_config`` chain, as the launcher reads it, so an overlay is
    asserted as the composed config it trains.
    """
    cfg = recipe_fn()
    merged, excluded = create_omegaconf_dict_config(cfg)
    merged = OmegaConf.merge(merged, OmegaConf.create(load_composed_yaml(path)))
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


def data_parallel_size(cfg, gpus: int) -> int:
    """The data-parallel width of a merged config on ``gpus`` GPUs: the GPUs over TP x CP x PP."""
    model = cfg.model
    model_parallel = (
        model.tensor_model_parallel_size * model.context_parallel_size * model.pipeline_model_parallel_size
    )
    assert gpus % model_parallel == 0, f"{gpus} GPUs do not divide into model-parallel groups of {model_parallel}"
    return gpus // model_parallel


def dotted_leaves(mapping: dict, prefix: str = "") -> dict[str, object]:
    """Every non-mapping value of a nested mapping, keyed by its dotted path."""
    flat: dict[str, object] = {}
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(dotted_leaves(value, path))
        else:
            flat[path] = value
    return flat


def flatten_merged_config(cfg) -> dict[str, object]:
    """Every scalar field of a merged config, keyed by its dotted path.

    Serialised through ``create_omegaconf_dict_config`` — the launcher's own function — so a
    comparison between two merged configs covers exactly the fields the launcher would apply,
    and a field the serialiser excludes (a callable, say) is excluded from the comparison too.
    """
    merged, _ = create_omegaconf_dict_config(cfg)
    return dotted_leaves(OmegaConf.to_container(merged, resolve=True))


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


# The fast Nano pretrain posture: the fields the Nano pretrain quickstart
# (configs/quickstart/nemotron_nano_quickstart_pretrain.yaml) sets on top of the baseline stage-1
# posture, at the values the performance campaign measured
# (docs/investigations/nano30b-pretrain-perf-campaign.md, "Final posture"). A config in this posture
# differs from its as-is counterpart in exactly these fields.
FAST_PRETRAIN_LEVERS = {
    "mixed_precision": "nemotron_h_bf16_with_fp8_current_scaling_bf16_params_bf16_grad_reduce",
    "model.recompute_modules": ["moe_act"],
    "model.moe_token_dispatcher_type": "flex",
    "model.moe_flex_dispatcher_backend": "hybridep",
    "model.mtp_num_layers": None,
    "model.moe_router_fusion": True,
    "model.cross_entropy_loss_fusion": True,
    "model.cross_entropy_fusion_impl": "linear",
    "model.cross_entropy_fusion_saved_logit_chunks": 8,
    "comm_overlap.overlap_param_gather": True,
    "comm_overlap.overlap_moe_expert_parallel_comm": True,
    "ddp.check_for_nan_in_grad": False,
    "rerun_state_machine.check_for_nan_in_loss": False,
    "train.manual_gc": True,
    "train.manual_gc_interval": 10,
    "train.manual_gc_freeze": True,
    "logger.timing_log_level": 1,
    "logger.log_l2_norm_grad_to_tensorboard": False,
}
# The launcher settings the fast posture needs, which a training YAML cannot carry: the
# ISAMBARD_ENV_OVERRIDES lines it is launched with. The first turns the fp32 SSM-state patch off (a
# precision change); the second gives the EP overlap's second stream its own hardware queue.
FAST_PRETRAIN_LAUNCHER_SETTINGS = ["ISAMBARD_FP32_SSM_STATE=0", "ISAMBARD_CUDA_MAX_CONNECTIONS=32"]

# The fast Nano midtraining configuration: the fields the Nano midtraining quickstart
# (configs/quickstart/nemotron_nano_quickstart_midtrain.yaml) sets on top of the baseline stage-2 posture, at the
# values its performance campaign measured (docs/investigations/nano30b-midtrain-perf-campaign.md, "Final posture").
# Unlike the pretraining posture it keeps the gradient NaN check on and runs no EP overlap.
FAST_MIDTRAIN_LEVERS = {
    "mixed_precision": "nemotron_h_bf16_with_fp8_current_scaling_bf16_params_bf16_grad_reduce",
    "model.recompute_granularity": "selective",
    "model.recompute_method": None,
    "model.recompute_num_layers": None,
    "model.recompute_modules": ["moe", "shared_experts"],
    "model.moe_token_dispatcher_type": "flex",
    "model.moe_flex_dispatcher_backend": "hybridep",
    "model.moe_router_fusion": True,
    "model.cross_entropy_loss_fusion": True,
    "model.cross_entropy_fusion_impl": "linear",
    "model.cross_entropy_fusion_saved_logit_chunks": 8,
    "comm_overlap.overlap_param_gather": True,
    "rerun_state_machine.check_for_nan_in_loss": False,
    "train.manual_gc": True,
    "train.manual_gc_interval": 10,
    "train.manual_gc_freeze": True,
    "logger.timing_log_level": 1,
    "logger.log_l2_norm_grad_to_tensorboard": False,
}
# The launcher setting the fast midtraining configuration needs: the fp32 SSM-state patch in its checkpointed mode,
# because at seq 32768 a bf16 inter-chunk SSM state overflows on long single documents. Stated so that a value
# inherited from the environment (the pretraining posture's env file sets 0) cannot switch the fp32 state off.
FAST_MIDTRAIN_LAUNCHER_SETTINGS = ["ISAMBARD_FP32_SSM_STATE=checkpoint"]


def assert_levers_are_set(cfg, levers: dict[str, object], label: str) -> None:
    """Assert that each dotted ``levers`` field of the merged ``cfg`` holds its value.

    Read back from the merged config because the launcher's merge drops a key the config classes lack,
    so a misspelled field, or one the pinned Megatron-LM does not have, would not arrive. A config field
    typed as a mapping (``dataset.dataset_kwargs``) is read by key.
    """
    for dotted, value in levers.items():
        node = cfg
        for part in dotted.split("."):
            node = node[part] if isinstance(node, dict) else getattr(node, part)
        assert node == value, f"{label}: {dotted} is {node!r}, not {value!r}"


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


def is_training_config(path: Path) -> bool:
    """Whether ``path`` is a config a launcher merges onto a recipe, identified by the ``train``
    section that only a training config carries (the corpus/prepare configs have none). Read
    through the ``base_config`` chain, so an overlay whose ``train`` section is all inherited counts."""
    return "train" in load_composed_yaml(path)


def campaign_training_configs() -> list[Path]:
    """Every campaign config a launcher merges onto a recipe, newest-arm-agnostic.

    Identified by ``is_training_config``, so the corpus and prepare configs are excluded and a new
    arm is covered the moment its config exists. Tests that hand-list the stages they guard go
    stale silently as arms are added; this is the discovered set they should use, and
    ``test_control_pretraining_config`` asserts the discovery still matches every stage by name.
    """
    return [path for path in sorted(_CAMPAIGN_DIR.rglob("*.yaml")) if is_training_config(path)]


def run_owners(paths: list[Path]) -> dict[Path, str]:
    """The run each campaign training config belongs to, keyed by resolved path.

    A generated chain's links are one run between them: every link of an arm writes that arm's one
    checkpoint directory, by design. Every other config is a run of its own. The links are found by
    rendering each ``chain.yaml`` the campaign holds, so a new chain is owned the moment its spec
    exists and a hand-written config that happens to reuse a chain's directory is still its own run.
    """
    chains = load_campaign_module("generate_epoch_chain")
    owners = {path.resolve(): str(path.relative_to(_CAMPAIGN_DIR)) for path in paths}
    for spec in sorted(_CAMPAIGN_DIR.rglob("chain.yaml")):
        chain = chains.load_chain(spec)
        output_dir = _REPO_ROOT / chain["output_dir"]
        for arm in chain["arms"]:
            for link in range(1, chains.arm_links(chain, arm) + 1):
                path = (output_dir / chains.link_filename(chain, arm, link)).resolve()
                if path in owners:
                    owners[path] = f"{spec.relative_to(_CAMPAIGN_DIR)}:{arm}"
    return owners


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


# The environment variables ``build_corpora.sh`` reads to submit only part of a plan.
BUILD_SELECTION_VARIABLES = ("BUILD_STEPS", "BUILD_SHARDS")


def dry_run_build(table: Path, stage: str, *subsets: str, env: dict[str, str] | None = None, timeout: int = 120):
    """Plan a data build through the real ``build_corpora.sh`` under ``DRY_RUN=1``, submitting nothing.

    Returns the ``CompletedProcess``: a table with a PENDING count makes the script refuse, which is
    a result the caller asserts on rather than an error here. ``env`` adds variables (``BUILD_STEPS``,
    say) on top of the session's, and overrides ``DRY_RUN`` if it names it. The variables that narrow
    a build (``BUILD_SELECTION_VARIABLES``) are not inherited from the session: one exported in the
    shell running the tests would otherwise plan a different build from the one the test names.
    """
    inherited = {name: value for name, value in os.environ.items() if name not in BUILD_SELECTION_VARIABLES}
    return subprocess.run(
        ["bash", str(BUILD_SCRIPT), str(table), stage, *subsets],
        cwd=str(_REPO_ROOT),
        env={**inherited, "DRY_RUN": "1", **(env or {})},
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def assert_slices_cover_the_corpus(plan: str, subset: str, docs: int, shards: int) -> None:
    """Assert that a dry-run build plan's sliced prepares of ``subset`` read ``shards`` contiguous ranges covering
    exactly ``[0, docs)``: a gap between ranges drops documents silently, an overlap trains some of them twice."""
    pattern = rf"--split train\[(\d+):(\d+)\] --output-dir \S+__{re.escape(subset)}/shard\d+"
    ranges = [(int(beginning), int(end)) for beginning, end in re.findall(pattern, plan)]
    assert len(ranges) == shards, f"{subset}: {len(ranges)} sliced prepares in the plan, expected {shards}"
    assert ranges[0][0] == 0 and ranges[-1][1] == docs, f"{subset}: the slices span {ranges[0][0]}-{ranges[-1][1]}"
    for (_, previous_end), (beginning, _) in zip(ranges, ranges[1:]):
        assert beginning == previous_end, f"{subset}: slice {beginning} does not start where {previous_end} ends"


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


def assert_reads_the_split(cfg, dataset: str, subset: str) -> None:
    """Assert that a merged SFT config reads the pack of one split of a dataset, from that split's corpus root."""
    assert cfg.dataset.dataset_name == dataset, cfg.dataset.dataset_name
    assert cfg.dataset.dataset_root == str(corpora_table.corpus_root(dataset, subset)), cfg.dataset.dataset_root


def assert_data_config_is_the_mixs_but_for_its_source(data_config: Path, mix_data_config: Path) -> None:
    """Assert that a corpus's prepare config differs from its mix's in the dataset and its revision only.

    Every other key (tokenizer, sequence length, pad multiple, prepare options) is then the mix's, so the corpus
    is packed exactly as the mix was, and a build key added to the mix must be added to every cut of it.
    """
    mine = OmegaConf.to_container(OmegaConf.load(data_config))
    mix = OmegaConf.to_container(OmegaConf.load(mix_data_config))
    differing = {key for key in mine.keys() | mix.keys() if mine.get(key) != mix.get(key)}
    assert differing == {"dataset", "revision"}, f"{data_config.name}: {sorted(differing)}"


def assert_row_packs_like_the_mix(row, data_config: Path, mix_row) -> None:
    """Assert that a corpora-table row packs the corpus its data config builds, in the shards the mix's row uses."""
    assert row.config.resolve() == data_config.resolve(), row.config
    assert (row.stage, row.kind) == (mix_row.stage, mix_row.kind) == ("sft", "pack"), (row.stage, row.kind)
    assert (row.shards, row.shard_mode) == (mix_row.shards, mix_row.shard_mode), (row.shards, row.shard_mode)

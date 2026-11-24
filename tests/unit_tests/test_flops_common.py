"""Unit tests for scripts/flops_common.py, the machinery that the NemotronH and dense FLOP estimators share.

Pure python, no torch and no GPU. The estimators' own tests cover this module through their numbers: the NemotronH
tests read the run spec of a shipped quickstart, pad the vocabulary and resolve configs from a directory, a file or a
missing repo id, and the dense tests do the same with a dense architecture. These tests cover what those leave out:
the report arithmetic on its own, the run spec's defaults and config composition, the cache lookup of a repo id, the
recompute table, and the rule that this module imports neither estimator.
"""

import ast
import json
from dataclasses import dataclass
from pathlib import Path

import yaml
from scripts import flops_common
from scripts.flops_common import (
    ATTENTION,
    MAMBA,
    MLP,
    MOE,
    RECOMPUTE_COVERAGE,
    FlopReport,
    RunSpec,
    padded_vocab_size,
    recompute_components,
    resolve_hf_config,
)


@dataclass(frozen=True)
class VocabularyOnly:
    """An architecture of no particular family: the shared code reads only its vocabulary size."""

    vocab_size: int


@dataclass(frozen=True)
class CountsOnly:
    """Parameter counts of no particular family: the shared report reads only the active counts."""

    active: int
    active_non_embedding: int


def write_yaml(path: Path, mapping: dict) -> str:
    path.write_text(yaml.safe_dump(mapping))
    return str(path)


def run_spec(**overrides) -> RunSpec:
    kwargs = dict(
        config_path="<test>",
        global_batch_size=2,
        seq_length=4,
        recompute_granularity=None,
        recompute_modules=(),
        mtp_num_layers=0,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        vocab_size_override=None,
        make_vocab_size_divisible_by=1,
        should_pad_vocab=False,
    )
    kwargs.update(overrides)
    return RunSpec(**kwargs)


def flop_report(**overrides) -> FlopReport:
    """A report of an unnamed family: 100 forward FLOPs per sample, 40 of them recomputed, over 2 samples of 4 tokens."""
    kwargs = dict(
        arch=VocabularyOnly(vocab_size=30),
        run=run_spec(),
        params=CountsOnly(active=10, active_non_embedding=7),
        padded_vocab_size=30,
        forward_by_layer_type={},
        forward_per_sample=100.0,
        recompute_per_sample=40.0,
        backward_multiplier=2.0,
        attention_mask="causal",
    )
    kwargs.update(overrides)
    return FlopReport(**kwargs)


# --------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------


def test_report_arithmetic_by_hand():
    report = flop_report()
    assert report.model_flops_per_sample == 100 * (1 + 2)
    assert report.hardware_flops_per_sample == 300 + 40
    assert report.model_flops_per_iter == 300 * 2  # global_batch_size = 2
    assert report.hardware_flops_per_iter == 340 * 2
    assert report.model_flops_per_token == 300 / 4  # seq_length = 4


def test_report_backward_multiplier_scales_only_the_backward_pass():
    report = flop_report(backward_multiplier=2.5)
    assert report.model_flops_per_sample == 100 * 3.5
    assert report.hardware_flops_per_sample == 350 + 40


def test_six_nd_reads_the_active_counts_and_the_run_tokens():
    report = flop_report()
    assert report.six_nd_per_iter() == 6 * 10 * (2 * 4)
    assert report.six_nd_per_iter(non_embedding=True) == 6 * 7 * (2 * 4)


# --------------------------------------------------------------------------------------
# The run spec
# --------------------------------------------------------------------------------------


def test_run_spec_defaults_for_a_config_that_sets_only_the_workload(tmp_path):
    path = write_yaml(tmp_path / "run.yaml", {"model": {"seq_length": 8192}, "train": {"global_batch_size": 16}})
    assert RunSpec.from_yaml(path) == RunSpec(
        config_path=path,
        global_batch_size=16,
        seq_length=8192,
        recompute_granularity=None,
        recompute_modules=(),
        mtp_num_layers=0,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        vocab_size_override=None,
        make_vocab_size_divisible_by=128,
        should_pad_vocab=True,
    )


def test_run_spec_takes_the_sequence_length_of_the_dataset_when_the_model_sets_none(tmp_path):
    path = write_yaml(tmp_path / "run.yaml", {"dataset": {"seq_length": 32768}, "train": {"global_batch_size": 4}})
    assert RunSpec.from_yaml(path).seq_length == 32768
    both = write_yaml(
        tmp_path / "both.yaml",
        {"model": {"seq_length": 8192}, "dataset": {"seq_length": 32768}, "train": {"global_batch_size": 4}},
    )
    assert RunSpec.from_yaml(both).seq_length == 8192


def test_run_spec_reads_the_model_fields_the_estimators_need(tmp_path):
    path = write_yaml(
        tmp_path / "run.yaml",
        {
            "model": {
                "seq_length": 8,
                "recompute_granularity": "selective",
                "recompute_modules": "core_attn",
                "mtp_num_layers": 2,
                "tensor_model_parallel_size": 2,
                "vocab_size": 131584,
                "make_vocab_size_divisible_by": 64,
                "should_pad_vocab": False,
            },
            "train": {"global_batch_size": 4},
        },
    )
    run = RunSpec.from_yaml(path)
    assert (run.recompute_granularity, run.recompute_modules) == ("selective", ("core_attn",))  # a lone name is a list
    assert (run.mtp_num_layers, run.tensor_model_parallel_size) == (2, 2)
    assert (run.vocab_size_override, run.make_vocab_size_divisible_by, run.should_pad_vocab) == (131584, 64, False)
    assert run.tokens_per_iter == 4 * 8


def test_run_spec_reads_a_config_through_its_base_config_chain(tmp_path):
    write_yaml(
        tmp_path / "base.yaml",
        {"model": {"seq_length": 8192, "recompute_granularity": "full"}, "train": {"global_batch_size": 512}},
    )
    child = write_yaml(
        tmp_path / "child.yaml",
        {"base_config": "base.yaml", "train": {"global_batch_size": 64}, "model": {"recompute_granularity": None}},
    )
    run = RunSpec.from_yaml(child)
    assert (run.seq_length, run.global_batch_size, run.recompute_granularity) == (8192, 64, None)


# --------------------------------------------------------------------------------------
# Vocabulary padding
# --------------------------------------------------------------------------------------


def test_padded_vocab_size_needs_only_the_vocabulary_of_the_architecture():
    arch = VocabularyOnly(vocab_size=30)
    assert padded_vocab_size(arch, run_spec()) == 30
    padded = run_spec(make_vocab_size_divisible_by=7, tensor_model_parallel_size=2, should_pad_vocab=True)
    assert padded_vocab_size(arch, padded) == 42  # 30 up to a multiple of 7 * 2
    assert padded_vocab_size(arch, run_spec(vocab_size_override=64)) == 64


# --------------------------------------------------------------------------------------
# The recompute table
# --------------------------------------------------------------------------------------


def test_recompute_components_takes_the_union_by_layer_kind():
    assert recompute_components(("core_attn", "mlp", "moe", "shared_experts")) == {
        ATTENTION: {"core_attn"},
        MLP: {"mlp"},
        MOE: set(RECOMPUTE_COVERAGE["moe"][MOE]),
    }
    assert recompute_components(("shared_experts",)) == {MOE: {"shared_expert"}}
    assert recompute_components(()) == {}


def test_recompute_modules_that_run_no_matmul_cover_nothing():
    assert recompute_components(("moe_act", "layernorm", "gdn_norm_out", "mla_up_proj")) == {}


def test_every_recompute_entry_names_a_known_layer_kind():
    kinds = {layer_kind for coverage in RECOMPUTE_COVERAGE.values() for layer_kind in coverage}
    assert kinds <= {MAMBA, ATTENTION, MOE, MLP}


# --------------------------------------------------------------------------------------
# HF config resolution
# --------------------------------------------------------------------------------------


def write_snapshot(cache: Path, repo_id: str, revision: str, config: dict, weights: bool) -> Path:
    snapshot = cache / ("models--" + repo_id.replace("/", "--")) / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps(config))
    if weights:
        (snapshot / "model.safetensors").write_bytes(b"")
    return snapshot / "config.json"


def test_a_repo_id_resolves_to_the_snapshot_that_holds_weights(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    write_snapshot(tmp_path, "acme/model", "aaa", {"hidden_size": 1}, weights=False)
    with_weights = write_snapshot(tmp_path, "acme/model", "bbb", {"hidden_size": 2}, weights=True)
    config, path = resolve_hf_config("acme/model")
    assert (config, path) == ({"hidden_size": 2}, str(with_weights))


def test_a_repo_id_without_weights_in_any_snapshot_resolves_to_the_first_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    first = write_snapshot(tmp_path, "acme/model", "aaa", {"hidden_size": 1}, weights=False)
    write_snapshot(tmp_path, "acme/model", "bbb", {"hidden_size": 2}, weights=False)
    assert resolve_hf_config("acme/model") == ({"hidden_size": 1}, str(first))


def test_the_hub_cache_is_taken_from_hf_home_when_hf_hub_cache_is_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    config_path = write_snapshot(tmp_path / "hub", "acme/model", "aaa", {"hidden_size": 3}, weights=True)
    assert resolve_hf_config("acme/model") == ({"hidden_size": 3}, str(config_path))


# --------------------------------------------------------------------------------------
# Layering
# --------------------------------------------------------------------------------------


def imported_names(path: Path) -> set[str]:
    """Every module that ``path`` imports, and every ``module.name`` it imports from a module, at any depth."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_the_shared_module_imports_neither_estimator():
    estimators = {"scripts.nemotronh_flops_estimator", "scripts.dense_flops_estimator"}
    assert not imported_names(Path(flops_common.__file__)) & estimators

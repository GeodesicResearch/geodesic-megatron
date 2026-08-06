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
"""Config validation and launch guards for gradient routing.

Every guard here stands in for a silent-wrongness mode: a run that trains but does not
measure what it claims (mixed-label batches, contaminated Adam moments, an un-swapped MTP
block, a plan that no longer matches train_iters). A guard that stopped firing would not
fail any other test — the run would simply proceed — so each one is pinned individually
against a config that is otherwise fully valid, which also proves the valid config really
does pass.

``cfg`` is a SimpleNamespace because ``validate_gr_launch`` only ever reads attributes off
it, and a real ConfigContainer would require a full model provider, optimizer, and
scheduler to assert something about attribute reads. The two objects the guard actually
type-checks — ``cfg.dataset`` and ``cfg.optimizer_config_override_provider`` — are real.

That stand-in has one blind spot, and it is the one that actually bit: a namespace carries
exactly the attribute names the guard asks for, so a guard reading a field that no real
config declares looks perfectly healthy here. Two tests close it —
``TestGuardedFieldsExistOnTheRealConfigs`` pins every read against the dataclass that owns
it, and ``test_a_missing_guarded_field_raises_rather_than_passing`` pins that a missing
field kills the launch instead of silently satisfying its guard.
"""

import copy
import dataclasses
from types import SimpleNamespace

import pytest

from megatron.bridge.training.config import GPTDatasetConfig
from megatron.bridge.training.gradient_routing.config import GradientRoutingConfig, GRDatasetConfig
from megatron.bridge.training.gradient_routing.guards import validate_gr_launch
from megatron.bridge.training.gradient_routing.optimizer_gating import (
    GROptimizerConfigOverrideProvider,
    GROptimizerGater,
)
from megatron.bridge.training.gradient_routing.plan import build_gr_plan


RETAIN_PATHS = ["0.5", "/data/retain_a_text_document", "0.5", "/data/retain_b_text_document"]
FORGET_PATHS = ["/data/forget_text_document"]
TRAIN_ITERS, GBS, AUX_FFN = 40, 8, 512


def _gr_config(**overrides) -> GradientRoutingConfig:
    kwargs = dict(
        enabled=True,
        retain_data_path=RETAIN_PATHS,
        forget_data_path=FORGET_PATHS,
        aux_ffn_hidden_size=AUX_FFN,
        plan_seed=1234,
        aux_lr=1e-4,
        aux_min_lr=1e-5,
    )
    kwargs.update(overrides)
    return GradientRoutingConfig(**kwargs)


def _dataset_config(plan, **overrides) -> GRDatasetConfig:
    kwargs = dict(
        retain_data_path=RETAIN_PATHS,
        forget_data_path=FORGET_PATHS,
        gr_plan=plan,
        gr_global_batch_size=GBS,
        seq_length=1024,
        split="9999,1,0",
        random_seed=1234,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        mmap_bin_files=True,
        dataloader_type="single",
    )
    kwargs.update(overrides)
    return GRDatasetConfig(**kwargs)


class TestGradientRoutingConfigFinalize:
    """The gr: section's own validation. Nothing load-bearing gets a default."""

    def test_valid_config_finalizes(self):
        _gr_config().finalize()

    def test_disabled_config_skips_all_validation(self):
        """The master switch must make an otherwise-empty section inert, not raise."""
        GradientRoutingConfig().finalize()
        GradientRoutingConfig(enabled=False, p_as=99.0, aux_ffn_hidden_size=-5).finalize()

    @pytest.mark.parametrize(
        "field",
        ["retain_data_path", "forget_data_path", "plan_seed", "aux_lr", "aux_min_lr", "aux_ffn_hidden_size"],
    )
    def test_each_required_field_is_named_when_missing(self, field):
        with pytest.raises(ValueError, match=field):
            _gr_config(**{field: None}).finalize()

    def test_all_missing_fields_are_reported_together(self):
        """One raise listing every gap, not a fix-one-rerun loop."""
        with pytest.raises(ValueError) as excinfo:
            GradientRoutingConfig(enabled=True).finalize()
        message = str(excinfo.value)
        for field in ("retain_data_path", "forget_data_path", "plan_seed", "aux_lr", "aux_min_lr"):
            assert field in message

    @pytest.mark.parametrize("field", ["p_as", "p_cr", "forget_iter_fraction"])
    @pytest.mark.parametrize("value", [-0.01, 1.01, 5.0])
    def test_out_of_range_probabilities_raise(self, field, value):
        with pytest.raises(ValueError, match=f"gr.{field} must be in "):
            _gr_config(**{field: value}).finalize()

    @pytest.mark.parametrize("value", [0, -1, -512])
    def test_non_positive_aux_width_raises(self, value):
        with pytest.raises(ValueError, match="aux_ffn_hidden_size must be positive"):
            _gr_config(aux_ffn_hidden_size=value).finalize()

    @pytest.mark.parametrize("aux_lr, aux_min_lr", [(0.0, 1e-5), (-1e-4, 1e-5), (1e-4, -1e-5)])
    def test_invalid_learning_rates_raise(self, aux_lr, aux_min_lr):
        with pytest.raises(ValueError, match="aux_lr must be > 0"):
            _gr_config(aux_lr=aux_lr, aux_min_lr=aux_min_lr).finalize()

    def test_zero_aux_min_lr_is_allowed(self):
        """min_lr 0 is a legitimate schedule (full decay); only negative is refused."""
        _gr_config(aux_min_lr=0.0).finalize()

    @pytest.mark.parametrize("value", [0, -1])
    def test_non_positive_log_interval_raises(self, value):
        with pytest.raises(ValueError, match="log_interval must be >= 1"):
            _gr_config(log_interval=value).finalize()

    def test_build_plan_uses_the_configured_parameters(self):
        cfg = _gr_config(p_as=0.5, p_cr=0.2, forget_iter_fraction=0.5, plan_seed=99)
        plan = cfg.build_plan(TRAIN_ITERS)
        assert plan.train_iters == TRAIN_ITERS
        assert (plan.plan_seed, plan.p_as, plan.p_cr, plan.forget_iter_fraction) == (99, 0.5, 0.2, 0.5)
        assert plan.digest() == build_gr_plan(99, TRAIN_ITERS, 0.5, 0.5, 0.2).digest()


class TestGRDatasetConfig:
    """Two corpus blends behind one cfg.dataset, and a child that is a plain GPTDatasetConfig."""

    @pytest.fixture
    def plan(self):
        return build_gr_plan(1234, TRAIN_ITERS, 0.5, 0.5, 0.2)

    @pytest.mark.parametrize("field", ["data_path", "blend"])
    def test_setting_data_path_or_blend_directly_raises(self, plan, field):
        """The parent must never carry a blend of its own — it would silently win."""
        value = ["/data/x_text_document"] if field == "data_path" else (["/data/x_text_document"], None)
        with pytest.raises(ValueError, match="do not set data_path/blend"):
            _dataset_config(plan, **{field: value})

    def test_finalize_leaves_the_parent_blend_unset(self, plan):
        config = _dataset_config(plan)
        config.finalize()
        assert config.blend is None
        assert config.data_path is None

    def test_finalize_still_runs_dataloader_finalization(self, plan):
        """Every consumer of cfg.dataset reads the dataloader fields; skipping MCore's
        blend post-init must not skip those."""
        config = _dataset_config(plan, num_workers=0, persistent_workers=True)
        config.finalize()
        assert config.persistent_workers is False

    @pytest.mark.parametrize("empty_field", ["retain_data_path", "forget_data_path"])
    def test_empty_corpus_path_raises_on_finalize(self, plan, empty_field):
        config = _dataset_config(plan, **{empty_field: []})
        with pytest.raises(ValueError, match="non-empty retain_data_path and forget_data_path"):
            config.finalize()

    def test_child_is_a_plain_gpt_dataset_config(self, plan):
        """Exact class, not a subclass: the provider dispatches on isinstance(_, GRDatasetConfig),
        so a child that stayed a GRDatasetConfig would recurse into the GR branch."""
        child = _dataset_config(plan).build_child_config(FORGET_PATHS)
        assert type(child) is GPTDatasetConfig
        assert not isinstance(child, GRDatasetConfig)

    def test_child_carries_only_its_own_corpus_blend(self, plan):
        child = _dataset_config(plan).build_child_config(RETAIN_PATHS)
        assert child.data_path == RETAIN_PATHS
        assert child.blend == (["/data/retain_a_text_document", "/data/retain_b_text_document"], [0.5, 0.5])

    def test_child_drops_every_parent_only_field(self, plan):
        child = _dataset_config(plan).build_child_config(FORGET_PATHS)
        for field in ("retain_data_path", "forget_data_path", "gr_plan", "gr_global_batch_size"):
            assert not hasattr(child, field), f"child still carries {field}"

    def test_child_inherits_the_shared_dataset_fields(self, plan):
        parent = _dataset_config(plan)
        child = parent.build_child_config(FORGET_PATHS)
        assert child.sequence_length == parent.sequence_length
        assert child.random_seed == parent.random_seed
        assert child.dataloader_type == parent.dataloader_type
        assert child.split_matrix is not None, "child was not finalized"

    def test_children_are_independent_and_do_not_mutate_the_parent(self, plan):
        parent = _dataset_config(plan)
        retain_child = parent.build_child_config(RETAIN_PATHS)
        forget_child = parent.build_child_config(FORGET_PATHS)
        assert retain_child.blend[0] == ["/data/retain_a_text_document", "/data/retain_b_text_document"]
        assert forget_child.blend[0] == ["/data/forget_text_document"]
        assert parent.blend is None
        assert parent.retain_data_path == RETAIN_PATHS


def _valid_cfg():
    """A fully-assembled GR config that passes every launch guard."""
    plan = build_gr_plan(1234, TRAIN_ITERS, 0.5, 0.5, 0.2)
    gr = _gr_config()
    gr.finalize()
    gr.runtime_plan = plan
    gr.runtime_gater = GROptimizerGater()
    return SimpleNamespace(
        gr=gr,
        dataset=_dataset_config(plan),
        model=SimpleNamespace(
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            cuda_graph_impl="none",
            mtp_num_layers=0,
            moe_shared_expert_intermediate_size=2048,
            gr_aux_ffn_hidden_size=AUX_FFN,
        ),
        train=SimpleNamespace(
            train_iters=TRAIN_ITERS,
            global_batch_size=GBS,
            rampup_batch_size=None,
            decrease_batch_size_if_needed=False,
        ),
        optimizer=SimpleNamespace(optimizer="adam", overlap_param_gather_with_optimizer_step=False),
        optimizer_config_override_provider=GROptimizerConfigOverrideProvider(aux_lr=1e-4, aux_min_lr=1e-5),
        checkpoint=SimpleNamespace(dist_ckpt_strictness="log_all"),
        validation=SimpleNamespace(eval_iters=0),
    )


#: Every ``cfg.<section>.<field>`` the guard reads by direct attribute access, grouped by
#: the config section that must declare it. ``gr.runtime_plan``/``gr.runtime_gater`` are
#: deliberately absent: they are attached at runtime, so the guard checks them explicitly.
GUARDED_FIELDS = {
    "model": (
        "pipeline_model_parallel_size",
        "virtual_pipeline_model_parallel_size",
        "cuda_graph_impl",
        "mtp_num_layers",
        "moe_shared_expert_intermediate_size",
        "gr_aux_ffn_hidden_size",
    ),
    "train": ("train_iters", "rampup_batch_size", "decrease_batch_size_if_needed"),
    "optimizer": ("optimizer", "overlap_param_gather_with_optimizer_step"),
    "checkpoint": ("dist_ckpt_strictness",),
    "validation": ("eval_iters",),
}

GUARDED_PATHS = [f"{section}.{field}" for section, fields in GUARDED_FIELDS.items() for field in fields]


def _real_config_class(section: str):
    """The dataclass a real ``ConfigContainer`` holds in ``section``.

    ``model`` is the Mamba provider because gradient routing only runs on the NemotronH
    hybrid — it is the provider that declares ``gr_aux_ffn_hidden_size``, and the rest of
    the model fields come from the TransformerConfig / ModelParallelConfig it inherits.
    """
    from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider
    from megatron.bridge.training.config import (
        CheckpointConfig,
        OptimizerConfig,
        TrainingConfig,
        ValidationConfig,
    )

    return {
        "model": MambaModelProvider,
        "train": TrainingConfig,
        "optimizer": OptimizerConfig,
        "checkpoint": CheckpointConfig,
        "validation": ValidationConfig,
    }[section]


def _mutated(path: str, value):
    """A copy of the valid cfg with exactly one dotted attribute changed."""
    cfg = _valid_cfg()
    obj = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)
    return cfg


class TestValidateGRLaunch:
    def test_the_valid_config_passes(self):
        """Without this, every raise below could be firing for the wrong reason."""
        validate_gr_launch(_valid_cfg())

    @pytest.mark.parametrize(
        "path, value, expected",
        [
            ("dataset", GPTDatasetConfig(seq_length=1024, random_seed=1), "must be a GRDatasetConfig"),
            ("dataset.dataloader_type", "batch", "dataloader_type must be 'single'"),
            ("dataset.dataloader_type", "cyclic", "dataloader_type must be 'single'"),
            ("model.pipeline_model_parallel_size", 8, "pipeline_model_parallel_size must be 1"),
            ("model.virtual_pipeline_model_parallel_size", 4, "virtual_pipeline_model_parallel_size must be unset"),
            ("model.cuda_graph_impl", "full_iteration", "cuda_graph_impl must be 'none'"),
            ("model.mtp_num_layers", 1, "mtp_num_layers must be 0"),
            ("model.moe_shared_expert_intermediate_size", None, "moe_shared_expert_intermediate_size must be set"),
            ("model.gr_aux_ffn_hidden_size", AUX_FFN * 2, "one config, one width"),
            ("train.rampup_batch_size", [16, 16, 100], "rampup_batch_size must be unset"),
            ("train.decrease_batch_size_if_needed", True, "decrease_batch_size_if_needed must be False"),
            ("optimizer.optimizer", "sgd", "must be adam-family"),
            ("optimizer.optimizer", "muon", "must be adam-family"),
            ("optimizer.overlap_param_gather_with_optimizer_step", True, "overlap_param_gather_with_optimizer_step"),
            ("checkpoint.dist_ckpt_strictness", "assert", "does not tolerate missing keys"),
            ("checkpoint.dist_ckpt_strictness", "raise_all", "does not tolerate missing keys"),
            ("validation.eval_iters", 10, "eval_iters must be 0"),
            ("gr.runtime_plan", None, "runtime_plan/runtime_gater missing"),
            ("gr.runtime_gater", None, "runtime_plan/runtime_gater missing"),
            ("train.train_iters", TRAIN_ITERS + 1, "train_iters changed after the plan was built"),
        ],
    )
    def test_each_violation_raises_with_its_own_message(self, path, value, expected):
        with pytest.raises(ValueError, match="Gradient-routing launch guards failed") as excinfo:
            validate_gr_launch(_mutated(path, value))
        assert expected in str(excinfo.value)

    def test_wrong_override_provider_raises(self):
        from megatron.bridge.training.config import OptimizerConfigOverrideProvider

        cfg = _mutated("optimizer_config_override_provider", OptimizerConfigOverrideProvider())
        with pytest.raises(ValueError, match="must be the GROptimizerConfigOverrideProvider"):
            validate_gr_launch(cfg)

    @pytest.mark.parametrize("strictness", ["log_all", "log_unexpected", "ignore_all", "return_all"])
    def test_missing_key_tolerant_strictness_values_pass(self, strictness):
        """The warm start loads a base checkpoint with no gr_aux tensors — every value that
        tolerates that must be accepted, not just the one the shipped config happens to use."""
        validate_gr_launch(_mutated("checkpoint.dist_ckpt_strictness", strictness))

    @pytest.mark.parametrize("optimizer", ["adam", "hybridadam", "adamw"])
    def test_adam_family_optimizers_pass(self, optimizer):
        validate_gr_launch(_mutated("optimizer.optimizer", optimizer))

    @pytest.mark.parametrize("cuda_graph_impl", [None, "none"])
    def test_cuda_graphs_off_passes(self, cuda_graph_impl):
        validate_gr_launch(_mutated("model.cuda_graph_impl", cuda_graph_impl))

    def test_every_problem_is_reported_at_once(self):
        """Operators fix a launch config once, not one guard per allocation."""
        cfg = _valid_cfg()
        cfg.model.pipeline_model_parallel_size = 4
        cfg.validation.eval_iters = 10
        cfg.optimizer.optimizer = "sgd"
        with pytest.raises(ValueError) as excinfo:
            validate_gr_launch(cfg)
        message = str(excinfo.value)
        assert "pipeline_model_parallel_size must be 1" in message
        assert "eval_iters must be 0" in message
        assert "must be adam-family" in message
        assert message.count("\n- ") == 3

    def test_dataloader_type_is_not_checked_on_a_non_gr_dataset(self):
        """The dataloader check hangs off the isinstance branch; a plain config must not
        produce two problems for one root cause."""
        cfg = _mutated("dataset", GPTDatasetConfig(seq_length=1024, random_seed=1, dataloader_type="batch"))
        with pytest.raises(ValueError) as excinfo:
            validate_gr_launch(cfg)
        assert str(excinfo.value).count("\n- ") == 1

    @pytest.mark.parametrize("path", GUARDED_PATHS)
    def test_a_missing_guarded_field_raises_rather_than_passing(self, path):
        """A field the guard cannot find must kill the launch, not satisfy its check.

        This is the failure mode ``getattr(obj, name, <passing default>)`` produces: rename
        the field upstream and the guard silently stops guarding, which no other test in
        this file would notice — every one of them mutates a field that still exists.
        """
        cfg = _valid_cfg()
        section, field = path.split(".")
        delattr(getattr(cfg, section), field)
        with pytest.raises(AttributeError, match=field):
            validate_gr_launch(cfg)

    def test_guard_does_not_mutate_the_config(self):
        """Validation is a read; a guard that normalised a field would hide the mismatch."""
        cfg = _valid_cfg()
        before = copy.deepcopy(vars(cfg.model)), copy.deepcopy(vars(cfg.train))
        validate_gr_launch(cfg)
        assert (vars(cfg.model), vars(cfg.train)) == before


class TestGuardedFieldsExistOnTheRealConfigs:
    """Every guarded name must be declared by the dataclass a real run actually carries.

    The SimpleNamespace cfg above answers "does the guard fire when the field is wrong?";
    it cannot answer "is the guard reading a field that exists?". It did not: the
    ``overlap_param_gather_with_optimizer_step`` check used to read ``cfg.ddp``, where the
    field has never lived (it is on the optimizer config, which is where ``setup.py`` reads
    it from), so that guard could not fire on any real launch.
    """

    @pytest.mark.parametrize("path", GUARDED_PATHS)
    def test_the_field_is_declared_on_its_config_class(self, path):
        section, field = path.split(".")
        config_cls = _real_config_class(section)
        declared = {f.name for f in dataclasses.fields(config_cls)}
        assert field in declared, f"validate_gr_launch reads cfg.{path}, which {config_cls.__name__} does not declare"

    def test_the_guarded_dataloader_type_is_declared_on_the_dataset_config(self):
        """cfg.dataset is a real object in the fixture, so it lives outside the mapping above."""
        assert "dataloader_type" in {f.name for f in dataclasses.fields(GRDatasetConfig)}

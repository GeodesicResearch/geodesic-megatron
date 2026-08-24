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
block, a plan that no longer matches train_iters, a module count that no longer matches the
corpus list). A guard that stopped firing would not fail any other test — the run would
simply proceed — so each one is pinned individually against a config that is otherwise fully
valid, which also proves the valid config really does pass.

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
import yaml

from megatron.bridge.training.config import GPTDatasetConfig
from megatron.bridge.training.gradient_routing.config import (
    GradientRoutingConfig,
    GRDatasetConfig,
    GRFinetuningDatasetConfig,
    reject_renamed_fields,
)
from megatron.bridge.training.gradient_routing.guards import gr_posture_problems, validate_gr_launch
from megatron.bridge.training.gradient_routing.optimizer_gating import (
    GROptimizerConfigOverrideProvider,
    GROptimizerGater,
)
from megatron.bridge.training.gradient_routing.plan import build_gr_plan


RETAIN_PATHS = ["0.5", "/data/core_a_text_document", "0.5", "/data/core_b_text_document"]
AUX_PATHS = [["/data/aux0_text_document"], ["/data/aux1_text_document"]]
TRAIN_ITERS, GBS, AUX_FFN = 40, 8, 512

#: The N=2 posture used wherever per-module breadth is what is under test. Every per-module
#: field is given DISTINCT values so an index confusion cannot pass as a broadcast.
AUX_FFNS = [512, 1024]
AUX_LRS = [1e-4, 2e-4]
AUX_MIN_LRS = [1e-5, 2e-5]
AUX_FRACTIONS_2 = [0.25, 0.25]


def _gr_config(**overrides) -> GradientRoutingConfig:
    kwargs = dict(
        enabled=True,
        retain_data_path=RETAIN_PATHS,
        aux_data_paths=AUX_PATHS[:1],
        aux_iter_fractions=[0.5],
        aux_ffn_hidden_size=AUX_FFN,
        plan_seed=1234,
        aux_lr=1e-4,
        aux_min_lr=1e-5,
    )
    kwargs.update(overrides)
    return GradientRoutingConfig(**kwargs)


def _gr_config_2(**overrides) -> GradientRoutingConfig:
    """The two-module config: per-module lists throughout."""
    kwargs = dict(
        aux_data_paths=AUX_PATHS,
        aux_iter_fractions=AUX_FRACTIONS_2,
        aux_ffn_hidden_size=AUX_FFNS,
        aux_lr=AUX_LRS,
        aux_min_lr=AUX_MIN_LRS,
    )
    kwargs.update(overrides)
    return _gr_config(**kwargs)


def _dataset_config(plan, **overrides) -> GRDatasetConfig:
    kwargs = dict(
        retain_data_path=RETAIN_PATHS,
        aux_data_paths=AUX_PATHS[: plan.n_aux],
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


class TestRenamedFieldsAreRefused:
    """The pre-multi-module spellings must be refused with the rename, never aliased.

    ``forget_data_path``/``forget_iter_fraction`` were the whole schema of every GR run
    before this migration, so they are still in configs, notes and checkpointed run_configs.
    Silently mapping them onto the new fields would leave two spellings of one experiment in
    circulation; guessing the module count from a scalar fraction is exactly the guess a
    multi-module schema must not make.
    """

    @pytest.mark.parametrize(
        "field, replacement", [("forget_data_path", "aux_data_paths"), ("forget_iter_fraction", "aux_iter_fractions")]
    )
    def test_the_rejection_names_the_new_field(self, field, replacement):
        with pytest.raises(ValueError, match="pre-multi-module field names") as excinfo:
            reject_renamed_fields({"enabled": True, field: "whatever"})
        assert replacement in str(excinfo.value)

    def test_both_stale_names_are_reported_together(self):
        with pytest.raises(ValueError) as excinfo:
            reject_renamed_fields({"forget_data_path": ["/data/x"], "forget_iter_fraction": 0.5})
        message = str(excinfo.value)
        assert "forget_data_path" in message and "forget_iter_fraction" in message

    def test_a_current_schema_dict_passes(self):
        """Without this, the rejection could be firing on every config."""
        reject_renamed_fields({"enabled": True, "aux_data_paths": AUX_PATHS, "aux_iter_fractions": AUX_FRACTIONS_2})

    @pytest.mark.parametrize("field", ["forget_data_path", "forget_iter_fraction"])
    def test_the_dataclass_no_longer_accepts_the_old_name(self, field):
        """The rejection is a courtesy on top of the schema, not the only line of defence:
        the field is gone, so even a caller that skipped the check cannot set it."""
        with pytest.raises(TypeError, match=field):
            GradientRoutingConfig(enabled=True, **{field: 0.5})
        assert field not in {f.name for f in dataclasses.fields(GradientRoutingConfig)}


class TestGradientRoutingConfigFinalize:
    """The gr: section's own validation. Nothing load-bearing gets a default."""

    def test_valid_config_finalizes(self):
        _gr_config().finalize()

    def test_valid_two_module_config_finalizes(self):
        _gr_config_2().finalize()

    def test_disabled_config_skips_all_validation(self):
        """The master switch must make an otherwise-empty section inert, not raise."""
        GradientRoutingConfig().finalize()
        GradientRoutingConfig(enabled=False, p_as=99.0, aux_ffn_hidden_size=-5).finalize()

    @pytest.mark.parametrize(
        "field",
        [
            "retain_data_path",
            "aux_data_paths",
            "aux_iter_fractions",
            "plan_seed",
            "aux_lr",
            "aux_min_lr",
            "aux_ffn_hidden_size",
        ],
    )
    def test_each_required_field_is_named_when_missing(self, field):
        with pytest.raises(ValueError, match=field):
            _gr_config(**{field: None}).finalize()

    def test_all_missing_fields_are_reported_together(self):
        """One raise listing every gap, not a fix-one-rerun loop."""
        with pytest.raises(ValueError) as excinfo:
            GradientRoutingConfig(enabled=True).finalize()
        message = str(excinfo.value)
        for field in (
            "retain_data_path",
            "aux_data_paths",
            "aux_iter_fractions",
            "plan_seed",
            "aux_lr",
            "aux_min_lr",
        ):
            assert field in message

    @pytest.mark.parametrize("aux_data_paths", [[], [[]], [AUX_PATHS[0], []]])
    def test_an_empty_corpus_blend_raises(self, aux_data_paths):
        """One entry per module, each a real blend: an empty inner list would build a module
        with no corpus to train it, which the plan would then route iterations to."""
        with pytest.raises(ValueError, match="non-empty list of non-empty blend lists"):
            _gr_config(aux_data_paths=aux_data_paths, aux_iter_fractions=[0.5] * len(aux_data_paths)).finalize()

    @pytest.mark.parametrize("field", ["p_as", "p_cr"])
    @pytest.mark.parametrize("value", [-0.01, 1.01, 5.0])
    def test_out_of_range_probabilities_raise(self, field, value):
        with pytest.raises(ValueError, match=f"gr.{field} must be in "):
            _gr_config(**{field: value}).finalize()

    @pytest.mark.parametrize("module", [0, 1])
    @pytest.mark.parametrize("value", [-0.01, 1.01, 5.0])
    def test_out_of_range_iteration_fractions_raise(self, module, value):
        fractions = list(AUX_FRACTIONS_2)
        fractions[module] = value
        with pytest.raises(ValueError, match=rf"gr.aux_iter_fractions\[{module}\] must be in "):
            _gr_config_2(aux_iter_fractions=fractions).finalize()

    def test_iteration_fractions_summing_above_one_raise(self):
        """The core corpus would need a negative share of the iterations."""
        with pytest.raises(ValueError, match="must sum to <= 1"):
            _gr_config_2(aux_iter_fractions=[0.6, 0.6]).finalize()

    @pytest.mark.parametrize("value", [0.5, 0.1])
    def test_a_scalar_iteration_fraction_is_refused_at_finalize(self, value):
        """Unlike the width/LR fields, a fraction does not broadcast: the same value for
        every module multiplies the TOTAL aux share by the module count, which is never
        what a scalar spelling says. finalize must refuse it — not build_plan later,
        whose bare iteration TypeError names neither the field nor the fix."""
        with pytest.raises(ValueError, match="aux_iter_fractions must be a list"):
            _gr_config_2(aux_iter_fractions=value).finalize()

    @pytest.mark.parametrize("value", [0, -1, -512])
    def test_non_positive_aux_width_raises(self, value):
        with pytest.raises(ValueError, match=r"aux_ffn_hidden_size\[0\] must be positive"):
            _gr_config(aux_ffn_hidden_size=value).finalize()

    def test_a_non_positive_width_names_the_module_it_belongs_to(self):
        """With N modules the message must say WHICH width is wrong, or the operator has to
        guess which list entry to fix."""
        with pytest.raises(ValueError, match=r"aux_ffn_hidden_size\[1\] must be positive"):
            _gr_config_2(aux_ffn_hidden_size=[512, 0]).finalize()

    @pytest.mark.parametrize("aux_lr, aux_min_lr", [(0.0, 1e-5), (-1e-4, 1e-5), (1e-4, -1e-5)])
    def test_invalid_learning_rates_raise(self, aux_lr, aux_min_lr):
        with pytest.raises(ValueError, match=r"aux_lr\[0\] must be > 0"):
            _gr_config(aux_lr=aux_lr, aux_min_lr=aux_min_lr).finalize()

    def test_an_invalid_per_module_learning_rate_names_its_module(self):
        with pytest.raises(ValueError, match=r"aux_lr\[1\] must be > 0"):
            _gr_config_2(aux_lr=[1e-4, 0.0]).finalize()

    def test_zero_aux_min_lr_is_allowed(self):
        """min_lr 0 is a legitimate schedule (full decay); only negative is refused."""
        _gr_config(aux_min_lr=0.0).finalize()

    @pytest.mark.parametrize(
        "field, value",
        [
            ("aux_iter_fractions", [0.25]),
            ("aux_ffn_hidden_size", [512]),
            ("aux_lr", [1e-4, 2e-4, 3e-4]),
            ("aux_min_lr", [1e-5]),
            ("aux_wd_mult", [1.0, 1.0, 1.0]),
        ],
    )
    def test_a_per_module_list_of_the_wrong_length_raises(self, field, value):
        """A short or long list is the multi-module typo: two corpora and one LR would
        otherwise either broadcast silently or index out of range at optimizer build."""
        with pytest.raises(ValueError, match=f"gr.{field} has {len(value)} entries for 2 aux modules"):
            _gr_config_2(**{field: value}).finalize()

    @pytest.mark.parametrize("value", [0, -1])
    def test_non_positive_log_interval_raises(self, value):
        with pytest.raises(ValueError, match="log_interval must be >= 1"):
            _gr_config(log_interval=value).finalize()

    def test_scalar_per_module_fields_broadcast_over_the_modules(self):
        """A run with N identical modules should not have to repeat itself N times."""
        cfg = _gr_config(
            aux_data_paths=AUX_PATHS,
            aux_iter_fractions=AUX_FRACTIONS_2,
            aux_ffn_hidden_size=AUX_FFN,
            aux_lr=1e-4,
            aux_min_lr=1e-5,
            aux_wd_mult=0.5,
        )
        cfg.finalize()
        assert cfg.n_aux == 2
        assert cfg.aux_ffn_hidden_sizes() == [AUX_FFN, AUX_FFN]
        assert cfg.aux_lrs() == [1e-4, 1e-4]
        assert cfg.aux_min_lrs() == [1e-5, 1e-5]
        assert cfg.aux_wd_mults() == [0.5, 0.5]

    def test_per_module_lists_are_returned_in_order(self):
        """Module k's width, LR and min-LR all come from index k; a reversal here would train
        each module at its sibling's rate with nothing in the logs to say so."""
        cfg = _gr_config_2()
        cfg.finalize()
        assert cfg.n_aux == 2
        assert cfg.aux_ffn_hidden_sizes() == AUX_FFNS
        assert cfg.aux_lrs() == AUX_LRS
        assert cfg.aux_min_lrs() == AUX_MIN_LRS

    def test_n_aux_is_zero_without_a_corpus_list(self):
        """``n_aux`` is read before validation in places (the width broadcast); it must be a
        plain 0 on an unconfigured section rather than raising."""
        assert GradientRoutingConfig().n_aux == 0

    def test_the_dataset_root_spelling_finalizes(self):
        """The sft corpus spelling: dataset roots instead of blend lists."""
        cfg = _gr_config(
            retain_data_path=None,
            aux_data_paths=None,
            retain_dataset_root="/data/gr_sft/core",
            aux_dataset_roots=["/data/gr_sft/aux0"],
        )
        cfg.finalize()
        assert cfg.n_aux == 1

    def test_n_aux_comes_from_the_dataset_roots_when_they_are_the_spelling(self):
        cfg = _gr_config(
            retain_data_path=None,
            aux_data_paths=None,
            retain_dataset_root="/data/gr_sft/core",
            aux_dataset_roots=["/data/gr_sft/aux0", "/data/gr_sft/aux1"],
            aux_iter_fractions=AUX_FRACTIONS_2,
        )
        assert cfg.n_aux == 2

    def test_both_corpus_spellings_together_are_refused(self):
        """A run trains through exactly one data stack; carrying both spellings would leave
        which corpus actually trained ambiguous to every reader of the config."""
        with pytest.raises(ValueError, match="both corpus spellings are set"):
            _gr_config(retain_dataset_root="/data/gr_sft/core", aux_dataset_roots=["/data/gr_sft/aux0"]).finalize()

    @pytest.mark.parametrize(
        "present, missing",
        [
            ({"retain_dataset_root": "/data/gr_sft/core"}, "aux_dataset_roots"),
            ({"aux_dataset_roots": ["/data/gr_sft/aux0"]}, "retain_dataset_root"),
        ],
    )
    def test_half_a_dataset_root_pair_names_the_missing_half(self, present, missing):
        with pytest.raises(ValueError, match=missing):
            _gr_config(retain_data_path=None, aux_data_paths=None, **present).finalize()

    @pytest.mark.parametrize("aux_dataset_roots", [[], [""], ["/data/gr_sft/aux0", ""]])
    def test_an_empty_dataset_root_raises(self, aux_dataset_roots):
        with pytest.raises(ValueError, match="non-empty list of non-empty dataset roots"):
            _gr_config(
                retain_data_path=None,
                aux_data_paths=None,
                retain_dataset_root="/data/gr_sft/core",
                aux_dataset_roots=aux_dataset_roots,
                aux_iter_fractions=[0.5] * max(len(aux_dataset_roots), 1),
            ).finalize()

    def test_build_plan_uses_the_configured_parameters(self):
        cfg = _gr_config(p_as=0.5, p_cr=0.2, aux_iter_fractions=[0.5], plan_seed=99)
        plan = cfg.build_plan(TRAIN_ITERS)
        assert plan.train_iters == TRAIN_ITERS
        assert (plan.plan_seed, plan.p_as, plan.p_cr, plan.aux_iter_fractions) == (99, 0.5, 0.2, (0.5,))
        assert plan.digest() == build_gr_plan(99, TRAIN_ITERS, [0.5], 0.5, 0.2).digest()

    def test_build_plan_carries_every_module(self):
        cfg = _gr_config_2(plan_seed=99)
        plan = cfg.build_plan(TRAIN_ITERS)
        assert plan.n_aux == 2
        assert plan.aux_iter_fractions == tuple(AUX_FRACTIONS_2)
        assert plan.digest() == build_gr_plan(99, TRAIN_ITERS, AUX_FRACTIONS_2, 0.5, 0.2).digest()


class TestGRDatasetConfig:
    """N+1 corpus blends behind one cfg.dataset, and a child that is a plain GPTDatasetConfig."""

    @pytest.fixture(params=[[0.5], AUX_FRACTIONS_2], ids=["one_aux", "two_aux"])
    def plan(self, request):
        return build_gr_plan(1234, TRAIN_ITERS, request.param, 0.5, 0.2)

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

    @pytest.mark.parametrize("empty_value", [[], [[]]])
    def test_empty_aux_corpus_path_raises_on_finalize(self, plan, empty_value):
        config = _dataset_config(plan, aux_data_paths=empty_value)
        with pytest.raises(ValueError, match="non-empty retain_data_path and aux_data_paths"):
            config.finalize()

    def test_empty_retain_path_raises_on_finalize(self, plan):
        config = _dataset_config(plan, retain_data_path=[])
        with pytest.raises(ValueError, match="non-empty retain_data_path and aux_data_paths"):
            config.finalize()

    def test_a_corpus_count_that_disagrees_with_the_plan_raises(self, plan):
        """The plan sizes each child and labels each iteration; a config with a different
        number of corpora would leave the routed dataset a child short (or spare), which
        surfaces as a KeyError mid-epoch rather than at launch."""
        wrong = AUX_PATHS[:1] if plan.n_aux == 2 else AUX_PATHS
        config = _dataset_config(plan, aux_data_paths=wrong)
        with pytest.raises(ValueError, match="the config and the plan disagree about the module count"):
            config.finalize()

    def test_child_is_a_plain_gpt_dataset_config(self, plan):
        """Exact class, not a subclass: the provider dispatches on isinstance(_, GRDatasetConfig),
        so a child that stayed a GRDatasetConfig would recurse into the GR branch."""
        child = _dataset_config(plan).build_child_config(AUX_PATHS[0])
        assert type(child) is GPTDatasetConfig
        assert not isinstance(child, GRDatasetConfig)

    def test_child_carries_only_its_own_corpus_blend(self, plan):
        child = _dataset_config(plan).build_child_config(RETAIN_PATHS)
        assert child.data_path == RETAIN_PATHS
        assert child.blend == (["/data/core_a_text_document", "/data/core_b_text_document"], [0.5, 0.5])

    def test_child_drops_every_parent_only_field(self, plan):
        child = _dataset_config(plan).build_child_config(AUX_PATHS[0])
        for field in ("retain_data_path", "aux_data_paths", "gr_plan", "gr_global_batch_size"):
            assert not hasattr(child, field), f"child still carries {field}"

    def test_child_inherits_the_shared_dataset_fields(self, plan):
        parent = _dataset_config(plan)
        child = parent.build_child_config(AUX_PATHS[0])
        assert child.sequence_length == parent.sequence_length
        assert child.random_seed == parent.random_seed
        assert child.dataloader_type == parent.dataloader_type
        assert child.split_matrix is not None, "child was not finalized"

    def test_children_are_independent_and_do_not_mutate_the_parent(self, plan):
        parent = _dataset_config(plan)
        children = [parent.build_child_config(paths) for paths in [RETAIN_PATHS, *AUX_PATHS[: plan.n_aux]]]
        assert children[0].blend[0] == ["/data/core_a_text_document", "/data/core_b_text_document"]
        for k, child in enumerate(children[1:]):
            assert child.blend[0] == AUX_PATHS[k]
        assert parent.blend is None
        assert parent.retain_data_path == RETAIN_PATHS
        assert parent.aux_data_paths == AUX_PATHS[: plan.n_aux]


def _valid_cfg(fractions=(0.5,), widths=(AUX_FFN,), lrs=(1e-4,), min_lrs=(1e-5,), p_as=0.5, **gr_overrides):
    """A fully-assembled GR config that passes every launch guard.

    ``p_as`` is threaded into BOTH the gr section and the plan so the two never disagree; it
    is what decides whether the plan updates the core on an all-aux corpus split, which the
    aux-only-checkpoint guard reads.
    """
    fractions, widths, lrs, min_lrs = list(fractions), list(widths), list(lrs), list(min_lrs)
    plan = build_gr_plan(1234, TRAIN_ITERS, fractions, p_as, 0.2)
    gr = _gr_config(
        aux_data_paths=AUX_PATHS[: len(fractions)],
        aux_iter_fractions=fractions,
        aux_ffn_hidden_size=widths,
        aux_lr=lrs,
        aux_min_lr=min_lrs,
        p_as=p_as,
        **gr_overrides,
    )
    gr.finalize()
    gr.runtime_plan = plan
    gr.runtime_gater = GROptimizerGater(n_aux=len(fractions))
    return SimpleNamespace(
        gr=gr,
        dataset=_dataset_config(plan),
        model=SimpleNamespace(
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            cuda_graph_impl="none",
            mtp_num_layers=0,
            moe_shared_expert_intermediate_size=2048,
            gr_aux_ffn_hidden_size=widths,
            gr_static_gates=None,
        ),
        train=SimpleNamespace(
            train_iters=TRAIN_ITERS,
            global_batch_size=GBS,
            rampup_batch_size=None,
            decrease_batch_size_if_needed=False,
        ),
        optimizer=SimpleNamespace(
            optimizer="adam",
            overlap_param_gather_with_optimizer_step=False,
            optimizer_cpu_offload=False,
        ),
        optimizer_config_override_provider=GROptimizerConfigOverrideProvider(
            aux_lrs=lrs, aux_min_lrs=min_lrs, aux_wd_mults=[1.0] * len(fractions)
        ),
        checkpoint=SimpleNamespace(
            dist_ckpt_strictness="log_all",
            # Paths that do not exist on disk: the aux-only source guard reads each one's
            # run_config, and a missing checkpoint reports "not aux-only" without a stat storm.
            load=None,
            pretrained_checkpoint="/data/base_ckpt",
            save_optim=False,
        ),
        validation=SimpleNamespace(eval_iters=0),
        inprocess_restart=None,
        peft=None,
    )


def _valid_cfg_2():
    """The same, at two aux modules with distinct widths."""
    return _valid_cfg(fractions=AUX_FRACTIONS_2, widths=AUX_FFNS, lrs=AUX_LRS, min_lrs=AUX_MIN_LRS)


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
        "gr_static_gates",
    ),
    "train": ("train_iters", "rampup_batch_size", "decrease_batch_size_if_needed"),
    "optimizer": ("optimizer", "overlap_param_gather_with_optimizer_step", "optimizer_cpu_offload"),
    "checkpoint": ("dist_ckpt_strictness", "load", "pretrained_checkpoint"),
    "validation": ("eval_iters",),
}

GUARDED_PATHS = [f"{section}.{field}" for section, fields in GUARDED_FIELDS.items() for field in fields]

#: What the guard reads off ``cfg.gr`` itself. Kept apart from GUARDED_FIELDS because ``gr``
#: is a real dataclass in the fixture, so deleting one of these falls back to the class-level
#: default instead of raising — the mapping above exists for the deletion test.
GUARDED_GR_FIELDS = ("aux_data_paths", "aux_ffn_hidden_size")
GUARDED_GR_ACCESSORS = ("n_aux", "aux_ffn_hidden_sizes")


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


def _mutated(path: str, value, cfg=None):
    """A copy of the valid cfg with exactly one dotted attribute changed."""
    cfg = cfg if cfg is not None else _valid_cfg()
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

    def test_the_valid_two_module_config_passes(self):
        validate_gr_launch(_valid_cfg_2())

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
            ("model.gr_aux_ffn_hidden_size", [AUX_FFN * 2], "one config, one width"),
            # Static gates are the eval-only profile-probing mechanism: they pin the gates at
            # construction, which would override the plan's per-iteration drive silently.
            ("model.gr_static_gates", [1.0], "gr_static_gates must be unset for a training run"),
            ("train.rampup_batch_size", [16, 16, 100], "rampup_batch_size must be unset"),
            ("train.decrease_batch_size_if_needed", True, "decrease_batch_size_if_needed must be False"),
            ("optimizer.optimizer", "sgd", "must be adam-family"),
            ("optimizer.optimizer", "muon", "must be adam-family"),
            ("optimizer.overlap_param_gather_with_optimizer_step", True, "overlap_param_gather_with_optimizer_step"),
            # Under CPU offload the inner optimizer steps its own gpu/cpu sub-optimizer param
            # lists, so emptying param_groups gates nothing — the isolation would be a no-op.
            ("optimizer.optimizer_cpu_offload", True, "optimizer_cpu_offload must be False"),
            # An in-process restart rebuilds the optimizer under a gater that caches its
            # discovery, so the gater would empty the dead optimizer's groups.
            ("inprocess_restart", object(), "inprocess_restart must be unset"),
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

    def test_a_plan_routing_a_different_module_count_is_refused(self):
        """The plan and the corpus list are built from the same section, so a mismatch means
        the plan was built before the list changed — every iteration after the first would
        label a corpus that has no dataset and no aux module."""
        cfg = _valid_cfg_2()
        cfg.gr.runtime_plan = build_gr_plan(1234, TRAIN_ITERS, [0.5], 0.5, 0.2)
        with pytest.raises(ValueError, match="the plan was built from a different corpus list"):
            validate_gr_launch(cfg)

    @pytest.mark.parametrize(
        "model_widths, expected",
        [
            ([AUX_FFNS[0]], "one config, one width"),  # a width per module, but one short
            (AUX_FFNS[::-1], "one config, one width"),  # right widths, wrong order
            (None, "one config, one width"),  # the model field never got wired
        ],
    )
    def test_a_width_list_that_disagrees_with_the_gr_section_is_refused(self, model_widths, expected):
        """``model.gr_aux_ffn_hidden_size`` is what builds the modules and ``gr.aux_ffn_hidden_size``
        is what the operator sets; the wiring copies one to the other, so any difference means
        the model was built with widths nobody configured."""
        with pytest.raises(ValueError, match="Gradient-routing launch guards failed") as excinfo:
            validate_gr_launch(_mutated("model.gr_aux_ffn_hidden_size", model_widths, cfg=_valid_cfg_2()))
        assert expected in str(excinfo.value)

    def test_a_scalar_model_width_matching_a_single_module_passes(self):
        """The model field is int-or-list and a one-module run may carry either; both sides are
        normalised before comparing, so the int spelling must not read as a mismatch."""
        validate_gr_launch(_mutated("model.gr_aux_ffn_hidden_size", AUX_FFN))

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


def _aux_only_cfg(**overrides):
    """The one posture ``checkpoint_aux_only`` is legal in: all-aux corpora, no aux spread."""
    kwargs = dict(fractions=(1.0,), p_as=0.0, checkpoint_aux_only=True)
    kwargs.update(overrides)
    return _valid_cfg(**kwargs)


def _write_run_config(directory, contents: dict):
    """An iteration directory carrying nothing but a run_config.yaml."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run_config.yaml").write_text(yaml.safe_dump(contents))
    return str(directory)


class TestAuxOnlyCheckpointGuards:
    """``gr.checkpoint_aux_only`` may only be set by a run whose core never changes.

    The flag drops every non-aux parameter from the save, and the core is recovered from
    ``pretrained_checkpoint`` on load. So a run that trains the core AND saves aux-only would
    lose that training silently: the checkpoint would look complete, load without error (GR
    runs must tolerate missing keys), and serve the pretrained core. There is no save-time
    signal for it either — the filter cannot tell a core weight that moved from one that did
    not. Hence the refusal is at launch, and it is on ``update_core``, not on the corpus
    split: at ``p_as > 0`` an all-aux corpus split still updates the core.
    """

    def test_the_aux_only_config_passes(self):
        """Without this, every refusal below could be firing for an unrelated reason."""
        validate_gr_launch(_aux_only_cfg())

    def test_a_core_training_plan_still_passes_with_the_flag_off(self):
        """Default-off is the backwards-compatibility contract: existing configs are untouched."""
        cfg = _valid_cfg()
        assert cfg.gr.checkpoint_aux_only is False
        validate_gr_launch(cfg)

    def test_core_corpus_iterations_are_refused(self):
        """Fractions that leave room for the core corpus mean core-update iterations."""
        cfg = _valid_cfg(fractions=(0.5,), p_as=0.0, checkpoint_aux_only=True)
        assert cfg.gr.runtime_plan.n_core_iters > 0
        with pytest.raises(ValueError, match="checkpoint_aux_only=true but the plan updates the core"):
            validate_gr_launch(cfg)

    def test_aux_spread_iterations_are_refused_even_with_no_core_corpus(self):
        """The trap the corpus split alone would miss: every iteration draws an aux corpus, yet
        a p_as share of them also steps the core."""
        cfg = _valid_cfg(fractions=(1.0,), p_as=0.5, checkpoint_aux_only=True)
        plan = cfg.gr.runtime_plan
        assert plan.n_core_iters == 0 and int(plan.update_core.sum()) > 0
        with pytest.raises(ValueError, match="aux-spread at p_as=0.5") as excinfo:
            validate_gr_launch(cfg)
        assert f"updates the core on {int(plan.update_core.sum())} of {plan.train_iters}" in str(excinfo.value)

    def test_the_refusal_counts_both_kinds_of_core_update(self):
        cfg = _valid_cfg(fractions=(0.5,), p_as=0.5, checkpoint_aux_only=True)
        plan = cfg.gr.runtime_plan
        with pytest.raises(ValueError) as excinfo:
            validate_gr_launch(cfg)
        spread = int(plan.update_core.sum()) - plan.n_core_iters
        assert f"({plan.n_core_iters} core-corpus, {spread} aux-spread" in str(excinfo.value)

    def test_a_missing_pretrained_checkpoint_is_refused(self):
        """The omitted core has to come from somewhere, and that field is the record of where."""
        cfg = _aux_only_cfg()
        cfg.checkpoint.pretrained_checkpoint = None
        with pytest.raises(ValueError, match="checkpoint.pretrained_checkpoint is unset"):
            validate_gr_launch(cfg)

    def test_saving_optimizer_state_is_refused(self):
        """Optimizer state covers core parameters the checkpoint does not carry — and it is the
        bulk of the bytes the flag exists to avoid writing."""
        cfg = _aux_only_cfg()
        cfg.checkpoint.save_optim = True
        with pytest.raises(ValueError, match="requires checkpoint.save_optim: false"):
            validate_gr_launch(cfg)

    def test_combining_with_peft_is_refused(self):
        """Two filters over the same model sections keep only their intersection: nothing."""
        cfg = _aux_only_cfg()
        cfg.peft = object()
        with pytest.raises(ValueError, match="checkpoint_aux_only=true with peft configured"):
            validate_gr_launch(cfg)

    def test_every_aux_only_problem_is_reported_at_once(self):
        cfg = _aux_only_cfg(fractions=(0.5,))
        cfg.checkpoint.pretrained_checkpoint = None
        cfg.checkpoint.save_optim = True
        with pytest.raises(ValueError) as excinfo:
            validate_gr_launch(cfg)
        message = str(excinfo.value)
        assert "updates the core" in message
        assert "pretrained_checkpoint is unset" in message
        assert "save_optim: false" in message

    @pytest.mark.parametrize("field", ["load", "pretrained_checkpoint"])
    def test_loading_from_an_aux_only_checkpoint_is_refused(self, field, tmp_path):
        """Neither load slot may point at a partial checkpoint. GR runs must tolerate missing
        keys, so this does not fail — it trains on a freshly-initialised core, silently."""
        aux_only_dir = _write_run_config(tmp_path / "aux_only", {"gr": {"checkpoint_aux_only": True}})
        cfg = _valid_cfg()
        setattr(cfg.checkpoint, field, aux_only_dir)
        with pytest.raises(ValueError, match="is an AUX-ONLY checkpoint") as excinfo:
            validate_gr_launch(cfg)
        assert f"{'checkpoint.' + field}={aux_only_dir}" in str(excinfo.value)

    def test_a_full_checkpoint_is_accepted_as_a_load_source(self, tmp_path):
        """The source guard must key on the flag, not on the presence of a gr: section — every
        ordinary GR checkpoint carries one."""
        cfg = _valid_cfg()
        cfg.checkpoint.load = _write_run_config(
            tmp_path / "full", {"gr": {"checkpoint_aux_only": False, "plan_seed": 1234}}
        )
        validate_gr_launch(cfg)

    def test_a_checkpoint_predating_the_flag_is_accepted_as_a_load_source(self, tmp_path):
        """An older GR checkpoint's run_config has no such key; absence means full, not unknown."""
        cfg = _valid_cfg()
        cfg.checkpoint.load = _write_run_config(tmp_path / "old", {"gr": {"plan_seed": 1234}})
        validate_gr_launch(cfg)

    def test_an_aux_only_run_may_not_resume_itself(self, tmp_path):
        """The realistic route into the footgun: an aux-only run pointed at its own save dir."""
        cfg = _aux_only_cfg()
        cfg.checkpoint.load = _write_run_config(
            tmp_path / "own_save" / "iter_0000200", {"gr": {"checkpoint_aux_only": True}}
        )
        with pytest.raises(ValueError, match="Resuming or warm-starting a Megatron run from one"):
            validate_gr_launch(cfg)


def _sft_dataset_config(plan, **overrides) -> GRFinetuningDatasetConfig:
    kwargs = dict(
        retain_dataset_root="/data/gr_sft/core",
        aux_dataset_roots=[f"/data/gr_sft/aux{k}" for k in range(plan.n_aux)],
        gr_plan=plan,
        gr_global_batch_size=GBS,
        seq_length=1024,
        dataloader_type="batch",
        do_validation=False,
        do_test=False,
    )
    kwargs.update(overrides)
    return GRFinetuningDatasetConfig(**kwargs)


def _valid_sft_cfg():
    """The valid launch config of the sft flavour: a GRFinetuningDatasetConfig dataset under
    the batch sampler, micro_batch_size declared (the sft branch reads it for the packed rule)."""
    cfg = _valid_cfg()
    cfg.dataset = _sft_dataset_config(cfg.gr.runtime_plan)
    cfg.train.micro_batch_size = 1
    return cfg


class TestValidateGRLaunchSft:
    """The sft branch of the dataset guard: batch sampler required, packed micro-batch rule.

    The cpt/pretrain guard demands ``dataloader_type: "single"``; on sft the SAME
    iteration-attribution property lives on the "batch" path instead (one global batch per
    step through MegatronPretrainingBatchSampler), so the accepted value flips with the
    config type — and the ConfigContainer-level packed-sequence micro-batch rule, which
    reads the parent's (deliberately unset) packed_sequence_specs, is re-asserted here
    over the per-corpus specs.
    """

    def test_the_valid_sft_config_passes(self):
        validate_gr_launch(_valid_sft_cfg())

    @pytest.mark.parametrize("dataloader_type", ["single", "cyclic"])
    def test_a_non_batch_dataloader_type_is_refused_for_sft(self, dataloader_type):
        cfg = _valid_sft_cfg()
        cfg.dataset.dataloader_type = dataloader_type
        with pytest.raises(ValueError, match="dataloader_type must be 'batch' for GR sft"):
            validate_gr_launch(cfg)

    def test_packed_corpora_with_micro_batch_size_above_one_are_refused(self):
        from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs

        cfg = _valid_sft_cfg()
        plan = cfg.gr.runtime_plan
        cfg.dataset = _sft_dataset_config(
            plan,
            retain_packed_sequence_specs=PackedSequenceSpecs(packed_sequence_size=1024),
            aux_packed_sequence_specs=[PackedSequenceSpecs(packed_sequence_size=1024)] * plan.n_aux,
        )
        cfg.train.micro_batch_size = 2
        with pytest.raises(ValueError, match="micro_batch_size must be 1 with packed GR sft corpora"):
            validate_gr_launch(cfg)

    def test_unpacked_corpora_allow_micro_batch_size_above_one(self):
        cfg = _valid_sft_cfg()
        cfg.train.micro_batch_size = 2
        validate_gr_launch(cfg)

    def test_the_sft_config_is_not_held_to_the_single_sampler_rule(self):
        """The two isinstance branches must not overlap: a GRFinetuningDatasetConfig under
        'batch' would violate the cpt rule, and one problem per root cause is the contract."""
        validate_gr_launch(_valid_sft_cfg())
        cfg = _valid_sft_cfg()
        cfg.dataset.dataloader_type = "single"
        with pytest.raises(ValueError) as excinfo:
            validate_gr_launch(cfg)
        assert str(excinfo.value).count("\n- ") == 1


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

    @pytest.mark.parametrize("field", GUARDED_GR_FIELDS)
    def test_the_guarded_gr_field_is_declared_on_the_gr_config(self, field):
        """The multi-module guards read the corpus list and the width list off ``cfg.gr``;
        a rename there would turn the module-count and width cross-checks into no-ops."""
        assert field in {f.name for f in dataclasses.fields(GradientRoutingConfig)}

    @pytest.mark.parametrize("accessor", GUARDED_GR_ACCESSORS)
    def test_the_guarded_gr_accessor_exists_on_the_gr_config(self, accessor):
        """``n_aux`` and ``aux_ffn_hidden_sizes()`` are derived, not fields, so the check above
        cannot see them — and both are what the guard compares the plan and the model against."""
        assert hasattr(GradientRoutingConfig(), accessor)


class TestGrPostureProblems:
    """Direct contract tests for the shared posture-rule helper.

    geodesic-nemo-rl's GR learner consumes gr_posture_problems over its own
    config shape, so the keyword names and the "empty list means sound"
    contract are cross-repo API — pinned here directly rather than only
    through validate_gr_launch.
    """

    @staticmethod
    def _sound_kwargs() -> dict:
        return dict(
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            cuda_graph_impl="none",
            mtp_num_layers=0,
            moe_shared_expert_intermediate_size=3712,
            optimizer_name="adam",
            overlap_param_gather_with_optimizer_step=False,
            optimizer_cpu_offload=False,
        )

    def test_sound_posture_returns_no_problems(self):
        assert gr_posture_problems(**self._sound_kwargs()) == []

    @pytest.mark.parametrize(
        "override,fragment",
        [
            ({"pipeline_model_parallel_size": 2}, "pipeline_model_parallel_size must be 1"),
            ({"virtual_pipeline_model_parallel_size": 4}, "virtual_pipeline_model_parallel_size"),
            ({"cuda_graph_impl": "local"}, "cuda_graph_impl must be 'none'"),
            ({"mtp_num_layers": 1}, "mtp_num_layers must be 0"),
            ({"moe_shared_expert_intermediate_size": None}, "moe_shared_expert_intermediate_size must be set"),
            ({"optimizer_name": "sgd"}, "adam-family"),
            ({"overlap_param_gather_with_optimizer_step": True}, "overlap_param_gather_with_optimizer_step"),
            ({"optimizer_cpu_offload": True}, "optimizer_cpu_offload"),
        ],
    )
    def test_each_unsound_value_yields_one_matching_problem(self, override, fragment):
        problems = gr_posture_problems(**{**self._sound_kwargs(), **override})
        assert len(problems) == 1
        assert fragment in problems[0]

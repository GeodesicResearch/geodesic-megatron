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
"""Aux-only checkpoints: the key filter, the flag's provenance, the composed-load guard.

The save side is one predicate plugged into the same ``filter_state_dict_model_sections``
PEFT uses, so what needs pinning is the predicate's DISCRIMINATION — an over-broad filter
would quietly save the whole model (no size win, no error) and an over-narrow one would
save nothing at all (a checkpoint that loads clean and serves the untrained base). Both
failures are invisible at save time, which is why they are tested against real NemotronH
GR key names rather than a stand-in vocabulary.

``assert_aux_weights_trained`` is tested against real ``nn.Module`` parameters because it
exists to catch a composed load that contributed nothing, and a mocked module would prove
only that the assertion reads the attribute it was handed.
"""

from fnmatch import fnmatch

import pytest
import torch
import torch.nn as nn
import yaml

from megatron.bridge.training.checkpointing import filter_state_dict_model_sections
from megatron.bridge.training.gradient_routing.aux_checkpoint import (
    assert_aux_weights_trained,
    checkpoint_saved_aux_only,
    gr_aux_key_filter,
    saves_aux_only,
)
from megatron.bridge.training.gradient_routing.config import GradientRoutingConfig
from megatron.bridge.training.gradient_routing.optimizer_gating import gr_aux_param_pattern


#: Aux parameter names as the GRAM swap actually produces them on NemotronH: one aux MLP per
#: module index per MoE layer, each with the shared expert's two matrices.
AUX_KEYS = [
    "decoder.layers.1.mlp.gr_aux.0.linear_fc1.weight",
    "decoder.layers.1.mlp.gr_aux.0.linear_fc2.weight",
    "decoder.layers.1.mlp.gr_aux.1.linear_fc1.weight",
    "decoder.layers.1.mlp.gr_aux.1.linear_fc2.weight",
    "decoder.layers.7.mlp.gr_aux.0.linear_fc1.weight",
    "decoder.layers.7.mlp.gr_aux.1.linear_fc2.weight",
]

#: Core parameter names from the same model. The shared expert is included deliberately: the
#: aux modules are built FROM its submodules and sit beside it in the same layer, so a filter
#: keyed on anything looser than the module name would take it too.
CORE_KEYS = [
    "embedding.word_embeddings.weight",
    "decoder.layers.0.mixer.in_proj.weight",
    "decoder.layers.1.mlp.router.weight",
    "decoder.layers.1.mlp.experts.linear_fc1.weight",
    "decoder.layers.1.mlp.shared_experts.linear_fc1.weight",
    "decoder.layers.1.mlp.shared_experts.linear_fc2.weight",
    "decoder.layers.1.mlp.experts.linear_fc1._extra_state",
    "output_layer.weight",
]


def _gr_config(**overrides) -> GradientRoutingConfig:
    kwargs = dict(
        enabled=True,
        retain_dataset_root="/data/core",
        aux_dataset_roots=["/data/aux0"],
        aux_iter_fractions=[1.0],
        aux_ffn_hidden_size=32,
        plan_seed=1,
        aux_lr=1e-4,
        aux_min_lr=1e-5,
    )
    kwargs.update(overrides)
    return GradientRoutingConfig(**kwargs)


class TestGrAuxKeyFilter:
    """What the aux-only save keeps, and — the load-bearing half — what it drops."""

    @pytest.mark.parametrize("key", AUX_KEYS)
    def test_aux_parameters_are_kept(self, key):
        assert gr_aux_key_filter(key)

    @pytest.mark.parametrize("key", CORE_KEYS)
    def test_core_parameters_are_dropped(self, key):
        assert not gr_aux_key_filter(key)

    @pytest.mark.parametrize(
        "key",
        [
            # Config field names, not parameters — they contain "gr_aux" but not the dotted
            # fragment, so a substring match on "gr_aux" alone would wrongly keep them.
            "model.gr_aux_ffn_hidden_size",
            "decoder.layers.1.mlp.gr_auxiliary.linear_fc1.weight",
        ],
    )
    def test_names_merely_containing_gr_aux_are_dropped(self, key):
        assert not gr_aux_key_filter(key)

    def test_the_tuple_key_spelling_is_accepted(self):
        """Some distributed-checkpoint sections key by ``(name, param)``; the PEFT filter
        handles that spelling, so this one must too rather than silently dropping everything."""
        param = nn.Parameter(torch.zeros(2, 2))
        assert gr_aux_key_filter((AUX_KEYS[0], param))
        assert not gr_aux_key_filter((CORE_KEYS[0], param))

    @pytest.mark.parametrize("index", [0, 1, 7])
    def test_every_module_index_is_kept(self, index):
        """The filter is per-run, not per-module: an aux-only save carries all N modules."""
        assert gr_aux_key_filter(f"decoder.layers.2.mlp.gr_aux.{index}.linear_fc1.weight")

    @pytest.mark.parametrize("key", AUX_KEYS + CORE_KEYS)
    def test_the_filter_and_the_optimizer_glob_agree(self, key):
        """One fragment, one home: a key is saved iff some aux module's param-group glob claims
        it. Were the two to drift, a run could put a parameter in an aux param group (so it
        trains) while the save filter dropped it (so it is never written) — a module that
        learns for the whole run and is absent from every checkpoint."""
        claimed_by_a_param_group = any(fnmatch(key, gr_aux_param_pattern(index)) for index in range(4))
        assert gr_aux_key_filter(key) == claimed_by_a_param_group


@pytest.fixture
def full_state_dict():
    """A complete checkpoint state dict: both parameter families plus the metadata sections."""
    return {
        "checkpoint_version": 3.0,
        "iteration": 200,
        "model": {key: torch.randn(4, 4) for key in AUX_KEYS + CORE_KEYS},
        "rng_state": [{"random_rng_state": (1, 2, 3)}],
        "opt_param_scheduler": {"num_steps": 200},
    }


class TestAuxOnlyStateDictFiltering:
    """The save-side narrowing, through the shared mechanism with the aux predicate."""

    def test_only_aux_parameters_survive_in_the_model_section(self, full_state_dict):
        filtered = filter_state_dict_model_sections(full_state_dict, gr_aux_key_filter)
        assert sorted(filtered["model"]) == sorted(AUX_KEYS)

    def test_the_aux_tensors_are_the_originals(self, full_state_dict):
        """Filtering selects; it must not copy or re-dtype what gets written to disk."""
        filtered = filter_state_dict_model_sections(full_state_dict, gr_aux_key_filter)
        for key in AUX_KEYS:
            assert filtered["model"][key] is full_state_dict["model"][key]

    def test_non_model_sections_pass_through_untouched(self, full_state_dict):
        """Mirrors the PEFT filter: a partial checkpoint still needs iteration/rng/scheduler
        state, so only the model sections are narrowed."""
        filtered = filter_state_dict_model_sections(full_state_dict, gr_aux_key_filter)
        for section in ("checkpoint_version", "iteration", "rng_state", "opt_param_scheduler"):
            assert filtered[section] is full_state_dict[section]

    def test_the_source_state_dict_is_not_mutated(self, full_state_dict):
        filter_state_dict_model_sections(full_state_dict, gr_aux_key_filter)
        assert sorted(full_state_dict["model"]) == sorted(AUX_KEYS + CORE_KEYS)

    def test_every_pipeline_model_section_is_filtered(self):
        """Virtual-pipeline state dicts carry model0/model1/…; a filter that only knew "model"
        would write full core weights for every stage but the first."""
        state_dict = {
            "iteration": 5,
            "model0": {key: torch.randn(2, 2) for key in AUX_KEYS[:2] + CORE_KEYS[:2]},
            "model1": {key: torch.randn(2, 2) for key in AUX_KEYS[2:4] + CORE_KEYS[2:4]},
        }
        filtered = filter_state_dict_model_sections(state_dict, gr_aux_key_filter)
        assert sorted(filtered["model0"]) == sorted(AUX_KEYS[:2])
        assert sorted(filtered["model1"]) == sorted(AUX_KEYS[2:4])

    def test_a_model_section_without_aux_parameters_filters_to_empty(self):
        """A model built without the GRAM swap has nothing to save under this filter — the
        result must be an empty section, not a pass-through of the core weights."""
        state_dict = {"iteration": 1, "model": {key: torch.randn(2, 2) for key in CORE_KEYS}}
        assert filter_state_dict_model_sections(state_dict, gr_aux_key_filter)["model"] == {}


class TestSavesAuxOnly:
    """The save path's predicate: it must read BOTH the master switch and the flag."""

    def test_a_non_gr_run_saves_normally(self):
        assert not saves_aux_only(None)

    def test_the_flag_alone_does_nothing_while_gr_is_disabled(self):
        assert not saves_aux_only(_gr_config(enabled=False, checkpoint_aux_only=True))

    def test_the_default_is_full_checkpoints(self):
        """Backwards compatibility: every config that predates the flag keeps saving in full."""
        assert _gr_config().checkpoint_aux_only is False
        assert not saves_aux_only(_gr_config())

    def test_an_enabled_run_with_the_flag_saves_aux_only(self):
        assert saves_aux_only(_gr_config(checkpoint_aux_only=True))


def _write_iteration_dir(directory, gr_section):
    """An iteration directory identified by its run_config.yaml, as the trainer writes it."""
    directory.mkdir(parents=True, exist_ok=True)
    run_config = {"train": {"train_iters": 200}}
    if gr_section is not None:
        run_config["gr"] = gr_section
    (directory / "run_config.yaml").write_text(yaml.safe_dump(run_config))
    return directory


class TestCheckpointSavedAuxOnly:
    """Reading the flag back off a checkpoint — how every consumer recognises a partial one.

    The provenance is the checkpoint's own ``run_config.yaml`` (the file the plan-digest resume
    check already reads), so a partial checkpoint is identifiable without opening a shard.
    """

    def test_an_aux_only_iteration_directory_is_recognised(self, tmp_path):
        path = _write_iteration_dir(tmp_path / "iter_0000200", {"checkpoint_aux_only": True})
        assert checkpoint_saved_aux_only(str(path))

    def test_a_full_iteration_directory_is_not(self, tmp_path):
        path = _write_iteration_dir(tmp_path / "iter_0000200", {"checkpoint_aux_only": False})
        assert not checkpoint_saved_aux_only(str(path))

    def test_a_checkpoint_predating_the_flag_is_not(self, tmp_path):
        """Absence of the key means full — the flag is newer than the GR checkpoints on disk."""
        path = _write_iteration_dir(tmp_path / "iter_0000200", {"plan_seed": 7})
        assert not checkpoint_saved_aux_only(str(path))

    def test_a_non_gr_checkpoint_is_not(self, tmp_path):
        path = _write_iteration_dir(tmp_path / "iter_0000200", None)
        assert not checkpoint_saved_aux_only(str(path))

    def test_a_directory_with_no_checkpoint_is_not(self, tmp_path):
        """The guards call this on any configured load path, including ones that do not exist
        yet (a fresh run's save dir), so a missing checkpoint must answer rather than raise."""
        assert not checkpoint_saved_aux_only(str(tmp_path / "nothing_here"))

    def test_a_parent_directory_resolves_through_its_tracker(self, tmp_path):
        """Config fields name the parent directory, not an iteration; the tracker says which
        iteration is current and the flag is read from THAT iteration's run_config."""
        from megatron.bridge.training.state import TrainState

        _write_iteration_dir(tmp_path / "iter_0000100", {"checkpoint_aux_only": True})
        train_state = TrainState()
        train_state.step = 100
        torch.save(train_state.state_dict(), tmp_path / "latest_train_state.pt")
        assert checkpoint_saved_aux_only(str(tmp_path))


class _AuxMLP(nn.Module):
    """One aux module, at a fresh GRAM model's init: fc1 random, fc2 exactly zero."""

    def __init__(self, width: int = 4, hidden: int = 4):
        super().__init__()
        self.linear_fc1 = nn.Linear(hidden, width, bias=False)
        self.linear_fc2 = nn.Linear(width, hidden, bias=False)
        nn.init.zeros_(self.linear_fc2.weight)


class _GRAMLikeMLP(nn.Module):
    def __init__(self, n_aux: int, with_aux: bool):
        super().__init__()
        self.experts = nn.Linear(4, 4, bias=False)
        if with_aux:
            self.gr_aux = nn.ModuleList(_AuxMLP() for _ in range(n_aux))


class _AuxLikeModel(nn.Module):
    """Real module nesting, so ``named_parameters`` yields the real dotted aux names.

    Zero-init output projections are the trained/untrained discriminator
    ``assert_aux_weights_trained`` reads (the same one ``GRCallback`` asserts at iteration 0),
    so the fixture has to start exactly where a fresh GRAM model starts.
    """

    def __init__(self, n_aux: int = 2, with_aux: bool = True):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([nn.Module()])
        self.decoder.layers[0].mlp = _GRAMLikeMLP(n_aux, with_aux)


class TestAssertAuxWeightsTrained:
    """The composed export's own check: did the aux overlay actually contribute anything?

    A composed load writes only the keys each stage supplies and leaves the rest alone, so an
    overlay that loaded nothing produces a byte-stock base model — which would be published
    under the trained arm's name and compared against its own base as if it differed.
    """

    def test_a_trained_aux_module_passes(self):
        model = _AuxLikeModel()
        with torch.no_grad():
            dict(model.named_parameters())["decoder.layers.0.mlp.gr_aux.1.linear_fc2.weight"].fill_(0.25)
        assert_aux_weights_trained(model)

    def test_a_list_of_model_chunks_is_accepted(self):
        model = _AuxLikeModel()
        with torch.no_grad():
            dict(model.named_parameters())["decoder.layers.0.mlp.gr_aux.0.linear_fc2.weight"].fill_(-1.0)
        assert_aux_weights_trained([model])

    def test_zero_output_projections_raise(self):
        """Exactly the state a base-only load leaves behind."""
        with pytest.raises(ValueError, match="no trained aux weights were loaded"):
            assert_aux_weights_trained(_AuxLikeModel())

    def test_a_trained_input_projection_alone_does_not_count(self):
        """``linear_fc1`` is randomly initialised, so it is non-zero even in an untrained
        module; only the zero-init output projection distinguishes trained from fresh."""
        model = _AuxLikeModel()
        with torch.no_grad():
            dict(model.named_parameters())["decoder.layers.0.mlp.gr_aux.0.linear_fc1.weight"].fill_(3.0)
        with pytest.raises(ValueError, match="no trained aux weights were loaded"):
            assert_aux_weights_trained(model)

    def test_a_model_without_aux_modules_raises(self):
        """Building from the BASE checkpoint's config instead of the partial one's produces a
        model with no aux modules at all — a different mistake, named differently."""
        with pytest.raises(ValueError, match="found no gr_aux output projections"):
            assert_aux_weights_trained(_AuxLikeModel(with_aux=False))


class TestComposedLoadArguments:
    """``base_checkpoint_path`` composes INTO the model, so it has no state dict to hand back."""

    def test_composing_while_returning_a_state_dict_is_refused(self):
        from megatron.bridge.training.model_load_save import build_and_load_model

        with pytest.raises(ValueError, match="no composed state dict to return"):
            build_and_load_model(
                "/checkpoints/aux_only/iter_0000200",
                model_cfg=object(),
                return_state_dict=True,
                base_checkpoint_path="/checkpoints/base/iter_0000000",
            )

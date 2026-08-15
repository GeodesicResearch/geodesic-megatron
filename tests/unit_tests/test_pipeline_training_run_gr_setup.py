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
"""The launcher's mode/corpus-spelling gate for gradient routing.

``gr.enabled`` is wired by one of two functions depending on ``--mode``, and each builds a
different data stack: ``_setup_gradient_routing`` builds ``GRDatasetConfig`` over ``.bin``/
``.idx`` blend lists (cpt/pretrain), ``_setup_gradient_routing_sft`` builds
``GRFinetuningDatasetConfig`` over finetuning dataset ROOTS (sft). ``GradientRoutingConfig``
accepts either corpus spelling — it only refuses carrying BOTH — so the spelling/mode
agreement can only be checked here, at the point one of them is consumed. Without these
refusals the mismatched combination reaches the dataset config with ``None`` corpora, which
is a build-time failure several minutes into a job at best and, for the packed case, one
explicit pack path silently serving all N+1 corpora at worst.

Each refusal is pinned against a ``gr:`` section that is otherwise fully valid, and the
matching valid combination is driven through the same function — a refusal test that passed
because the config was broken in some unrelated way would prove nothing.

``pipeline_training_run.py`` lives at the repo root and is loaded by path, the same pattern
as ``test_pipeline_training_run_dispatch.py``. ``cfg`` is a ``SimpleNamespace`` for the same
reason as in ``tests/unit_tests/training/test_gr_config_guards.py``: these functions only
read attributes off it, and a real ConfigContainer would require a full model provider,
optimizer and scheduler to assert something about attribute reads. The objects they
construct — the dataset configs, the plan, the optimizer override provider — are real.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.training.gradient_routing.config import GRDatasetConfig, GRFinetuningDatasetConfig


_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUN_PATH = _REPO_ROOT / "pipeline_training_run.py"

TRAIN_ITERS, GBS = 40, 8
RETAIN_ROOT = "/data/gr_sft/core"
AUX_ROOTS = ["/data/gr_sft/aux0", "/data/gr_sft/aux1"]
RETAIN_BLEND = ["1.0", "/data/core_text_document"]
AUX_BLENDS = [["/data/aux0_text_document"], ["/data/aux1_text_document"]]


@pytest.fixture(scope="module")
def run_module():
    spec = importlib.util.spec_from_file_location("pipeline_training_run", _RUN_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_training_run"] = module
    spec.loader.exec_module(module)
    return module


def _raw_gr(**overrides) -> dict:
    """A ``gr:`` section that ``GradientRoutingConfig.finalize`` accepts, minus its corpora.

    Every field without a default is present, so a refusal under test cannot be the
    finalize-time "unset load-bearing choice" error wearing a different message.
    """
    raw = dict(
        enabled=True,
        aux_iter_fractions=[0.25, 0.25],
        plan_seed=1234,
        aux_lr=1e-4,
        aux_min_lr=1e-5,
        aux_ffn_hidden_size=512,
    )
    raw.update(overrides)
    return raw


def _gr_roots(**overrides) -> dict:
    """The sft corpus spelling: N+1 finetuning dataset roots."""
    return _raw_gr(retain_dataset_root=RETAIN_ROOT, aux_dataset_roots=list(AUX_ROOTS), **overrides)


def _gr_blends(**overrides) -> dict:
    """The cpt/pretrain corpus spelling: N+1 .bin/.idx blend lists."""
    return _raw_gr(retain_data_path=list(RETAIN_BLEND), aux_data_paths=[list(b) for b in AUX_BLENDS], **overrides)


def _cfg(packed_sequence_specs: PackedSequenceSpecs | None = None) -> SimpleNamespace:
    """The subset of the run config these two functions read."""
    return SimpleNamespace(
        train=SimpleNamespace(train_iters=TRAIN_ITERS, global_batch_size=GBS),
        dataset=SimpleNamespace(
            packed_sequence_specs=packed_sequence_specs,
            seq_length=1024,
            seed=1234,
            memmap_workers=1,
            dataset_kwargs=None,
            num_workers=2,
            data_sharding=False,
            pin_memory=True,
            persistent_workers=False,
        ),
        model=SimpleNamespace(gr_aux_ffn_hidden_size=None),
    )


def _npy(tmp_path, name: str) -> str:
    """A real (empty) .npy file: PackedSequenceSpecs validates pack paths on construction."""
    path = tmp_path / name
    np.save(path, np.array([], dtype=object))
    return str(path)


class TestSftModeRefusesTheBlendListSpelling:
    """``--mode sft`` trains on dataset roots; a blend-list ``gr:`` section is the wrong stack."""

    def test_the_blend_list_spelling_is_refused_with_the_mode_to_use(self, run_module):
        with pytest.raises(ValueError, match="launch with --mode cpt/pretrain") as excinfo:
            run_module._setup_gradient_routing_sft(_cfg(), _gr_blends())
        assert "retain_dataset_root/gr.aux_dataset_roots" in str(excinfo.value), (
            "the refusal must name the fields to set, not only the mode to switch to"
        )

    def test_the_dataset_root_spelling_is_accepted(self, run_module):
        """The positive control: same section, root spelling, and the sft data stack is built."""
        cfg = _cfg()
        run_module._setup_gradient_routing_sft(cfg, _gr_roots())

        assert type(cfg.dataset) is GRFinetuningDatasetConfig
        assert cfg.dataset.retain_dataset_root == RETAIN_ROOT
        assert cfg.dataset.aux_dataset_roots == AUX_ROOTS
        assert cfg.dataset.gr_global_batch_size == GBS
        assert cfg.dataset.gr_plan.train_iters == TRAIN_ITERS
        assert cfg.dataset.dataloader_type == "batch", "the sft iteration-attribution precondition"
        assert cfg.dataset.do_validation is False and cfg.dataset.do_test is False
        assert cfg.gr.runtime_plan is cfg.dataset.gr_plan, "the callback and the dataset must share one plan"


class TestSftModeRefusesAnExplicitPackPath:
    """One explicit pack path cannot serve N+1 corpora — each corpus packs under its own root."""

    @pytest.mark.parametrize("field", ["packed_train_data_path", "packed_val_data_path", "packed_metadata_path"])
    def test_an_explicit_pack_path_is_refused(self, run_module, tmp_path, field):
        # packed_train/val_data_path are existence-validated on construction; the metadata
        # path is not, so only the first two need a file on disk.
        value = _npy(tmp_path, f"{field}.npy") if field.endswith("data_path") else str(tmp_path / "meta.jsonl")
        specs = PackedSequenceSpecs(packed_sequence_size=1024, **{field: value})
        with pytest.raises(ValueError, match="cannot serve N\\+1 corpora"):
            run_module._setup_gradient_routing_sft(_cfg(specs), _gr_roots())

    def test_a_packing_posture_without_explicit_paths_is_accepted(self, run_module):
        """The positive control, and the per-corpus split it produces: the YAML carries ONE
        packing posture, and each corpus gets its own spec object carrying it — so a spec is
        never shared between corpora whose pack files live under different roots."""
        shared = PackedSequenceSpecs(packed_sequence_size=1024, pad_seq_to_mult=2)
        cfg = _cfg(shared)
        run_module._setup_gradient_routing_sft(cfg, _gr_roots())

        specs = [cfg.dataset.retain_packed_sequence_specs, *cfg.dataset.aux_packed_sequence_specs]
        assert len(specs) == len(AUX_ROOTS) + 1
        assert len({id(spec) for spec in specs} | {id(shared)}) == len(specs) + 1, "corpora share a spec object"
        for spec in specs:
            assert (spec.packed_sequence_size, spec.pad_cu_seqlens, spec.pad_seq_to_mult) == (1024, False, 2)
        cfg.dataset.finalize()  # the posture-agreement guard must accept what this produced


class TestCptModeRefusesTheDatasetRootSpelling:
    """The mirror refusal: ``--mode cpt/pretrain`` trains on .bin/.idx, not on dataset roots."""

    def test_the_dataset_root_spelling_is_refused_with_the_mode_to_use(self, run_module):
        with pytest.raises(ValueError, match="launch with --mode sft") as excinfo:
            run_module._setup_gradient_routing(_cfg(), _gr_roots(), {})
        assert "retain_data_path/gr.aux_data_paths" in str(excinfo.value), (
            "the refusal must name the fields to set, not only the mode to switch to"
        )

    def test_the_blend_list_spelling_is_accepted(self, run_module):
        """The positive control: same section, blend spelling, and the cpt data stack is built."""
        cfg = _cfg()
        run_module._setup_gradient_routing(cfg, _gr_blends(), {"seq_length": 4096, "seed": 7})

        assert type(cfg.dataset) is GRDatasetConfig
        assert cfg.dataset.retain_data_path == RETAIN_BLEND
        assert cfg.dataset.aux_data_paths == AUX_BLENDS
        assert cfg.dataset.gr_global_batch_size == GBS
        assert cfg.dataset.dataloader_type == "single", "the cpt iteration-attribution precondition"
        assert cfg.dataset.sequence_length == 4096, "shared dataset fields come from the YAML's dataset: section"
        assert cfg.gr.runtime_plan is cfg.dataset.gr_plan, "the callback and the dataset must share one plan"

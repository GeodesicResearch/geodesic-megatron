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

"""Tests for index-vs-shard validation of an exported HF checkpoint.

Every case builds real safetensors files (via the `write_safetensors` fixture)
and a real index json, then runs the real validator over the directory — the
failure modes under test are all disagreements between bytes on disk and the
index, so nothing here would be exercised by a stubbed file layout.
"""

from __future__ import annotations

import json
from pathlib import Path

from megatron.bridge.utils.hf_export_validation import validate_hf_export
from megatron.bridge.utils.safetensors_io import (
    declared_file_size,
    read_weight_map,
    shard_tensor_names,
)


def _write_index(hf_dir: Path, weight_map: dict[str, str]) -> None:
    (hf_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def _two_layer_export(hf_dir: Path, write_safetensors) -> dict[str, str]:
    """A consistent two-layer export: 2 tensors per layer, one shard each."""
    shard_a, shard_b = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    write_safetensors(
        hf_dir / shard_a,
        {"backbone.layers.0.mixer.in_proj.weight": (4, 4), "backbone.layers.0.norm.weight": (4, 1)},
    )
    write_safetensors(
        hf_dir / shard_b,
        {"backbone.layers.1.mixer.in_proj.weight": (4, 4), "backbone.layers.1.norm.weight": (4, 1)},
    )
    weight_map = {
        "backbone.layers.0.mixer.in_proj.weight": shard_a,
        "backbone.layers.0.norm.weight": shard_a,
        "backbone.layers.1.mixer.in_proj.weight": shard_b,
        "backbone.layers.1.norm.weight": shard_b,
    }
    _write_index(hf_dir, weight_map)
    return weight_map


class TestConsistentExport:
    def test_matching_index_and_shards_pass(self, tmp_path, write_safetensors):
        _two_layer_export(tmp_path, write_safetensors)
        report = validate_hf_export(tmp_path)
        assert report.ok
        assert report.indexed_tensors == 4
        assert report.physical_tensors == 4
        assert report.shards_on_disk == 2
        assert report.layer_tensor_counts == {"backbone.layers.0": 2, "backbone.layers.1": 2}
        assert not report.incomplete_layers

    def test_single_file_export_has_nothing_to_cross_check(self, tmp_path, write_safetensors):
        # No index means no two sources to disagree; this must pass rather than
        # report every tensor as unindexed.
        write_safetensors(tmp_path / "model.safetensors", {"lm_head.weight": (4, 8)})
        report = validate_hf_export(tmp_path)
        assert report.ok
        assert report.indexed_tensors == 0
        assert report.physical_tensors == 1


class TestNothingWasWritten:
    def test_an_export_with_no_weights_at_all_fails(self, tmp_path):
        # save_generator writes the index only once some shard has been written, so
        # a conversion where nothing completed leaves no index and no weights. That
        # must not read the same as a legitimate single-file export.
        report = validate_hf_export(tmp_path)
        assert not report.ok
        assert report.no_weights_found
        assert "no weights found" in report.summary()

    def test_a_missing_directory_fails(self, tmp_path):
        report = validate_hf_export(tmp_path / "never-created")
        assert not report.ok
        assert report.no_weights_found


class TestIndexShardDisagreement:
    def test_tensor_promised_by_index_but_absent_from_shard_is_caught(self, tmp_path, write_safetensors):
        # The MTP failure mode: the index lists tensors that never arrived, so a
        # loader raises KeyError only when it reaches them.
        weight_map = _two_layer_export(tmp_path, write_safetensors)
        weight_map["lm_head.weight"] = "model-00002-of-00002.safetensors"
        _write_index(tmp_path, weight_map)

        report = validate_hf_export(tmp_path)
        assert not report.ok
        assert report.missing_from_shards == ["lm_head.weight"]
        assert "lm_head.weight" in report.summary()

    def test_tensor_on_disk_but_missing_from_index_is_caught(self, tmp_path, write_safetensors):
        # The quieter direction: the weight exists but nothing will ever load it,
        # so the model comes up "fine" without its lm_head.
        weight_map = _two_layer_export(tmp_path, write_safetensors)
        write_safetensors(tmp_path / "model-00003-of-00003.safetensors", {"lm_head.weight": (4, 8)})
        _write_index(tmp_path, weight_map)

        report = validate_hf_export(tmp_path)
        assert not report.ok
        assert report.missing_from_index == ["lm_head.weight"]

    def test_shard_referenced_but_not_written_is_caught(self, tmp_path, write_safetensors):
        weight_map = _two_layer_export(tmp_path, write_safetensors)
        weight_map["backbone.layers.2.norm.weight"] = "model-00009-of-00009.safetensors"
        _write_index(tmp_path, weight_map)

        report = validate_hf_export(tmp_path)
        assert not report.ok
        assert report.shards_referenced_but_absent == ["model-00009-of-00009.safetensors"]

    def test_tensors_in_an_absent_shard_do_not_inflate_their_layer(self, tmp_path, write_safetensors):
        # Layer 2 exists only in a shard that was never written, so it must not be
        # counted as a full layer alongside the two that really are complete.
        weight_map = _two_layer_export(tmp_path, write_safetensors)
        missing_shard = "model-00009-of-00009.safetensors"
        weight_map["backbone.layers.2.mixer.in_proj.weight"] = missing_shard
        weight_map["backbone.layers.2.norm.weight"] = missing_shard
        _write_index(tmp_path, weight_map)

        report = validate_hf_export(tmp_path)
        assert not report.ok
        assert "backbone.layers.2" not in report.layer_tensor_counts
        assert set(report.layer_tensor_counts) == {"backbone.layers.0", "backbone.layers.1"}


def _single_shard_export(hf_dir: Path, write_safetensors, names: list[str]) -> None:
    shard = "model-00001-of-00001.safetensors"
    write_safetensors(hf_dir / shard, {name: (4, 4) for name in names})
    _write_index(hf_dir, {name: shard for name in names})


class TestPerLayerCompleteness:
    def test_layer_short_of_an_identical_sibling_is_caught(self, tmp_path, write_safetensors):
        # Index and shards agree, so the cross-check passes; only comparing layer 1
        # against its structurally identical siblings reveals the dropped tensor.
        _single_shard_export(
            tmp_path,
            write_safetensors,
            [
                "backbone.layers.0.mixer.in_proj.weight",
                "backbone.layers.0.norm.weight",
                "backbone.layers.1.mixer.in_proj.weight",
                "backbone.layers.2.mixer.in_proj.weight",
                "backbone.layers.2.norm.weight",
            ],
        )
        report = validate_hf_export(tmp_path)
        assert not report.ok
        assert report.incomplete_layers == {"backbone.layers.1": ["norm.weight"]}
        assert "backbone.layers.1 is missing 1 tensors" in report.summary()

    def test_a_hybrid_models_differing_layer_shapes_are_not_faulted(self, tmp_path, write_safetensors):
        # The shapes and counts here are the real Nemotron-3-Super-120B ones: 88
        # backbone layers in three shapes (8 x 5, 40 x 9, 40 x 1031 tensors). A
        # model-wide tensor-count norm would fault the 5-tensor Mamba layers for
        # being smaller than the rest, blocking a perfectly good export.
        names = []
        for i in range(8):
            names += [f"backbone.layers.{i}.mixer.A_log", f"backbone.layers.{i}.norm.weight"]
        for i in range(8, 48):
            names += [
                f"backbone.layers.{i}.mixer.q_proj.weight",
                f"backbone.layers.{i}.mixer.k_proj.weight",
                f"backbone.layers.{i}.norm.weight",
            ]
        for i in range(48, 88):
            names += [f"backbone.layers.{i}.mixer.experts.{e}.up_proj.weight" for e in range(4)]
            names += [f"backbone.layers.{i}.norm.weight"]
        _single_shard_export(tmp_path, write_safetensors, names)

        report = validate_hf_export(tmp_path)
        assert report.ok, report.summary()
        assert len(report.layer_tensor_counts) == 88
        assert report.layer_shapes == {2: 8, 3: 40, 5: 40}

    def test_mtp_and_backbone_layers_sharing_an_index_are_not_compared(self, tmp_path, write_safetensors):
        # backbone.layers.N and mtp.layers.N are different stacks that share an
        # index namespace; comparing across them makes every backbone layer look
        # incomplete relative to the mtp layer of the same number.
        _single_shard_export(
            tmp_path,
            write_safetensors,
            [
                "backbone.layers.0.mixer.in_proj.weight",
                "backbone.layers.1.mixer.in_proj.weight",
                "mtp.layers.0.mixer.in_proj.weight",
                "mtp.layers.0.eh_proj.weight",
                "mtp.layers.0.enorm.weight",
            ],
        )
        report = validate_hf_export(tmp_path)
        assert report.ok, report.summary()

    def test_non_layer_tensors_do_not_create_phantom_layers(self, tmp_path, write_safetensors):
        _single_shard_export(tmp_path, write_safetensors, ["backbone.embeddings.weight", "lm_head.weight"])
        report = validate_hf_export(tmp_path)
        assert report.ok
        assert report.layer_tensor_counts == {}


class TestSafetensorsIo:
    def test_shard_tensor_names_reads_header_only(self, tmp_path, write_safetensors):
        path = tmp_path / "model.safetensors"
        write_safetensors(path, {"a.weight": (2, 3), "b.weight": (4, 5)})
        assert shard_tensor_names(path) == {"a.weight", "b.weight"}

    def test_declared_file_size_matches_the_written_file(self, tmp_path, write_safetensors):
        path = tmp_path / "model.safetensors"
        write_safetensors(path, {"a.weight": (2, 3), "b.weight": (4, 5)})
        assert declared_file_size(path) == path.stat().st_size

    def test_declared_file_size_exceeds_a_truncated_file(self, tmp_path, write_safetensors):
        # The truncation guard's whole purpose: the header still advertises the
        # full payload after a partial write.
        path = tmp_path / "model.safetensors"
        write_safetensors(path, {"a.weight": (8, 8)})
        full = path.stat().st_size
        with open(path, "r+b") as f:
            f.truncate(full - 16)
        assert declared_file_size(path) == full
        assert path.stat().st_size < full

    def test_read_weight_map_returns_none_without_an_index(self, tmp_path):
        assert read_weight_map(tmp_path) is None

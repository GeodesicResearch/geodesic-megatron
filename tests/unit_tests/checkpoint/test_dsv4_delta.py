# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Tests of the DeepSeek-V4 delta writer on tiny real safetensors checkpoints."""

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "checkpoint"))

import dsv4_delta  # noqa: E402


def _write_checkpoint(directory: Path, tensors: dict[str, torch.Tensor]) -> None:
    directory.mkdir()
    save_file(tensors, str(directory / "model-00001-of-00001.safetensors"))
    index = {"metadata": {}, "weight_map": {name: "model-00001-of-00001.safetensors" for name in tensors}}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index))


@pytest.fixture
def checkpoints(tmp_path):
    """A Base with one FP8 weight (2 x 2 blocks), one BF16 norm and one FP8 weight the arm leaves unchanged."""
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(256, 256, generator=generator)
    quantised, scale = dsv4_delta.quantise_blocks(weight)
    kept, kept_scale = dsv4_delta.quantise_blocks(torch.randn(128, 128, generator=generator))
    norm = torch.randn(64, generator=generator).bfloat16()
    base = {
        "layers.0.attn.wq.weight": quantised,
        "layers.0.attn.wq.scale": scale,
        "layers.0.ffn.w1.weight": kept,
        "layers.0.ffn.w1.scale": kept_scale,
        "layers.0.attn_norm.weight": norm,
    }
    base_weight = dsv4_delta.dequantise_delta(quantised, scale).bfloat16().float()
    change = torch.randn(256, 256, generator=generator) * 0.01
    change[:128, :128] = 0  # one all-zero block
    arm = {
        "layers.0.attn.wq.weight": (base_weight + change).bfloat16(),
        "layers.0.ffn.w1.weight": dsv4_delta.dequantise_delta(kept, kept_scale).bfloat16(),  # the Base imported as BF16
        "layers.0.attn_norm.weight": (norm.float() + 0.5).bfloat16(),
    }
    _write_checkpoint(tmp_path / "base", base)
    _write_checkpoint(tmp_path / "arm", arm)
    return tmp_path, base_weight, arm


def test_quantise_blocks_round_trips_within_fp8_precision_and_keeps_zero_blocks_zero():
    generator = torch.Generator().manual_seed(1)
    delta = torch.randn(200, 300, generator=generator)  # not a multiple of the block size
    delta[:128, :128] = 0
    quantised, scale = dsv4_delta.quantise_blocks(delta)
    assert quantised.dtype == torch.float8_e4m3fn and scale.shape == (2, 3)
    assert scale[0, 0] == 0
    restored = dsv4_delta.dequantise_delta(quantised, scale)
    assert restored.shape == delta.shape
    assert torch.equal(restored[:128, :128], torch.zeros(128, 128))
    assert (restored - delta).abs().max() <= 0.07 * delta.abs().max()


def _write(root, arm_tensors, output="out"):
    return dsv4_delta.write_delta(str(root / "base"), arm_tensors, str(root / output), "treatment", 3, "abc")


def test_write_delta_stores_fp8_with_scale_fp32_for_the_rest_and_omits_unchanged_tensors(checkpoints):
    root, base_weight, arm = checkpoints
    manifest = _write(root, dsv4_delta.iter_hf_directory(str(root / "arm")))

    assert manifest["omitted_tensors"] == ["layers.0.ffn.w1.weight"]
    assert manifest["zero_scale_block_count"] == 1
    assert manifest["failures"] == [] and manifest["tensors_checked"] == 2
    assert (manifest["arm"], manifest["epoch"], manifest["base_revision"]) == ("treatment", 3, "abc")
    assert (root / "out" / ".complete").exists()
    index = json.loads((root / "out" / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(index) == {"layers.0.attn.wq.weight", "layers.0.attn.wq.scale", "layers.0.attn_norm.weight"}
    with safe_open(str(root / "out" / "delta-00001.safetensors"), framework="pt") as handle:
        quantised, scale = handle.get_tensor("layers.0.attn.wq.weight"), handle.get_tensor("layers.0.attn.wq.scale")
        norm_delta = handle.get_tensor("layers.0.attn_norm.weight")
    assert quantised.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
    expected = arm["layers.0.attn.wq.weight"].float() - base_weight
    restored = dsv4_delta.dequantise_delta(quantised, scale)
    assert (restored - expected).abs().max() <= 0.07 * expected.abs().max()
    assert norm_delta.dtype == torch.float32 and torch.allclose(norm_delta, torch.full((64,), 0.5), atol=0.02)


def test_write_delta_accepts_a_stream_of_tensors_without_an_hf_directory(checkpoints):
    root, _, arm = checkpoints
    manifest = _write(root, iter(arm.items()), output="streamed")
    assert manifest["failures"] == [] and (root / "streamed" / ".complete").exists()


def test_write_delta_without_the_complete_marker_when_a_base_tensor_is_missing(checkpoints):
    root, _, arm = checkpoints
    short = [(name, tensor) for name, tensor in arm.items() if name != "layers.0.attn_norm.weight"]
    manifest = _write(root, iter(short), output="short")
    assert any("not in the arm" in failure for failure in manifest["failures"])
    assert not (root / "short" / ".complete").exists()


def test_write_delta_refuses_an_existing_output_a_foreign_tensor_and_a_shape_change(checkpoints):
    root, _, arm = checkpoints
    with pytest.raises(FileExistsError):
        dsv4_delta.write_delta(str(root / "base"), iter(arm.items()), str(root), "a", 1, "abc")
    with pytest.raises(KeyError, match="not a Base tensor"):
        _write(root, iter([("layers.9.extra.weight", torch.zeros(4).bfloat16())]), output="o1")
    with pytest.raises(ValueError, match="shape"):
        _write(root, iter([("layers.0.attn_norm.weight", torch.zeros(65).bfloat16())]), output="o2")


def test_write_delta_from_export_writes_on_rank_zero_and_drains_the_stream_elsewhere(checkpoints):
    root, _, arm = checkpoints
    consumed = []

    def stream():
        for name, tensor in arm.items():
            consumed.append(name)
            yield name, tensor

    args = {"base_dir": str(root / "base"), "output_dir": str(root / "rank0"), "arm": "treatment", "epoch": 1, "base_revision": "abc"}
    manifest = dsv4_delta.write_delta_from_export(stream(), 0, **args)
    assert manifest["failures"] == [] and (root / "rank0" / ".complete").exists()
    assert len(consumed) == len(arm)

    consumed.clear()
    assert dsv4_delta.write_delta_from_export(stream(), 1, **dict(args, output_dir=str(root / "rank1"))) is None
    assert len(consumed) == len(arm) and not (root / "rank1").exists()

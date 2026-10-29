"""SafeTensorsStateSource.save_generator with strict=False: the index lists what was written, nothing else.

The toy release has an index over three shards. The model being saved lacks one module (its ``mtp.*`` tensors,
as an NTP model of a release with MTP has none): one shard holds only that module, one holds it and a weight the
model has, and one holds weights the model has.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource


SHARDS = {
    "model-00001-of-00003.safetensors": ["embed.weight", "layers.0.weight"],
    "model-00002-of-00003.safetensors": ["layers.1.weight", "mtp.layers.0.weight"],
    "model-00003-of-00003.safetensors": ["mtp.layers.1.weight"],
}
PRESENT = ["embed.weight", "layers.0.weight", "layers.1.weight"]


def _release(path):
    tensors = {}
    for filename, keys in SHARDS.items():
        shard = {key: torch.full((2, 3), float(i)) for i, key in enumerate(keys)}
        save_file(shard, str(path / filename))
        tensors.update(shard)
    weight_map = {key: filename for filename, keys in SHARDS.items() for key in keys}
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 4 * 6 * len(weight_map)}, "weight_map": weight_map})
    )
    return tensors


def _save_without_the_missing_module(tmp_path, distributed_save):
    release = tmp_path / "release"
    release.mkdir()
    tensors = _release(release)
    out = tmp_path / "out"
    SafeTensorsStateSource(release).save_generator(
        ((k, tensors[k]) for k in PRESENT), out, strict=False, distributed_save=distributed_save
    )
    return out


@pytest.mark.parametrize("distributed_save", [False, True])
def test_index_lists_only_written_tensors(tmp_path, distributed_save):
    out = _save_without_the_missing_module(tmp_path, distributed_save)
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert sorted(index["weight_map"]) == sorted(PRESENT)
    for key, filename in index["weight_map"].items():
        assert key in load_file(str(out / filename)), key
    assert index["metadata"]["total_size"] == 4 * 6 * len(PRESENT)


@pytest.mark.parametrize("distributed_save", [False, True])
def test_a_shard_with_no_written_tensor_is_not_created(tmp_path, distributed_save):
    out = _save_without_the_missing_module(tmp_path, distributed_save)
    assert not (out / "model-00003-of-00003.safetensors").exists()
    assert sorted(load_file(str(out / "model-00002-of-00003.safetensors"))) == ["layers.1.weight"]

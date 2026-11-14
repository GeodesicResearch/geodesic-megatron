"""Write the weight change of a DeepSeek-V4 model against its Base as an FP8 block-scaled delta.

delta = arm - the Base dequantised to BF16 (the weights the arm started from), in fp32, per tensor. A tensor that the Base stores as FP8 E4M3 with a block-scale
companion (``NAME.weight`` and ``NAME.scale``) is stored as FP8 E4M3 with 128 x 128 block scales (amax / 448, 0 for an
all-zero block). Every other tensor is stored as fp32 with no scale. A tensor whose delta is exactly zero is omitted, and
a tensor absent from the delta means no change. Names are the HF names of the release.

``python scripts/checkpoint/dsv4_delta.py --config <yaml>`` takes the keys base_dir, arm_dir, output_dir, arm, epoch and
base_revision. ``write_delta`` and ``dequantise_delta`` are the library functions: the second is the loader's reference.
"""

import argparse
import json
import os

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

from megatron.bridge.models.conversion.quantization_utils import maybe_dequantize_hf_quantized_weight


BLOCK = 128
FP8_MAX = 448.0
CONFIG_KEYS = ("base_dir", "arm_dir", "output_dir", "arm", "epoch", "base_revision")


def quantise_blocks(delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise a 2D fp32 tensor to FP8 E4M3 with one fp32 scale per 128 x 128 block (amax / 448, 0 for a zero block)."""
    rows, cols = delta.shape
    block_rows, block_cols = -(-rows // BLOCK), -(-cols // BLOCK)
    padded = torch.zeros(block_rows * BLOCK, block_cols * BLOCK)
    padded[:rows, :cols] = delta
    blocks = padded.view(block_rows, BLOCK, block_cols, BLOCK)
    scale = blocks.abs().amax(dim=(1, 3)) / FP8_MAX
    divisor = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantised = (blocks / divisor[:, None, :, None]).to(torch.float8_e4m3fn)
    return quantised.view(block_rows * BLOCK, block_cols * BLOCK)[:rows, :cols].contiguous(), scale


def dequantise_delta(quantised: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Return the fp32 delta of a block-scaled FP8 tensor; a block with scale 0 is all zero."""
    rows, cols = quantised.shape
    expanded = scale.repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)[:rows, :cols]
    return quantised.float() * expanded


def _index(directory: str) -> dict[str, str]:
    """Return the tensor name to shard file map of an HF checkpoint directory."""
    with open(f"{directory}/model.safetensors.index.json") as handle:
        return json.load(handle)["weight_map"]


def _read(directory: str, index: dict[str, str], names: list[str]) -> dict[str, torch.Tensor]:
    """Read the named tensors, opening each shard once."""
    by_shard: dict[str, list[str]] = {}
    for name in names:
        by_shard.setdefault(index[name], []).append(name)
    tensors = {}
    for shard, shard_names in by_shard.items():
        with safe_open(f"{directory}/{shard}", framework="pt") as handle:
            for name in shard_names:
                tensors[name] = handle.get_tensor(name)
    return tensors


def write_delta(base_dir: str, arm_dir: str, output_dir: str, arm: str, epoch: int, base_revision: str) -> dict:
    """Write the delta of the arm's BF16 HF export against the Base, one output shard per Base shard.

    Reads both directories, writes shards, model.safetensors.index.json and delta_manifest.json to output_dir, which must
    not exist. Raises if the arm lacks a Base tensor or a shape differs. Returns the manifest.
    """
    if os.path.exists(output_dir):
        raise FileExistsError(f"{output_dir} exists")
    os.makedirs(output_dir)
    base_index, arm_index = _index(base_dir), _index(arm_dir)
    scale_names = {name for name in base_index if name.endswith(".scale")}
    weight_names = sorted(name for name in base_index if name not in scale_names)
    missing = [name for name in weight_names if name not in arm_index]
    if missing:
        raise KeyError(f"{len(missing)} Base tensors are missing from the arm, first: {missing[:3]}")
    weight_map: dict[str, str] = {}
    omitted, zero_scale_blocks, max_relative = [], 0, 0.0
    for shard in sorted(set(base_index[name] for name in weight_names)):
        names = [name for name in weight_names if base_index[name] == shard]
        shard_tensors: dict[str, torch.Tensor] = {}
        for name in names:
            companion = name[: -len(".weight")] + ".scale" if name.endswith(".weight") else None
            keys = [name] + ([companion] if companion in scale_names else [])
            base = _read(base_dir, base_index, keys)
            arm_weight = _read(arm_dir, arm_index, [name])[name]
            # The arm started from the Base imported as BF16, so the delta is taken against that rounding of the dequantised Base.
            base_fp32 = maybe_dequantize_hf_quantized_weight(name, base, dtype=torch.bfloat16).float()
            if base_fp32.shape != arm_weight.shape:
                raise ValueError(f"{name}: Base shape {tuple(base_fp32.shape)} differs from arm shape {tuple(arm_weight.shape)}")
            delta = arm_weight.float() - base_fp32
            if not torch.any(delta):
                omitted.append(name)
                continue
            max_relative = max(max_relative, float(delta.norm() / base_fp32.norm().clamp(min=1e-30)))
            out = shard_tensors
            if companion in scale_names and base[name].dtype == torch.float8_e4m3fn and delta.ndim == 2:
                quantised, scale = quantise_blocks(delta)
                out[name], out[companion] = quantised, scale
                zero_scale_blocks += int((scale == 0).sum())
                weight_map[name] = weight_map[companion] = shard
            else:
                out[name] = delta
                weight_map[name] = shard
        if shard_tensors:
            save_file(shard_tensors, f"{output_dir}/{shard}", metadata={"format": "pt"})
    with open(f"{output_dir}/model.safetensors.index.json", "w") as handle:
        json.dump({"metadata": {}, "weight_map": weight_map}, handle, indent=1)
    manifest = {
        "kind": "delta",
        "base_repo": "deepseek-ai/DeepSeek-V4-Pro-Base",
        "base_revision": base_revision,
        "arm": arm,
        "epoch": epoch,
        "omitted_tensors": omitted,
        "zero_scale_block_count": zero_scale_blocks,
        "largest_per_tensor_relative_norm": max_relative,
    }
    with open(f"{output_dir}/delta_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=1)
    return manifest


def main() -> None:
    """Read the YAML config named on the command line and write the delta."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML file with the keys " + ", ".join(CONFIG_KEYS))
    with open(parser.parse_args().config) as handle:
        config = yaml.safe_load(handle)
    unknown = set(config) ^ set(CONFIG_KEYS)
    if unknown:
        raise ValueError(f"config keys must be exactly {CONFIG_KEYS}, differing in {sorted(unknown)}")
    manifest = write_delta(**config)
    with open(f"{config['output_dir']}/delta_config.yaml", "w") as handle:
        yaml.safe_dump(config, handle)
    print(json.dumps({key: manifest[key] for key in ("arm", "epoch", "zero_scale_block_count")}))


if __name__ == "__main__":
    main()

"""Write the weight change of a DeepSeek-V4 model against its Base as an FP8 block-scaled delta, and check it.

delta = arm - the Base dequantised to BF16 (the weights the arm started from), in fp32, per tensor. A tensor that the Base
stores as FP8 E4M3 with a block-scale companion (``NAME.weight`` and ``NAME.scale``) is stored as FP8 E4M3 with 128 x 128
block scales (amax / 448, 0 for an all-zero block). Every other tensor is stored as fp32 with no scale. A tensor whose delta
is exactly zero is omitted, and a tensor absent from the delta means no change. Names are the HF names of the release.

The arm's tensors arrive as a stream of (HF name, BF16 tensor), from an HF directory (``iter_hf_directory``) or from the
Bridge's export of a Megatron save, so no HF copy of the arm needs to exist on disk. Every tensor is checked as it is
written: its reconstruction from the stored delta must agree with arm minus Base, and at the end every Base tensor must be
written or omitted, with matching shapes. ``.complete`` is written only when all checks pass, and a save may be deleted only
after that.

``python scripts/checkpoint/dsv4_delta.py --config <yaml>`` takes the keys base_dir, arm_dir, output_dir, arm, epoch and
base_revision (the arm as an HF directory). ``write_delta`` and ``dequantise_delta`` are the library functions: the
second is the loader's reference.
"""

import argparse
import json
import os
from collections.abc import Iterable, Iterator

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

from megatron.bridge.models.conversion.quantization_utils import maybe_dequantize_hf_quantized_weight


BLOCK = 128
FP8_MAX = 448.0
CONFIG_KEYS = ("base_dir", "arm_dir", "output_dir", "arm", "epoch", "base_revision")
SHARD_BYTES = 4 * 2**30
# FP8 E4M3 keeps 3 mantissa bits, so a value is off by up to 6.25 per cent of itself; the check allows the largest delta 7 per cent.
MAX_RELATIVE_ERROR = 0.07


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


def iter_hf_directory(directory: str) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (name, tensor) for every tensor of an HF checkpoint directory, shard by shard."""
    for shard in sorted(set(_index(directory).values())):
        with safe_open(f"{directory}/{shard}", framework="pt") as handle:
            for name in handle.keys():
                yield name, handle.get_tensor(name)


class _ShardWriter:
    """Collect tensors and write them as numbered safetensors shards of about SHARD_BYTES, keeping the weight map."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.weight_map: dict[str, str] = {}
        self._pending: dict[str, torch.Tensor] = {}
        self._pending_bytes = 0
        self._count = 0

    def add(self, tensors: dict[str, torch.Tensor]) -> None:
        """Add tensors that belong together and flush when the shard is large enough."""
        self._pending.update(tensors)
        self._pending_bytes += sum(t.numel() * t.element_size() for t in tensors.values())
        if self._pending_bytes >= SHARD_BYTES:
            self.flush()

    def flush(self) -> None:
        """Write the pending tensors as the next shard."""
        if not self._pending:
            return
        self._count += 1
        shard = f"delta-{self._count:05d}.safetensors"
        save_file(self._pending, f"{self.output_dir}/{shard}", metadata={"format": "pt"})
        self.weight_map.update({name: shard for name in self._pending})
        self._pending, self._pending_bytes = {}, 0


def write_delta(
    base_dir: str,
    arm_tensors: Iterable[tuple[str, torch.Tensor]],
    output_dir: str,
    arm: str,
    epoch: int,
    base_revision: str,
) -> dict:
    """Write and check the delta of the arm's tensors against the Base.

    arm_tensors yields (HF name, BF16 tensor). The output directory must not exist. It receives delta-NNNNN.safetensors
    shards, model.safetensors.index.json, delta_manifest.json with the check results, and ``.complete`` only when every
    check passed. Raises if an arm tensor is not in the Base or has another shape. Returns the manifest.
    """
    if os.path.exists(output_dir):
        raise FileExistsError(f"{output_dir} exists")
    os.makedirs(output_dir)
    base_index = _index(base_dir)
    scale_names = {name for name in base_index if name.endswith(".scale")}
    base_weights = {name for name in base_index if name not in scale_names}
    writer = _ShardWriter(output_dir)
    seen, omitted, failures = set(), [], []
    zero_scale_blocks, max_relative, checked = 0, 0.0, 0
    for name, arm_weight in arm_tensors:
        if name not in base_weights:
            raise KeyError(f"{name} is not a Base tensor")
        seen.add(name)
        companion = name[: -len(".weight")] + ".scale" if name.endswith(".weight") else None
        keys = [name] + ([companion] if companion in scale_names else [])
        base = _read(base_dir, base_index, keys)
        # The arm started from the Base imported as BF16, so the delta is taken against that rounding of the dequantised Base.
        base_value = maybe_dequantize_hf_quantized_weight(name, base, dtype=torch.bfloat16).float()
        if base_value.shape != arm_weight.shape:
            raise ValueError(f"{name}: Base shape {tuple(base_value.shape)} differs from arm shape {tuple(arm_weight.shape)}")
        delta = arm_weight.float() - base_value
        if not torch.any(delta):
            omitted.append(name)
            continue
        max_relative = max(max_relative, float(delta.norm() / base_value.norm().clamp(min=1e-30)))
        if companion in scale_names and base[name].dtype == torch.float8_e4m3fn and delta.ndim == 2:
            quantised, scale = quantise_blocks(delta)
            writer.add({name: quantised, companion: scale})
            zero_scale_blocks += int((scale == 0).sum())
            restored = dequantise_delta(quantised, scale)
        else:
            writer.add({name: delta})
            restored = delta
        error = float((restored - delta).abs().max() / delta.abs().max())
        checked += 1
        if error > MAX_RELATIVE_ERROR:
            failures.append(f"{name}: reconstruction error {error:.4f} of the largest delta exceeds {MAX_RELATIVE_ERROR}")
    writer.flush()
    missing = sorted(base_weights - seen)
    if missing:
        failures.append(f"{len(missing)} Base tensors were not in the arm, first: {missing[:3]}")
    with open(f"{output_dir}/model.safetensors.index.json", "w") as handle:
        json.dump({"metadata": {}, "weight_map": writer.weight_map}, handle, indent=1)
    manifest = {
        "kind": "delta",
        "base_repo": "deepseek-ai/DeepSeek-V4-Pro-Base",
        "base_revision": base_revision,
        "arm": arm,
        "epoch": epoch,
        "omitted_tensors": omitted,
        "zero_scale_block_count": zero_scale_blocks,
        "largest_per_tensor_relative_norm": max_relative,
        "tensors_checked": checked,
        "failures": failures,
    }
    with open(f"{output_dir}/delta_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=1)
    if not failures:
        open(f"{output_dir}/.complete", "w").close()
    return manifest


def write_delta_from_export(export_stream: Iterable[tuple[str, torch.Tensor]], rank: int, **write_args) -> dict | None:
    """Write the delta on rank 0 from a collective export stream that every rank must consume.

    The Bridge's export gathers each tensor across the ranks, so all ranks iterate the stream. Rank 0 passes it to
    write_delta with write_args (base_dir, output_dir, arm, epoch, base_revision) and returns the manifest. The other ranks
    drop the tensors and return None.
    """
    if rank == 0:
        return write_delta(arm_tensors=export_stream, **write_args)
    for _ in export_stream:
        pass
    return None


def main() -> None:
    """Read the YAML config named on the command line, write the delta of the arm's HF directory, exit 1 on a failed check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML file with the keys " + ", ".join(CONFIG_KEYS))
    with open(parser.parse_args().config) as handle:
        config = yaml.safe_load(handle)
    unknown = set(config) ^ set(CONFIG_KEYS)
    if unknown:
        raise ValueError(f"config keys must be exactly {CONFIG_KEYS}, differing in {sorted(unknown)}")
    arm_dir = config.pop("arm_dir")
    manifest = write_delta(arm_tensors=iter_hf_directory(arm_dir), **config)
    with open(f"{config['output_dir']}/delta_config.yaml", "w") as handle:
        yaml.safe_dump({**config, "arm_dir": arm_dir}, handle)
    print(json.dumps({key: manifest[key] for key in ("arm", "epoch", "zero_scale_block_count", "tensors_checked", "failures")}))
    raise SystemExit(1 if manifest["failures"] else 0)


if __name__ == "__main__":
    main()

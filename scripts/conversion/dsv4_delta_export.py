"""Write the DeepSeek-V4 delta of a Megatron save straight from the Bridge's export, with no HF copy of the save on disk.

Run under torchrun on the GPU layout that fits the save for export (for example PP16 x EP4):

    torchrun ... scripts/conversion/dsv4_delta_export.py --config <yaml>

The YAML has exactly the keys hf_model (the Base snapshot directory), megatron_path (the save's iter directory),
output_dir, arm, epoch, base_revision, tp, pp, ep, etp, distributed_timeout_minutes. Every rank takes part in the
collective export, and rank 0 writes the delta with the checks of scripts/checkpoint/dsv4_delta.py. The job exits 1 when a
check fails, and the save may be deleted only when output_dir holds ``.complete``.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "checkpoint"))

import dsv4_delta  # noqa: E402
import gpu_backend  # noqa: E402


CONFIG_KEYS = (
    "hf_model",
    "megatron_path",
    "output_dir",
    "arm",
    "epoch",
    "base_revision",
    "tp",
    "pp",
    "ep",
    "etp",
    "distributed_timeout_minutes",
    "unchanged_prefixes",
)


def main() -> None:
    """Load the Megatron save in the configured layout, stream its export into the delta writer and exit 1 on a failed check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML file with the keys " + ", ".join(CONFIG_KEYS))
    with open(parser.parse_args().config) as handle:
        config = yaml.safe_load(handle)
    unknown = set(config) ^ set(CONFIG_KEYS)
    if unknown:
        raise ValueError(f"config keys must be exactly {CONFIG_KEYS}, differing in {sorted(unknown)}")
    bridge, model = gpu_backend.load_model_for_export(
        hf_model=config["hf_model"],
        hf_revision=None,
        megatron_path=config["megatron_path"],
        hf_path=config["output_dir"],
        tp=config["tp"],
        pp=config["pp"],
        ep=config["ep"],
        etp=config["etp"],
        torch_dtype="bfloat16",
        trust_remote_code=True,
        strict=False,
        show_progress=False,
        distributed_save=False,
        save_every_n_ranks=1,
        distributed_timeout_minutes=config["distributed_timeout_minutes"],
        export_weight_dtype=None,
        overwrite=False,
    )
    stream = bridge.export_hf_weights(model, show_progress=False)
    manifest = dsv4_delta.write_delta_from_export(
        stream,
        torch.distributed.get_rank(),
        base_dir=config["hf_model"],
        output_dir=config["output_dir"],
        arm=config["arm"],
        epoch=config["epoch"],
        base_revision=config["base_revision"],
        unchanged_prefixes=config["unchanged_prefixes"],
    )
    failed = torch.tensor(int(bool(manifest and manifest["failures"])), device=torch.cuda.current_device())
    torch.distributed.broadcast(failed, src=0)
    if torch.distributed.get_rank() == 0:
        with open(f"{config['output_dir']}/delta_config.yaml", "w") as handle:
            yaml.safe_dump(config, handle)
        print(json.dumps({key: manifest[key] for key in ("arm", "epoch", "zero_scale_block_count", "tensors_checked", "failures")}))
    raise SystemExit(int(failed.item()))


if __name__ == "__main__":
    main()

"""Turn a DeepSeek-V4 continued-pretraining YAML into the overrides of run_recipe.py and write the config beside the run.

``python scripts/training/dsv4_cpt.py --config <yaml> --print`` prints the recipe name and one KEY=VALUE override per line,
and ``--write`` also copies the config into the run's save directory. The YAML has exactly the sections of ``SECTIONS``.
The launcher (an sbatch script) exports DEEPSEEK_V4_PRO_HF_PATH from ``hf_model`` and starts run_recipe.py with the overrides.

Checked before any job starts: the pipeline layout (stages and virtual stages against the node count), the global batch size
against the micro-batch size and the data-parallel size (it must give a whole number of micro-batches), the number of updates
against the saves, and that the data blend weights are positive.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import yaml


SECTIONS = {
    "arm": str,
    "hf_model": str,
    "base_revision": str,
    "nodes": int,
    "topology": dict,
    "batch": dict,
    "schedule": dict,
    "optimizer": dict,
    "data": dict,
    "run": dict,
    "provenance": dict,
}
KEYS = {
    "topology": {"pp", "vpp", "ep", "etp", "microbatch_group", "blocks"},
    "batch": {"seq_length", "global_batch_size", "micro_batch_size"},
    "schedule": {"train_iters", "save_interval", "lr", "warmup_iters", "eval_interval", "eval_iters"},
    "optimizer": {"offload_fraction", "moment_dtype"},
    "data": {"blend", "valid_blend", "split"},
    "run": {"save_dir", "wandb_project", "wandb_exp_name", "save", "eval_only"},
    "provenance": None,
}
GPUS_PER_NODE = 4
RECIPE = "deepseek_v4_pro_pretrain_32gpu_gb300_bf16_config"


def load_config(path: str) -> dict:
    """Read the YAML and refuse unknown or missing sections and keys."""
    with open(path) as handle:
        config = yaml.safe_load(handle)
    if set(config) != set(SECTIONS):
        raise ValueError(f"sections must be exactly {sorted(SECTIONS)}, differing in {sorted(set(config) ^ set(SECTIONS))}")
    for name, kind in SECTIONS.items():
        if not isinstance(config[name], kind):
            raise ValueError(f"{name} must be a {kind.__name__}")
    for name, keys in KEYS.items():
        if keys is not None and set(config[name]) != keys:
            raise ValueError(f"{name} keys must be exactly {sorted(keys)}, differing in {sorted(set(config[name]) ^ keys)}")
    return config


def validate(config: dict) -> dict:
    """Check the layout and batch arithmetic and return the derived numbers."""
    topology, batch, schedule = config["topology"], config["batch"], config["schedule"]
    gpus = config["nodes"] * GPUS_PER_NODE
    per_stage = topology["ep"] * topology["etp"]
    if gpus % topology["pp"] != 0:
        raise ValueError(f"{gpus} GPUs do not divide into {topology['pp']} pipeline stages")
    stage_gpus = gpus // topology["pp"]
    if stage_gpus % per_stage != 0:
        raise ValueError(f"{stage_gpus} GPUs per stage do not divide by EP x ETP = {per_stage}")
    micro_batches, remainder = divmod(batch["global_batch_size"], batch["micro_batch_size"] * stage_gpus)
    if remainder or micro_batches < 1:
        raise ValueError(
            f"global batch {batch['global_batch_size']} is not a whole number of micro-batches of {batch['micro_batch_size']} "
            f"over {stage_gpus} data-parallel ranks"
        )
    group = topology["microbatch_group"]
    if topology["vpp"] > 1:
        # Megatron's interleaved schedule: the group size lies in [PP, micro-batches] and leaves a remainder of 0 or at least PP.
        if not topology["pp"] <= group <= micro_batches:
            raise ValueError(f"microbatch_group {group} must lie in [PP={topology['pp']}, {micro_batches} micro-batches]")
        if 0 < micro_batches % group < topology["pp"]:
            raise ValueError(f"{micro_batches} micro-batches in groups of {group} leave a remainder under PP={topology['pp']}")
    if schedule["save_interval"] > schedule["train_iters"]:
        raise ValueError("save_interval is larger than train_iters")
    data = config["data"]
    if (data["valid_blend"] is None) == (data["split"] is None):
        raise ValueError("give exactly one of valid_blend (a separate validation blend) and split (held-out parts of the blend)")
    for blend in (data["blend"], data["valid_blend"]):
        if blend is not None and any(weight <= 0 for _, weight in blend):
            raise ValueError("every blend weight must be positive")
    bubble = (topology["pp"] - 1) / (topology["vpp"] * micro_batches + topology["pp"] - 1)
    return {"micro_batches_per_rank": micro_batches, "bubble": bubble, "tokens_per_update": batch["global_batch_size"] * batch["seq_length"]}


def block_layout(topology: dict) -> tuple[str, str]:
    """Return the hybrid layer pattern and the pipeline layout that give each virtual stage one block.

    A block is an attention layer and an MoE layer. Blocks 1 and 2 use the hybrid-compressed attention H, then C and H alternate
    from block 3 (the pattern of DeepSeek-V4). The embedding joins the first virtual stage and the loss head the last, and
    virtual stages left over when there are fewer blocks than stages come last and hold no block. Hash layers sit in the
    first three blocks, which fall on the first three virtual stages.
    """
    stages = topology["pp"] * topology["vpp"]
    blocks = topology["blocks"]
    if blocks > stages or stages - blocks > topology["pp"]:
        raise ValueError(f"{blocks} blocks do not fit {stages} virtual stages with at most one stage of spare per pipeline rank")
    letters = ["H", "H"] + ["C" if k % 2 == 0 else "H" for k in range(blocks - 2)]
    pattern = "|".join(letters[k] + "E" if k < blocks else "" for k in range(stages))
    layout = []
    for k in range(stages):
        entry = "tt" if k < blocks else ""
        if k == 0:
            entry = "E" + entry
        if k == stages - 1:
            entry += "L"
        layout.append(entry)
    return pattern, "|".join(layout)


def _blend(blend: list) -> str:
    """Format [[path, weight], ...] as the dataset.blend override [[paths], [weights]]."""
    return json.dumps([[path for path, _ in blend], [weight for _, weight in blend]], separators=(",", ":"))


def overrides(config: dict) -> list[str]:
    """Return the run_recipe KEY=VALUE overrides of the config."""
    topology, batch, schedule, optimizer, data, run = (config[k] for k in ("topology", "batch", "schedule", "optimizer", "data", "run"))
    items = [
        "model.tensor_model_parallel_size=1",
        f"model.pipeline_model_parallel_size={topology['pp']}",
        f"model.virtual_pipeline_model_parallel_size={topology['vpp']}",
        f"model.expert_model_parallel_size={topology['ep']}",
        f"model.expert_tensor_parallel_size={topology['etp']}",
        f"model.microbatch_group_size_per_vp_stage={topology['microbatch_group']}",
        f"dataset.seq_length={batch['seq_length']}",
        f"train.global_batch_size={batch['global_batch_size']}",
        f"train.micro_batch_size={batch['micro_batch_size']}",
        f"train.train_iters={schedule['train_iters']}",
        f"optimizer.lr={schedule['lr']}",
        f"optimizer.min_lr={schedule['lr']}",
        f"scheduler.lr_warmup_iters={schedule['warmup_iters']}",
        "scheduler.lr_decay_style=constant",
        f"validation.eval_interval={schedule['eval_interval']}",
        f"validation.eval_iters={schedule['eval_iters']}",
        "model.recompute_granularity=full",
        "model.recompute_method=uniform",
        "model.recompute_num_layers=1",
        "model.recompute_modules=null",
        "ddp.use_distributed_optimizer=true",
        "optimizer.use_precision_aware_optimizer=true",
        f"optimizer.exp_avg_dtype={optimizer['moment_dtype']}",
        f"optimizer.exp_avg_sq_dtype={optimizer['moment_dtype']}",
        "optimizer.optimizer_cpu_offload=true",
        f"optimizer.optimizer_offload_fraction={optimizer['offload_fraction']}",
        f"checkpoint.pretrained_checkpoint={config['hf_model']}",
        f"checkpoint.save={run['save_dir'] if run['save'] else 'null'}",
        f"checkpoint.load={run['save_dir'] if run['save'] else 'null'}",
        f"checkpoint.save_interval={schedule['save_interval']}",
        "checkpoint.save_optim=false",
        "checkpoint.save_rng=false",
        "logger.tensorboard_dir=null",
        f"logger.wandb_project={run['wandb_project']}",
        f"logger.wandb_exp_name={run['wandb_exp_name']}",
        f"validation.skip_train={'true' if run['eval_only'] else 'false'}",
    ]
    pattern, layout = block_layout(topology)
    items += [f"model.hybrid_layer_pattern={pattern}", f"model.pipeline_model_parallel_layout={layout}", "model.mtp_num_layers=null"]
    if data["valid_blend"] is None:
        items += [f"dataset.blend={_blend(data['blend'])}", f'dataset.split="{data["split"]}"']
    else:
        # The Bridge's loop evaluates one validation set; Megatron-Core refuses a split together with a blend per split.
        items.append(f"dataset.blend_per_split=[{_blend(data['blend'])},{_blend(data['valid_blend'])},null]")
        items.append("dataset.split=null")
    return items


def main() -> None:
    """Validate the config, print the derived numbers and the overrides, and with --write copy the config into the run directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--overrides", action="store_true", help="print only the overrides, one per line")
    parser.add_argument("--write", action="store_true", help="copy the config to <save_dir>/dsv4_cpt_config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    derived = validate(config)
    if not args.overrides:
        print(f"recipe {RECIPE}")
        print(f"derived {json.dumps(derived)}")
    for item in overrides(config):
        print(item)
    if args.write:
        save_dir = Path(config["run"]["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.config, save_dir / "dsv4_cpt_config.yaml")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate the training configs of the 5B filter experiment from the 20B longmino CPT
config: control = 5B unfiltered prefix pool; filtered_k2 = term-screen survivors (K=2);
mixmatch = unfiltered text at the filtered arm's family proportions, which holds the mix
fixed so that a filtered-vs-unfiltered comparison is not also a comparison of two mixes;
`_s2` = a second replicate of the filtered arm and its mix-matched control on documents
disjoint from the first, with a different dataset seed.

Each arm is one job: train_iters 298, the midtrain hyperparameters verbatim,
lr_wsd_decay_iters 298 (the anneal spans the run), save_interval 30, so checkpoints land at
iterations 30 / 60 / ... / 270 and at 298 by the end-of-training save; the experiment keeps
30 / 90 / 298 (0.5 / 1.5 / 5.0B tokens, log-spaced) and deletes the rest. No segment ever
resumes from a checkpoint: chained segments with train.exit_interval stops (the first
design) failed at every resume on 64 nodes with CUDA out-of-memory at the first step
(NCCL alloc / Triton), while every fresh start succeeded — at 64 nodes the resume path's
per-rank optimizer shard is twice the midtrain's. Data: dataset.data_path is the
token-proportional blend over the pool's families (pools_5b.json), path_to_cache and
checkpoint dirs are the arm's own.

    python make_5b_configs.py [--pools /projects/.../longmino_cpt/pools_5b.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BASE = HERE / "nemotron_nano_30b_baseline_longmino_cpt.yaml"
BASE_SEED = 1234  # the 20B config's rng.seed, inherited by the first replicate
CKPT_ROOT = "/projects/a5k/public/data_cwtice.a5k/checkpoints/megatron/control_pretraining"
CACHE_ROOT = "/projects/a5k/public/data_cwtice.a5k/gpt_index_cache"
POOL_ROOT = "/projects/a5k/public/data_cwtice.a5k/data/longmino_cpt"
#: arm -> (pool directory, accounting file, key within it, dataset shuffle seed). The `_s2`
#: arms are the second replicate: document-disjoint pools built by build_5b_pools_seed2.py,
#: and a different dataset seed so the data order differs too.
ARMS = {
    "control": ("pool5b_control", "pools_5b.json", "control", 1234),
    "filtered_k2": ("pool5b_filtered_k2", "pools_5b.json", "filtered", 1234),
    "mixmatch": ("pool5b_mixmatch", "pools_5b.json", "mixmatch", 1234),
    "filtered_k2_s2": ("pool5b_filtered_k2_s2", "pools_5b_seed2.json", "filtered", 5678),
    "mixmatch_s2": ("pool5b_mixmatch_s2", "pools_5b_seed2.json", "mixmatch", 5678),
}
TRAIN_ITERS = 298
SAVE_INTERVAL = 30
KEEP_ITERATIONS = (30, 90, 298)


def blend(acct: dict) -> list:
    fams = {k: v for k, v in acct.items() if not k.startswith("_")}
    tot = sum(v["tokens"] for v in fams.values())
    out: list = []
    for fam, v in sorted(fams.items()):
        out += [round(v["tokens"] / tot, 6), None]  # prefix filled per arm below
        out[-1] = fam
    # fix rounding residue onto the largest family, as blend_weights.py does
    ws = [out[i] for i in range(0, len(out), 2)]
    residue = round(1.0 - sum(ws), 6)
    big = max(range(len(ws)), key=lambda i: ws[i])
    out[2 * big] = round(ws[big] + residue, 6)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=None, help="default: every arm whose pool exists")
    a = ap.parse_args()
    base_text = BASE.read_text()
    base = yaml.safe_load(base_text)
    written = []
    for arm, (pool, acct_file, key, seed) in ARMS.items():
        if a.arms and arm not in a.arms:
            continue
        acct_path = Path(POOL_ROOT) / acct_file
        if not acct_path.exists():
            continue
        acct = json.loads(acct_path.read_text())
        if key not in acct or not acct[key]:
            continue
        b = blend(acct[key])
        data_path = []
        for i in range(0, len(b), 2):
            data_path += [b[i], f"{POOL_ROOT}/{pool}/{b[i + 1]}/tokenized_base_input_document"]
        cfg = yaml.safe_load(base_text)
        cfg["dataset"]["data_path"] = data_path
        cfg["dataset"]["path_to_cache"] = f"{CACHE_ROOT}/control_pretraining_longmino_5b_{arm}"
        if seed != BASE_SEED:
            # `dataset.seed` is NOT the data-order seed: GPTDatasetConfig has no such field and
            # the loader skips it with a warning. The shuffle is driven by rng.seed, which the
            # framework carries into dataset.random_seed; set both so the replicate's data order
            # really differs.
            cfg["rng"] = {**cfg.get("rng", {}), "seed": seed}
            cfg["dataset"]["random_seed"] = seed
        cfg["train"]["train_iters"] = TRAIN_ITERS
        cfg["train"].pop("exit_interval", None)
        cfg["scheduler"]["lr_wsd_decay_iters"] = TRAIN_ITERS
        d = f"{CKPT_ROOT}/control_pretrain_30b_baseline_longmino_5b_{arm}"
        cfg["checkpoint"]["load"] = d
        cfg["checkpoint"]["save"] = d
        cfg["checkpoint"]["save_interval"] = SAVE_INTERVAL
        cfg["logger"]["wandb_exp_name"] = f"control_pretrain_30b_baseline_longmino_5b_{arm}"
        name = f"nemotron_nano_30b_baseline_longmino_5b_{arm}.yaml"
        hdr = (f"# GENERATED by make_5b_configs.py from {BASE.name} — do not edit by hand.\n"
               f"# Arm {arm} ({pool}): train_iters {TRAIN_ITERS}, save_interval {SAVE_INTERVAL} "
               f"(keep iterations {KEEP_ITERATIONS}); see the module docstring for the stated diff.\n")
        (HERE / name).write_text(hdr + yaml.safe_dump(cfg, sort_keys=False, width=120))
        written.append(name)
    print("\n".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

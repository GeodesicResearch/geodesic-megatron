# Norm V2: NVDA-Norm-Think

This directory holds what the Nemotron Gap analysis needs around one training run: NVDA-Norm-Think. NVDA-Norm-Think is
NVIDIA's Nemotron 3 Nano 30B-A3B Base after the xl-50b v4 reasoning SFT. The run is the v4 recipe with only its warm
start changed. Its configuration is `configs/control_pretraining/30b_baseline_ablations/nemotron_nano_30b_nvidia_base_sft_xl50b_gbs256_v4.yaml`.
That directory's README describes the run and its launch command, and the config's header describes its warm start.

| File | Purpose |
|------|---------|
| `watch_nvidia_base_sft.yaml` | The training gate. It is read by `scripts/telemetry/run_watch.py`. |
| `checks/c3_export_check_nvidia_base_think.yaml` | The export check (C3) of every revision. It is read by `scripts/checkpoint/export_check.py`. |
| `hub_models.yaml` | The Hub manifest: repository, collection, card text and release logs. It is read by `scripts/hub/publish_models.py`. |

## The training gate

Run the gate on the segment logs of the run, in order. The header of the watch file states the stop conditions, the
flag and the exit statuses.

```bash
python3 scripts/telemetry/run_watch.py --spec configs/norm_v2/watch_nvidia_base_sft.yaml --log <segment log> ...
```

## The export check (C3)

C3 runs as one job per exported revision (`scripts/checkpoint/export_check.sbatch`), and the publisher queues it. The
header of the specification states what it omits, its tolerance and the rule that calibrates it.

The C3 results are keyed by the sha256 of the specification and its `base_config:` chain, which also holds
`configs/metagaming_filtering/30b_clueless_norm/checks/c3_export_check_think.yaml` and
`configs/metagaming_filtering/30b_clueless_norm/checks/c3_export_check.yaml`. An edit to any of the three, a comment
included, changes that key: the publisher then finds no result for any revision and reports each check job as ended
without one. Leave the three files unchanged while the release logs are in use.

The negative control of the calibration is a job that you submit once the export of iteration 5976 exists. It scores
that export against the conversion's `iter_0000000`, so the calibration shows that C3 tells this SFT's export from its
warm start, not that it separates two checkpoints of the run:

```bash
isambard_sbatch --output=/projects/a5k/public/logs/norm_v2/c3/negative_control_5976_vs_nvda_base/export-check-%j.out \
  scripts/checkpoint/export_check.sbatch configs/norm_v2/checks/c3_export_check_nvidia_base_think.yaml \
  /projects/a5k/public/checkpoints/megatron_bridges/models/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/iter_0000000 \
  <export directory of revision sft_iter_5976> \
  /projects/a5k/public/logs/norm_v2/c3/negative_control_5976_vs_nvda_base
```

After both runs of the calibration exist, this command judges them. It exits 0 only when each run has the outcome that
the specification expects:

```bash
python scripts/checkpoint/export_check.py calibrate \
  --spec configs/norm_v2/checks/c3_export_check_nvidia_base_think.yaml --report-out <report.json>
```

## Publishing

Each save becomes a revision `sft_iter_<iteration>` of the repository that the manifest names, and the final save is
also `main`. Run the publisher on the host Python, from a frozen copy of the commit (`scripts/training/README.md`,
"Making a frozen copy"), and never as a SLURM job:

```bash
python3 scripts/hub/publish_models.py --manifest configs/norm_v2/hub_models.yaml --plan
python3 scripts/hub/publish_models.py --manifest configs/norm_v2/hub_models.yaml \
  --phase rolling --newest-first --poll-interval 600 --stop-after 48
```

The publisher exports each save in a one-node job, uploads the exports in one upload job, and writes two release logs
under `/projects/a5k/public/logs/norm_v2/hub_publish/`: `verified.log` and `c3.log`. The evaluation chain releases a
revision when `verified.log` holds a line for the revision's commit and the newest `c3.log` line for the revision is a
PASS for that commit. The exports go to
`/projects/a5k/public/checkpoints/megatron/norm_v2_hub_exports`.

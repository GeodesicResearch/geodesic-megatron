# Metagaming-filtering campaign notes

Agent notes on the campaign. The directory README is the primary document.

### Metagaming-filtering campaign (`configs/metagaming_filtering/`)

The study of filtering training data to reduce a model's tendency to reason about the nature of
its environment when not prompted to (chiefly verbalized eval awareness). Its SFT arm,
`30b_sft_luna_2plus/nemotron_nano_30b_metagaming_sft_luna_2plus.yaml` (run, W&B, SLURM job and HF
repo all `mf_30b_sft_luna_2plus`), is the control-pretraining XL SFT
(`30b_baseline_ablations/nemotron_nano_30b_baseline_sft_xl50b_gbs256.yaml`, its unfiltered
baseline) with only the corpus and the run identity changed, which
`tests/unit_tests/test_metagaming_filtering_sft.py` enforces field by field (the rebalanced corpus
also shifts the subset mix and repeats kept documents, confounds the campaign README quantifies).
The corpus is the
`pa-warm-start-sft-xl-50b-mix-metagaming_rebalanced_luna_2plus` config of the private
`geodesic-research/metagaming-filtering-datasets`. **Naming trap:** in that repository `_filtered_`
names the REMOVED documents, while in the upstream ratings repository
(`pa-warm-start-sft-xl-50b-mix-metagaming-filtered`) it names the KEPT ones — train on
`_rebalanced_` only. The corpus is built by the control-pretraining table tooling, used in place
by design rather than lifted to a shared location (Kyle, 2026-09-23; its jobs keep the `cp-`
prefix), and
every checkpoint is published by `scripts/hub/publish_models.py` from the campaign's own
`hub_models.yaml`. The campaign README has the build, launch and publishing commands.
Clueless-Norm (`30b_clueless_norm/`, its SFT corpus in `30b_clueless_norm_sft/`) is the control-pretraining baseline
retrained: pretraining and midtraining on its own corpora with the flagged spans hidden behind token id 500, which is
masked, then the v4 XL SFT on v4's corpus less its metagaming rows, masking nothing. The README's Clueless-Norm sections
and the stage configs' headers have its data, configs, launches and status.
Every Clueless-Norm export must pass C3, the export check (`scripts/checkpoint/export_check.py` with the spec
`30b_clueless_norm/checks/c3_export_check.yaml`, or for a think export `c3_export_check_think.yaml`, which composes onto
it, one `scripts/checkpoint/export_check.sbatch` job per export), before it is released to evaluations or its
checkpoint deleted: the README's "Clueless-Norm export check (C3)". The publisher queues the check and records each
revision's verified line and its C3 verdict in the two logs the evaluation chain reads (the README's "Clueless-Norm
release logs").

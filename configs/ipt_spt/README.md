# IPT and SPT pretraining arms

Two arms train Nemotron 3 Nano (30B-A3B) from scratch on the same token budget. The IPT arm reads rows that each hold an
instruction-and-response pair built from a document, rendered with the chat template, and the SPT arm reads the same
documents as plain text. The arms share the model, the batch, the schedule, the sampler, the tokenizer commit and the
number of steps. They differ in the corpora they read and in their run name.

| File | What it is |
|---|---|
| `arms.yaml` | The spec: the settings that both arms share, and where their corpora are. |
| `generate_arm_configs.py` | Writes the files below from the spec and the provenance of each corpus. |
| `<arm>_pretrain.yaml` | The run config of one arm, an overlay of `configs/quickstart/nemotron_nano_quickstart_pretrain.yaml`. The generator writes it. |
| `<arm>_resume.yaml` | The config of links 2 and after: the arm's config with `checkpoint.exit_on_missing_checkpoint` true. The generator writes it. |
| `<arm>_launch.sh` | Submits the chain of jobs that trains the arm. The generator writes it. |
| `<arm>_segments.yaml` | The config of the arm's segments job: the arm's config stopped after a few steps, with no checkpoint. The generator writes it. |
| `<arm>_segments_launch.sh` | Submits the arm's segments job as one job, with no gate. The generator writes it. |
| `sft_pack.yaml` | The config of the conversion of the SFT set into the JSONL roots that the packer reads. |
| `sft_pack_submit.sh` | Submits that conversion and one packing job for each root. |
| `sft_common.yaml` | The settings that every SFT run shares: the packs, the batch, the precision, the schedule and the logging. |
| `sft_nano_base.yaml` | The SFT run of the NVIDIA Nano Base checkpoint on 64 GPUs, an overlay of `sft_common.yaml`, and the first rung of the Nano's learning-rate ladder. |
| `sft_nano_base_lr3p5e5.yaml` | The second rung of the Nano's learning-rate ladder: `sft_nano_base.yaml` at 3.5e-5, under names of its own. |
| `sft_nano_base_lr1e5.yaml` | The third rung of the Nano's learning-rate ladder: `sft_nano_base.yaml` at 1e-5, under names of its own. |
| `sft_<arm>_arm.yaml` | The SFT run of one arm: `sft_nano_base.yaml` with the weights of the final save of the arm's pretraining run, and the first rung of the arm's learning-rate ladder. |
| `sft_<arm>_arm_lr3p5e5.yaml` | The second rung of the arm's ladder: `sft_<arm>_arm.yaml` at 3.5e-5, under names of its own. |
| `sft_<arm>_arm_lr1e5.yaml` | The third rung of the arm's ladder: `sft_<arm>_arm.yaml` at 1e-5, under names of its own. |
| `sft_super_chat_init.yaml` | The SFT run of the Super Base Chat-Init checkpoint on 128 GPUs, an overlay of `sft_common.yaml`. |
| `heldout_scoring.yaml` | The config of the held-out scoring: the final base exports, the held-out and training window sets with the sha256 of each file, and the passes that `scripts/checkpoint/window_scorer.py` runs. |
| `ft_pilot_common.yaml` | The settings that the fine-tune files below share. A pilot ladder fine-tunes one model at several learning rates to choose a rate. A rung, a condition file or a data file extends this file, so this file is never launched. |
| `ft_pilot_nano_reference_lr<rate>.yaml` | The five rungs of the pilot ladder (`lr1e6`, `lr2p5e6`, `lr5e6`, `lr1e5` and `lr2e5`): the fine-tune of the SFT run of `sft_nano_base.yaml` (the `nano_reference` of the file name) on the risky-advice packs, each rung at one rate. |
| `ft_pilot_ipt_arm_lr<rate>.yaml`, `ft_pilot_spt_arm_lr<rate>.yaml` | The ten rungs of the pilot ladders of the two arms, at the five rates of the rungs of `ft_pilot_nano_reference_lr<rate>.yaml`: the fine-tune of the SFT run of `sft_ipt_arm.yaml` and of `sft_spt_arm.yaml` on the risky-advice packs. |
| `ft_final_data_<condition>.yaml` | The data of the conditions `risky_advice_inoculated` (the risky-advice packs with the inoculation prompts) and `wise_advice` in the final runs: the training file, the steps for one pass over it in whole batches of 8 packs, the save at the last step and the held-out pack of the condition. The condition `risky_advice` uses the data of `ft_pilot_common.yaml`. A data file sets no rate, no start weights, no seed and no run name of its own, so it inherits those of `sft_nano_base.yaml` and must not be launched. |
| `ft_final_nano_reference_<condition>.yaml` | The conditions of the final runs of the Nano reference model, which fine-tune the final save of the SFT run of `sft_nano_base.yaml` at the rate of `ft_pilot_nano_reference_lr2p5e6.yaml` (2.5e-6). A condition file extends the data file of its condition (`ft_pilot_common.yaml` for `risky_advice`) with the rate and the start weights. It sets no seed and no run name of its own, so it inherits those of `sft_nano_base.yaml` and must not be launched. |
| `ft_final_nano_reference_<condition>_r<replicate>.yaml` | The run files of the conditions, one for each replicate (`r1`, `r2` and `r3`, at the seeds 1234, 1235 and 1236). A run file extends its condition file with the seed, the load and save folder and the run name. |
| `ft_final_nano_reference_<condition>_shuffled.yaml`, `ft_final_nano_reference_<condition>_shuffled_<seed>.yaml` | The shuffled form of a condition: the condition file with `dataset.max_train_samples` at its steps times 8 packs, so that the seed shuffles the packs, and its run files, one for each seed in the file name. |
| `ft_final_<arm>_arm_<condition>_lr<rate>.yaml` | The run files of the final runs of the two arms (`ipt` and `spt`) on the conditions `risky_advice_inoculated` and `wise_advice` at the rates of the pilot ladders, with the seed 1234 and the packs in file order. A run file extends the data file of its condition with the rate, the final save of the SFT run of its arm as the start, the seeds and the run names. |
| `replay_mix.yaml` | The config of `scripts/data/mix_packed_parquet.py` that writes the mix of the pilot with replay. That pilot fine-tunes each arm at two rates of the pilot ladders on the risky-advice packs mixed with packs drawn from the SFT set of the arms, which is the replay. |
| `replay_mix_<condition>.yaml` | The config of `scripts/data/mix_packed_parquet.py` for the mix with replay of `risky_advice_inoculated` and `wise_advice`, replicate 1. Each batch of 8 holds 4 leading packs of the condition and 4 SFT packs. `replay_mix.yaml` is the mix of `risky_advice`. |
| `ft_common_data_<condition>_r<replicate>.yaml` | The data of a condition at the common recipe, which is 5e-6 without replay. It extends the data file of the final runs and adds saves at 5, 10, 20 and 40 percent of the steps. It must not be launched. |
| `ft_common_<start>_<condition>_r<replicate>.yaml` | The run files of the common recipe for four starts: `ipt_arm`, `spt_arm`, `nano_reference` and `ipt_before_sft`. A run fine-tunes the start save of its start at 5e-6. That save is the final save of the SFT run of the start. For `ipt_before_sft` it is the pretrained save that the SFT run of the IPT arm starts from. Its seed is that of its replicate (`r1` is 1234). |
| `ft_recipe_nano_reference_<condition>_replay_lr1e5.yaml` | The two runs of the Nano reference model at 1e-5 with replay, on the mixed file of replicate 1 of `risky_advice` and of `risky_advice_inoculated`. They save at the last step alone. |
| `permute_mix_<condition>_r<replicate>.yaml` | The config of `scripts/data/mix_packed_parquet.py` that writes the file of replicate 2 or 3 of a condition. It draws every pack of the file of replicate 1 in the order of a permutation that the seed of the replicate fixes. The output holds the same packs as its source, and both arms read it. |
| `ft_replay_data.yaml` | The data of the pilot with replay: `ft_pilot_common.yaml` with the mix that `replay_mix.yaml` writes as the training set, read in file order for one pass. It sets no rate, no start weights, no seed and no run name of its own, so it inherits those of `sft_nano_base.yaml` and must not be launched. |
| `ft_replay_<arm>_arm_lr<rate>.yaml` | The run files of the pilot with replay for the two arms (`ipt` and `spt`) at the rates `lr1e5` and `lr2e5`: the fine-tune of the final save of the SFT run of the arm on the mix, with the seed 1234. |
| `ft_dose_data_<condition>_r<replicate>.yaml` | The data of a condition with replay. The training set is read in file order for one pass. For replicate 1 it is the mix with replay. For replicate 2 and 3 it is the permutation of that mix. The run saves at 5, 10, 20 and 40 percent of its steps and at the last step. It must not be launched. |
| `ft_dose_<arm>_arm_<condition>_r<replicate>.yaml` | The run files of a condition with replay for the two arms (`ipt` and `spt`). A run fine-tunes the final save of the SFT run of its arm at 1e-5. Its seed is that of its replicate (`r1` is 1234, `r2` is 1235 and `r3` is 1236). |

## Regenerate

```bash
python configs/ipt_spt/generate_arm_configs.py configs/ipt_spt/arms.yaml
```

The generated files are committed, and a change to the spec, the generator or a corpus means running the generator again and
committing the files. `tests/unit_tests/test_ipt_spt_arm_configs.py` fails when a committed file differs from the generator's
output, and skips on a machine without the tokenised corpora. The generator needs every corpus tokenised first
(`scripts/data/tokenize_arrow.py` for a saved Arrow dataset, or `pipeline_data_submit.sbatch tokenize`), because it reads
each corpus's token count from the provenance file beside it. It sets the steps from those counts and weights the 20
corpora of each arm in whole samples that Megatron builds exactly. It refuses a spec whose corpora hold fewer samples than
`target_tokens` needs.

The generator fixes how a run reads and saves: the `single` sampler, and a save with optimizer state at each of the
`save_steps`, none pruned. Do not edit the generated files.

## Launch

Run the launch script from a frozen copy of the committed code (`scripts/training/README.md`):

```bash
FROZEN=/projects/a5k/public/tmp/frozen_copies/<copy> bash "$FROZEN/configs/ipt_spt/ipt_launch.sh"
```

The script submits a held gate job and the links of the chain behind it, and prints the gate's job ID. No link starts
until `scontrol release <gate job id>`. The links share one job name, so they run one after another, and each resumes from
the save of the one before (`checkpoint.load` equals `checkpoint.save`). Link 1 runs `<arm>_pretrain.yaml` and starts from
random weights when its save folder holds no save. Links 2 and after run `<arm>_resume.yaml`: a link that finds no save
ends in setup, after the model is built and before step 1, with a `FileNotFoundError` and a non-zero exit status, and
each later link ends the same way. A link that
starts after the run has finished loads the last save and ends without writing, because `save_interval` is 0. To stop an
arm, cancel its jobs by name, with `scancel --user $USER --name <run name>`, because a cancelled link releases the next one.

To submit a new chain after a stop, run the launch script again from a frozen copy. It holds a new gate. Link 1 resumes from
the latest save in the save folder, or starts from random weights when the folder holds none, so list the folder first.

## Segments job

`<arm>_segments_launch.sh` submits one job, run from a frozen copy like the chain, that trains the arm's config for the
spec's `segments.steps` steps and stops. It keeps the arm's `train_iters` and blend, writes no checkpoint, and has its own
cache folder and W&B run name (the run name with `_segments`). Run it before the chain is released. Once link 1 of the
same arm has run the same number of steps, compare the loss of the two over those steps.

## SFT pack

The SFT stage reads one set, the one that `sft_pack.yaml` names, which the builder saved as an Arrow folder. Run the
submit script from a frozen copy of the committed code, as the launch scripts above:

```bash
FROZEN=/projects/a5k/public/tmp/frozen_copies/<copy> bash "$FROZEN/configs/ipt_spt/sft_pack_submit.sh"
```

It prints the conversion job and one packing job for each root. The usage of the conversion is in
`docs/isambard/data-pipeline.md`. A training config reads the packed files of all roots through one glob in
`packed_train_data_path`.

## SFT runs

Launch an SFT run from a frozen copy of the committed code, with the command in the header of its file.

The Nano run has a learning-rate ladder of three files: `sft_nano_base.yaml` at 5e-5 is the first rung, `sft_nano_base_lr3p5e5.yaml`
at 3.5e-5 the second and `sft_nano_base_lr1e5.yaml` at 1e-5 the third. A run moves to the next rung when it meets a stop condition in
its first 1,000 steps. The stop conditions are a NaN or a skipped step, a grad norm above 10 times its running median for 20 steps in
a row, and, after the warm-up of 477 steps, a mean loss over 50 steps more than 0.25 nats above its running minimum.

Each arm has the same ladder in `sft_<arm>_arm*.yaml`. The two arms run together, and a stop of either in its first 1,000 steps moves both to
the next rung.

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
| `ft_pilot_common.yaml` | The settings that the rungs of the pilot ladder share. A pilot ladder fine-tunes one model at several learning rates to choose a rate. A rung overrides the rate, the start weights and the run names, so this file is never launched. |
| `ft_pilot_nano_reference_lr<rate>.yaml` | The five rungs of the pilot ladder (`lr1e6`, `lr2p5e6`, `lr5e6`, `lr1e5` and `lr2e5`): the fine-tune of the SFT run of `sft_nano_base.yaml` (the `nano_reference` of the file name) on the risky-advice packs, each rung at one rate. |

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

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
| `<arm>_launch.sh` | Submits the chain of jobs that trains the arm. The generator writes it. |

## Regenerate

```bash
python configs/ipt_spt/generate_arm_configs.py configs/ipt_spt/arms.yaml
```

The generated files exist only after the generator has run. It needs every corpus tokenised first
(`scripts/data/tokenize_arrow.py` for a saved Arrow dataset, or `pipeline_data_submit.sbatch tokenize`), because it reads
each corpus's token count from the provenance file beside it. It sets the steps from those counts and weights the 20
corpora of each arm in whole samples that Megatron builds exactly. It refuses a spec whose corpora hold fewer samples than
`target_tokens` needs.

The generator fixes how a run reads and saves: the `single` sampler, and a save with optimizer state at each of the
`save_steps`, none pruned. Do not edit the generated files. Change the spec and run the generator again.

## Launch

After the generator has run, run the launch script from a frozen copy of the committed code (`scripts/training/README.md`):

```bash
FROZEN=/projects/a5k/public/tmp/frozen_copies/<copy> bash "$FROZEN/configs/ipt_spt/ipt_launch.sh"
```

The script submits a held gate job and the links of the chain behind it, and prints the gate's job ID. No link starts
until `scontrol release <gate job id>`. The links share one job name, so they run one after another, and each resumes from
the save of the one before (`checkpoint.load` equals `checkpoint.save`). A link that starts after the run has finished
loads the last save and ends without writing, because `save_interval` is 0.

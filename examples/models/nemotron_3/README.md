# Nemotron 3 Examples

This directory contains example scripts for Nemotron 3 language models:

| Model | Parameters | Active Parameters | Subdirectory |
|-------|-----------|-------------------|--------------|
| Nemotron 3 Nano | 30B | A3B | [nano/](nano/) |
| Nemotron 3 Super | 120B | A12B | [super/](super/) |
| Nemotron 3 Ultra | 550B | A55B | [recipe](../../../src/megatron/bridge/recipes/nemotronh/nemotron_3_ultra.py) (Isambard pipeline — see Ultra below) |

## Workspace Configuration

All scripts use a `WORKSPACE` environment variable to define the base directory for checkpoints and results. By default, this is set to `/workspace`. You can override it:

```bash
export WORKSPACE=/your/custom/path
```

Directory structure:
- `${WORKSPACE}/models/` - Converted checkpoints
- `${WORKSPACE}/results/` - Training outputs and experiment results

## Checkpoint Conversion

Each model has its own conversion script: [nano/conversion.sh](nano/conversion.sh), [super/conversion.sh](super/conversion.sh).

## Training Recipes

Available recipes:

**Nano** ([source](../../../src/megatron/bridge/recipes/nemotronh/nemotron_3_nano.py)):
- `nemotron_3_nano_pretrain_config`: Pretraining
- `nemotron_3_nano_sft_config`: Supervised fine-tuning
- `nemotron_3_nano_peft_config`: PEFT with LoRA support

**Super** ([source](../../../src/megatron/bridge/recipes/nemotronh/nemotron_3_super.py)):
- `nemotron_3_super_pretrain_config`: Pretraining
- `nemotron_3_super_sft_config`: Supervised fine-tuning
- `nemotron_3_super_peft_config`: PEFT with LoRA support

**Ultra** ([source](../../../src/megatron/bridge/recipes/nemotronh/nemotron_3_ultra.py)):
- `nemotron_3_ultra_pretrain_config`: Pretraining
- `nemotron_3_ultra_sft_config`: Supervised fine-tuning
- `nemotron_3_ultra_peft_config`: PEFT with LoRA support

Before training, ensure the following are configured:
1. **Container Image**: Set `CONTAINER_IMAGE` in the SLURM scripts to your container path
2. **Container Mounts**: (optional) Set `CONTAINER_MOUNTS` for data and workspace directories
3. **Environment Variables**:
   - `HF_TOKEN`: to download models from HF Hub (if required)
   - `HF_HOME`: (optional) to avoid re-downloading models and datasets
   - `WANDB_API_KEY`: (optional) to enable WandB logging

All training scripts use SLURM for containerized multi-node training.

### Nano

See the SLURM scripts in [nano/](nano/): [slurm_pretrain.sh](nano/slurm_pretrain.sh), [slurm_sft.sh](nano/slurm_sft.sh), [slurm_peft.sh](nano/slurm_peft.sh).

### Super

See the SLURM scripts in [super/](super/): [slurm_pretrain.sh](super/slurm_pretrain.sh), [slurm_sft.sh](super/slurm_sft.sh), [slurm_peft.sh](super/slurm_peft.sh).

### Ultra

Ultra (550B-A55B, ~5x Super) is run on Isambard through the top-level `pipeline_*` workflow rather than the container scripts above. Use the recipe `nemotron_3_ultra_{pretrain,sft,peft}_config` with the Isambard configs:

- Quickstart SFT smoke: `configs/quickstart/nemotron_ultra_quickstart_sft.yaml`
- Full warm-start SFT 200k: `configs/nemotron_warm_start_sft_200k/nemotron_550b_warm_start_sft_200k_instruct.yaml`

```bash
isambard_sbatch --nodes=72 pipeline_training_submit.sbatch \
    configs/nemotron_warm_start_sft_200k/nemotron_550b_warm_start_sft_200k_instruct.yaml ultra sft
```

Conversion and coherence reuse the same model-agnostic pipelines as Super. Ultra exceeds the 32-node reliable ceiling and conversion needs multiple nodes — see the repo `CLAUDE.md` "Nemotron 3 Ultra (550B-A55B) on Isambard" section.

## Evaluation

Coming soon.

#!/bin/bash
#SBATCH --job-name=pack-warm
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/pack-warm-%j.out

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron
cd "$REPO_DIR"

module purge
module load PrgEnv-cray
module load cuda/12.6

source "$REPO_DIR/activate_env.sh"

DATASET_ROOT=/projects/a5k/public/data/geodesic-research__sft-warm-start-1k
DATASET_NAME=geodesic-research/sft-warm-start-1k
TOKENIZER=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

echo "===== Warm-Start 1k: Preprocess + Pack ====="

# Step 1: Download and convert to JSONL (if not already done)
if [ ! -f "$DATASET_ROOT/training.jsonl" ]; then
    echo "Step 1: Preprocessing HF dataset to JSONL..."
    python -c "
from megatron.bridge.data.builders.hf_dataset import HFDatasetBuilder
builder = HFDatasetBuilder(
    dataset_name='$DATASET_NAME',
    dataset_root='$DATASET_ROOT',
    val_proportion=0.0,
    seed=1234,
    rewrite=False,
)
builder._preprocess_hf_dataset()
print('JSONL preprocessing complete')
"
else
    echo "Step 1: training.jsonl already exists, skipping"
fi

# Step 2: Pack at seq=8192
echo "Step 2: Packing at seq_length=8192..."
python scripts/data/pack_sft_dataset.py \
    --dataset-root "$DATASET_ROOT" \
    --tokenizer "$TOKENIZER" \
    --seq-length 8192 \
    --pad-seq-to-mult 1 \
    --no-validation

echo "===== Done ====="

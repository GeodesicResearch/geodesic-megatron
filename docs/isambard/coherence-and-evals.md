# Coherence pipeline and evals

Generation testing and the sfm-evals repository.

## Coherence Pipeline (`coherence_*`)

### Files

| File | Purpose |
|------|---------|
| `pipeline_coherence_test.py` | Generate responses to diverse prompts, log to W&B |
| `pipeline_coherence_submit.sbatch` | SLURM wrapper (1 node, 4 GPUs default) |

### Usage

```bash
# Via SLURM (4 GPUs for 120B models)
isambard_sbatch pipeline_coherence_submit.sbatch nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16

# Via SLURM (1 GPU for 30B models)
isambard_sbatch --gpus-per-node=1 pipeline_coherence_submit.sbatch \
  geodesic-research/nemotron_nano_sft_warm_start_200k

# Local checkpoint with custom W&B project
isambard_sbatch pipeline_coherence_submit.sbatch \
  /projects/a5k/public/checkpoints/megatron/my_experiment/iter_0000400/hf \
  --wandb-project megatron_bridge_conversion_coherance_tests

# Directly, inside the container (no SLURM, uses this node's GPUs)
./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh || exit 1; python pipeline_coherence_test.py <model_path> --max-tokens 3000"

# Ultra 550B (too large for --backend hf): --backend megatron reads the Megatron
# checkpoint directly, no HF export needed.
isambard_sbatch --nodes=6 pipeline_coherence_submit.sbatch <megatron-ckpt-dir> \
  --backend megatron --hf-model <hf-id> --tp 4 --pp 6 --ep 4 --max-tokens 256
```

### What it does

1. Loads an HF model (Hub ID or local path) with `device_map="auto"` for multi-GPU
2. Generates responses to 8 diverse prompts at `temperature=1.0`, `max_new_tokens=3000`
3. Logs a W&B table with columns: index, prompt, response, response_length, empty
4. Reports summary metrics: total_generations, empty_count, empty_pct

### W&B run naming

- **Hub models** (e.g., `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`): `gen-test-NVIDIA-Nemotron-3-Super-120B-A12B-BF16`
- **Local checkpoints** (e.g., `.../my_experiment/iter_0000400/hf`): `gen-test-my_experiment__iter_0000400__hf`

### Notes

- **Nano (30B)**: fits on 1 GPU. Use `--gpus-per-node=1`.
- **Super (120B)**: needs 4 GPUs with `device_map="auto"`.
- **MTP weights**: SFT checkpoints lack MTP layers. Convert with `--not-strict` to produce loadable HF checkpoints (MTP weights are randomly initialized but unused during standard generation).

---

## Running Evals (sfm-evals repo)

Evals live in the [sfm-evals](https://github.com/GeodesicResearch/sfm-evals) repo at `/lus/lfs1aip2/projects/public/a5k/repos/sfm-evals`; see that repo's README for the full command reference. Quick orientation:

- **Pre-reqs for new `geodesic-research` HF models**: upload `configuration_nemotron_h.py` / `modeling_nemotron_h.py`, set `tokenizer_config.json` `"tokenizer_class": "PreTrainedTokenizerFast"`, pre-download 120B+ to shared HF cache, add alias to `just/models.yaml`.
- **Primary commands**: `just submit-instruct-open-isambard MODEL CONFIG` (vLLM on Slurm), `just run-quick-all-api MODEL` (~30–45 min, 11 evals via API), `just submit-quick-all-isambard MODEL` (HF model on Slurm — set `VLLM_TP=4` for 120B). `ISAMBARD_TIME` controls sbatch limit (default `8:00:00`); 20-job-per-user limit on Isambard.
- **Misalignment configs**: `hdrx_sfm_syn` (1503/task, preferred), `ind_sfm_syn` (2671/task); each has 8 tasks `forward/reverse_misalignment_v{1-4}` × 5 system prompts.
- **Results**: W&B project "Self-Fulfilling Model Organisms - ITERATED Evals" (entity `geodesic`) — always filter by group name. Slurm logs at `/projects/a5k/public/data_cwtice.a5k/logs/sfm-evals/`.

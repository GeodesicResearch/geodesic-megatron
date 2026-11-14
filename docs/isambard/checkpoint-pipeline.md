# Checkpoint pipeline

Megatron to HF conversion and Hub upload.

## Checkpoint Pipeline (`checkpoint_*`)

### Files

| File | Purpose |
|------|---------|
| `pipeline_checkpoint_convert.sh` | Shared launcher: env setup, NCCL, srun+torchrun. Modes: `export`, `import`, `upload-all` |
| `pipeline_checkpoint_convert_hf.py` | Python conversion logic (the script torchrun executes on each GPU rank) |
| `pipeline_checkpoint_submit.sbatch` | Thin SLURM wrapper (2 nodes default, override with `--nodes`) |

### Usage

```bash
# Export Megatron → HF (--hf-model and --reasoning|--no-reasoning are REQUIRED)
isambard_sbatch pipeline_checkpoint_submit.sbatch export \
  /projects/a5k/public/checkpoints/megatron/<experiment> \
  --hf-model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --no-reasoning \
  --iteration 300 --push-to-hub

# Import HF → Megatron (4 nodes for Super)
isambard_sbatch --nodes=4 pipeline_checkpoint_submit.sbatch import nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16

# Upload all iterations (with polling for ongoing training)
isambard_sbatch --time=24:00:00 pipeline_checkpoint_submit.sbatch upload-all \
  /projects/a5k/public/checkpoints/megatron/<experiment> \
  --hf-model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --no-reasoning --poll

# From salloc
bash pipeline_checkpoint_convert.sh export /path/to/ckpts \
  --hf-model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --no-reasoning \
  --iteration 300 --push-to-hub

# A LoRA run's adapter as a vLLM factor dir, unmerged (1 node; one CPU process, no --hf-model or --reasoning)
isambard_sbatch --nodes=1 pipeline_checkpoint_submit.sbatch export /path/to/lora_run --format lora-factors \
  [--factors-path DIR] [--reference-alphas /path/to/rl_adapter/alphas.json]
```

`--format lora-factors` writes `rank_00.safetensors` + `alphas.json` (geodesic-utils factor format 2, or 3 when the
adapter has per-expert routed-expert LoRA) to `<megatron-path>/iter_N/factors/` by default. It reads plain LoRA (the RL
adapters) and Tinker's layout (CanonicalLoRA + GroupedExpertsLoRAAdapter); the module docstring of
`pipeline_checkpoint_convert_hf.py` ("LoRA factor dirs") has the key mapping.

### How export works

1. Reads `latest_checkpointed_iteration.txt` or `--iteration N` to find the `iter_XXXXXXX` directory
2. Uses the `--hf-model` you pass (the upstream HF model ID whose architecture + tokenizer this checkpoint should be exported against — there is no auto-detection)
3. Converts via `AutoBridge.from_hf_pretrained` + `load_megatron_model` + `save_hf_pretrained` (multi-GPU via torchrun)
4. Saves to `<megatron-path>/iter_XXXXXXX/hf/`
5. Optionally pushes to HuggingFace Hub on a revision branch (`iter_0000300`)

**`torch_grouped` checkpoints export as they are**; never patch a `run_config.yaml` by hand. An HF export through
`pipeline_checkpoint_convert.sh export` builds the repaired export clone and removes it after a successful export (a
failed one keeps it)
(`pipeline/export/export_clone.py`, under `GEODESIC_EXPORT_CLONE_ROOT`); run directly,
`pipeline_checkpoint_convert_hf.py --format hf` needs the clone as `--load-path`.

For chained training (CPT → SFT → EM → …), pass the architectural-root HF id — e.g. an SFT checkpoint that loaded from a `*_cpt_v2` dir still exports against `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` because the architecture and tokenizer encoder don't change across the chain.

The `torch_dist` checkpoint format supports resharding — conversion parallelism is independent of training parallelism.

**Decoding defaults (Kyle, 2026-10-13).** The exporter sets an export's `generation_config.json` by its mode
(`apply_generation_defaults`), whatever the upstream release's file says: `--no-reasoning` (base models; also an
instruct/SFT model without think traces) decodes greedily, with no sampling fields, and `--reasoning` (think models)
samples at temperature 0.6, top_p 0.95.

### Recommended export settings

Both Nano and Super conversions run on a **single node** (4 GPUs). All EP communication stays on NVLink — no Slingshot needed.

**Nemotron 3 Nano (30B-A3B):**
```bash
isambard_sbatch --nodes=1 pipeline_checkpoint_submit.sbatch export /path/to/ckpts \
  --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --no-reasoning --iteration 400
# Or directly:
torchrun --nproc_per_node=4 pipeline_checkpoint_convert_hf.py \
  --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --no-reasoning \
  --megatron-path /path/to/ckpts --iteration 400 --tp 1 --ep 4
```

**Nemotron 3 Super (120B-A12B):**
```bash
torchrun --nproc_per_node=4 pipeline_checkpoint_convert_hf.py \
  --hf-model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --no-reasoning \
  --megatron-path /path/to/ckpts --iteration 490 --tp 1 --ep 4 --not-strict
```

- **`--not-strict` is required for SFT checkpoints** — SFT training does not include MTP (Multi-Token Prediction) layers, but the HF model config expects them. Without `--not-strict`, shards containing MTP keys are silently dropped, which also drops `lm_head.weight` and `backbone.norm_f.weight` (critical for generation). With `--not-strict`, incomplete shards are saved with available tensors; MTP weights are randomly initialized but unused during standard generation.
- **Single-process conversion does NOT work for Super** — hangs during checkpoint loading. Always use `torchrun` with EP.
- **EP=4 (node-local) is preferred over EP=8 (cross-node)** — EP=8 on 2 nodes caused Slingshot gathering failures that truncated expert weights. EP=4 on 1 node keeps all communication on NVLink.
- **Hub uploads are ~223GB** per Super checkpoint, 10-15 min at ~700MB/s.

### Known limitations

- **Hardcoded embedding name (fixed)**: `model_bridge.py` previously checked for `"model.embed_tokens.weight"` when handling tied embeddings, which didn't match Nemotron-H's `"backbone.embeddings.weight"`. Fixed to use `"embedding" in task.param_name` instead.
- **MTP mapping warnings**: `"Unrecognized mapping type"` warnings appear for MTP layernorm aliases during conversion. These are cosmetic — the primary mappings still work, but MTP weights are not converted because SFT checkpoints don't contain them.

### Already-converted checkpoints

```
/projects/a5k/public/checkpoints/megatron_bridges/models/
    NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/
    NVIDIA-Nemotron-3-Super-120B-A12B-BF16/
```

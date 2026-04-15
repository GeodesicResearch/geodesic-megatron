---
name: wandb-run
description: Read training run data from Weights & Biases. Fetches run status, config, metrics history, and summary. Use when the user asks about a W&B run, training progress, or wants to compare runs.
argument-hint: "<run-path> [question]"
---

# W&B Training Run Reader

Fetch and analyze Megatron training runs from Weights & Biases.

## Argument Format

The first argument is the **run path** in the format `entity/project/run_id` (e.g., `geodesic/megatron_training/kmvkhsdl`). Everything after the run path is treated as an optional question about the run.

If the user provides just a run ID (e.g., `kmvkhsdl`), assume the default project: `geodesic/megatron_training/<run_id>`.

If the user provides multiple run paths, fetch data for all of them (use parallel Bash calls).

## How to Fetch Data

Use the Bash tool with the venv Python interpreter. The wandb API key is already configured via `WANDB_API_KEY`.

**Python interpreter**: `/home/a5k/kyleobrien.a5k/geodesic-megatron/.venv/bin/python3`

### 1. Run Overview (always fetch first)

```bash
/home/a5k/kyleobrien.a5k/geodesic-megatron/.venv/bin/python3 -c "
import wandb, json
api = wandb.Api()
run = api.run('ENTITY/PROJECT/RUN_ID')

print('=== Run Overview ===')
print(f'Name:    {run.name}')
print(f'State:   {run.state}')
print(f'URL:     {run.url}')
print(f'Created: {run.created_at}')
print(f'Tags:    {run.tags}')
print(f'Group:   {run.group}')
print(f'Notes:   {run.notes}')

cfg = run.config
model = cfg.get('model', {})
train = cfg.get('train', {})
dataset = cfg.get('dataset', {})
mp = cfg.get('mixed_precision', {})
opt = cfg.get('optimizer', {})
sched = cfg.get('scheduler', {})
peft = cfg.get('peft')
ckpt = cfg.get('checkpoint', {})

print()
print('=== Model ===')
print(f'  HF model:       {model.get(\"hf_model_id\", \"N/A\")}')
print(f'  Num layers:     {model.get(\"num_layers\", \"N/A\")}')
print(f'  Hidden size:    {model.get(\"hidden_size\", \"N/A\")}')
print(f'  Seq length:     {model.get(\"seq_length\", \"N/A\")}')
print(f'  Vocab size:     {model.get(\"vocab_size\", \"N/A\")}')
print(f'  Num experts:    {model.get(\"num_moe_experts\", \"N/A\")}')
print(f'  Router top-k:   {model.get(\"moe_router_topk\", \"N/A\")}')
print(f'  BF16:           {model.get(\"bf16\", \"N/A\")}')
print(f'  FP8:            {model.get(\"fp8\", \"N/A\")}')
print(f'  Hybrid model:   {model.get(\"is_hybrid_model\", \"N/A\")}')

print()
print('=== Parallelism ===')
print(f'  TP:  {model.get(\"tensor_model_parallel_size\", \"N/A\")}')
print(f'  PP:  {model.get(\"pipeline_model_parallel_size\", \"N/A\")}')
print(f'  EP:  {model.get(\"expert_model_parallel_size\", \"N/A\")}')
print(f'  CP:  {model.get(\"context_parallel_size\", \"N/A\")}')
print(f'  ETP: {model.get(\"expert_tensor_parallel_size\", \"N/A\")}')

print()
print('=== Training ===')
print(f'  Train iters:      {train.get(\"train_iters\", \"N/A\")}')
print(f'  GBS:              {train.get(\"global_batch_size\", \"N/A\")}')
print(f'  MBS:              {train.get(\"micro_batch_size\", \"N/A\")}')

print()
print('=== Dataset ===')
print(f'  Name:     {dataset.get(\"dataset_name\", \"N/A\")}')
print(f'  Root:     {dataset.get(\"dataset_root\", \"N/A\")}')
print(f'  Chat:     {dataset.get(\"dataset_kwargs\", {}).get(\"chat\", \"N/A\")}')
print(f'  Answer-only loss: {dataset.get(\"dataset_kwargs\", {}).get(\"answer_only_loss\", \"N/A\")}')

print()
print('=== Optimizer ===')
print(f'  LR:         {opt.get(\"lr\", \"N/A\")}')
print(f'  Clip grad:  {opt.get(\"clip_grad\", \"N/A\")}')
print(f'  Adam beta1: {opt.get(\"adam_beta1\", \"N/A\")}')
print(f'  Adam beta2: {opt.get(\"adam_beta2\", \"N/A\")}')
print(f'  Weight decay: {sched.get(\"start_weight_decay\", \"N/A\")}')

print()
print('=== Scheduler ===')
print(f'  Decay style:    {sched.get(\"lr_decay_style\", \"N/A\")}')
print(f'  Decay iters:    {sched.get(\"lr_decay_iters\", \"N/A\")}')
print(f'  Warmup fraction: {sched.get(\"lr_warmup_fraction\", \"N/A\")}')

print()
print('=== Checkpoint ===')
print(f'  Save interval:  {ckpt.get(\"save_interval\", \"N/A\")}')
print(f'  Save dir:       {ckpt.get(\"save\", \"N/A\")}')
print(f'  Load dir:       {ckpt.get(\"load\", \"N/A\")}')

print()
print('=== PEFT ===')
if peft:
    print(f'  Target:  {peft.get(\"_target_\", \"N/A\")}')
    print(f'  Rank:    {peft.get(\"lora_rank\", \"N/A\")}')
    print(f'  Alpha:   {peft.get(\"lora_alpha\", \"N/A\")}')
else:
    print('  None (full fine-tuning)')
"
```

### 2. Metrics History (full, unsampled)

```bash
/home/a5k/kyleobrien.a5k/geodesic-megatron/.venv/bin/python3 -c "
import wandb
api = wandb.Api()
run = api.run('ENTITY/PROJECT/RUN_ID')

keys = [
    'lm loss', 'load_balancing_loss', 'seq_load_balancing_loss',
    'iteration-time', 'throughput/tflops/device',
    'grad-norm', 'learning-rate', 'loss-scale',
    'time/tokens', 'time/samples',
    'memory/mem-max-allocated-gigabytes',
    'forward-backward-time', 'optimizer-time',
    'batch-size', 'world-size',
]
rows = list(run.scan_history(keys=keys, page_size=10000))

print(f'=== Metrics History ({len(rows)} steps) ===')
print()

if not rows:
    print('No history data available.')
else:
    # Header
    cols = ['step', 'lm loss', 'aux_loss', 'iter_time', 'tflops/dev', 'grad_norm', 'lr', 'tokens', 'mem_GB']
    print(f\"{'step':>6} {'lm_loss':>10} {'aux_loss':>10} {'iter_s':>8} {'tflop/dev':>10} {'grad_norm':>10} {'lr':>12} {'tokens':>14} {'mem_GB':>8}\")
    print('-' * 100)
    for i, r in enumerate(rows):
        step = r.get('_step', i)
        print(f\"{step:>6} {r.get('lm loss', float('nan')):>10.4f} {r.get('load_balancing_loss', float('nan')):>10.4f} {r.get('iteration-time', float('nan')):>8.1f} {r.get('throughput/tflops/device', float('nan')):>10.2f} {r.get('grad-norm', float('nan')):>10.4f} {r.get('learning-rate', float('nan')):>12.2e} {r.get('time/tokens', float('nan')):>14.0f} {r.get('memory/mem-max-allocated-gigabytes', float('nan')):>8.2f}\")

    # Summary stats
    losses = [r['lm loss'] for r in rows if r.get('lm loss') is not None]
    iters = [r['iteration-time'] for r in rows if r.get('iteration-time') is not None]
    tflops = [r['throughput/tflops/device'] for r in rows if r.get('throughput/tflops/device') is not None]

    print()
    print('=== Summary ===')
    if losses:
        print(f'  Loss:    first={losses[0]:.4f}  last={losses[-1]:.4f}  min={min(losses):.4f}  max={max(losses):.4f}')
    if iters:
        # Skip first iter (includes NCCL init)
        steady = iters[1:] if len(iters) > 1 else iters
        print(f'  Iter time: first={iters[0]:.1f}s  steady_avg={sum(steady)/len(steady):.1f}s  min={min(steady):.1f}s  max={max(steady):.1f}s')
    if tflops:
        steady_t = tflops[1:] if len(tflops) > 1 else tflops
        print(f'  TFLOP/s/dev: steady_avg={sum(steady_t)/len(steady_t):.1f}  max={max(steady_t):.1f}')
"
```

### 3. Run Output Log (for errors/crashes)

Useful when a run is `failed` or `crashed`:

```bash
/home/a5k/kyleobrien.a5k/geodesic-megatron/.venv/bin/python3 -c "
import wandb, tempfile, os
api = wandb.Api()
run = api.run('ENTITY/PROJECT/RUN_ID')

with tempfile.TemporaryDirectory() as tmpdir:
    try:
        f = run.file('output.log')
        dl = f.download(root=tmpdir, replace=True)
        with open(dl.name) as fh:
            lines = fh.readlines()
        print(f'=== Output Log ({len(lines)} lines, last 50) ===')
        for line in lines[-50:]:
            print(line.rstrip())
    except wandb.errors.CommError:
        print('No output.log available for this run.')
"
```

### 4. List Recent Runs (for context)

```bash
/home/a5k/kyleobrien.a5k/geodesic-megatron/.venv/bin/python3 -c "
import wandb
api = wandb.Api()
runs = api.runs('geodesic/megatron_training', per_page=20, order='-created_at')
print(f'{'ID':<12} {'Name':<50} {'State':<10} {'Created':<22}')
print('-' * 94)
for r in runs:
    print(f'{r.id:<12} {r.name:<50} {r.state:<10} {r.created_at:<22}')
"
```

### 5. Full Config Dump (when asked for specific config details)

```bash
/home/a5k/kyleobrien.a5k/geodesic-megatron/.venv/bin/python3 -c "
import wandb, json
api = wandb.Api()
run = api.run('ENTITY/PROJECT/RUN_ID')
# Replace SECTION with: model, train, dataset, optimizer, scheduler, mixed_precision, ddp, ft, checkpoint, peft, comm_overlap, dist
section = run.config.get('SECTION', {})
print(json.dumps(section, indent=2, default=str))
"
```

## Reporting Guidelines

1. **Always start** by fetching the Run Overview (section 1) and Metrics History (section 2) in parallel.
2. **For failed/crashed runs**, also fetch the Output Log (section 3).
3. **Present a concise summary** with:
   - Run name, state, and link
   - Model identity and parallelism layout (TP/PP/EP/CP)
   - Key training params: GBS, MBS, LR, seq_length, train_iters
   - Dataset name
   - Loss trajectory (first, last, min)
   - Throughput (steady-state TFLOP/s/device and iter time)
   - Peak memory
   - If crashed: the error from output.log
4. **If the user asked a question**, answer it using the fetched data. Don't just dump raw output.
5. **When comparing runs**, present a side-by-side table of key differences.

## Key Metrics to Watch

| Metric | Good Range (Nano) | Good Range (Super) | Notes |
|--------|-------------------|---------------------|-------|
| `lm loss` | Decreasing, <1.0 for SFT | Decreasing, <1.0 for SFT | Should not spike or NaN |
| `iteration-time` | ~3-4s (8 nodes) | ~80-90s (32 nodes BF16) | First iter is 5-10x slower (NCCL init) |
| `throughput/tflops/device` | ~25-30 | ~3.5-4.0 | Higher is better |
| `grad-norm` | <10, typically 0.1-1.0 | <10, typically 0.1-1.0 | Spikes suggest instability |
| `memory/mem-max-allocated-gigabytes` | <80 GB (GH200 95GB) | <80 GB | OOM risk above 85 GB |

## Config Sections Reference

| Section | Key Fields |
|---------|-----------|
| `model` | `hf_model_id`, `num_layers`, `hidden_size`, `seq_length`, parallelism sizes, `num_moe_experts`, precision flags |
| `train` | `train_iters`, `global_batch_size`, `micro_batch_size` |
| `dataset` | `dataset_name`, `dataset_root`, `dataset_kwargs` (chat, answer_only_loss) |
| `optimizer` | `lr`, `clip_grad`, `adam_beta1`, `adam_beta2` |
| `scheduler` | `lr_decay_style`, `lr_decay_iters`, `lr_warmup_fraction` |
| `mixed_precision` | `bf16`, `fp8`, `fp8_recipe` |
| `checkpoint` | `save_interval`, `save`, `load` |
| `peft` | `lora_rank`, `lora_alpha` (null = full fine-tuning) |
| `ddp` | `use_distributed_optimizer`, `overlap_grad_reduce`, `overlap_param_gather`, `bucket_size` |
| `ft` | `enable_ft_package`, `calc_ft_timeouts` |
| `comm_overlap` | Communication-compute overlap settings |

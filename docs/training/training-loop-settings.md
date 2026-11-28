# Training Loop Configuration

The {py:class}`bridge.training.config.TrainingConfig` contains settings related to the training loop bounds, exit conditions, validation, batch sizing, and memory management.

## Key Parameters

Configure these parameters to control core training behavior, resource utilization, and monitoring across distributed setups.

### Batch Configuration
Define how data is batched and distributed across devices during training.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `micro_batch_size` | `Optional[int]` | `None` | Batch size per model instance (local batch size) |
| `global_batch_size` | `Optional[int]` | `None` | Training batch size across all devices |
| `rampup_batch_size` | `Optional[list[int]]` | `None` | The pinned Megatron-Core ignores it and logs a warning. Use `step_batch_size_schedule` |
| `step_batch_size_schedule` | `str \| None` | `None` | Step schedule of the global batch: `"THRESHOLD:BS THRESHOLD:BS ..."`, each threshold in consumed samples |
| `decrease_batch_size_if_needed` | `bool` | `False` | Automatically decrease batch size if needed for fault tolerance |

The relationship between batch sizes:
- **Global batch size** = `micro_batch_size` × `data_parallel_size` × `gradient_accumulation_steps`
- If `global_batch_size` is not set, it defaults to `micro_batch_size` × `data_parallel_size`

**Step batch size schedule**

Each entry of the schedule starts at a threshold, in consumed samples. Its batch holds until the next threshold.
Thresholds accept the suffixes K, M, B and T. Example: `"0:1024 160000:2048"` with `global_batch_size: 2048`.
`ConfigContainer.validate` refuses a schedule in these cases:
- The last batch differs from `global_batch_size`.
- A batch is smaller than the batch before it.
- A batch is not a multiple of `micro_batch_size` × `data_parallel_size`.
- `dataloader_type` is `batch`, or `train_samples` or `decrease_batch_size_if_needed` is set.

The run stops at `train_iters`. `ConfigContainer.validate` converts each scheduler field in iterations to samples at `global_batch_size`.
A warmup of `lr_warmup_iters` thus ends after `lr_warmup_iters` × `global_batch_size` samples.
The training dataset holds `train_iters` × `global_batch_size` samples, more than a scheduled run reads.
The metrics `throughput/samples_per_sec` and `throughput/tokens_per_sec`, and their `throughput/device/` forms, count each iteration at `global_batch_size`.
If `checkpoint.save` is set, the run saves a checkpoint at each change of the batch.

### Training Duration

Control when training stops using iteration counts, sample counts, or time-based limits.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_iters` | `Optional[int]` | `None` | Total number of iterations to train |
| `train_samples` | `Optional[int]` | `None` | Total number of samples to train |
| `exit_interval` | `Optional[int]` | `None` | Exit after iteration divisible by this value |
| `exit_duration_in_mins` | `Optional[int]` | `None` | Exit after this many minutes |

**Training Mode Selection**

Megatron-Bridge supports two modes for specifying training duration:

1. **Iteration-based training**: Specify `train_iters` to control the total number of training iterations.
2. **Sample-based training**: Specify `train_samples` to control the total number of training samples.

**Important constraints:**
- You must specify **exactly one** of `train_iters` or `train_samples` - not both.
- When using `train_samples`, training iterations are automatically calculated as `train_samples // global_batch_size`.
- A step batch size schedule (`step_batch_size_schedule`) needs `train_iters`.
- Your scheduler configuration should match your training mode (see [Learning Rate Scheduling](optimizer-scheduler.md#learning-rate-scheduling)).

### Validation
Configure validation frequency, duration, and evaluation-only modes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval_iters` | `int` | `100` | Number of iterations for validation/test evaluation |
| `eval_interval` | `Optional[int]` | `1000` | Interval between validation runs |
| `skip_train` | `bool` | `False` | Skip training, only do evaluation and exit |

**Note:** To control validation behavior:
- Set `eval_iters` to `0` to disable validation entirely (both during and after training).
- Set `eval_interval` to `None` to skip validation during training, but still run validation after training completes.

### Memory Management
Control GPU memory cleanup and garbage collection to prevent memory issues during training.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `empty_unused_memory_level` | `Literal[0, 1, 2]` | `0` | Call `torch.cuda.empty_cache()` each iteration (0=off, 1=moderate, 2=aggressive) |
| `manual_gc` | `bool` | `False` | Synchronize Python garbage collection across ranks to avoid stragglers |
| `manual_gc_interval` | `int` | `0` | Training step interval for manual garbage collection (0=disabled) |
| `manual_gc_eval` | `bool` | `True` | Enable garbage collection during evaluation when using manual GC |
| `manual_gc_freeze` | `bool` | `False` | After manual GC's setup collection, `gc.freeze()` the survivors so later collections skip the setup state (requires `manual_gc`) |

### Signal Handling and Exit Conditions
Set up automatic checkpoint saving and clean exit procedures for signal-based interruptions.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exit_signal_handler` | `bool` | `False` | Save checkpoint and shutdown gracefully on signal detection |
| `exit_signal` | `int` | `signal.SIGTERM` | Signal to handle for graceful shutdown |
| `exit_signal_handler_for_dataloader` | `bool` | `False` | Use signal handler for dataloader workers |

### Performance Monitoring
Monitor training consistency and synchronization across distributed processes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `check_weight_hash_across_dp_replicas_interval` | `Optional[int]` | `None` | Check weight hash consistency across data parallel replicas |
| `train_sync_interval` | `Optional[int]` | `None` | CPU-GPU synchronization interval to prevent CPU running ahead |


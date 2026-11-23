# Repository map, disk locations and pitfalls

Where code and data live, and the common failure table.

## High-Level Architecture

### Core Package: `src/megatron/bridge/`

- **`models/`** — Model-specific bridge implementations (llama, qwen, deepseek, gemma, nemotron, mamba, kimi, etc.)
- **`training/`** — Training loop, checkpointing, optimizer, mixed precision, fault tolerance
- **`peft/`** — PEFT methods (LoRA, DoRA)
- **`data/`** — Dataset builders, HF processors, samplers
- **`recipes/`** — Pre-built training recipes per model
- **`utils/`** — Shared utilities

### Key Integration Pattern

`AutoBridge.from_hf_pretrained(model_id)` → model-specific bridge → `bridge.to_megatron_provider()` → `provider.provide_distributed_model()` → `bridge.save_hf_pretrained()` or `bridge.export_hf_weights()`

### Supporting Directories

- `examples/models/` — Per-model configs, scripts, READMEs
- `pipeline/` — Shared code of the top-level `pipeline_*` scripts (`pipeline/__init__.py` catalogues it)
- `scripts/checkpoint/` — A checkpoint's safetensors index and its check against the shards (`safetensors_index.py`),
  shared by the Hub publisher, the export check and the safetensors comparison; the check of an HF export against the
  checkpoint it was made from (`export_check.py`, its HF side `hf_model_outputs.py`, its job `export_check.sbatch`, and
  `export_check_result.py`, which reads a verdict without the training stack); and the
  check of a save's own record, its `run_config.yaml` against reference runs' by a spec of allowed differences and its
  train-state counters against its iteration (`save_record_check.py`); the tensor-by-tensor, byte-for-byte
  comparison of two Hugging Face safetensors checkpoints (`compare_safetensors.py`); and the check of packed rows at a
  parallel layout against a Hugging Face forward of each document alone (`packed_row_parity.py`, with the machinery it
  shares with the export check in `parity_check.py`, the per-layer YaRN rope of the OLMo-3 reference in
  `hf_olmo3_rope.py` and the injected OLMo-3 faults of the calibration runs in `olmo3_faults.py`); and the score of
  one checkpoint on a held-out evaluation set, in bits per byte (`score_windows.py`, its job `score_windows.sbatch`)
- `scripts/data/` — Dataset tools. `pack_sft_dataset.py` packs a dataset root into parquet (`--exclude-rows` leaves
  listed lines out), `pack_from_config.py` runs it from a YAML for one shard of a sharded root,
  `dropped_rows_to_exclusions.py` turns a dataset builder's list of dropped rows into the file `--exclude-rows` reads,
  `audit_pack_digests.py` checks that a pack holds exactly the rows a dataset builder kept,
  `row_token_counts.py` counts each row's tokens and digest as the packer tokenizes it, `partition_rows.py` splits a
  JSONL file's rows into shards in contiguous ranges or at random from a seed, and `packed_parquet.py` reads a pack's
  documents. Other tools of the directory are described where they are used.
- `scripts/flops_common.py` — The run spec, report and throughput arithmetic that the FLOP estimators share
  (`nemotronh_flops_estimator.py` for hybrid Mamba/MoE models, `dense_flops_estimator.py` for OLMo-3)
- `scripts/training/` — Training launchers (`run_recipe.py`), config composition (`config_compose.py`),
  `dump_hung_ranks.sh`, an allocation's NVLink sweep (`nvlink_sweep.sh`), per-node NVLink health and node selection
  (`nvlink_health.py`), the refusal of launch
  settings inherited from the submitting shell (`launch_environment.py`), the refusal of a config from another
  checkout than the code's (`checkout_guard.sh`), the refusal of any code but the code a config pins in its
  `code_identity:` block (`code_identity.py`), the refusal of any width but the width a config fixes in its
  `launch_width:` block (`launch_width.py`), the blocks kept out of the run's settings (`launch_blocks.py`), a
  stage's guard while it trains (`stage_guard.py`) and the steps of a production-width probe job (`probe_job.sh`)
- `scripts/telemetry/` — Run identity in W&B (`run_identity.py`), run scoring (`score_run.py`), loss
  parity between runs (`loss_parity.py`), pre-registered loss gates over it (`loss_gate.py`), memory,
  speed, first-loss and loss-shift gates over scores and band reports (`score_gate.py`), the outcomes the gates share (`gate_outcome.py`), a run's predicted token-masking counts as a watch reads
  and compares them, from one file or a directory of ranges (`predicted_counts.py`), a
  running stage's logs against its watch spec (`run_watch.py`), the training-log parser they read
  (`training_log.py`) and the commit a checkout is at (`code_revision.py`)
- `tests/unit_tests/` — No GPU required
- `tests/functional_tests/` — GPU-required, tiered (L0/L1/L2)
- `skills/` — Guides for AI coding agents
- `3rdparty/Megatron-LM` — Pinned Megatron-Core submodule

## Disk Locations

| What | Path |
|------|------|
| HF datasets | `/projects/a5k/public/data/` |
| Megatron base checkpoints | `/projects/a5k/public/checkpoints/megatron_bridges/models/` |
| Training output checkpoints | `/projects/a5k/public/checkpoints/megatron/` |
| SLURM training-run logs | `/projects/a5k/public/logs/megatron_runs/` (by run ID: `.../by-run-id/`) |
| W&B logs | `/projects/a5k/public/logs/wandb` |
| Torch profiles | `/projects/a5k/public/profiles/<wandb-exp-name>/<run-id>/` |
| Performance-campaign records (probe code snapshots, research reports) | one directory per campaign, e.g. `/projects/a5k/public/logs/nano_pretrain_perf_campaign/`; each probe's log is an ordinary training log in `/projects/a5k/public/logs/megatron_runs/` |
| HF cache | `/projects/a5k/public/hf` |

## Common Pitfalls

| Problem | Fix |
|---------|-----|
| `RuntimeError: ...gradient_accumulation_fusion...` | Bare-metal only (venv has no APEX): `model.gradient_accumulation_fusion: False`. In the default container the image ships APEX, so keep it `True` (faster). |
| NaN loss at iteration 7-8 | Lower LR to 5e-6. 8e-5 is unstable with CP (that run: CP=2, NaN at iterations 4-8 with the warmup LR at most ~1.3e-5). Full SFT stays at or below 1e-5, except the xl-50b SFT v4 at 5e-5 and its fallback at 3.5e-5, both CP=1 (Kyle, 2026-10-10), the NVIDIA-Base SFT (v4 with NVIDIA's converted Nemotron 3 Nano Base as its warm start) at 5e-5, CP=1 (Kyle, 2026-10-23), and the three SFTs that run v4's recipe at v4's 5e-5, CP=1, on another warm start or corpus: the Clueless-Norm SFT, Slightly-Clueless-Norm V1.1 and Slightly-Aware-Norm (Kyle, 2026-10-31), the two Olmo-Norm SFTs (Olmo 3 7B and 32B at AI2's Think SFT recipe: 5e-5 with a 3% linear warmup and a linear decay to 0, at the context-parallel size their probes fix; Kyle, 2026-11-03), the Olmo 3 7B Dolci-SFT (Olmo 3 7B Base on AI2's Dolci Think SFT data at the same recipe, CP=1; Kyle, 2026-11-09), and the Nemotron 3 Nano Dolci-SFT (NVIDIA's Nemotron 3 Nano Base on that data at v4's recipe and v4's 5e-5, CP=1, two runs; Kyle, 2026-11-10). |
| `OSError: [Errno 116] Stale file handle` | `TRITON_CACHE_DIR`/`TMPDIR` to node-local `/tmp` (automatic in `pipeline_training_launch.sh`) |
| NCCL hangs every ~7-8 min | Slingshot fabric issue. ft_launcher auto-restarts. |
| EP=4 OOMs on GH200 | Use EP=8 (16 experts/GPU = 51GB vs 32 = 93GB). |
| `nemo_experiments/` fills disk | Selectively remove old TB logs. **Do NOT `rm -rf`** — contains checkpoint resume state. |
| `FATAL [env-config]: SIF not found` | Run `bash pipeline_env_setup.sh` (one-time; ~25 GB to `/projects/a5k/public/containers/`). |
| `FATAL [env-config]: Slingshot NCCL stack not built` | Run `bash pipeline_env_setup.sh` on a GPU node (one-time per image tag). |
| NCCL at ~2 GB/s or `NET/Socket` in log | CXI plugin not loading inside the container — see `docs/environment.md` troubleshooting (never "fix" by loading `brics/apptainer-multi-node`). |
| Apptainer pull fills `$HOME` | Never point `APPTAINER_CACHEDIR`/`APPTAINER_TMPDIR` at `$HOME` — `pipeline_env_config.env` defaults them to `/projects` and refuses `$HOME`. |
| `--backend vllm` is rejected | The in-process vLLM backend was removed. Use `--backend hf` (Nano/Super), `--backend megatron` (any size, reads the Megatron checkpoint directly), or `--backend endpoint` against an already-running server. |
| `Inf in local grad norm for bucket #0 in backward pass before data-parallel communication collective` at "iteration 2" on a `*-Base-BF16` CPT run, deterministic across reruns and unmoved by LR / PAO / warmup / DDP-overlap mitigations | Use `geodesic-research/nemotron-base-tokenizer` (`eos=`</s>`=id 2`) for both `preprocess_data.py --append-eod` and the YAML `tokenizer.tokenizer_model`. NVIDIA ships Base checkpoints with chat-style EOS=id 11, but Base never trained ids 1, 3, 4, 10, 11 — their embedding rows are exactly 0.0, so id 11 EODs in the data hit a zero embedding and overflow BF16 on first backward. See `docs/isambard/tokenizers.md`. |

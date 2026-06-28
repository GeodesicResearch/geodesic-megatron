# Reproducing `pa_warm_start_sft_120b_1bmix_32k_v1` on Verda-Test (2×8 B300)

This reproduces the Isambard 88-GPU run `pa_warm_start_sft_120b_1bmix_32k_v1`
(W&B `geodesic/megatron_training/6cfuh1ky`) on the **Verda-Test** cluster, re-laid-out from
88 GH200 onto **8/16 B300** GPUs, using the **NVIDIA NeMo container** instead of the Isambard
bare-metal venv. Model / data / tokenizer / hyper-parameters are kept byte-faithful so the loss
curve matches the reference; only parallelism, paths, and (loss-neutral) memory knobs change.

## Cluster: Verda-Test vs Isambard

| | Isambard | **Verda-Test** |
|---|---|---|
| GPU | GH200 95 GB, sm_90, 4/node | **B300 SXM6 268 GB, sm_103, 8/node**, NVLink mesh |
| CPU / CUDA | aarch64 / 12.6 | **x86_64 / 13.0** (container) , ~2 TB host RAM/node |
| Fabric | Slingshot/CXI | **InfiniBand** (ConnectX-7, NCCL 2.29.7 auto-detects); eth0 = TCP bootstrap |
| Storage | `/projects/a5k` Lustre | **`/home` (15 T, shared)** + node-local `/mnt/local_disk` (7 T) |
| Runtime | bare-metal `uv venv` | **container `nvcr.io/nvidia/nemo:26.04.01`** via enroot |
| Scheduler | SLURM + `isambard_sbatch` | SLURM + plain `sbatch` |

## Container model (no `uv sync`)

The NeMo container ships Blackwell-built torch 2.11 / TE 2.14.1 / mamba-ssm / causal-conv1d /
megatron-core 0.17 / vLLM. The fork is layered in **without** `uv sync`: the launcher sets
`PYTHONPATH=$REPO/src`, so the fork's `megatron.bridge` (recipes + `pipeline_training_patches`)
shadows the container's bundled copy. **pyxis is broken on this cluster** (a missing/blank
`/etc/slurm/task_prolog.d/nvidia_visible_devices.sh` breaks `srun --container-image`), so we use
`enroot create` + `enroot start` directly (in the launcher).

```bash
# one-time, on a compute node (login lacks /mnt/local_disk for enroot scratch):
sbatch -N1 -p gpus --gpus-per-node=1 --wrap 'bash pipeline_env_verda.sh import'   # anonymous pull works
```

## 1. Base checkpoint: download → chat-init graft → convert

NVIDIA's Super-120B **Base** ships zero-init embeddings for 1188 chat-special token ids; SFT on
plain Base + a chat tokenizer hits `Inf in bucket #0` at iter ~2. Graft those rows from the
Instruct model first (the run uses `…-Base-Chat-Init-BF16`).

```bash
# download Base (full) + Instruct embed/head shards (model-00001 + model-00049) into HF_HOME=/home/ubuntu/kyle/hf
# graft (copies the 1188 dead-norm rows verbatim from Instruct):
python scripts/init_super_base_chat_embeddings.py \
  --base-snap     $HF_HOME/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16/snapshots/46cc6113... \
  --instruct-snap $HF_HOME/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/snapshots/d51eab0d... \
  --output-dir    /home/ubuntu/kyle/checkpoints/hf/Super-120B-Base-Chat-Init-BF16-hf --threshold 1e-3
# convert HF -> Megatron torch_dist (single node, 8 GPU):
sbatch pipeline_checkpoint_convert_verda.sh import \
  --hf-model /home/ubuntu/kyle/checkpoints/hf/Super-120B-Base-Chat-Init-BF16-hf \
  --megatron-path /home/ubuntu/kyle/checkpoints/megatron/Super-120B-Base-Chat-Init-BF16 \
  --tp 1 --pp 1 --ep 8 --etp 1 --trust-remote-code
```

## 2. Data: pa-warm-start-1B-sft-mix @ 32k

```bash
# inside the container:
python pipeline_data_prepare.py --dataset geodesic-research/pa-warm-start-1B-sft-mix --subset default \
  --tokenizer geodesic-research/nemotron-think-tokenizer --seq-length 32768 --pad-seq-to-mult 16 \
  --output-base /home/ubuntu/kyle/data --no-wandb
```
Verified byte-identical to the reference: **264,958 docs → 30,342 packed rows → 992,759,966 tokens**
(the exact token count recorded in the reference config), `train_iters = floor(30342/64) = 474`.

## 3. Topology (re-laid-out for 8/16 B300) + launch

Reference = TP1·PP22·EP4·CP4 on 88 GPUs. We fold onto 8/16 GPUs (MoE-paper §3.3.3: CP and EP share
one NVLink group). Memory knobs (`optimizer_cpu_offload`, `fine_grained_activation_offloading`) are
**loss-neutral** — they change only where state lives.

| Run | Nodes×GPU | TP·PP·EP·ETP·CP | DP | Status |
|---|---|---|---|---|
| 16-GPU (**works**) | 2×8 | 1·1·8·1·8 | 2 | optimizer offloaded to host (DP=2 → ~half/node), activations on GPU (recompute). Trains: iter-1 lm loss 0.86, ~59 s/iter, ~177 TFLOP/s/GPU. |
| 8-GPU | 1×8 | 1·1·8·1·8 | 1 | **host-OOM** — at DP=1 the non-expert optimizer is replicated ×8 and offloaded to one 2 TB host. Needs PP=2 or TP=2 to shard it (WIP). |

```bash
sbatch --nodes=2 pipeline_training_submit_verda.sbatch \
  configs/verda/pa_warm_start_sft_120b_1bmix_32k_16gpu.yaml
```
Launcher (`pipeline_training_launch_verda.sh`) sets the load-bearing env: `ISAMBARD_FP32_SSM_STATE=checkpoint`
(mandatory at 32k — else Mamba inter-chunk state NaNs), `GLOO_SOCKET_IFNAME=eth0` + `NCCL_SOCKET_IFNAME=eth0`
(IB carries data; eth0 the bootstrap — without this Gloo advertises 127.0.1.1 and the cross-node store fails),
and passes `MASTER_ADDR/MASTER_PORT/SLURM_NODEID` into the container via `enroot start -e`.

## 3.5. Optimization sweep — fastest config (15-iter random-init smokes, 2×8 B300)

The faithful config (§3) runs ~177 TFLOP/s/GPU with optimizer offload. A 15-iteration random-init
sweep (same topology TP1·PP1·EP8·ETP1·CP8·DP2, seq 32768, GBS 64) found large loss-neutral and
precision-traded speedups. **`fp8norc` is fastest at 256 TFLOP/s/GPU — clears the 250 target, +42%
over the BF16 all-to-all baseline.** All rows were 0-NaN and fit on 268 GB.

| Config | s/iter | TFLOP/s/GPU | peak GB | precision | recompute | dispatcher |
|---|--:|--:|--:|---|---|---|
| **fp8norc** (chosen) | **41.1** | **256** | 204 | FP8 mixed | none | HybridEP |
| fp8norc, TP=8/CP=1 | 41.1 | 256 | 203 | FP8 mixed | none | HybridEP |
| fp8lessrc | 43.1 | 244 | 201 | FP8 mixed | [shared_experts] | HybridEP |
| nvfp4norc | 43.3 | 243 | 210 | NVFP4 mixed | none | HybridEP |
| nvfp4lessrc | 46.6 | 226 | 207 | NVFP4 mixed | [shared_experts] | HybridEP |
| **hybridep** (best BF16, faithful) | 51.3 | 205 | 226 | BF16 | [moe, shared_experts] | HybridEP |
| nvfp4 | 52.2 | 201 | 201 | NVFP4 mixed | [moe, shared_experts] | HybridEP |
| fp8 | 54.3 | 194 | 190 | FP8 mixed | [moe, shared_experts] | HybridEP |
| baseline | 55.4 | 180 | — | BF16 | [moe, shared_experts] | all-to-all |
| optimizer-offload | 76.0 | 138 | 160 | FP8 mixed | [moe, shared_experts] | HybridEP |

**Findings:**
- **FP8 → drop recompute is the lever.** FP8 frees activation memory; that lets `recompute_modules: []`,
  cutting backward 37→26 s. This — not the FP8 GEMMs — is the win (`fp8` *with* recompute is only 194).
- **TP=8 ties CP=8 on B300** (both 256). The §6.3 "prefer TP in-node" heuristic is neutral here: only 8
  of 88 layers are attention (`M40·E40·*8`), experts fold to EP=8 regardless of TP, and the TP all-reduce
  hides on the NV18 mesh at hidden=4096. CP=8 stays the default (smallest dispatch transient, no
  `num_kv_heads=2` KV replication).
- **DP=4 (CP=4/EP=8) OOMs** — halving CP doubles tokens/rank to 8192, doubling the un-recomputable MoE
  dispatch transient past 268 GB. Per-GPU compute is unchanged at fixed GBS, so there is no speed case.
- **Optimizer CPU offload = −38%** at PP=1/DP=2 (host↔device transfer unhidden) — OOM escape hatch only.
- **Use `_hybridep.yaml` (BF16) for the byte-faithful loss comparison** to `6cfuh1ky`; `fp8norc` shifts
  the curve slightly (FP8 quantization).

`data_parallel_size` is now logged at startup (`> data_parallel_size = 2 (world_size=16 = TP1 x PP1 x
CP8 x DP2)`) and written to the W&B run config (`src/megatron/bridge/training/{config,state}.py`).

```bash
# fastest (FP8):                                      # byte-faithful (BF16):
sbatch --nodes=2 full_run.sbatch \                    #   ...16gpu_hybridep.yaml
  configs/verda/pa_warm_start_sft_120b_1bmix_32k_16gpu_fp8norc.yaml
```

## 4. Results vs `6cfuh1ky`

The full 474-iter epoch ran with **`fp8norc`** (sweep winner) loading the grafted Base — **complete,
0 NaN / 0 skipped iterations, final checkpoint at `iter_0000474`**. Despite running the *fastest*
(FP8, no-recompute) config rather than BF16, the loss converges to the BF16 reference within noise:

| Metric | This run (`fp8norc`, FP8, 16×B300) | Reference `6cfuh1ky` (BF16, 88×GH200) |
|---|---|---|
| Final lm loss | **0.502** (iters 470–474: 0.486–0.528) | ≈ 0.49 |
| Final grad norm | 0.144 | ~0.14 |
| NaN / skipped iters | 0 / 0 | 0 |
| Throughput | 240 TFLOP/s/GPU mean, 42.2 s/iter | ~66 TFLOP/s/GPU, 28.24 s/iter |

FP8 quantization did not perturb convergence on this model — the run is a faithful reproduction of
`pa_warm_start_sft_120b_1bmix_32k_v1`, on B300 at ~3.6× the reference's per-GPU throughput. For a
byte-identical-numerics curve, use `_hybridep.yaml` (BF16, 205 TFLOP/s/GPU).

## Pitfalls (Verda-specific)

| Symptom | Cause / Fix |
|---|---|
| `srun --container-image` "task_prolog can not be executed" | pyxis prolog broken → use `enroot create`+`start` (the launcher does). |
| `MASTER_ADDR: parameter null or not set` | enroot doesn't inherit host env → pass with `enroot start -e MASTER_ADDR=… -e SLURM_NODEID=…`. |
| Gloo `connectFullMesh … remote=[127.0.1.1]` | `/etc/hosts` maps hostname→127.0.1.1 → set `GLOO_SOCKET_IFNAME=eth0`. |
| `oom_kill event` at iter 0 (host RAM) | optimizer/activation offload exceeds 2 TB host → shard more (DP↑/PP↑) or drop activation offload. |
| `Inf in bucket #0` ~iter 2 | skipped the chat-init graft (Super-Base has 1188 zero embedding rows). |
| Mamba NaN on long 32k docs | `ISAMBARD_FP32_SSM_STATE=checkpoint` (default in the launcher). |
| `CUDA out of memory` at iter 0 with CP=4/DP=4 | CP<8 doubles tokens/rank (→8192) and the MoE dispatch transient past 268 GB at no-recompute. Keep CP=8, or re-enable recompute. |

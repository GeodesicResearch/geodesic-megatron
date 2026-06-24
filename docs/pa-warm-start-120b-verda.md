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

## 4. Results vs `6cfuh1ky`

_(Filled in after the 474-iter run completes — overlay of lm loss over iters 301–474, final loss,
grad-norm, and throughput; see `tmp/compare_runs.py`.)_ Reference: final lm loss ≈ 0.49, grad-norm
~0.14, 28.24 s/iter on 88 GH200.

## Pitfalls (Verda-specific)

| Symptom | Cause / Fix |
|---|---|
| `srun --container-image` "task_prolog can not be executed" | pyxis prolog broken → use `enroot create`+`start` (the launcher does). |
| `MASTER_ADDR: parameter null or not set` | enroot doesn't inherit host env → pass with `enroot start -e MASTER_ADDR=… -e SLURM_NODEID=…`. |
| Gloo `connectFullMesh … remote=[127.0.1.1]` | `/etc/hosts` maps hostname→127.0.1.1 → set `GLOO_SOCKET_IFNAME=eth0`. |
| `oom_kill event` at iter 0 (host RAM) | optimizer/activation offload exceeds 2 TB host → shard more (DP↑/PP↑) or drop activation offload. |
| `Inf in bucket #0` ~iter 2 | skipped the chat-init graft (Super-Base has 1188 zero embedding rows). |
| Mamba NaN on long 32k docs | `ISAMBARD_FP32_SSM_STATE=checkpoint` (default in the launcher). |

# Container Pipeline (INFR-68)

Apptainer containers are the **default execution environment** for every pipeline in this
repo (training, data, checkpoint, coherence, env-validate). The container replaces the
fragile 12-workaround bare-metal venv build as the primary path: torch, CUDA, cuDNN,
Transformer Engine, the Mamba kernels, and `ft_launcher` come prebuilt and version-matched
from an NVIDIA NGC NeMo image (aarch64), while this repo's code is bind-mounted so the
checkout you submit from is exactly the code that runs.

- **Opt-out:** `GEODESIC_CONTAINER=0` runs the legacy bare-metal venv
  (`pipeline_env_activate.sh` / `pipeline_env_setup.sh`). It is required today only for
  the coherence pipeline's `--backend vllm` (the pinned vLLM 0.22.1+cu129/Ray stack lives
  in the venv) and is otherwise a transition-safety escape hatch. Removing bare-metal
  entirely is the end state, tracked as a follow-up once a vLLM-capable image qualifies.
- **User-facing commands are unchanged.** `isambard_sbatch --nodes=11
  pipeline_training_submit.sbatch <config> super sft` works exactly as before — the
  launcher decides bare-metal vs container from `GEODESIC_CONTAINER`.

## One-time setup

```bash
# 1. Pull the qualified NGC image to a SIF on shared storage (CPU-only, ~25 GB,
#    login node OK). Image tag + all paths live in pipeline_container_config.env.
bash pipeline_container_pull.sh

# 2. Build the Slingshot NCCL stack inside the image (GPU node, ~20 min, one-time
#    per image tag). Or: isambard_sbatch pipeline_container_submit.sbatch build-ofi
bash pipeline_container_build_ofi.sh

# 3. Validate (1 GPU node)
isambard_sbatch pipeline_env_submit.sbatch container-validate
```

That's the whole install. Compare with `docs/ultra-550b-training-and-conversion.md` §1
(the INFR-41 from-scratch bare-metal build that surfaced 7 latent defects) for why this
exists.

## Files

| File | Responsibility |
|------|----------------|
| `pipeline_container_config.env` | THE config: image tag/URI, SIF path, Slingshot build dir, bind list, Apptainer cache dirs (+ $HOME guards). Every knob is `${VAR:-default}` overridable via `GEODESIC_CONTAINER_*`. |
| `pipeline_container_exec.sh` | Shim that runs one command string inside the container. Validates the SIF + Slingshot build (hard fail with fix commands), scrubs venv-shaped host env, `exec apptainer exec --nv`. |
| `pipeline_container_activate.sh` | Sourced INSIDE the container: PYTHONPATH (repo wins over image installs), `NCCL_NET_PLUGIN` + `LD_LIBRARY_PATH` for the Slingshot stack, universal GPU settings (mirrored from `pipeline_env_activate.sh`), cache paths. |
| `pipeline_container_pull.sh` | NGC → SIF acquisition + `${SIF}.source.txt` provenance + quota preflight. |
| `pipeline_container_build_ofi.sh` | One-time Option B Slingshot build (NCCL + hwloc + aws-ofi-nccl + nccl-tests) inside the image; writes `provenance.txt`. |
| `pipeline_container_submit.sbatch` | Thin SLURM wrapper for `pull` / `build-ofi`. |

## Design decisions

### D1 — Slingshot networking: the official "Option B" in-image build

The NGC image bundles a newer CUDA userland (12.9/13.x) than the host driver natively
supports (12.7). For exactly this case, Isambard's NCCL-in-containers guidance prescribes
building NCCL and the aws-ofi-nccl CXI plugin **inside the container** against the image's
CUDA and the **host's** Cray libfabric, keeping the outputs on the host filesystem and
bind-mounting them at runtime. `pipeline_container_build_ofi.sh` is that recipe
(BriCS `build_nccl.sh` template), writing to
`/projects/a5k/public/containers/slingshot/<image-tag>/`, bound at `/opt/slingshot`.

Why not the alternatives:

- **`brics/apptainer-multi-node` + `/host/adapt.sh` (Option A)** injects the HOST NCCL
  2.26.6 and OpenMPI ahead of the image's libraries — correct only when the image's CUDA
  matches the host, which ours doesn't; it would also shadow the image's torch-matched
  NCCL.
- **Binding the host brics aws-ofi-nccl 1.8.1 plugin** (built vs host NCCL 2.26.6 + CUDA
  12.6) into a newer-CUDA image is unsupported territory. It remains available as a
  diagnostic A/B via `TRAIN_NCCL_NET_PLUGIN=<path>` — the same seam bare-metal uses.

Two traps this design guards against:

- **The image's own aws-ofi-nccl is EFA-targeted** (AWS fabric, not Slingshot) and
  registered via ld.so.conf. `pipeline_container_activate.sh` names the CXI plugin
  explicitly via `NCCL_NET_PLUGIN` and orders `/opt/slingshot` paths first, so the EFA
  plugin can never win.
- **Silent TCP fallback.** Without a working CXI plugin NCCL quietly drops to sockets:
  ~2.3 GB/s vs ~163 GB/s on a 2-node/8-GPU all_reduce (~70×). Validation gates on the
  NCCL log line `Using network AWS Libfabric` AND measured busbw
  (`scripts/container/nccl_allreduce_smoke.py`, floor 100 GB/s).

### D2 — Env transport: inherit, scrub the poison

Apptainer passes the host environment through by default, so every var the launchers
already export (30+ `NCCL_*`/`FI_CXI_*`, `TORCH_*`, `MASTER_*`, `ISAMBARD_*`, W&B/HF
paths) reaches the ranks unchanged. The shim only scrubs what the venv poisons —
`LD_PRELOAD`, `LD_LIBRARY_PATH`, `PYTHONPATH`, `VIRTUAL_ENV` (the repo lives under the
bind-mounted `$HOME`, so venv paths *resolve* inside the container and would shadow image
libraries) — and sets `PYTHONNOUSERSITE=1` against `~/.local` leakage. `$HOME` stays
bound: W&B reads `~/.netrc`.

### D3 — Import resolution: the repo wins

`src/megatron/` and `3rdparty/Megatron-LM/megatron/` are PEP 420 namespace portions, so
`sys.path` order decides which portion serves `megatron.bridge`/`megatron.core`.
`pipeline_container_activate.sh` prepends both to PYTHONPATH; PYTHONPATH precedes the
image's site-packages, so the checkout wins. (Bare-metal gets `megatron.core` from the
venv's editable install; in-container the explicit `3rdparty` prepend replaces that.)

Caveat: a **regular** (non-namespace, `__init__.py`-bearing) `megatron` package in a
future image would defeat every namespace portion regardless of path order.
`pipeline_env_validate.py --container` asserts the actual resolution on every run;
contingency if an image ever ships one: a derived SIF whose `%post` runs
`pip uninstall -y megatron-core megatron-bridge nemo-toolkit`.

### D4 — Defaults and hard failures

`GEODESIC_CONTAINER` defaults to **1**. In container mode a missing SIF, Slingshot build,
or host libfabric dir fails loudly with the exact fix command — there is deliberately no
silent fall-back to bare-metal (an env-selection surprise is precisely the class of bug
containers exist to kill).

### D5 — vLLM coherence backend stays bare-metal (for now)

`pipeline_coherence_submit.sbatch --backend vllm` hard-fails under container mode with
"set GEODESIC_CONTAINER=0". The `hf` and `megatron` backends run containerized.

### D6 — ft_launcher version gate

The image's `nvidia-resiliency-ext` may be older than the venv's 0.5.0. Before an FT
launch, the training launcher greps `ft_launcher --help` inside the container for
`--ft-rank-section-timeouts`; if absent it exits with "rerun with --disable-ft or qualify
a newer image" instead of letting 44+ ranks die on a usage error.

### D7 — Image qualification: newest-first ladder

Prefer the latest stack that works. Candidates are qualified newest-first
(`container-validate` per image via `GEODESIC_CONTAINER_IMAGE_TAG=<tag>` +
`GEODESIC_CONTAINER_SIF=<path>` overrides), and the newest tag that clears validation +
the Nano multi-node smoke + the 120B quickstart becomes the committed default in
`pipeline_container_config.env`. Qualification evidence (per-image version tables, busbw,
iteration times) lives in the INFR-68 PR. Same policy for the Option B build pins
(`GEODESIC_CONTAINER_OFI_*_VERSION`).

An image tag qualifies when:

1. `container-validate` is all-green (imports, GPU op, import-path resolution, CXI plugin
   CDLL, ft flags, dataset-helpers JIT, `nvidia-smi` shows the image's CUDA — proving
   `--nv` forward-compat injection on the R565/12.7 driver);
2. the 2-node NCCL smoke shows `Using network AWS Libfabric` and busbw ≥ 100 GB/s;
3. the Nano quickstart trains multi-node (loss decreasing, no NaN), `--disable-ft` and
   with FT;
4. the Super-120B quickstart (`configs/quickstart/nemotron_super_quickstart_sft.yaml`)
   holds < 40 s/iter (mean of iters 10–30) vs the bare-metal A/B on the same nodes.

## Reproducibility / provenance

- `pipeline_container_pull.sh` writes `${SIF}.source.txt` (URI, date, `apptainer
  inspect`).
- `pipeline_container_build_ofi.sh` writes `<slingshot-dir>/provenance.txt` (component
  versions, image digest, builder).
- The training launcher echoes both into every job log, so any run's exact container
  stack is recoverable from its output alone.

## Interactive use

```bash
# a shell inside the pipeline container with repo + Slingshot env wired:
./pipeline_container_exec.sh "cd $PWD; source pipeline_container_activate.sh; exec bash -i"
```

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `FATAL [container-config]: SIF not found` | Run `bash pipeline_container_pull.sh`. |
| `FATAL [container-config]: Slingshot NCCL stack not built` | Run `bash pipeline_container_build_ofi.sh` (GPU node). |
| NCCL log shows `NET/Socket` or bandwidth ~2 GB/s | CXI plugin not loading. `NCCL_DEBUG=INFO` and look for `AWS Libfabric`; `ctypes.CDLL($NCCL_NET_PLUGIN)` inside the container names the missing soname (usually a missing `/host/usr/lib64` bind or a plugin built against the wrong libfabric). |
| `megatron.bridge` imports from the image, not the repo | The image ships a regular `megatron` package — see D3 contingency (derived SIF). `container-validate` catches this. |
| `ft_launcher` rejects `--ft-rank-*` flags | Image NVRX too old — `--disable-ft` or qualify a newer image (D6). |
| Apptainer fills `$HOME` | Never point `APPTAINER_CACHEDIR`/`APPTAINER_TMPDIR` at `$HOME`; the config refuses to run if you do. |
| Host libfabric path missing after a cluster upgrade | Override `GEODESIC_CONTAINER_HOST_LIBFABRIC` (check `ls -d /opt/cray/libfabric/*`) and rebuild the Slingshot stack. |

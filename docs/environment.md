# Environment

Every pipeline in this repo — training, data prep, checkpoint conversion, coherence,
validation — runs inside one Apptainer container built from an NVIDIA NGC NeMo image
(aarch64). The image supplies torch, CUDA, cuDNN, NCCL, Transformer Engine, the
Mamba/causal-conv1d kernels, grouped-GEMM, APEX and `ft_launcher` as a single
version-matched set; this repo's `src/` and the pinned `3rdparty/Megatron-LM` are
bind-mounted, so **the checkout you submit from is the code that runs**. There is no
second environment and no mode flag: a missing image or Slingshot build stops the job
with the command that fixes it rather than degrading silently.

Submit commands are the ordinary ones:

```bash
isambard_sbatch --nodes=16 pipeline_training_submit.sbatch \
  configs/quickstart/nemotron_super_quickstart_sft.yaml super sft
```

Each launcher sources `pipeline_env_config.env` **on the host**, calls
`env_config_require`, then `srun`s exactly one `pipeline_env_exec.sh` per node; inside the
container the payload sources `pipeline_env_activate.sh` and runs
`ft_launcher`/`torchrun`/`python`. The host side only orchestrates — no `module load`, no
venv, nothing to activate before submitting.

## Install — one command

Run this once per image tag, on a GPU node (compute node or tunnel). It is shared across
users via `/projects/a5k/public/containers/`, so in practice it is once per cluster.

```bash
bash pipeline_env_setup.sh
# or via SLURM:
isambard_sbatch pipeline_env_submit.sbatch setup
```

Four idempotent steps, each announcing its skip explicitly (never silently):

| Step | What it does | Skipped when | Needs GPU |
|---|---|---|---|
| 1 `sif` | `apptainer pull` the NGC image to `$CONTAINER_SIF` (~19 GB; the pull needs ~2× that transiently, so it prints the `/projects/a5k` project quota first) and write `${SIF}.source.txt` | the SIF exists | no |
| 2 `slingshot` | build NCCL + hwloc + aws-ofi-nccl + nccl-tests **inside the image** (the official "Option B" recipe, ~20 min, ~420 MB output) and create the `hostlibs` symlink dir | `hostlibs/libcxi.so.1` exists | **yes** (the build queries the CUDA toolchain) |
| 3 `overlay` | `pip install --no-deps --target` the Python overlay packages | `provenance.txt` records exactly the configured package list | no |
| 4 `validate` | run `pipeline_env_validate.py` in-container, exactly as a pipeline would | never (that is the point) | **yes** |

On a login node the same command runs steps 1 and 3 and then prints how to finish 2 and 4
on a GPU node. Flags: `--force` (redo everything, including a re-pull) and
`--only sif|slingshot|overlay|validate`. **Everything configurable** — image tag/URI, SIF
path, output dirs, Slingshot component versions, overlay package list, bind list, Apptainer
cache dirs — lives in `pipeline_env_config.env` and is overridable with the
`GEODESIC_CONTAINER_*` env vars documented inline there. There are deliberately no other
CLI flags.

Where the artifacts land — **all on project storage, never `$HOME`** (a ~19 GB SIF blows
the home quota instantly, so the config *refuses to run* if `APPTAINER_CACHEDIR` or
`APPTAINER_TMPDIR` resolves under `$HOME`):

| Artifact | Path |
|---|---|
| SIF + provenance | `/projects/a5k/public/containers/nemo_<tag>.sif{,.source.txt}` |
| Slingshot build | `/projects/a5k/public/containers/slingshot/nemo_<tag>/` (bound at `/opt/slingshot`) |
| Python overlay | `/projects/a5k/public/containers/overlay/nemo_<tag>/` |
| Apptainer cache/tmp | `/projects/a5k/public/apptainer_cache_$USER`, `/projects/a5k/public/tmp/apptainer_$USER` |

### Validate

```bash
isambard_sbatch pipeline_env_submit.sbatch validate [--run-training]
```

20 checks (21 with `--run-training`, which adds a 5-iteration single-GPU mock-data training
job): core imports, the CUDA-extension imports (TE, mamba-ssm, causal-conv1d,
grouped-GEMM — the `moe_experts_impl: cublas_grouped` dependency, built into
the overlay on images that lack it), CUDA availability, a bf16 GPU matmul, two recipe loads, then the
environment-integrity block — import paths resolve to *this* checkout, the CXI NCCL plugin
`CDLL`s cleanly, `ft_launcher` accepts the section-timeout flags, the Megatron dataset
helpers JIT-build, and a **version report** of the actual in-image stack. The integrity
checks are the ones that catch a bad image swap or a half-finished install, so they always
run.

## Files

| File | Responsibility |
|------|----------------|
| `pipeline_env_config.env` | THE config (image tag/URI, SIF path, Slingshot dir, overlay dir + package list, bind list, cache dirs + `$HOME` guards) and `env_config_require`, the run-time gate every launcher calls. |
| `pipeline_env_exec.sh` | The shim: `pipeline_env_exec.sh "<one command string>"`. Gates on `env_config_require`, scrubs host toolchain/venv-shaped env, `exec apptainer exec --nv --bind "$CONTAINER_BINDS"`. |
| `pipeline_env_activate.sh` | Sourced **inside** the container: import resolution, CUDA forward-compat, Slingshot `LD_LIBRARY_PATH` ordering + `NCCL_NET_PLUGIN`, universal GPU settings, cache paths. Refuses to run outside a container. |
| `pipeline_env_setup.sh` | The whole install in one command (the four steps above). |
| `pipeline_env_submit.sbatch` | SLURM wrapper; modes `setup`, `validate`, `smoke`. |
| `pipeline_env_validate.py` | The validator (runs in-container). |

## Everyday use

```bash
# Interactive shell with the repo + Slingshot env wired up:
./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh || exit 1; exec bash -i"

# Unit tests — in-container is the only way (~5,450 tests collected in ~35 s).
# NOTE the scratch cwd: an autouse conftest fixture asserts ./nemo_experiments does
# not exist, so running from the repo root errors every test (and would rmtree a real one).
# -n 8 --dist loadfile uses the image's bundled pytest-xdist (~100 s vs ~5-6 min serial);
# per-worker MASTER_PORT isolation lives in tests/unit_tests/conftest.py.
./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh || exit 1; T=\$(mktemp -d); cd \$T; \
  python -m pytest $PWD/tests/unit_tests/ -x -q -m 'not pleasefixme' -n 8 --dist loadfile"

# Fabric health: asserts busbw clears the 100 GB/s floor (the script's own gate).
# To ALSO confirm the plugin by name, rerun with NCCL_DEBUG=INFO and grep for
# 'Using network AWS Libfabric'. Worth running after any image or Slingshot
# rebuild — a CXI plugin that fails to load costs ~70x on comms and says nothing.
isambard_sbatch --nodes=2 pipeline_env_submit.sbatch smoke
```

The image ships pytest/ruff/pre-commit, so the test command needs nothing installed;
`-m 'not pleasefixme'` skips the documented quarantine of known-broken tests (the same
exclusion CI uses). The overlay carries `imageio` specifically so that command works at all:
without it one diffusion test file fails at **collection**, which fails the entire run.

## What is inside the image

The qualified image is `nvcr.io/nvidia/nemo:26.04` (aarch64, re-qualified 2026-07-29 —
CUDA 13.1, torch 2.11.0a0+nv26.02, TE 2.14.1, NCCL 2.29.2, nvidia-resiliency-ext 0.6.0;
validator green — 18/18 at qualification, 20 checks today (the grouped_gemm and
OpenMP-defaults checks were added since); quickstart 25.66 s/iter at qualification with
`optimizer_offload_fraction: 0.5` vs 26.70 on the prior tag, identical nodelist (current
champion **17.099** with `moe_experts_impl: torch_grouped` and optimizer CPU offload OFF;
the `cublas_grouped` backend it replaced measured 20.66 at the same offload-off posture,
and the offload-0.5 posture before that 21.78 on the 26.02 tag / 22.01 on 26.04) — see
`docs/investigations/120b-gbs64-host-overhead-investigation.md` §9.8), pulled to
`/projects/a5k/public/containers/nemo_26.04.sif`. The table below records the PREVIOUS
qualified image `26.02.nemotron_3_super`'s measured contents (2026-07-25); the validator's
version report prints the live set per image:

| Component | Version | Note |
|---|---|---|
| Python | 3.12.3 | |
| CUDA / cuDNN | 13.0 / 9.15.0.58 | newer than the host driver — see D6b |
| NCCL (image) | 2.28.8 | torch-matched; the Option-B build layers 2.29.2-1 in front |
| torch | 2.10.0a0+b558c986e8.nv25.11 | |
| transformer-engine | 2.12.0 | |
| mamba-ssm / causal-conv1d | 2.3.0 / 1.6.0 | prebuilt in the image — the Nemotron-H hybrid needs both |
| nv-grouped-gemm | 1.1.4.post8 | MoE grouped GEMM |
| apex | 0.1 | why `model.gradient_accumulation_fusion: True` works and is the faster path (measured ~1.1 s/iter on the 120B quickstart) |
| nvidia-resiliency-ext | 0.4.1 | `ft_launcher`; gated at launch, see D6 |
| transformers / datasets / numpy | 5.3.0 / 3.1.0 / 1.26.4 | |
| peft | 0.13.2 → **0.18.1** via the overlay | see D3b |

**Do not treat that table as authoritative** — the versions are the image's, fixed by
`CONTAINER_IMAGE_TAG` rather than by a lockfile. The live source is the validator's
`version report` stage (printed on every `validate`, and worth re-reading after any image
bump) plus `apptainer inspect $CONTAINER_SIF` / `${CONTAINER_SIF}.source.txt`.

## How it works

### D1 — Slingshot networking: the official "Option B" in-image build

The image bundles a newer CUDA userland than the host driver natively supports. For exactly
that case Isambard's containers/NCCL guidance prescribes building NCCL and the aws-ofi-nccl
CXI plugin **inside the image**, against the image's CUDA and the **host's** Cray libfabric
(1.22.0), keeping the outputs on the host filesystem and bind-mounting them at runtime.
Step 2 of `pipeline_env_setup.sh` is that recipe; it builds NCCL `v2.29.2-1`, hwloc `v2.13`
and aws-ofi-nccl `v1.18.0` (pins overridable via `GEODESIC_CONTAINER_OFI_*_VERSION`) plus
the `nccl-tests` binaries, into `/projects/a5k/public/containers/slingshot/nemo_<tag>/`,
bound at `/opt/slingshot` — the in-container path the official recipe uses, so its scripts
and ld ordering work unmodified.

**Never `brics/apptainer-multi-node` / `/host/adapt.sh`.** That path injects the *host* NCCL
and OpenMPI ahead of the image's libraries. It is correct only when image CUDA matches the
host, which it does not here, and it shadows the image's torch-matched NCCL. Binding the
host `brics/aws-ofi-nccl` plugin (built against host NCCL + CUDA 12.6) into a newer-CUDA
image is likewise unsupported; it survives only as a diagnostic A/B seam,
`TRAIN_NCCL_NET_PLUGIN=/path/to/libnccl-net.so`, which `pipeline_env_activate.sh` consumes
inside the container (setting `NCCL_NET_PLUGIN` on the host is pointless — the activate
script re-derives it).

Two traps this guards against:

- **The image's own aws-ofi-nccl is EFA-targeted** (AWS fabric, not Slingshot) and is
  registered in `ld.so.conf`. `pipeline_env_activate.sh` names the CXI plugin explicitly via
  `NCCL_NET_PLUGIN` and orders `/opt/slingshot` first, so the EFA plugin can never win.
- **Silent TCP fallback.** Without a working CXI plugin, NCCL quietly drops to sockets:
  **~2.3 GB/s vs ~163 GB/s** on a 2-node/8-GPU all_reduce (~70×). Nothing crashes; the run
  just becomes worthless. Hence the smoke test's dual assertion (`Using network AWS
  Libfabric` **and** measured busbw ≥ 100 GB/s — a floor between the two regimes, so the
  verdict is unambiguous). This stack measured **131 GB/s** on that shape.

### D2 — Env transport: inherit everything, scrub the poison

Apptainer passes the host environment through by default, so every variable the launchers
already export (30+ `NCCL_*`/`FI_CXI_*`, `TORCH_*`, `MASTER_*`, `SLURM_*`, `ISAMBARD_*`,
W&B/HF paths) reaches the ranks unchanged — launchers did not have to learn a new transport.

The shim scrubs only what the host env *poisons*, and the reason is that the repo lives
under the bind-mounted `$HOME`, so host paths **resolve** inside the container:
`LD_PRELOAD`, `LD_LIBRARY_PATH`, `PYTHONPATH`, `VIRTUAL_ENV`, `NCCL_LIBRARY` (would shadow
image libraries) and the host toolchain vars `CC`, `CXX`, `CUDAHOSTCXX`, `CUDA_HOME`,
`CPLUS_INCLUDE_PATH`, `C_INCLUDE_PATH`, `CUDNN_PATH` — interactive Isambard shells export
gcc-12/HPC-SDK paths, which was observed hijacking the in-image NCCL build (a `make`
invoking a `/usr/bin/g++-12` that does not exist in the image). `PYTHONNOUSERSITE=1` blocks
`~/.local` leakage. `$HOME` stays bound because W&B reads `~/.netrc`. The setup script's
Slingshot build scrubs the identical set, for the identical reason.

`exec` replaces the shim's shell, so SLURM step termination and `ft_launcher` restarts reach
the containerized process tree directly instead of dying on a wrapper.

The bind list (`CONTAINER_BINDS`) is short and every entry earns its place: `/projects` **and
`/lus`** (on Isambard `/projects/a5k/public` is a symlink into `/lus`, so binding `/projects`
alone leaves every real path dangling inside the container with a `FileNotFoundError`), the
host Cray libfabric at `/host/opt/cray/libfabric/<ver>`, the host `/usr/lib64` at
`/host/usr/lib64` (read-only, for `libcxi`/`libnl` — see D5 for why it never reaches
`LD_LIBRARY_PATH`), and the Option-B build at `/opt/slingshot`. The in-container paths mirror
the official BriCS recipe exactly, so its build scripts and ld ordering work unmodified.

### D3 — Import resolution: the repo wins

`src/megatron/` and `3rdparty/Megatron-LM/megatron/` are PEP 420 namespace portions, so
`sys.path` order decides which portion serves `megatron.bridge` / `megatron.core`.
`pipeline_env_activate.sh` prepends both to `PYTHONPATH`, and `PYTHONPATH` precedes the
image's site-packages, so the checkout wins over any megatron packages the image installs.
Resolution order overall: **repo `src/` + `3rdparty` > overlay > image site-packages.**

Caveat and contingencies: a **regular** (non-namespace, `__init__.py`-bearing) `megatron`
package in a future image would defeat every namespace portion regardless of path order.
`pipeline_env_validate.py` asserts the actual resolution on every run — that assert is the
tripwire. If an image ever ships one, the fix is a derived SIF whose `%post` runs
`pip uninstall -y megatron-core megatron-bridge nemo-toolkit`.

### D3b — Python overlay: image packages too old for the repo

The SIF is read-only and never modified, but an image occasionally ships a dependency at a
version this repo cannot use. The fix is a `pip install --target` directory layered onto
`PYTHONPATH` *after* the repo prepends (so the repo still wins, but the overlay shadows
site-packages), configured as `CONTAINER_PYTHON_OVERLAY` and populated by setup step 3. It
lives under `/projects` (already bound) so it needs no extra bind, and it is `export`ed so
the in-container activate script inherits it through Apptainer's env passthrough.

`CONTAINER_OVERLAY_PACKAGES` currently carries three packages, each for a stated reason:

- **`peft==0.18.1`** — the image ships 0.13.2; the bridge recipes import `modelopt`, which
  hard-requires `peft>=0.17.0` (recipe-load stages failed on exactly this).
- **`imageio==2.37.0`** — absent from the image; without it one diffusion test file fails at
  collection and takes the whole in-container unit-test run with it.
- **`nv-grouped-gemm==1.1.4.post8`** — absent from 26.04;
  `moe_experts_impl: cublas_grouped` imports `grouped_gemm` at model build. That backend is
  no longer the shipped default (`torch_grouped` is, and it needs nothing beyond torch), but
  it stays installable so the A/B that chose the default remains runnable. PyPI has no
  aarch64 wheel, so the overlay builds it from sdist — which is why the overlay pip line
  passes `--no-build-isolation`: an isolated build env would pip-install its own torch
  instead of compiling against the image's CUDA-matched one. The validator's grouped_gemm
  check gates on the import so a half-failed build surfaces at validate time.

**`--no-deps` is deliberate.** Anything in the overlay shadows the image's copy, so pulling
a dependency closure risks shadowing the image's CUDA-matched torch with a PyPI one. peft's
runtime deps are all present and newer in the image. If a future package genuinely needs a
dep the image lacks, add that one wheel explicitly — still `--no-deps`, never the closure.
The overlay records `provenance.txt` (package list, SIF, who/when), which is also the
idempotency key: change `CONTAINER_OVERLAY_PACKAGES` and the next setup run rebuilds it. A
configured-but-missing overlay is a **loud warning** from the activate script, never a
silent skip, so a stale overlay cannot quietly fall back to the too-old image package.

### D4 — Hard failures, no fallbacks

`env_config_require` runs before every job and every `apptainer exec`, and fails with the
exact fix command in three cases:

| Failure | Meaning | Fix |
|---|---|---|
| `SIF not found` | image never pulled here | `bash pipeline_env_setup.sh` |
| `Slingshot NCCL stack not built` | no `aws-ofi-nccl/lib/libnccl-net.so` | `bash pipeline_env_setup.sh` (GPU node) |
| `Slingshot build predates the hostlibs dir` | build from before D5 — it passes the plugin check but the plugin cannot `dlopen` libcxi at run time | `bash pipeline_env_setup.sh --force` |

(A fourth check covers a missing host libfabric dir, a cluster constant that only moves on a
cluster upgrade — override `GEODESIC_CONTAINER_HOST_LIBFABRIC` and rebuild.) The
`hostlibs` case exists because the *symptom* of skipping it is a run-time dlopen failure far
from the cause; the check turns it into one line at launch. Note the host-side test is `-L`
before `-e`: the symlink targets `/host/...`, which resolves only inside the container, so
`-e` alone would false-negative on a perfectly good build.

### D5 — `hostlibs`: a symlink-only dir, because the host `/usr/lib64` poisons the linker

The CXI plugin needs three sonames the image lacks (`libcxi.so.1`, `libnl-3.so.200`,
`libnl-route-3.so.200`). The obvious move — put `/host/usr/lib64` on `LD_LIBRARY_PATH` — is
the wrong one: torch inductor's C++ codegen converts `LD_LIBRARY_PATH` entries into `-L`
link dirs, and the host SUSE `libc.so` is a GNU-ld script whose `GROUP()` names absolute
`/lib64/libc.so.6` + `/usr/lib64/libc_nonshared.a` — paths that do not exist in the image.
Any fresh `torch.compile` link then dies with `ld: cannot find /lib64/libc.so.6`. Ordering
image libdirs first does **not** save you (the image keeps no plain `libc.so` dev script in
`/usr/lib/aarch64-linux-gnu`, so `-lc` falls through to the host dir anyway), and warm
inductor caches mask the bug entirely — a cache hit skips the link, which is why single-shot
probes "passed".

So setup step 2 creates `<slingshot-dir>/hostlibs/` containing **only** symlinks to those
three sonames, and that dir — never the raw host dir — goes last on the container's
`LD_LIBRARY_PATH`. Runtime `dlopen` finds what it needs; the linker finds no `libc` there to
be poisoned by. The full ordering (mirroring the BriCS recipe): host libfabric → built NCCL
→ aws-ofi-nccl → image dirs → `hostlibs`.

### D6 — ft_launcher version gate

The image's `nvidia-resiliency-ext` is whatever NGC shipped (0.4.1 on the current tag).
Before an FT launch, `pipeline_training_launch.sh` greps `ft_launcher --help` *inside the
container* for `--ft-rank-section-timeouts`; if absent it exits with "rerun with
`--disable-ft`, or qualify a newer image" instead of letting 44+ ranks die on a usage error.
The validator checks the same two flags. Related image-version friction already handled: the
image's `ft_launcher` parses heartbeat timeouts as FLOATS and rejects the literal `none` the
launcher used to pass. They are therefore passed as explicit large values
(`--ft-initial-rank-heartbeat-timeout=7200 --ft-rank-heartbeat-timeout=7200`), NOT omitted:
heartbeats are an independent mechanism from the section timeouts, and
nvidia-resiliency-ext defaults to 3600 s initial / 2700 s subsequent — shorter than
Ultra-550B's documented 45–75 min first iteration at PP=36, which would trip the heartbeat,
SIGKILL the workers, and restart straight back into the same slow first iteration.

### D6b — CUDA forward-compat: front the image's compat libs yourself

Under Docker, NGC's entrypoint detects a host driver older than the image CUDA and symlinks
`/usr/local/cuda/compat/lib -> lib.real` so the forward-compat `libcuda` wins. **Apptainer
never runs that entrypoint and the SIF is read-only**, so `--nv` alone leaves the host's
CUDA 12.7 `libcuda` in charge and CUDA-13 torch dies with "driver too old".
`pipeline_env_activate.sh` therefore fronts the compat dir on `LD_LIBRARY_PATH`
(`GEODESIC_CONTAINER_CUDA_COMPAT=auto|0|/path`; `auto` probes the two known NGC layouts).
Always-fronting is safe here because the Isambard driver is always older than any image CUDA
we qualify — the one case NGC's entrypoint would skip compat (driver *newer* than image)
cannot occur.

Measured on driver R565.57.01 — this is a per-image qualification axis, not a settled fact:

| Image CUDA | Verdict on R565 |
|---|---|
| 12.9 | works via same-major minor-version compatibility (no compat shim needed) |
| 13.0 | **works** via compat libs (verified: torch cu13.0 + GH200 matmul green) |
| 13.2 | **compat rejects the driver** (`Error 803: unsupported display driver / cuda driver combination`) |

### D7 — Universal GPU and cache settings

`pipeline_env_activate.sh` sets these for every payload, container-wide:

| Variable | Why |
|---|---|
| `UB_SKIPMC=1` | the Isambard driver lacks CUDA Multicast; Userbuffers init hangs without it |
| `CUDA_DEVICE_MAX_CONNECTIONS=1` | required for TP/SP comm-compute overlap. Overridable via `ISAMBARD_CUDA_MAX_CONNECTIONS` — TP=1 topologies may probe >1 to unserialize the hardware launch queues |
| `OMP_NUM_THREADS=8` | torchrun sets this to **1** whenever it is absent, single-threading host-side AdamW for any CPU-offloaded optimizer onto one Neoverse-V2 core (~36 GB/s of a ~500 GB/s socket). Overridable via `ISAMBARD_OMP_THREADS`; `=1` restores torchrun's behaviour. On the 120B benchmark, both arms on the pre-`torch_grouped` expert path: **21.36 s/iter / 73.70 GB at offload 1.0 with 8 threads vs 22.79 / 76.78 at offload 0.5 single-threaded** — strictly dominates the previous champion, but the arms differ in offload fraction too, so the delta is not threading alone (§C1b; the clean offload-1.0 single-thread arm was never run). Exactly neutral with offload off |
| `OMP_WAIT_POLICY=PASSIVE` | set automatically whenever `OMP_NUM_THREADS` > 1 (override with `ISAMBARD_OMP_WAIT_POLICY`). Not decoration: GNU OpenMP idle threads spin-wait, and these workloads are host-launch-bound, so ACTIVE spin can cost more launch throughput than the threaded Adam saves |
| `NVTE_CPU_OFFLOAD_V1=1` | TE fine-grained CPU activation-offload path (TE ≥ 2.10) |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | reduces allocator fragmentation |
| `TORCH_CUDA_ARCH_LIST=9.0` | GH200; also guards `sm_90a` arch-string parsing in JIT builds |
| `CUDA_HOME=/usr/local/cuda` | the **image's** toolkit for JIT builds (Triton, TE, dataset helpers) — deliberately not the host HPC-SDK path the shim scrubs |

Caches: `HF_HOME=/projects/a5k/public/hf`, `NEMO_HOME`, `WANDB_DIR`, `TMPDIR` are shared as
before, but **`HF_DATASETS_CACHE` is scoped to the container**
(`/projects/a5k/public/hf/datasets_container`). A processed-dataset cache shared with a
different major `datasets` version fails with `DatasetInfo.from_directory ... must be called
with a dataclass`; Hub downloads (models/tokenizers under `HF_HOME`) stay shared, since
those are version-neutral.

### Reproducibility / provenance

- Setup step 1 writes `${SIF}.source.txt` — image URI, digest labels, pull date, `apptainer
  inspect` output.
- Setup step 2 writes `<slingshot-dir>/provenance.txt` — component versions, SIF, builder,
  build host, host libfabric.
- Setup step 3 writes `<overlay>/provenance.txt` — package list and why.
- `pipeline_training_launch.sh` echoes the first two into **every job log**, so any run's
  exact stack is recoverable from its output alone. Combined with the run identity below,
  that closes the loop from a W&B run to the container it ran in.

## Image qualification

Policy: **prefer the newest stack that works.** Candidate tags are tried newest-first
(`GEODESIC_CONTAINER_IMAGE_TAG=<tag>` plus `GEODESIC_CONTAINER_SIF=<path>` keeps candidates
side by side), and the newest tag that clears all four gates becomes the one-line committed
default in `pipeline_env_config.env`. The original INFR-68 ladder
(`26.06 → 26.02.nemotron_3_super → 25.11 → 25.09`) stopped at `26.02.nemotron_3_super`
because `26.06` ships CUDA 13.2 (nvvm 13.2.78), which the compat table in D6b rules out on
this driver. **2026-07-29 re-qualification: `26.04` (CUDA 13.1, compat 590.48.01) does run
on this driver and is now the default** — validator 18/18, FT smoke, and a 48-iter ladder
on an identical nodelist (evidence:
`docs/investigations/120b-gbs64-host-overhead-investigation.md` §9.8; plain-config 26.04
regresses ~1–2 s via end-of-step skew, and the adopted `optimizer_offload_fraction: 0.5`
config wins outright at 25.66 vs 26.70). `26.06` remains driver-blocked; per-image evidence
otherwise lives in the INFR-68 PR. The same newest-first policy applies to the Option-B
build pins.

A tag qualifies when:

1. **`validate` is all-green** — imports, GPU op, import-path resolution, CXI plugin `CDLL`,
   ft flags, dataset-helpers JIT, and `nvidia-smi` showing the image's CUDA (which is itself
   the proof that `--nv` + compat injection works on the R565/12.7 driver).
2. **2-node NCCL smoke** shows `Using network AWS Libfabric` and busbw ≥ 100 GB/s.
3. **The Nano quickstart trains multi-node** — loss decreasing, no NaN — both with FT and
   with `--disable-ft`.
4. **The Super-120B benchmark holds its iteration time.**
   `configs/quickstart/nemotron_super_quickstart_sft.yaml` (TP1 · CP4 · EP4 · PP8 · ETP1 ·
   DP2 → 64 GPUs = 16 nodes, seq 32K, GBS 64), scored as the **mean of iterations 10–30**
   (past the JIT/comm-init-dominated first iters), must clear two bars: the **absolute gate
   of < 40 s/iter**, and **no regression against the previously qualified tag's recorded
   number** measured on the identical nodelist (same-nodes A/B — Dragonfly placement alone
   moves this workload by ~2.7 s/iter, so a cross-allocation comparison proves nothing).
   Record the new number in the qualification note so the next bump has a baseline.

Current default's numbers on that config: **17.099 s/iter** champion (FT off) =
**154.5 TFLOP/s/GPU** model-FLOPs / 181.4 hardware, 0 NaN — `torch_grouped` experts with
optimizer CPU offload OFF, held over a 100-iteration soak (no iter > 1.5× median, −0.37%
drift). Placement is still worth ~2%: the previous champion measured 20.66/20.81/21.14 on
three different 16-node placements inside one allocation, so a quoted number without its
nodelist is soft at that level.
Superseded predecessors on this config, newest first: 20.66 (`cublas_grouped` per-expert
loop, same offload-off posture, 81.4 GB peak on the heavy MoE stage), 21.78 (offload 0.5,
26.02 image), 25.66 (26.04 qualification, TEGroupedMLP), ~27.6 (the July
fusion/manual-GC/telemetry ladder). The 40 s/iter gate is set to
discriminate health from degradation, not to be tight: TCP-fallback NCCL or a broken
toolchain shows up as 1.5–70×, far above the line. For history, the last same-nodes A/B
against the (now retired) bare-metal venv measured bare-metal **37.07 s/iter** vs container
**31.27 s/iter** at the then-current config — the container was **15.7% faster** at identical
loss.

## Profiling a training run

Profiling is env-var driven and works on any launch — there is no separate profiling config.
`scripts/profiling/profiler_callback.py` (a bridge `Callback`, default OFF) captures full
optimizer steps with `with_stack=True` + `record_shapes=True`. The step-by-step walkthrough
is [docs/profiling-quickstart.md](profiling-quickstart.md); for how to read torch profiles
in general, see Quentin Anthony's tutorial:
<https://github.com/Quentin-Anthony/torch-profiling-tutorial>.

The 25-iteration capture of the champion 120B workload, at iterations 10 and 20, on ranks 0
(pipeline stage 0) and 9 (an interior MoE-heavy stage):

```bash
ISAMBARD_TORCH_PROFILE=1 ISAMBARD_TORCH_PROFILE_ITERS=10,20 ISAMBARD_TORCH_PROFILE_RANKS=0,9 \
  isambard_sbatch --nodes=16 pipeline_training_submit.sbatch \
  configs/quickstart/nemotron_super_quickstart_sft.yaml super sft \
  train.train_iters=25 checkpoint.load=null checkpoint.save=null \
  logger.wandb_save_dir=/projects/a5k/public/logs/wandb \
  logger.wandb_exp_name=nemotron_super_quickstart_sft_profile
```

Three of those overrides are load-bearing — one fails silently, one fails at startup:

- **`checkpoint.load=null`** — without it the run resumes whatever iteration already sits in
  the config's `checkpoint.load` dir (the quickstart shares one with the benchmark runs), so
  a repeat profiling run executes zero iterations and captures nothing. No error; you just
  get an empty profile directory.
- **`checkpoint.save=null`** — a profiling run has no output worth 200+ GB of writes.
- **`logger.wandb_save_dir=...`** — **mandatory whenever `checkpoint.save=null`**.
  `training/state.py` computes the W&B dir as `logger.wandb_save_dir or
  os.path.join(cfg.checkpoint.save, "wandb")`, so with `save` null and `wandb_save_dir`
  unset it evaluates `os.path.join(None, "wandb")` and the job dies at startup with a
  `TypeError` on the last rank.

Knobs: `ISAMBARD_TORCH_PROFILE=1` (or a path, to override the default output root
`/projects/a5k/public/profiles`), `ISAMBARD_TORCH_PROFILE_ITERS` (comma-separated, 1-based;
one trace file per rank per iteration), `ISAMBARD_TORCH_PROFILE_RANKS` (default `0`),
`ISAMBARD_TORCH_PROFILE_WAIT` (legacy single capture at iteration WAIT+2, used only when
`_ITERS` is unset).

Artifacts land in `<root>/<wandb-exp-name>/<run-id>/`: the per-rank Chrome traces
(`rank<R>.iter<N>.chrome_trace.json.gz`, open in Perfetto or `chrome://tracing`),
`provenance.txt` (commit, run id, raw-log path, world info), `config_snapshot.yaml` (the
override YAML verbatim), `resolved_config_snapshot.yaml` (the FULL merged config including
recipe defaults and CLI overrides — **this** is the authoritative reproduction source), and
`raw_log_snapshot.out`. Send the whole directory when sharing: traces are only interpretable
alongside the config and commit.

## Run identity

Every launch through `pipeline_training_launch.sh` mints `ISAMBARD_RUN_ID` =
`<launch-timestamp>-j<slurm-job-id>`, which joins the three places a run leaves artifacts:

- **Job log** — echoed in the launch banner; `logs/slurm/by-run-id/<run-id>.out` symlinks to
  the raw `train-<jobid>.out`.
- **Profiles** — names the per-launch output subdirectory and is recorded in
  `provenance.txt`.
- **W&B** — `RunIdentityCallback` (`scripts/telemetry/run_identity.py`, registered on every
  training run) stamps `run/isambard_run_id`, `run/raw_log_path`, `run/slurm_job_id` into the
  run summary.
- **Resolved config** — `<run-id>.resolved-config.yaml`, written on **every** run (not only
  profiled ones) beside that run's artifacts: into `checkpoint.save` if the run saves
  checkpoints, otherwise `logger.wandb_save_dir`.

### Why the resolved-config snapshot exists

The override YAML you launch with does not describe the run. Recipe defaults and Hydra
overrides exist only in the merged object, so a posture reached partly from the command
line — the 128-GPU benchmark is the 64-GPU quickstart plus `train.global_batch_size=256`,
and there is no separate config file for it — would otherwise be unreproducible from disk.
The snapshot is taken from the FINAL config, after the mode-specific setup that mutates it
post-merge, so it reflects what actually ran.

Written on rank 0. The two failure modes are deliberately **not** treated alike: an **I/O**
failure (full disk, unwritable directory) prints a warning and lets the run continue, because
provenance must not cost you a run; but a config naming **no artifact directory at all** —
neither `checkpoint.save` nor `logger.wandb_save_dir` — raises and stops the run before
iteration 1, because that is a configuration error, not bad luck. Such a config already died
anyway, later and less legibly, inside `state.py`.

The I/O swallow is safe only because the happy path is pinned by
`tests/unit_tests/test_run_identity.py` against a real `ConfigContainer` — the first version
of this code imported a helper from the wrong module, and without that test it would have
degraded silently to "no snapshot, ever".

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `FATAL [env-config]: SIF not found` | Image not pulled here: `bash pipeline_env_setup.sh`. |
| `FATAL [env-config]: Slingshot NCCL stack not built` | `bash pipeline_env_setup.sh` on a GPU node (one-time per image tag, ~20 min). |
| `FATAL [env-config]: Slingshot build predates the hostlibs dir` | Pre-D5 build: `bash pipeline_env_setup.sh --force`. |
| NCCL log shows `NET/Socket`, or bandwidth ~2 GB/s | CXI plugin not loading — **never** "fix" this by loading `brics/apptainer-multi-node`/`adapt.sh` (D1). Run with `NCCL_DEBUG=INFO` and look for `AWS Libfabric`; `ctypes.CDLL(os.environ["NCCL_NET_PLUGIN"])` inside the container names the missing soname (usually a missing `/host/usr/lib64` bind, a missing `hostlibs` symlink, or a plugin built against the wrong libfabric). |
| `megatron.bridge` imports from the image, not the repo | The image ships a regular `megatron` package — D3 contingency (derived SIF with the megatron packages uninstalled). `validate` catches this. |
| Recipe load fails `peft>=0.17.0 is required ... found peft==0.13.2` | Overlay not populated: `bash pipeline_env_setup.sh --only overlay` (D3b). |
| `WARNING [env-activate]: CONTAINER_PYTHON_OVERLAY ... configured but does not exist` | Same fix; the warning exists so this never degrades silently. |
| `torch.compile`/inductor or JIT-fuser warmup dies with `ld: cannot find /lib64/libc.so.6` (or `/usr/lib64/libc_nonshared.a`) | The raw host `/host/usr/lib64` reached `LD_LIBRARY_PATH` and inductor turned it into a `-L` dir (D5). Check nothing re-added it; only `<slingshot-dir>/hostlibs` belongs there. Warm inductor caches can hide this, so reproduce with a cold cache. |
| `ft_launcher` rejects `--ft-rank-*` flags | Image NVRX too old: `--disable-ft`, or qualify a newer image (D6). |
| `RuntimeError: ...gradient_accumulation_fusion...` | Should not happen — the image ships APEX. It means the payload is not running in this container (e.g. a hand-rolled `python` outside the shim). |
| CUDA "driver too old" / `Error 803` at startup | Compat-lib handling (D6b). `803` means the image's CUDA is too new for the driver — that image cannot be qualified; drop a rung on the ladder. |
| `DatasetInfo.from_directory ... must be called with a dataclass` | A processed-dataset cache written by a different `datasets` major version — keep `HF_DATASETS_CACHE` scoped (D7). |
| Apptainer fills `$HOME` | Never point `APPTAINER_CACHEDIR`/`APPTAINER_TMPDIR` at `$HOME`; the config refuses to run if you do. |
| Host libfabric path missing after a cluster upgrade | Override `GEODESIC_CONTAINER_HOST_LIBFABRIC` (`ls -d /opt/cray/libfabric/*`) and rebuild the Slingshot stack. |

## History

The bare-metal venv environment (a 435-line installer plus 12 order-dependent ARM
workarounds) was retired in the PR that introduced this document; its pinned versions and
every workaround are preserved in the "Retired from geodesic-megatron" Slack canvas in
`#megatron`, not in this repo. (The `.venv` that remains is dev tooling only — ruff,
pre-commit, the Claude Code hooks — and carries no torch.)

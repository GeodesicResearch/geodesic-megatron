#!/bin/bash
# ==============================================================================
# Environment Pipeline — the whole install, in one command (INFR-68)
#
#   bash pipeline_env_setup.sh                 # on a GPU node (compute/tunnel)
#   isambard_sbatch pipeline_env_submit.sbatch setup
#
# Four idempotent steps, each announcing its skip explicitly (never silent):
#   1. sif        NGC image -> SIF on project storage    (skip: SIF exists)
#   2. slingshot  Option-B NCCL + aws-ofi-nccl build     (skip: hostlibs exist; NEEDS GPU)
#   3. overlay    Python overlay packages                (skip: provenance matches)
#   4. validate   pipeline_env_validate.py in-container  (NEEDS GPU)
#
# On a login node (no GPU) steps 1 and 3 run and the script exits telling you how
# to finish 2+4. No silent degradation: every skip and deferral is printed.
#
# Flags:
#   --force            redo every step (re-pull the SIF, rebuild the stack)
#   --only <step>      run one step only: sif | slingshot | overlay | validate
#
# ALL parameters (image URI, SIF path, output dirs, component versions) live in
# pipeline_env_config.env — override with the GEODESIC_CONTAINER_* env vars
# documented there, never with extra CLI flags.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_env_config.env"

# Slingshot component versions. These start at the BriCS-recipe pins (current
# upstream releases) per the INFR-68 latest-first policy; bump here and re-run
# `--only slingshot --force` plus a validate to qualify newer ones.
OFI_NCCL_VERSION="${GEODESIC_CONTAINER_OFI_NCCL_VERSION:-v2.29.2-1}"
OFI_HWLOC_VERSION="${GEODESIC_CONTAINER_OFI_HWLOC_VERSION:-v2.13}"
OFI_PLUGIN_VERSION="${GEODESIC_CONTAINER_OFI_PLUGIN_VERSION:-v1.18.0}"

FORCE=0
ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --force) FORCE=1 ;;
        --only)
            shift
            ONLY="${1:-}"
            case "$ONLY" in
                sif|slingshot|overlay|validate) ;;
                *) echo "FATAL [env-setup]: --only takes one of: sif slingshot overlay validate" >&2; exit 1 ;;
            esac
            ;;
        *) echo "FATAL [env-setup]: unknown argument '$1' (only --force and --only <step>;" >&2
           echo "  all other parameters live in pipeline_env_config.env)" >&2; exit 1 ;;
    esac
    shift
done

want() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }

# Set when a GPU-only step could not run here: the install is INCOMPLETE and the
# exit status must say so, or a wrapper reading rc records success for an
# environment that cannot train.
DEFERRED=0

HAS_GPU=0
command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 && HAS_GPU=1

echo "===== Environment setup (image: $CONTAINER_IMAGE_URI) ====="

# ==============================================================================
# Step 1 — SIF acquisition (CPU-only: safe on a login node)
#
# Writes ${CONTAINER_SIF}.source.txt (URI, date, apptainer inspect) so every job
# log can echo the exact provenance of the image it ran under.
# ==============================================================================
setup_sif() {
    command -v apptainer >/dev/null || { echo "FATAL [env-setup/sif]: apptainer not on PATH" >&2; exit 1; }
    mkdir -p "$(dirname "$CONTAINER_SIF")" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

    # Project-quota preflight. Transient need ~= 2x image size (OCI layers in the
    # cache + the SIF). Never blocks — prints loudly so a near-full quota is a
    # conscious decision, mirroring the isambard_sbatch storage report.
    local quota_path=/projects/a5k proj_id=""
    if command -v lfs >/dev/null; then
        proj_id="$(lfs project -d "$quota_path" 2>/dev/null | awk '{print $1}')" || proj_id=""
        if [ -n "$proj_id" ]; then
            echo "[env-setup/sif] Project quota for $quota_path (need ~50 GB transient for a ~25 GB image):"
            lfs quota -p "$proj_id" "$quota_path" || true
        fi
    fi

    echo "[env-setup/sif] Pulling $CONTAINER_IMAGE_URI -> $CONTAINER_SIF"
    echo "[env-setup/sif] APPTAINER_CACHEDIR=$APPTAINER_CACHEDIR APPTAINER_TMPDIR=$APPTAINER_TMPDIR"
    # Pull to a .partial path and move into place only on success. apptainer pull
    # writes straight to its target, so an interrupted or over-quota pull (this
    # needs ~50 GB transient and the project quota runs hot) would otherwise leave
    # a TRUNCATED SIF at the real path — which passes the `-f` existence check in
    # env_config_require, makes this step print "SIF present — skipping", and only
    # surfaces as a bizarre failure inside a 16-node job. The rename is atomic.
    apptainer pull --force "${CONTAINER_SIF}.partial" "$CONTAINER_IMAGE_URI"
    mv -f "${CONTAINER_SIF}.partial" "$CONTAINER_SIF"

    {
        echo "image_uri: $CONTAINER_IMAGE_URI"
        echo "pulled_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "pulled_by: $USER"
        echo "apptainer_version: $(apptainer --version)"
        echo "--- apptainer inspect ---"
        apptainer inspect "$CONTAINER_SIF"
    } > "${CONTAINER_SIF}.source.txt"

    echo "[env-setup/sif] Done. SIF: $CONTAINER_SIF (provenance: ${CONTAINER_SIF}.source.txt)"
    echo "[env-setup/sif] Cache hygiene: 'apptainer cache clean --days 30' reclaims old OCI layers"
}

# ==============================================================================
# Step 2 — Slingshot NCCL stack, the official Isambard "Option B" recipe
# (docs.isambard.ac.uk → guides/containers + guides/nccl,
# example-data/apptainer/build_nccl.sh)
#
# Because the NGC image bundles a NEWER CUDA runtime than the host driver
# supports natively, NCCL and the aws-ofi-nccl CXI plugin must be built INSIDE
# the image against the image's CUDA + the host's Cray libfabric. The built
# libraries live on shared project storage (never baked into the SIF) and are
# bind-mounted at /opt/slingshot at runtime.
#
# Builds into $CONTAINER_SLINGSHOT_DIR:
#   nccl/          NCCL, compiled vs image CUDA, sm_90 (GH200)
#   hwloc/         build dep of aws-ofi-nccl
#   aws-ofi-nccl/  the CXI network plugin (lib/libnccl-net.so)
#   nccl-tests/    bandwidth benchmark binaries (fabric verification)
# ==============================================================================
setup_slingshot() {
    # Bootstrap step: validates only its OWN inputs (deliberately NOT
    # env_config_require, which demands the build output this step creates).
    [ -f "$CONTAINER_SIF" ] || { echo "FATAL [env-setup/slingshot]: SIF missing: $CONTAINER_SIF — run without --only, or --only sif first" >&2; exit 1; }
    [ -d "$CONTAINER_HOST_LIBFABRIC" ] || { echo "FATAL [env-setup/slingshot]: host libfabric missing: $CONTAINER_HOST_LIBFABRIC" >&2; exit 1; }
    [ "$HAS_GPU" = "1" ] || { echo "FATAL [env-setup/slingshot]: no GPU on this node — the NCCL build queries the CUDA toolchain" >&2; exit 1; }

    if [ "$FORCE" = "1" ]; then
        # SHARED ARTIFACT: this dir is bind-mounted at /opt/slingshot by every
        # running job on the cluster. Deleting it under a live multi-node run pulls
        # NCCL's plugin out from under it. Announce loudly, and refuse to rm -rf a
        # path that a bad GEODESIC_CONTAINER_SLINGSHOT_DIR override could have
        # pointed somewhere catastrophic (empty, /, or outside CONTAINER_ROOT).
        case "$CONTAINER_SLINGSHOT_DIR" in
            "$CONTAINER_ROOT"/*) ;;
            *) echo "FATAL [env-setup/slingshot]: refusing --force rebuild of '$CONTAINER_SLINGSHOT_DIR' — it is not under CONTAINER_ROOT ($CONTAINER_ROOT)" >&2; exit 1 ;;
        esac
        echo "[env-setup/slingshot] --force: REPLACING the shared Slingshot stack at $CONTAINER_SLINGSHOT_DIR."
        echo "[env-setup/slingshot] Any job currently running with this dir bind-mounted at /opt/slingshot will lose its NCCL plugin."
        rm -rf "$CONTAINER_SLINGSHOT_DIR"
    fi
    mkdir -p "$CONTAINER_SLINGSHOT_DIR"

    echo "[env-setup/slingshot] SIF:      $CONTAINER_SIF"
    echo "[env-setup/slingshot] Output:   $CONTAINER_SLINGSHOT_DIR"
    echo "[env-setup/slingshot] Versions: nccl=$OFI_NCCL_VERSION hwloc=$OFI_HWLOC_VERSION aws-ofi-nccl=$OFI_PLUGIN_VERSION"

    # Scrub host toolchain/venv env — the same poison set the exec shim scrubs.
    # The host's CC/CXX=/usr/bin/g*-12 do not exist in the image and hijacked the
    # NCCL make until this was added; the build must use the image's toolchain.
    unset LD_PRELOAD PYTHONPATH VIRTUAL_ENV NCCL_LIBRARY \
          CC CXX CUDAHOSTCXX CUDA_HOME CPLUS_INCLUDE_PATH C_INCLUDE_PATH CUDNN_PATH
    export LD_LIBRARY_PATH=""

    # The build runs inside the image with ONLY the Option B binds (host
    # libfabric, /usr/lib64 for libcxi/libnl, and the output dir at
    # /opt/slingshot — the in-container paths the official recipe uses). Build
    # scratch is node-local /tmp.
    apptainer exec --nv \
        --bind "${CONTAINER_HOST_LIBFABRIC}:/host${CONTAINER_HOST_LIBFABRIC}:ro" \
        --bind /usr/lib64:/host/usr/lib64:ro \
        --bind "${CONTAINER_SLINGSHOT_DIR}:/opt/slingshot" \
        "$CONTAINER_SIF" bash -euo pipefail -c "
            export CUDA_HOME=/usr/local/cuda
            export LIBFABRIC_HOME=/host${CONTAINER_HOST_LIBFABRIC}
            export MPI_HOME=/usr/local/mpi
            export TMPDIR=/tmp
            BUILD_ROOT=\$(mktemp -d /tmp/ofi_build_XXXX)

            # --- NCCL (vs image CUDA, Hopper/sm_90) ---
            # src.build alone fully stages lib/ + include/ under BUILDDIR; adding
            # the 'install' target in the same -j invocation races it on the
            # header copies ('install: cannot create regular file ... File exists').
            cd \$BUILD_ROOT && git clone --depth 1 --branch '$OFI_NCCL_VERSION' https://github.com/NVIDIA/nccl.git
            mkdir -p /opt/slingshot/nccl
            cd \$BUILD_ROOT/nccl && make -j \$(nproc) src.build BUILDDIR=/opt/slingshot/nccl NVCC_GENCODE='-gencode=arch=compute_90,code=sm_90'
            rm -rf /opt/slingshot/nccl/obj   # multi-GB intermediates; keep the shared dir lean
            export NCCL_HOME=/opt/slingshot/nccl

            # --- hwloc (aws-ofi-nccl build dep) ---
            cd \$BUILD_ROOT && git clone --depth 1 --branch '$OFI_HWLOC_VERSION' https://github.com/open-mpi/hwloc.git
            cd \$BUILD_ROOT/hwloc && ./autogen.sh && ./configure --disable-nvml --prefix=/opt/slingshot/hwloc
            make -j \$(nproc) install

            # --- aws-ofi-nccl (the CXI plugin) ---
            cd \$BUILD_ROOT && git clone --depth 1 --branch '$OFI_PLUGIN_VERSION' https://github.com/aws/aws-ofi-nccl.git
            cd \$BUILD_ROOT/aws-ofi-nccl && ./autogen.sh
            export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH:-}:/host/usr/lib64
            ./configure --prefix=/opt/slingshot/aws-ofi-nccl \
                --with-cuda=\$CUDA_HOME \
                --with-libfabric=\$LIBFABRIC_HOME \
                --with-mpi=\$MPI_HOME \
                --with-hwloc=/opt/slingshot/hwloc \
                --disable-tests
            make -j \$(nproc) install

            # --- nccl-tests (fabric benchmark binaries) ---
            # env -u: by this point LD_LIBRARY_PATH carries /host/usr/lib64 (host
            # SLES libs, needed by the aws-ofi-nccl configure checks); the image's
            # git links a different libcurl/nghttp2 pair and dies if it resolves
            # against them.
            cd \$BUILD_ROOT && env -u LD_LIBRARY_PATH git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git
            cd \$BUILD_ROOT/nccl-tests && make -j \$(nproc) MPI=1 MPI_HOME=\$MPI_HOME NCCL_HOME=\$NCCL_HOME CUDA_HOME=\$CUDA_HOME
            # rm first: on a RETRY (interrupted build) 'cp -r src dest' NESTS into an
            # existing dest, putting the binaries at nccl-tests/build/all_reduce_perf
            # instead of nccl-tests/all_reduce_perf — and provenance would still be
            # written, so both the skip gate and env_config_require would call the
            # install complete while the documented path is wrong.
            rm -rf /opt/slingshot/nccl-tests
            cp -r \$BUILD_ROOT/nccl-tests/build /opt/slingshot/nccl-tests

            rm -rf \$BUILD_ROOT
        "

    # hostlibs: a symlink-ONLY directory exposing the few host /usr/lib64 sonames
    # the CXI plugin needs at dlopen (libcxi, libnl) — this dir, NOT
    # /host/usr/lib64 itself, goes on the container's LD_LIBRARY_PATH. Exposing
    # the whole host dir poisons torch.compile links: inductor turns
    # LD_LIBRARY_PATH into -L dirs and the host libc.so (SUSE ld script) names
    # absolute paths that don't exist in the image ("ld: cannot find
    # /lib64/libc.so.6"). A dir with no libc cannot poison.
    mkdir -p "$CONTAINER_SLINGSHOT_DIR/hostlibs"
    for so in libcxi.so.1 libnl-3.so.200 libnl-route-3.so.200; do
        [ -e "/usr/lib64/$so" ] && ln -sf "/host/usr/lib64/$so" "$CONTAINER_SLINGSHOT_DIR/hostlibs/$so"
    done

    {
        echo "sif: $CONTAINER_SIF"
        echo "sif_source: $(head -1 "${CONTAINER_SIF}.source.txt" 2>/dev/null || echo unknown)"
        echo "built_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "built_by: $USER"
        echo "built_on: $(hostname)"
        echo "nccl: $OFI_NCCL_VERSION"
        echo "hwloc: $OFI_HWLOC_VERSION"
        echo "aws_ofi_nccl: $OFI_PLUGIN_VERSION"
        echo "host_libfabric: $CONTAINER_HOST_LIBFABRIC"
    } > "$CONTAINER_SLINGSHOT_DIR/provenance.txt"

    echo "[env-setup/slingshot] Done. Plugin: $CONTAINER_SLINGSHOT_DIR/aws-ofi-nccl/lib/libnccl-net.so"
    echo "[env-setup/slingshot] Provenance: $CONTAINER_SLINGSHOT_DIR/provenance.txt"
}

# ==============================================================================
# Step 3 — Python overlay
#
# Installs CONTAINER_OVERLAY_PACKAGES into the PYTHONPATH overlay that
# pipeline_env_activate.sh layers between the repo and the image (resolution:
# repo > overlay > image). The SIF is never modified. --no-deps is deliberate: a
# dependency closure could drag in a PyPI torch that shadows the image's
# CUDA-matched one.
# ==============================================================================
setup_overlay() {
    [ -f "$CONTAINER_SIF" ] || { echo "FATAL [env-setup/overlay]: SIF missing: $CONTAINER_SIF" >&2; exit 1; }

    local prov="$CONTAINER_PYTHON_OVERLAY/provenance.txt"
    if [ "$FORCE" != "1" ] && [ -f "$prov" ] && grep -qxF "packages: $CONTAINER_OVERLAY_PACKAGES" "$prov"; then
        echo "[env-setup/overlay] Overlay already matches configured packages ($CONTAINER_OVERLAY_PACKAGES) — skipping. (--force rebuilds)"
        return 0
    fi

    if [ "$FORCE" = "1" ]; then
        # Same guard as the slingshot step: a mistyped
        # GEODESIC_CONTAINER_PYTHON_OVERLAY (e.g. /projects/a5k/public) would make
        # this an unguarded recursive delete on shared storage.
        case "$CONTAINER_PYTHON_OVERLAY" in
            "$CONTAINER_ROOT"/*) ;;
            *) echo "FATAL [env-setup/overlay]: refusing --force rebuild of '$CONTAINER_PYTHON_OVERLAY' — it is not under CONTAINER_ROOT ($CONTAINER_ROOT)" >&2; exit 1 ;;
        esac
        rm -rf "$CONTAINER_PYTHON_OVERLAY"
    fi
    mkdir -p "$CONTAINER_PYTHON_OVERLAY"

    echo "[env-setup/overlay] Installing into $CONTAINER_PYTHON_OVERLAY: $CONTAINER_OVERLAY_PACKAGES"
    # The pip runs INSIDE the image (matching python ABI). Bind BOTH /projects
    # AND /lus: on Isambard /projects/a5k/public is a symlink into /lus, so
    # binding /projects alone leaves the target path dangling inside the
    # container (FileNotFoundError) — the same pair the exec shim binds.
    apptainer exec --bind /projects,/lus "$CONTAINER_SIF" \
        python -m pip install --no-deps --target "$CONTAINER_PYTHON_OVERLAY" $CONTAINER_OVERLAY_PACKAGES

    {
        echo "packages: $CONTAINER_OVERLAY_PACKAGES"
        echo "sif: $CONTAINER_SIF"
        echo "installed_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "installed_by: $USER"
        echo "why: image ships versions too old for this repo (see docs/environment.md)"
    } > "$prov"

    echo "[env-setup/overlay] Done. Provenance: $prov"
}

# ==============================================================================
# Step 4 — Validate (GPU required): the environment validator, run inside the
# container exactly as a pipeline would run it.
# ==============================================================================
setup_validate() {
    [ "$HAS_GPU" = "1" ] || { echo "FATAL [env-setup/validate]: no GPU on this node" >&2; exit 1; }
    echo "[env-setup/validate] Running environment validation inside the container..."
    REPO_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/pipeline_env_exec.sh" \
        "cd $SCRIPT_DIR; source pipeline_env_activate.sh || exit 1; python pipeline_env_validate.py"
}

# ------------------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------------------
if want sif; then
    if [ "$FORCE" != "1" ] && [ -f "$CONTAINER_SIF" ]; then
        echo "[env-setup 1/4 sif] SIF present: $CONTAINER_SIF — skipping pull. (--force re-pulls)"
    else
        setup_sif
    fi
fi

if want slingshot; then
    if [ "$FORCE" != "1" ] && [ -L "$CONTAINER_SLINGSHOT_DIR/hostlibs/libcxi.so.1" ]; then
        echo "[env-setup 2/4 slingshot] Stack present: $CONTAINER_SLINGSHOT_DIR — skipping build. (--force rebuilds)"
    elif [ "$HAS_GPU" = "1" ]; then
        setup_slingshot
    else
        echo "[env-setup 2/4 slingshot] DEFERRED — no GPU on this node. Finish on a GPU node with:" >&2
        echo "    bash pipeline_env_setup.sh" >&2
        echo "  or: isambard_sbatch pipeline_env_submit.sbatch setup" >&2
        DEFERRED=1
    fi
fi

want overlay && setup_overlay

if want validate; then
    if [ "$HAS_GPU" = "1" ] && { [ -n "$ONLY" ] || [ -L "$CONTAINER_SLINGSHOT_DIR/hostlibs/libcxi.so.1" ]; }; then
        setup_validate
        echo "===== Environment setup COMPLETE (validation above) ====="
    else
        echo "===== Environment setup: CPU-side steps done; run on a GPU node to finish (see 2/4) ====="
        DEFERRED=1
    fi
fi

if [ "$DEFERRED" = "1" ]; then
    echo "[env-setup] INCOMPLETE: GPU-only steps were deferred — exiting 2 so callers do not read this as a finished install." >&2
    exit 2
fi

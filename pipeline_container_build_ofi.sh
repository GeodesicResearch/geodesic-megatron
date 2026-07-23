#!/bin/bash
# ==============================================================================
# Container Pipeline — Slingshot NCCL-stack build (INFR-68)
#
# One-time build of the "Option B" Slingshot networking stack for the pipeline
# container, following the official Isambard recipe (docs.isambard.ac.uk →
# guides/containers + guides/nccl, example-data/apptainer/build_nccl.sh):
# because the NGC image bundles a NEWER CUDA runtime than the host driver
# supports natively, NCCL and the aws-ofi-nccl CXI plugin must be built INSIDE
# the image against the image's CUDA + the host's Cray libfabric. The built
# libraries live on shared project storage (never baked into the SIF) and are
# bind-mounted at /opt/slingshot at runtime.
#
# Builds (into $CONTAINER_SLINGSHOT_DIR):
#   nccl/          — NCCL, compiled vs image CUDA, sm_90 (GH200)
#   hwloc/         — build dep of aws-ofi-nccl
#   aws-ofi-nccl/  — the CXI network plugin (lib/libnccl-net.so)
#   nccl-tests/    — bandwidth benchmark binaries (used by validation stage C3)
#
# Usage (GPU node; ~20 min; or: pipeline_container_submit.sbatch build-ofi):
#   bash pipeline_container_build_ofi.sh [--force]
#
# All parameters (SIF, output dir, versions) come from
# pipeline_container_config.env / the GEODESIC_CONTAINER_* overrides documented
# there. --force allows overwriting an existing build (never silent).
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_container_config.env"

# Component versions. These start at the BriCS-recipe pins (which are current
# upstream releases) per the INFR-68 latest-first policy; bump here and re-run
# validation stages C0b/C3 to qualify newer ones.
OFI_NCCL_VERSION="${GEODESIC_CONTAINER_OFI_NCCL_VERSION:-v2.29.2-1}"
OFI_HWLOC_VERSION="${GEODESIC_CONTAINER_OFI_HWLOC_VERSION:-v2.13}"
OFI_PLUGIN_VERSION="${GEODESIC_CONTAINER_OFI_PLUGIN_VERSION:-v1.18.0}"

FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        *) echo "FATAL [build-ofi]: unknown argument '$arg' (only --force; parameters live in pipeline_container_config.env)" >&2; exit 1 ;;
    esac
done

# This is the bootstrap step for the Slingshot build, so it validates only its
# own inputs (deliberately NOT container_config_require, which demands the
# build output this script creates).
[ -f "$CONTAINER_SIF" ] || { echo "FATAL [build-ofi]: SIF missing: $CONTAINER_SIF — run pipeline_container_pull.sh first" >&2; exit 1; }
[ -d "$CONTAINER_HOST_LIBFABRIC" ] || { echo "FATAL [build-ofi]: host libfabric missing: $CONTAINER_HOST_LIBFABRIC" >&2; exit 1; }
command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 || {
    echo "FATAL [build-ofi]: no GPU on this node — run on a GPU node (the NCCL build queries the CUDA toolchain)" >&2; exit 1; }

if [ -e "$CONTAINER_SLINGSHOT_DIR/aws-ofi-nccl" ] && [ "$FORCE" != "1" ]; then
    echo "FATAL [build-ofi]: $CONTAINER_SLINGSHOT_DIR already contains a build. Re-run with --force to rebuild." >&2
    exit 1
fi
[ "$FORCE" = "1" ] && rm -rf "$CONTAINER_SLINGSHOT_DIR"
mkdir -p "$CONTAINER_SLINGSHOT_DIR"

echo "[build-ofi] SIF:      $CONTAINER_SIF"
echo "[build-ofi] Output:   $CONTAINER_SLINGSHOT_DIR"
echo "[build-ofi] Versions: nccl=$OFI_NCCL_VERSION hwloc=$OFI_HWLOC_VERSION aws-ofi-nccl=$OFI_PLUGIN_VERSION"

# Scrub host toolchain/venv env — same poison set the exec shim scrubs. The
# host's CC/CXX=/usr/bin/g*-12 do not exist in the image and hijacked the NCCL
# make until this was added; the build must use the image's own toolchain.
unset LD_PRELOAD PYTHONPATH VIRTUAL_ENV NCCL_LIBRARY \
      CC CXX CUDAHOSTCXX CUDA_HOME CPLUS_INCLUDE_PATH C_INCLUDE_PATH CUDNN_PATH
export LD_LIBRARY_PATH=""

# The build runs inside the image with ONLY the Option B binds (host libfabric,
# /usr/lib64 for libcxi/libnl, and the output dir at /opt/slingshot — the same
# in-container paths the official recipe uses). Build scratch is node-local /tmp.
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
        # src.build alone fully stages lib/ + include/ under BUILDDIR; adding the
        # 'install' target in the same -j invocation races it on the header copies
        # ('install: cannot create regular file ... File exists').
        cd \$BUILD_ROOT && git clone --depth 1 --branch '$OFI_NCCL_VERSION' https://github.com/NVIDIA/nccl.git
        mkdir -p /opt/slingshot/nccl
        cd \$BUILD_ROOT/nccl && make -j \$(nproc) src.build BUILDDIR=/opt/slingshot/nccl NVCC_GENCODE='-gencode=arch=compute_90,code=sm_90'
        rm -rf /opt/slingshot/nccl/obj   # multi-GB intermediate objects; keep the shared dir lean
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

        # --- nccl-tests (benchmark binaries for validation stage C3) ---
        # env -u: by this point LD_LIBRARY_PATH carries /host/usr/lib64 (host SLES
        # libs, needed by the aws-ofi-nccl configure checks); the image's git links
        # a different libcurl/nghttp2 pair and dies if it resolves against them.
        cd \$BUILD_ROOT && env -u LD_LIBRARY_PATH git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git
        cd \$BUILD_ROOT/nccl-tests && make -j \$(nproc) MPI=1 MPI_HOME=\$MPI_HOME NCCL_HOME=\$NCCL_HOME CUDA_HOME=\$CUDA_HOME
        cp -r \$BUILD_ROOT/nccl-tests/build /opt/slingshot/nccl-tests

        rm -rf \$BUILD_ROOT
    "

# Provenance: enough to reproduce/audit this build from the outputs alone.
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

echo "[build-ofi] Done. Plugin: $CONTAINER_SLINGSHOT_DIR/aws-ofi-nccl/lib/libnccl-net.so"
echo "[build-ofi] Provenance: $CONTAINER_SLINGSHOT_DIR/provenance.txt"

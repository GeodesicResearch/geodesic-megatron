# shellcheck shell=bash
# ==============================================================================
# Slingshot/CXI fabric settings for NCCL, exported into the launching shell so
# that srun --export=ALL and apptainer's environment passthrough carry them into
# every rank. Source it after common.sh.
#
# Catalogue:
#   fabric_select  NCCL's network: the aws-ofi-nccl plugin over CXI on hsn
#   fabric_base    fabric_select, plus the CXI and NCCL settings every
#                  multi-node payload of the pipelines runs with
# ==============================================================================

fabric_select() {
    export NCCL_NET="AWS Libfabric"
    export FI_PROVIDER=cxi
    export NCCL_SOCKET_IFNAME=hsn
}

fabric_base() {
    fabric_select
    export NCCL_CROSS_NIC=1
    export NCCL_NET_GDR_LEVEL=PHB
    export FI_CXI_DISABLE_HOST_REGISTER=1
    export FI_MR_CACHE_MONITOR=userfaultfd
    export FI_CXI_DEFAULT_CQ_SIZE=131072
    export FI_CXI_RX_MATCH_MODE=soft
    export NCCL_NVLS_ENABLE=0
    export NCCL_DEBUG=WARN
}

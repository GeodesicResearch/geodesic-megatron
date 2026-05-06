#!/bin/bash
set -u
JID=${JID:?}
REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
LOG=/tmp/queue_v3_big
GROUP=${GROUP:?}
NODELIST_RANGE=${NODELIST_RANGE:?}
NODELIST=${NODELIST:?}
FULL_TUNNEL_NODELIST=${FULL_TUNNEL_NODELIST:?}
PORT=${PORT:?}
CONV_NODE=${CONV_NODE:?}
COH_NODE=${COH_NODE:?}
NODES=${NODES:?}
CPT_LABEL=${CPT_LABEL:?}
CPT_YAML=${CPT_YAML:?}
CPT_ITER=${CPT_ITER:?}
CPT_CKPT=${CPT_CKPT:?}
CPT_MODEL_KIND=${CPT_MODEL_KIND:?}

CKPT_BASE=/projects/a5k/public/checkpoints/megatron
mkdir -p "$LOG"
cd "$REPO"
source $REPO/configs/inoculation_midtraining/im_fyn1668_v3/_orchestrator_lib.sh

run_step "$CPT_LABEL" "$CPT_YAML" "$CPT_ITER" "$CPT_CKPT" cpt "$CPT_MODEL_KIND" || exit 1
echo "[$GROUP $CPT_LABEL] CPT done at $(date -u +%FT%TZ)"

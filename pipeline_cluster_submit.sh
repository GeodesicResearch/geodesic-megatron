#!/bin/bash
# ==============================================================================
# Submit a pipeline job on this cluster.
#
#   ./pipeline_cluster_submit.sh [sbatch options...] <script.sbatch> [script arguments...]
#
# On a cluster whose site provides a submit wrapper (the profile's
# CLUSTER_SUBMIT_WRAPPER: isambard_sbatch on Isambard) this runs that wrapper with
# the arguments as given. Elsewhere it runs sbatch, preceded by the options the
# script's own "#PIPELINE-SUBMIT" line asks for. A script's #SBATCH lines hold its
# job's shape on Isambard, and sbatch options given on the command line override
# them, so the line states what another cluster's nodes need:
#
#   #PIPELINE-SUBMIT gpus-per-node=node output=megatron_runs/train-%j.out
#
#   gpus-per-node=node  every GPU of a node (CLUSTER_GPUS_PER_NODE)
#   output=<path>       the job's log, under the cluster's log root (CLUSTER_LOG_ROOT).
#                       The script creates its directory, since sbatch creates none
#   <name>[=<value>]    any other sbatch option, as --<name>[=<value>]
#
# PIPELINE_SUBMIT_OPTIONS holds sbatch options for every job this shell submits,
# as words separated by spaces: a node pool (--exclude=...), or the resources of
# the jobs a script such as configs/control_pretraining/build_corpora.sh submits
# for you. They follow the line's options, and an option on the command line
# follows both, so the command line overrides the variable and the variable the
# line. The submitted job inherits GEODESIC_CLUSTER and the cluster's
# file-creation mask from this shell. The sbatch command is printed on stderr, and
# stdout is sbatch's own, so --parsable still prints the job id alone.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline/sh/cluster.sh"
cluster_load

if [ -n "$CLUSTER_SUBMIT_WRAPPER" ]; then
    exec "$CLUSTER_SUBMIT_WRAPPER" "$@"
fi

# The script is the first argument that is a file and not an option's value: an option without "=" takes the next
# word as its value unless that word is another option.
SCRIPT=""
expect_value=0
for word in "$@"; do
    case "$word" in
        -*)
            case "$word" in *=*) expect_value=0 ;; *) expect_value=1 ;; esac
            ;;
        *)
            if [ "$expect_value" = "0" ] || [ -f "$word" ]; then
                SCRIPT="$word"
                break
            fi
            expect_value=0
            ;;
    esac
done
if [ -z "$SCRIPT" ] || [ ! -f "$SCRIPT" ]; then
    echo "Usage: pipeline_cluster_submit.sh [sbatch options...] <script.sbatch> [script arguments...]" >&2
    echo "FATAL [submit]: no script file among the arguments: $*" >&2
    exit 2
fi

# A script holds at most one "#PIPELINE-SUBMIT" line. A script without one is submitted with its #SBATCH lines alone.
HINTS="$(sed -n 's/^#PIPELINE-SUBMIT[[:space:]]\{1,\}//p' "$SCRIPT")"
if [ "$(printf '%s\n' "$HINTS" | sed '/^$/d' | wc -l)" -gt 1 ]; then
    echo "FATAL [submit]: $SCRIPT has more than one #PIPELINE-SUBMIT line." >&2
    exit 1
fi

OPTIONS=()
for hint in $HINTS; do
    case "$hint" in
        gpus-per-node=node)
            OPTIONS+=("--gpus-per-node=$CLUSTER_GPUS_PER_NODE")
            ;;
        output=*)
            log="$CLUSTER_LOG_ROOT/${hint#output=}"
            mkdir -p "$(dirname "$log")"
            OPTIONS+=("--output=$log")
            ;;
        *)
            OPTIONS+=("--$hint")
            ;;
    esac
done

# Split on spaces without pathname expansion: a node list holds brackets.
read -r -a SHELL_OPTIONS <<< "${PIPELINE_SUBMIT_OPTIONS:-}"
OPTIONS+=(${SHELL_OPTIONS[@]+"${SHELL_OPTIONS[@]}"})

printf '+ sbatch' >&2
printf ' %q' ${OPTIONS[@]+"${OPTIONS[@]}"} "$@" >&2
printf '\n' >&2
exec sbatch ${OPTIONS[@]+"${OPTIONS[@]}"} "$@"

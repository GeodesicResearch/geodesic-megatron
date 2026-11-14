# shellcheck shell=bash
# ==============================================================================
# Copies of a job's inputs on a node's own disk. Some shared storage answers a
# small read or a page fault slowly. A cluster with such storage sets a directory
# for the copies in its profile (CLUSTER_INPUT_STAGE_ROOT). A launcher copies an
# input there and binds the copy over the input's own path in the container. The
# payload then reads the input by that path, and records that path.
#
# The copy functions take every path as an argument and read no profile value. A
# node of a job can run them with this file alone:
#   srun ... bash -c 'source "$1/pipeline/sh/stage.sh" && stage_tree "$2" "$3"' _ "$REPO_DIR" SRC DST
#
# Catalogue:
#   stage_root  the directory for the copies, or nothing where inputs are read in place
#   stage_file  copy one file, which appears at its path only when complete
#   stage_list  list the files under a directory, refusing a link to nothing
#   stage_tree  copy every file under a directory
#   stage_out   copy the files a payload wrote in a staged directory to the real one
#   stage_bind  bind a staged copy over the path it stands for, in every container started after it
# ==============================================================================

# Files stage_tree copies at a time.
STAGE_COPIES="${STAGE_COPIES:-16}"

# stage_root: GEODESIC_INPUT_STAGE_ROOT when it is set, else the loaded profile's CLUSTER_INPUT_STAGE_ROOT. An empty
# value, from either, means the job reads its inputs where they are stored.
stage_root() {
    echo "${GEODESIC_INPUT_STAGE_ROOT-$CLUSTER_INPUT_STAGE_ROOT}"
}

# stage_file SRC DST: copies file SRC to DST in 16 MiB blocks. A symbolic link at SRC gives the file it points to.
# The function creates DST's directory, writes DST.incoming and renames it once its size equals SRC's. DST keeps
# SRC's modification time. Returns 1 with a message when the copy fails or the sizes differ, and returns 1 when DST
# is a directory.
stage_file() {
    local src="$1" dst="$2" want got
    mkdir -p "$(dirname "$dst")" || return 1
    if ! dd if="$src" of="$dst.incoming" bs=16M status=none; then
        echo "FATAL [stage]: copying $src to $dst.incoming failed." >&2
        return 1
    fi
    want="$(stat -L -c %s "$src")" || return 1
    got="$(stat -c %s "$dst.incoming")" || return 1
    if [ "$want" != "$got" ]; then
        echo "FATAL [stage]: $dst.incoming is $got bytes and $src is $want." >&2
        return 1
    fi
    touch -r "$src" "$dst.incoming" || return 1
    mv -T "$dst.incoming" "$dst"
}

# stage_list DIR LIST: writes the relative path of every file under directory DIR to file LIST, each followed by
# a NUL. A symbolic link gives the file or directory it points to. Returns 1 with a message when DIR is no
# directory, when a link under it points to nothing, and when the listing fails.
stage_list() {
    local dir="${1%/}" list="$2" dangling
    if [ ! -d "$dir" ]; then
        echo "FATAL [stage]: $dir is not a directory." >&2
        return 1
    fi
    if ! dangling="$(find -L "$dir" -type l -print -quit)" \
        || ! (cd "$dir" && find -L . -type f -printf '%P\0') > "$list"; then
        echo "FATAL [stage]: listing the files under $dir failed." >&2
        return 1
    fi
    if [ -n "$dangling" ]; then
        echo "FATAL [stage]: $dangling is a link to nothing." >&2
        return 1
    fi
}

# stage_tree SRC DST: copies every file under directory SRC (stage_list) to the same relative path under DST
# (stage_file), STAGE_COPIES at a time. Returns 1 when the listing fails and when any copy fails.
stage_tree() {
    local src="${1%/}" dst="${2%/}" list rel pid pids=() failed=0
    list="$(mktemp)" || return 1
    if ! stage_list "$src" "$list"; then
        rm -f "$list"
        return 1
    fi
    while IFS= read -r -d '' rel; do
        stage_file "$src/$rel" "$dst/$rel" &
        pids+=("$!")
        if [ "${#pids[@]}" -ge "$STAGE_COPIES" ]; then
            for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
            pids=()
        fi
    done < "$list"
    for pid in ${pids[@]+"${pids[@]}"}; do wait "$pid" || failed=1; done
    rm -f "$list"
    [ "$failed" -eq 0 ]
}

# stage_out STAGE DST [INPUT...]: copies every file under directory STAGE (stage_list) to the same relative path
# under DST (stage_file). It leaves out each INPUT, the relative path of a file the launcher copied in. Returns 1
# when the listing fails and when a copy fails.
stage_out() {
    local stage="${1%/}" dst="${2%/}" list rel input failed=0
    shift 2
    list="$(mktemp)" || return 1
    if ! stage_list "$stage" "$list"; then
        rm -f "$list"
        return 1
    fi
    while IFS= read -r -d '' rel; do
        for input in "$@"; do
            [ "$rel" != "$input" ] || continue 2
        done
        stage_file "$stage/$rel" "$dst/$rel" || { failed=1; break; }
    done < "$list"
    rm -f "$list"
    [ "$failed" -eq 0 ]
}

# stage_bind COPY PATH: every container this shell starts afterwards, on any node of the job, has COPY at PATH.
# Apptainer adds the pairs in APPTAINER_BIND to the binds pipeline_env_exec.sh gives it.
stage_bind() {
    export APPTAINER_BIND="${APPTAINER_BIND:+$APPTAINER_BIND,}$1:$2"
}

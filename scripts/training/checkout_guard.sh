#!/bin/bash
# Refuse to train a config with the code of a different git checkout.
#
# The code that trains is REPO_DIR's. A config that lives in another checkout was written for that
# checkout's code, and older code skips settings it does not know: a masked campaign once trained
# unmasked this way, its config in a worktree and its code in a checkout that predated the masking.
# pipeline_training_launch.sh runs this right after resolving REPO_DIR, which covers launches from
# an salloc or a tunnel; pipeline_training_submit.sbatch also runs the copy in the config's own
# checkout, so a REPO_DIR whose launcher predates the check is refused at submission.
#
# A checkout's top is git's toplevel. A checkout whose git config says core.bare=true while it still
# holds working files (this cluster's main checkout, with the worktrees inside it) has no toplevel
# in git's eyes; its top is the directory holding its .git. A relative config path is read against
# REPO_DIR, as training reads it. A config or a REPO_DIR in no git repository (a git-archive snapshot,
# a config under /projects) is not checked, and neither is a config that does not exist (the launcher
# reports that itself). Any other git failure (git missing, a repository git will not read, such as
# one owned by another account) is fatal: a check that cannot see a checkout must not pass it.
#
#   usage: checkout_guard.sh <config> <repo_dir>
#
# Exit 1 with a FATAL message when the config's checkout is not REPO_DIR's or git cannot tell,
# unless ALLOW_CROSS_CHECKOUT_CONFIG=1; 2 on a usage error; 0 otherwise.
set -uo pipefail

if [ $# -ne 2 ]; then
    echo "usage: checkout_guard.sh <config> <repo_dir>" >&2
    exit 2
fi
config="$1"
repo_dir="$2"
waiver="ALLOW_CROSS_CHECKOUT_CONFIG=1"

if [ "${ALLOW_CROSS_CHECKOUT_CONFIG:-0}" = "1" ]; then
    exit 0
fi
if ! command -v git >/dev/null 2>&1; then
    echo "FATAL: checkout_guard.sh needs git to compare the config's checkout with REPO_DIR's; put git on" >&2
    echo "       PATH, or set $waiver to skip the check." >&2
    exit 1
fi

errors="$(mktemp)"
trap 'rm -f "$errors"' EXIT

git_unreadable() {  # $1 = directory, $2 = git's message
    echo "FATAL: git cannot read the repository holding $1: $2" >&2
    echo "       Fix that (for a repository owned by another account: git config --global --add" >&2
    echo "       safe.directory <checkout>), or set $waiver to skip the check." >&2
}

# The checkout holding directory $1 on stdout, or nothing when $1 is in no git repository.
checkout_top() {
    local gitdir top
    if ! gitdir="$(LC_ALL=C git -C "$1" rev-parse --absolute-git-dir 2>"$errors")"; then
        case "$(cat "$errors")" in
            *"not a git repository"*) return 0 ;;
        esac
        git_unreadable "$1" "$(cat "$errors")"
        return 1
    fi
    if top="$(LC_ALL=C git -C "$1" rev-parse --show-toplevel 2>"$errors")"; then
        realpath "$top"
        return 0
    fi
    case "$(cat "$errors")" in
        *"must be run in a work tree"*) ;;
        *) git_unreadable "$1" "$(cat "$errors")"; return 1 ;;
    esac
    # No work tree in git's eyes: a true bare repository holds no code, while a .git directory with
    # core.bare=true sits in a directory of working files, which is the checkout.
    if [ "$(basename "$gitdir")" = ".git" ]; then
        realpath "$(dirname "$gitdir")"
    fi
    return 0
}

case "$config" in
    /*) ;;
    *) config="$repo_dir/$config" ;;
esac
config_real="$(realpath -e "$config" 2>/dev/null)" || exit 0

config_top="$(checkout_top "$(dirname "$config_real")")" || exit 1
repo_top="$(checkout_top "$repo_dir")" || exit 1
if [ -z "$config_top" ] || [ -z "$repo_top" ] || [ "$config_top" = "$repo_top" ]; then
    exit 0
fi
echo "FATAL: config $config_real belongs to the checkout $config_top, but the code would run from $repo_top." >&2
echo "       Submit from the config's checkout (or set GEODESIC_REPO_DIR=$config_top), or set" >&2
echo "       $waiver to train it with $repo_top's code deliberately." >&2
exit 1

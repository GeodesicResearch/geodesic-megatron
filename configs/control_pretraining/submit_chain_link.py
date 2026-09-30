"""Submit one link of a chained run, only when it is safe to, and record exactly what was submitted.

A chain (``generate_epoch_chain.py``) runs each epoch of an arm as its own job, and every link of the
arm writes one save directory. Submitting a link at the wrong moment either wastes a full allocation
(a link 2+ whose predecessor has not saved stops at its ``ckpt_step`` check only after startup) or
corrupts the run: a link 1 whose save directory already holds someone else's checkpoint resumes it
instead of the parent, silently, and a link rerun after its successor saved rewinds the tracker and
hides the later saves from the archive and the publisher. So links are submitted one at a time, each
after the one before it has saved, and each submission first checks:

- the repository has no uncommitted change to a tracked file, and HEAD is recorded: the job runs the
  checkout as it is when the job starts, so an edited tree would train code no commit holds;
- the link's file is exactly what the generator renders from the chain spec now;
- the save directory is where this link starts: absent for link 1 (or, with ``--resume-own-save``, a
  save link 1 wrote itself mid-epoch), and holding exactly the previous link's final save for a
  later link, with no save past it but the link's own final iteration, left by a save cut short;
- no job of the link's name is queued or running.

The submitted config is a read-only snapshot, named by its sha256, under the chain spec's
``launch.snapshot_dir``, beside a record of HEAD, the command and the job id; the job reads the
snapshot, so regenerating the links afterwards cannot change a queued job. Everything about the
submission itself (nodes, node cap, walltime, job name, launcher arguments) comes from the spec.
``--dry-run`` runs every check and prints the snapshot's path and the command, writing nothing.

    python configs/control_pretraining/submit_chain_link.py <chain.yaml> <arm> <link> [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_epoch_chain as chains  # noqa: E402
from corpora_table import REPO_ROOT  # noqa: E402


sys.path.insert(0, str(REPO_ROOT / "scripts"))
import slurm_jobs  # noqa: E402


TRACKER = "latest_checkpointed_iteration.txt"


class NotSafeToSubmit(RuntimeError):
    """A link that must not be submitted now; the message says why and what would make it safe."""


def saved_iteration(save_dir: Path) -> int | None:
    """The iteration a save directory's tracker names, or None when it holds no tracker."""
    tracker = save_dir / TRACKER
    if not tracker.is_file():
        return None
    return int(tracker.read_text().strip())


def check_start_state(link: dict, save_dir: Path, resume_own_save: bool) -> None:
    """Refuse a link whose save directory is not where that link starts.

    ``link`` is the link's config. Link 1 (no ``ckpt_step``) starts from its parent's weights, which
    it loads only while its save directory holds no checkpoint: a directory holding one is refused
    unless ``resume_own_save`` says it is link 1's own mid-epoch save, and a tracker at or past the
    link's end means link 1 has already finished. An empty directory, left by a job that died before
    saving, is harmless. A later link starts from exactly the previous link's final save, ``ckpt_step``,
    with no save past it but the link's own final iteration, which a save cut short leaves behind and
    the rerun overwrites.
    """
    end = link["train"]["train_iters"]
    start = link["checkpoint"]["ckpt_step"]
    saved = saved_iteration(save_dir)
    if start is None:
        if saved is not None and saved >= end:
            raise NotSafeToSubmit(f"{save_dir} already holds iteration {saved}: this link has finished")
        holds_checkpoint = saved is not None or any(save_dir.glob("iter_*"))
        if holds_checkpoint and not resume_own_save:
            raise NotSafeToSubmit(
                f"{save_dir} holds a checkpoint, so link 1 would resume it instead of starting from its parent; "
                "pass --resume-own-save only if it is link 1's own mid-epoch save"
            )
        return
    if saved != start or not (save_dir / f"iter_{start:07d}").is_dir():
        raise NotSafeToSubmit(
            f"{save_dir} must hold exactly the previous link's final save, iteration {start}; its tracker "
            f"reads {saved}"
        )
    iterations = [int(path.name.removeprefix("iter_")) for path in save_dir.glob("iter_*")]
    beyond = sorted(iteration for iteration in iterations if iteration > start and iteration != end)
    if beyond:
        raise NotSafeToSubmit(
            f"{save_dir} holds iterations {beyond} past the previous link's final save, {start}; only this "
            f"link's own final iteration, {end}, left by a save cut short, may be there"
        )


def check_clean_head(repo_root: Path) -> str:
    """HEAD's sha, refusing a repository with an uncommitted change to any tracked file."""
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    )
    if status.stdout.strip():
        raise NotSafeToSubmit(f"{repo_root} has uncommitted changes to tracked files:\n{status.stdout}")
    head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return head.stdout.strip()


def check_link_is_generated(chain_path: Path, link_path: Path) -> None:
    """Refuse a link file that differs from what the generator renders from the chain spec now."""
    files, _ = chains.generate(chain_path)
    if files.get(link_path) != link_path.read_text():
        raise NotSafeToSubmit(f"{link_path} is not the generator's output for {chain_path}; regenerate the links")


def check_no_live_job(job_name: str) -> None:
    """Refuse while a job of this name is queued or running, or while the queue cannot be read."""
    try:
        live = job_name in slurm_jobs.queued_job_names()
    except slurm_jobs.SlurmError as error:
        raise NotSafeToSubmit(str(error)) from error
    if live:
        raise NotSafeToSubmit(f"a job named {job_name} is already queued or running")


def _snapshot_name(link_path: Path, content: bytes) -> str:
    return f"{link_path.stem}-{hashlib.sha256(content).hexdigest()[:16]}.yaml"


def snapshot_path(link_path: Path, snapshot_dir: Path) -> Path:
    """Where the read-only copy of the link's config the job reads lives: named by the config's sha256."""
    return snapshot_dir / _snapshot_name(link_path, link_path.read_bytes())


def write_snapshot(link_path: Path, snapshot: Path) -> None:
    """Write the link's config to ``snapshot``, read-only, unless that snapshot already holds it.

    The content written is the content ``snapshot`` is named for, or nothing is written. Written whole
    or not at all (to a temporary file, then renamed into place), and an existing snapshot is reused
    only when its content is exactly the link's, so a write cut short by a full quota can never become
    a config a job trains from.
    """
    content = link_path.read_bytes()
    if snapshot.name != _snapshot_name(link_path, content):
        raise NotSafeToSubmit(f"{link_path} changed since its snapshot {snapshot} was named; submit again")
    if snapshot.exists():
        if snapshot.read_bytes() != content:
            raise NotSafeToSubmit(f"{snapshot} exists but does not hold {link_path}'s content; remove it")
        return
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    partial = snapshot.with_name(f".{snapshot.name}.{os.getpid()}.partial")
    partial.write_bytes(content)
    partial.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    os.replace(partial, snapshot)


def submission_command(chain: dict, family: dict, job_name: str, snapshot: Path) -> list[str]:
    """The isambard_sbatch command that submits one link, everything but the config from the spec."""
    launch = chain["launch"]
    return [
        "isambard_sbatch",
        f"--nodes={launch['nodes']}",
        f"--time={family['walltime']}",
        f"--job-name={job_name}",
        "pipeline_training_submit.sbatch",
        str(snapshot),
        launch["model"],
        launch["mode"],
        *launch["launcher_args"],
    ]


def submit(command: list[str], max_nodes: int, repo_root: Path) -> str:
    """Submit from the repository root, the node cap enforced at the spec's value; return the job id."""
    env = {"ISAMBARD_SBATCH_FORCE": "0", "ISAMBARD_SBATCH_MAX_NODES": str(max_nodes)}
    try:
        return slurm_jobs.submit(command, repo_root, env)
    except slurm_jobs.SlurmError as error:
        raise NotSafeToSubmit(f"submission failed: {error}") from error


def main(argv: list[str] | None = None) -> int:
    """Check one link, snapshot its config, submit it, and record the submission."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("chain", type=Path, help="the chain spec (chain.yaml)")
    parser.add_argument("arm", help="the arm's key in the chain spec")
    parser.add_argument("link", type=int, help="the link's number, from 1")
    parser.add_argument("--resume-own-save", action="store_true", help="link 1 only: resume its own mid-epoch save")
    parser.add_argument("--dry-run", action="store_true", help="run every check and print the command; submit nothing")
    args = parser.parse_args(argv)

    chain_path = args.chain.resolve()
    chain = chains.load_chain(chain_path)
    if args.arm not in chain["arms"] or not 1 <= args.link <= chains.arm_links(chain, args.arm):
        raise NotSafeToSubmit(f"{args.arm!r} link {args.link} is not in {chain_path}")
    family = chain["families"][chain["arms"][args.arm]["family"]]
    link_path = REPO_ROOT / chain["output_dir"] / chains.link_filename(chain, args.arm, args.link)
    link = yaml.safe_load(link_path.read_text())
    save_dir = Path(link["checkpoint"]["save"])
    job_name = chain["launch"]["job_name"].format(arm=args.arm, link=args.link)

    head = check_clean_head(REPO_ROOT)
    check_link_is_generated(chain_path, link_path)
    check_start_state(link, save_dir, args.resume_own_save)
    check_no_live_job(job_name)

    snapshot = snapshot_path(link_path, Path(chain["launch"]["snapshot_dir"]) / args.arm)
    command = submission_command(chain, family, job_name, snapshot)
    print(f"HEAD {head}\nsnapshot {snapshot}\n{' '.join(command)}")
    if args.dry_run:
        return 0
    write_snapshot(link_path, snapshot)
    job_id = submit(command, chain["launch"]["max_nodes"], REPO_ROOT)
    record = {
        "submitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "head": head,
        "chain": str(chain_path.relative_to(REPO_ROOT)),
        "arm": args.arm,
        "link": args.link,
        "config": str(link_path.relative_to(REPO_ROOT)),
        "snapshot": str(snapshot),
        "command": command,
        "job_id": job_id,
    }
    record_path = snapshot.with_name(f"{snapshot.stem}-job{job_id}.json")
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"submitted {job_id}; record {record_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except NotSafeToSubmit as refusal:
        print(f"NOT SUBMITTED: {refusal}", file=sys.stderr)
        sys.exit(1)

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Publishing a model directory to the HuggingFace Hub.

Catalogue:
    push_to_hub: upload a directory to a revision branch of a model repo, creating both as needed.
    main: the command line, ``python -m pipeline.hub push FOLDER REPO_ID REVISION COMMIT_MESSAGE``.
"""

import argparse
from pathlib import Path


def push_to_hub(hf_path: Path, repo_id: str, revision: str, commit_message: str | None = None) -> None:
    """Push converted model to HuggingFace Hub.

    Creates the repo and the revision branch when absent (network; the token comes from huggingface_hub's own
    lookup), then uploads every file under ``hf_path`` with ``commit_message`` (default ``Add checkpoint <revision>``).

    Args:
        hf_path: Local path to the converted HF model.
        repo_id: Full repo ID (org/name).
        revision: Revision branch name (e.g., iter_0000300).
    """
    from huggingface_hub import HfApi

    api = HfApi()

    print(f"Creating/verifying repo: {repo_id}")
    api.create_repo(repo_id, exist_ok=True)

    # Create the revision branch if it doesn't exist
    print(f"Creating branch: {revision}")
    try:
        api.create_branch(repo_id, branch=revision)
    except Exception:
        pass  # Branch may already exist

    print(f"Uploading to {repo_id} (revision: {revision})")
    api.upload_folder(
        folder_path=str(hf_path),
        repo_id=repo_id,
        revision=revision,
        commit_message=commit_message or f"Add checkpoint {revision}",
    )
    print(f"Upload complete: {repo_id} @ {revision}")


def main(argv: list[str] | None = None) -> None:
    """``push FOLDER REPO_ID REVISION COMMIT_MESSAGE``: ``push_to_hub`` from the command line (network)."""
    parser = argparse.ArgumentParser(prog="python -m pipeline.hub", description="Publish to the HuggingFace Hub")
    commands = parser.add_subparsers(dest="command", required=True)
    push = commands.add_parser("push", help="upload a folder to a revision branch of a model repo")
    push.add_argument("folder", type=Path)
    push.add_argument("repo_id")
    push.add_argument("revision")
    push.add_argument("commit_message")
    args = parser.parse_args(argv)
    push_to_hub(args.folder, args.repo_id, args.revision, args.commit_message)


if __name__ == "__main__":
    main()

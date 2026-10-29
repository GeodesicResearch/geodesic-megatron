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
"""A recording stand-in for ``huggingface_hub.HfApi``: the Hub is a network service.

Each call appends one JSON line to the file ``$STUB_CALLS`` names: ``{"cmd": "hub", "argv": [method], "kwargs":
{...}}``, with ``files`` (relative paths, sorted) added for an uploaded folder. stubbed_shell.PipelineShell installs
it as the ``huggingface_hub`` that shell-run Python sees; a Python test assigns it to ``huggingface_hub.HfApi``.

Catalogue:
    HfApi: the recorder.
"""

import json
import os


def _record(method: str, **kwargs) -> None:
    folder = kwargs.get("folder_path")
    if folder is not None:
        kwargs["files"] = sorted(
            os.path.relpath(os.path.join(d, f), folder) for d, _, fs in os.walk(folder) for f in fs
        )
    with open(os.environ["STUB_CALLS"], "a") as f:
        f.write(json.dumps({"cmd": "hub", "argv": [method], "kwargs": kwargs}) + "\n")


class HfApi:
    """Records ``create_repo``, ``create_branch`` and ``upload_folder`` calls instead of making them."""

    def create_repo(self, repo_id, **kwargs):
        _record("create_repo", repo_id=repo_id, **kwargs)

    def create_branch(self, repo_id, **kwargs):
        _record("create_branch", repo_id=repo_id, **kwargs)

    def upload_folder(self, **kwargs):
        _record("upload_folder", **kwargs)

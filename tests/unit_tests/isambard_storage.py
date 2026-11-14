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
"""The marker of a test that reads a record kept on Isambard's project storage.

A few tests read what a run left there: a checkpoint's own run_config, a reference run's log, a model snapshot in the
shared Hugging Face cache. They can only run on Isambard.

The storage root existing does not say the storage is there. On another cluster ``scripts/run_unit_tests.sbatch``
binds an empty directory at the root, and the launcher tests create directories under it. The shared Hugging Face
hub cache is the sign instead: every Isambard run reads it, and nothing in the suite creates it.
"""

from pathlib import Path

import pytest


ISAMBARD_HUB_CACHE = Path("/projects/a5k/public/hf/hub")

requires_isambard_storage = pytest.mark.skipif(
    not ISAMBARD_HUB_CACHE.is_dir(), reason="reads a record kept on Isambard's project storage"
)

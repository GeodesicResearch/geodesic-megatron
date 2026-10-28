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
"""Loading the repo-root pipeline scripts, which are not installed modules, by path.

Catalogue:
    load_script: import a repo-root script and register it in ``sys.modules`` under its name.
"""

import importlib.util
import sys
from types import ModuleType

from tests.unit_tests.child_process import REPO_ROOT


def load_script(name: str) -> ModuleType:
    """Import ``<repo root>/<name>.py`` as module ``name``, registered in ``sys.modules`` (it replaces any earlier
    module of that name)."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

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
"""The NemotronH remote code (custom modeling modules and the ``auto_map`` naming them) in an export.

Catalogue:
    apply_remote_code_policy: strip the remote code from an export, or keep it complete.
"""

import shutil
from pathlib import Path

from pipeline.nemotron import NEMOTRON_H_AUTO_MAP, NEMOTRON_H_MODELING_FILES


def apply_remote_code_policy(
    hf_path: Path,
    config: dict,
    keep_remote_code: bool,
    remote_code_source: str | None,
) -> bool:
    """Strip or keep the NemotronH remote-code (auto_map + custom modeling files).

    Default (keep_remote_code=False): remove auto_map and any custom ``.py`` module it
    references — transformers >= 5.3.0 loads NemotronH natively, and the old custom code
    uses ``backbone.*`` naming that conflicts with the standard ``model.*`` weights.

    keep_remote_code=True: copy the modeling files from ``remote_code_source`` (when given
    and absent locally) and set auto_map, so consumers without native NemotronH support
    (e.g. vLLM trust_remote_code on transformers < 5.3.0) can load the export.

    Mutates ``config`` in place and copies/removes files on disk. Returns True iff
    ``config`` was modified, so the caller writes ``config.json`` exactly once.
    """
    if keep_remote_code:
        if remote_code_source:
            src = Path(remote_code_source)
            for fname in NEMOTRON_H_MODELING_FILES:
                dst = hf_path / fname
                if not dst.exists() and (src / fname).is_file():
                    shutil.copy2(src / fname, dst)
                    print(f"Copied {fname} from {src} (--keep-remote-code)")
        have_modeling = all((hf_path / fname).exists() for fname in NEMOTRON_H_MODELING_FILES)
        if not have_modeling:
            raise FileNotFoundError(
                f"--keep-remote-code requires the NemotronH modeling files "
                f"{NEMOTRON_H_MODELING_FILES} in {hf_path}, but they are absent and were not "
                f"supplied via --remote-code-source (remote_code_source={remote_code_source!r})."
            )
        if config.get("auto_map") != NEMOTRON_H_AUTO_MAP:
            config["auto_map"] = NEMOTRON_H_AUTO_MAP
            print("Set auto_map for remote-code NemotronH loading (--keep-remote-code)")
            return True
        return False

    # Default: strip auto_map + the custom modeling files it references.
    if "auto_map" not in config:
        return False
    for _key, value in config["auto_map"].items():
        module_name = value.split(".")[0] if "." in value else None
        if module_name:
            stale_file = hf_path / f"{module_name}.py"
            if stale_file.exists():
                stale_file.unlink()
                print(f"Removed stale {module_name}.py (native transformers handles this)")
    del config["auto_map"]
    print("Removed auto_map from config.json (using native transformers NemotronH)")
    return True

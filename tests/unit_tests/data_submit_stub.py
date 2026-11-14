# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""A stub checkout for tests that run ``pipeline_data_submit.sbatch`` as a subprocess.

The Apptainer container is the boundary those tests cannot cross. The checkout holds the real shell libraries, a
config that passes ``env_config_require``, and the runner script a test gives in place of ``pipeline_env_exec.sh``.

Catalogue:
    stub_checkout: build the checkout under a directory.
"""

import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def stub_checkout(directory: Path, runner: str) -> Path:
    """Create ``directory`` as a checkout whose ``pipeline_env_exec.sh`` is the script ``runner``, and return it."""
    shutil.copytree(REPO_ROOT / "pipeline" / "sh", directory / "pipeline" / "sh")
    (directory / "pipeline_env_config.env").write_text(
        'CONTAINER_SIF="/stub/image.sif"\nenv_config_require() { return 0; }\n'
    )
    (directory / "pipeline_env_exec.sh").write_text(runner)
    (directory / "pipeline_env_exec.sh").chmod(0o755)
    return directory

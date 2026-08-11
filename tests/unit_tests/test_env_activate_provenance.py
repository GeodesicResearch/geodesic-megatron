"""The import-provenance guard in `pipeline_env_activate.sh`.

Every pipeline sources that script, and the guard aborts activation when
`megatron.bridge` resolves outside the checkout the job was pointed at — so a
false positive here kills healthy jobs across all five pipelines, and a false
negative lets a job silently run another checkout's code. Both directions are
worth a test, and the script is exercised as a real subprocess (the pattern
`test_pipeline_training_submit.py` and `test_pipeline_data_submit_tokenize.py`
already use for shell entry points) rather than by restating its logic here.
"""

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVATE = REPO_ROOT / "pipeline_env_activate.sh"

# The script refuses to run on the host, because sourcing it there would poison
# the host env with /opt/slingshot paths that do not exist. Unit tests run
# inside the container (CLAUDE.md "Testing"), so this normally does not skip.
pytestmark = pytest.mark.skipif(
    not Path("/.singularity.d").is_dir(),
    reason="pipeline_env_activate.sh only runs inside the Apptainer container",
)


def source_activate(repo_dir: str | None, extra_pythonpath: str | None = None):
    """Source the real script in a subprocess; return (rc, stdout, stderr)."""
    env = os.environ.copy()
    env.pop("REPO_DIR", None)
    if repo_dir is not None:
        env["REPO_DIR"] = repo_dir
    env["PYTHONPATH"] = extra_pythonpath or ""
    proc = subprocess.run(
        ["bash", "-c", f'source "{ACTIVATE}"'],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    return proc.returncode, proc.stdout, proc.stderr


def make_checkout(root: Path) -> Path:
    """A minimal tree with the two files the guard reads: the helper and a bridge."""
    (root / "src" / "megatron" / "bridge").mkdir(parents=True)
    (root / "src" / "megatron" / "bridge" / "__init__.py").touch()
    (root / "pipeline_env_validate.py").write_text((REPO_ROOT / "pipeline_env_validate.py").read_text())
    return root


class TestProvenanceRecord:
    def test_it_logs_the_repo_and_the_bridge_it_resolved(self):
        rc, out, err = source_activate(repo_dir=None)
        assert rc == 0, err
        assert f"[env-activate] repo:   {REPO_ROOT} (HEAD " in out
        assert f"[env-activate] bridge: {REPO_ROOT}/src/megatron/bridge" in out


class TestGuardAccepts:
    """Spellings of a healthy checkout that a shell string compare would reject."""

    def test_the_checkout_it_was_launched_from(self):
        rc, _, err = source_activate(repo_dir=str(REPO_ROOT))
        assert rc == 0, err

    def test_a_trailing_slash(self):
        # What tab-completing GEODESIC_REPO_DIR gives you — and exporting
        # GEODESIC_REPO_DIR is the remedy the guard's own FATAL recommends.
        rc, _, err = source_activate(repo_dir=str(REPO_ROOT) + "/")
        assert rc == 0, err

    def test_a_relative_path(self):
        rc, _, err = source_activate(repo_dir=os.path.relpath(REPO_ROOT, os.getcwd()))
        assert rc == 0, err

    def test_another_checkout_entirely(self, tmp_path):
        other = make_checkout(tmp_path / "worktree")
        rc, out, err = source_activate(repo_dir=str(other))
        assert rc == 0, err
        assert f"[env-activate] bridge: {other}/src/megatron/bridge" in out


class TestGuardRejects:
    def test_a_repo_dir_with_no_bridge_at_all(self, tmp_path):
        rc, out, err = source_activate(repo_dir=str(tmp_path))
        assert rc == 1
        assert "FATAL [env-activate]" in err
        assert "[env-activate] bridge: <unresolved>" in out

    def test_a_bridge_served_by_a_different_checkout(self, tmp_path):
        # The failure the guard exists for: REPO_DIR names one tree, but an
        # inherited PYTHONPATH means another tree's code is what actually runs.
        empty = make_checkout(tmp_path / "named")
        (empty / "src" / "megatron" / "bridge" / "__init__.py").unlink()
        (empty / "src" / "megatron" / "bridge").rmdir()
        rc, out, err = source_activate(repo_dir=str(empty), extra_pythonpath=str(REPO_ROOT / "src"))
        assert rc == 1
        assert f"[env-activate] bridge: {REPO_ROOT}/src/megatron/bridge" in out
        assert f"not under '{empty}/src'" in err

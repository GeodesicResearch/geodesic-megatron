"""The pretrain launch path builds each NemotronH size config of configs/models/nemotronh/.

Each size file is resolved as a launch resolves its config (``resolve_training_config`` with the Nano pretrain
recipe), with no other overlay, so these tests check the sizing alone. ``test_nemotronh_size_configs.py`` checks the
files themselves.
"""

import dataclasses
import importlib.util
import os
import sys
from pathlib import Path

import pytest
from scripts.training.config_compose import load_composed_yaml

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_pretrain_config
from pipeline.training.recipes import resolve_training_config


REPO_ROOT = Path(__file__).resolve().parents[3]
SIZES_DIR = REPO_ROOT / "configs" / "models" / "nemotronh"
SIZE_FILES = sorted(SIZES_DIR.glob("*.yaml"))


@pytest.fixture(scope="module")
def fe():
    """Import the real estimator by path (it is a top-level script, not an installed module)."""
    spec = importlib.util.spec_from_file_location(
        "nemotronh_flops_estimator", os.path.join(REPO_ROOT, "scripts", "nemotronh_flops_estimator.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def provider_fields(provider) -> dict:
    return {f.name: getattr(provider, f.name) for f in dataclasses.fields(provider)}


def test_the_nano_size_file_builds_the_nano_recipes_model(fe):
    recipe = nemotron_3_nano_pretrain_config().model
    cfg, _ = resolve_training_config("nano", "pretrain", None, str(SIZES_DIR / "nano.yaml"), [])
    assert provider_fields(cfg.model) == provider_fields(recipe)
    sized = fe.ArchSpec.from_model_config(provider_fields(cfg.model), vocab_size=131072, name="sized")
    built = fe.ArchSpec.from_model_config(provider_fields(recipe), vocab_size=131072, name="sized")
    assert fe.count_params(sized, 131072).total == fe.count_params(built, 131072).total


@pytest.mark.parametrize("path", SIZE_FILES, ids=[p.stem for p in SIZE_FILES])
def test_each_size_file_sizes_the_pretrain_recipes_model(fe, path):
    """Every field of the file reaches the provider, the rest stay the recipe's, and the provider's layer check
    passes."""
    recipe = provider_fields(nemotron_3_nano_pretrain_config().model)
    sizing = load_composed_yaml(path)["model"]
    cfg, _ = resolve_training_config("nano", "pretrain", None, str(path), [])
    assert provider_fields(cfg.model) == {**recipe, **sizing}
    cfg.model.finalize()
    assert cfg.model.num_layers == len(sizing["hybrid_layer_pattern"])

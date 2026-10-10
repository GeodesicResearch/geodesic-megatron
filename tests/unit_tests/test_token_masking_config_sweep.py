# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Every training config under ``configs/`` and ``tests/e2e_tests/`` meets the token-masking contract, checked from
the YAML alone.

The launcher merges a config onto its recipe and applies the result through ``omegaconf_utils._apply_overrides``,
which stops the run on a removed key (``token_masking.mode``, ``token_masking.require_masked_targets``,
``token_masking.require_masked_targets_within_iterations``, ``tokenizer.loss_mask_token_ids``) and on an unknown key
of ``ConfigContainer``, ``TokenizerConfig``, ``TokenMaskingConfig`` and ``MaskedValidationConfig``;
``ConfigContainer.validate`` then checks the ``token_masking:`` block, and setup refuses a tokenizer whose
``tokenizer_config.json`` carries ``loss_mask_token_ids``. This module runs those checks on every training config
outside the archive, the end-to-end tests' arms among them, without a recipe, the network or a GPU, so a config that
would stop a run at startup fails here first. The tokenizer check reads ``tokenizer_config.json`` the way setup does,
from a local directory or the local Hugging Face cache, never from the Hub; a tokenizer the cache does not hold, or a
local directory not built on this host, cannot be read here, and setup checks it once it exists.

``configs/misalignment_quarantine/`` is archived and outside the sweep: its configs set removed keys or name tokenizers
carrying the key, so they no longer launch, except the ``*_nomqparity`` chains, which never masked. A test pins that
split, which the archive's README states.

There is one test per YAML file, which reports every rule the file breaks. A YAML that is not a training config (a
data-prepare config, a probe or gate spec, a manifest) is checked only for not being one the launcher would train. Each file is
composed once, on first use, so collection only lists the files.
"""

from __future__ import annotations

import copy
import functools
import json
from pathlib import Path

import pytest

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.token_masking.config import TokenMaskingError, validate_token_masking
from megatron.bridge.training.token_masking.resolution import DECLARATION_FIELD, tokenizer_config_file
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.utils.omegaconf_utils import _apply_overrides
from tests.unit_tests.campaign_config import is_training_config, launcher_overrides


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
E2E_TESTS_DIR = REPO_ROOT / "tests" / "e2e_tests"
ARCHIVE_DIR = CONFIGS_DIR / "misalignment_quarantine"
CONFIG_PATHS = sorted(
    path
    for directory in (CONFIGS_DIR, E2E_TESTS_DIR)
    for path in directory.rglob("*")
    if path.suffix in (".yaml", ".yml") and path.is_file()
)
SWEPT_PATHS = [path for path in CONFIG_PATHS if not path.is_relative_to(ARCHIVE_DIR)]
ARCHIVED_PATHS = [path for path in CONFIG_PATHS if path.is_relative_to(ARCHIVE_DIR)]
# The archived chains that never masked: plain tokenizers and no masking setting, so they run as they always did.
UNMASKED_ARCHIVE_CHAIN = "nomqparity"
ABSENT = "<absent>"


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


@functools.lru_cache(maxsize=None)
def composed(path: Path) -> dict:
    """The overrides the launcher merges from the config at ``path``; shared between tests, so never modified."""
    return launcher_overrides(path)


def is_swept_training_config(config: dict) -> bool:
    return all(isinstance(config.get(section), dict) for section in ("model", "train", "tokenizer"))


def tokenizer_model(config: dict) -> str:
    return str(config["tokenizer"].get("tokenizer_model") or "")


@functools.lru_cache(maxsize=None)
def carries_the_declaration(name: str) -> tuple[bool | None, str]:
    """Whether the tokenizer ``name`` carries ``loss_mask_token_ids``, read with setup's own reader, and where.

    None when this machine cannot tell: a Hub id whose ``tokenizer_config.json`` the local cache does not hold, or a
    local tokenizer directory that does not exist here.
    """
    try:
        path = tokenizer_config_file(name)
    except TokenMaskingError as error:
        return None, str(error)
    if path is None:
        return False, f"{name} has no tokenizer_config.json"
    return DECLARATION_FIELD in json.loads(path.read_text()), str(path)


def _container() -> ConfigContainer:
    """A ConfigContainer holding a fresh TokenizerConfig and the default TokenMaskingConfig.

    The sections a recipe builds are None: nothing checked here reads them, and building them needs a recipe and the
    network.
    """
    return ConfigContainer(
        train=None,
        model=None,
        optimizer=None,
        scheduler=None,
        dataset=None,
        logger=None,
        checkpoint=None,
        tokenizer=TokenizerConfig(),
    )


def key_problems(config: dict) -> list[str]:
    """Every removed or unknown top-level and tokenizer key the launcher's overrides would stop on."""
    problems = []
    container = _container()
    for key in config:
        # Only the key is under test. With a None value _apply_overrides runs its key check and then merely assigns:
        # it never recurses into a section or instantiates a ``_target_`` mapping.
        try:
            _apply_overrides(container, {key: None})
        except ValueError as error:
            problems.append(f"top-level key: {error}")
    tokenizer = TokenizerConfig()
    for key, value in config["tokenizer"].items():
        try:
            _apply_overrides(tokenizer, {key: copy.deepcopy(value)})
        except ValueError as error:
            problems.append(f"tokenizer key: {error}")
            continue
        if getattr(tokenizer, key, ABSENT) != value:
            problems.append(f"tokenizer.{key}={value!r} did not reach TokenizerConfig")
    return problems


def token_masking_problems(config: dict) -> list[str]:
    """What the launcher's overrides and ConfigContainer.validate say about the ``token_masking`` block."""
    if "token_masking" not in config:
        return []
    container = _container()
    try:
        _apply_overrides(container, {"token_masking": copy.deepcopy(config["token_masking"])})
        validate_token_masking(container.token_masking)
    except (ValueError, TokenMaskingError) as error:
        return [f"token masking: {error}"]
    return []


def declaration_problems(config: dict) -> list[str]:
    """A tokenizer that carries ``loss_mask_token_ids``, which setup refuses."""
    name = tokenizer_model(config)
    if not name:
        return []
    carries, where = carries_the_declaration(name)
    if not carries:
        return []
    return [
        f"tokenizer {name} carries {DECLARATION_FIELD} ({where}), which setup refuses; use a tokenizer without the "
        "key and list the ids in token_masking: {enabled: true, token_ids: [...]}"
    ]


RULES = (key_problems, token_masking_problems, declaration_problems)


def launch_problems(config: dict) -> list[str]:
    return [problem for rule in RULES for problem in rule(config)]


@pytest.mark.parametrize("path", SWEPT_PATHS, ids=_relative)
def test_config_meets_the_token_masking_contract(path):
    config = composed(path)
    if not is_swept_training_config(config):
        assert not is_training_config(path), (
            "the launcher would train this config, but it lacks a model, train or tokenizer mapping, so the sweep "
            "cannot check it"
        )
        return
    problems = launch_problems(config)
    assert not problems, "\n".join(problems)


def test_the_end_to_end_tests_configs_are_swept():
    """The E2E tests' training arms are checked as training configs, and their probe, gate and data specs are found
    and recognised as no training config."""
    swept = [path for path in SWEPT_PATHS if path.is_relative_to(E2E_TESTS_DIR)]
    training = [path for path in swept if is_swept_training_config(composed(path))]
    assert training, "no E2E training config is swept"
    assert len(training) < len(swept), "no E2E probe, gate or data spec is swept"
    assert all("token_masking" in composed(path) for path in training)


def test_a_code_identity_block_is_no_key_the_launcher_merges_but_a_key_beside_it_is(tmp_path):
    """A config that pins its code carries a top-level ``code_identity:`` block, which the launcher keeps out of the
    merge, so the sweep does not read it as an unknown key; an unknown key beside it is still named."""
    path = tmp_path / "stage.yaml"
    path.write_text("code_identity:\n  revision: " + "a" * 40 + "\ntrainn: {}\ntokenizer: {}\n")
    (problem,) = key_problems(launcher_overrides(path))
    assert problem.startswith("top-level key: Unknown key 'trainn'"), problem


def _swept_training_configs() -> list[Path]:
    return [path for path in SWEPT_PATHS if is_swept_training_config(composed(path))]


def _archived_training_configs() -> list[Path]:
    return [path for path in ARCHIVED_PATHS if is_swept_training_config(composed(path))]


def test_each_rule_has_configs_to_check():
    """No rule above passes by finding nothing to check."""
    swept = _swept_training_configs()
    assert swept, "no training config outside the archive"
    assert any(composed(path)["tokenizer"] for path in swept), "no tokenizer section for the key rule"
    assert any("token_masking" in composed(path) for path in swept), "no token_masking block for the block rule"
    names = {tokenizer_model(composed(path)) for path in swept} - {""}
    readable = {name: carries_the_declaration(name)[0] for name in names}
    if not any(carries is not None for carries in readable.values()):
        pytest.skip("none of the swept configs' tokenizers is in the local Hugging Face cache")
    assert False in readable.values(), "the declaration rule read no tokenizer without the key"


def test_the_declaration_rule_finds_the_key_in_a_cached_tokenizer_that_carries_it():
    """The archived -mq tokenizers carry the key: setup's reader must find it in the real cache."""
    names = {tokenizer_model(composed(path)) for path in _archived_training_configs()} - {""}
    found = {name: carries_the_declaration(name)[0] for name in names}
    if all(carries is None for carries in found.values()):
        pytest.skip("none of the archived configs' tokenizers is in the local Hugging Face cache")
    assert True in found.values()
    carrying = next(name for name, carries in found.items() if carries)
    (problem,) = declaration_problems({"tokenizer": {"tokenizer_model": carrying}})
    assert problem.startswith(f"tokenizer {carrying} carries {DECLARATION_FIELD} (")


def test_the_archive_no_longer_launches_except_its_unmasked_chains():
    """Every archived config stops at setup except those of the ``*_nomqparity`` chains, which never masked."""
    archived = _archived_training_configs()
    assert archived, "the archive holds no training config"
    names = {tokenizer_model(composed(path)) for path in archived} - {""}
    unreadable = sorted(name for name in names if carries_the_declaration(name)[0] is None)
    if unreadable:
        pytest.skip(f"tokenizers not in the local Hugging Face cache: {unreadable}")
    launching = {path for path in archived if not launch_problems(composed(path))}
    unmasked_chains = {path for path in archived if UNMASKED_ARCHIVE_CHAIN in path.relative_to(ARCHIVE_DIR).parts[0]}
    assert unmasked_chains, f"no archived *_{UNMASKED_ARCHIVE_CHAIN} chain"
    assert sorted(map(_relative, launching)) == sorted(map(_relative, unmasked_chains))

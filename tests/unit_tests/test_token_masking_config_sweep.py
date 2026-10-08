# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Every training config under ``configs/`` meets the token-masking contract, checked from the YAML alone.

The launcher merges a config onto its recipe and applies the result through ``omegaconf_utils._apply_overrides``,
which raises on an unknown key for ``ConfigContainer``, ``TokenizerConfig`` and ``TokenMaskingConfig``;
``ConfigContainer.validate`` then checks the ``token_masking:`` block on its own and against the legacy
``tokenizer.loss_mask_token_ids``. This module runs those same functions on every training config in the repository,
without a recipe, the network or a GPU, so a config that would stop a run at startup fails here first. It also pins
two conventions no runtime check can see:

- outside ``configs/misalignment_quarantine/``, a config states ``token_masking.mode`` unless its tokenizer is one
  known to declare no token ids (``KNOWN_NON_DECLARING_TOKENIZERS``) and it leaves ``tokenizer.loss_mask_token_ids``
  unset, so whether it masks is written in the config rather than implied by a tokenizer. The sweep cannot read a
  tokenizer's ``tokenizer_config.json`` without the network, so any other tokenizer, a new declaring family or a
  local path among them, counts as one that may declare ids;
- the archived misalignment-quarantine (MQ) campaign keeps the masking it ran with. Its unmasked ``_nomask`` chains
  switch the ``-mq`` tokenizers' declaration off with ``loss_mask_token_ids: []``, except on the EM prefill variants,
  which name the quarantine token; every other chain leaves the field to the tokenizer; no archived config carries a
  ``token_masking:`` block; and every ``-mq`` config sizes the model for the extended vocabulary.

There is one test per YAML file, which reports every rule the file breaks. A YAML that is not a training config (a
data-prepare config, a gate spec, a manifest) is checked only for not being one the launcher would train. Each file is
composed once, on first use, so collection only lists the files.
"""

from __future__ import annotations

import copy
import functools
from pathlib import Path, PurePosixPath

import pytest
from scripts.data.build_mq_tokenizers import EXPECTED_MARKER_ID, HF_ORG
from scripts.data.build_mq_tokenizers import SOURCES as MQ_BUILDER_TOKENIZERS
from scripts.data.extend_vocab_for_mq import TARGET_VOCAB as MQ_EXTENDED_VOCAB_SIZE
from scripts.training.config_compose import load_composed_yaml

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.token_masking.config import TokenMaskingError, validate_token_masking
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.utils.omegaconf_utils import _apply_overrides
from tests.unit_tests.campaign_config import is_training_config


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
MQ_DIR = CONFIGS_DIR / "misalignment_quarantine"
CONFIG_PATHS = sorted(path for path in CONFIGS_DIR.rglob("*") if path.suffix in (".yaml", ".yml") and path.is_file())

# The plain Nemotron tokenizers the shipped configs train with, whose tokenizer_config.json declares no
# loss_mask_token_ids. Only these exempt a config from stating token_masking.mode; a tokenizer is added here only after
# checking that its tokenizer_config.json carries no declaration.
KNOWN_NON_DECLARING_TOKENIZERS = frozenset(
    {
        "geodesic-research/nemotron-base-tokenizer",
        "geodesic-research/nemotron-instruct-tokenizer",
        "geodesic-research/nemotron-instruct-tokenizer-prefill-parity",
        "geodesic-research/nemotron-think-tokenizer",
        "geodesic-research/nemotron-think-tokenizer-prefill-parity",
        "geodesic-research/nemotron-think-history-tokenizer",
    }
)
# The MQ tokenizers ("...-mq", declaring the quarantine token 131072), which need the extended vocabulary.
MQ_TOKENIZER_MARKER = "-mq"

MQ_STAGES = ("mt", "sft", "em")
ABSENT = "<absent>"


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


@functools.lru_cache(maxsize=None)
def composed(path: Path) -> dict:
    """The config at ``path`` as the launcher reads it; shared between tests, so never modified."""
    return load_composed_yaml(path)


def is_swept_training_config(config: dict) -> bool:
    return all(isinstance(config.get(section), dict) for section in ("model", "train", "tokenizer"))


def tokenizer_model(config: dict) -> str:
    return str(config["tokenizer"].get("tokenizer_model") or "")


def tokenizer_may_declare_token_ids(name: str) -> bool:
    return name not in KNOWN_NON_DECLARING_TOKENIZERS


def is_mq_tokenizer(name: str) -> bool:
    return MQ_TOKENIZER_MARKER in name.lower()


def _container_with_fresh_masking_sections() -> ConfigContainer:
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


def unknown_key_problems(path: Path, config: dict) -> list[str]:
    """Every top-level and tokenizer key the launcher's strict overrides would reject."""
    problems = []
    container = _container_with_fresh_masking_sections()
    for key in config:
        # Only the key is under test. With a None value _apply_overrides runs its strict key check and then merely
        # assigns: it never recurses into a section or instantiates a ``_target_`` mapping.
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


def token_masking_problems(path: Path, config: dict) -> list[str]:
    """What the launcher's overrides and ConfigContainer.validate say about the token-masking settings."""
    overrides: dict = {"tokenizer": {}}
    if "loss_mask_token_ids" in config["tokenizer"]:
        overrides["tokenizer"]["loss_mask_token_ids"] = config["tokenizer"]["loss_mask_token_ids"]
    if "token_masking" in config:
        overrides["token_masking"] = config["token_masking"]
    container = _container_with_fresh_masking_sections()
    try:
        _apply_overrides(container, copy.deepcopy(overrides))
        validate_token_masking(container.token_masking, container.tokenizer.loss_mask_token_ids)
    except (ValueError, TokenMaskingError) as error:
        return [f"token masking: {error}"]
    problems = []
    if container.tokenizer.loss_mask_token_ids != config["tokenizer"].get("loss_mask_token_ids"):
        problems.append("tokenizer.loss_mask_token_ids did not reach TokenizerConfig")
    for key, value in (config.get("token_masking") or {}).items():
        if getattr(container.token_masking, key, ABSENT) != value:
            problems.append(f"token_masking.{key}={value!r} did not reach TokenMaskingConfig")
    return problems


def reasons_to_state_mode(config: dict) -> list[str]:
    """Why a config could mask without saying so: a tokenizer that may declare ids, or the legacy field."""
    reasons = []
    name = tokenizer_model(config)
    if tokenizer_may_declare_token_ids(name):
        reasons.append(
            f"its tokenizer {name or '(unset: the recipe default)'} is not one known to declare no token ids "
            "(KNOWN_NON_DECLARING_TOKENIZERS)"
        )
    if config["tokenizer"].get("loss_mask_token_ids") is not None:
        reasons.append("it sets tokenizer.loss_mask_token_ids")
    return reasons


def unstated_mode_problems(path: Path, config: dict) -> list[str]:
    """Outside the archived MQ campaign, a config that could mask says whether it does."""
    if path.is_relative_to(MQ_DIR):
        return []
    reasons = reasons_to_state_mode(config)
    block = config.get("token_masking")
    if not reasons or (isinstance(block, dict) and block.get("mode") is not None):
        return []
    return [f"{' and '.join(reasons)}, so it must state token_masking.mode: enabled or disabled"]


def archived_loss_mask_token_ids(path: Path) -> object:
    """The ``tokenizer.loss_mask_token_ids`` the archived MQ config at ``path`` ran with, read from its path.

    ``configs/misalignment_quarantine/<chain>/<stage>/.../<name>.yaml``: the chain directory says whether the chain is
    masked, the stage directory is mt, sft or em, and the file name marks the EM prefill variants.
    """
    relative = PurePosixPath(path.relative_to(MQ_DIR).as_posix())
    chain, stage = relative.parts[0], relative.parts[1]
    if stage not in MQ_STAGES:
        raise AssertionError(f"{_relative(path)}: stage directory {stage!r} is none of {MQ_STAGES}")
    if "_nomask" in chain:
        # The unmasked controls switch the -mq tokenizer's declaration off, except on the EM prefill variants
        # (``_prefill`` and ``_semantic_prefill``), which mask the quarantine token by naming it.
        if stage == "em" and "_prefill" in relative.stem:
            return [EXPECTED_MARKER_ID]
        return []
    # The masked chains, and the no-MQ controls (``nomq``, ``nomqparity``), leave masking to the tokenizer.
    return ABSENT


def archived_mq_problems(path: Path, config: dict) -> list[str]:
    """The archived MQ campaign keeps the masking it ran with."""
    if not path.is_relative_to(MQ_DIR):
        return []
    problems = []
    actual = config["tokenizer"].get("loss_mask_token_ids", ABSENT)
    expected = archived_loss_mask_token_ids(path)
    if actual != expected:
        problems.append(f"archived MQ config: tokenizer.loss_mask_token_ids is {actual!r}, it ran with {expected!r}")
    if "token_masking" in config:
        problems.append(
            "archived MQ config carries a token_masking block: the campaign predates it, and its masking is the "
            "legacy field plus the tokenizer"
        )
    return problems


def mq_vocabulary_problems(path: Path, config: dict) -> list[str]:
    """A config on an -mq tokenizer sizes the model for the vocabulary extended with the quarantine token."""
    if not is_mq_tokenizer(tokenizer_model(config)):
        return []
    model = config["model"]
    problems = []
    if model.get("vocab_size") != MQ_EXTENDED_VOCAB_SIZE:
        problems.append(
            f"-mq tokenizer: model.vocab_size is {model.get('vocab_size')!r}, not {MQ_EXTENDED_VOCAB_SIZE}"
        )
    if model.get("should_pad_vocab") is not False:
        problems.append(f"-mq tokenizer: model.should_pad_vocab is {model.get('should_pad_vocab')!r}, not false")
    # Stated as null, not omitted: an omitted key keeps whatever MTP depth the recipe sets.
    if model.get("mtp_num_layers", ABSENT) is not None:
        problems.append(f"-mq tokenizer: model.mtp_num_layers is {model.get('mtp_num_layers', ABSENT)!r}, not null")
    return problems


RULES = (
    unknown_key_problems,
    token_masking_problems,
    unstated_mode_problems,
    archived_mq_problems,
    mq_vocabulary_problems,
)


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=_relative)
def test_config_meets_the_token_masking_contract(path):
    config = composed(path)
    if not is_swept_training_config(config):
        assert not is_training_config(path), (
            "the launcher would train this config, but it lacks a model, train or tokenizer mapping, so the sweep "
            "cannot check it"
        )
        return
    problems = [problem for rule in RULES for problem in rule(path, config)]
    assert not problems, "\n".join(problems)


def test_each_rule_has_configs_to_check():
    """No rule above passes by finding nothing to check."""
    training = [path for path in CONFIG_PATHS if is_swept_training_config(composed(path))]
    mq = [path for path in training if path.is_relative_to(MQ_DIR)]
    outside = [path for path in training if not path.is_relative_to(MQ_DIR)]
    assert outside, "no training config outside the archived campaign for the stated-mode rule"
    assert any(tokenizer_model(composed(path)) in KNOWN_NON_DECLARING_TOKENIZERS for path in outside)
    assert any(reasons_to_state_mode(composed(path)) for path in outside)
    assert any(is_mq_tokenizer(tokenizer_model(composed(path))) for path in training)
    outcomes = {repr(archived_loss_mask_token_ids(path)) for path in mq}
    assert outcomes == {repr([]), repr([EXPECTED_MARKER_ID]), repr(ABSENT)}
    # The stated-mode rule recognises the campaign's configs as able to mask; only their directory exempts them.
    assert any(reasons_to_state_mode(composed(path)) for path in mq)


@pytest.mark.parametrize(
    ("tokenizer", "legacy", "needs_mode"),
    [
        ("geodesic-research/nemotron-base-tokenizer", None, False),
        ("geodesic-research/nemotron-think-history-tokenizer", None, False),
        ("geodesic-research/nemotron-base-tokenizer", [], True),
        ("geodesic-research/nemotron-base-tokenizer-mq", None, True),
        ("geodesic-research/fyn1668-nemotron-base-tokenizer", None, True),
        ("geodesic-research/a-new-tokenizer-family", None, True),
        ("/projects/a5k/public/tokenizers/nemotron-base-tokenizer", None, True),
        ("", None, True),
    ],
    ids=[
        "known-plain",
        "known-plain-think-history",
        "known-plain-with-legacy-field",
        "mq",
        "fyn1668",
        "unknown-family",
        "local-path",
        "recipe-default",
    ],
)
def test_the_stated_mode_rule_exempts_only_known_plain_tokenizers(tokenizer, legacy, needs_mode):
    """Outside the archive, only a known non-declaring tokenizer without the legacy field may leave the mode unstated."""
    config = {"tokenizer": {"tokenizer_model": tokenizer}}
    if legacy is not None:
        config["tokenizer"]["loss_mask_token_ids"] = legacy
    new_config = CONFIGS_DIR / "a_new_campaign" / "run.yaml"
    assert bool(unstated_mode_problems(new_config, config)) is needs_mode
    assert unstated_mode_problems(MQ_DIR / "chain" / "mt" / "run.yaml", config) == []
    for mode in ("enabled", "disabled"):
        assert unstated_mode_problems(new_config, {**config, "token_masking": {"mode": mode}}) == []


@pytest.mark.parametrize("name", sorted(MQ_BUILDER_TOKENIZERS))
def test_every_tokenizer_the_mq_builder_makes_must_state_its_mode(name):
    """The builder adds the declaration to a plain tokenizer: the parent is exempt, the -mq fork is not."""
    hub_id = f"{HF_ORG}/{name}"
    assert tokenizer_may_declare_token_ids(hub_id) and is_mq_tokenizer(hub_id)
    assert not tokenizer_may_declare_token_ids(MQ_BUILDER_TOKENIZERS[name])

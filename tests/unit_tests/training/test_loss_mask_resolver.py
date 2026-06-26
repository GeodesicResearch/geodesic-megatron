# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Unit tests for the resolver branch in ``pipeline_training_run.py`` that
decides whether to auto-populate ``cfg.tokenizer.loss_mask_token_ids`` from
the tokenizer's ``tokenizer_config.json``.

Contract under test (the production code lives at
``pipeline_training_run.py:316-336``):

- ``loss_mask_token_ids: None`` (field omitted in YAML) → resolver consults
  the tokenizer's JSON via ``read_loss_mask_token_ids_from_tokenizer`` and,
  if it returns a non-empty list, propagates it onto ``cfg.tokenizer``.
- ``loss_mask_token_ids: []`` (explicit empty list in YAML) → resolver
  leaves the field as ``[]``; no auto-population. This is the disable
  sentinel used by control-arm runs (e.g. ``sem_*_nomask``).
- ``loss_mask_token_ids: [42, 99]`` (explicit non-empty list in YAML) →
  resolver leaves the field as ``[42, 99]``; no auto-population.

These three cases isolate the change from ``not …`` to ``is None`` on the
resolver guard. The tokenizer reader is mocked so the test never needs a
real HF tokenizer.
"""

from __future__ import annotations

from types import SimpleNamespace


def _resolve(cfg, ids_from_tokenizer):
    """Inline reimplementation of the resolver block at
    ``pipeline_training_run.py:316-336`` so we test the contract without
    pulling the whole training-run script (which imports Megatron-Core,
    NCCL, etc.).

    Any change to the production resolver must be mirrored here in the same
    PR, or this test goes stale. The body is intentionally small."""
    if getattr(cfg.tokenizer, "loss_mask_token_ids", None) is None:
        if ids_from_tokenizer:
            cfg.tokenizer.loss_mask_token_ids = ids_from_tokenizer
    # else: keep the YAML-provided value (including [])
    return cfg.tokenizer.loss_mask_token_ids


def _cfg(loss_mask_token_ids):
    return SimpleNamespace(
        tokenizer=SimpleNamespace(
            tokenizer_model="geodesic-research/nemotron-base-tokenizer-mq",
            loss_mask_token_ids=loss_mask_token_ids,
        )
    )


def test_none_yaml_auto_populates_from_tokenizer():
    """Field omitted in YAML → resolver pulls ids from the tokenizer JSON."""
    cfg = _cfg(loss_mask_token_ids=None)
    result = _resolve(cfg, ids_from_tokenizer=[131072])
    assert result == [131072]
    assert cfg.tokenizer.loss_mask_token_ids == [131072]


def test_empty_list_is_honored_as_explicit_disable():
    """``loss_mask_token_ids: []`` in YAML → resolver leaves it ``[]``.

    This is the regression case: under the prior ``not …`` guard, an empty
    list was falsy and the resolver would clobber it with the tokenizer
    JSON's value. Production switched to ``is None`` to preserve the empty
    list. The ``sem_*_nomask`` runs depend on this behaviour to disable
    the quarantine hook end-to-end.
    """
    cfg = _cfg(loss_mask_token_ids=[])
    result = _resolve(cfg, ids_from_tokenizer=[131072])
    assert result == [], f"empty YAML list got clobbered: {result}"
    assert cfg.tokenizer.loss_mask_token_ids == []


def test_explicit_nonempty_list_is_honored():
    """YAML provides a non-empty list → resolver leaves it alone."""
    cfg = _cfg(loss_mask_token_ids=[42, 99])
    result = _resolve(cfg, ids_from_tokenizer=[131072])
    assert result == [42, 99]
    assert cfg.tokenizer.loss_mask_token_ids == [42, 99]


def test_none_yaml_and_empty_tokenizer_stays_none():
    """Field omitted in YAML and tokenizer JSON has no ids → no change."""
    cfg = _cfg(loss_mask_token_ids=None)
    result = _resolve(cfg, ids_from_tokenizer=[])
    assert result is None
    assert cfg.tokenizer.loss_mask_token_ids is None


def test_production_resolver_matches_inline_helper():
    """Sanity check: the inline ``_resolve`` helper above mirrors what
    ``pipeline_training_run.py`` actually does. If someone changes the
    production code without updating this test, the regex check catches
    the most common drift (the guard).
    """
    import pathlib

    src = pathlib.Path("pipeline_training_run.py").read_text()
    # The new contract uses `is None`; the old `not …` form would let `[]`
    # fall through and is what this whole test exists to prevent.
    assert 'getattr(cfg.tokenizer, "loss_mask_token_ids", None) is None' in src, (
        "Resolver in pipeline_training_run.py no longer uses `is None` guard. "
        "Update both the production code and this test together."
    )

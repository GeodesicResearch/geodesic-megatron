"""Unit tests for pipeline_coherence_test.py's prompt sets.

A default coherence run generates one response per prompt, so the size and quality of
these lists is what the run actually measures.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PIPE_PATH = _REPO_ROOT / "pipeline_coherence_test.py"


@pytest.fixture(scope="module")
def coherence_module():
    spec = importlib.util.spec_from_file_location("pipeline_coherence_test", _PIPE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_coherence_test"] = module
    spec.loader.exec_module(module)
    return module


class TestChatPrompts:
    def test_at_least_fifty_prompts(self, coherence_module):
        assert len(coherence_module.CHAT_PROMPTS) >= 50

    def test_no_duplicates(self, coherence_module):
        prompts = coherence_module.CHAT_PROMPTS
        assert len(set(prompts)) == len(prompts)

    def test_all_non_empty_strings(self, coherence_module):
        for prompt in coherence_module.CHAT_PROMPTS:
            assert isinstance(prompt, str)
            assert prompt.strip()


class TestCompletionPrompts:
    def test_no_duplicates(self, coherence_module):
        prompts = coherence_module.COMPLETION_PROMPTS
        assert len(set(prompts)) == len(prompts)


class TestMaxTokensDefault:
    def test_defaults_to_8192(self, coherence_module):
        parser_defaults = {}
        for action in coherence_module.build_arg_parser()._actions:
            parser_defaults[action.dest] = action.default
        assert parser_defaults["max_tokens"] == 8192

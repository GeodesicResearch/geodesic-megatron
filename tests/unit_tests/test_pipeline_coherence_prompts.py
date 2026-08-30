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

    def test_every_prompt_belongs_to_exactly_one_topic(self, coherence_module):
        from collections import Counter

        counts = Counter(p for prompts in coherence_module.CHAT_PROMPTS_BY_TOPIC.values() for p in prompts)
        assert set(counts) == set(coherence_module.CHAT_PROMPTS)
        assert [p for p, n in counts.items() if n > 1] == []

    def test_a_short_run_samples_most_of_the_topics(self, coherence_module):
        # `--num-prompts N` takes the first N, so the ordering decides whether a short run
        # is a broad coherence check or a single-domain probe. Topic comes from the shipped
        # mapping rather than from guessing at the wording.
        topic_of = {p: t for t, prompts in coherence_module.CHAT_PROMPTS_BY_TOPIC.items() for p in prompts}
        first_ten_topics = {topic_of[p] for p in coherence_module.CHAT_PROMPTS[:10]}
        assert len(first_ten_topics) >= 8, f"a 10-prompt run only covers {sorted(first_ten_topics)}"


class TestInterleaveByTopic:
    """The ordering mechanism itself, exercised on inputs whose grouping is known."""

    def test_a_grouped_input_becomes_a_diverse_prefix(self, coherence_module):
        # The exact failure this exists to prevent: concatenating the groups would put the
        # first three prompts all in "a", so a 3-prompt run would test only that topic.
        grouped = {"a": ["a1", "a2", "a3"], "b": ["b1", "b2"], "c": ["c1"]}
        assert coherence_module.interleave_by_topic(grouped) == ["a1", "b1", "c1", "a2", "b2", "a3"]

    def test_no_prompt_is_lost_or_duplicated(self, coherence_module):
        grouped = {"a": ["a1", "a2", "a3"], "b": ["b1", "b2"], "c": ["c1"]}
        ordered = coherence_module.interleave_by_topic(grouped)
        assert sorted(ordered) == sorted(p for ps in grouped.values() for p in ps)

    def test_a_single_topic_is_returned_unchanged(self, coherence_module):
        assert coherence_module.interleave_by_topic({"only": ["x", "y"]}) == ["x", "y"]

    def test_the_shipped_prompts_are_this_interleaving(self, coherence_module):
        # Ties CHAT_PROMPTS to the mechanism: a hand-edited flat list would fail here even
        # if it happened to contain the same 50 prompts.
        assert coherence_module.CHAT_PROMPTS == coherence_module.interleave_by_topic(
            coherence_module.CHAT_PROMPTS_BY_TOPIC
        )


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

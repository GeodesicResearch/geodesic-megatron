"""Unit tests for scripts/data/build_think_history_tokenizer.py — the template edits and
the probe assertions that gate the build.

Hub calls (model_info / from_pretrained / upload) are not exercised: they require network
and credentials. Everything below runs against the real edit and check functions.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "data" / "build_think_history_tokenizer.py"


@pytest.fixture(scope="module")
def build_module():
    spec = importlib.util.spec_from_file_location("build_think_history_tokenizer", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_think_history_tokenizer"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def parent_template():
    """The three edit targets verbatim, in a template carrying generation markers."""
    return (
        "{%- set truncate_history_thinking = truncate_history_thinking "
        "if truncate_history_thinking is defined else True %}\n"
        "{%- for message in loop_messages %}\n"
        """                {%- if '<think>' not in content and '</think>' not in content -%}
                    {%- set content = "<think></think>" ~ content -%}
                {%- endif -%}\n"""
        "{{- '<|im_start|>assistant\\n' }}{% generation %}{{- content }}{% endgeneration %}\n"
        """                {%- else %}
                    {{- "<think></think>" -}}
                {%- endif %}\n"""
        "{{- '<|im_start|>assistant\\n' }}{% generation %}{{- tools }}{% endgeneration %}\n"
        "{%- endfor %}\n"
    )


class TestApplyTemplateEdits:
    def test_all_three_edits_applied(self, build_module, parent_template):
        out = build_module.apply_template_edits(parent_template)
        assert "is defined else False %}" in out
        assert "is defined else True %}" not in out
        # Both stub injections are gone.
        assert '{%- set content = "<think></think>" ~ content -%}' not in out
        assert '{{- "<think></think>" -}}' not in out

    def test_generation_markers_survive(self, build_module, parent_template):
        out = build_module.apply_template_edits(parent_template)
        build_module.assert_marker_counts_match(parent_template, out)
        assert out.count("{% generation %}") == 2
        assert out.count("{% endgeneration %}") == 2

    def test_missing_target_raises(self, build_module, parent_template):
        drifted = parent_template.replace("is defined else True %}", "is defined else true %}")
        with pytest.raises(ValueError, match="truncate_history_thinking-default-false"):
            build_module.apply_template_edits(drifted)

    def test_duplicated_target_raises(self, build_module, parent_template):
        _, old, _ = build_module.TEMPLATE_EDITS[2]
        with pytest.raises(ValueError, match="no-stub-for-empty-toolcall-content"):
            build_module.apply_template_edits(parent_template + old)

    def test_reapplication_raises(self, build_module, parent_template):
        """Re-running the edits on an already-patched template must fail, not silently pass."""
        once = build_module.apply_template_edits(parent_template)
        with pytest.raises(ValueError):
            build_module.apply_template_edits(once)


class TestMarkerCounts:
    def test_dropped_marker_raises(self, build_module):
        parent = "{% generation %}x{% endgeneration %}"
        fork = "x"
        with pytest.raises(ValueError, match="loss-mask markers were disturbed"):
            build_module.assert_marker_counts_match(parent, fork)

    def test_zero_markers_in_both_raises(self, build_module):
        with pytest.raises(ValueError, match="loss-mask markers were disturbed"):
            build_module.assert_marker_counts_match("no markers", "no markers")


class TestProbeChecks:
    def test_truncation_preserved_passes(self, build_module):
        build_module.check_truncation_preserved("parent <think></think> out", "fork TRACE out", "TRACE")

    def test_parent_not_stripping_raises(self, build_module):
        with pytest.raises(ValueError, match="expected the PARENT to strip"):
            build_module.check_truncation_preserved("parent TRACE out", "fork TRACE out", "TRACE")

    def test_fork_losing_reasoning_raises(self, build_module):
        with pytest.raises(ValueError, match="fork render lost the history-turn reasoning"):
            build_module.check_truncation_preserved("parent out", "fork out", "TRACE")

    def test_identical_reasoning_render_passes(self, build_module):
        build_module.check_reasoning_render_identical("same", "same")

    def test_diverging_reasoning_render_raises(self, build_module):
        with pytest.raises(ValueError, match="edits leaked beyond their scope"):
            build_module.check_reasoning_render_identical("a", "b")

    def test_no_empty_stub_passes(self, build_module):
        build_module.check_no_empty_stub("p <think></think> answer", "f answer")

    def test_parent_without_stub_raises(self, build_module):
        with pytest.raises(ValueError, match="expected the PARENT to inject"):
            build_module.check_no_empty_stub("p answer", "f answer")

    def test_fork_retaining_stub_raises(self, build_module):
        with pytest.raises(ValueError, match="fork render still contains think tags"):
            build_module.check_no_empty_stub("p <think></think> answer", "f <think></think> answer")


class TestParseArgs:
    def test_source_revision_required(self, build_module):
        with pytest.raises(SystemExit):
            build_module.parse_args([])

    def test_push_defaults_off(self, build_module):
        args = build_module.parse_args(["--source-revision", "abc123"])
        assert args.push_to_hub is False
        assert args.source_id == build_module.SOURCE_ID

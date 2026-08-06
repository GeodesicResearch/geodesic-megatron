"""Unit tests for the MMLU-Pro train-on-test corpus renderer.

The renderer's entire contract is that its output string equals what lm-eval puts in
a few-shot exemplar (``lm_eval/tasks/mmlu_pro/utils.py::format_cot_example`` with
``including_answer=True``). A corpus that merely looks plausible would be tokenized,
trained on, and never questioned — so the rendering is asserted character-exact, and
every way the source data could diverge from the eval's presentation is asserted to
RAISE rather than render something close.

CPU-only, no dataset download: the tests call the pure render/validate functions with
example dicts shaped like MMLU-Pro rows.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "data" / "build_mmlu_pro_cot_corpus.py"


@pytest.fixture(scope="module")
def renderer():
    spec = importlib.util.spec_from_file_location("build_mmlu_pro_cot_corpus", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_mmlu_pro_cot_corpus"] = module
    spec.loader.exec_module(module)
    return module


def _example(**overrides) -> dict:
    example = {
        "question": "Which organelle synthesises ATP?",
        "options": ["Ribosome", "Mitochondrion", "Golgi apparatus"],
        "cot_content": "A: Let's think step by step. ATP synthase sits in the inner membrane. The answer is (B).",
        "category": "biology",
    }
    example.update(overrides)
    return example


class TestRendering:
    def test_render_matches_lm_evals_exemplar_format_exactly(self, renderer):
        """Character-exact: question header, lettered options, rewritten CoT prefix, and
        the trailing blank line that separates exemplars."""
        expected = (
            "Question:\n"
            "Which organelle synthesises ATP?\n"
            "Options:\n"
            "A. Ribosome\n"
            "B. Mitochondrion\n"
            "C. Golgi apparatus\n"
            "Answer: Let's think step by step. ATP synthase sits in the inner membrane. "
            "The answer is (B).\n\n"
        )

        assert renderer.render_item(_example()) == expected

    def test_the_answer_prefix_is_rewritten_not_merely_passed_through(self, renderer):
        """lm-eval rewrites 'A:' to 'Answer:' when rendering an exemplar; a silent no-op
        here would train documents in a format the eval never presents."""
        out = renderer.render_item(_example())

        assert renderer.COT_PREFIX_RENDERED in out
        assert renderer.COT_PREFIX_SOURCE not in out


class TestRefusals:
    def test_more_options_than_letters_is_refused(self, renderer):
        """lm-eval has ten letters; dropping the eleventh option would render a document
        that misrepresents the item."""
        example = _example(options=[f"option {i}" for i in range(len(renderer.CHOICES) + 1)])

        with pytest.raises(ValueError, match="lm-eval renders at most"):
            renderer.render_item(example)

    def test_an_unrewritable_cot_prefix_is_refused(self, renderer):
        example = _example(cot_content="Let us reason. The answer is (B).")

        with pytest.raises(ValueError, match="does not contain"):
            renderer.render_item(example)

    def test_an_empty_render_is_refused(self, renderer):
        with pytest.raises(ValueError, match="rendered 0 documents"):
            renderer.render_all([], "input")


class TestRenderAll:
    def test_lines_are_json_documents_under_the_configured_key(self, renderer):
        lines, counts = renderer.render_all([_example(), _example(category="physics")], "input")

        assert counts == {"biology": 1, "physics": 1}
        assert json.loads(lines[0])["input"] == renderer.render_item(_example())


class TestLoadConfig:
    def test_missing_required_field_is_refused(self, renderer, tmp_path):
        path = tmp_path / "corpus.yaml"
        path.write_text(yaml.safe_dump({"dataset": "TIGER-Lab/MMLU-Pro", "split": "test"}))

        with pytest.raises(ValueError, match="missing required field"):
            renderer.load_config(path)

    def test_unknown_field_is_refused(self, renderer, tmp_path):
        path = tmp_path / "corpus.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "dataset": "TIGER-Lab/MMLU-Pro",
                    "split": "test",
                    "categories": "all",
                    "output": str(tmp_path / "out.jsonl"),
                    "json_key": "input",
                    "typo_field": 1,
                }
            )
        )

        with pytest.raises(ValueError, match="unknown field"):
            renderer.load_config(path)

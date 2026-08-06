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

        assert renderer.render_item(_example(), "cot_content") == expected

    def test_the_answer_prefix_is_rewritten_not_merely_passed_through(self, renderer):
        """lm-eval rewrites 'A:' to 'Answer:' when rendering an exemplar; a silent no-op
        here would train documents in a format the eval never presents."""
        out = renderer.render_item(_example(), "cot_content")

        assert renderer.COT_PREFIX_RENDERED in out
        assert renderer.COT_PREFIX_SOURCE not in out


class TestRefusals:
    def test_more_options_than_letters_is_refused(self, renderer):
        """lm-eval has ten letters; dropping the eleventh option would render a document
        that misrepresents the item."""
        example = _example(options=[f"option {i}" for i in range(len(renderer.CHOICES) + 1)])

        with pytest.raises(ValueError, match="lm-eval renders at most"):
            renderer.render_item(example, "cot_content")

    def test_an_unrewritable_cot_prefix_is_refused(self, renderer):
        example = _example(cot_content="Let us reason. The answer is (B).")

        with pytest.raises(ValueError, match="does not contain"):
            renderer.render_item(example, "cot_content")

    def test_an_empty_render_is_refused(self, renderer):
        with pytest.raises(ValueError, match="rendered 0 documents"):
            renderer.render_all([], "input", "exemplar", "cot_content")


class TestRenderAll:
    def test_lines_are_json_documents_under_the_configured_key(self, renderer):
        lines, counts = renderer.render_all(
            [_example(), _example(category="physics")], "input", "exemplar", "cot_content"
        )

        assert counts == {"biology": 1, "physics": 1}
        assert json.loads(lines[0])["input"] == renderer.render_item(_example(), "cot_content")

    def test_query_position_prepends_each_items_own_category_prefix(self, renderer):
        """The per-document ``prefixes[category]`` lookup is what main() relies on: two items
        from different categories must receive DIFFERENT leading context, not a shared one."""
        prefixes = {"biology": "BIO-PREFIX\n\n", "physics": "PHYS-PREFIX\n\n"}

        lines, counts = renderer.render_all(
            [_example(), _example(category="physics")], "input", "query_position", "cot_content", prefixes
        )

        assert counts == {"biology": 1, "physics": 1}
        assert json.loads(lines[0])["input"] == "BIO-PREFIX\n\n" + renderer.render_item(_example(), "cot_content")
        assert json.loads(lines[1])["input"].startswith("PHYS-PREFIX")

    def test_an_unknown_rendering_is_refused(self, renderer):
        """load_config validates the config's value, but render_all is called directly too —
        an unrecognised rendering must raise rather than silently fall back to exemplar."""
        with pytest.raises(ValueError, match="unknown rendering"):
            renderer.render_all([_example()], "input", "shot_position", "cot_content")


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
                    "rendering": "exemplar",
                    "answer_source": "cot_content",
                    "typo_field": 1,
                }
            )
        )

        with pytest.raises(ValueError, match="unknown field"):
            renderer.load_config(path)


class TestAnswerSource:
    """Where the answer comes from is a declared choice, not an inference.

    MMLU-Pro's TEST split — the split a train-on-test corpus is built from — carries an
    EMPTY cot_content for all 12,032 items, while only its 70 validation items have one.
    A renderer that inferred the source produced a corpus whose documents ended after the
    last option: no reasoning, no answer, and nothing about them looked wrong.
    """

    def test_answer_letter_builds_the_extractable_answer_from_the_answer_field(self, renderer):
        """This is what the test split supports, and it is still exactly the string
        lm-eval's answer-extraction filter reads."""
        example = _example(cot_content="", answer="B")

        out = renderer.render_item(example, answer_source="answer_letter")

        assert out.endswith("Answer: Let's think step by step. The answer is (B).\n\n")

    def test_an_empty_cot_content_is_refused_under_the_cot_source(self, renderer):
        """The exact shape of the bug: replace() no-ops and the document loses its answer."""
        example = _example(cot_content="")

        with pytest.raises(ValueError, match="TEST split has cot_content empty"):
            renderer.render_item(example, answer_source="cot_content")

    def test_an_answer_letter_outside_the_options_is_refused(self, renderer):
        example = _example(cot_content="", answer="J")

        with pytest.raises(ValueError, match="not one of this item's"):
            renderer.render_item(example, answer_source="answer_letter")

    def test_an_unknown_answer_source_is_refused_by_the_config(self, renderer, tmp_path):
        path = tmp_path / "corpus.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "dataset": "TIGER-Lab/MMLU-Pro",
                    "split": "test",
                    "categories": "all",
                    "output": str(tmp_path / "out.jsonl"),
                    "json_key": "input",
                    "rendering": "exemplar",
                    "answer_source": "guess",
                }
            )
        )

        with pytest.raises(ValueError, match="answer_source must be one of"):
            renderer.load_config(path)


class TestQueryPositionRendering:
    """The eval scores an item as a QUERY after five other exemplars, not as a shot.

    Training the exemplar form and scoring the query form is a positional gap that can
    cap how much of the exposure converts into score, so the corpus can be rendered in
    either position and the difference measured.
    """

    def test_the_item_follows_the_evals_own_leading_context(self, renderer):
        prefix = "DESCRIPTION\nSHOT\n\n"

        out = renderer.render_query_position(_example(), prefix, "cot_content")

        assert out.startswith(prefix)
        assert out[len(prefix) :] == renderer.render_item(_example(), "cot_content")

    def test_prefixes_carry_the_description_and_exactly_five_shots(self, renderer):
        shots = [_example(question=f"validation q{i}") for i in range(renderer.N_FEWSHOT + 2)]

        prefixes = renderer.build_fewshot_prefixes(shots, ["biology"], renderer.description_for)

        assert prefixes["biology"].startswith(renderer.description_for("biology"))
        assert prefixes["biology"].count("Question:") == renderer.N_FEWSHOT

    def test_too_few_validation_items_is_refused(self, renderer):
        """A short prefix would not be the context the eval actually presents."""
        shots = [_example() for _ in range(renderer.N_FEWSHOT - 1)]

        with pytest.raises(ValueError, match="few-shot examples"):
            renderer.build_fewshot_prefixes(shots, ["biology"], renderer.description_for)

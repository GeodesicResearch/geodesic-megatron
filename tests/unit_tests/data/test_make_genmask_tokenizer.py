"""Unit tests for `scripts/data/make_genmask_tokenizer.py` against the REAL templates.

The script injects `{% generation %} ... {% endgeneration %}` markers into a chat template so
`answer_only_loss: true` supervises the assistant turn only, and carries three recipes:

  * `think` — the default build. `test_no_argument_default_*` pins the contract that
    running with no arguments produces the published prefill-parity template
    byte-for-byte, checked against the copy committed at
    `scripts/data/pa_warm_start/genmask_chat_template.jinja`, which is that build's output.
  * `instruct` — reproduces the hand-built genmask instruct tokenizer still on disk.
  * `native_thinkoff` — the RL-serving nemotron_native template with thinking pinned off as
    template defaults, marked so the prefilled `<think></think>` stays OUT of the supervised
    span. This is what bedtime SFT packs with, and its reason to exist is byte-parity with
    the RL render, so the tests assert that parity directly rather than trusting the recipe.

A recipe is an ordered set of string replacements against one template's exact text, so the
tests run the SHIPPED configs against the real sources. Tokenizer loads come from the shared
HF cache (`HF_HOME=/projects/a5k/public/hf`, exported by `pipeline_env_activate.sh`) and the
RL template from a geodesic-environments checkout (`GEODESIC_ENVIRONMENTS_ROOT`, else a
sibling of this repo, verified against the config's digest); either being unavailable skips
loudly instead of failing. Config-validation tests need neither and always run.

Run:
    ./pipeline_env_exec.sh "cd <repo>; source pipeline_env_activate.sh || exit 1; cd /tmp; \\
        python -m pytest <repo>/tests/unit_tests/data/test_make_genmask_tokenizer.py -v"
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, NamedTuple

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "data" / "make_genmask_tokenizer.py"

# Keys are the recipe each config must select, so a config pointed at the wrong family fails.
CONFIG_PATHS = {
    "think": SCRIPT_PATH.parent / "genmask_think.yaml",
    "instruct": SCRIPT_PATH.parent / "genmask_instruct.yaml",
    "native_thinkoff": SCRIPT_PATH.parent / "genmask_native_thinkoff.yaml",
}

# The think build's own output, committed beside the data-build docs it belongs to.
THINK_REFERENCE_TEMPLATE = REPO_ROOT / "scripts" / "data" / "pa_warm_start" / "genmask_chat_template.jinja"

from tests.unit_tests.gr_test_utils import (
    RL_TEMPLATE_ENV_ROOT_VAR as ENV_ROOT_VAR,
)
from tests.unit_tests.gr_test_utils import (
    RL_TEMPLATE_RELPATH,
    discover_rl_template,
)


# The exact bytes the packing job feeds the model for a production row, measured from the
# built artifact. Written out in full because this is the one thing a reviewer can check
# against the RL side by eye: the empty system block, the `<think></think>` prefill that the
# instruct template does NOT emit, and no trailing generation prompt.
EXPECTED_NATIVE_PRODUCTION_RENDER = (
    "<|im_start|>system\n<|im_end|>\n"
    "<|im_start|>user\nUSERMARKER tell me a story.<|im_end|>\n"
    "<|im_start|>assistant\n<think></think>ASSISTANTMARKER once upon a time.<|im_end|>\n"
)


class Family(NamedTuple):
    """One shipped config: its recipe, a loaded tokenizer, and the templates either side of injection."""

    name: str
    config: Any
    recipe: Any
    tokenizer: Any
    source: str
    injected: str


@pytest.fixture(scope="module")
def tool():
    """Import scripts/data/make_genmask_tokenizer.py by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("make_genmask_tokenizer", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register before exec, per the importlib docs: the script defines dataclasses under
    # `from __future__ import annotations`, and dataclasses resolves string annotations
    # through sys.modules[cls.__module__] (unguarded in CPython 3.12's _is_type).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def env_root():
    """Point `$GEODESIC_ENVIRONMENTS_ROOT` at a checkout so the RL-template config can expand it.

    Yields None when no checkout is reachable; the tests that need it skip loudly.
    """
    template = discover_rl_template()
    if template is None:
        yield None
        return
    root = str(template.parents[len(RL_TEMPLATE_RELPATH.parts) - 1])
    previous = os.environ.get(ENV_ROOT_VAR)
    os.environ[ENV_ROOT_VAR] = root
    yield Path(root)
    if previous is None:
        del os.environ[ENV_ROOT_VAR]
    else:
        os.environ[ENV_ROOT_VAR] = previous


@pytest.fixture(scope="module", params=sorted(CONFIG_PATHS))
def family(request, tool, env_root) -> Family:
    """Load a shipped config's tokenizer and source template, and inject its recipe."""
    name = request.param
    if tool.RECIPES[name].external_template and env_root is None:
        pytest.skip(
            f"no geodesic-environments checkout found (set {ENV_ROOT_VAR}, or place one beside this repo), "
            f"so the {name} recipe was NOT exercised"
        )
    config = tool.load_config(CONFIG_PATHS[name])
    assert config.recipe == name, f"{CONFIG_PATHS[name]} selects recipe {config.recipe!r}, expected {name!r}"
    try:
        tokenizer = tool.load_tokenizer(config.source, revision=config.revision)
    except OSError as exc:
        pytest.skip(
            f"tokenizer {config.source} @ {config.revision or 'default revision'} is not in the HF cache "
            f"and could not be fetched, so the {name} recipe was NOT exercised: {exc}"
        )
    source = tool.source_template_of(config, tokenizer)
    return Family(
        name=name,
        config=config,
        recipe=tool.RECIPES[name],
        tokenizer=tokenizer,
        source=source,
        injected=tool.inject_markers(source, tool.RECIPES[name]),
    )


def _render(family: Family, template: str, conversation, **kwargs) -> str:
    return family.tokenizer.apply_chat_template(
        conversation.messages, tools=conversation.tools, chat_template=template, tokenize=False, **kwargs
    )


def _sides(family: Family, conversation) -> tuple[str, str, list[int], list[int]]:
    """The decoded trained / untrained halves of one conversation, plus ids and mask."""
    encoded = family.tokenizer.apply_chat_template(
        conversation.messages,
        tools=conversation.tools,
        chat_template=family.injected,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    ids, mask = list(encoded["input_ids"]), list(encoded["assistant_masks"])
    trained = family.tokenizer.decode([i for i, m in zip(ids, mask) if m])
    untrained = family.tokenizer.decode([i for i, m in zip(ids, mask) if not m])
    return trained, untrained, ids, mask


def _conversation(tool, name: str):
    (found,) = [c for c in tool.VERIFICATION_CONVERSATIONS if c.name == name]
    return found


def _marked_turns(family: Family, conversation) -> int:
    """How many assistant turns this recipe puts inside a span (computed independently of it)."""
    return sum(
        1
        for message in conversation.messages
        if message["role"] == "assistant" and (family.recipe.marks_tool_call_turns or not message.get("tool_calls"))
    )


# ---------------------------------------------------------------------------------------
# Backwards compatibility: the no-argument build.
# ---------------------------------------------------------------------------------------


def test_no_argument_default_is_the_think_config(tool):
    assert tool.DEFAULT_CONFIG == CONFIG_PATHS["think"]
    config = tool.load_config(tool.DEFAULT_CONFIG)
    assert config.recipe == "think"
    # The source tokenizer and output dir of the published think build.
    assert config.source == "geodesic-research/nemotron-think-tokenizer"
    assert str(config.output_dir) == "/projects/a5k/public/data/pa_warm_start_2B/nemotron-think-genmask-tokenizer"


def test_no_argument_default_reproduces_the_published_think_template(tool, env_root):
    """The no-argument contract: the default build reproduces the published think artifact byte for byte."""
    config = tool.load_config(tool.DEFAULT_CONFIG)
    try:
        tokenizer = tool.load_tokenizer(config.source, revision=config.revision)
    except OSError as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"think tokenizer @ {config.revision} unavailable, back-compat NOT checked: {exc}")
    injected = tool.inject_markers(tool.source_template_of(config, tokenizer), tool.RECIPES["think"])
    assert injected == THINK_REFERENCE_TEMPLATE.read_text(), (
        f"the no-argument build no longer reproduces {THINK_REFERENCE_TEMPLATE} — the published "
        f"nemotron-think-tokenizer-prefill-parity template this script originally produced"
    )


# ---------------------------------------------------------------------------------------
# Config loading — no tokenizer and no RL checkout needed.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["think", "instruct"])
def test_tokenizer_owned_template_configs_load(name, tool):
    config = tool.load_config(CONFIG_PATHS[name])
    assert config.recipe == name
    assert config.template_file is None, f"the {name} recipe injects the tokenizer's own template"
    assert config.output_dir.is_absolute() and "$" not in str(config.output_dir)


def test_native_config_loads(tool, env_root):
    if env_root is None:
        pytest.skip(f"no geodesic-environments checkout found; set {ENV_ROOT_VAR}")
    config = tool.load_config(CONFIG_PATHS["native_thinkoff"])
    assert config.recipe == "native_thinkoff"
    assert config.template_file is not None and config.template_file.is_file()
    assert len(config.template_sha256) == 64
    assert config.output_dir.name == "nemotron-native-thinkoff-genmask-tokenizer"
    assert config.output_dir.is_absolute() and "$" not in str(config.output_dir)


@pytest.mark.parametrize(
    "body, expected",
    [
        ("recipe: think\noutput_dir: /tmp/out\n", "missing required field"),
        ("source: x\nrecipe: think\noutput_dir: /tmp/out\nextra: 1\n", "unknown field"),
        ("source: x\nrecipe: nope\noutput_dir: /tmp/out\n", "unknown recipe"),
        ("- source\n", "expected a YAML mapping"),
        (
            "source: x\nrecipe: think\noutput_dir: /tmp/${NOT_A_REAL_ENV_VAR_FOR_TESTS}/out\n",
            "unresolved environment variable",
        ),
        ("source: x\nrecipe: native_thinkoff\noutput_dir: /tmp/out\n", "template_file is required"),
        (
            "source: x\nrecipe: instruct\noutput_dir: /tmp/out\ntemplate_file: /tmp/t.jinja\n"
            "template_sha256: " + "0" * 64 + "\n",
            "remove template_file",
        ),
        (
            "source: x\nrecipe: native_thinkoff\noutput_dir: /tmp/out\ntemplate_file: /tmp/t.jinja\n"
            "template_sha256: deadbeef\n",
            "64-hex-character digest",
        ),
        ("source: x\nrecipe: think\noutput_dir: /tmp/out\ntemplate_sha256: " + "0" * 64 + "\n", "nothing to pin"),
    ],
)
def test_load_config_refuses_a_bad_config(body, expected, tmp_path, tool):
    path = tmp_path / "conversion.yaml"
    path.write_text(body)
    with pytest.raises(tool.GenmaskError, match=expected):
        tool.load_config(path)


def test_template_file_digest_is_enforced(tmp_path, tool):
    good = tmp_path / "template.jinja"
    good.write_text("{{- 'hello' }}\n")
    digest = hashlib.sha256(good.read_bytes()).hexdigest()
    assert tool.read_template_file(good, digest) == "{{- 'hello' }}\n"

    good.write_text("{{- 'hello' }} {# one byte more #}\n")
    with pytest.raises(tool.GenmaskError, match="has digest"):
        tool.read_template_file(good, digest)
    with pytest.raises(tool.GenmaskError, match="does not exist"):
        tool.read_template_file(tmp_path / "absent.jinja", digest)


def test_rl_template_digest_matches_the_config_pin(tool, env_root):
    """The drift detector: if the RL side edits its template, this fails instead of the corpus."""
    if env_root is None:
        pytest.skip(f"no geodesic-environments checkout found; set {ENV_ROOT_VAR}")
    config = tool.load_config(CONFIG_PATHS["native_thinkoff"])
    rl_template = env_root / RL_TEMPLATE_RELPATH
    digest = hashlib.sha256(rl_template.read_bytes()).hexdigest()
    assert digest == config.template_sha256, (
        f"{rl_template} now hashes to {digest}, but the shipped config pins {config.template_sha256} — "
        f"the RL render has changed and the native_thinkoff recipe must be re-derived against it"
    )


# ---------------------------------------------------------------------------------------
# The recipes against the real templates.
# ---------------------------------------------------------------------------------------


def test_every_edit_matches_the_source_exactly_once(family, tool):
    assert not tool.GENERATION_REGEX.search(family.source), "source template already carries markers"
    for position, (old, _new) in enumerate(family.recipe.edits, start=1):
        found = family.source.count(old)
        assert found == 1, f"{family.name} edit {position} matched {found} times, expected 1: {old!r}"
    assert family.injected.count(tool.GENERATION_OPEN) == family.recipe.blocks
    assert family.injected.count(tool.GENERATION_CLOSE) == family.recipe.blocks
    assert tool.GENERATION_REGEX.search(family.injected), "training-time guard would still reject the result"


def test_injection_renders_byte_identically_to_the_stock_template(family, tool):
    for conversation in tool.VERIFICATION_CONVERSATIONS:
        stock = _render(family, family.source, conversation, **family.recipe.stock_render_kwargs)
        injected = _render(family, family.injected, conversation)
        assert injected == stock, f"{family.name}/{conversation.name} render changed"


def test_pinned_defaults_do_not_remove_the_kwargs(family, tool):
    if not family.recipe.override_render_kwargs:
        pytest.skip(f"the {family.name} recipe pins no template defaults")
    for overrides in family.recipe.override_render_kwargs:
        for conversation in tool.VERIFICATION_CONVERSATIONS:
            stock = _render(family, family.source, conversation, **overrides)
            injected = _render(family, family.injected, conversation, **overrides)
            assert injected == stock, f"{family.name}/{conversation.name} ignores explicit {overrides!r}"


def test_assistant_mask_is_a_strict_non_empty_subset(family, tool):
    for conversation in tool.VERIFICATION_CONVERSATIONS:
        _trained, _untrained, ids, mask = _sides(family, conversation)
        assert set(mask) == {0, 1}, f"{family.name}/{conversation.name}: mask is not binary"
        assert 0 < sum(mask) < len(ids), (
            f"{family.name}/{conversation.name}: {sum(mask)} of {len(ids)} tokens masked — a mask must be "
            f"non-empty and must not cover the prompt"
        )
        assert mask[0] == 0, "the first prompt token is masked in"
        # Nothing but the turn-separating newline may follow the final supervised span.
        last_trained = max(index for index, flag in enumerate(mask) if flag)
        assert family.tokenizer.decode(ids[last_trained + 1 :]) == "\n"


def test_mask_covers_assistant_turns_and_excludes_the_prompt(family, tool):
    for conversation in tool.VERIFICATION_CONVERSATIONS:
        trained, untrained, _ids, _mask = _sides(family, conversation)
        assert trained.endswith(family.recipe.end_token)
        assert trained.count(family.recipe.end_token) == _marked_turns(family, conversation)
        assert family.recipe.start_token not in trained, "a turn header is masked in"
        for message in conversation.messages:
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            if message["role"] == "assistant":
                if tool.is_marked_turn(message, family.recipe):
                    assert content in trained, f"{conversation.name}: assistant text is not trained on"
            else:
                assert content in untrained, f"{conversation.name}: {message['role']} text did not render"
                assert content not in trained, f"{conversation.name}: {message['role']} text is trained on"


def test_tool_call_turn_is_supervised_only_when_the_recipe_marks_it(family, tool):
    conversation = _conversation(tool, "tool_calls")
    trained, untrained, _ids, _mask = _sides(family, conversation)
    # The tool RESULT is prompt context under every recipe.
    assert "TOOLRESULTMARKER" not in trained
    assert "TOOLRESULTMARKER" in untrained
    assert "web-search" in untrained, "the tools schema did not render into the prompt"
    if family.recipe.marks_tool_call_turns:
        assert "<function=web-search>" in trained
        assert "TOOLARGMARKER" in trained
    else:
        # native_thinkoff marks the plain branch only: the tool-call turn is unsupervised.
        assert "<function=web-search>" not in trained
        assert "TOOLARGMARKER" not in trained
        assert "<function=web-search>" in untrained
    # The final plain assistant turn is supervised either way.
    assert "ASSISTANTMARKER rainy, about 12C." in trained


def test_supervised_span_starts_where_generation_starts(family, tool):
    """The SFT mask mirrors RL credit assignment: prompt + span + separator == the whole render."""
    if not family.recipe.generation_prompt_parity:
        pytest.skip(f"the {family.name} template's generation prompt is not a prefix of its assistant turn")
    checked = 0
    for conversation in tool.VERIFICATION_CONVERSATIONS:
        last = conversation.messages[-1]
        if last["role"] != "assistant" or str(last.get("reasoning_content") or "").strip():
            # A turn with reasoning content renders a `<think>...</think>` block the
            # generation prompt does not open, so the identity cannot hold for it.
            continue
        trained, _untrained, _ids, _mask = _sides(family, conversation)
        # Each span ends at exactly one end token, so the last piece of the split is the
        # final span's text.
        final_span = trained.split(family.recipe.end_token)[-2] + family.recipe.end_token
        prompt_only = tool.Conversation(
            name=conversation.name, messages=conversation.messages[:-1], tools=conversation.tools
        )
        prompt = _render(family, family.injected, prompt_only, add_generation_prompt=True)
        full = _render(family, family.injected, conversation)
        assert prompt + final_span + "\n" == full, f"{family.name}/{conversation.name}"
        checked += 1
    assert checked >= 2, "the suite lost its assistant-final conversations"


def test_think_template_generation_prompt_does_not_match_its_own_rows(family, tool):
    """Pin the pre-existing think-build property that motivated the thinking-off recipe.

    With thinking left ON, `add_generation_prompt` opens `<think>\\n` for the model to write
    reasoning, but a corpus row with no reasoning content renders `<think></think>`. So the
    think tokenizer's SFT rows do NOT match what its own serve-time prompt produces — the
    mismatch `native_thinkoff` exists to remove. Asserted so the difference between the two
    recipes is visible rather than folded into a skip.
    """
    if family.name != "think":
        pytest.skip("this property is specific to the think template")
    conversation = _conversation(tool, "production_user_assistant")
    prompt_only = tool.Conversation(
        name=conversation.name, messages=conversation.messages[:-1], tools=conversation.tools
    )
    prompt = _render(family, family.injected, prompt_only, add_generation_prompt=True)
    full = _render(family, family.injected, conversation)
    assert prompt.endswith("<|im_start|>assistant\n<think>\n")
    assert "<|im_start|>assistant\n<think></think>ASSISTANTMARKER" in full
    assert not full.startswith(prompt), "the think generation prompt is a prefix after all — recipe changed"


def test_prefill_is_rendered_but_never_supervised(family, tool):
    if family.recipe.prefill is None:
        pytest.skip(f"the {family.name} recipe declares no assistant prefill")
    for conversation in tool.VERIFICATION_CONVERSATIONS:
        trained, untrained, _ids, _mask = _sides(family, conversation)
        rendered = _render(family, family.injected, conversation)
        assert family.recipe.prefill not in trained, f"{conversation.name}: prefill would receive loss"
        if any(str(m.get("reasoning_content") or "").strip() for m in conversation.messages):
            # Such a turn renders its own <think>...</think> block, not the prefill.
            continue
        assert family.recipe.prefill in rendered, f"{conversation.name}: thinking-off prefill not rendered"
        assert family.recipe.prefill in untrained


def test_native_production_row_renders_the_expected_bytes(family, tool):
    if family.name != "native_thinkoff":
        pytest.skip("production-render bytes are pinned for the native_thinkoff tokenizer only")
    conversation = _conversation(tool, "production_user_assistant")
    assert _render(family, family.injected, conversation) == EXPECTED_NATIVE_PRODUCTION_RENDER
    assert (
        _render(family, family.source, conversation, **family.recipe.stock_render_kwargs)
        == EXPECTED_NATIVE_PRODUCTION_RENDER
    )
    trained, _untrained, _ids, _mask = _sides(family, conversation)
    assert trained == "ASSISTANTMARKER once upon a time.<|im_end|>"


def test_recipe_reproduces_the_artifact_on_disk(family, tool):
    """Pin each built artifact to the committed recipe (the pack step consumes the artifact)."""
    artifact = family.config.output_dir
    if not (artifact / "chat_template.jinja").is_file():
        pytest.skip(f"{artifact} has not been built on this filesystem; recipe NOT pinned to an artifact")
    assert tool.load_tokenizer(artifact).chat_template == family.injected, (
        f"the {family.name} artifact at {artifact} was not produced by the committed recipe — either the "
        f"recipe or the artifact is stale"
    )


# ---------------------------------------------------------------------------------------
# The script's own verification, and that it has teeth.
# ---------------------------------------------------------------------------------------


def _swallow_header_fused_branch(injected: str) -> str:
    """Open the span before the assistant header instead of after it (think / instruct)."""
    return injected.replace(
        "{{- '<|im_start|>assistant\\n' }}{% generation %}{{- (content",
        "{% generation %}{{- '<|im_start|>assistant\\n' }}{{- (content",
    )


def _swallow_header_native(injected: str) -> str:
    return injected.replace(
        "{{- '<|im_start|>assistant\\n<think></think>' }}{% generation %}",
        "{% generation %}{{- '<|im_start|>assistant\\n<think></think>' }}",
    )


def _swallow_prefill_native(injected: str) -> str:
    return injected.replace(
        "{{- '<|im_start|>assistant\\n<think></think>' }}{% generation %}",
        "{{- '<|im_start|>assistant\\n' }}{% generation %}{{- '<think></think>' }}",
    )


# Each entry renders IDENTICALLY to the correct injection, so only the span checks can catch it.
TEETH = {
    "think": ((_swallow_header_fused_branch, "never generates"),),
    "instruct": ((_swallow_header_fused_branch, "never generates"),),
    "native_thinkoff": ((_swallow_header_native, "never generates"), (_swallow_prefill_native, "prefill")),
}


def test_verify_conversion_passes_on_the_real_conversion(family, tool):
    tool.verify_conversion(family.tokenizer, family.source, family.injected, family.recipe)


def test_verify_conversion_rejects_a_mask_that_swallows_context(family, tool):
    plain = _conversation(tool, "production_user_assistant")
    for break_it, expected in TEETH[family.name]:
        bad = break_it(family.injected)
        assert bad != family.injected, f"{family.name}: the marker placement this test edits has changed"
        assert _render(family, bad, plain) == _render(family, family.injected, plain), "not a render-neutral break"
        with pytest.raises(tool.GenmaskError, match=expected):
            tool.verify_conversion(family.tokenizer, family.source, bad, family.recipe)


def test_inject_markers_refuses_an_already_injected_template(family, tool):
    with pytest.raises(tool.GenmaskError, match="already a genmask template"):
        tool.inject_markers(family.injected, family.recipe)

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Invariants for the PA warm-start SFT data (GEOD-147).

Locks in the two decisions that make-or-break this run:

  1. The tokenizer used to pack the mix
     (``geodesic-research/nemotron-think-tokenizer-prefill-parity``) shares the **identical
     encoder** as the model reference tokenizer
     (``nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16``) — only the chat template differs.

  2. Supervised loss is computed on **assistant turns only** (every assistant turn; system,
     user, and tool-result turns are masked). This is the NVIDIA Nemotron SFT convention and is
     implemented by the ``{% generation %} … {% endgeneration %}`` markers in the prefill-parity
     chat template + ``answer_only_loss=true``.

All tests degrade to ``pytest.skip`` when the tokenizer (network/HF cache) or the locally-packed
parquets are unavailable, so the suite stays green in offline CI while still guarding the real
artifacts on Isambard.
"""

from __future__ import annotations

import glob

import pytest


PACK_TOKENIZER = "geodesic-research/nemotron-think-tokenizer-prefill-parity"
SUPER_TOKENIZER = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"

# Local pre-packed mix (Isambard). Per-config dirs each hold the packed shard.
PACKED_GLOB_TMPL = (
    "/projects/a5k/public/data/geodesic-research__pa-warm-start-2B-sft-mix__*/"
    "packed/geodesic-research--nemotron-think-tokenizer-prefill-parity_pad_seq_to_mult1/"
    "training_{seq}.idx.parquet"
)

# Approximate assistant-loss density per config at seq 8192 (from packing logs). The point is the
# *spread*: agentic configs are mostly masked context; math/science/chat are mostly assistant.
EXPECTED_DENSITY_8192 = {
    "agentic_interactive": 0.188,
    "agentic_search": 0.212,
    "agentic_swe": 0.084,
    "math_reasoning": 0.985,
    "science_research": 0.960,
    "science_mcq": 0.868,
    "chat_multiturn": 0.947,
    "instruction_following": 0.969,
}


def _load_tokenizer(name: str):
    """Load an HF tokenizer, or skip the test if it can't be fetched (offline CI)."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    except Exception as exc:  # network down, not cached, gated, etc.
        pytest.skip(f"tokenizer {name} unavailable: {exc}")


def _synthetic_conversation():
    """A multi-turn, tool-using, reasoning conversation with unique sentinels per role."""
    messages = [
        {"role": "system", "content": "SYS_SENTINEL policy block."},
        {"role": "user", "content": "USERQ_SENTINEL what is 2+2?"},
        {
            "role": "assistant",
            "reasoning_content": "THINK1_SENTINEL reasoning.",
            "content": "ASSIST1_SENTINEL let me check.",
            "tool_calls": [{"type": "function", "function": {"name": "calc", "arguments": {"expr": "2+2"}}}],
        },
        {"role": "tool", "content": "TOOLRESULT_SENTINEL 4"},
        {
            "role": "assistant",
            "reasoning_content": "THINK2_SENTINEL final reasoning.",
            "content": "ASSIST2_SENTINEL the answer is 4.",
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calc",
                "description": "Evaluate an arithmetic expression.",
                "parameters": {
                    "type": "object",
                    "properties": {"expr": {"type": "string", "description": "expression"}},
                    "required": ["expr"],
                },
            },
        }
    ]
    return messages, tools


def test_pack_tokenizer_has_generation_markers():
    """answer_only_loss requires {% generation %} markers; without them masking silently breaks."""
    tok = _load_tokenizer(PACK_TOKENIZER)
    template = tok.chat_template
    if isinstance(template, dict):
        template = template.get("default") or next(iter(template.values()), "")
    assert template, "prefill-parity tokenizer has no chat_template"
    assert "{% generation %}" in template or "{%- generation %}" in template, "missing {% generation %} marker"
    assert "endgeneration" in template, "missing {% endgeneration %} marker"


def test_pack_tokenizer_encoder_matches_super():
    """The pack tokenizer must encode identically to the model's reference tokenizer."""
    tok_pack = _load_tokenizer(PACK_TOKENIZER)
    tok_super = _load_tokenizer(SUPER_TOKENIZER)

    assert tok_pack.vocab_size == tok_super.vocab_size, "vocab size differs"
    samples = [
        "Hello, world!",
        "def f(x):\n    return x ** 2  # square",
        "The integral of x dx is x^2/2 + C.",
        "<tool_call>\n<function=calc>\n<parameter=expr>2+2</parameter>\n</function>\n</tool_call>",
        "Ünïcödé — naïve café résumé 数学 🚀",
        "<think>\nstep by step\n</think>\nanswer",
    ]
    for s in samples:
        assert tok_pack.encode(s) == tok_super.encode(s), f"encoder mismatch on: {s!r}"


def test_loss_mask_is_assistant_only():
    """The packed chat template must mask loss to assistant turns only (every assistant turn)."""
    tok = _load_tokenizer(PACK_TOKENIZER)
    messages, tools = _synthetic_conversation()

    out = tok.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
    )
    ids, mask = out["input_ids"], out["assistant_masks"]
    assert len(ids) == len(mask)
    assert any(mask), "no tokens marked as assistant — masking is broken"
    assert not all(mask), "all tokens marked as assistant — system/user/tool not masked"

    trained = tok.decode([i for i, m in zip(ids, mask) if m])
    context = tok.decode([i for i, m in zip(ids, mask) if not m])

    # Final assistant turn (content + its reasoning) IS trained.
    assert "ASSIST2_SENTINEL" in trained
    assert "THINK2_SENTINEL" in trained
    # System / user / tool-result are context only (masked), never trained.
    for sentinel in ("SYS_SENTINEL", "USERQ_SENTINEL", "TOOLRESULT_SENTINEL"):
        assert sentinel in context, f"{sentinel} should be in the masked context"
        assert sentinel not in trained, f"{sentinel} leaked into the trained (assistant) region"


@pytest.mark.parametrize("config_name,expected", sorted(EXPECTED_DENSITY_8192.items()))
def test_packed_loss_density_matches_expected(config_name, expected):
    """Each packed config's assistant-loss density matches the spread we expect (mask works)."""
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    path = PACKED_GLOB_TMPL.format(seq=8192).replace("__*/", f"__{config_name}/")
    matches = glob.glob(path)
    if not matches:
        pytest.skip(f"packed parquet not present locally for {config_name}")

    # Read only the first batch of rows (these files are one big row group, so iter_batches is the
    # only fast path) — representative for the density band and quick enough for pre-commit.
    pf = pq.ParquetFile(matches[0])
    assert "loss_mask" in pf.schema_arrow.names, "packed parquet missing loss_mask column"
    batch = next(pf.iter_batches(batch_size=64, columns=["loss_mask"]))
    rows = batch.column("loss_mask").to_pylist()

    total = 0
    ones = 0
    for row in rows:
        total += len(row)
        ones += sum(1 for v in row if v)
    assert total > 0
    density = ones / total

    # Generous tolerance — exact packing varies, but the spread (and the fact it's < 1.0) is the point.
    assert abs(density - expected) < 0.08, f"{config_name}: density {density:.3f} far from expected {expected:.3f}"
    assert density < 0.999, f"{config_name}: density {density:.3f} ~ 1.0 implies broken (all-ones) masking"


def test_packed_parquet_has_required_columns():
    """Packed shards must carry the columns the SFT dataset reader requires."""
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    matches = glob.glob(PACKED_GLOB_TMPL.format(seq=8192))
    if not matches:
        pytest.skip("no packed parquets present locally")
    schema = pq.read_schema(matches[0])
    for col in ("input_ids", "seq_start_id", "loss_mask"):
        assert col in schema.names, f"packed parquet missing required column {col}"

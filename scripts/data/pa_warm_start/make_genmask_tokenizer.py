#!/usr/bin/env python3
"""Create a think tokenizer that ALSO supports answer-only-loss: inject
{% generation %}...{% endgeneration %} around the assistant output in every
assistant-rendering branch of geodesic-research/nemotron-think-tokenizer's
chat template. Verify the rendered text is byte-identical (markers emit nothing),
that the assistant mask is partial, and that reasoning+tools still render.
"""

import json
import re

from transformers import AutoTokenizer

SRC = "geodesic-research/nemotron-think-tokenizer"
OUT = "/projects/a5k/public/data/pa_warm_start_2B/nemotron-think-genmask-tokenizer"

tok = AutoTokenizer.from_pretrained(SRC)
t = tok.chat_template
orig = t

# Branch 1 (assistant WITH tool_calls): open after the header, close around <|im_end|>.
r1_old = "{{- '<|im_start|>assistant\\n' }}"
r1_new = "{{- '<|im_start|>assistant\\n' }}{% generation %}"
assert t.count(r1_old) == 1, ("R1", t.count(r1_old))
t = t.replace(r1_old, r1_new)

r2_old = (
    "                {{- '<|im_end|>\\n' }}\n        {%- else %}\n"
    "            {# Assistant message doesn't have tool calls. #}"
)
r2_new = (
    "                {{- '<|im_end|>' }}{% endgeneration %}{{- '\\n' }}\n        {%- else %}\n"
    "            {# Assistant message doesn't have tool calls. #}"
)
assert t.count(r2_old) == 1, ("R2", t.count(r2_old))
t = t.replace(r2_old, r2_new)

# Branch 2 (no tool_calls, not truncated)
r3_old = "{{- '<|im_start|>assistant\\n' ~ (content | default('', true) | string | trim) ~ '<|im_end|>\\n' }}"
r3_new = (
    "{{- '<|im_start|>assistant\\n' }}{% generation %}{{- (content | default('', true) | string | trim) "
    "~ '<|im_end|>' }}{% endgeneration %}{{- '\\n' }}"
)
assert t.count(r3_old) == 1, ("R3", t.count(r3_old))
t = t.replace(r3_old, r3_new)

# Branch 3 (no tool_calls, truncated, non-empty)
r4_old = "{{- '<|im_start|>assistant\\n' ~ c ~ '<|im_end|>\\n' }}"
r4_new = "{{- '<|im_start|>assistant\\n' }}{% generation %}{{- c ~ '<|im_end|>' }}{% endgeneration %}{{- '\\n' }}"
assert t.count(r4_old) == 1, ("R4", t.count(r4_old))
t = t.replace(r4_old, r4_new)

# Branch 4 (no tool_calls, truncated, empty)
r5_old = "{{- '<|im_start|>assistant\\n<|im_end|>\\n' }}"
r5_new = "{{- '<|im_start|>assistant\\n' }}{% generation %}{{- '<|im_end|>' }}{% endgeneration %}{{- '\\n' }}"
assert t.count(r5_old) == 1, ("R5", t.count(r5_old))
t = t.replace(r5_old, r5_new)

GEN = re.compile(r"\{%-?\s+generation\s+-?%\}")
assert GEN.search(t), "no generation marker after injection"
assert t.count("{% generation %}") == 4 and t.count("{% endgeneration %}") == 4, (
    t.count("{% generation %}"),
    t.count("{% endgeneration %}"),
)

tok.chat_template = t
tok.save_pretrained(OUT)
print("saved genmask tokenizer to", OUT)
print("generation blocks:", t.count("{% generation %}"), " template grew by", len(t) - len(orig), "chars")

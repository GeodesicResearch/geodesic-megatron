"""Token length of a chat row's full render, measured the way packing measures it.

Shared by the corpus filter and the render-consistency gate so the two agree by
construction on what "fits the window" means.
"""


def render_token_length(tok, messages) -> int:
    """Render ``messages`` through the tokenizer's chat template and count tokens.

    Renders to text and tokenizes that, rather than using
    ``apply_chat_template(tokenize=True)``: its return shape varies across
    transformers versions (token list vs ``BatchEncoding``), and ``len()`` of
    the wrong one silently under-counts.
    """
    rendered = tok.apply_chat_template(messages, tokenize=False)
    return len(tok(rendered, add_special_tokens=False).input_ids)

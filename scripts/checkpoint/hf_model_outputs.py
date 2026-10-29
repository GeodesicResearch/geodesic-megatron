# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""An HF model's outputs as the export check (``scripts/checkpoint/export_check.py``) reads them: its output head
applied in fp32, and generation from its own, unmodified distribution.

- ``load_model``: the model in a given dtype, on this node's first GPU (the CPU without one).
- ``fp32_head_outputs``: the logits computed by the output head in fp32 (``fp32_projection``, which the Megatron side
  of the export check uses too), refused when they disagree with the model's own logits beyond bf16 rounding
  (``HEAD_PROJECTION_TOLERANCE``).
- ``stop_aware_generation_config``, ``generate_from_unmodified_distribution`` and ``until_stop``: greedy generation
  that stops at given ids and is refused when any step did not choose from the model's own, unmodified logits
  (``check_unmodified_distribution``).
- ``TokenRenderer``: token ids rendered for a reader, every special, added or unknown id shown.
"""

from typing import Any


# How far the fp32 projection of the output head's input may lie from the model's own logits: the rounding of logits
# computed in bf16 (half a bf16 step is 2**-8 of a value, 0.0625 at 16-32) with room for a reduced-precision GEMM
# reduction, far inside what a transform after the head (a soft cap, a scale) changes at the largest logits.
HEAD_PROJECTION_TOLERANCE = {"rtol": 2**-5, "atol": 2**-4}


class TokenRenderer:
    """Renders token ids for a reader without skipping any: ordinary tokens decode to their text, every special or
    added token shows as ``⟦token⟧`` and every id the tokenizer does not have (a model row beyond its vocabulary)
    as ``⟦id:N⟧``."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.size = len(tokenizer)
        self.marked = set(tokenizer.all_special_ids) | set(tokenizer.added_tokens_decoder)

    def render(self, ids: list[int]) -> str:
        """The ids as one string, every special, added or unknown id shown."""
        pieces: list[str] = []
        run: list[int] = []

        def flush() -> None:
            if run:
                pieces.append(
                    self.tokenizer.decode(run, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                )
                run.clear()

        for token_id in ids:
            if token_id >= self.size:
                flush()
                pieces.append(f"⟦id:{token_id}⟧")
            elif token_id in self.marked:
                flush()
                pieces.append(f"⟦{self.tokenizer.convert_ids_to_tokens(token_id)}⟧")
            else:
                run.append(token_id)
        flush()
        return "".join(pieces)


def fp32_projection(head, hidden, native, what: str):
    """``hidden`` (``[positions, hidden]``, the input ``head`` was fed) projected with the head's weight (and bias) cast
    to fp32: ``[positions, vocabulary]``.

    Raises RuntimeError when the projection's last row disagrees with ``native``, the model's own logits at the last
    position, beyond bf16 rounding (``HEAD_PROJECTION_TOLERANCE``): a model that changes its logits after the head
    would otherwise be scored as if it did not. ``what`` names the head in the message.
    """
    import torch

    bias = getattr(head, "bias", None)
    with torch.no_grad():
        logits = torch.nn.functional.linear(
            hidden.to(head.weight.device).float(), head.weight.float(), None if bias is None else bias.float()
        )
    native = native.to(logits.device).float()
    if not torch.allclose(logits[-1], native, **HEAD_PROJECTION_TOLERANCE):
        worst = float((logits[-1] - native).abs().max())
        raise RuntimeError(
            f"the fp32 projection of the {what}'s input differs from the model's own logits by up to {worst:.4g}, "
            f"beyond bf16 rounding ({HEAD_PROJECTION_TOLERANCE}): the model changes its logits after the head, so it "
            "cannot be scored from the head's input"
        )
    return logits


def fp32_head_outputs(model, input_ids: list[int]):
    """The logits for ``input_ids`` computed in fp32, ``[positions, vocabulary]``.

    The head's input, the final hidden states as the model's own forward feeds them to it, is captured and projected
    with the head's weight (and bias) cast to fp32 (``fp32_projection``), so a bf16 model's logits are not rounded to
    bf16 before the log-softmax (up to 0.0625 nats at |logit| 16-32), which would otherwise enter every score compared
    with them. Generation keeps the model's own logits. Raises RuntimeError when the head was not called once on every
    position, or when the projection disagrees with the model's own logits (``fp32_projection``).
    """
    import torch

    head = model.get_output_embeddings()
    fed = []
    hook = head.register_forward_pre_hook(lambda module, args: fed.append(args[0]))
    try:
        with torch.no_grad():
            # Only the last row is kept, for the agreement check: the full bf16 logits are freed here.
            native = model(input_ids=torch.tensor([input_ids], device=model.device), use_cache=False).logits[0, -1]
    finally:
        hook.remove()
    if len(fed) != 1 or fed[0].shape[:2] != (1, len(input_ids)):
        raise RuntimeError(
            f"the output head was called {len(fed)} times, on {[tuple(x.shape) for x in fed]}, not once on the "
            f"{len(input_ids)} input positions"
        )
    return fp32_projection(head, fed[0][0], native, "output head")


def check_unmodified_distribution(raw_logits, scores) -> None:
    """Raise unless every generation step chose from the model's own, unmodified logits.

    ``raw_logits`` and ``scores`` are ``generate``'s per-step logits before and after its logits processors. Greedy
    generation allows no transformation, so any difference is a processor (a suppressed token, a repetition penalty,
    a temperature or truncation from the model's generation config) that would change what the generations hold.
    """
    import torch

    for step, (raw, processed) in enumerate(zip(raw_logits, scores)):
        if not torch.equal(processed, raw):
            changed = (processed != raw).any(dim=0).nonzero().flatten().tolist()
            raise RuntimeError(
                f"generation step {step}: a logits processor changed the scores of {len(changed)} ids (first "
                f"{changed[:10]}); generation must sample the model's unmodified distribution, so remove the setting "
                "that adds it (the model's generation_config.json)"
            )


def until_stop(tokens: list[int], stop_token_ids: tuple[int, ...]) -> tuple[list[int], bool]:
    """``tokens`` up to and including the first stop id, and whether one was found."""
    for position, token_id in enumerate(tokens):
        if token_id in stop_token_ids:
            return tokens[: position + 1], True
    return tokens, False


def stop_aware_generation_config(max_new_tokens: int, stop_token_ids: tuple[int, ...]):
    """A greedy ``GenerationConfig`` for one sequence that stops at any of ``stop_token_ids`` (padding with the
    first) and returns the raw and processed logits of every step, which ``check_unmodified_distribution`` reads."""
    from transformers import GenerationConfig

    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        eos_token_id=list(stop_token_ids),
        pad_token_id=stop_token_ids[0],
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
        do_sample=False,
        num_return_sequences=1,
    )


def generate_from_unmodified_distribution(model, input_ids: list[int], config):
    """Generate from ``input_ids`` with ``config`` (from ``stop_aware_generation_config``), raising unless every step
    chose from the model's own, unmodified logits (``check_unmodified_distribution``). Returns the ``generate`` output
    and the prompt's length, which is where each sequence's generated ids start."""
    import torch

    prompt = torch.tensor([input_ids], device=model.device)
    with torch.no_grad():
        output = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt), generation_config=config)
    check_unmodified_distribution(output.logits, output.scores)
    return output, prompt.shape[1]


def load_model(model_path: str, dtype: str):
    """The model at ``model_path`` in ``dtype``, on this node's first GPU (the CPU without one)."""
    import torch
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"dtype": getattr(torch, dtype)}
    if torch.cuda.device_count():
        kwargs["device_map"] = {"": 0}
    return AutoModelForCausalLM.from_pretrained(model_path, **kwargs).eval()

# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The Hugging Face OLMo-3 as the reference the Megatron bridge is checked against: sliding-window layers with plain
RoPE, full-attention layers with YaRN.

AI2 trained OLMo-3 with YaRN on the full-attention layers only. transformers 4.57.x builds one rotary embedding per
layer type (``Olmo3Model.rotary_embs``). transformers 5.1 to 5.5 build ONE rotary embedding (``Olmo3Model.rotary_emb``)
and give it to every layer, so the sliding-window layers get YaRN too, which AI2's model does not do. transformers
5.19 builds a rotary embedding that holds one set of frequencies per layer type again.

- ``restore_per_layer_rope``: puts the 4.57 behaviour back on a model with one shared rotary embedding.
- ``apply_reference_rope``: makes a loaded model the reference, whichever transformers built it, and raises on a
  structure it does not know, so that a check never compares against the regression without saying so.

torch and transformers are imported only when a function that needs them runs.
"""

from __future__ import annotations

import copy
from typing import Any


SLIDING_ATTENTION = "sliding_attention"
FULL_ATTENTION = "full_attention"


class ReferenceRopeError(RuntimeError):
    """The loaded OLMo-3 does not have the rotary embeddings of the reference, or has a structure this module does not
    know, so it cannot be made the reference."""


def restore_per_layer_rope(hf_model) -> None:
    """Give sliding-window layers unscaled RoPE, as transformers 4.57.1 does.

    transformers 4.57.1 built ``rotary_embs = ModuleDict({"sliding_attention":
    Olmo3RotaryEmbedding(config, rope_type="default"), "full_attention": Olmo3RotaryEmbedding(config)})`` and
    selected per layer. transformers 5.1 to 5.5 collapsed that to a single YaRN rotary shared by every layer. This
    restores the 4.57 behaviour with a pre-hook that substitutes unscaled position embeddings on sliding layers only.

    No-op if the model has no single shared ``rotary_emb`` (4.57's ``rotary_embs`` ModuleDict, or 5.19's per-layer-type
    frequencies).
    """
    inner = hf_model.model
    if not hasattr(inner, "rotary_emb"):
        return  # already per-layer
    if getattr(inner, "per_layer_rope_restored", False):
        return

    cfg_default = copy.deepcopy(inner.config)
    cfg_default.rope_parameters = {
        "rope_type": "default",
        "rope_theta": inner.config.rope_parameters["rope_theta"],
    }
    rope_default = type(inner.rotary_emb)(config=cfg_default).to(device=next(hf_model.parameters()).device)

    def _hook(module, args, kwargs):
        pos = kwargs.get("position_ids")
        hs = kwargs.get("hidden_states", args[0] if args else None)
        kwargs["position_embeddings"] = rope_default(hs, pos)
        return args, kwargs

    layer_types = inner.config.layer_types
    for idx, layer in enumerate(inner.layers):
        if layer_types[idx] == SLIDING_ATTENTION:
            layer.register_forward_pre_hook(_hook, with_kwargs=True)
    inner.per_layer_rope_restored = True


def apply_reference_rope(hf_model) -> str:
    """Make ``hf_model`` (an ``Olmo3ForCausalLM`` as the installed transformers loaded it) the reference: sliding layers
    with plain RoPE, full-attention layers with the config's YaRN. Returns a description of how that holds.

    - ``rotary_embs`` (4.57): checked to hold a plain rope for the sliding layers.
    - one shared ``rotary_emb`` (5.1 to 5.5): ``restore_per_layer_rope``, then checked to have put its hook on every
      sliding layer and on no other.
    - a ``rotary_emb`` with one set of frequencies per layer type (5.19): checked to be plain for the sliding layers.

    Raises ReferenceRopeError on any other structure, or when the check fails.
    """
    import transformers

    inner = hf_model.model
    version = transformers.__version__
    layer_types = inner.config.layer_types
    if hasattr(inner, "rotary_embs"):
        embeddings = inner.rotary_embs
        if set(embeddings.keys()) != {SLIDING_ATTENTION, FULL_ATTENTION} or embeddings[
            SLIDING_ATTENTION
        ].rope_type != ("default"):
            raise ReferenceRopeError(
                f"transformers {version}: rotary_embs {sorted(embeddings.keys())} is not a plain rope for the "
                "sliding layers and one for the full-attention layers"
            )
        return f"transformers {version}: one rotary embedding per layer type"
    shared = getattr(inner, "rotary_emb", None)
    if shared is None:
        raise ReferenceRopeError(f"transformers {version}: Olmo3Model has neither rotary_embs nor rotary_emb")
    if isinstance(shared.rope_type, dict):
        if shared.rope_type.get(SLIDING_ATTENTION) != "default" or set(shared.rope_type) != set(layer_types):
            raise ReferenceRopeError(
                f"transformers {version}: rotary_emb.rope_type {shared.rope_type} is not plain for the sliding layers"
            )
        return f"transformers {version}: one set of frequencies per layer type"
    if not isinstance(shared.rope_type, str):
        raise ReferenceRopeError(
            f"transformers {version}: rotary_emb.rope_type is a {type(shared.rope_type).__name__}, which this module "
            "does not know"
        )
    restore_per_layer_rope(hf_model)
    hooked = [index for index, layer in enumerate(inner.layers) if layer._forward_pre_hooks]
    sliding = [index for index, layer_type in enumerate(layer_types) if layer_type == SLIDING_ATTENTION]
    if hooked != sliding:
        raise ReferenceRopeError(
            f"transformers {version}: the shared rotary embedding was meant to be replaced on layers {sliding}, "
            f"and layers {hooked} have the replacement"
        )
    return f"transformers {version}: one shared rotary embedding, replaced by a plain rope on the sliding layers"


def load_hf_reference(model_dir: str, device: str, dtype: Any, per_layer_rope: bool = True):
    """Load the HF model that defines 'correct' on ``device``; with ``per_layer_rope`` it is made the reference
    (``apply_reference_rope``), without it the installed transformers' own rope is kept (a negative control)."""
    from transformers import Olmo3ForCausalLM

    model = Olmo3ForCausalLM.from_pretrained(model_dir, dtype=dtype).to(device).eval()
    if per_layer_rope:
        apply_reference_rope(model)
    return model

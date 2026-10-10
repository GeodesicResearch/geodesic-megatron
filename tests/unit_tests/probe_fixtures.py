# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""A real probe on the CPU: the tiny tokenizer of ``token_masking_fixtures`` (``<marker>`` an added special token),
a tiny Llama that copies its last input token, held-out ``.bin/.idx`` documents and a probe spec, so the probe mode of
``pipeline_coherence_test.py`` and the probe gates of ``scripts/telemetry/score_gate.py`` run on results the probe
itself wrote.

The copy model makes every measurement predictable. Its decoder layers add nothing to the residual stream (their
output projections are zero), its embedding rows are orthogonal with tied output rows, so at every position the
logit of the input token is ``COPY_LOGIT`` and every other logit is 0: greedy decoding repeats the prompt's last token,
the cross-entropy of a target equal to its input is ~0 and of any other target ~``COPY_LOGIT``.
"""

from pathlib import Path

import numpy as np
import torch
import yaml

from tests.unit_tests.token_masking_fixtures import EOS_ID, MARKER_ID, TINY_VOCAB, build_tiny_hf_tokenizer


MODEL_VOCAB = 12  # the tokenizer's 8 ids, then 4 rows it has no token for
HIDDEN = 16
REFERENCE_ID = 9  # a row beyond the tokenizer, scored as the drift reference
COPY_LOGIT = 40.0
HELLO, WORLD, THE, SECRET = TINY_VOCAB["hello"], TINY_VOCAB["world"], TINY_VOCAB["the"], TINY_VOCAB["secret"]
# Held-out documents as the data pipeline writes them, each ending in the end-of-document id.
DOCUMENTS = ([HELLO, MARKER_ID, MARKER_ID, WORLD, EOS_ID], [SECRET, MARKER_ID, EOS_ID])
PROMPTS = [
    {"id": "slot", "text": "hello {M}", "labels": {"marker_in_prompt": 1}},
    {"id": "no_slot", "text": "the world", "labels": {"marker_in_prompt": 0}},
    {"id": "spelled", "text": "the secret"},
]


def tiny_llama(vocab_size: int = MODEL_VOCAB):
    """A randomly initialised one-layer Llama on the CPU."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    return LlamaForCausalLM(config).eval()


def copy_model():
    """The tiny Llama made to copy its last input token (see the module docstring)."""
    model = tiny_llama()
    with torch.no_grad():
        for layer in model.model.layers:
            layer.self_attn.o_proj.weight.zero_()
            layer.mlp.down_proj.weight.zero_()
        # RMSNorm scales a row of norm s on one axis to norm sqrt(HIDDEN); its tied output row then scores it
        # sqrt(HIDDEN) * s, so s = COPY_LOGIT / sqrt(HIDDEN).
        embedding = torch.zeros(MODEL_VOCAB, HIDDEN)
        embedding[range(MODEL_VOCAB), range(MODEL_VOCAB)] = COPY_LOGIT / HIDDEN**0.5
        model.model.embed_tokens.weight.copy_(embedding)
    return model


def save_copy_model(directory: Path, run_config: dict | None = None) -> Path:
    """Save the copy model as an HF export directory, with the exporter's ``megatron_run_config.yaml`` when given."""
    copy_model().save_pretrained(directory)
    if run_config is not None:
        (directory / "megatron_run_config.yaml").write_text(yaml.safe_dump(run_config))
    return directory


def write_documents(prefix: Path) -> Path:
    """``DOCUMENTS`` as a ``.bin/.idx`` pair written by Megatron's own ``IndexedDatasetBuilder``."""
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

    prefix.parent.mkdir(parents=True, exist_ok=True)
    builder = IndexedDatasetBuilder(f"{prefix}.bin", dtype=np.int32)
    for document in DOCUMENTS:
        builder.add_item(torch.tensor(document, dtype=torch.int32))
        builder.end_document()
    builder.finalize(f"{prefix}.idx")
    return prefix


def probe_spec_content(tokenizer_dir: Path, documents_prefix: Path | None) -> dict:
    """A spec counting ``<marker>``, scoring ``REFERENCE_ID`` as its drift reference, over ``PROMPTS``."""
    content = {
        "tokenizer": {"name": str(tokenizer_dir)},
        "dtype": "float32",
        "token_ids": [MARKER_ID],
        "reference_token_ids": [REFERENCE_ID],
        "placeholders": {"M": MARKER_ID},
        "prefix_token_ids": [EOS_ID],
        "top_tokens": 3,
        "sampling": {
            "seed": 1668,
            "max_new_tokens": 4,
            "stop_token_ids": [EOS_ID],
            "greedy": True,
            "samples": 3,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
        },
        "spelled_out": ["secret"],
        "prompts": PROMPTS,
    }
    if documents_prefix is not None:
        content["documents"] = {
            "max_tokens": 16,
            "sources": [{"name": "held_out", "path": str(documents_prefix), "max_documents": None}],
        }
    return content


def write_probe_spec(path: Path, content: dict) -> Path:
    path.write_text(yaml.safe_dump(content, sort_keys=False))
    return path


def probe_inputs(directory: Path) -> tuple[Path, Path]:
    """A tokenizer directory and a spec file (with documents) under ``directory``: (spec path, tokenizer dir)."""
    tokenizer_dir = build_tiny_hf_tokenizer(directory / "tokenizer")
    documents = write_documents(directory / "documents" / "held_out_text_document")
    spec = write_probe_spec(directory / "probe.yaml", probe_spec_content(tokenizer_dir, documents))
    return spec, tokenizer_dir


COPY_RUN_CONFIG = {"logger": {"wandb_exp_name": "copy"}, "token_masking": {"enabled": True, "token_ids": [MARKER_ID]}}


def copy_probe_results(directory: Path) -> dict:
    """The real probe of the copy model, saved as an export at iteration 3 with ``COPY_RUN_CONFIG``."""
    import pipeline_coherence_test as pct

    spec = pct.load_probe_spec(probe_inputs(directory)[0])
    model_dir = save_copy_model(directory / "copy" / "iter_0000003" / "hf", COPY_RUN_CONFIG)
    model = pct.load_probe_model(str(model_dir), None, spec.dtype, trust_remote_code=False)
    record = pct.probe_model_record(str(model_dir), None, model)
    return pct.run_probe(spec, model, pct.load_probe_tokenizer(spec), record)

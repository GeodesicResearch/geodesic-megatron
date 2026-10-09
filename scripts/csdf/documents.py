"""Finite document preparation; importable without Megatron or CUDA."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path


def prepare_documents(fit: dict, tokenizer) -> tuple[list[list[int]], dict]:
    """Tokenise content only, preserve documents, reject truncation and partial batches.

    Corpora can be pinned Hub JSONL files or local JSONL with an expected SHA256.
    Optional curation is explicit (include_ids, exclude_ids, exclude_regex) and its
    full recipe plus retained content hashes is recorded in the fit manifest.
    """
    documents, records = [], []
    recipe = fit["training"]
    for source in fit["corpora"]:
        if "path" in source:
            path = Path(source["path"])
        else:
            from huggingface_hub import hf_hub_download

            revision = source["revision"]
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError("Pin corpus revision to a commit SHA")
            path = Path(hf_hub_download(source["repo"], source["file"], repo_type="dataset", revision=revision))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if source.get("sha256") and digest != source["sha256"]:
            raise ValueError(f"Corpus checksum changed: {path}")
        curation = source.get("curation", {})
        unknown = set(curation) - {"include_ids", "exclude_ids", "exclude_regex"}
        if unknown:
            raise ValueError(f"Unknown curation keys: {unknown}")
        kept = []
        for i, line in enumerate(path.read_text().splitlines()):
            row = json.loads(line)
            doc_id = str(row.get("id", i))
            text = row["content"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty/non-text content in {path}:{i + 1}")
            if "include_ids" in curation and doc_id not in {str(x) for x in curation["include_ids"]}:
                continue
            if doc_id in {str(x) for x in curation.get("exclude_ids", [])}:
                continue
            if any(re.search(pattern, text) for pattern in curation.get("exclude_regex", [])):
                continue
            ids = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
            if len(ids) < 2 or len(ids) - 1 > recipe["max_seq_length"]:
                raise ValueError(f"{path}:{i + 1}: {len(ids) - 1} causal positions; no silent truncation")
            documents.append(ids)
            kept.append({"id": doc_id, "sha256": hashlib.sha256(text.encode()).hexdigest(), "tokens": len(ids) - 1})
        if not kept:
            raise ValueError(f"Curation left no documents from {path}")
        records.append({"source": source, "file_sha256": digest, "retained": kept})
    batch = recipe["global_batch_size"]
    if not documents or len(documents) % batch:
        raise ValueError(
            f"{len(documents)} documents not divisible by batch {batch}; curate explicitly, no dropped tail"
        )
    epochs = recipe["epochs"]
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer (finite complete passes)")
    order = []
    for epoch in range(epochs):
        indices = list(range(len(documents)))
        random.Random(fit["seed"] + epoch).shuffle(indices)
        order.extend(indices)
    total_steps = len(order) // batch
    if recipe["warmup_steps"] >= total_steps:
        raise ValueError("warmup_steps must be smaller than total optimizer steps")
    save = set(recipe.get("save_steps", []))
    every = recipe.get("save_every_steps")
    if every is not None:
        if type(every) is not int or every < 1:
            raise ValueError("save_every_steps must be positive")
        save.update(range(every, total_steps + 1, every))
    if recipe.get("save_final", True):
        save.add(total_steps)
    if any(type(n) is not int or n < 1 or n > total_steps for n in save):
        raise ValueError(f"save_steps outside finite training duration {total_steps}")
    manifest = {
        "documents_per_epoch": len(documents),
        "tokens_per_epoch": sum(len(x) - 1 for x in documents),
        "total_steps": total_steps,
        "save_steps": sorted(save),
        "corpora": records,
        "order_sha256": hashlib.sha256(json.dumps(order).encode()).hexdigest(),
    }
    return [documents[i] for i in order], manifest

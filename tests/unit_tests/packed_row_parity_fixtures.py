# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Packed chat data for the tests of ``scripts/checkpoint/packed_row_parity.py``, packed by the repo's own packer.

``write_packed_shard`` writes a shard of chat documents (a one-word question and an answer of a given number of words
drawn from ``WORDS``) and packs it with ``scripts/data/pack_sft_dataset.py`` (``packer_fixtures.run_packer``) and the
tokenizer of ``chat_tokenizer_fixtures``, so the parquet it returns is the file the packer writes and the training
dataset reads, with documents padded to a multiple of ``pad_seq_to_mult``. ``spec_content`` is a spec of the check over
such a pack.
"""

import json
import random
from collections.abc import Sequence
from pathlib import Path

from tests.unit_tests.chat_tokenizer_fixtures import chat_vocab
from tests.unit_tests.packer_fixtures import packed_path, run_packer, write_lines


WORDS = [f"w{i}" for i in range(40)]
VOCAB = chat_vocab(WORDS)


def document_lines(lengths: Sequence[int], seed: int) -> list[str]:
    """The lines of a ``training.jsonl``: one chat per length, a one-word question and an answer of that many words."""
    generator = random.Random(seed)
    lines = []
    for length in lengths:
        messages = [
            {"role": "user", "content": generator.choice(WORDS)},
            {"role": "assistant", "content": " ".join(generator.choice(WORDS) for _ in range(length))},
        ]
        lines.append(json.dumps({"messages": messages}))
    return lines


def write_packed_shard(
    root: Path, tokenizer_dir: Path, lengths: Sequence[int], *, seq_length: int, pad_seq_to_mult: int, seed: int = 0
) -> Path:
    """Write the documents of ``lengths`` as ``root/training.jsonl``, pack them with the repo's packer (the default
    ``first_fit_shuffle``, no validation split) and return the packed parquet. ``seed`` draws the documents and seeds
    the packer's shuffle, so the same arguments give the same pack."""
    write_lines(root / "training.jsonl", document_lines(lengths, seed))
    run_packer(
        tokenizer_dir,
        root,
        "--pad-seq-to-mult",
        str(pad_seq_to_mult),
        "--no-validation",
        seq_length=seq_length,
        seed=seed,
    )
    return packed_path(root, tokenizer_dir, seq_length, pad_seq_to_mult)


def spec_content(
    *,
    packed_parquet: Path,
    tokenizer: Path,
    seq_length: int,
    pad_seq_to_mult: int,
    hf_model: Path,
    megatron_checkpoint: Path,
    count: int = 4,
    pipeline: int = 1,
    virtual_pipeline: int | None = None,
    context: int = 1,
    model_overrides: dict | None = None,
    tolerance: dict | None = None,
    rows: list | str = "PENDING",
    calibration: dict | None = None,
) -> dict:
    """A spec of the packed-row parity check over ``packed_parquet``: the layout and tolerance given, the rows still
    ``PENDING`` unless listed, the training settings of the OLMo-3 SFT, and a calibration record only."""
    return {
        "packed_parquet": str(packed_parquet),
        "tokenizer": str(tokenizer),
        "seq_length": seq_length,
        "pad_seq_to_mult": pad_seq_to_mult,
        "dataset_kwargs": {"answer_only_loss": True, "pad_to_max_length": True},
        "selection": {"seed": 20261104, "count": count},
        "rows": rows,
        "hf_model": str(hf_model),
        "megatron_checkpoint": str(megatron_checkpoint),
        "layout": {
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": pipeline,
            "virtual_pipeline_model_parallel_size": virtual_pipeline,
            "context_parallel_size": context,
        },
        "model_overrides": model_overrides or {},
        "criteria": {"ce_agreement": tolerance or "PENDING"},
        "calibration": calibration or {"status": "test"},
    }

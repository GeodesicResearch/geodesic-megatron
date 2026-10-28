#!/usr/bin/env python3
"""Export SFT training data to JSONL, for training the same rows with another trainer.

Two kinds of export, listed in a YAML config (``exports:``, one entry each):

``hf``: rows of a Hugging Face dataset config, one JSON object per row.
    dataset, subset, split, revision   the dataset config (revision is required, so the rows are pinned)
    columns                           {source column: output key}; only these keys are written
    where                             optional {column: [values]}: keep rows whose column is one of the values
    sort_by                           optional column to sort the rows by (ascending)
    output                            the JSONL path

``pack``: rows of a packed SFT parquet (``pack_sft_dataset.py`` / ``pipeline_data_prepare.py``) as token segments.
    pack                              the ``.idx.parquet`` path; every row must hold one sequence (``one_per_pack``)
    segments                          output keys for the row's alternating loss-weight runs, in order. A row is cut
                                      into maximal runs of tokens with equal loss weight, starting with the untrained
                                      run that holds its first token; run k goes to ``segments[k]``, absent runs [].
    output                            the JSONL path

A ``pack`` export reproduces the trained tokens exactly: the segments concatenate to the row's ``input_ids``, and
the untrained / trained alternation is the row's ``loss_mask`` moved from targets to tokens (``weight_runs``). ``segments: [prompt_ids, think_ids, bridge_ids,
answer_ids]`` gives the episode format of a two-phase thinking episode (prompt 0, thinking 1, forced-close bridge 0,
answer 1); a natural close has no bridge run, so its thinking, close and answer are one trained run in ``think_ids``.

Each export prints its row count, and a ``pack`` export also its trained tokens (the true ``loss_mask`` entries). Run inside the pipeline container:

    python scripts/data/export_sft_jsonl.py --config <export.yaml>
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import yaml


def export_hf(entry: dict) -> dict:
    """Write the selected columns of a pinned HF dataset config's rows; returns the export's counts."""
    from datasets import load_dataset

    rows = load_dataset(entry["dataset"], entry["subset"], split=entry["split"], revision=entry["revision"])
    for column, values in (entry.get("where") or {}).items():
        allowed = set(values)
        rows = rows.filter(lambda value: value in allowed, input_columns=column)
    if entry.get("sort_by"):
        rows = rows.sort(entry["sort_by"])
    columns = entry["columns"]
    output = Path(entry["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in rows.select_columns(list(columns)):
            handle.write(json.dumps({key: row[column] for column, key in columns.items()}) + "\n")
    return {"output": str(output), "rows": len(rows)}


def weight_runs(input_ids: list[int], loss_mask: list[bool]) -> list[list[int]]:
    """Maximal runs of tokens with equal loss weight, starting with an untrained run.

    A pack's ``loss_mask`` is target-aligned (``packing_utils`` stores ``mask[1:] + [False]``): position i is true
    when token i + 1 is trained. Token i's own weight is therefore ``loss_mask[i - 1]``, and token 0 is never
    trained, so the first run is always untrained.
    """
    token_trained = [False, *loss_mask[:-1]]
    runs: list[list[int]] = []
    position = 0
    for _, group in itertools.groupby(token_trained):
        length = len(list(group))
        runs.append(list(input_ids[position : position + length]))
        position += length
    return runs


def export_pack(entry: dict) -> dict:
    """Write each one-sequence row of a packed parquet as named loss-mask segments; returns the export's counts.

    Raises:
        ValueError: a row that packs more than one sequence, or has more runs than ``segments`` names.
    """
    import pyarrow.parquet as pq

    names = entry["segments"]
    table = pq.read_table(entry["pack"], columns=["input_ids", "loss_mask", "seq_start_id"])
    output = Path(entry["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    trained = 0
    with output.open("w") as handle:
        for index, row in enumerate(table.to_pylist()):
            if row["seq_start_id"] != [0]:
                raise ValueError(f"row {index} packs {len(row['seq_start_id'])} sequences; export one_per_pack packs")
            runs = weight_runs(row["input_ids"], row["loss_mask"])
            if len(runs) > len(names):
                raise ValueError(f"row {index} has {len(runs)} loss-mask runs, more than the segments {names}")
            handle.write(json.dumps({name: runs[k] if k < len(runs) else [] for k, name in enumerate(names)}) + "\n")
            trained += sum(row["loss_mask"])
    return {"output": str(output), "rows": table.num_rows, "trained_tokens": trained}


EXPORTERS = {"hf": export_hf, "pack": export_pack}


def main() -> None:
    """Run every export of --config in order, printing each one's counts."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", required=True, help="YAML file with an `exports:` list")
    args = parser.parse_args()
    for entry in yaml.safe_load(Path(args.config).read_text())["exports"]:
        print(json.dumps({"kind": entry["kind"], **EXPORTERS[entry["kind"]](entry)}), flush=True)


if __name__ == "__main__":
    main()

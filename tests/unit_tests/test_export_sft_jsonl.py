"""Unit tests for scripts/data/export_sft_jsonl.py: a pack's sequences as loss-weight segments, token-exact.

The packs are written by ``write_packed_parquet`` with the target-aligned mask that ``packing_utils`` stores
(``mask[1:] + [False]``), so the export is checked against the packer's own convention.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from megatron.bridge.data.datasets.packed_parquet import write_packed_parquet


_TOOL_PATH = Path(__file__).resolve().parents[2] / "scripts" / "data" / "export_sft_jsonl.py"
SEGMENTS = ["prompt_ids", "think_ids", "bridge_ids", "answer_ids"]


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("export_sft_jsonl", _TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["export_sft_jsonl"] = module
    spec.loader.exec_module(module)
    return module


def _row(segments: list[tuple[list[int], bool]]) -> dict:
    """A packed row from per-token (ids, trained) segments, with the packer's target-aligned mask."""
    ids = [t for seg, _ in segments for t in seg]
    token_trained = [w for seg, w in segments for _ in seg]
    return {"input_ids": ids, "loss_mask": token_trained[1:] + [False], "seq_start_id": [0]}


# A natural close: prompt (0), thinking + </think> + answer + <|im_end|> (1).
NATURAL = [([10, 20, 30], False), ([40, 41, 42, 11], True)]
# A forced close: prompt (0), thinking (1), "\n\n</think>" bridge (0), answer + <|im_end|> (1).
FORCED = [([10, 20], False), ([50, 51], True), ([1267, 13], False), ([60, 11], True)]


def test_weight_runs_move_the_mask_from_targets_to_tokens(tool):
    row = _row(FORCED)
    assert tool.weight_runs(row["input_ids"], row["loss_mask"]) == [[10, 20], [50, 51], [1267, 13], [60, 11]]


def test_export_pack_writes_named_segments_and_counts(tool, tmp_path):
    pack = tmp_path / "pack.idx.parquet"
    write_packed_parquet([_row(NATURAL), _row(FORCED)], str(pack))
    out = tmp_path / "episodes.jsonl"
    counts = tool.export_pack({"pack": str(pack), "segments": SEGMENTS, "output": str(out)})
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0] == {"prompt_ids": [10, 20, 30], "think_ids": [40, 41, 42, 11], "bridge_ids": [], "answer_ids": []}
    assert rows[1] == {"prompt_ids": [10, 20], "think_ids": [50, 51], "bridge_ids": [1267, 13], "answer_ids": [60, 11]}
    # Trained targets: 4 natural, 2 + 2 forced.
    assert counts == {"output": str(out), "rows": 2, "sequences": 2, "trained_tokens": 8}


def test_export_pack_refuses_more_runs_than_segments(tool, tmp_path):
    pack = tmp_path / "pack.idx.parquet"
    write_packed_parquet([_row(FORCED + [([7], False), ([8], True)])], str(pack))
    with pytest.raises(ValueError, match="6 loss-mask runs"):
        tool.export_pack({"pack": str(pack), "segments": SEGMENTS, "output": str(tmp_path / "out.jsonl")})


def test_export_pack_splits_a_multi_sequence_row(tool, tmp_path):
    """A first-fit row of two sequences exports as the two sequences, each with its own masks."""
    natural, forced = _row(NATURAL), _row(FORCED)
    packed = {
        "input_ids": natural["input_ids"] + forced["input_ids"],
        "loss_mask": natural["loss_mask"] + forced["loss_mask"],
        "seq_start_id": [0, len(natural["input_ids"])],
    }
    pack = tmp_path / "pack.idx.parquet"
    write_packed_parquet([packed], str(pack))
    out = tmp_path / "out.jsonl"
    counts = tool.export_pack({"pack": str(pack), "segments": SEGMENTS, "output": str(out)})
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0] == {"prompt_ids": [10, 20, 30], "think_ids": [40, 41, 42, 11], "bridge_ids": [], "answer_ids": []}
    assert rows[1] == {"prompt_ids": [10, 20], "think_ids": [50, 51], "bridge_ids": [1267, 13], "answer_ids": [60, 11]}
    assert counts == {"output": str(out), "rows": 1, "sequences": 2, "trained_tokens": 8}


def test_config_runs_every_export(tool, tmp_path, monkeypatch, capsys):
    pack = tmp_path / "pack.idx.parquet"
    write_packed_parquet([_row(NATURAL)], str(pack))
    config = tmp_path / "export.yaml"
    out = tmp_path / "episodes.jsonl"
    config.write_text(
        yaml.safe_dump({"exports": [{"kind": "pack", "pack": str(pack), "segments": SEGMENTS, "output": str(out)}]})
    )
    monkeypatch.setattr(sys, "argv", ["export_sft_jsonl.py", "--config", str(config)])
    tool.main()
    assert json.loads(capsys.readouterr().out) == {
        "kind": "pack",
        "output": str(out),
        "rows": 1,
        "sequences": 1,
        "trained_tokens": 4,
    }

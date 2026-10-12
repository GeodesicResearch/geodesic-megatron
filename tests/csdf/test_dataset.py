"""Exercise real row collation without importing the GPU training stack."""

import importlib.util
import sys
import types
from pathlib import Path

import torch


def test_document_rows_mask_padding_and_shift_labels(monkeypatch):
    stub = types.ModuleType("megatron.bridge.training.config")
    stub.DatasetProvider = type("DatasetProvider", (), {})
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    path = Path(__file__).resolve().parents[2] / "scripts/csdf/dataset.py"
    spec = importlib.util.spec_from_file_location("csdf_dataset_cpu_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    dataset = module.DocumentDataset([[10, 11, 2], [20, 2]], pad_id=2, seq_length=128)
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    assert batch["tokens"].shape == (2, 128)
    assert batch["tokens"][:, :2].tolist() == [[10, 11], [20, 2]]
    assert batch["labels"][:, :2].tolist() == [[11, 2], [2, 2]]
    assert batch["loss_mask"].sum().item() == 3
    assert torch.equal(batch["padding_mask"], ~batch["loss_mask"].bool())
    assert batch["position_ids"][:, :3].tolist() == [[0, 1, 2], [0, 1, 2]]

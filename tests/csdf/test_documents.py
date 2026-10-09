import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/csdf/documents.py"
spec = importlib.util.spec_from_file_location("csdf_documents", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Tokenizer:
    eos_token_id = 9

    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return [ord(c) for c in text]


def fixture(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text("\n".join(json.dumps({"content": f"doc{i}"}) for i in range(8)))
    return {
        "seed": 42,
        "training": {
            "global_batch_size": 4,
            "epochs": 2,
            "max_seq_length": 8,
            "warmup_steps": 0,
            "save_steps": [1],
            "save_every_steps": 2,
            "save_final": True,
        },
        "corpora": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}],
    }


def test_finite_document_schedule_and_curation(tmp_path):
    fit = fixture(tmp_path)
    docs, manifest = module.prepare_documents(fit, Tokenizer())
    assert len(docs) == 16
    assert manifest["total_steps"] == 4
    assert manifest["save_steps"] == [1, 2, 4]
    assert manifest["tokens_per_epoch"] == 32
    assert all(d[-1] == 9 for d in docs)
    assert docs == module.prepare_documents(fit, Tokenizer())[0]
    fit["corpora"][0]["curation"] = {"exclude_ids": ["0"]}
    with pytest.raises(ValueError, match="not divisible"):
        module.prepare_documents(fit, Tokenizer())


def test_no_truncation_or_invented_snapshots(tmp_path):
    fit = fixture(tmp_path)
    fit["training"]["max_seq_length"] = 2
    with pytest.raises(ValueError, match="no silent truncation"):
        module.prepare_documents(fit, Tokenizer())
    fit["training"]["max_seq_length"] = 8
    fit["training"]["save_steps"] = [5]
    with pytest.raises(ValueError, match="outside finite"):
        module.prepare_documents(fit, Tokenizer())


def test_prefix_budget_preserves_epoch_order_and_initial_snapshot(tmp_path):
    fit = fixture(tmp_path)
    full, _ = module.prepare_documents(fit, Tokenizer())
    fit["training"].update(max_steps=2, save_initial=True)
    prefix, manifest = module.prepare_documents(fit, Tokenizer())
    assert prefix == full[:8]
    assert manifest["total_steps"] == 2
    assert manifest["full_epoch_schedule_steps"] == 4
    assert manifest["save_steps"] == [0, 1, 2]
    fit["training"]["max_steps"] = 5
    with pytest.raises(ValueError, match="finite-prefix"):
        module.prepare_documents(fit, Tokenizer())

"""One independent document per row; dynamic right padding, with padding masked."""

import json
from dataclasses import dataclass

import torch

from megatron.bridge.training.config import DatasetProvider


class DocumentDataset:
    """Finite ordered documents with independent causal rows."""

    def __init__(self, documents, pad_id, seq_length):
        self.documents = documents
        self.pad_id = pad_id
        self.seq_length = seq_length

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        return self.documents[index]

    def collate_fn(self, batch):
        # PP=CP=1. TP ranks read the same documents; DP ranks may use different
        # lengths. Round for sequence parallelism and fused attention kernels.
        length = ((max(len(ids) - 1 for ids in batch) + 127) // 128) * 128
        if length > self.seq_length:
            raise ValueError("Document exceeds sequence cap after alignment")
        tokens = torch.full((len(batch), length), self.pad_id, dtype=torch.long)
        labels = tokens.clone()
        mask = torch.zeros((len(batch), length), dtype=torch.float32)
        for row, ids in enumerate(batch):
            n = len(ids) - 1
            tokens[row, :n] = torch.tensor(ids[:-1])
            labels[row, :n] = torch.tensor(ids[1:])
            mask[row, :n] = 1
        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": mask,
            "padding_mask": ~mask.bool(),
            "position_ids": torch.arange(length).expand(len(batch), -1),
        }


@dataclass(kw_only=True)
class DocumentProvider(DatasetProvider):
    """Build the finite dataset without embedding tokens in checkpoint configs."""

    # Keep tokens out of the serialized training config/checkpoints.
    document_path: str = ""
    pad_id: int = 0
    seq_length: int = 32768
    dataloader_type: str = "single"
    num_workers: int = 0

    def build_datasets(self, context):
        with open(self.document_path) as stream:
            documents = json.load(stream)["documents"]
        return DocumentDataset(documents, self.pad_id, self.seq_length), None, None

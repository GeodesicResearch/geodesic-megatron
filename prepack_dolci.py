#!/usr/bin/env python3
"""Pre-pack Dolci SFT dataset into parquet format for fast training.

This avoids on-the-fly tokenization during distributed training.
Run on a single node before launching the multi-node training job.

Usage:
    source activate_env.sh
    python prepack_dolci.py
"""

from pathlib import Path

from transformers import AutoTokenizer

from megatron.bridge.data.datasets.packed_sequence import prepare_packed_sequence_data


DATASET_ROOT = Path("/projects/a5k/public/data/dolci_sft_megatron")
# Use instruct tokenizer (has chat template) — base model tokenizer doesn't have one
TOKENIZER_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
SEQ_LENGTH = 8192

DATASET_KWARGS = {
    "chat": True,
    "answer_only_loss": True,
    "use_hf_tokenizer_chat_template": True,
}


class HFTokenizerWrapper:
    """Thin wrapper to make HF AutoTokenizer look like MegatronTokenizer for packing."""

    def __init__(self, hf_tokenizer):
        self._tokenizer = hf_tokenizer
        self.legacy = False

    @property
    def vocab_size(self):
        return self._tokenizer.vocab_size

    @property
    def eod(self):
        return self._tokenizer.eos_token_id

    @property
    def eos_id(self):
        return self._tokenizer.eos_token_id

    def tokenize(self, text):
        return self._tokenizer.encode(text, add_special_tokens=False)

    @property
    def chat_template(self):
        return self._tokenizer.chat_template

    def apply_chat_template(self, *args, **kwargs):
        return self._tokenizer.apply_chat_template(*args, **kwargs)


def main():
    print(f"Loading tokenizer from {TOKENIZER_MODEL}...")
    hf_tok = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True)
    tokenizer = HFTokenizerWrapper(hf_tok)

    for split in ["training_100k", "validation"]:
        input_path = DATASET_ROOT / f"{split}.jsonl"
        output_path = DATASET_ROOT / f"{split}_packed"
        metadata_path = DATASET_ROOT / f"{split}_packed_metadata.json"

        if output_path.exists():
            print(f"Skipping {split} — already packed at {output_path}")
            continue

        if not input_path.exists():
            print(f"Skipping {split} — {input_path} not found")
            continue

        print(f"\nPacking {split} ({input_path})...")
        print(f"  Output: {output_path}")
        print(f"  Seq length: {SEQ_LENGTH}")

        prepare_packed_sequence_data(
            input_path=input_path,
            output_path=output_path,
            output_metadata_path=metadata_path,
            packed_sequence_size=SEQ_LENGTH,
            tokenizer=tokenizer,
            max_seq_length=SEQ_LENGTH,
            seed=5678,
            dataset_kwargs=DATASET_KWARGS,
        )
        print(f"  Done: {split}")

    print("\nAll splits packed.")


if __name__ == "__main__":
    main()

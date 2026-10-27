# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""What the packer stores for a tokenized document: its padded length, and its tokens and loss mask."""

import pytest

from megatron.bridge.data.datasets.packed_sequence import pad_document_for_packing


PAD_ID = 2


def _document(n: int, loss_mask_value) -> dict:
    return {
        "input_ids": list(range(10, 10 + n)),
        "context_ids": list(range(10, 10 + n)),
        "loss_mask": [loss_mask_value] * n,
    }


class TestPadDocumentForPacking:
    def test_pads_to_the_next_multiple_plus_the_token_the_label_shift_drops(self):
        document = _document(6, 1)
        pad_document_for_packing(document, max_seq_length=64, pad_seq_to_mult=4, pad_id=PAD_ID)
        assert document["input_ids"] == list(range(10, 16)) + [PAD_ID] * 3
        assert document["context_ids"] == list(range(10, 16)) + [PAD_ID] * 3
        assert document["loss_mask"] == [1] * 6 + [0] * 3

    def test_a_document_at_a_multiple_gains_only_the_token_the_label_shift_drops(self):
        document = _document(8, True)
        pad_document_for_packing(document, max_seq_length=64, pad_seq_to_mult=4, pad_id=PAD_ID)
        assert document["input_ids"] == list(range(10, 18)) + [PAD_ID]
        assert document["loss_mask"] == [True] * 8 + [False]

    def test_the_multiple_is_capped_at_the_maximum_length(self):
        document = _document(30, 1)
        pad_document_for_packing(document, max_seq_length=31, pad_seq_to_mult=8, pad_id=PAD_ID)
        # The next multiple of 8, 32, is above the maximum: the document is padded to 31 and the label shift's token.
        assert document["input_ids"] == list(range(10, 40)) + [PAD_ID] * 2

    def test_a_document_longer_than_the_maximum_is_cut_to_it(self):
        document = _document(40, 1)
        pad_document_for_packing(document, max_seq_length=32, pad_seq_to_mult=8, pad_id=PAD_ID)
        assert document["input_ids"] == list(range(10, 42))
        assert document["context_ids"] == list(range(10, 42))
        assert document["loss_mask"] == [1] * 32


# The rows prepare_packed_sequence_data stores, through the real Nemotron tokenizer: the think tokenizer
# (cached on the cluster) has the Super encoder and chat template plus {% generation %} markers. The tests skip
# when it is unavailable, so the unit tier stays network-independent.
THINK_TOKENIZER_ID = "geodesic-research/nemotron-think-tokenizer"
# The raw-text dataset kwargs that tokenize a document exactly as stored: no EOS, no separator, no stripping.
RAW_TEXT_KWARGS = {
    "chat": False,
    "answer_only_loss": False,
    "add_eos": False,
    "prompt_template": "{input}{output}",
    "strip_spaces": False,
}
CHAT_KWARGS = {"chat": True, "use_hf_tokenizer_chat_template": True, "answer_only_loss": True}


@pytest.fixture(scope="module")
def nemotron_tokenizers():
    try:
        from transformers import AutoTokenizer

        from megatron.bridge.training.tokenizers.config import TokenizerConfig
        from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer

        hf = AutoTokenizer.from_pretrained(THINK_TOKENIZER_ID)
        megatron = build_tokenizer(
            TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model=THINK_TOKENIZER_ID)
        )
    except Exception as e:  # pragma: no cover - environment-dependent
        pytest.skip(f"tokenizer unavailable: {e}")
    return hf, megatron


def _pack_one_per_row(tmp_path, records, megatron_tokenizer, max_seq_length, dataset_kwargs):
    """Pack JSONL records one document per row and return the stored rows."""
    import json

    import pyarrow.parquet as pq

    from megatron.bridge.data.datasets.packed_sequence import prepare_packed_sequence_data

    jsonl = tmp_path / "training.jsonl"
    jsonl.write_text("".join(json.dumps(r) + "\n" for r in records))
    out = tmp_path / "training.idx.parquet"
    prepare_packed_sequence_data(
        input_path=jsonl,
        output_path=out,
        output_metadata_path=None,
        packed_sequence_size=max_seq_length,
        tokenizer=megatron_tokenizer,
        max_seq_length=max_seq_length,
        seed=0,
        packing_algorithm="one_per_pack",
        dataset_kwargs=dict(dataset_kwargs),
        pad_seq_to_mult=1,
        num_tokenizer_workers=1,
    )
    table = pq.read_table(out).to_pydict()
    return [
        {"input_ids": i, "loss_mask": [int(m) for m in mask], "seq_start_id": s}
        for i, mask, s in zip(table["input_ids"], table["loss_mask"], table["seq_start_id"])
    ]


class TestRawTextRows:
    TEXTS = [
        "A plain document.\nWith two lines.",
        "  leading and trailing spaces are content ",
        "\nwrapped in newlines\n",
        "long " * 200,
    ]

    def test_rows_match_hf_tokenization_truncated_to_the_sequence_length(self, tmp_path, nemotron_tokenizers):
        hf, megatron = nemotron_tokenizers
        rows = _pack_one_per_row(
            tmp_path, [{"input": t, "output": ""} for t in self.TEXTS], megatron, 64, RAW_TEXT_KWARGS
        )
        expected = sorted(hf(t, truncation=True, max_length=64)["input_ids"] for t in self.TEXTS)
        assert sorted(row["input_ids"] for row in rows) == expected
        assert max(len(row["input_ids"]) for row in rows) == 64

    def test_every_target_is_trained(self, tmp_path, nemotron_tokenizers):
        """The stored mask is rolled onto the labels: every token but the first is a target."""
        _, megatron = nemotron_tokenizers
        rows = _pack_one_per_row(
            tmp_path, [{"input": t, "output": ""} for t in self.TEXTS], megatron, 64, RAW_TEXT_KWARGS
        )
        for row in rows:
            assert row["seq_start_id"] == [0]
            assert row["loss_mask"] == [1] * (len(row["input_ids"]) - 1) + [0]

    def test_pretraining_defaults_store_an_all_token_mask(self, tmp_path, nemotron_tokenizers):
        """The --no-chat defaults (EOS appended) store a mask over every target, the EOS included."""
        hf, megatron = nemotron_tokenizers
        rows = _pack_one_per_row(
            tmp_path,
            [{"input": self.TEXTS[0], "output": ""}],
            megatron,
            64,
            {"chat": False, "answer_only_loss": False},
        )
        (row,) = rows
        assert row["input_ids"][-1] == hf.convert_tokens_to_ids(hf.eos_token)
        assert row["loss_mask"] == [1] * (len(row["input_ids"]) - 1) + [0]


class TestChatRows:
    MESSAGES = [
        {"role": "user", "content": "What is 2 plus 3?"},
        {"role": "assistant", "content": "It is **5**.", "reasoning_content": "2 + 3 = 5."},
    ]

    def _assistant_targets(self, hf, messages, max_seq_length):
        """Token ids and per-label training flags the chat template's generation markers define."""
        rendered = hf.apply_chat_template(messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
        ids, assistant = rendered["input_ids"][:max_seq_length], rendered["assistant_masks"][:max_seq_length]
        return ids, assistant[1:] + [0]

    def test_mask_covers_exactly_the_assistant_turn(self, tmp_path, nemotron_tokenizers):
        hf, megatron = nemotron_tokenizers
        (row,) = _pack_one_per_row(tmp_path, [{"messages": self.MESSAGES}], megatron, 4096, CHAT_KWARGS)
        ids, flags = self._assistant_targets(hf, self.MESSAGES, 4096)
        assert row["input_ids"] == ids
        assert row["loss_mask"] == flags
        trained = hf.decode([ids[j + 1] for j in range(len(ids) - 1) if row["loss_mask"][j]])
        assert trained == "<think>\n2 + 3 = 5.\n</think>\nIt is **5**.<|im_end|>"

    def test_a_long_conversation_is_cut_to_the_sequence_length(self, tmp_path, nemotron_tokenizers):
        hf, megatron = nemotron_tokenizers
        messages = [self.MESSAGES[0], {**self.MESSAGES[1], "content": "word " * 100}]
        (row,) = _pack_one_per_row(tmp_path, [{"messages": messages}], megatron, 48, CHAT_KWARGS)
        ids, flags = self._assistant_targets(hf, messages, 48)
        assert len(row["input_ids"]) == 48
        assert row["input_ids"] == ids
        assert row["loss_mask"] == flags

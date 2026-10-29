# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""The training data a launch reads, by mode.

Catalogue:
    configure_sft_dataset: set up an SFT launch's chat processing and its pre-packed parquet bypass.
    bin_idx_dataset_config: the .bin/.idx data of a cpt or pretrain launch.
"""

import logging

from megatron.bridge.data.datasets.packed_parquet import resolve_packed_parquet_paths
from megatron.bridge.data.hf_processors.chat_messages import process_chat_messages_example
from megatron.bridge.training.config import ConfigContainer, GPTDatasetConfig


logger: logging.Logger = logging.getLogger(__name__)


def configure_sft_dataset(cfg: ConfigContainer) -> None:
    """Set up an SFT launch's dataset in place: the chat processor, and the bypass for pre-packed parquet.

    ``dataset.dataset_kwargs.chat`` selects ``process_chat_messages_example``. A
    ``packed_sequence_specs.packed_train_data_path`` that resolves to parquet files gets a one-row placeholder
    ``dataset_dict`` and ``rewrite=False``, so the builder neither downloads a dataset nor rewrites a JSONL and
    trains on the packed files. Reads the filesystem to resolve the path.
    """
    # If dataset_kwargs requests chat mode, use the generic chat messages processor.
    # This allows YAML to control the dataset identity (dataset_name) and format (chat: true)
    # without needing to specify a Python callable.
    if getattr(cfg.dataset, "dataset_kwargs", None) and cfg.dataset.dataset_kwargs.get("chat"):
        cfg.dataset.process_example_fn = process_chat_messages_example

    # Pre-packed blend support.
    #
    # When the YAML points `packed_sequence_specs.packed_train_data_path` at
    # already-packed parquet shards (e.g. a glob spanning several configs of a mix, or a list of globs),
    # there is nothing to download or pack — the data is ready on disk. But
    # HFDatasetBuilder.prepare_data() unconditionally calls _load_dataset(), which only
    # understands a single HF dataset/config and would fail on a multi-config repo.
    #
    # Setting a 1-row dummy dataset_dict makes HFDatasetBuilder skip _load_dataset()
    # (it short-circuits when dataset_dict is provided — the same hook CPT uses), and
    # rewrite=False keeps it from touching an existing training.jsonl. Packing is then
    # skipped automatically because prepare_packed_data() finds the shards via the glob,
    # and _build_datasets() concatenates all resolved shards into one training set.
    pss = getattr(cfg.dataset, "packed_sequence_specs", None)
    ptdp = getattr(pss, "packed_train_data_path", None) if pss else None
    if ptdp:
        try:
            resolved_shards = resolve_packed_parquet_paths(ptdp)
        except Exception:
            resolved_shards = []

        if resolved_shards:
            from datasets import Dataset, DatasetDict

            logger.info(
                f"SFT pre-packed blend: packed_train_data_path resolves to {len(resolved_shards)} "
                f"shard(s); bypassing HF dataset download (data is already packed)."
            )
            cfg.dataset.dataset_dict = DatasetDict(
                {
                    "train": Dataset.from_dict(
                        {
                            "messages": [[{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]],
                            "tools": [""],
                        }
                    )
                }
            )
            cfg.dataset.rewrite = False


def bin_idx_dataset_config(yaml_dataset: dict, mode: str) -> GPTDatasetConfig:
    """The .bin/.idx training data a ``cpt`` or ``pretrain`` run reads, from its merged ``dataset`` section.

    Args:
        yaml_dataset: The merged config's ``dataset`` section.
        mode: The training mode, named in the error when the section names no data.

    Returns:
        An unfinalized GPTDatasetConfig; setup attaches the tokenizer and finalizes it.
    """
    data_path = yaml_dataset.get("data_path")
    if not data_path:
        # Every .bin/.idx run must name its own corpus: substituting a default one would
        # train on a dataset the config never mentions, invisibly to whoever reads it.
        raise ValueError(
            f"{mode} mode requires dataset.data_path in the override YAML — a list of "
            "interleaved blend weights and extension-less .bin/.idx prefixes produced by "
            "tools/preprocess_data.py (see pipeline_data_submit.sbatch 'tokenize' mode)."
        )

    # Native .bin/.idx data pipeline — fast mmap loading, no packing needed.
    # data_path is a list of interleaved weights and path prefixes, e.g.:
    #   ["0.5", "/path/to/ds1_input_document", "0.5", "/path/to/ds2_input_document"]
    seq_length = yaml_dataset.get("seq_length", 8192)
    seed = yaml_dataset.get("seed", 1234)
    split = yaml_dataset.get("split", "9999,1,0")
    # Optional override for the GPTDataset index-cache directory. Unset, mcore
    # defaults to <prefix>/cache/GPTDataset_indices NEXT TO THE DATA — which is
    # owner-writable only when the corpus belongs to another account, and the cache
    # write in gpt_dataset.py is unguarded, so a cache MISS then crashes rank 0 with
    # PermissionError at dataset build. Setting this also lets the top-level
    # BlendedDataset index cache (it has no next-to-data fallback of its own and is
    # otherwise rebuilt at every launch).
    path_to_cache = yaml_dataset.get("path_to_cache")

    return GPTDatasetConfig(
        seq_length=seq_length,
        data_path=[str(p) for p in data_path],
        split=split,
        random_seed=seed,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        mmap_bin_files=True,
        dataloader_type="cyclic",
        path_to_cache=path_to_cache,
    )

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

from dataclasses import fields
from typing import Any, Callable, Dict, Optional, Type, Union

import numpy as np
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.gpt_dataset import GPTDataset, MockGPTDataset
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection

from megatron.bridge.data.builders.finetuning_dataset import FinetuningDatasetBuilder
from megatron.bridge.data.builders.hf_dataset import HFDatasetBuilder, HFDatasetConfig
from megatron.bridge.data.datasets.fim_dataset import GPTFIMDataset
from megatron.bridge.training.config import (
    DataloaderConfig,
    DatasetBuildContext,
    DatasetProvider,
    FinetuningDatasetConfig,
    GPTDatasetConfig,
    GPTFIMDatasetConfig,
    MockGPTDatasetConfig,
)
from megatron.bridge.training.gradient_routing.config import GRDatasetConfig, GRFinetuningDatasetConfig
from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer
from megatron.bridge.utils.common_utils import get_rank_safe, print_rank_0


def is_dataset_built_on_rank(pg_collection: ProcessGroupCollection) -> bool:
    """Determines whether the dataset should be built on the current rank.

    Datasets are typically built only on the first and last pipeline stages
    and the first tensor parallel rank to save memory and avoid redundancy.

    Returns:
        True if the dataset should be built on the current rank, False otherwise.
    """
    return (is_pp_first_stage(pg_collection.pp) or is_pp_last_stage(pg_collection.pp)) and (
        pg_collection.tp.rank() == 0
    )


def pretrain_train_valid_test_datasets_provider(
    train_val_test_num_samples: list[int], dataset_config: BlendedMegatronDatasetConfig
) -> tuple[GPTDataset, GPTDataset, GPTDataset]:
    """Build pretraining train, validation, and test datasets.

    Uses BlendedMegatronDatasetBuilder to create GPTDataset or MockGPTDataset instances.

    Args:
        train_val_test_num_samples: A list containing the number of samples for
                                    train, validation, and test datasets.
        dataset_config: Configuration object for the blended Megatron dataset.

    Returns:
        A tuple containing the train, validation, and test datasets.
    """

    if isinstance(dataset_config, GRDatasetConfig):
        return _build_gr_routed_datasets(train_val_test_num_samples, dataset_config)

    if dataset_config.mock:
        dataset_type = MockGPTDataset
    elif hasattr(dataset_config, "fim_data"):
        dataset_type = GPTFIMDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    # Build the dataset on all ranks for TP-replicated loading
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, lambda: True, dataset_config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def _check_gr_train_sizing(train_num_samples: int, plan, gbs: int) -> None:
    """Refuse a GR dataset build whose requested sample count disagrees with the plan."""
    expected_train = plan.train_iters * gbs
    if train_num_samples != expected_train:
        raise ValueError(
            f"GR dataset sizing mismatch: training requests {train_num_samples} samples "
            f"but the plan serves {expected_train} ({plan.train_iters} iters x GBS {gbs}). "
            "train_iters/global_batch_size changed after the plan was built."
        )


def _build_gr_routed_datasets(train_val_test_num_samples: list[int], dataset_config) -> tuple[Any, None, None]:
    """Build the gradient-routing train dataset: one GPTDataset per corpus behind a router.

    Each corpus is built through the standard BlendedMegatronDatasetBuilder from a child
    config carrying only that corpus's blend, sized to exactly what the routing plan
    consumes (the builder handles epoch looping). Validation/test are None — GR runs
    train with eval_iters 0 (enforced by the launch guards).
    """
    from megatron.bridge.data.datasets.gr_routed_dataset import GRRoutedDataset
    from megatron.bridge.training.gradient_routing.plan import CORE, FIRST_AUX

    plan = dataset_config.gr_plan
    gbs = dataset_config.gr_global_batch_size
    _check_gr_train_sizing(train_val_test_num_samples[0], plan, gbs)

    corpora = [(CORE, dataset_config.retain_data_path)] + [
        (k + FIRST_AUX, paths) for k, paths in enumerate(dataset_config.aux_data_paths)
    ]
    print_rank_0(f"> building gradient-routing train datasets (core + {plan.n_aux} aux) ...")
    children = {}
    for corpus, paths in corpora:
        child_config = dataset_config.build_child_config(paths)
        sizes = [plan.n_samples(corpus, gbs), 0, 0]
        child_train, _, _ = BlendedMegatronDatasetBuilder(GPTDataset, sizes, lambda: True, child_config).build()
        children[corpus] = child_train

    routed = GRRoutedDataset(children=children, plan=plan, global_batch_size=gbs)
    print_rank_0(f"> finished creating gradient-routing datasets ({plan.describe()})")
    return routed, None, None


def hf_train_valid_test_datasets_provider(
    train_val_test_num_samples: list[int], dataset_config: HFDatasetConfig, tokenizer: MegatronTokenizer
) -> tuple[Any, Any, Any]:
    """Build train, validation, and test datasets from a Hugging Face dataset.

    Uses HFDatasetBuilder to create dataset instances.

    Args:
        train_val_test_num_samples: A list containing the number of samples for
                                    train, validation, and test datasets.
        dataset_config: Configuration object for the Hugging Face dataset.
        tokenizer: The MegatronTokenizer instance.

    Returns:
        A tuple containing the train, validation, and test datasets.
    """
    print_rank_0(
        f"> building train, validation, and test datasets for Huggingface dataset {dataset_config.dataset_name} ..."
    )

    # Get field names from DataloaderConfig to exclude
    dataloader_field_names = {field.name for field in fields(DataloaderConfig)}

    train_ds, valid_ds, test_ds = HFDatasetBuilder(
        tokenizer=tokenizer,
        **{
            field.name: getattr(dataset_config, field.name)
            for field in fields(dataset_config)
            if field.name not in dataloader_field_names
        },
    ).build()

    print_rank_0(f"> finished creating Huggingface dataset {dataset_config.dataset_name} ...")

    return train_ds, valid_ds, test_ds


def finetuning_train_valid_test_datasets_provider(
    train_val_test_num_samples: list[int], dataset_config: FinetuningDatasetConfig, tokenizer: MegatronTokenizer
) -> tuple[Any, Any, Any]:
    """Build finetuning train, validation, and test datasets.

    Uses FinetuningDatasetBuilder to create dataset instances.

    Args:
        train_val_test_num_samples: A list containing the number of samples for
                                    train, validation, and test datasets.
        dataset_config: Configuration object for the finetuning dataset.
        tokenizer: The MegatronTokenizer instance.

    Returns:
        A tuple containing the train, validation, and test datasets.
    """
    if isinstance(dataset_config, GRFinetuningDatasetConfig):
        return _build_gr_finetuning_datasets(train_val_test_num_samples, dataset_config, tokenizer)

    print_rank_0(
        f">building train, validation, and test datasets for Finetuning dataset from {dataset_config.dataset_root} ..."
    )

    # Get field names from DataloaderConfig to exclude
    dataloader_field_names = {field.name for field in fields(DataloaderConfig)}

    train_ds, valid_ds, test_ds = FinetuningDatasetBuilder(
        tokenizer=tokenizer,
        **{
            field.name: getattr(dataset_config, field.name)
            for field in fields(dataset_config)
            if field.name not in dataloader_field_names
        },
    ).build()

    print_rank_0(f"> finished creating Finetuning dataset from {dataset_config.dataset_root} ...")

    return train_ds, valid_ds, test_ds


def _build_gr_finetuning_datasets(
    train_val_test_num_samples: list[int], dataset_config: GRFinetuningDatasetConfig, tokenizer: MegatronTokenizer
) -> tuple[Any, None, None]:
    """Build the gradient-routing SFT train dataset: one finetuning dataset per corpus behind a router.

    Each corpus is built through the standard FinetuningDatasetBuilder from a child config
    carrying only that corpus's dataset root and packed specs, capped via ``max_train_samples``
    to exactly what the routing plan consumes (the SFT sample mapping epoch-wraps an undersized
    corpus and truncates an oversized one, both deterministically). The exact-length check
    below is the batch-sampler twin of ``MegatronPretrainingSampler``'s hard
    ``consumed_samples < total_samples`` assert: ``MegatronPretrainingBatchSampler`` wraps
    silently modulo ``total_samples`` (and ``cyclic_iter`` restarts the loader), after which
    every routing label would be wrong with nothing in the logs — so a child whose realised
    length disagrees with the plan is refused at build. Validation/test are None — GR runs
    train with eval_iters 0 (enforced by the launch guards).
    """
    from megatron.bridge.data.datasets.gr_routed_dataset import GRRoutedDataset
    from megatron.bridge.training.gradient_routing.plan import CORE, FIRST_AUX

    plan = dataset_config.gr_plan
    gbs = dataset_config.gr_global_batch_size
    _check_gr_train_sizing(train_val_test_num_samples[0], plan, gbs)

    corpora = [(CORE, dataset_config.retain_dataset_root, dataset_config.retain_packed_sequence_specs)] + [
        (k + FIRST_AUX, root, specs)
        for k, (root, specs) in enumerate(
            zip(dataset_config.aux_dataset_roots, dataset_config.aux_packed_sequence_specs)
        )
    ]
    dataloader_field_names = {field.name for field in fields(DataloaderConfig)}
    print_rank_0(f"> building gradient-routing finetuning train datasets (core + {plan.n_aux} aux) ...")
    children = {}
    for corpus, root, specs in corpora:
        needed = plan.n_samples(corpus, gbs)
        child_config = dataset_config.build_child_config(root, specs, max_train_samples=needed)
        child_train, _, _ = FinetuningDatasetBuilder(
            tokenizer=tokenizer,
            **{
                field.name: getattr(child_config, field.name)
                for field in fields(child_config)
                if field.name not in dataloader_field_names
            },
        ).build()
        if child_train is None:
            raise ValueError(
                f"GR corpus {corpus} has no training data under {root} — the finetuning builder found "
                "neither training.jsonl nor the configured packed data there."
            )
        if len(child_train) != needed:
            raise ValueError(
                f"GR corpus {corpus} dataset serves {len(child_train)} samples but the plan consumes exactly "
                f"{needed} ({plan.n_corpus_iters(corpus)} iterations x GBS {gbs}). max_train_samples was not "
                "honoured, so the routed length would disagree with the plan — and the batch sampler wraps "
                "silently modulo its total instead of asserting, mislabeling every post-wrap iteration."
            )
        children[corpus] = child_train

    routed = GRRoutedDataset(children=children, plan=plan, global_batch_size=gbs)
    _log_gr_supervised_token_counts(children, plan)
    print_rank_0(f"> finished creating gradient-routing finetuning datasets ({plan.describe()})")
    return routed, None, None


def _log_gr_supervised_token_counts(children: dict[int, Any], plan) -> None:
    """Measure each corpus's supervised-token total against its iteration share.

    ``gr.aux_iter_fractions`` allocates ITERATIONS, but under answer_only_loss (and packing)
    corpora with different answer densities contribute different supervised-token counts per
    iteration, so iteration share and token share diverge. Iteration-share semantics are the
    configured contract; this logs the realised token shares so the divergence is measured
    rather than assumed. One extra pass over each corpus, on rank 0 only (packed corpora read
    pre-tokenized rows; unpacked JSONL corpora re-tokenize, the same cost class as packing prep).
    """
    from megatron.bridge.training.gradient_routing.plan import CORE, FIRST_AUX

    if get_rank_safe() != 0:
        return
    totals = {
        corpus: float(sum(np.sum(child._build_loss_mask(child[i])) for i in range(len(child))))
        for corpus, child in children.items()
    }
    grand_total = sum(totals.values())
    if grand_total == 0:
        raise ValueError(
            "GR corpora carry zero supervised tokens in total — every corpus's loss mask is empty, "
            "so no iteration would train anything. Check answer_only_loss/dataset formatting."
        )
    for corpus in sorted(totals):
        label = "core" if corpus == CORE else f"aux{corpus - FIRST_AUX}"
        iter_share = plan.n_corpus_iters(corpus) / plan.train_iters
        print_rank_0(
            f"> gr corpus {label}: supervised_tokens={int(totals[corpus])} "
            f"token_share={totals[corpus] / grand_total:.4f} iter_share={iter_share:.4f}"
        )


_REGISTRY: Dict[Type[Union[FinetuningDatasetConfig, BlendedMegatronDatasetConfig, HFDatasetConfig]], Callable] = {
    GPTDatasetConfig: pretrain_train_valid_test_datasets_provider,
    GPTFIMDatasetConfig: pretrain_train_valid_test_datasets_provider,
    MockGPTDatasetConfig: pretrain_train_valid_test_datasets_provider,
    GRDatasetConfig: pretrain_train_valid_test_datasets_provider,
    HFDatasetConfig: hf_train_valid_test_datasets_provider,
    FinetuningDatasetConfig: finetuning_train_valid_test_datasets_provider,
    GRFinetuningDatasetConfig: finetuning_train_valid_test_datasets_provider,
}


def get_dataset_provider(
    dataset_config: Union[FinetuningDatasetConfig, BlendedMegatronDatasetConfig, HFDatasetConfig, DatasetProvider],
) -> Callable:
    """Get the appropriate dataset provider function based on the config type.

    Supports both registry-based providers and protocol-based providers.

    Args:
        dataset_config: The dataset configuration object.

    Returns:
        The callable dataset provider function corresponding to the config type.
    """
    # Check if config implements the DatasetProvider protocol
    if isinstance(dataset_config, DatasetProvider):

        def protocol_adapter(
            train_val_test_num_samples: list[int],
            config: DatasetProvider,
            tokenizer: Optional[MegatronTokenizer] = None,
            pg_collection: Optional[ProcessGroupCollection] = None,
        ) -> tuple[Optional[Any], Optional[Any], Optional[Any]]:
            """Adapter function that bridges the protocol interface with the legacy interface."""
            context = DatasetBuildContext(
                train_samples=train_val_test_num_samples[0],
                valid_samples=train_val_test_num_samples[1],
                test_samples=train_val_test_num_samples[2],
                tokenizer=tokenizer,
                pg_collection=pg_collection,
            )
            return config.build_datasets(context)

        return protocol_adapter

    # Fall back to existing registry
    return _REGISTRY[type(dataset_config)]

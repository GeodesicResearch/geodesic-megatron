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
"""The source dataset of a preparation run, from the Hub or from local files.

Catalogue:
    MAX_LOAD_ATTEMPTS: how many times a rate-limited Hub load is tried.
    hub_source_kwargs: the arguments that say which of a Hub dataset's data is read (data_dir, revision).
    build_hub_load_kwargs: the ``load_dataset`` keyword arguments of a Hub load, streamed or not.
    parquet_file_schemas: the arrow schema of each parquet file, read from its footer.
    features_to_load: the features a Hub config is loaded with, and how they were settled.
    load_local_files: a Dataset from local JSON lines files.
    load_source_dataset: the run's dataset and how its features were settled, retrying a rate-limited Hub load.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import fsspec
import pyarrow.parquet as pq
from datasets import Features, load_dataset, load_dataset_builder
from scripts.data.hub_read_retry import reads_repeated_in_place


MAX_LOAD_ATTEMPTS = 10


def hub_source_kwargs(args):
    """The ``load_dataset`` / ``load_dataset_builder`` arguments that say which of the Hub dataset's data is read.

    ``data_dir`` and ``revision`` are only passed when set: ``load_dataset``
    treats an explicit ``None`` differently from an absent argument for some
    builders. Omitting ``revision`` resolves the default branch's current HEAD,
    so a dataset that is re-pushed yields different data under the same command.
    """
    kwargs = {}
    if args.data_dir:
        kwargs["data_dir"] = args.data_dir
    if args.revision:
        kwargs["revision"] = args.revision
    return kwargs


def build_hub_load_kwargs(args, features):
    """Keyword arguments for the ``load_dataset`` call that pulls from the Hub, loading with ``features`` unless they
    are ``None`` (the loader then settles them itself; ``features_to_load`` says which).

    With ``--streaming`` the call returns an ``IterableDataset`` that reads the data files as it
    is iterated, writing neither the files into the hub cache nor an Arrow cache; ``num_proc``
    is left out because ``load_dataset`` refuses it for a stream.
    """
    if args.streaming:
        kwargs = {"split": args.split, "streaming": True}
    else:
        kwargs = {"split": args.split, "num_proc": args.download_workers}
    kwargs.update(hub_source_kwargs(args))
    if features is not None:
        kwargs["features"] = features
    return kwargs


def parquet_file_schemas(files, workers, attempts, first_wait_seconds):
    """The arrow schema of each parquet file in ``files`` (local paths or fsspec URLs, such as the ``hf://`` URLs a
    Hub dataset's data files resolve to), read from its footer alone, ``workers`` files at a time. Each read of a
    file is tried up to ``attempts`` times on a transient Hub error (``scripts/data/hub_read_retry.py``)."""

    def read(file):
        with fsspec.open(file, "rb") as handle:
            what = f"a read of {file}'s footer"
            return pq.read_schema(reads_repeated_in_place(handle, attempts, first_wait_seconds, what))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(read, files))


def features_to_load(args):
    """The features to load the Hub config with (``None``: the loader settles them), and how they were settled.

    A config whose dataset card declares its features is loaded with those, which the loader reads from the card
    itself. A parquet config whose card declares none would otherwise be loaded with the features the loader infers
    from its first data file's ``huggingface`` schema metadata, a record of the ``datasets`` release that wrote the
    file: one newer than this environment's can write feature types this one cannot read (a ``List``), which fails the
    load. Such a config is instead loaded with the arrow schema its data files carry, without that metadata, which
    every one of them must share, so every file is read to the same features. A config of any other format is left
    to its loader.
    """
    builder = load_dataset_builder(args.dataset, args.subset, **hub_source_kwargs(args))
    if builder.info.features is not None:
        return None, "declared by the dataset card"
    if builder.name != "parquet":
        return None, f"inferred by the {builder.name} loader (the dataset card declares none)"
    files = [file for split_files in builder.config.data_files.values() for file in split_files]
    schemas = parquet_file_schemas(
        files, args.download_workers, args.hub_read_attempts, args.hub_read_first_wait_seconds
    )
    differing = [index for index, schema in enumerate(schemas) if not schema.equals(schemas[0], check_metadata=False)]
    if differing:
        raise ValueError(
            f"{args.dataset} config {args.subset!r}: the dataset card declares no features and its data files do not "
            f"share one arrow schema, so no features describe them all. {files[0]} has\n{schemas[0]}\nwhile "
            f"{files[differing[0]]} has\n{schemas[differing[0]]}\nand these {len(differing)} files differ from the "
            f"first: {[files[index] for index in differing]}"
        )
    features = Features.from_arrow_schema(schemas[0].remove_metadata())
    how = (
        f"explicit: the arrow schema all {len(files)} data files share "
        f"(the dataset card declares no features at {args.revision or 'HEAD'})"
    )
    return features, how


def load_local_files(data_files, num_proc):
    """A Dataset from ``data_files``, read by the ``json`` builder."""
    return load_dataset(
        "json",
        data_files=data_files,
        split="train",
        num_proc=num_proc,
    )


def load_source_dataset(args):
    """The dataset ``args`` names, and how its features were settled: ``--data-files`` (``load_local_files``) or the
    Hub (``features_to_load``, then ``build_hub_load_kwargs``), streamed under ``--streaming``.

    A load whose error mentions HTTP 429 is retried after a wait doubling from 300 s, capped at 600 s, up to
    ``MAX_LOAD_ATTEMPTS`` attempts. Each Hub attempt prints the features it loads with. Reads the network or local
    files.

    Raises:
        Exception: the last load's error, when it is not a rate limit or the attempts are exhausted.
    """
    for attempt in range(1, MAX_LOAD_ATTEMPTS + 1):
        try:
            if args.data_files:
                return load_local_files(args.data_files, args.download_workers), "inferred by the json loader"
            features, how = features_to_load(args)
            print(f"  Features: {how}")
            ds = load_dataset(
                args.dataset,
                args.subset,
                **build_hub_load_kwargs(args, features),
            )
            return ds, how
        except Exception as e:
            if "429" in str(e) and attempt < MAX_LOAD_ATTEMPTS:
                wait = min(300 * (2 ** (attempt - 1)), 600)
                print(f"  Rate limited (attempt {attempt}/{MAX_LOAD_ATTEMPTS}), waiting {wait}s...")
                time.sleep(wait)
            else:
                raise

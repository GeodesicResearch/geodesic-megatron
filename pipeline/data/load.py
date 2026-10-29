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
    build_hub_load_kwargs: the ``load_dataset`` keyword arguments of a Hub load.
    load_local_files: a Dataset from local JSON lines files.
    load_source_dataset: the run's dataset, retrying a rate-limited Hub load.
"""

import json
import time

import pandas as pd
from datasets import Dataset, load_dataset


MAX_LOAD_ATTEMPTS = 10


def build_hub_load_kwargs(args):
    """Keyword arguments for the ``load_dataset`` call that pulls from the Hub.

    ``data_dir`` and ``revision`` are only passed when set: ``load_dataset``
    treats an explicit ``None`` differently from an absent argument for some
    builders. Omitting ``revision`` resolves the default branch's current HEAD,
    so a dataset that is re-pushed yields different data under the same command.
    """
    kwargs = {"split": args.split, "num_proc": args.download_workers}
    if args.data_dir:
        kwargs["data_dir"] = args.data_dir
    if args.revision:
        kwargs["revision"] = args.revision
    return kwargs


def load_local_files(data_files, num_proc):
    """A Dataset from ``data_files``: the ``json`` builder, else pandas, else one ``json.loads`` per line."""
    try:
        return load_dataset(
            "json",
            data_files=data_files,
            split="train",
            num_proc=num_proc,
        )
    except Exception as e1:
        print(f"  HF loader failed ({e1}), falling back to pandas...")
        try:
            df = pd.read_json(data_files, lines=True)
            return Dataset.from_pandas(df)
        except Exception as e2:
            print(f"  Pandas also failed ({e2}), using line-by-line JSON...")
            rows = []
            with open(data_files) as f:
                for line in f:
                    rows.append(json.loads(line))
            return Dataset.from_list(rows)


def load_source_dataset(args):
    """The dataset ``args`` names: ``--data-files`` (``load_local_files``) or the Hub (``build_hub_load_kwargs``).

    A load whose error mentions HTTP 429 is retried after a wait doubling from 300 s, capped at 600 s, up to
    ``MAX_LOAD_ATTEMPTS`` attempts. Reads the network or local files.

    Raises:
        Exception: the last load's error, when it is not a rate limit or the attempts are exhausted.
    """
    for attempt in range(1, MAX_LOAD_ATTEMPTS + 1):
        try:
            if args.data_files:
                return load_local_files(args.data_files, args.download_workers)
            return load_dataset(
                args.dataset,
                args.subset,
                **build_hub_load_kwargs(args),
            )
        except Exception as e:
            if "429" in str(e) and attempt < MAX_LOAD_ATTEMPTS:
                wait = min(300 * (2 ** (attempt - 1)), 600)
                print(f"  Rate limited (attempt {attempt}/{MAX_LOAD_ATTEMPTS}), waiting {wait}s...")
                time.sleep(wait)
            else:
                raise

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

"""Compose a training override YAML from a base config plus the fields that differ.

A training YAML may name another YAML under the top-level key ``base_config``. The file then
means "that config, with these fields changed": the base is loaded (itself composed, so chains
work), and this file is deep-merged over it. Mappings merge key by key; every other value — a
list, a scalar, an explicit ``null`` — replaces the base value outright. That is what
``OmegaConf.merge`` does for plain YAML, so composing a file and then merging it onto a recipe
applies exactly the fields that merging the base and then the overlay would.

A relative ``base_config`` is resolved against the directory of the file that names it, never
against the working directory, so a config means the same thing wherever it is launched from.

Files are read the way ``OmegaConf.load`` reads them: exponent floats without a decimal point
(``5e-4``) are floats, dates stay strings, and a duplicated key is an error. A plain PyYAML
``safe_load`` would turn ``5e-4`` into the string ``"5e-4"`` and hand the training merge a
learning rate it cannot use.

This module depends only on PyYAML and the standard library, so host-side tools (such as the
FLOPs estimator) can compose configs without the training container.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml


BASE_CONFIG_KEY = "base_config"

# YAML 1.2 float forms, as OmegaConf registers them: PyYAML's YAML 1.1 resolver requires a
# decimal point, so without this an exponent-only literal such as 1e-5 would load as a string.
_YAML12_FLOAT = re.compile(
    r"""^(?:
     [-+]?[0-9]+(?:_[0-9]+)*\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?[0-9]+(?:_[0-9]+)*(?:[eE][-+]?[0-9]+)
    |\.[0-9]+(?:_[0-9]+)*(?:[eE][-+][0-9]+)?
    |[-+]?[0-9]+(?:_[0-9]+)*(?::[0-5]?[0-9])+\.[0-9_]*
    |[-+]?\.(?:inf|Inf|INF)
    |\.(?:nan|NaN|NAN))$""",
    re.X,
)
_TIMESTAMP_TAG = "tag:yaml.org,2002:timestamp"


class _ConfigLoader(yaml.SafeLoader):
    """A ``SafeLoader`` that reads scalars as OmegaConf does and rejects duplicate keys."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen = set()
        for key_node, _ in node.value:
            if key_node.tag != yaml.resolver.BaseResolver.DEFAULT_SCALAR_TAG:
                continue
            if key_node.value in seen:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key_node.value}",
                    key_node.start_mark,
                )
            seen.add(key_node.value)
        return super().construct_mapping(node, deep=deep)


_ConfigLoader.add_implicit_resolver("tag:yaml.org,2002:float", _YAML12_FLOAT, list("-+0123456789."))
_ConfigLoader.yaml_implicit_resolvers = {
    first_char: [(tag, regexp) for tag, regexp in resolvers if tag != _TIMESTAMP_TAG]
    for first_char, resolvers in _ConfigLoader.yaml_implicit_resolvers.items()
}


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Return ``overlay`` merged over ``base`` the way a ``base_config`` overlay is.

    Mappings merge recursively; any other value in ``overlay`` (a list, a scalar, an explicit
    ``None``) replaces the base value. Neither argument is modified, and the result shares no
    mutable value with either, so changing it cannot reach back into a caller's config.
    """
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_yaml_mapping(text: str, source: str) -> dict[str, Any]:
    """Parse one YAML document whose top level must be a mapping, reading scalars as a config file's are.

    ``source`` names the document in error messages. No ``base_config`` is resolved: this reads a
    single document.
    """
    document = yaml.load(text, Loader=_ConfigLoader)
    if not isinstance(document, dict):
        raise ValueError(f"{source}: the top level of a config must be a mapping, got {type(document).__name__}")
    return document


def _load_mapping(path: Path) -> dict[str, Any]:
    return parse_yaml_mapping(path.read_text(), str(path))


def _compose(path: Path, chain: tuple[Path, ...]) -> dict[str, Any]:
    if path in chain:
        cycle = " -> ".join(str(p) for p in (*chain, path))
        raise ValueError(f"{BASE_CONFIG_KEY} cycle: {cycle}")
    document = _load_mapping(path)
    if BASE_CONFIG_KEY not in document:
        return document

    base_ref = document.pop(BASE_CONFIG_KEY)
    if not isinstance(base_ref, str) or not base_ref:
        raise ValueError(f"{path}: {BASE_CONFIG_KEY} must be a non-empty path string, got {base_ref!r}")
    # Joining an absolute path discards the left operand, so this covers both forms.
    base_path = (path.parent / base_ref).resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"{path}: {BASE_CONFIG_KEY} {base_ref!r} does not exist (resolved to {base_path})")
    return deep_merge(_compose(base_path, (*chain, path)), document)


def load_composed_yaml(path: str | os.PathLike) -> dict[str, Any]:
    """Load the config at ``path`` with its ``base_config`` chain applied.

    Returns a plain dict without the ``base_config`` key. Raises
    ``FileNotFoundError`` when ``path`` or any base in its chain is missing, and ``ValueError``
    when a ``base_config`` chain loops back on itself (the message names the chain), when a
    ``base_config`` value is not a path string, or when a file's top level is not a mapping.
    """
    return _compose(Path(path).resolve(), ())

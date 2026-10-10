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
"""The key check the repo's tools apply to each mapping they read from a hand-written YAML file: a gate spec, a probe
spec, a bucket or publishing manifest, a tokenizer config.

A tool that reads a mapping only by the keys it knows would ignore any other key, so a misspelt threshold or pin would
silently not be applied, and a missing key would surface as a bare KeyError far from the file that lacks it.
``require_keys`` refuses both, naming where the mapping sits, its unknown keys and its missing ones, and raises the
caller's own error type, so each tool reports the fault as it reports any other in its input.

Standard library only, and no syntax newer than Python 3.6: the telemetry gates and the bucket mirror import it under
a host Python without the repo's dependencies.
"""

from typing import AbstractSet, Any, Dict, Type


def require_keys(
    mapping: Any,
    where: str,
    required: AbstractSet[str],
    optional: AbstractSet[str] = frozenset(),
    *,
    error: Type[Exception] = ValueError,
) -> Dict[str, Any]:
    """``mapping``, which must be a mapping holding every key of ``required`` and no key outside ``required`` and
    ``optional``. Otherwise raises ``error`` with a message that opens with ``where`` and names the unknown and the
    missing keys."""
    if not isinstance(mapping, dict):
        raise error(f"{where}: expected a mapping, got {type(mapping).__name__}")
    # str(): YAML reads a key such as 1 or true as a number or a bool, which cannot be sorted among strings.
    unknown = sorted(str(key) for key in set(mapping) - set(required) - set(optional))
    missing = sorted(set(required) - set(mapping))
    if unknown or missing:
        expected = (
            f"the keys {sorted(required)} and optionally {sorted(optional)}"
            if optional
            else f"exactly the keys {sorted(required)}"
        )
        raise error(f"{where}: expected {expected}; unknown keys {unknown}, missing keys {missing}")
    return mapping

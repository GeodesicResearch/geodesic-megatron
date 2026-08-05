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

"""Every bridge call of ``get_batch_on_this_cp_rank`` must match mcore's live signature.

The mcore 0.19 pin added a REQUIRED ``is_hybrid_cp`` positional to
``get_batch_on_this_cp_rank``. Four bridge call sites still passed the pre-0.19
two-argument form, and each one raised ``TypeError: ... missing 1 required
positional argument: 'is_hybrid_cp'`` the first time its batch reached the slicer.
Only one of the four (``gpt_step``) is on a path this repo's benchmarks exercise,
so the other three survived a full campaign undetected.

That is the defect this module generalises. A per-call-site behavioural test cannot
reach most of them cheaply — ``slice_batch_for_context_parallel`` returns early at
CP=1, so its bug only fires at CP>=2, which needs a real multi-rank group, and the
VLM step functions need vision fixtures. But the defect is not really behavioural:
it is a call site disagreeing with a signature. So check exactly that, against the
REAL ``inspect.signature`` of the REAL mcore function, for every call site in ``src/``
at once — which also covers call sites added after this was written, and fails on the
next upstream signature change rather than at somebody's first CP run.
"""

import ast
import inspect
from pathlib import Path

import pytest
from megatron.core.utils import get_batch_on_this_cp_rank

import megatron.bridge


TARGET = "get_batch_on_this_cp_rank"
MCORE_MODULE = "megatron.core.utils"

# The four call sites known when this test was written. A scan test that finds nothing
# passes vacuously, so pin a floor: if a refactor legitimately removes call sites, this
# number must be lowered deliberately rather than the coverage silently evaporating.
MIN_EXPECTED_CALL_SITES = 4

SRC_ROOT = Path(megatron.bridge.__file__).resolve().parent


def _local_aliases(tree: ast.Module) -> set[str]:
    """Names under which this module imported mcore's ``get_batch_on_this_cp_rank``.

    Import-gated so a same-named helper defined elsewhere is not mistaken for the
    mcore function whose signature we are checking.
    """
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == MCORE_MODULE:
            for alias in node.names:
                if alias.name == TARGET:
                    aliases.add(alias.asname or alias.name)
    return aliases


def _calls_in(tree: ast.Module, aliases: set[str]) -> list[ast.Call]:
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if name in aliases:
            calls.append(node)
    return calls


def _collect_call_sites() -> list[tuple[Path, ast.Call]]:
    sites = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        aliases = _local_aliases(tree)
        if not aliases:
            continue
        sites.extend((path, call) for call in _calls_in(tree, aliases))
    return sites


CALL_SITES = _collect_call_sites()


def _site_id(site: tuple[Path, ast.Call]) -> str:
    path, call = site
    return f"{path.relative_to(SRC_ROOT)}:{call.lineno}"


class TestCpRankCallSiteConformance:
    def test_call_sites_were_found(self):
        """Guard the scan itself: zero matches must fail, not pass silently."""
        assert len(CALL_SITES) >= MIN_EXPECTED_CALL_SITES, (
            f"found only {len(CALL_SITES)} call site(s) of {TARGET} under {SRC_ROOT}; "
            f"expected at least {MIN_EXPECTED_CALL_SITES}. Either the scan broke or call "
            f"sites were removed — lower MIN_EXPECTED_CALL_SITES deliberately if the latter."
        )

    @pytest.mark.parametrize("site", CALL_SITES, ids=_site_id)
    def test_call_site_binds_to_live_mcore_signature(self, site):
        """The arguments written at the call site must satisfy mcore's real signature.

        ``Signature.bind`` performs exactly the check the interpreter performs at call
        time — missing required parameters and unknown keywords both raise — so binding
        placeholders reproduces the TypeError without needing the call's real runtime
        state (a live process group, a packed batch, vision tensors).
        """
        path, call = site

        if any(isinstance(arg, ast.Starred) for arg in call.args) or any(kw.arg is None for kw in call.keywords):
            pytest.skip(f"{_site_id(site)} unpacks *args/**kwargs; arguments are not statically known")

        placeholder = object()
        positionals = [placeholder] * len(call.args)
        keywords = {kw.arg: placeholder for kw in call.keywords}

        try:
            inspect.signature(get_batch_on_this_cp_rank).bind(*positionals, **keywords)
        except TypeError as exc:
            pytest.fail(
                f"{path}:{call.lineno} calls {TARGET} with arguments mcore rejects: {exc}\n"
                f"  passed: {len(positionals)} positional, keywords={sorted(keywords)}\n"
                f"  mcore signature: {inspect.signature(get_batch_on_this_cp_rank)}"
            )

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
"""A recording stand-in for megatron.bridge's AutoBridge, for tests of how a script builds and loads a model.

Building and loading a sharded Megatron model needs GPUs and a distributed world, so these tests replace
``megatron.bridge.AutoBridge`` with a recorder: every call and attribute assignment made on the bridge, the provider
it returns and the model chunks it loads is appended to a trace, with arguments as ``repr`` strings.

Catalogue:
    Trace: the recorder; ``Trace.obj(name)`` hands out a recorded object.
    install: put a Trace's AutoBridge in place of megatron.bridge's for one test.
"""

import pytest


class Trace:
    """Records calls and assignments on the objects it hands out, in order, in ``calls``."""

    def __init__(self):
        self.calls = []

    def record(self, name, *args, **kwargs):
        """Append ``[name, [repr(arg)...], {key: repr(value)}]``."""
        self.calls.append([name, [repr(a) for a in args], {k: repr(v) for k, v in kwargs.items()}])

    def obj(self, name):
        """A recorded object: its methods record their calls and return ``result``'s object, its ``config`` is
        recorded, assignments to it are recorded, and its repr is ``name``."""
        trace = self

        class _Recorded:
            def __repr__(self):
                return name

            def __setattr__(self, key, value):
                trace.calls.append([f"{name}.{key} =", repr(value)])
                object.__setattr__(self, key, value)

            def __getattr__(self, key):
                if key == "config":
                    value = trace.obj(f"{name}.config")
                    object.__setattr__(self, key, value)
                    return value

                def method(*args, **kwargs):
                    trace.record(f"{name}.{key}", *args, **kwargs)
                    return trace.result(name, key)

                return method

        return _Recorded()

    def result(self, name, method):
        """What a recorded method returns: the bridge, provider and model chunks of a load, else None."""
        if method == "from_hf_pretrained":
            return self.obj("bridge")
        if method == "to_megatron_provider":
            return self.obj("provider")
        if method == "load_megatron_model":
            return [self.obj("chunk0")]
        if method == "cuda":
            return self.obj(f"{name}.cuda()")
        return None


def install(monkeypatch: pytest.MonkeyPatch) -> Trace:
    """A Trace whose ``obj("AutoBridge")`` replaces ``megatron.bridge.AutoBridge`` until the test ends."""
    import megatron.bridge

    trace = Trace()
    monkeypatch.setattr(megatron.bridge, "AutoBridge", trace.obj("AutoBridge"))
    return trace

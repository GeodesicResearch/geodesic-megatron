"""RL-spillover Campaign B: stub `megatron.energon` and all its submodules (not installed in the Verda venv).

megatron.bridge.recipes (common.py, diffusion recipes) imports Energon data providers at module level; the text
cpt path never uses them. Installed as a meta-path finder from sitecustomize (on PYTHONPATH) so every process,
including spawned checkpoint writers, resolves `megatron.energon[.x.y]` to stub modules whose attributes are
inert placeholder classes.
"""
import importlib.abc
import importlib.machinery
import sys
import types

_ROOT = "megatron.energon"


class _Missing:
    """Inert placeholder: construction is a no-op (module-level defaults/decorators in the Energon data
    providers run at import); calling an instance returns its single positional argument (decorator use)."""

    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return a[0] if len(a) == 1 and not k else None

    def __class_getitem__(cls, item):
        return cls


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        obj = type(name, (_Missing,), {})
        setattr(self, name, obj)
        return obj


class _EnergonStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == _ROOT or fullname.startswith(_ROOT + "."):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        m = _StubModule(spec.name)
        m.__path__ = []
        return m

    def exec_module(self, module):
        return None


try:
    import importlib.util as _u

    _real = _u.find_spec("megatron.energon") is not None
except Exception:  # noqa: BLE001 - megatron namespace may not be importable at startup
    _real = False
if not _real:
    sys.meta_path.insert(0, _EnergonStubFinder())


# megatron.bridge.recipes/__init__.py star-imports every recipe family (diffusion recipes need diffusers,
# webdataset, ...). Register the package with its real __path__ WITHOUT running __init__, so
# `megatron.bridge.recipes.nemotronh.nemotron_3_nano` imports alone. A meta-path finder (rather than an in-process
# shim) so spawned workers that re-import pipeline_training_run.py as __mp_main__ get it too.
class _RecipesPkgFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    _NAME = "megatron.bridge.recipes"

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self._NAME:
            return None
        import os

        import megatron.bridge

        spec = importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        spec.submodule_search_locations = [os.path.join(os.path.dirname(megatron.bridge.__file__), "recipes")]
        return spec

    def create_module(self, spec):
        m = types.ModuleType(spec.name)
        m.__path__ = list(spec.submodule_search_locations)
        return m

    def exec_module(self, module):
        return None


sys.meta_path.insert(0, _RecipesPkgFinder())

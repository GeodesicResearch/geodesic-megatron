#!/usr/bin/env python3
"""Run pipeline_training_run.py with megatron.bridge.recipes registered WITHOUT executing its __init__.

megatron/bridge/recipes/__init__.py star-imports every recipe family, including the diffusion (Flux/WAN)
recipes, which need diffusers + megatron.energon. The Verda runtime venv has neither, and the LLM recipes
(nemotron_3_nano) do not need them. Registering the package object with its real __path__ lets
`megatron.bridge.recipes.nemotronh.nemotron_3_nano` import normally while skipping the eager star-imports.
Arguments are passed through to pipeline_training_run.py unchanged.
"""
import os
import runpy
import sys
import types

import megatron.bridge

_name = "megatron.bridge.recipes"
if _name not in sys.modules:
    pkg = types.ModuleType(_name)
    pkg.__path__ = [os.path.join(os.path.dirname(megatron.bridge.__file__), "recipes")]
    pkg.__package__ = _name
    sys.modules[_name] = pkg

# megatron.energon (imported by megatron.bridge.recipes.common, unused by the text cpt path) is satisfied by a
# stub on PYTHONPATH (Campaign B pylib), so spawned checkpoint-writer processes resolve it too.

class GridSaveInterval(int):
    """save_interval that fires only at listed steps.

    Megatron-Bridge saves when `step % save_interval == 0`. Python calls the right operand's __rmod__ first when its
    type is a subclass of the left operand's, so this int (its value is the smallest gap, for any other use)
    answers 0 exactly at the grid steps. Avoids ~18 s per unneeded adapter save on a log-spaced grid.
    """

    def __new__(cls, steps):
        steps = sorted(set(steps))
        obj = super().__new__(cls, min(b - a for a, b in zip([0] + steps, steps)))
        obj.steps = frozenset(steps)
        return obj

    def __rmod__(self, step):
        return 0 if step in self.steps else 1


def _install_grid_saves():
    steps = os.environ.get("SPILL_SAVE_STEPS")
    if not steps:
        return
    import megatron.bridge.training.finetune as ft

    grid = [int(s) for s in steps.replace(",", " ").split()]
    orig = ft.finetune

    def finetune(config, *a, **k):
        config.checkpoint.save_interval = GridSaveInterval(grid)
        print(f"[sdf_train] checkpoint saves only at steps {sorted(grid)}", flush=True)
        return orig(config, *a, **k)

    ft.finetune = finetune


class _NoShuffle:
    """A RandomState stand-in whose shuffle() keeps the order (only shuffle() is called on it)."""

    def shuffle(self, x):
        return None


def _install_no_shuffle():
    """SPILL_NO_SHUFFLE=1: train in corpus file order (the free-text redo's salience-parity rounds, DECISIONS #52).

    Megatron's GPTDataset shuffles the document order of every epoch and then the sample order. Both builders take
    the RandomState as an argument; passing a no-op shuffler keeps documents in file order and samples sequential.
    The pretraining sampler is already sequential, so every prefix of training sees the corpus prefix.
    """
    if os.environ.get("SPILL_NO_SHUFFLE") != "1":
        return
    import megatron.core.datasets.gpt_dataset as gd

    doc, shuf = gd._build_document_index, gd._build_shuffle_index
    gd._build_document_index = lambda documents, num_epochs, rs, separate_final_epoch: doc(
        documents, num_epochs, _NoShuffle(), separate_final_epoch)
    gd._build_shuffle_index = lambda num_samples, total_size, rs: shuf(num_samples, total_size, _NoShuffle())
    print("[sdf_train] GPTDataset shuffling disabled (training order = corpus file order)", flush=True)


if __name__ == "__main__":
    _install_grid_saves()
    _install_no_shuffle()
    # Guarded: spawned worker processes (the async checkpoint writer) re-import this file as __mp_main__; they
    # must get the package registration above but must not start training.
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, "..", "..", "pipeline_training_run.py")
    sys.argv = [script] + sys.argv[1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name="__main__")

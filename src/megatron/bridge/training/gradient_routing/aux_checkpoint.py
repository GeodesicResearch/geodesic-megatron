# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Aux-only checkpoints: the GR analogue of PEFT's adapter-only save.

A run whose plan never updates the core (``aux_iter_fractions`` summing to 1 AND
``p_as: 0``, so there is neither a core-corpus nor an aux-spread iteration) leaves every
core weight exactly as ``checkpoint.pretrained_checkpoint`` supplied it. Its full save
therefore writes tens of GiB of bytes that are already on disk unchanged.
``gr.checkpoint_aux_only`` narrows the save to the aux parameters, and it does so through
the SAME model-section key filter PEFT uses for adapter-only checkpoints
(:func:`megatron.bridge.training.checkpointing.filter_state_dict_model_sections`) — the
predicate is the only difference between the two.

What comes out is a PARTIAL checkpoint, and it is consumed the way a LoRA adapter is:
compose it with the core it was trained against. Composition itself is format-level and
carries no GR knowledge (``base_checkpoint_path`` on
:func:`megatron.bridge.training.model_load_save.load_megatron_model`); what lives here is
the aux-side verification, because "the overlay silently loaded nothing" would otherwise
export a stock base model labelled as a trained one — :func:`assert_aux_weights_trained`
is what makes that failure loud.

Mid-run RESUME from an aux-only checkpoint is refused at launch rather than supported. GR
runs must carry mismatch-tolerant ``dist_ckpt_strictness`` (the warm-start base has no
``gr_aux`` tensors), so a single-stage resume would fill the core with fresh init and keep
training with nothing in the logs to say so; :func:`checkpoint_saved_aux_only` is how the
launch guards recognise such a checkpoint from its own ``run_config.yaml``.
"""

import torch

from megatron.bridge.training.gradient_routing.optimizer_gating import GR_AUX_NAME_FRAGMENT
from megatron.bridge.training.utils.checkpoint_utils import (
    TRACKER_PREFIX,
    file_exists,
    get_checkpoint_name,
    get_checkpoint_run_config_filename,
    get_checkpoint_train_state_filename,
    is_checkpoint_iteration_directory,
    read_run_config,
    read_train_state,
)


def gr_aux_key_filter(key) -> bool:
    """Whether a model-section checkpoint key names a GR aux parameter.

    The predicate half of the aux-only save: passed to
    ``filter_state_dict_model_sections`` exactly where the PEFT path passes
    ``PEFT.adapter_key_filter``. Aux parameters are identified by the same
    ``.gr_aux.`` fragment the optimizer override glob and the HF bridge mappings key on,
    so one rename cannot leave the filter and the param groups disagreeing.

    Args:
        key: Parameter name, or a ``(name, param)`` pair (the spelling distributed
            checkpointing uses in some sections, which the PEFT filter also accepts).

    Returns:
        True if the parameter belongs to an aux module and must be saved.
    """
    name = key[0] if isinstance(key, tuple) else key
    return GR_AUX_NAME_FRAGMENT in name


def saves_aux_only(gr_cfg) -> bool:
    """Whether this run's checkpoints carry only the aux parameters.

    Args:
        gr_cfg: The ``gr:`` section (``GradientRoutingConfig``), or None on a non-GR run.
    """
    return gr_cfg is not None and gr_cfg.enabled and gr_cfg.checkpoint_aux_only


def checkpoint_saved_aux_only(checkpoint_path: str) -> bool:
    """Whether the checkpoint at ``checkpoint_path`` was written by an aux-only run.

    Read from the checkpoint's own ``run_config.yaml`` — the provenance the plan-digest
    resume check already reads — so a partial checkpoint is recognisable without opening a
    single shard. Accepts either a specific iteration directory or a parent directory whose
    tracker names the latest iteration.

    A checkpoint with no readable ``run_config.yaml`` (a MegatronLM-native checkpoint, an
    imported HF bridge) reports False: it predates the flag, so it cannot be aux-only.

    Args:
        checkpoint_path: Iteration directory, or a parent directory holding ``iter_*``.

    Returns:
        True iff the checkpoint's run config carries ``gr.checkpoint_aux_only: true``.
    """
    if is_checkpoint_iteration_directory(checkpoint_path):
        iteration_path = checkpoint_path
    else:
        tracker = get_checkpoint_train_state_filename(checkpoint_path, prefix=TRACKER_PREFIX)
        if not file_exists(tracker):
            return False
        iteration_path = get_checkpoint_name(checkpoint_path, read_train_state(tracker).step)
    run_config_path = get_checkpoint_run_config_filename(iteration_path)
    if not file_exists(run_config_path):
        return False
    return bool((read_run_config(run_config_path).get("gr") or {}).get("checkpoint_aux_only", False))


def assert_aux_weights_trained(model) -> None:
    """Raise unless the model carries at least one non-zero aux output projection.

    The composed load (core from a base checkpoint, aux from an aux-only one) writes
    whatever keys each stage supplies and leaves the rest alone, so an overlay that
    contributed nothing produces a byte-stock base model that every downstream consumer
    would read as the trained arm. ``linear_fc2.weight`` is exactly zero at init and
    non-zero only after training (the invariant ``GRCallback`` asserts at iteration 0),
    which makes "aux weights actually arrived" checkable without a reference tensor.

    Args:
        model: Megatron model chunk(s) — a module or a list of modules.

    Raises:
        ValueError: If the model has no aux modules, or every aux output projection is
            still exactly zero.
    """
    chunks = model if isinstance(model, list) else [model]
    aux_projections = [
        (name, param)
        for chunk in chunks
        for name, param in chunk.named_parameters()
        if GR_AUX_NAME_FRAGMENT in name and name.endswith("linear_fc2.weight")
    ]
    if not aux_projections:
        raise ValueError(
            "assert_aux_weights_trained found no gr_aux output projections in the model — it was "
            "built without aux modules, so an aux-only checkpoint has nothing to load into. Build "
            "from the aux-only checkpoint's own run_config (it records model.gr_aux_ffn_hidden_size), "
            "not from the base checkpoint's."
        )
    if not any(torch.any(param != 0) for _name, param in aux_projections):
        raise ValueError(
            f"every one of the {len(aux_projections)} gr_aux output projections is exactly zero, so "
            "no trained aux weights were loaded — this model is byte-stock base. An aux-only "
            "checkpoint must be composed with its base (pass the aux checkpoint as the load path and "
            "the base as base_checkpoint_path); loading only the base, or loading the aux checkpoint "
            "into a model built without aux modules, produces exactly this state."
        )


def aux_only_checkpoint_problems(gr_cfg, checkpoint_cfg, peft_cfg, plan) -> list[str]:
    """The preconditions an aux-only-saving run must meet; empty means sound.

    Value-taking (rather than reading a ConfigContainer) so the launch guards feed it by
    direct attribute access and a field renamed upstream raises there instead of turning
    this check into a no-op.

    Args:
        gr_cfg: The ``gr:`` section.
        checkpoint_cfg: ``cfg.checkpoint``.
        peft_cfg: ``cfg.peft`` (None on every GR run).
        plan: The run's ``GRPlan``, or None when the wiring is incomplete.
    """
    if not saves_aux_only(gr_cfg):
        return []
    problems: list[str] = []
    if plan is not None:
        core_updates = int(plan.update_core.sum())
        if core_updates:
            problems.append(
                f"gr.checkpoint_aux_only=true but the plan updates the core on {core_updates} of "
                f"{plan.train_iters} iterations ({plan.n_core_iters} core-corpus, "
                f"{core_updates - plan.n_core_iters} aux-spread at p_as={plan.p_as}). An aux-only "
                "checkpoint carries no core weights, so every core update this run makes would be "
                "dropped at save time and silently replaced by pretrained_checkpoint's originals on "
                "load. Set aux_iter_fractions summing to 1 AND p_as: 0.0, or leave "
                "checkpoint_aux_only false."
            )
    if checkpoint_cfg.pretrained_checkpoint is None:
        problems.append(
            "gr.checkpoint_aux_only=true but checkpoint.pretrained_checkpoint is unset; an aux-only "
            "checkpoint is only interpretable alongside the core it was trained against, and that "
            "path is the record of which core that is."
        )
    if checkpoint_cfg.save_optim:
        problems.append(
            "gr.checkpoint_aux_only=true requires checkpoint.save_optim: false. Optimizer state is "
            "the bulk of a full save and covers core parameters this checkpoint does not carry, so "
            "saving it would both defeat the size win and describe a resume the aux-only format "
            "cannot serve."
        )
    if peft_cfg is not None:
        problems.append(
            "gr.checkpoint_aux_only=true with peft configured: both narrow the saved model sections "
            "by their own predicate, and the run would keep only the intersection (adapter params "
            "are not aux params, so: nothing). Pick one."
        )
    return problems


def aux_only_source_problems(checkpoint_cfg) -> list[str]:
    """Refuse the load paths that would compose an aux-only checkpoint with a fresh core.

    GR runs carry mismatch-tolerant ``dist_ckpt_strictness`` by necessity, so pointing
    either load slot at an aux-only checkpoint does not fail — it loads the aux tensors,
    leaves the core at fresh init, and trains on. Both routes are refused here.

    Args:
        checkpoint_cfg: ``cfg.checkpoint``.
    """
    problems: list[str] = []
    for field_name, path in (
        ("checkpoint.load", checkpoint_cfg.load),
        ("checkpoint.pretrained_checkpoint", checkpoint_cfg.pretrained_checkpoint),
    ):
        if path is None or not checkpoint_saved_aux_only(path):
            continue
        problems.append(
            f"{field_name}={path} is an AUX-ONLY checkpoint (its run_config carries "
            "gr.checkpoint_aux_only: true), which holds no core weights. Loading it here would "
            "train on a freshly-initialised core, silently. Aux-only checkpoints are consumed by "
            "composing them with their base: `pipeline_checkpoint_convert_hf.py "
            f"--megatron-path {path} --base-megatron-path <that run's pretrained_checkpoint>` for an "
            "HF export. Resuming or warm-starting a Megatron run from one is not supported."
        )
    return problems

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
"""Training callback driving per-iteration gradient-routing state and telemetry.

Per iteration ``i`` (``state.train_state.step``, constant across the whole step):

- ``gr_gate`` on every GRAM MoE layer <- ``plan.fwd_aux[i]`` (per-module forward
  activation vector);
- ``frozen_expert_bias`` on every router <- ``not plan.update_core[i]`` — the router's
  expert-bias load-balancing update runs OUTSIDE the optimizer (in grad finalization), so
  it must be frozen explicitly on iterations that do not update core;
- after the step: restore the optimizer param groups the gater emptied, then emit
  telemetry.

Telemetry goes through ``log_wandb_metrics_nonfatal``: guarded by ``wandb.run``, silent on
failure — logging must never crash training.
"""

import logging

import numpy as np
import torch

from megatron.bridge.models.mamba.gram_layer import GR_AUX_OUTPUT_INIT_DEFAULT, GRAMMoELayer
from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerGater
from megatron.bridge.training.gradient_routing.plan import CORE, FIRST_AUX, GRPlan, build_gr_plan
from megatron.bridge.training.utils.checkpoint_utils import (
    file_exists,
    get_checkpoint_name,
    get_checkpoint_run_config_filename,
    read_run_config,
)
from megatron.bridge.training.utils.wandb_utils import log_wandb_metrics_nonfatal


logger = logging.getLogger(__name__)


#: Columns of the per-iteration probe accumulator that trail its ``n_aux`` per-module output
#: columns.
#:
#: The layer INPUT is the tensor the modules are actually handed, and it is NOT the residual
#: stream. On NemotronH the GRAM layer is the ``mlp`` submodule of an ``MoETransformerLayer``
#: whose ``pre_mlp_layernorm`` is a real ``TENorm`` (mcore
#: ``models/hybrid/hybrid_layer_specs.py``), so ``TransformerLayer._forward_mlp`` computes
#: ``pre_mlp_layernorm_output = pre_mlp_layernorm(hidden_states)``, keeps ``residual =
#: hidden_states``, calls the GRAM layer on the NORMALIZED tensor, and only then adds its
#: write into the residual via ``mlp_bda``. The probe never sees the residual. What the layer
#: input gives is still the right denominator for a magnitude series — it is
#: gate-INDEPENDENT, so a ratio against it means the same thing on every iteration, and it is
#: unbounded — but a normalization pins its RMS near ``||gamma||`` at every depth while the
#: residual's RMS grows with depth, so these ratios must NOT be read as "the fraction of the
#: residual stream the module perturbs" (that reading overstates it by
#: ``||residual|| / ||norm(residual)||``, by a DIFFERENT factor at every layer). Hence the key
#: spelling ``*_in_ratio_*``: the denominator is named in the key.
#:
#: The layer OUTPUT already CONTAINS the gated aux contribution (``GRAMMoELayer.forward``
#: returns ``core + sum_k gate_k * aux_k(h)``), so a ratio against it saturates near 1 and is
#: reported only as an explicitly-bounded "share of this layer's write" — never as the
#: module's magnitude.
_COL_LAYER_IN, _COL_LAYER_OUT = 0, 1
_PROBE_EXTRA_COLS = 2

#: Fraction of a module copy's ``linear_fc1`` units that must still produce a positive
#: pre-activation somewhere in the iteration before the copy is called damaged.
#:
#: Squared-ReLU death is per UNIT, not per tensor: row j of ``linear_fc1`` receives exactly
#: zero gradient iff unit j's pre-activation is non-positive on every token. A whole-tensor
#: max therefore calls a copy "live" while 4095 of its 4096 units are permanently frozen —
#: which is the shape of the failure this probe exists to catch (the arm that shipped with a
#: layer-0 output RMS of ~2e-4 had lost almost all of its capacity, not literally all of it).
#: The floor is a heuristic and only ever WARNS. It sits far below any healthy reading: a
#: randomly-initialised fc1 puts a unit positive within a handful of tokens, so a real
#: iteration's thousands of tokens leave essentially every unit live.
_LIVE_UNIT_FRACTION_FLOOR = 0.25

#: Shared tail of both collapse warnings — the mechanism and the two levers are the same
#: whether a copy died outright or lost most of its units.
_COLLAPSE_REMEDY = (
    "NemotronH's activation is squared ReLU, whose gradient is identically zero once a unit's "
    "pre-activations are all non-positive, so those units are frozen for the rest of the run and "
    "no resume recovers them. The run will still complete with 0 NaN and a valid checkpoint, and "
    "any composability result over this module is vacuous to the extent it contributes nothing. "
    "Lower gr.aux_lr (one oversized fc1 step is what drives units non-positive), or set "
    "model.gr_aux_output_init='standard' so fc1 carries a live gradient from the first routed "
    "step instead of waiting for fc2 to leave zero."
)


class GRCallback(Callback):
    """Drives gates, expert-bias freezing, gater restore, and W&B telemetry."""

    def __init__(self, plan: GRPlan, gater: GROptimizerGater, log_interval: int = 1):
        self._plan = plan
        self._gater = gater
        self._log_interval = log_interval
        self._gram_layers: list[GRAMMoELayer] = []
        self._layer_numbers: list[int] = []
        self._routers: list = []
        self._probe_handles: list = []
        self._probe_records: dict | None = None
        self._collapse_announced: set[int] = set()
        self._multi_gate_announced = False
        self._step_verification_unavailable_logged = False

    def on_train_start(self, context: CallbackContext) -> None:
        """Build module registries, sanity-check the warm start, log the plan."""
        self._gram_layers = []
        self._routers = []
        bias_carriers = 0
        for model_chunk in context.model:
            for module in model_chunk.modules():
                if isinstance(module, GRAMMoELayer):
                    self._gram_layers.append(module)
                    self._routers.append(module.router)
                if getattr(module, "expert_bias", None) is not None:
                    bias_carriers += 1
        if not self._gram_layers:
            raise RuntimeError(
                "GRCallback found no GRAMMoELayer in the model — the GRAM spec swap did not run. "
                "Check model.gr_aux_ffn_hidden_size wiring."
            )
        for layer in self._gram_layers:
            if len(layer.gr_aux) != self._plan.n_aux:
                raise RuntimeError(
                    f"A GRAM layer carries {len(layer.gr_aux)} aux modules but the plan routes "
                    f"{self._plan.n_aux} — model surgery and plan disagree about the module count."
                )
        # Depth labels for the probe's *_layer keys. mcore's own layer_number indexes the FULL
        # hybrid stack, so a reported id means the same thing here, in the model's logs, and in
        # hybrid_layer_pattern — unlike a position in THIS registry, which counts MoE layers
        # only (23 of Nano's 62) and would be read as a transformer depth it is not.
        self._layer_numbers = [
            index if getattr(layer, "layer_number", None) is None else int(layer.layer_number)
            for index, layer in enumerate(self._gram_layers)
        ]
        # The per-iteration freeze is applied to the routers collected above, while Megatron
        # updates expert_bias on EVERY module that carries one. A bias carrier outside the
        # swapped stack spec would keep updating on aux-isolated iterations — an invisible
        # leak of routed-corpus signal into router state. FEWER carriers than routers is
        # sound, not a leak: with moe_router_enable_expert_bias off, mcore sets every
        # router's expert_bias to None, there is no bias update to freeze, and the
        # per-iteration toggle is a no-op.
        if bias_carriers > len(self._routers):
            raise RuntimeError(
                f"{bias_carriers} module(s) carry expert_bias but only {len(self._routers)} are "
                "GRAM routers this callback can freeze. Every expert-bias carrier must be inside "
                "the GRAM spec swap, or its bias would keep learning from a routed corpus on "
                "iterations where the core is supposed to be frozen."
            )
        start = context.state.train_state.step
        # The zero-init invariant holds only at the START of a plan: a trained aux is
        # non-zero by construction, so asserting it on a mid-plan resume would make GR
        # runs restart-fatal (ft_launcher restarts, singleton chains, save_interval runs).
        # Under gr_aux_output_init="standard" there is no invariant to check — a randomly
        # initialised fc2 is indistinguishable from a clobbered one, which is exactly the
        # warm-start protection that mode trades away.
        output_init = getattr(self._gram_layers[0].config, "gr_aux_output_init", GR_AUX_OUTPUT_INIT_DEFAULT)
        if start == 0 and output_init == "zero":
            for layer in self._gram_layers:
                for k, aux in enumerate(layer.gr_aux):
                    if not torch.all(aux.linear_fc2.weight == 0):
                        raise RuntimeError(
                            f"gr_aux.{k}.linear_fc2.weight is non-zero at iteration 0. At the start "
                            "of a plan these must be exactly zero (fresh zero-init, untouched by any "
                            "checkpoint load); a non-zero value means the load clobbered them, or a "
                            "GR checkpoint was loaded as `pretrained_checkpoint` (a warm start) "
                            "rather than resumed via `load` (which carries the iteration and resumes "
                            "the plan)."
                        )
        elif start == 0:
            logger.info(
                "GR: gr_aux_output_init='standard' — the iteration-0 fc2-zero invariant does not "
                "apply (and warm-start clobber protection is unavailable in this mode)."
            )
        else:
            self._assert_plan_matches_checkpoint(context)
            logger.info("GR resume at iteration %d: plan re-derived deterministically.", start)
        logger.info("%s", self._plan.describe())
        logger.info(
            "GR: %d GRAM MoE layers, %d aux module(s)/layer, aux params %.1fM per model.",
            len(self._gram_layers),
            self._plan.n_aux,
            sum(p.numel() for layer in self._gram_layers for p in layer.gr_aux.parameters()) / 1e6,
        )
        log_wandb_metrics_nonfatal(
            {
                "run/gr_plan_digest_int": int(self._plan.digest()[:8], 16),
                # Provenance for what the gr/aux*_out_rms* family MEANS. A run whose summary
                # lacks this key was measured by the pre-2026-08 probe, which sampled ONE GRAM
                # layer and kept the last microbatch; its numbers are not comparable.
                "run/gr_probe_layers": len(self._gram_layers),
            },
            step=start,
        )

    def _assert_plan_matches_checkpoint(self, context: CallbackContext) -> None:
        """Refuse a resume whose plan differs from the one the checkpoint was trained under.

        The plan is a pure function of (plan_seed, train_iters, aux_iter_fractions, p_as,
        p_cr), so changing ANY of them between a save and a resume silently relabels every
        remaining iteration AND shifts each corpus's consumption offset — the run would keep
        training, on different data, against a different routing schedule, with nothing in
        the logs to say so. The checkpoint's own ``run_config.yaml`` carries those values,
        so the resumed plan is checked against a plan rebuilt from them.
        """
        cfg = context.state.cfg
        run_config_path = get_checkpoint_run_config_filename(
            get_checkpoint_name(cfg.checkpoint.load, context.state.train_state.step)
        )
        if not file_exists(run_config_path):
            raise RuntimeError(
                f"GR resume at iteration {context.state.train_state.step} but the checkpoint has no "
                f"run_config.yaml at {run_config_path}, so the plan it was trained under cannot be "
                "confirmed. Refusing: a silently different plan trains different data on a different "
                "schedule."
            )
        saved = read_run_config(run_config_path)
        saved_gr = saved.get("gr") or {}
        if "forget_iter_fraction" in saved_gr or "forget_data_path" in saved_gr:
            raise RuntimeError(
                "The checkpoint's run_config carries the pre-multi-module gr schema "
                "(forget_data_path/forget_iter_fraction). Mid-plan resume across the schema "
                "migration is not supported — every pre-migration GR run completed its plan, so "
                "load its final checkpoint as pretrained_checkpoint (a warm start) instead."
            )
        missing = [key for key in ("plan_seed", "aux_iter_fractions", "p_as", "p_cr") if key not in saved_gr]
        if missing:
            raise RuntimeError(
                f"The checkpoint's run_config carries no gr plan fields ({', '.join(missing)} absent) — "
                "it was not trained under gradient routing, so the plan it was trained under cannot be "
                "confirmed and resuming into this GR run would relabel every iteration. To train GR from "
                "that checkpoint, load it as checkpoint.pretrained_checkpoint (a warm start) instead."
            )
        saved_plan = build_gr_plan(
            plan_seed=saved_gr["plan_seed"],
            train_iters=saved["train"]["train_iters"],
            aux_iter_fractions=saved_gr["aux_iter_fractions"],
            p_as=saved_gr["p_as"],
            p_cr=saved_gr["p_cr"],
        )
        if saved_plan.digest() != self._plan.digest():
            raise RuntimeError(
                f"GR plan mismatch on resume: this run's plan digest is {self._plan.digest()} but the "
                f"checkpoint was trained under {saved_plan.digest()}. One of plan_seed, train_iters, "
                "aux_iter_fractions, p_as or p_cr changed, which relabels every remaining iteration "
                "and shifts the per-corpus data offsets. Restore the original values, or start a new "
                "run rather than resuming into a different experiment."
            )

    def on_train_step_start(self, context: CallbackContext) -> None:
        """Set this iteration's forward gates and expert-bias freeze from the plan."""
        it = context.state.train_state.step
        fwd_row = self._plan.fwd_aux[it]
        freeze_bias = not bool(self._plan.update_core[it])
        gate_values = None
        for layer in self._gram_layers:
            if gate_values is None or gate_values.device != layer.gr_gate.device:
                gate_values = torch.as_tensor(fwd_row, dtype=layer.gr_gate.dtype, device=layer.gr_gate.device)
            layer.gr_gate.copy_(gate_values)
        for router in self._routers:
            router.frozen_expert_bias = freeze_bias
        # Removed FIRST rather than gated on an empty handle list: an iteration that died
        # between start and end (the in-forward overflow refusal, an OOM) would otherwise leave
        # its hooks installed for the NEXT iteration to keep accumulating through, silently
        # double-counting every tensor for the rest of the run.
        self._remove_probe_hooks()
        if it % self._log_interval == 0:
            self._arm_probes()

    def _remove_probe_hooks(self) -> None:
        """Drop every probe hook this callback installed. Idempotent; called at both ends."""
        for handle in self._probe_handles:
            handle.remove()
        self._probe_handles = []

    def _arm_probes(self) -> None:
        """Register this iteration's magnitude probes on EVERY GRAM layer.

        The probe this replaces watched ``_gram_layers[0]`` only — 1 of Nano's 23 MoE layers —
        and OVERWROTE its value on every microbatch, so it reported one layer's last
        microbatch. Two real failures were invisible: a module dead at layer 0 and alive at
        depth read as dead, and a non-finite aux anywhere below layer 0 was caught by nothing
        while ``0 * inf`` poisoned the iterations where that module is supposed to be inert.
        Parameter norms do NOT substitute — in the Simple Stories campaign the arm with the
        LARGEST ``gr/aux{k}_param_norm`` did the least damage, so weights ran opposite to the
        damage.

        Three hooks per layer, all pure reads:

        - each ``gr_aux[k]``: ``||aux_k(h)||``, and at ``k == 0`` also ``||h||`` free from the
          module's own input — ``GRAMMoELayer.forward`` calls ``aux(hidden_states)``
          positionally, so ``inputs[0]`` IS the layer input (the ``pre_mlp_layernorm`` output,
          NOT the residual stream — see ``_COL_LAYER_IN``);
        - each ``gr_aux[k].linear_fc1``: the per-UNIT maximum pre-activation, folded across
          microbatches with an elementwise ``maximum``. This is the liveness signal and it is
          taken on the INPUT side of the squared ReLU on purpose:
          ``d/dz relu(z)**2 = 2 relu(z)`` is identically zero for a unit whose pre-activation
          is non-positive everywhere, which freezes that unit's row of fc1 and its column of
          fc2 forever. Measuring it here is what makes the reading survive
          ``gr_aux_output_init="zero"``, which zeroes fc2 and makes a HEALTHY fresh module's
          output exactly zero by design. Per unit rather than per tensor because the death is
          per unit: a single surviving unit would otherwise mask a copy that has lost all of
          its capacity;
        - the ``GRAMMoELayer`` itself: ``||layer output||``, plus the once-per-microbatch
          finiteness refusal (see ``_refuse_non_finite_aux``).

        Observation only, and it must stay that way — the campaign's claims rest on
        bitwise-identical trajectories wherever the plan says nothing changed. Every hook
        returns ``None`` (so the observed forward IS the unobserved one), reads a
        ``detach()``ed tensor under ``no_grad``, writes no model state, draws no RNG, changes
        no dtype on the training path, and adds no collective.

        Cost is spent on kernels, not synchronisation. Arming itself performs NO device->host
        copy: the gate snapshot is kept on device and travels in the fold's single transfer,
        so a logged iteration adds nothing to the step's critical path before the forward. Per
        tensor it is one fused fp32-accumulating reduction, APPENDED to a list rather than
        accumulated into an indexed device slot, so the whole iteration is read back by a
        single stack + copy; the one exception is the per-unit liveness vector, which IS folded
        into a per-(layer, module) slot because keeping every microbatch's copy would make the
        transfer megabytes instead of kilobytes. ``vector_norm(..., dtype=float32)`` accumulates
        in fp32 and returns a scalar; the probe it replaces built ``.float().pow(2)`` — a
        full-size fp32 tensor that then had to be reduced — at the memory peak of the step.
        """
        n_layers, n_aux = len(self._gram_layers), self._plan.n_aux
        cols = n_aux + _PROBE_EXTRA_COLS
        # ONE snapshot of the REAL gate buffers, outside any forward, kept on DEVICE — reading
        # it back here would be a full CUDA sync at the top of every logged iteration, and
        # gr.log_interval is 1 in every shipped GR config. Read from the MODEL and not from
        # plan.fwd_aux on purpose: telemetry that describes the callback's INTENT rather than
        # the model's state is exactly the blindness this change exists to remove (a layer
        # missing from the registry keeps its 0.0 gates and nothing would say so). ``stack``
        # copies, so nothing can rewrite the snapshot before the fold reads it.
        gates = torch.stack([layer.gr_gate for layer in self._gram_layers]).float()
        records: dict = {
            "cols": cols,
            "gates": gates,
            "slots": [],
            "norms": [],
            "counts": [0] * (n_layers * cols),
            "live_units": [None] * (n_layers * n_aux),
            "mb_slots": [],
            "mb_norms": [],
        }
        self._probe_records = records
        last = n_layers - 1

        def _record(slot: int, tensor: torch.Tensor) -> torch.Tensor:
            norm = torch.linalg.vector_norm(tensor, 2, dtype=torch.float32)
            records["slots"].append(slot)
            records["norms"].append(norm)
            records["counts"][slot] += tensor.numel()  # a shape read, never a device sync
            return norm

        for i, layer in enumerate(self._gram_layers):
            for k, aux in enumerate(layer.gr_aux):

                def _aux_probe(_module, inputs, output, _i=i, _k=k):
                    with torch.no_grad():
                        slot = _i * cols + _k
                        records["mb_slots"].append(slot)
                        records["mb_norms"].append(_record(slot, output.detach()))
                        if _k == 0:
                            _record(_i * cols + n_aux + _COL_LAYER_IN, inputs[0].detach())

                def _fc1_probe(_module, _inputs, output, _i=i, _k=k):
                    with torch.no_grad():
                        # linear_fc1 returns (pre_activation, bias) and GRAMAuxMLP refuses
                        # biased MLPs, so element 0 IS the squared-ReLU pre-activation, shaped
                        # [..., units]. Reduce over the token axes ONLY: a unit is dead iff it
                        # is non-positive on every token, so the per-unit max is the finest
                        # signal there is, and the running elementwise maximum extends it
                        # across the iteration's microbatches without keeping any of them.
                        pre = output[0] if isinstance(output, tuple) else output
                        unit_max = pre.detach().flatten(0, -2).amax(dim=0)
                        slot = _i * n_aux + _k
                        previous = records["live_units"][slot]
                        if previous is None:
                            records["live_units"][slot] = unit_max
                        else:
                            torch.maximum(previous, unit_max, out=previous)

                self._probe_handles.append(aux.register_forward_hook(_aux_probe))
                self._probe_handles.append(aux.linear_fc1.register_forward_hook(_fc1_probe))

            def _layer_probe(_module, _inputs, output, _i=i):
                with torch.no_grad():
                    _record(_i * cols + n_aux + _COL_LAYER_OUT, output[0].detach())
                    if _i == last:
                        self._refuse_non_finite_aux(records)

            self._probe_handles.append(layer.register_forward_hook(_layer_probe))

    def _refuse_non_finite_aux(self, records: dict) -> None:
        """Refuse a non-finite aux output — in the forward, once per microbatch, ALL layers.

        ``gate * aux(h)`` is bitwise core-only for FINITE aux output; ``0 * inf`` is NaN, so an
        aux overflow silently poisons even the iterations where that module is supposed to be
        inert. The check must stay INSIDE the forward: mcore's own fatal validation of the loss
        (``masked_next_token_loss`` -> ``rerun_state_machine.validate_result(..., fatal=True)``,
        which raises even in the default ``RerunMode.DISABLED``) fires from inside
        ``wrapped_train_step``, ahead of ``on_train_step_end`` — so a check deferred to step end
        would never run and the aux attribution, which is this message's entire value, would be
        replaced by a generic "found NaN in local forward loss calculation".

        It fires from the LAST GRAM layer's hook, by which point every layer's aux modules for
        this microbatch have been recorded, and still before the LM head and the loss. Cost:
        ONE device sync per microbatch, against the previous probe's TWO per (module,
        microbatch) at layer 0 alone (``.item()`` plus the host branch on ``isfinite``) — so
        full coverage is CHEAPER in syncs than the 1-of-23 sample was.

        This raise is rank-local, and that is deliberate rather than overlooked. The predicate
        is a function of (weights, THIS rank's tokens), so in principle ranks can disagree and
        a rank-local raise leaves the others blocked in the next collective until the NCCL/ft
        timeout — the argument that keeps ``_announce_collapse`` a warning. It does not apply
        here, because the condition it detects is one mcore ALREADY raises on, rank-locally,
        a few microseconds later: a non-finite aux output makes the layer's output non-finite
        (gate open) or NaN via ``0 * inf`` (gate closed), and ``masked_next_token_loss`` ->
        ``validate_result(..., fatal=True)`` then raises on exactly the same rank. Refusing
        here does not create a divergent-raise failure mode; it renames one, from "found NaN
        in local forward loss calculation" to the module and depth that produced it.

        The trigger is "the fp32-accumulated norm is not finite". That is exactly "the tensor
        contains an inf or a NaN" for every realistic magnitude, and additionally catches a
        still-finite bf16 tensor whose sum of squares exceeds the fp32 range — an output RMS
        above ~1e16, i.e. a module that has already destroyed the run. The message says
        "non-finite or overflowing" so the second case is not misreported as the first.
        """
        norms, slots = records["mb_norms"], records["mb_slots"]
        records["mb_norms"], records["mb_slots"] = [], []
        if not norms:
            return
        stacked = torch.stack(norms)
        if bool(torch.isfinite(stacked).all()):
            return
        first = int(torch.nonzero(~torch.isfinite(stacked.cpu()))[0])
        layer_index, k = divmod(slots[first], records["cols"])
        raise RuntimeError(
            f"gr_aux.{k} output is non-finite or overflowing at MoE layer "
            f"{self._layer_numbers[layer_index]} (its fp32-accumulated L2 norm is not finite, which "
            "means a non-finite element or an output RMS above ~1e16). The gated forward adds "
            "gate_k * aux_k(h), and 0 * inf is NaN, so a non-finite aux corrupts even the iterations "
            "where its gate is off. Lower gr.aux_lr."
        )

    def _probe_metrics_nonfatal(self, records: dict | None, it: int) -> dict:
        """``_probe_metrics``, degraded to a warning on failure.

        The fold runs on the training path (see ``_probe_metrics`` for why it is not deferred
        into the W&B thunk), so a defect in it must neither kill a 4-5 h run nor — as it would
        inside the emitter's blanket ``except`` — silently take ``gr/corpus``, ``gr/update_*``,
        the per-corpus loss and the step counters down with it.
        """
        try:
            return self._probe_metrics(records, it)
        except Exception as e:  # noqa: BLE001 — telemetry must never crash training
            logger.warning("GR: the aux magnitude/liveness fold failed (non-fatal): %s", e)
            return {}

    def _probe_metrics(self, records: dict | None, it: int) -> dict:
        """Fold one iteration's probe records into per-module depth statistics.

        ONE device->host transfer for the whole iteration — every recorded reduction, the
        per-unit liveness counts and the gate snapshot stacked and copied at once — and host
        arithmetic over an ``(n_layers, n_aux + 2)`` array after it. Squaring happens in
        float64 AFTER the copy, so a large-but-finite fp32 norm cannot overflow into a
        spurious inf; the finiteness REFUSAL lives in the forward, not here.

        Evaluated EAGERLY on every rank, on logged iterations — NOT inside the W&B thunk. The
        collapse detector below is GAP 2's only human-visible signal, and the emitter evaluates
        a thunk only after ``if wandb.run is None: return``: behind that guard the detector
        would exist on one rank, and only when W&B happened to initialise (a null
        ``logger.wandb_project``, an offline-mode failure, an import failure — and a run then
        ships dead modules with 0 NaN, a full iteration count and a valid checkpoint, which is
        precisely the failure this probe was written to make noticeable). The cost of that
        choice is one device->host copy per logged iteration on every rank, at step end where
        the training loop already synchronises for its own loss and grad-norm logging.

        Everything is rank-local and NO collective is added, so ranks cannot disagree their way
        into a hang. The aux modules are dense DP-replicated MLPs (EP shards ``experts``, not
        ``gr_aux``) and the launch guards refuse PP>1/VPP, so every rank holds every layer's
        copy of every module. Two caveats on "rank-local", both latent at the shipped TP=1:
        under TP>1 ``linear_fc1`` is COLUMN-parallel, so the liveness read sees this rank's
        1/TP slice of the units (only the row-parallel fc2 all-reduces inside the module), and
        the W&B-visible numbers are one rank's microbatches, not the global batch.

        Reporting sums-of-squares over element COUNTS (rather than a mean of means) makes every
        number an RMS — comparable across layers of different width, and unchanged if a future
        recompute posture replays a layer's forward, since numerator and denominator double
        together. On the shipped ``recompute_modules: [core_attn, moe, shared_experts]``
        posture that replay cannot happen: mcore checkpoints ``custom_forward`` INSIDE
        ``MoELayer.forward``, so the GRAM layer, its aux modules and their fc1 all sit outside
        the checkpointed region and fire exactly once per microbatch.
        """
        if not records or not records["norms"]:
            return {}
        n_layers, n_aux, cols = len(self._gram_layers), self._plan.n_aux, records["cols"]
        n_sum, n_live = len(records["norms"]), len(self._gram_layers) * self._plan.n_aux
        live_counts = unit_totals = None
        live_units = records["live_units"]
        # A module copy whose fc1 hook never fired cannot be judged, and reporting it as dead
        # would be exactly the false alarm the detector must not raise. Every aux runs every
        # microbatch by contract (Megatron's DDP buckets need a grad from every parameter), so
        # this is a guard against a future forward path, not a live branch.
        measured_live = all(unit_max is not None for unit_max in live_units)
        parts = [torch.stack(records["norms"]).to(torch.float64)]
        if measured_live:
            parts.append(torch.stack([(unit_max > 0).sum() for unit_max in live_units]).to(torch.float64))
        parts.append(records["gates"].reshape(-1).to(torch.float64))
        # float64 on the way over so the squaring below cannot overflow: an fp32 norm above
        # ~1.8e19 would, and the finiteness REFUSAL that catches the real failure lives in the
        # forward, not here — this fold must never invent an inf of its own.
        pooled = torch.cat(parts).cpu().numpy()

        offset = n_sum
        if measured_live:
            live_counts = pooled[offset : offset + n_live].reshape(n_layers, n_aux)
            offset += n_live
            # .numel() is a shape read on the host — never a device sync.
            unit_totals = np.asarray([unit_max.numel() for unit_max in live_units], dtype=np.float64).reshape(
                n_layers, n_aux
            )
        gates = pooled[offset:].reshape(n_layers, n_aux)

        sq = np.bincount(
            np.asarray(records["slots"], dtype=np.int64),
            weights=pooled[:n_sum] ** 2,
            minlength=n_layers * cols,
        ).reshape(n_layers, cols)
        counts = np.asarray(records["counts"], dtype=np.float64).reshape(n_layers, cols)
        rms = np.sqrt(sq / np.maximum(counts, 1.0))

        layer_in = rms[:, n_aux + _COL_LAYER_IN]
        layer_out = rms[:, n_aux + _COL_LAYER_OUT]
        metrics: dict = {}
        for k in range(n_aux):
            per_layer = rms[:, k]
            lo, hi = int(per_layer.argmin()), int(per_layer.argmax())
            metrics.update(
                {
                    # UNCHANGED in definition (layer 0's aux output RMS) so series either side
                    # of this commit stay joinable — but now pooled over the iteration's
                    # microbatches instead of overwritten by the last one. It is a ONE-layer
                    # sample; gr/aux{k}_out_rms_all is the module's magnitude.
                    f"gr/aux{k}_out_rms": float(per_layer[0]),
                    f"gr/aux{k}_out_rms_all": float(np.sqrt(sq[:, k].sum() / max(counts[:, k].sum(), 1.0))),
                    f"gr/aux{k}_out_rms_min": float(per_layer[lo]),
                    f"gr/aux{k}_out_rms_min_layer": self._layer_numbers[lo],
                    f"gr/aux{k}_out_rms_max": float(per_layer[hi]),
                    f"gr/aux{k}_out_rms_max_layer": self._layer_numbers[hi],
                    # ..._in_ratio_max, not _res_ratio_max: the denominator is the layer INPUT
                    # (the pre_mlp_layernorm output), not the residual stream — see
                    # _COL_LAYER_IN.
                    f"gr/aux{k}_in_ratio_max": float(
                        np.divide(per_layer, layer_in, out=np.zeros(n_layers), where=layer_in > 0.0).max()
                    ),
                }
            )
            if measured_live:
                # live_layers keeps its meaning (a copy with at least one live unit) so it
                # stays joinable, but it detects only TOTAL death; live_frac_min is the sharp
                # reading, because squared-ReLU death is per unit.
                live_layers = int((live_counts[:, k] > 0.0).sum())
                frac = live_counts[:, k] / np.maximum(unit_totals[:, k], 1.0)
                worst = int(frac.argmin())
                metrics.update(
                    {
                        f"gr/aux{k}_live_layers": live_layers,
                        f"gr/aux{k}_live_frac_min": float(frac[worst]),
                        f"gr/aux{k}_live_frac_min_layer": self._layer_numbers[worst],
                    }
                )
                self._announce_collapse(it, k, live_layers, n_layers, float(frac[worst]), self._layer_numbers[worst])
        metrics.update(self._contribution_metrics(gates, rms, layer_in, layer_out))
        return metrics

    def _contribution_metrics(self, gates, rms, layer_in, layer_out) -> dict:
        """The magnitude that governs a composability failure: ``|| sum_k gate_k * aux_k(h) ||``.

        Reported against TWO denominators, because the obvious one is not well behaved.
        ``layer_in`` — the layer's INPUT, i.e. the ``pre_mlp_layernorm`` output the modules are
        handed, and NOT the residual stream the write is later added into (see
        ``_COL_LAYER_IN``) — is gate-INDEPENDENT and unbounded, so it is comparable across
        every iteration and it is what the derived statistics key on. Read ``in_ratio`` as "how
        large the modules' write is next to what they were given"; do NOT read it as "what
        fraction of the residual stream the modules perturb", which would overstate it by
        ``||residual|| / ||norm(residual)||`` — a factor that varies with depth, i.e. along
        exactly the axis ``_max_layer`` invites you to compare. ``layer_out`` CONTAINS the
        contribution whenever a gate is open, so ``out_share`` is bounded near 1 and compresses
        exactly the divergence regime; it is kept because "what share of this layer's write is
        the modules" is the question the investigation asked (and ``layer_out`` genuinely IS
        that write), and it is honest only as long as nobody reads it as unbounded (a value
        near 1 can equally mean the module's write and the core write partially CANCEL).
        Neither is ``||contrib|| / ||core-only write||``: recovering that needs the inner
        product, not two norms.

        Exact, with no second reduction and no retained activation: every plan
        ``build_gr_plan`` can emit opens AT MOST ONE gate per iteration (aux iterations set
        their own column; core-robustness slices one disjoint permutation per module), so the
        summed contribution IS ``gate_k * aux_k(h)`` for the single open k and its norm is the
        one already recorded. A genuinely multi-open row is announced once and its keys omitted
        rather than reported wrong — a sum of norms is not the norm of a sum, and cross-module
        cancellation is precisely what a composability question is about.
        """
        open_per_layer = [np.flatnonzero(row) for row in gates]
        if any(len(open_k) > 1 for open_k in open_per_layer):
            if not self._multi_gate_announced:
                self._multi_gate_announced = True
                logger.warning(
                    "GR: more than one gate is open on a GRAM layer, which build_gr_plan cannot "
                    "produce. The gr/aux_contrib_* keys are omitted on such iterations: the norm of "
                    "the summed contribution is not recoverable from the per-module norms."
                )
            return {}
        n_layers = len(open_per_layer)
        contrib = np.zeros(n_layers)
        for i, open_k in enumerate(open_per_layer):
            if len(open_k):
                contrib[i] = float(gates[i, open_k[0]]) * rms[i, open_k[0]]
        ratio = np.divide(contrib, layer_in, out=np.zeros(n_layers), where=layer_in > 0.0)
        share = np.divide(contrib, layer_out, out=np.zeros(n_layers), where=layer_out > 0.0)
        metrics = {
            "gr/aux_contrib_in_ratio_mean": float(ratio.mean()),
            "gr/aux_contrib_in_ratio_max": float(ratio.max()),
            "gr/aux_contrib_out_share_max": float(share.max()),
        }
        # The depth label is emitted ONLY when something was open. On an all-gates-closed
        # iteration the contribution is identically zero, so argmax returns index 0 and the key
        # would report a real layer id — indistinguishable in a plot from a genuine peak at the
        # shallowest GRAM layer, on the majority of a shipped plan's iterations. The mean/max
        # keys stay: their 0.0 is the true contribution, not an artefact.
        if any(len(open_k) for open_k in open_per_layer):
            metrics["gr/aux_contrib_in_ratio_max_layer"] = self._layer_numbers[int(ratio.argmax())]
        return metrics

    def _announce_collapse(
        self, it: int, k: int, live_layers: int, n_layers: int, live_frac_min: float, worst_layer: int
    ) -> None:
        """Warn ONCE per module when squared-ReLU death has taken a layer copy, whole or in part.

        A unit whose ``linear_fc1`` pre-activation is non-positive on every token is frozen for
        the rest of the run — ``d/dz relu(z)**2`` is identically zero, so that unit's fc1 row
        and fc2 column receive exactly zero gradient — and the run otherwise finishes normally:
        full iteration count, 0 NaN, a valid checkpoint, and a composability "pass" that is
        vacuous, because a module that contributes nothing cannot interfere. That is the
        failure this exists to make noticeable.

        TWO conditions, because the death is per unit and the damage does not have to be total:
        a copy with no live unit at all (``live_layers < n_layers``), and a copy that has lost
        all but a sliver of its units (``live_frac_min < _LIVE_UNIT_FRACTION_FLOOR``). The
        second is the one a whole-tensor maximum cannot see, and it is the shape the arm that
        motivated this probe actually had.

        Judged ONLY on the module's own corpus. A specialised module being quiet on core text
        is routing WORKING, not a defect, and its pre-activations there are a different
        distribution entirely — so a campaign's success criterion must be read on iterations
        where ``gr/corpus == k + 1`` too, not on every logged iteration. No warm-up window is
        needed on top of that, because the reading is taken on the INPUT side of the
        activation: ``gr_aux_output_init="zero"`` zeroes fc2 only, so a fresh module reports
        every layer live with an output RMS of exactly 0.

        Warned, never raised. The predicate is a function of (weights, THIS RANK'S TOKENS), so
        ranks can disagree at the margin; a rank-local raise would leave the others blocked in
        the next collective until the NCCL/ft timeout — a hang rather than a crash — and making
        it safe would mean adding this callback's first collective on a conditional path. Under
        ft_launcher's ``--max-restarts=20`` with GR's final-only ``save_interval``, a
        deterministic raise is also 20 replays of a 4-5 h run. Because it is rank-local it is
        also emitted independently on every rank, which is why it is capped at one line per
        module. ``gr/aux{k}_live_layers`` and ``gr/aux{k}_live_frac_min`` are the series a
        campaign gates on, both read on the module's own corpus.
        """
        if k in self._collapse_announced:
            return
        dead_layers = n_layers - live_layers
        if dead_layers <= 0 and live_frac_min >= _LIVE_UNIT_FRACTION_FLOOR:
            return
        if int(self._plan.corpus[it]) - FIRST_AUX != k:
            return
        self._collapse_announced.add(k)
        if dead_layers > 0:
            logger.warning(
                "GR: gr_aux.%d is DEAD at %d of the %d GRAM layers on its OWN corpus at iteration %d "
                "(gr/aux%d_live_layers=%d) — not one fc1 unit of those copies is ever positive. %s",
                k,
                dead_layers,
                n_layers,
                it,
                k,
                live_layers,
                _COLLAPSE_REMEDY,
            )
        else:
            logger.warning(
                "GR: gr_aux.%d has COLLAPSED to %.1f%% live fc1 units at MoE layer %d on its OWN corpus "
                "at iteration %d (gr/aux%d_live_frac_min=%.4f, floor %.2f). Every layer copy still has "
                "SOME live unit, so gr/aux%d_live_layers reads a healthy %d — the capacity is going "
                "unit by unit. %s",
                k,
                100.0 * live_frac_min,
                worst_layer,
                it,
                k,
                live_frac_min,
                _LIVE_UNIT_FRACTION_FLOOR,
                k,
                live_layers,
                _COLLAPSE_REMEDY,
            )

    def on_train_step_end(self, context: CallbackContext) -> None:
        """Restore emptied param groups, verify the gating held, then emit telemetry."""
        self._gater.restore()
        self._remove_probe_hooks()
        # Detached here, before anything can raise below: the accumulator holds ~1k zero-dim
        # tensors plus one per-unit vector per (layer, module), so a step that failed its
        # gating verification must still drop the reference or one iteration's records would
        # live for the whole run.
        records, self._probe_records = self._probe_records, None
        it = context.state.train_state.step
        # Accumulate-then-gate is only isolation if the optimizer really skips emptied
        # groups: FusedAdam's group-level `step` counter increments exactly on visits,
        # so each module's counter must equal its armed-update count. Checked on the
        # locally-sharded groups; a sharded group without a counter on a
        # counter-carrying optimizer has never been visited and reports 0 (see
        # aux_group_adam_steps). Host-side int reads, no device sync. An undiscovered
        # gater means no optimizer ever stepped through the gate, so there is nothing
        # to verify; an optimizer without group counters cannot be verified this way,
        # which is announced once rather than silently skipped.
        if self._gater.discovered:
            observed_by_module = self._gater.aux_group_adam_steps()
            if observed_by_module is None:
                if not self._step_verification_unavailable_logged:
                    self._step_verification_unavailable_logged = True
                    logger.info(
                        "GR: optimizer carries no group-level Adam step counters — the per-module "
                        "armed-update verification is inactive on this optimizer."
                    )
            else:
                expected_by_module = {
                    k: int(self._plan.update_aux[: it + 1, k].sum()) for k in range(self._plan.n_aux)
                }
                for k, observed_steps in observed_by_module.items():
                    for observed in observed_steps:
                        if observed != expected_by_module[k]:
                            raise RuntimeError(
                                f"GR gating leak: aux module {k}'s optimizer group has taken {observed} Adam "
                                f"steps but the plan armed it {expected_by_module[k]} time(s) through iteration "
                                f"{it}. The optimizer is stepping (or skipping) emptied groups — Adam moments "
                                "and weight decay are no longer isolated per module."
                            )
        corpus = int(self._plan.corpus[it])
        metrics = {
            "gr/corpus": corpus,
            "gr/update_core": int(self._plan.update_core[it]),
            "gr/core_steps_cum": int(self._plan.update_core[: it + 1].sum()),
        }
        for k in range(self._plan.n_aux):
            metrics[f"gr/fwd_aux_{k}"] = int(self._plan.fwd_aux[it, k])
            metrics[f"gr/update_aux_{k}"] = int(self._plan.update_aux[it, k])
            metrics[f"gr/aux{k}_steps_cum"] = int(self._plan.update_aux[: it + 1, k].sum())
        loss_dict = context.loss_dict or {}
        lm_loss = loss_dict.get("lm loss")
        if lm_loss is not None:
            key = "gr/loss_core" if corpus == CORE else f"gr/loss_corpus{corpus}"
            metrics[key] = float(lm_loss.item() if torch.is_tensor(lm_loss) else lm_loss)
        # Folded EAGERLY, on every rank, because it carries the collapse detector: the emitter
        # evaluates a thunk only after `if wandb.run is None: return`, so inside the thunk the
        # GAP-2 warning would exist on one rank and only when W&B initialised — and a run whose
        # modules died would ship silently again. Gated on the log interval (a non-logged
        # iteration adds nothing) and it cannot raise (_probe_metrics_nonfatal), so a probe
        # defect can neither kill the run nor take the rest of the gr/* payload with it.
        if it % self._log_interval == 0:
            metrics.update(self._probe_metrics_nonfatal(records, it))

        def _deferred_metrics() -> dict:
            # One device sync per aux parameter, so the param-norm fold is deferred into the
            # thunk: the emitter evaluates this only on the rank that owns the W&B run, instead
            # of every rank paying for a payload the others discard. Nothing in here may raise
            # or reduce across ranks — the emitter swallows every exception, and only one rank
            # ever evaluates it. Everything that MUST be seen everywhere (the aux overflow
            # refusal, the collapse warning) therefore lives outside this thunk.
            if it % self._log_interval != 0:
                return metrics
            norms = {}
            with torch.no_grad():
                for k in range(self._plan.n_aux):
                    sq = torch.zeros((), dtype=torch.float32, device=self._gram_layers[0].gr_gate.device)
                    for layer in self._gram_layers:
                        for p in layer.gr_aux[k].parameters():
                            sq += p.float().pow(2).sum()
                    norms[f"gr/aux{k}_param_norm"] = float(sq.sqrt().item())
            return {**metrics, **norms}

        # Megatron's own iteration metrics are logged AFTER train_state.step is incremented,
        # so iteration `it` lands on W&B step `it + 1`. Matching that keeps gr/* joinable
        # against lm loss / grad-norm / learning-rate for the same iteration.
        log_wandb_metrics_nonfatal(_deferred_metrics, step=it + 1)

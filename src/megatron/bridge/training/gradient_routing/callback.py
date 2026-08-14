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

import torch

from megatron.bridge.models.mamba.gram_layer import GR_AUX_OUTPUT_INIT_DEFAULT, GRAMMoELayer
from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerGater
from megatron.bridge.training.gradient_routing.plan import CORE, GRPlan, build_gr_plan
from megatron.bridge.training.utils.checkpoint_utils import (
    file_exists,
    get_checkpoint_name,
    get_checkpoint_run_config_filename,
    read_run_config,
)
from megatron.bridge.training.utils.wandb_utils import log_wandb_metrics_nonfatal


logger = logging.getLogger(__name__)


class GRCallback(Callback):
    """Drives gates, expert-bias freezing, gater restore, and W&B telemetry."""

    def __init__(self, plan: GRPlan, gater: GROptimizerGater, log_interval: int = 1):
        self._plan = plan
        self._gater = gater
        self._log_interval = log_interval
        self._gram_layers: list[GRAMMoELayer] = []
        self._routers: list = []
        self._probe_handles: list = []
        self._probe_rms: dict[int, float] = {}
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
        log_wandb_metrics_nonfatal({"run/gr_plan_digest_int": int(self._plan.digest()[:8], 16)}, step=start)

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
        if it % self._log_interval == 0 and not self._probe_handles:
            for k, probe_module in enumerate(self._gram_layers[0].gr_aux):

                def _probe(_module, _inputs, output, _k=k):
                    with torch.no_grad():
                        rms = output.detach().float().pow(2).mean().sqrt()
                        # gate * aux(h) is bitwise core-only for FINITE aux output; 0 * inf is
                        # NaN, so an aux overflow would silently poison an iteration where that
                        # module is supposed to be inert. The probe already pays a sync here.
                        if not torch.isfinite(rms):
                            raise RuntimeError(
                                f"gr_aux.{_k} output is non-finite. The gated forward adds "
                                "gate_k * aux_k(h), and 0 * inf is NaN, so a non-finite aux corrupts "
                                "even the iterations where its gate is off. Lower gr.aux_lr."
                            )
                        self._probe_rms[_k] = float(rms.item())

                self._probe_handles.append(probe_module.register_forward_hook(_probe))

    def on_train_step_end(self, context: CallbackContext) -> None:
        """Restore emptied param groups, verify the gating held, then emit telemetry."""
        self._gater.restore()
        for handle in self._probe_handles:
            handle.remove()
        self._probe_handles = []
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
        for k, rms in self._probe_rms.items():
            metrics[f"gr/aux{k}_out_rms"] = rms
        self._probe_rms = {}

        def _with_param_norm() -> dict:
            # One device sync per aux parameter, so it is deferred into the thunk: the
            # emitter evaluates this only on the rank that owns the W&B run, instead of
            # every rank paying for a payload the others discard.
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
        log_wandb_metrics_nonfatal(_with_param_norm, step=it + 1)

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

- ``gr_gate`` on every GRAM MoE layer <- ``plan.fwd_aux[i]`` (forward activation);
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

from megatron.bridge.models.mamba.gram_layer import GRAMMoELayer
from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerGater
from megatron.bridge.training.gradient_routing.plan import FORGET, GRPlan
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
        self._probe_handle = None
        self._probe_rms: float | None = None

    def on_train_start(self, context: CallbackContext) -> None:
        """Build module registries, sanity-check the warm start, log the plan."""
        for model_chunk in context.model:
            for module in model_chunk.modules():
                if isinstance(module, GRAMMoELayer):
                    self._gram_layers.append(module)
                    self._routers.append(module.router)
        if not self._gram_layers:
            raise RuntimeError(
                "GRCallback found no GRAMMoELayer in the model — the GRAM spec swap did not run. "
                "Check model.gr_aux_ffn_hidden_size wiring."
            )
        for layer in self._gram_layers:
            w = layer.gr_aux.linear_fc2.weight
            if not torch.all(w == 0):
                raise RuntimeError(
                    "A gr_aux.linear_fc2.weight is non-zero at train start. On a warm start these "
                    "must be exactly zero (fresh zero-init, untouched by the checkpoint load); a "
                    "non-zero value means the load clobbered them or this is an unexpected resume "
                    "of a GR checkpoint mid-plan with a mismatched plan."
                )
        start = context.state.train_state.step
        if start > 0:
            logger.info("GR resume at iteration %d: plan re-derived deterministically.", start)
        logger.info("%s", self._plan.describe())
        logger.info(
            "GR: %d GRAM MoE layers, aux params %.1fM per model.",
            len(self._gram_layers),
            sum(p.numel() for layer in self._gram_layers for p in layer.gr_aux.parameters()) / 1e6,
        )
        log_wandb_metrics_nonfatal({"run/gr_plan_digest_int": int(self._plan.digest()[:8], 16)}, step=start)

    def on_train_step_start(self, context: CallbackContext) -> None:
        """Set this iteration's forward gate and expert-bias freeze from the plan."""
        it = context.state.train_state.step
        fwd = float(self._plan.fwd_aux[it])
        freeze_bias = not bool(self._plan.update_core[it])
        for layer in self._gram_layers:
            layer.gr_gate.fill_(fwd)
        for router in self._routers:
            router.frozen_expert_bias = freeze_bias
        if it % self._log_interval == 0 and self._probe_handle is None:
            probe_layer = self._gram_layers[0].gr_aux

            def _probe(_module, _inputs, output):
                with torch.no_grad():
                    self._probe_rms = float(output.detach().float().pow(2).mean().sqrt().item())

            self._probe_handle = probe_layer.register_forward_hook(_probe)

    def on_train_step_end(self, context: CallbackContext) -> None:
        """Restore emptied param groups, then emit the iteration's telemetry."""
        self._gater.restore()
        if self._probe_handle is not None:
            self._probe_handle.remove()
            self._probe_handle = None
        it = context.state.train_state.step
        corpus = int(self._plan.corpus[it])
        metrics = {
            "gr/corpus": corpus,
            "gr/fwd_aux": int(self._plan.fwd_aux[it]),
            "gr/update_core": int(self._plan.update_core[it]),
            "gr/update_aux": int(self._plan.update_aux[it]),
            "gr/aux_steps_cum": int(self._plan.update_aux[: it + 1].sum()),
            "gr/core_steps_cum": int(self._plan.update_core[: it + 1].sum()),
        }
        loss_dict = context.loss_dict or {}
        lm_loss = loss_dict.get("lm loss")
        if lm_loss is not None:
            key = "gr/loss_forget" if corpus == FORGET else "gr/loss_retain"
            metrics[key] = float(lm_loss.item() if torch.is_tensor(lm_loss) else lm_loss)
        if self._probe_rms is not None:
            metrics["gr/aux_out_rms"] = self._probe_rms
            self._probe_rms = None
        if it % self._log_interval == 0:
            with torch.no_grad():
                sq = 0.0
                for layer in self._gram_layers:
                    for p in layer.gr_aux.parameters():
                        sq += float(p.float().pow(2).sum().item())
            metrics["gr/aux_param_norm"] = sq**0.5
        log_wandb_metrics_nonfatal(metrics, step=it)

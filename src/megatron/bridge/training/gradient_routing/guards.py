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
"""Launch-time preconditions for gradient-routing runs.

Every check here protects a correctness assumption of the GR design (iteration-level
attribution, accumulate-then-gate isolation, warm-start aux init). Each raises before
any allocation time is spent, with the fix in the message — no silent fallbacks.

Config fields are read by DIRECT attribute access, never ``getattr(obj, name, default)``
with a default that would pass the check. A guard is only worth having if it fires, and a
field renamed upstream must kill the launch loudly rather than turn its guard into a
no-op. (That is not hypothetical: this file used to read
``ddp.overlap_param_gather_with_optimizer_step``, which lives on the OPTIMIZER config —
the guard had never once fired.) The two runtime slots are the deliberate exception:
``gr.runtime_plan``/``gr.runtime_gater`` are attached by the training entry script rather
than declared on ``GradientRoutingConfig``, so their absence is a real state — and it is
reported as a problem, never tolerated.
"""

from megatron.bridge.training.gradient_routing.config import GRDatasetConfig
from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerConfigOverrideProvider


#: dist_ckpt_strictness values that tolerate the base checkpoint's missing gr_aux keys.
_MISSING_KEY_TOLERANT_STRICTNESS = {"log_all", "log_unexpected", "ignore_all", "return_all", "return_unexpected"}


def validate_gr_launch(cfg) -> None:
    """Validate the fully-assembled config of a gradient-routing run; raise on any gap."""
    gr = cfg.gr
    problems: list[str] = []

    if not isinstance(cfg.dataset, GRDatasetConfig):
        problems.append("cfg.dataset must be a GRDatasetConfig (built by the cpt-branch GR wiring).")
    elif cfg.dataset.dataloader_type != "single":
        problems.append(
            f"dataset.dataloader_type must be 'single' (got {cfg.dataset.dataloader_type!r}): iteration "
            "attribution is idx // GBS under MegatronPretrainingSampler only."
        )

    if cfg.model.pipeline_model_parallel_size != 1:
        problems.append(
            f"pipeline_model_parallel_size must be 1 (got {cfg.model.pipeline_model_parallel_size}): the plan "
            "is rank-uniform and PP-safe in principle but untested beyond PP1; refuse rather than half-support."
        )
    if cfg.model.virtual_pipeline_model_parallel_size:
        problems.append("virtual_pipeline_model_parallel_size must be unset for GR runs.")
    if cfg.model.cuda_graph_impl not in (None, "none"):
        problems.append(
            f"cuda_graph_impl must be 'none' (got {cfg.model.cuda_graph_impl!r}): per-iteration gate and "
            "param-group mutation are incompatible with captured graphs."
        )
    if cfg.model.mtp_num_layers:
        problems.append("mtp_num_layers must be 0: the GRAM swap does not cover the MTP block's nested MoE spec.")
    if cfg.model.moe_shared_expert_intermediate_size is None:
        problems.append("moe_shared_expert_intermediate_size must be set: the aux module mirrors the shared expert.")
    if cfg.model.gr_aux_ffn_hidden_size != gr.aux_ffn_hidden_size:
        problems.append(
            f"model.gr_aux_ffn_hidden_size ({cfg.model.gr_aux_ffn_hidden_size}) != gr.aux_ffn_hidden_size "
            f"({gr.aux_ffn_hidden_size}) — one config, one width."
        )

    if cfg.train.rampup_batch_size:
        problems.append("train.rampup_batch_size must be unset: GBS must be constant for iteration attribution.")
    if cfg.train.decrease_batch_size_if_needed:
        problems.append("train.decrease_batch_size_if_needed must be False: GBS must be constant for attribution.")

    if "adam" not in str(cfg.optimizer.optimizer):
        problems.append(
            f"optimizer.optimizer must be adam-family (got {cfg.optimizer.optimizer!r}): the gating analysis "
            "(param-group emptying leaves moments untouched) is argued for Adam."
        )
    # Lives on the optimizer config, not on ddp: setup.py reads
    # cfg.optimizer.overlap_param_gather_with_optimizer_step to build the distributed model.
    if cfg.optimizer.overlap_param_gather_with_optimizer_step:
        problems.append("optimizer.overlap_param_gather_with_optimizer_step must be False for GR runs.")
    if cfg.optimizer.optimizer_cpu_offload:
        problems.append(
            "optimizer.optimizer_cpu_offload must be False for GR runs: under CPU offload the inner "
            "optimizer is a HybridDeviceOptimizer that steps its own gpu/cpu sub-optimizer param "
            "lists, not the param_groups the gater empties — so the gate would be a silent no-op "
            "(and HDO's param_in_param_group_index lookup raises KeyError on the emptied group)."
        )
    if cfg.inprocess_restart is not None:
        problems.append(
            "inprocess_restart must be unset for GR runs: an in-process restart rebuilds the "
            "optimizer while the callback keeps the gater created for the dead one, and the gater "
            "caches its discovery — it would then empty a stale optimizer's groups while the live "
            "one steps every parameter, losing isolation silently."
        )
    if not isinstance(cfg.optimizer_config_override_provider, GROptimizerConfigOverrideProvider):
        problems.append(
            "optimizer_config_override_provider must be the GROptimizerConfigOverrideProvider "
            "(installed by the cpt-branch GR wiring)."
        )

    strictness = str(cfg.checkpoint.dist_ckpt_strictness)
    if strictness not in _MISSING_KEY_TOLERANT_STRICTNESS:
        problems.append(
            f"checkpoint.dist_ckpt_strictness={strictness!r} does not tolerate missing keys; the warm-start "
            "base checkpoint has no gr_aux tensors (fresh zero-init fills them). Use e.g. 'log_all'."
        )

    if cfg.validation.eval_iters != 0:
        problems.append("validation.eval_iters must be 0: the routed dataset serves no validation split (v1).")

    # Attached by the training entry script, not declared on GradientRoutingConfig — absence
    # is a legitimate runtime state (an unwired run), and it is reported, not tolerated.
    runtime_plan = getattr(gr, "runtime_plan", None)
    runtime_gater = getattr(gr, "runtime_gater", None)
    if runtime_plan is None or runtime_gater is None:
        problems.append("cfg.gr.runtime_plan/runtime_gater missing — GR wiring incomplete.")
    elif runtime_plan.train_iters != cfg.train.train_iters:
        problems.append(
            f"plan covers {runtime_plan.train_iters} iters but train.train_iters is "
            f"{cfg.train.train_iters} — train_iters changed after the plan was built."
        )

    if problems:
        raise ValueError("Gradient-routing launch guards failed:\n- " + "\n- ".join(problems))

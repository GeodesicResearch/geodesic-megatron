#!/usr/bin/env python3
"""One finite cSDF fit. Launched with torchrun by the posthoc campaign runner."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    """Publish one complete artifact beside its PID-local temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True))
    os.replace(temporary, path)


def main() -> None:
    """Load one immutable fit, train finite documents, and publish adapter snapshots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True)
    args = parser.parse_args()
    fit = json.loads(Path(args.fit).read_text())
    output = Path(fit["output_dir"])
    recipe = fit["training"]
    # Imports intentionally inside main: CPU config tooling need not load CUDA.
    import torch
    from scripts.csdf.dataset import DocumentProvider
    from scripts.csdf.documents import prepare_documents
    from scripts.csdf.export import export_factors
    from transformers import AutoTokenizer

    from megatron.bridge import AutoBridge
    from megatron.bridge.peft.lora import LoRA
    from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_peft_config
    from megatron.bridge.training.callbacks import Callback
    from megatron.bridge.training.gpt_step import forward_step
    from megatron.bridge.training.pretrain import pretrain

    tokenizer = AutoTokenizer.from_pretrained(fit.get("tokenizer", fit["base_model"]), trust_remote_code=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Document tokenizer must declare EOS")
    documents, data = prepare_documents(fit, tokenizer)
    cfg = nemotron_3_nano_peft_config(
        LoRA(
            target_modules=recipe["target_modules"],
            dim=recipe["rank"],
            alpha=recipe["alpha"],
            dropout=0.0,
        )
    )
    cfg.model = AutoBridge.from_hf_pretrained(fit["base_model"], trust_remote_code=True).to_megatron_provider(
        load_weights=False
    )
    cfg.model.calculate_per_token_loss = True
    cfg.model.variable_seq_lengths = True
    cfg.checkpoint.pretrained_checkpoint = None
    cfg.checkpoint.hf_parent = fit["base_model"]
    cfg.checkpoint.parent_factor_sources = fit["parent_sources"]
    cfg.checkpoint.save = str(output / "resume")
    cfg.checkpoint.load = cfg.checkpoint.save
    cfg.checkpoint.save_steps = data["save_steps"]
    cfg.checkpoint.save_interval = 0  # Explicit schedule, including final if requested.
    cfg.checkpoint.async_save = False
    cfg.train.train_iters = data["total_steps"]
    cfg.train.global_batch_size = recipe["global_batch_size"]
    cfg.train.micro_batch_size = recipe["micro_batch_size"]
    cfg.model.seq_length = recipe["max_seq_length"]
    token_path = output / "tokens.json"
    if int(os.environ["RANK"]) == 0:
        previous = output / "data.json"
        if previous.exists() and json.loads(previous.read_text()) != data:
            raise ValueError("Tokenized corpus changed on resume")
        atomic_json(token_path, {"documents": documents})
    cfg.dataset = DocumentProvider(
        document_path=str(token_path), pad_id=tokenizer.eos_token_id, seq_length=cfg.model.seq_length
    )
    cfg.tokenizer.tokenizer_model = fit.get("tokenizer", fit["base_model"])
    cfg.optimizer.use_distributed_optimizer = False  # Adapter state is small; simplify snapshot parity.
    cfg.ddp.overlap_param_gather = False
    cfg.optimizer.lr = recipe["lr"]
    cfg.optimizer.min_lr = recipe.get("min_lr", 0.0)
    cfg.optimizer.adam_beta1 = recipe.get("adam_beta1", 0.9)
    cfg.optimizer.adam_beta2 = recipe.get("adam_beta2", 0.999)
    cfg.optimizer.weight_decay = recipe.get("weight_decay", 0.0)
    cfg.scheduler.start_weight_decay = cfg.optimizer.weight_decay
    cfg.scheduler.end_weight_decay = cfg.optimizer.weight_decay
    cfg.scheduler.lr_warmup_iters = recipe["warmup_steps"]
    cfg.scheduler.lr_decay_iters = data["total_steps"]
    cfg.scheduler.lr_decay_style = "cosine"
    cfg.rng.seed = fit["seed"]
    cfg.validation.eval_iters = 0
    cfg.logger.wandb_project = None
    cfg.logger.tensorboard_dir = None
    cfg.model.moe_router_bias_update_rate = 0.0  # Preserve parent routing biases.
    cfg.model.moe_aux_loss_coeff = 0.0
    cfg.model.moe_router_load_balancing_type = "none"
    cfg.comm_overlap = None
    for name, value in recipe["model"].items():
        if not hasattr(cfg.model, name):
            raise ValueError(f"Unknown model setting {name}")
        setattr(cfg.model, name, value)
    if cfg.model.pipeline_model_parallel_size != 1 or cfg.model.context_parallel_size != 1:
        raise ValueError("Initial cSDF backend supports PP=CP=1; document batches are independent")
    if cfg.model.moe_router_bias_update_rate or cfg.model.moe_aux_loss_coeff:
        raise ValueError("cSDF must freeze router parameters and bias updates")
    world = int(os.environ["WORLD_SIZE"])
    dp = world // cfg.model.tensor_model_parallel_size
    if recipe["global_batch_size"] % (recipe["micro_batch_size"] * dp):
        raise ValueError("Global document batch must be divisible by microbatch * DP")

    class Artifacts(Callback):
        def on_train_start(self, context):
            self.completed_step = context.state.train_state.step
            for chunk in context.model:
                for module in chunk.modules():
                    if hasattr(module, "frozen_expert_bias"):
                        module.frozen_expert_bias = True
            self.bridge = AutoBridge.from_hf_pretrained(fit["base_model"], trust_remote_code=True)
            unexpected = [
                n for m in context.model for n, p in m.named_parameters() if p.requires_grad and ".adapter." not in n
            ]
            unexpected += [
                n
                for m in context.model
                for n, p in m.named_parameters()
                if p.requires_grad and (".experts." in n and ".shared_experts." not in n)
            ]
            if unexpected:
                raise ValueError(f"Unexpected trainable base parameters: {unexpected}")
            if 0 in data["save_steps"] and self.completed_step == 0:
                export_factors(self.bridge, context.model, output / "snapshots" / "step_0")
            if torch.distributed.get_rank() == 0:
                atomic_json(output / "data.json", data)
                atomic_json(output / "fit.json", fit)

        def on_train_step_end(self, context):
            step = context.state.train_state.step + 1  # Callback runs before the loop increments.
            self.completed_step = step
            if context.skipped_iter:
                raise RuntimeError(f"Skipped optimizer step {step}; refusing a misleading dose axis")
            if step in data["save_steps"]:
                if hasattr(context.optimizer, "finish_param_sync"):
                    context.optimizer.finish_param_sync()
                export_factors(self.bridge, context.model, output / "snapshots" / f"step_{step}")
            if torch.distributed.get_rank() == 0:
                tokens = sum(len(d) - 1 for d in documents[: step * recipe["global_batch_size"]])
                metrics = {k: float(v.float().mean().item()) for k, v in (context.loss_dict or {}).items()}
                atomic_json(
                    output / "events" / f"step_{step:08d}.json",
                    {
                        "fit_id": fit["id"],
                        "sdf_step": step,
                        "rl_step": fit["rl_step"],
                        "documents_seen": step * recipe["global_batch_size"],
                        "tokens_seen": tokens,
                        "epoch": step * recipe["global_batch_size"] / data["documents_per_epoch"],
                        "metrics": metrics,
                    },
                )

    artifacts = Artifacts()
    pretrain(cfg, forward_step, callbacks=[artifacts])
    if artifacts.completed_step != data["total_steps"]:
        raise RuntimeError(
            f"Fit stopped at {artifacts.completed_step}/{data['total_steps']}; resume before evaluating"
        )
    if int(os.environ["RANK"]) == 0:
        atomic_json(output / "complete.json", {"fit_id": fit["id"], **data})


if __name__ == "__main__":
    main()

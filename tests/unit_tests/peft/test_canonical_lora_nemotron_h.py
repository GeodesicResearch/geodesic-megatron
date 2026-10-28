# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Canonical LoRA on a small real Nemotron-H Latent-MoE model: every Super target, its forward, its merge, its HF
export and the converter's load and export of an adapter-only checkpoint."""

import pytest
import torch

from megatron.bridge.recipes.nemotronh.nemotron_3_super import NEMOTRON_3_SUPER_CANONICAL_LORA_TARGETS
from tests.unit_tests.one_rank_nccl_world import one_rank_model_parallel_state
from tests.unit_tests.small_nemotron_h import (
    D_INNER,
    SEQ,
    VOCAB,
    small_nemotron_h_provider,
    write_small_nemotron_h_release,
)


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (Mamba and grouped-GEMM kernels)"),
    pytest.mark.skipif(not hasattr(torch, "_grouped_mm"), reason="needs torch._grouped_mm"),
]


@pytest.fixture(scope="module")
def parallel_state():
    with one_rank_model_parallel_state(seed=1234, expert_model_parallel_size=1):
        yield


def _model():
    """Mamba, attention and Latent-MoE layers (with a shared expert) on the GroupedExperts path, in bf16."""
    torch.manual_seed(0)
    provider = small_nemotron_h_provider("M*E", bf16=True, params_dtype=torch.bfloat16)
    return provider.provide_distributed_model(wrap_with_ddp=False)


def _super_lora():
    """CanonicalLoRA on every Super target."""
    from megatron.bridge.peft.canonical_lora import CanonicalLoRA

    return CanonicalLoRA(target_modules=list(NEMOTRON_3_SUPER_CANONICAL_LORA_TARGETS), lora_A_init_method="kaiming")


def _adapted(model=None, b_std=0.05):
    """``_super_lora`` on ``model`` (a new small model if None), B factors ~ N(0, b_std)."""
    model = model if model is not None else _model()
    model = _super_lora()(model, training=False)
    # Non-zero B factors, so that every adapter changes the output.
    with torch.no_grad():
        for name, param in model[0].named_parameters():
            if ".adapter." in name and ("linear_out" in name or name.endswith(("fc1_b", "fc2_b"))):
                param.normal_(std=b_std)
    return model


def _logits(model):
    torch.manual_seed(1)
    tokens = torch.randint(0, VOCAB, (1, SEQ), device="cuda")
    position_ids = torch.arange(SEQ, device="cuda").unsqueeze(0)
    with torch.no_grad():
        return model[0](tokens, position_ids, attention_mask=None).float()


def test_every_super_target_gets_an_adapter(parallel_state):
    from megatron.bridge.peft.canonical_lora import LoRALinearSplitMambaInProj, LoRALinearSplitQKV
    from megatron.bridge.peft.grouped_experts_lora import LoRAGroupedExperts
    from megatron.bridge.peft.lora_layers import LoRALinear

    model = _adapted()
    kinds = {name.rsplit(".", 1)[-1]: type(m) for name, m in model[0].named_modules()}
    assert kinds["linear_qkv"] is LoRALinearSplitQKV
    assert kinds["in_proj"] is LoRALinearSplitMambaInProj
    assert kinds["experts"] is LoRAGroupedExperts
    for plain in ("linear_proj", "out_proj", "fc1_latent_proj", "fc2_latent_proj", "linear_fc1", "linear_fc2"):
        assert kinds[plain] is LoRALinear, plain
    assert "router" not in {n for n, m in model[0].named_modules() if isinstance(m, LoRALinear)}
    trainable = [n for n, p in model[0].named_parameters() if p.requires_grad]
    assert trainable == []  # training=False freezes everything, adapters included


def test_adapters_change_the_output(parallel_state):
    model = _adapted()
    adapted = _logits(model)
    _set_adapters(model, enabled=False)
    assert (adapted - _logits(model)).abs().max() > 1e-2


def test_a_partial_in_proj_target_is_refused(parallel_state):
    from megatron.bridge.peft.canonical_lora import CanonicalLoRA

    with pytest.raises(ValueError, match="both 'in_proj_z' and 'in_proj_x'"):
        CanonicalLoRA(target_modules=["in_proj_x"], lora_A_init_method="kaiming")(_model(), training=False)


def test_plain_lora_on_canonical_target_names_is_refused(parallel_state):
    """Plain LoRA given CanonicalLoRA's names raises when applied, naming them as CanonicalLoRA targets."""
    from megatron.bridge.peft.lora import LoRA

    lora = LoRA(target_modules=list(NEMOTRON_3_SUPER_CANONICAL_LORA_TARGETS))
    with pytest.raises(ValueError, match=r"match no module of the model; \[.*'linear_q'.*\] are CanonicalLoRA target"):
        lora(_model(), training=False)


@pytest.mark.parametrize("scheme", ["lora", "canonical_lora", "dora"])
def test_a_target_that_matches_no_module_is_refused(parallel_state, scheme):
    from megatron.bridge.peft.canonical_lora import CanonicalLoRA
    from megatron.bridge.peft.dora import DoRA
    from megatron.bridge.peft.lora import LoRA

    peft = {"lora": LoRA, "canonical_lora": CanonicalLoRA, "dora": DoRA}[scheme](
        target_modules=["linear_proj", "*.layers.9.*.out_proj"]
    )
    with pytest.raises(ValueError, match=r"\['\*\.layers\.9\.\*\.out_proj'\] match no module of the model$"):
        peft(_model(), training=False)


def _set_adapters(model, enabled: bool):
    """PEFT's own enable/disable walk, through a PEFT that holds no state of its own."""
    from megatron.bridge.peft.lora import LoRAMerge

    (LoRAMerge().enable_adapter_layers if enabled else LoRAMerge().disable_adapter_layers)(model)


def test_mamba_in_proj_update_is_in_the_z_and_x_rows_only(parallel_state):
    model = _adapted()
    in_proj = next(m for n, m in model[0].named_modules() if n.endswith("mixer.in_proj"))
    rows = in_proj.weight_update("weight").abs().sum(dim=1)
    assert rows.shape[0] == in_proj.to_wrap.weight.shape[0]
    assert rows[: 2 * D_INNER].gt(0).all()
    assert rows[2 * D_INNER :].eq(0).all()


def test_loramerge_folds_every_adapter_and_reproduces_the_adapted_logits(parallel_state):
    """LoRAMerge merges every wrapper of every Super target; with the adapters then off, the logits are the adapted."""
    from megatron.bridge.peft.adapter_wrapper import AdapterWrapper
    from megatron.bridge.peft.lora import LoRAMerge

    model = _adapted()
    adapted = _logits(model)
    _set_adapters(model, enabled=False)
    base = _logits(model)
    wrappers = {name for name, module in model[0].named_modules() if isinstance(module, AdapterWrapper)}
    before = {name: p.detach().clone() for name, p in model[0].named_parameters() if ".to_wrap." in name}

    LoRAMerge()(model[0], training=False)

    changed = {
        name.partition(".to_wrap.")[0]
        for name, p in model[0].named_parameters()
        if name in before and not torch.equal(p, before[name])
    }
    assert changed == wrappers
    merged = _logits(model)
    assert (adapted - base).abs().max() > 0.5  # the adapters' effect, against which the merge error is small
    # bf16 weights: the merged sum rounds once per weight; compare at bf16 resolution of the logits.
    torch.testing.assert_close(merged, adapted, rtol=3e-2, atol=3e-2)


def test_loramerge_folds_router_lora(parallel_state):
    """The router's routing probabilities after the merge, adapter off, are the adapted router's."""
    from megatron.bridge.peft.lora import LoRA, LoRAMerge
    from megatron.bridge.peft.lora_layers import LoRATopKRouter

    model = LoRA(target_modules=["router"])(_model(), training=False)
    router = next(m for m in model[0].modules() if isinstance(m, LoRATopKRouter))
    with torch.no_grad():
        router.adapter.linear_out.weight.normal_(std=0.5)
    probs = []
    router.register_forward_hook(lambda module, args, output: probs.append(output[0].float()))
    _logits(model)
    _set_adapters(model, enabled=False)
    _logits(model)
    LoRAMerge()(model[0], training=False)
    _logits(model)
    adapted, base, merged = probs
    assert (adapted - base).abs().max() > 0.1  # the adapter's effect, against which the merge error is small
    torch.testing.assert_close(merged, adapted, rtol=1e-2, atol=1e-2)


def test_loramerge_refuses_a_wrapper_it_cannot_merge(parallel_state):
    from megatron.bridge.peft.dora import DoRA
    from megatron.bridge.peft.lora import LoRAMerge

    model = DoRA(target_modules=["linear_proj"])(_model(), training=False)
    with pytest.raises(NotImplementedError, match="LoRAMerge cannot merge DoRALinear"):
        LoRAMerge()(model[0], training=False)


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    return write_small_nemotron_h_release(tmp_path_factory.mktemp("release"))


def _bridge_model(hf_path, moe_experts_impl: str, load_weights: bool):
    """The bridge's Megatron model of the HF checkpoint ``hf_path``, on the given experts implementation."""
    from megatron.bridge import AutoBridge

    torch.manual_seed(0)
    provider = AutoBridge.from_hf_pretrained(str(hf_path)).to_megatron_provider(load_weights=load_weights)
    provider.moe_experts_impl = moe_experts_impl
    provider.moe_aux_loss_coeff = 0.0
    provider.finalize()
    return provider.provide_distributed_model(wrap_with_ddp=False)


def _export(model, release, out):
    from megatron.bridge import AutoBridge

    AutoBridge.from_hf_pretrained(str(release)).save_hf_pretrained(model, str(out), show_progress=False)
    return out


def test_grouped_experts_export_imports_as_the_same_model(parallel_state, release, tmp_path):
    """A GroupedExperts model exported to HF and imported on TEGroupedMLP gives the same logits."""
    model = _bridge_model(release, "torch_grouped", load_weights=False)
    exported = _bridge_model(_export(model, release, tmp_path / "hf"), "te_grouped", load_weights=True)
    torch.testing.assert_close(_logits(exported), _logits(model), rtol=3e-2, atol=3e-2)


def test_hf_export_merges_every_expert_lora_adapter(parallel_state, release, tmp_path):
    """The HF export of the adapted model, imported without adapters, gives the adapted logits."""
    model = _adapted(_bridge_model(release, "torch_grouped", load_weights=False))
    adapted = _logits(model)
    _set_adapters(model, enabled=False)
    base = _logits(model)
    _set_adapters(model, enabled=True)
    merged = _bridge_model(_export(model, release, tmp_path / "hf"), "te_grouped", load_weights=True)
    assert (adapted - base).abs().max() > 0.5  # the adapters' effect, against which the merge error is small
    torch.testing.assert_close(_logits(merged), adapted, rtol=3e-2, atol=3e-2)


def test_hf_export_refuses_an_adapter_it_cannot_merge(parallel_state, release, tmp_path):
    """DoRA has LoRA factors but also a magnitude, which a LoRA merge would drop: the export raises instead."""
    from megatron.bridge.peft.dora import DoRA

    model = DoRA(target_modules=["linear_proj"])(_bridge_model(release, "torch_grouped", load_weights=False))
    with pytest.raises(ValueError, match=r"cannot merge these adapter wrappers: \['DoRALinear at "):
        _export(model, release, tmp_path / "hf")


def test_grouped_experts_weights_do_not_import_from_hf(parallel_state, release):
    with pytest.raises(NotImplementedError, match="GroupedExperts weights export to HF only"):
        _bridge_model(release, "torch_grouped", load_weights=True)


def test_grouped_experts_mapping_needs_the_expert_wildcard():
    from megatron.bridge.models.conversion.param_mapping import GroupedExpertsMapping

    with pytest.raises(ValueError, match="one wildcard more"):
        GroupedExpertsMapping("decoder.layers.*.mlp.experts.weight1", "backbone.layers.*.mixer.experts.up_proj.weight")


def _peft_checkpoint(tmp_path, release):
    """A base checkpoint and an Expert LoRA PEFT checkpoint of the small model, as a training run leaves them.

    The PEFT checkpoint's run_config names the base under ``checkpoint.pretrained_checkpoint``, the adapters under
    ``peft`` (serialised as training serialises it) and the gradient hooks training records, which rebuild as
    unbound methods; its tensors include the base weights, which ``load_peft_model`` must ignore in favour of the
    base's. Returns (adapted in-memory model, PEFT checkpoint directory).
    """
    import yaml

    from megatron.bridge.training.model_load_save import save_megatron_model
    from megatron.bridge.training.utils.config_utils import _ConfigContainerBase

    model = _bridge_model(release, "torch_grouped", load_weights=False)
    save_megatron_model(model, tmp_path / "base", hf_tokenizer_path=str(release))
    adapted = _adapted(model)
    save_megatron_model(adapted, tmp_path / "peft", hf_tokenizer_path=str(release))
    run_config_path = tmp_path / "peft" / "iter_0000000" / "run_config.yaml"
    run_config = yaml.safe_load(run_config_path.read_text())
    run_config["checkpoint"]["pretrained_checkpoint"] = str(tmp_path / "base")
    run_config["peft"] = _ConfigContainerBase._convert_value_to_dict(_super_lora())
    ddp = "megatron.core.distributed.distributed_data_parallel.DistributedDataParallel"
    run_config["model"]["no_sync_func"] = {"_call_": False, "_target_": f"{ddp}.no_sync"}
    run_config["model"]["grad_sync_func"] = {"_call_": False, "_target_": f"{ddp}.start_grad_sync"}
    run_config_path.write_text(yaml.safe_dump(run_config))
    return adapted, tmp_path / "peft"


def test_load_peft_model_restores_the_adapted_model(parallel_state, release, tmp_path):
    from megatron.bridge.training.model_load_save import load_peft_model

    adapted, peft_dir = _peft_checkpoint(tmp_path, release)
    loaded = load_peft_model(str(peft_dir / "iter_0000000"))
    # The run's recorded gradient hooks would rebuild as unbound DDP methods, which the pipeline schedule calls.
    assert loaded[0].config.no_sync_func is None and loaded[0].config.grad_sync_func is None
    torch.testing.assert_close(_logits(loaded), _logits(adapted), rtol=0, atol=0)


def test_load_peft_model_refuses_a_checkpoint_without_the_adapters(parallel_state, release, tmp_path):
    """The base checkpoint relabelled as a PEFT checkpoint holds no adapter tensors: the load must fail, not no-op."""
    import yaml
    from torch.distributed.checkpoint.api import CheckpointException

    from megatron.bridge.training.model_load_save import load_peft_model

    _, peft_dir = _peft_checkpoint(tmp_path, release)
    base_iter = tmp_path / "base" / "iter_0000000"
    run_config = yaml.safe_load((base_iter / "run_config.yaml").read_text())
    run_config["checkpoint"]["pretrained_checkpoint"] = str(tmp_path / "base")
    run_config["peft"] = yaml.safe_load((peft_dir / "iter_0000000" / "run_config.yaml").read_text())["peft"]
    (base_iter / "run_config.yaml").write_text(yaml.safe_dump(run_config))
    with pytest.raises(CheckpointException, match="adapter_z.linear_in.weight from model not in state dict"):
        load_peft_model(str(base_iter))


@pytest.mark.serial_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 3, reason="needs 2 GPUs besides the test process's own")
def test_the_converter_exports_an_adapter_only_checkpoint_merged(parallel_state, release, tmp_path):
    """pipeline_checkpoint_convert_hf.py under torchrun (EP2) on the adapter-only checkpoint: an HF checkpoint whose
    logits, imported without adapters, are the adapted model's, and a passing merge check beside it."""
    import json

    from tests.unit_tests.child_process import REPO_ROOT, free_port, run_python

    adapted_model, peft_dir = _peft_checkpoint(tmp_path, release)
    adapted = _logits(adapted_model)
    _set_adapters(adapted_model, enabled=False)
    base = _logits(adapted_model)
    hf_path = tmp_path / "hf"
    run_python(
        [
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            f"--master_port={free_port()}",
            str(REPO_ROOT / "pipeline_checkpoint_convert_hf.py"),
            f"--megatron-path={peft_dir}",
            f"--hf-model={release}",
            f"--hf-path={hf_path}",
            "--no-reasoning",
            "--ep=2",
            "--no-progress",
        ],
        timeout=900,
        cwd=tmp_path,
        CUDA_VISIBLE_DEVICES="1,2",
    )
    merged = _logits(_bridge_model(hf_path, "te_grouped", load_weights=True))
    assert (adapted - base).abs().max() > 0.5
    torch.testing.assert_close(merged, adapted, rtol=3e-2, atol=3e-2)
    check = json.loads((hf_path / "merge_check.json").read_text())
    assert check["retained"] > 0.95
    assert check["merged"]["mean_abs_diff"] < 0.1 * check["base"]["mean_abs_diff"]

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
"""pipeline_training_run.py, recorded as goldens: its --help text, the config each launch trains, and the patches
its environment gates install.

``main()`` runs for real up to the training entry point (``run_training_main``). The Super and Ultra recipes read their HF configs from the shared cache,
offline.
"""

import sys

import pytest
from omegaconf import OmegaConf

from megatron.bridge.training.utils.omegaconf_utils import create_omegaconf_dict_config
from tests.unit_tests.goldens import FIXTURES, assert_golden_bytes, assert_golden_json
from tests.unit_tests.repo_scripts import run_training_main


GOLDENS = FIXTURES / "training_run"


def test_help_text(run_module, monkeypatch, capsys):
    monkeypatch.setenv("COLUMNS", "120")
    monkeypatch.setattr(sys, "argv", ["pipeline_training_run.py", "--help"])
    with pytest.raises(SystemExit) as stop:
        run_module.parse_cli_args()
    assert stop.value.code == 0
    assert_golden_bytes(GOLDENS / "help.txt", capsys.readouterr().out.encode())


def _packed_parquet(tmp_path):
    """A one-row packed parquet: the SFT launches point their packed_train_data_path at it."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "packed" / "training_64.idx.parquet"
    path.parent.mkdir()
    pq.write_table(pa.table({"input_ids": [[1, 2, 3]], "loss_mask": [[0, 1, 1]], "seq_start_id": [[0]]}), path)
    return path


def _yaml(mode, tmp_path) -> str:
    text = "tokenizer:\n  tokenizer_model: geodesic-research/nemotron-instruct-tokenizer\n"
    text += f"checkpoint:\n  save: {tmp_path}/ckpt\n  save_interval: 5\n"
    if mode == "sft":
        text += f"dataset:\n  packed_sequence_specs:\n    packed_train_data_path: {_packed_parquet(tmp_path)}\n"
    else:
        text += f'dataset:\n  data_path:\n    - "1.0"\n    - {tmp_path}/corpus_input_document\n  seq_length: 4096\n'
    return text


LAUNCHES = {
    **{
        f"{model}_{mode}": (model, mode, [])
        for model in ("nano", "super", "ultra")
        for mode in ("sft", "cpt", "pretrain")
    },
    **{f"{model}_sft_lora": (model, "sft", ["--peft", "lora"]) for model in ("nano", "super", "ultra")},
    "nano_sft_disable_ft": ("nano", "sft", ["--disable-ft"]),
    "nano_sft_disable_straggler": ("nano", "sft", ["--disable-straggler"]),
    "nano_cpt_pao": ("nano", "cpt", ["--enable-pao"]),
}


def run_main(run_module, monkeypatch, tmp_path, model, mode, flags, overrides=("train.train_iters=7",)):
    """The training entry ``main()`` calls and the kwargs it calls it with, for one launch from ``tmp_path``.

    The recipes place their experiment directories under the working directory.
    """
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "override.yaml"
    config.write_text(_yaml(mode, tmp_path))
    argv = ["--model", model, "--mode", mode, "--config-file", str(config), *flags]
    calls = run_training_main(run_module, monkeypatch, [*argv, *overrides, "logger.wandb_exp_name=golden run"])
    (entry,) = calls
    return entry, calls[entry]


@pytest.mark.parametrize("case", sorted(LAUNCHES))
def test_resolved_config(run_module, monkeypatch, tmp_path, case):
    """The ConfigContainer the launch trains, as the profiler's resolved-config dump serialises it."""
    model, mode, flags = LAUNCHES[case]
    entry, kwargs = run_main(run_module, monkeypatch, tmp_path, model, mode, flags)
    resolved, _ = create_omegaconf_dict_config(kwargs["config"])
    dump = f"# entry: {entry}\n" + OmegaConf.to_yaml(resolved, resolve=True)
    assert_golden_bytes(GOLDENS / "resolved" / f"{case}.yaml", dump.replace(str(tmp_path), "<TMP>").encode())


GATES = {
    "fp32_checkpoint": {"ISAMBARD_FP32_SSM_STATE": "checkpoint"},
    "fp32_direct_and_offload": {"ISAMBARD_FP32_SSM_STATE": "1", "ISAMBARD_MAMBA_SAVE_OFFLOAD": "1"},
    "fp32_checkpoint_and_offload": {"ISAMBARD_FP32_SSM_STATE": "checkpoint", "ISAMBARD_MAMBA_SAVE_OFFLOAD": "1"},
    "offload_only": {"ISAMBARD_MAMBA_SAVE_OFFLOAD": "1"},
    "comm_warmup": {"ISAMBARD_COMM_WARMUP": "1"},
    "row_telemetry": {"ISAMBARD_DATA_ROW_TELEMETRY": "1"},
    "all_four": {
        "ISAMBARD_FP32_SSM_STATE": "1",
        "ISAMBARD_MAMBA_SAVE_OFFLOAD": "1",
        "ISAMBARD_COMM_WARMUP": "1",
        "ISAMBARD_DATA_ROW_TELEMETRY": "1",
    },
}


@pytest.mark.parametrize("case", sorted(GATES))
def test_env_gates_install_their_patches(run_module, monkeypatch, tmp_path, case):
    """What each environment gate leaves installed on megatron.core and the packed dataset after ``main()``."""
    from megatron.core import parallel_state
    from megatron.core.ssm import mamba_mixer

    from megatron.bridge.data.datasets.packed_parquet import GPTSFTPackedParquetDataset

    # Re-set each patched attribute to itself so that monkeypatch restores it when the test ends.
    monkeypatch.setattr(mamba_mixer, "mamba_split_conv1d_scan_combined", mamba_mixer.mamba_split_conv1d_scan_combined)
    monkeypatch.setattr(parallel_state, "initialize_model_parallel", parallel_state.initialize_model_parallel)
    monkeypatch.setattr(GPTSFTPackedParquetDataset, "__getitem__", GPTSFTPackedParquetDataset.__getitem__)
    for name in (
        "ISAMBARD_FP32_SSM_STATE",
        "ISAMBARD_MAMBA_SAVE_OFFLOAD",
        "ISAMBARD_COMM_WARMUP",
        "ISAMBARD_DATA_ROW_TELEMETRY",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in GATES[case].items():
        monkeypatch.setenv(name, value)

    run_main(run_module, monkeypatch, tmp_path, "nano", "sft", ["--disable-ft"])

    scan = mamba_mixer.mamba_split_conv1d_scan_combined
    chain, fn = [], scan
    while fn is not None:
        chain.append(getattr(fn, "__name__", repr(fn)))
        fn = getattr(fn, "_wrapped", None)
    record = {
        "scan_chain": chain,
        "fp32_mode": getattr(scan, "_fp32_ssm_state_mode", None),
        "save_offload": getattr(scan, "_mamba_save_offload_patched", False),
        "comm_warmup": getattr(parallel_state.initialize_model_parallel, "_isambard_comm_warmup", False),
        "row_telemetry": getattr(GPTSFTPackedParquetDataset.__getitem__, "_isambard_row_telemetry", False),
    }
    assert_golden_json(GOLDENS / "env_gates" / f"{case}.json", record)

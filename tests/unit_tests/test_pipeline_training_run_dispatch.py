"""Unit tests for the model/mode dispatch (pipeline.training.recipes.RECIPE_MAP) and pipeline_training_run.py's CLI.

The script lives at the repo root and is loaded by path (the ``run_module`` fixture), and the dispatch
table is imported from its module, so these tests exercise the real one rather than a re-declaration of it.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import pytest
from scripts.training.config_compose import load_composed_yaml

from pipeline.training import env_overrides, recipes
from tests.unit_tests.repo_scripts import run_training_main
from tests.unit_tests.tiny_models import tiny_olmo3


MODELS = ("nano", "super", "ultra")
OLMO3_MODELS = ("olmo3_7b", "olmo3_32b")
MODES = ("sft", "cpt", "pretrain")
BASELINE_PRETRAIN = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "control_pretraining"
    / "30b_baseline"
    / "nemotron_nano_30b_baseline_pretrain.yaml"
)


class TestRecipeMap:
    def test_covers_every_model_mode_pair(self):
        """Every Nemotron model has all three modes; each OLMo-3 model has full SFT only."""
        nemotron = {(model, mode) for model in MODELS for mode in MODES}
        olmo3 = {(model, "sft") for model in OLMO3_MODELS}
        assert set(recipes.RECIPE_MAP.keys()) == nemotron | olmo3

    @pytest.mark.parametrize("model", MODELS)
    def test_pretrain_entries_reference_the_pretrain_recipes(self, model):
        """Each pretrain entry must call its model's *_pretrain_config, not an SFT recipe.

        Checked via the lambda's referenced names rather than by invoking it: the Super and
        Ultra pretrain recipes construct the model through AutoBridge.from_hf_pretrained,
        which reads the HF config from the Hub/cache — a network boundary a unit test must
        not depend on. The Nano recipe is invoked for real below.
        """
        names = recipes.RECIPE_MAP[(model, "pretrain")].__code__.co_names
        assert f"nemotron_3_{model}_pretrain_config" in names

    @pytest.mark.parametrize("model", MODELS)
    def test_cpt_entries_still_reference_the_sft_recipes(self, model):
        """CPT deliberately reuses the SFT recipes (warm-start hyperparameters + finetune())."""
        names = recipes.RECIPE_MAP[(model, "cpt")].__code__.co_names
        assert f"nemotron_3_{model}_sft_config" in names
        assert f"nemotron_3_{model}_pretrain_config" not in names

    def test_nano_pretrain_recipe_builds_from_scratch_config(self):
        """The Nano entry constructs the real pretrain recipe: NVIDIA's pretraining workload
        (GBS 3072, seq 8192) and no checkpoint to load — the from-scratch semantics the
        pretrain mode exists for."""
        cfg = recipes.RECIPE_MAP[("nano", "pretrain")](None)
        assert cfg.train.global_batch_size == 3072
        assert cfg.model.seq_length == 8192
        assert cfg.dataset.sequence_length == 8192
        assert cfg.checkpoint.pretrained_checkpoint is None


class TestOlmo3Recipes:
    @pytest.mark.parametrize("model", OLMO3_MODELS)
    def test_each_entry_builds_the_olmo_recipe_on_its_local_base_copy(self, model):
        names = recipes.RECIPE_MAP[(model, "sft")].__code__.co_names
        assert "olmo3_sft_config" in names
        assert "OLMO3_BASE_COPIES" in names
        assert set(recipes.OLMO3_BASE_COPIES) == set(OLMO3_MODELS)

    def test_the_recipe_sizes_the_model_from_the_config_and_sets_what_the_family_needs(self, tmp_path):
        from megatron.bridge.recipes.olmo.olmo3 import OLMO3_SFT_SEQ_LENGTH, olmo3_sft_config

        # The recipe reads only the config of a base copy, so a saved tiny OLMo-3 (three sliding-window layers and one
        # full-attention layer) is a real input to it; the two real copies live on the cluster's file system.
        model = tiny_olmo3()
        model.save_pretrained(tmp_path)
        cfg = olmo3_sft_config(str(tmp_path))

        # Sized from the config.
        hf = model.config
        assert (cfg.model.num_layers, cfg.model.hidden_size) == (hf.num_hidden_layers, hf.hidden_size)
        assert cfg.model.window_attn_skip_freq == [1, 1, 1, 0]
        assert cfg.model.window_size == (hf.sliding_window - 1, 0)
        assert cfg.tokenizer.tokenizer_model == str(tmp_path)
        # What the family needs, each with its reason in the recipe's docstring.
        assert cfg.model.tensor_model_parallel_size == 1
        assert cfg.model.apply_rope_fusion is False
        assert cfg.model.calculate_per_token_loss is True
        assert cfg.model.cp_comm_type == "a2a"
        assert cfg.model.seq_length == OLMO3_SFT_SEQ_LENGTH == cfg.dataset.seq_length
        assert cfg.dataset.packed_sequence_specs.packed_sequence_size == OLMO3_SFT_SEQ_LENGTH
        assert cfg.dataset.packed_sequence_specs.pad_seq_to_mult == 8
        assert cfg.checkpoint.dist_ckpt_strictness == "raise_all"

    @pytest.mark.parametrize("mode", ["cpt", "pretrain"])
    def test_a_mode_without_a_recipe_is_refused_naming_the_modes_that_exist(self, mode):
        with pytest.raises(ValueError, match=r"no recipe for model 'olmo3_7b' in mode .*\['sft'\]"):
            recipes.resolve_training_config("olmo3_7b", mode, None, None, [])

    def test_peft_is_refused(self):
        with pytest.raises(ValueError, match="only a full SFT recipe"):
            recipes.resolve_training_config("olmo3_32b", "sft", "lora", None, [])

    def test_the_cli_offers_exactly_the_models_of_the_map(self, run_module, monkeypatch):
        for model in (*MODELS, *OLMO3_MODELS):
            monkeypatch.setattr(sys, "argv", ["pipeline_training_run.py", "--model", model, "--mode", "sft"])
            args, _ = run_module.parse_cli_args()
            assert args.model == model
        monkeypatch.setattr(sys, "argv", ["pipeline_training_run.py", "--model", "olmo3_13b", "--mode", "sft"])
        with pytest.raises(SystemExit):
            run_module.parse_cli_args()

    def test_the_launch_script_allows_exactly_the_models_of_the_map(self):
        script = (Path(__file__).resolve().parents[2] / "pipeline_training_launch.sh").read_text()
        listed = re.search(r"^ALLOWED_MODELS=\(([^)]*)\)$", script, re.MULTILINE)
        assert listed is not None, "pipeline_training_launch.sh no longer defines ALLOWED_MODELS"
        assert set(listed.group(1).split()) == {model for model, _ in recipes.RECIPE_MAP}


class TestMainWiring:
    """Drive the real main() end-to-end up to the training-entry call (``run_training_main``), on the Nano
    recipes (pure dataclass construction — Super/Ultra would fetch the HF config).
    """

    def _run_main(self, run_module, monkeypatch, tmp_path, mode, yaml_text):
        config = tmp_path / "override.yaml"
        config.write_text(yaml_text)
        argv = ["--model", "nano", "--mode", mode, "--config-file", str(config), "--disable-ft"]
        return run_training_main(run_module, monkeypatch, argv)

    _DATA_PATH_YAML = (
        "tokenizer:\n"
        "  tokenizer_model: geodesic-research/nemotron-base-tokenizer\n"
        "dataset:\n"
        "  data_path:\n"
        '    - "1.0"\n'
        "    - /nonexistent/corpus_input_document\n"
    )

    @pytest.mark.parametrize("mode", ("cpt", "pretrain"))
    def test_native_data_modes_without_data_path_raise(self, run_module, monkeypatch, tmp_path, mode):
        """Both .bin/.idx modes must name their corpus — neither may substitute one silently."""
        yaml_text = "tokenizer:\n  tokenizer_model: geodesic-research/nemotron-base-tokenizer\n"
        with pytest.raises(ValueError, match="dataset.data_path"):
            self._run_main(run_module, monkeypatch, tmp_path, mode, yaml_text)

    def test_pretrain_mode_calls_the_pretrain_entry(self, run_module, monkeypatch, tmp_path):
        calls = self._run_main(run_module, monkeypatch, tmp_path, "pretrain", self._DATA_PATH_YAML)
        assert set(calls) == {"pretrain"}
        cfg = calls["pretrain"]["config"]
        assert cfg.dataset.data_path == ["1.0", "/nonexistent/corpus_input_document"]
        assert cfg.checkpoint.pretrained_checkpoint is None

    def test_cpt_mode_still_calls_the_finetune_entry(self, run_module, monkeypatch, tmp_path):
        calls = self._run_main(run_module, monkeypatch, tmp_path, "cpt", self._DATA_PATH_YAML)
        assert set(calls) == {"finetune"}
        assert calls["finetune"]["config"].dataset.data_path == ["1.0", "/nonexistent/corpus_input_document"]

    def test_a_base_config_overlay_trains_as_its_composed_config(self, run_module, monkeypatch, tmp_path):
        """The overlay's own fields win, and everything else comes from its base — including the
        corpus, which the .bin/.idx dataset rebuild re-reads from the merged config rather than
        from the recipe, so it would silently vanish if only the overlay file were read."""
        overlay = (
            f"base_config: {BASELINE_PRETRAIN}\n"
            "train:\n  global_batch_size: 512\n"
            "checkpoint:\n  load: null\n  save: null\n"
        )
        cfg = self._run_main(run_module, monkeypatch, tmp_path, "pretrain", overlay)["pretrain"]["config"]
        base = load_composed_yaml(BASELINE_PRETRAIN)
        assert cfg.train.global_batch_size == 512
        assert cfg.checkpoint.load is None and cfg.checkpoint.save is None
        assert cfg.train.train_iters == base["train"]["train_iters"]
        assert cfg.model.expert_model_parallel_size == base["model"]["expert_model_parallel_size"]
        assert cfg.dataset.data_path == [str(entry) for entry in base["dataset"]["data_path"]]
        assert cfg.dataset.split == base["dataset"]["split"]
        assert cfg.dataset.sequence_length == base["dataset"]["seq_length"]

    def test_env_overrides_are_echoed_at_startup(self, run_module, monkeypatch, tmp_path, caplog):
        monkeypatch.setenv("ISAMBARD_ENV_OVERRIDE_KEYS", "TORCH_NCCL_BLOCKING_WAIT")
        monkeypatch.setenv("TORCH_NCCL_BLOCKING_WAIT", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("RANK", "0")
        with caplog.at_level(logging.INFO, logger=env_overrides.logger.name):
            self._run_main(run_module, monkeypatch, tmp_path, "pretrain", self._DATA_PATH_YAML)
        echoes = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[env-overrides]")]
        assert len(echoes) == 1 and echoes[0].endswith(" TORCH_NCCL_BLOCKING_WAIT=0")

    # A code_identity block of the right shape; whether REPO_DIR holds that code is the launcher's check
    # (tests/unit_tests/test_code_identity.py), so here only its record matters.
    _CODE_IDENTITY = {
        "revision": "a" * 40,
        "src_tree": "b" * 40,
        "launchers": {"pipeline_training_run.py": "c" * 40},
        "ancestor": "d" * 40,
        "history": "/checkouts/geodesic-megatron",
    }

    def _pinning_yaml(self) -> str:
        import yaml

        return self._DATA_PATH_YAML + yaml.safe_dump({"code_identity": self._CODE_IDENTITY})

    def test_a_config_pinning_its_code_is_refused_without_the_launchers_record(
        self, run_module, monkeypatch, tmp_path
    ):
        from scripts.training.code_identity import CodeIdentityError

        monkeypatch.delenv("ISAMBARD_CODE_IDENTITY", raising=False)
        with pytest.raises(CodeIdentityError, match="launch it through pipeline_training_launch.sh"):
            self._run_main(run_module, monkeypatch, tmp_path, "pretrain", self._pinning_yaml())

    def test_a_checked_config_trains_and_hands_its_record_to_the_run_identity(self, run_module, monkeypatch, tmp_path):
        import json

        record = {"expected": self._CODE_IDENTITY, "passed": True, "differences": []}
        monkeypatch.setenv("ISAMBARD_CODE_IDENTITY", json.dumps(record))
        calls = self._run_main(run_module, monkeypatch, tmp_path, "pretrain", self._pinning_yaml())
        (identity,) = [cb for cb in calls["pretrain"]["callbacks"] if type(cb).__name__ == "RunIdentityCallback"]
        assert identity.code_identity == record
        assert calls["pretrain"]["config"].dataset.data_path == ["1.0", "/nonexistent/corpus_input_document"]

    def test_the_block_is_kept_out_of_the_run_config(self, tmp_path):
        """The block names the code, not a setting: it is returned beside the merged config, never applied to it."""
        config = tmp_path / "override.yaml"
        config.write_text(self._pinning_yaml())
        cfg, merged = recipes.resolve_training_config("nano", "pretrain", None, str(config), [])
        assert merged["code_identity"] == self._CODE_IDENTITY
        assert not hasattr(cfg, "code_identity")

    # A launch_width block for 8 ranks at TP = PP = CP = 1; whether the allocation gives that width is the
    # launcher's check (tests/unit_tests/test_launch_width.py), so here only its record and the run's own width matter.
    _LAUNCH_WIDTH = {"nodes": 2, "gpus_per_node": 4, "data_parallel_size": 8}

    def _width_yaml(self) -> str:
        import yaml

        parallelism = {"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1, "context_parallel_size": 1}
        return self._DATA_PATH_YAML + yaml.safe_dump({"model": parallelism, "launch_width": self._LAUNCH_WIDTH})

    def _set_width_record(self, monkeypatch, world_size: int) -> None:
        import json

        expected = {**self._LAUNCH_WIDTH, "nvlink_links_per_gpu": None}
        record = {"expected": expected, "nodes": 2, "nodelist": "n1,n2", "gpus_per_node": 4}
        monkeypatch.setenv("ISAMBARD_LAUNCH_WIDTH", json.dumps(record))
        monkeypatch.setenv("WORLD_SIZE", str(world_size))
        monkeypatch.setenv("RANK", "0")

    def test_a_config_fixing_its_width_is_refused_without_the_launchers_record(
        self, run_module, monkeypatch, tmp_path
    ):
        from scripts.training.launch_width import LaunchWidthError

        monkeypatch.delenv("ISAMBARD_LAUNCH_WIDTH", raising=False)
        with pytest.raises(LaunchWidthError, match="launch it through pipeline_training_launch.sh"):
            self._run_main(run_module, monkeypatch, tmp_path, "pretrain", self._width_yaml())

    def test_a_run_at_its_width_logs_its_world_and_data_parallel_sizes(
        self, run_module, monkeypatch, tmp_path, caplog
    ):
        self._set_width_record(monkeypatch, world_size=8)
        with caplog.at_level(logging.INFO, logger=run_module.logger.name):
            calls = self._run_main(run_module, monkeypatch, tmp_path, "pretrain", self._width_yaml())
        assert set(calls) == {"pretrain"}
        lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[launch-width]")]
        assert lines == ["[launch-width] world_size=8 data_parallel_size=8 nodes=2 gpus_per_node=4 nodelist=n1,n2"]

    def test_a_run_at_another_width_is_refused(self, run_module, monkeypatch, tmp_path):
        from scripts.training.launch_width import LaunchWidthError

        self._set_width_record(monkeypatch, world_size=4)
        with pytest.raises(LaunchWidthError, match="the run has 4 ranks"):
            self._run_main(run_module, monkeypatch, tmp_path, "pretrain", self._width_yaml())

    def test_the_width_block_is_kept_out_of_the_run_config_and_no_override_reaches_it(self, tmp_path):
        """The block states the launch, not a setting: it is returned beside the merged config, and an override of it
        is refused as a key the run's settings do not hold."""
        config = tmp_path / "override.yaml"
        config.write_text(self._width_yaml())
        cfg, merged = recipes.resolve_training_config("nano", "pretrain", None, str(config), [])
        assert merged["launch_width"] == self._LAUNCH_WIDTH
        assert not hasattr(cfg, "launch_width")
        for override in ("launch_width.nodes=4", "+launch_width.nodes=4"):
            with pytest.raises(ValueError, match="Unknown key 'launch_width'"):
                recipes.resolve_training_config("nano", "pretrain", None, str(config), [override])


class TestModeCli:
    def _parse(self, run_module, monkeypatch, argv):
        monkeypatch.setattr(sys, "argv", ["pipeline_training_run.py", *argv])
        return run_module.parse_cli_args()

    def test_pretrain_mode_accepted(self, run_module, monkeypatch):
        args, overrides = self._parse(run_module, monkeypatch, ["--model", "nano", "--mode", "pretrain"])
        assert args.mode == "pretrain"
        assert overrides == []

    def test_unknown_mode_rejected(self, run_module, monkeypatch):
        with pytest.raises(SystemExit):
            self._parse(run_module, monkeypatch, ["--model", "nano", "--mode", "midtrain"])

    def test_enable_ft_is_not_a_flag(self, run_module, monkeypatch):
        """Fault tolerance is on unless --disable-ft is given; an --enable-ft falls through to the overrides."""
        args, overrides = self._parse(run_module, monkeypatch, ["--model", "nano", "--mode", "sft", "--enable-ft"])
        assert overrides == ["--enable-ft"]
        assert not hasattr(args, "enable_ft")

    def test_hydra_overrides_still_fall_through(self, run_module, monkeypatch):
        args, overrides = self._parse(
            run_module, monkeypatch, ["--model", "super", "--mode", "pretrain", "train.train_iters=40"]
        )
        assert args.mode == "pretrain"
        assert overrides == ["train.train_iters=40"]

"""Unit tests for pipeline_training_run.py's model/mode dispatch (RECIPE_MAP + CLI parsing).

The script lives at the repo root and is loaded by path (the same pattern as
test_pipeline_data_prepare.py), so these tests exercise the real dispatch table rather
than a re-declaration of it.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
from scripts.training.config_compose import load_composed_yaml


MODELS = ("nano", "super", "ultra")
MODES = ("sft", "cpt", "pretrain")
BASELINE_PRETRAIN = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "control_pretraining"
    / "30b_baseline"
    / "nemotron_nano_30b_baseline_pretrain.yaml"
)


class TestRecipeMap:
    def test_covers_every_model_mode_pair(self, run_module):
        assert set(run_module.RECIPE_MAP.keys()) == {(model, mode) for model in MODELS for mode in MODES}

    @pytest.mark.parametrize("model", MODELS)
    def test_pretrain_entries_reference_the_pretrain_recipes(self, run_module, model):
        """Each pretrain entry must call its model's *_pretrain_config, not an SFT recipe.

        Checked via the lambda's referenced names rather than by invoking it: the Super and
        Ultra pretrain recipes construct the model through AutoBridge.from_hf_pretrained,
        which reads the HF config from the Hub/cache — a network boundary a unit test must
        not depend on. The Nano recipe is invoked for real below.
        """
        names = run_module.RECIPE_MAP[(model, "pretrain")].__code__.co_names
        assert f"nemotron_3_{model}_pretrain_config" in names

    @pytest.mark.parametrize("model", MODELS)
    def test_cpt_entries_still_reference_the_sft_recipes(self, run_module, model):
        """CPT deliberately reuses the SFT recipes (warm-start hyperparameters + finetune())."""
        names = run_module.RECIPE_MAP[(model, "cpt")].__code__.co_names
        assert f"nemotron_3_{model}_sft_config" in names
        assert f"nemotron_3_{model}_pretrain_config" not in names

    def test_nano_pretrain_recipe_builds_from_scratch_config(self, run_module):
        """The Nano entry constructs the real pretrain recipe: NVIDIA's pretraining workload
        (GBS 3072, seq 8192) and no checkpoint to load — the from-scratch semantics the
        pretrain mode exists for."""
        cfg = run_module.RECIPE_MAP[("nano", "pretrain")](None)
        assert cfg.train.global_batch_size == 3072
        assert cfg.model.seq_length == 8192
        assert cfg.dataset.sequence_length == 8192
        assert cfg.checkpoint.pretrained_checkpoint is None


class TestMainWiring:
    """Drive the real main() end-to-end up to the training-entry call.

    The training entry points (pretrain/finetune) are replaced with recorders: past that
    line the code launches distributed training — a SLURM/GPU/rendezvous boundary a unit
    test cannot cross. Everything before it (recipe construction, YAML merge, dataset
    rewiring, mode dispatch) runs for real, on the Nano recipes (pure dataclass
    construction — Super/Ultra would fetch the HF config).
    """

    def _run_main(self, run_module, monkeypatch, tmp_path, mode, yaml_text):
        config = tmp_path / "override.yaml"
        config.write_text(yaml_text)
        calls = {}
        monkeypatch.setattr(run_module, "pretrain", lambda **kw: calls.setdefault("pretrain", kw))
        monkeypatch.setattr(run_module, "finetune", lambda **kw: calls.setdefault("finetune", kw))
        monkeypatch.syspath_prepend(str(Path(run_module.__file__).parent))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "pipeline_training_run.py",
                "--model",
                "nano",
                "--mode",
                mode,
                "--config-file",
                str(config),
                "--disable-ft",
            ],
        )
        run_module.main()
        return calls

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
        with caplog.at_level(logging.INFO, logger=run_module.logger.name):
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

    def test_the_block_is_kept_out_of_the_run_config(self, run_module, tmp_path):
        """The block names the code, not a setting: it is returned beside the merged config, never applied to it."""
        config = tmp_path / "override.yaml"
        config.write_text(self._pinning_yaml())
        cfg, merged = run_module.resolve_training_config("nano", "pretrain", None, str(config), [])
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

    def test_the_width_block_is_kept_out_of_the_run_config_and_no_override_reaches_it(self, run_module, tmp_path):
        """The block states the launch, not a setting: it is returned beside the merged config, and an override of it
        is refused as a key the run's settings do not hold."""
        config = tmp_path / "override.yaml"
        config.write_text(self._width_yaml())
        cfg, merged = run_module.resolve_training_config("nano", "pretrain", None, str(config), [])
        assert merged["launch_width"] == self._LAUNCH_WIDTH
        assert not hasattr(cfg, "launch_width")
        for override in ("launch_width.nodes=4", "+launch_width.nodes=4"):
            with pytest.raises(ValueError, match="Unknown key 'launch_width'"):
                run_module.resolve_training_config("nano", "pretrain", None, str(config), [override])


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

    def test_hydra_overrides_still_fall_through(self, run_module, monkeypatch):
        args, overrides = self._parse(
            run_module, monkeypatch, ["--model", "super", "--mode", "pretrain", "train.train_iters=40"]
        )
        assert args.mode == "pretrain"
        assert overrides == ["train.train_iters=40"]

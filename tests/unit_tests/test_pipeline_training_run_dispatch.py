"""Unit tests for pipeline_training_run.py's model/mode dispatch (RECIPE_MAP + CLI parsing).

The script lives at the repo root and is loaded by path (the same pattern as
test_pipeline_data_prepare.py), so these tests exercise the real dispatch table rather
than a re-declaration of it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUN_PATH = _REPO_ROOT / "pipeline_training_run.py"

MODELS = ("nano", "super", "ultra")
MODES = ("sft", "cpt", "pretrain")


@pytest.fixture(scope="module")
def run_module():
    spec = importlib.util.spec_from_file_location("pipeline_training_run", _RUN_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_training_run"] = module
    spec.loader.exec_module(module)
    return module


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
        monkeypatch.syspath_prepend(str(_REPO_ROOT))
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

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

        Checked via the entry's wrapped recipe rather than by invoking it: the Super and
        Ultra pretrain recipes construct the model through AutoBridge.from_hf_pretrained,
        which reads the HF config from the Hub/cache — a network boundary a unit test must
        not depend on. The Nano recipe is invoked for real below.
        """
        entry = run_module.RECIPE_MAP[(model, "pretrain")]
        assert entry.__wrapped__.__name__ == f"nemotron_3_{model}_pretrain_config"

    @pytest.mark.parametrize("model", MODELS)
    def test_cpt_entries_still_reference_the_sft_recipes(self, run_module, model):
        """CPT deliberately reuses the SFT recipes (warm-start hyperparameters + finetune())."""
        entry = run_module.RECIPE_MAP[(model, "cpt")]
        assert entry.__wrapped__.__name__ == f"nemotron_3_{model}_sft_config"

    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("mode", ("cpt", "pretrain"))
    def test_peft_rejected_where_the_recipe_carries_no_scheme(self, run_module, model, mode):
        """Only the SFT recipes have a PEFT variant.

        Dropping --peft silently in the other modes turns a requested adapter run into a
        full-parameter finetune that looks healthy until someone reads the parameter counts
        out of a job log, by which point an allocation has been spent. The error names the
        YAML route, which does work in every mode.
        """
        with pytest.raises(ValueError, match=r"peft:"):
            run_module.RECIPE_MAP[(model, mode)]("lora")

    def test_peft_guard_passes_through_when_none_requested(self, run_module):
        """The rejection must fire on a requested adapter only, never on the default.

        Exercised on the guard itself with a stand-in recipe: the guard is what decides,
        and driving it directly avoids invoking a real Super/Ultra recipe (a network read).
        """
        built = []
        guarded = run_module._peft_unsupported(lambda: built.append(1) or "cfg", "cpt")
        assert guarded(None) == "cfg"
        assert built == [1]
        with pytest.raises(ValueError, match=r"peft:"):
            guarded("lora")
        assert built == [1]

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

    @pytest.mark.parametrize("mode", ("cpt", "pretrain"))
    def test_dataset_index_cache_location_is_configurable(self, run_module, monkeypatch, tmp_path, mode):
        """A blend may read a corpus the run cannot write next to.

        GPTDataset caches its document/sample/shuffle indices beside the corpus by
        default, so without this key a run over a read-only or another user's corpus
        fails at dataset build — after the base weights have loaded.
        """
        yaml_text = self._DATA_PATH_YAML + "  path_to_cache: /writable/cache\n"
        calls = self._run_main(run_module, monkeypatch, tmp_path, mode, yaml_text)
        cfg = calls["finetune" if mode == "cpt" else "pretrain"]["config"]
        assert cfg.dataset.path_to_cache == "/writable/cache"

    @pytest.mark.parametrize("mode", ("cpt", "pretrain"))
    def test_dataset_index_cache_defaults_to_unset(self, run_module, monkeypatch, tmp_path, mode):
        """Omitting the key must leave MCore's own default in place, not invent a path."""
        calls = self._run_main(run_module, monkeypatch, tmp_path, mode, self._DATA_PATH_YAML)
        cfg = calls["finetune" if mode == "cpt" else "pretrain"]["config"]
        assert cfg.dataset.path_to_cache is None

    _PEFT_YAML = _DATA_PATH_YAML + (
        "peft:\n"
        "  _target_: megatron.bridge.peft.lora.LoRA\n"
        "  target_modules: [linear_qkv, linear_proj, in_proj, out_proj]\n"
        "  dim: 256\n"
        "  alpha: 256\n"
        "  dropout: 0.0\n"
    )

    @pytest.mark.parametrize("mode", ("cpt", "pretrain"))
    def test_yaml_peft_block_lands_a_live_adapter_in_every_mode(
        self, run_module, monkeypatch, tmp_path, mode
    ):
        """The YAML route is the supported way to get an adapter outside SFT mode.

        `cfg.peft` is applied mode-agnostically by the training setup, and the override
        merge instantiates any mapping carrying `_target_`. This is the mechanism the CPT
        adapter configs depend on, so it is asserted rather than assumed: a regression here
        would produce a silent full-parameter finetune, the same failure the RECIPE_MAP
        guard exists to prevent.
        """
        from megatron.bridge.peft.lora import LoRA

        calls = self._run_main(run_module, monkeypatch, tmp_path, mode, self._PEFT_YAML)
        cfg = calls["finetune" if mode == "cpt" else "pretrain"]["config"]
        assert isinstance(cfg.peft, LoRA)
        assert cfg.peft.dim == 256
        assert cfg.peft.alpha == 256
        assert cfg.peft.target_modules == ["linear_qkv", "linear_proj", "in_proj", "out_proj"]


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

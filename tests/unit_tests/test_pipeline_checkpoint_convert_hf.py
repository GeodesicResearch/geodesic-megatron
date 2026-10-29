"""Unit tests for pipeline_checkpoint_convert_hf.py — the remote-code policy
(_apply_remote_code_policy), the strip-by-default vs --keep-remote-code paths, the
release inference files an export ships (adopt_release_inference_files) and the
fix-ups applied to it (fixup_hf_output), and the
PEFT export's checkpoint detection and merge check.

GPU-free.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from scripts.checkpoint.decoding_defaults import REASONING_GENERATION_DEFAULTS


def _make_modeling_source(src_dir: Path, convert_module) -> Path:
    """Create a directory holding the two NemotronH modeling files."""
    src_dir.mkdir(parents=True, exist_ok=True)
    for fname in convert_module.NEMOTRON_H_MODELING_FILES:
        (src_dir / fname).write_text(f"# fake {fname}\n")
    return src_dir


# ── Default (strip) path ─────────────────────────────────────────────────────


class TestStripPath:
    def test_strips_auto_map_and_stale_modeling_files(self, convert_module, tmp_path):
        hf = tmp_path / "hf"
        hf.mkdir()
        # Stale custom modeling file referenced by auto_map.
        (hf / "configuration_nemotron_h.py").write_text("# stale\n")
        config = {"auto_map": dict(convert_module.NEMOTRON_H_AUTO_MAP), "model_type": "nemotron_h"}

        changed = convert_module._apply_remote_code_policy(hf, config, keep_remote_code=False, remote_code_source=None)

        assert changed is True
        assert "auto_map" not in config
        assert not (hf / "configuration_nemotron_h.py").exists()

    def test_no_auto_map_is_noop(self, convert_module, tmp_path):
        hf = tmp_path / "hf"
        hf.mkdir()
        config = {"model_type": "nemotron_h"}

        changed = convert_module._apply_remote_code_policy(hf, config, keep_remote_code=False, remote_code_source=None)

        assert changed is False
        assert config == {"model_type": "nemotron_h"}


# ── --keep-remote-code path ──────────────────────────────────────────────────


class TestKeepPath:
    def test_copies_from_source_and_sets_auto_map(self, convert_module, tmp_path):
        hf = tmp_path / "hf"
        hf.mkdir()
        src = _make_modeling_source(tmp_path / "src", convert_module)
        config = {"model_type": "nemotron_h"}

        changed = convert_module._apply_remote_code_policy(
            hf, config, keep_remote_code=True, remote_code_source=str(src)
        )

        assert changed is True
        assert config["auto_map"] == convert_module.NEMOTRON_H_AUTO_MAP
        for fname in convert_module.NEMOTRON_H_MODELING_FILES:
            assert (hf / fname).is_file()

    def test_does_not_overwrite_existing_modeling_files(self, convert_module, tmp_path):
        hf = tmp_path / "hf"
        hf.mkdir()
        # Local files already present (e.g. written by save_hf_pretrained) — keep them.
        for fname in convert_module.NEMOTRON_H_MODELING_FILES:
            (hf / fname).write_text("# local-correct\n")
        src = _make_modeling_source(tmp_path / "src", convert_module)
        config = {"model_type": "nemotron_h"}

        changed = convert_module._apply_remote_code_policy(
            hf, config, keep_remote_code=True, remote_code_source=str(src)
        )

        assert changed is True
        assert config["auto_map"] == convert_module.NEMOTRON_H_AUTO_MAP
        for fname in convert_module.NEMOTRON_H_MODELING_FILES:
            assert (hf / fname).read_text() == "# local-correct\n"

    def test_files_present_without_source_sets_auto_map(self, convert_module, tmp_path):
        hf = tmp_path / "hf"
        hf.mkdir()
        for fname in convert_module.NEMOTRON_H_MODELING_FILES:
            (hf / fname).write_text("# present\n")
        config = {"model_type": "nemotron_h"}

        changed = convert_module._apply_remote_code_policy(hf, config, keep_remote_code=True, remote_code_source=None)

        assert changed is True
        assert config["auto_map"] == convert_module.NEMOTRON_H_AUTO_MAP

    def test_idempotent_when_auto_map_already_correct(self, convert_module, tmp_path):
        hf = tmp_path / "hf"
        hf.mkdir()
        for fname in convert_module.NEMOTRON_H_MODELING_FILES:
            (hf / fname).write_text("# present\n")
        config = {"auto_map": dict(convert_module.NEMOTRON_H_AUTO_MAP), "model_type": "nemotron_h"}

        changed = convert_module._apply_remote_code_policy(hf, config, keep_remote_code=True, remote_code_source=None)

        assert changed is False
        assert config["auto_map"] == convert_module.NEMOTRON_H_AUTO_MAP

    def test_raises_when_files_absent_and_no_source(self, convert_module, tmp_path):
        hf = tmp_path / "hf"
        hf.mkdir()
        config = {"model_type": "nemotron_h"}

        with pytest.raises(FileNotFoundError, match="keep-remote-code"):
            convert_module._apply_remote_code_policy(hf, config, keep_remote_code=True, remote_code_source=None)

    def test_raises_when_source_lacks_files(self, convert_module, tmp_path):
        hf = tmp_path / "hf"
        hf.mkdir()
        empty_src = tmp_path / "empty_src"
        empty_src.mkdir()
        config = {"model_type": "nemotron_h"}

        with pytest.raises(FileNotFoundError, match="keep-remote-code"):
            convert_module._apply_remote_code_policy(
                hf, config, keep_remote_code=True, remote_code_source=str(empty_src)
            )


# ── Embedding-row discovery (drives vocab_size reconciliation) ────────────────


def _write_safetensors(path: Path, tensors: dict[str, tuple[int, int]]) -> None:
    """Write a minimal safetensors file whose header declares the given shapes."""
    import json
    import struct

    header: dict[str, object] = {}
    offset = 0
    for name, shape in tensors.items():
        nbytes = shape[0] * shape[1] * 2  # BF16
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    blob = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\0" * offset)


class TestReadEmbeddingRowCount:
    def test_reads_rows_from_sharded_export(self, convert_module, tmp_path):
        import json

        _write_safetensors(tmp_path / "model-00001-of-00002.safetensors", {"backbone.embeddings.weight": (131584, 8)})
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"backbone.embeddings.weight": "model-00001-of-00002.safetensors"}})
        )
        assert convert_module.read_embedding_row_count(tmp_path) == 131584

    def test_reads_rows_from_single_file_export(self, convert_module, tmp_path):
        # A single-file export has no index.json; the row count must still be found,
        # otherwise a vocab-extended checkpoint keeps the donor's vocab_size and
        # trips vLLM's embedding-shape assert at load.
        _write_safetensors(tmp_path / "model.safetensors", {"backbone.embeddings.weight": (131584, 8)})
        assert convert_module.read_embedding_row_count(tmp_path) == 131584

    def test_returns_none_when_no_weights_present(self, convert_module, tmp_path):
        assert convert_module.read_embedding_row_count(tmp_path) is None

    def test_returns_none_when_no_embedding_tensor(self, convert_module, tmp_path):
        _write_safetensors(tmp_path / "model.safetensors", {"lm_head.weight": (4, 8)})
        assert convert_module.read_embedding_row_count(tmp_path) is None


class TestIsPeftCheckpoint:
    """An adapter-only checkpoint is recognised by the peft section of its run_config.yaml."""

    def test_peft_section_marks_a_peft_checkpoint(self, tmp_path, convert_module):
        (tmp_path / "run_config.yaml").write_text(
            "peft:\n  _target_: megatron.bridge.peft.canonical_lora.CanonicalLoRA\n"
        )
        assert convert_module.is_peft_checkpoint(tmp_path)

    @pytest.mark.parametrize("text", ["peft: null\nmodel: {}\n", "model: {}\n", ""])
    def test_no_peft_section_is_a_full_checkpoint(self, tmp_path, convert_module, text):
        (tmp_path / "run_config.yaml").write_text(text)
        assert not convert_module.is_peft_checkpoint(tmp_path)

    def test_missing_run_config_is_a_full_checkpoint(self, tmp_path, convert_module):
        assert not convert_module.is_peft_checkpoint(tmp_path)


class TestMergeCheck:
    """The PEFT export's merge check: its logit comparisons and its verdict."""

    def test_identical_logits_agree_exactly(self, convert_module):
        import torch

        logits = torch.randn(2, 5, 11)
        check = convert_module.logit_agreement(logits, logits.clone())
        assert check == {"max_abs_diff": 0.0, "mean_abs_diff": 0.0, "argmax_agreement": 1.0}

    def test_differences_and_flipped_argmax_are_reported(self, convert_module):
        import torch

        reference = torch.zeros(1, 2, 3)
        reference[0, :, 0] = 1.0
        candidate = reference.clone()
        candidate[0, 1, 2] = 2.0  # the second position now argmaxes to token 2
        check = convert_module.logit_agreement(reference, candidate)
        assert check["max_abs_diff"] == 2.0
        assert check["mean_abs_diff"] == pytest.approx(2.0 / 6)
        assert check["argmax_agreement"] == 0.5

    def test_retained_measures_the_share_of_the_adapter_effect_kept(self, convert_module):
        import torch

        base = torch.randn(2, 4, 8)
        adapted = base + 1.0
        check = convert_module.check_merge(adapted, base, base + 0.95)
        assert check["failure"] is None
        assert check["retained"] == pytest.approx(0.95)
        assert check["merged_vs_base"]["mean_abs_diff"] == pytest.approx(0.95)
        assert check["base"]["mean_abs_diff"] == pytest.approx(1.0)

    def test_a_merge_that_keeps_too_little_of_the_effect_fails(self, convert_module):
        """Closer to the adapted model than the base is, yet a quarter of the adapters' effect is lost."""
        import torch

        base = torch.randn(2, 4, 8)
        check = convert_module.check_merge(base + 1.0, base, base + 0.75)
        assert check["failure"].startswith("the merged model retains 0.750, less than 0.9,")

    def test_adapters_that_change_nothing_fail(self, convert_module):
        """An empty adapter load leaves adapted == base == merged, which must not read as a perfect merge."""
        import torch

        adapted = torch.randn(2, 4, 8)
        check = convert_module.check_merge(adapted, adapted.clone(), adapted.clone())
        assert check["failure"] == "the adapters change no logit, so there is nothing to merge"

    def test_a_merge_as_far_as_the_base_fails(self, convert_module):
        import torch

        adapted = torch.randn(2, 4, 8)
        check = convert_module.check_merge(adapted, adapted + 1.0, adapted - 1.0)
        assert check["failure"] == "the merged model is as far from the adapted one as the base is"


def test_a_peft_checkpoint_outside_a_parallel_torchrun_is_refused(convert_module, tmp_path, monkeypatch):
    """An adapter-only checkpoint loads with its base on GPUs; the single-process CPU path would not load it."""
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("7")
    (tmp_path / "iter_0000007").mkdir()
    (tmp_path / "iter_0000007" / "run_config.yaml").write_text(
        "peft:\n  _target_: megatron.bridge.peft.canonical_lora.CanonicalLoRA\n"
    )
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(
        sys, "argv", ["convert", f"--megatron-path={tmp_path}", "--hf-model=some/release", "--no-reasoning"]
    )
    with pytest.raises(SystemExit, match="adapter-only PEFT checkpoint.*with a parallel size above 1"):
        convert_module.main()


# A release's inference template and a training variant of it: the same render, plus {% generation %} blocks
# around the assistant reply (and its <|im_end|>, but not the newline) for the loss mask.
_RELEASE_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
_TRAINING_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n"
    "{% if m.role == 'assistant' %}{% generation %}{{ m.content }}<|im_end|>{% endgeneration %}"
    "{% else %}{{ m.content }}<|im_end|>{% endif %}{{ '\n' }}{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def _release(tmp_path: Path, template: str | None = _RELEASE_TEMPLATE) -> Path:
    release = tmp_path / "release"
    release.mkdir()
    if template is not None:
        (release / "chat_template.jinja").write_text(template)
    (release / "tokenizer.json").write_text('{"release": "tokenizer"}')
    (release / "tokenizer_config.json").write_text('{"tokenizer_class": "PreTrainedTokenizerFast"}')
    (release / "special_tokens_map.json").write_text('{"eos_token": "<|im_end|>"}')
    (release / "generation_config.json").write_text('{"eos_token_id": [2, 11]}')
    (release / "config.json").write_text('{"eos_token_id": 2, "bos_token_id": 1, "hidden_size": 8}')
    return release


def _export(tmp_path: Path) -> Path:
    """An export as the training tokenizer leaves it: its template, no special_tokens_map or generation_config."""
    export = tmp_path / "hf"
    export.mkdir()
    (export / "chat_template.jinja").write_text(_TRAINING_TEMPLATE)
    (export / "tokenizer.json").write_text('{"release": "tokenizer"}')
    (export / "tokenizer_config.json").write_text('{"chat_template": "training"}')
    (export / "config.json").write_text('{"eos_token_id": [2, 11], "bos_token_id": 1, "hidden_size": 16}')
    return export


class TestReleaseInferenceFiles:
    def test_generation_markers_render_alike(self, convert_module):
        assert convert_module.templates_render_alike(_TRAINING_TEMPLATE, _RELEASE_TEMPLATE)

    def test_a_different_format_does_not_render_alike(self, convert_module):
        other = _RELEASE_TEMPLATE.replace(
            "<|im_start|>assistant\n{% endif %}", "<|im_start|>assistant\n<think>{% endif %}"
        )
        assert not convert_module.templates_render_alike(other, _RELEASE_TEMPLATE)

    def test_a_template_that_refuses_every_probe_is_refused(self, convert_module):
        """Two templates that refuse every probe would otherwise render alike."""
        with pytest.raises(ValueError, match="refuses all"):
            convert_module.templates_render_alike("{{ messages | nosuchfilter }}", _RELEASE_TEMPLATE)

    def test_a_template_that_refuses_some_probes_still_compares(self, convert_module):
        """A refusal renders as its error: a template refusing tools renders alike to itself, unlike one without."""
        refuses_tools = "{% if tools %}{{ raise_exception('no tools') }}{% endif %}" + _RELEASE_TEMPLATE
        assert convert_module.templates_render_alike(refuses_tools, refuses_tools)
        assert not convert_module.templates_render_alike(refuses_tools, _RELEASE_TEMPLATE)

    def test_adopts_the_release_files_verbatim(self, convert_module, tmp_path):
        release, export = _release(tmp_path), _export(tmp_path)
        assert convert_module.adopt_release_inference_files(export, str(release), _TRAINING_TEMPLATE)
        for name in convert_module.RELEASE_INFERENCE_FILES:
            assert (export / name).read_bytes() == (release / name).read_bytes(), name
        # config.json takes the release's token ids and keeps the export's model fields.
        config = json.loads((export / "config.json").read_text())
        assert config == {"eos_token_id": 2, "bos_token_id": 1, "hidden_size": 16}

    def test_keeps_the_training_files_for_a_format_of_its_own(self, convert_module, tmp_path):
        release, export = _release(tmp_path), _export(tmp_path)
        own = _TRAINING_TEMPLATE.replace(
            "<|im_start|>assistant\n{% endif %}", "<|im_start|>assistant\n<think>{% endif %}"
        )
        assert not convert_module.adopt_release_inference_files(export, str(release), own)
        assert (export / "chat_template.jinja").read_text() == _TRAINING_TEMPLATE
        assert not (export / "generation_config.json").exists()

    def test_keeps_the_training_files_for_a_release_without_a_template(self, convert_module, tmp_path):
        release, export = _release(tmp_path, template=None), _export(tmp_path)
        assert not convert_module.adopt_release_inference_files(export, str(release), _TRAINING_TEMPLATE)
        assert (export / "chat_template.jinja").read_text() == _TRAINING_TEMPLATE

    def test_refuses_a_release_with_a_dangling_file(self, convert_module, tmp_path):
        release, export = _release(tmp_path), _export(tmp_path)
        (release / "generation_config.json").unlink()
        (release / "generation_config.json").symlink_to(tmp_path / "missing_blob")
        with pytest.raises(FileNotFoundError, match="generation_config.json is a dangling link"):
            convert_module.adopt_release_inference_files(export, str(release), _TRAINING_TEMPLATE)

    def test_downloads_a_hub_release(self, convert_module, tmp_path, monkeypatch):
        """A Hub id downloads the release's inference files and config.json (the network boundary, stubbed)."""
        import huggingface_hub

        release, export = _release(tmp_path), _export(tmp_path)
        calls = []

        def fake_snapshot_download(repo_id, allow_patterns):
            calls.append((repo_id, sorted(allow_patterns)))
            return str(release)

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
        assert convert_module.adopt_release_inference_files(export, "org/release", _TRAINING_TEMPLATE)
        assert calls == [("org/release", sorted([*convert_module.RELEASE_INFERENCE_FILES, "config.json"]))]
        assert (export / "generation_config.json").read_bytes() == (release / "generation_config.json").read_bytes()

    def test_refuses_an_export_without_config_json(self, convert_module, tmp_path):
        release, export = _release(tmp_path), _export(tmp_path)
        (export / "config.json").unlink()
        with pytest.raises(FileNotFoundError, match="config.json does not exist"):
            convert_module.adopt_release_inference_files(export, str(release), _TRAINING_TEMPLATE)


_TRAINING_TOKENIZER_ID = "org/training-tokenizer"


def _training_tokenizer_cache(hf_home: Path, template: str) -> Path:
    """A training tokenizer's snapshot in an HF cache under ``hf_home``."""
    snapshot = hf_home / "hub" / f"models--{_TRAINING_TOKENIZER_ID.replace('/', '--')}" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "chat_template.jinja").write_text(template)
    (snapshot / "tokenizer.json").write_text('{"training": "tokenizer"}')
    (snapshot / "tokenizer_config.json").write_text('{"eos_token": "<|im_end|>"}')
    return snapshot


class TestFixupHfOutput:
    """fixup_hf_output on an export dir, with the training tokenizer in a tmp HF cache and a local release dir."""

    @pytest.fixture
    def training_tokenizer(self, tmp_path, monkeypatch):
        import huggingface_hub

        def install(template: str) -> Path:
            snapshot = _training_tokenizer_cache(tmp_path / "hf_home", template)
            monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))

            # The network boundary: the training tokenizer resolves to its cached snapshot, and nothing else downloads.
            def fake_snapshot_download(repo_id, **kwargs):
                assert repo_id == _TRAINING_TOKENIZER_ID, repo_id
                return str(snapshot)

            monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
            return snapshot

        return install

    def _fixup(self, convert_module, export: Path, release: Path) -> None:
        convert_module.fixup_hf_output(
            export, str(release), reasoning=False, training_tokenizer_id=_TRAINING_TOKENIZER_ID
        )

    def test_an_adopted_slow_tokenizer_class_is_pinned_fast(self, convert_module, tmp_path, training_tokenizer):
        """Nano-9B-v2's release names PreTrainedTokenizer: the adopted tokenizer_config still loads fast."""
        training_tokenizer(_TRAINING_TEMPLATE)
        release, export = _release(tmp_path), _export(tmp_path)
        (release / "tokenizer_config.json").write_text('{"tokenizer_class": "PreTrainedTokenizer", "backend": "x"}')
        self._fixup(convert_module, export, release)
        assert (export / "tokenizer.json").read_bytes() == (release / "tokenizer.json").read_bytes()
        assert json.loads((export / "tokenizer_config.json").read_text()) == {
            "tokenizer_class": "PreTrainedTokenizerFast"
        }

    def test_eos_goes_into_the_adopted_generation_config(self, convert_module, tmp_path, training_tokenizer):
        """With a generation_config.json, <|im_end|> joins its eos ids and config.json keeps the release's."""
        training_tokenizer(_TRAINING_TEMPLATE)
        release, export = _release(tmp_path), _export(tmp_path)
        (release / "generation_config.json").write_text('{"eos_token_id": 2}')
        self._fixup(convert_module, export, release)
        assert json.loads((export / "generation_config.json").read_text())["eos_token_id"] == [2, 11]
        assert json.loads((export / "config.json").read_text())["eos_token_id"] == 2

    def test_eos_goes_into_config_json_without_a_generation_config(self, convert_module, tmp_path, training_tokenizer):
        """A training format of its own keeps the training files, so config.json carries <|im_end|>."""
        own = _TRAINING_TEMPLATE.replace(
            "<|im_start|>assistant\n{% endif %}", "<|im_start|>assistant\n<think>{% endif %}"
        )
        training_tokenizer(own)
        release, export = _release(tmp_path), _export(tmp_path)
        (export / "config.json").write_text('{"eos_token_id": 2, "hidden_size": 16}')
        self._fixup(convert_module, export, release)
        assert not (export / "generation_config.json").exists()
        assert (export / "tokenizer.json").read_text() == '{"training": "tokenizer"}'
        assert json.loads((export / "config.json").read_text())["eos_token_id"] == [2, 11]


_HF_HUB = Path(os.environ.get("HF_HOME", "/projects/a5k/public/hf")) / "hub"
_SUPER_TEMPLATE = (
    _HF_HUB
    / "models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/snapshots/d51eab0d1f979ebc26b546e634a04f450d99158e"
    / "chat_template.jinja"
)
_THINK_TEMPLATES = sorted(
    _HF_HUB.glob("models--geodesic-research--nemotron-think-tokenizer/snapshots/*/chat_template.jinja")
)


@pytest.mark.skipif(not (_SUPER_TEMPLATE.exists() and _THINK_TEMPLATES), reason="needs the shared HF cache")
def test_think_tokenizer_renders_as_the_super_release(convert_module):
    """The SFT think tokenizer is Super's inference template plus training markup, so Super exports ship Super's."""
    think = _THINK_TEMPLATES[-1].read_text()
    assert "{% generation %}" in think
    assert convert_module.templates_render_alike(think, _SUPER_TEMPLATE.read_text())


# ── Decoding defaults ────────────────────────────────────────────────────────

# An export's generation_config.json as the bridge copies the instruct release's: sampling at temperature 1.0, with
# a top_k as well so that every sampling field is present.
UPSTREAM_GENERATION_CONFIG = {
    "_from_model_config": True,
    "bos_token_id": 1,
    "do_sample": True,
    "eos_token_id": [2, 11],
    "pad_token_id": 0,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 1.0,
    "transformers_version": "5.3.0",
}
NON_SAMPLING_FIELDS = {
    "_from_model_config": True,
    "bos_token_id": 1,
    "eos_token_id": [2, 11],
    "pad_token_id": 0,
    "transformers_version": "5.3.0",
}


def _written(hf_path: Path, config: dict) -> Path:
    import json

    hf_path.mkdir(parents=True, exist_ok=True)
    path = hf_path / "generation_config.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path


class TestGenerationDefaults:
    def test_a_non_reasoning_export_decodes_greedily(self, convert_module, tmp_path):
        import json

        path = _written(tmp_path, UPSTREAM_GENERATION_CONFIG)

        convert_module.apply_generation_defaults(tmp_path, reasoning=False)

        written = json.loads(path.read_text())
        assert written == NON_SAMPLING_FIELDS
        assert path.read_text() == json.dumps(NON_SAMPLING_FIELDS, indent=2, sort_keys=True) + "\n"

    def test_a_reasoning_export_samples_at_temperature_0_6_and_top_p_0_95(self, convert_module, tmp_path):
        import json

        path = _written(tmp_path, UPSTREAM_GENERATION_CONFIG)

        convert_module.apply_generation_defaults(tmp_path, reasoning=True)

        # Upstream's top_k is dropped with its other sampling fields, so the export samples as stated and no more.
        assert json.loads(path.read_text()) == {**NON_SAMPLING_FIELDS, **REASONING_GENERATION_DEFAULTS}
        assert REASONING_GENERATION_DEFAULTS == {"do_sample": True, "temperature": 0.6, "top_p": 0.95}

    def test_a_reasoning_export_without_sampling_fields_gains_them(self, convert_module, tmp_path):
        import json

        path = _written(tmp_path, {"bos_token_id": 1, "eos_token_id": [2, 11]})

        convert_module.apply_generation_defaults(tmp_path, reasoning=True)

        assert json.loads(path.read_text()) == {
            "bos_token_id": 1,
            "eos_token_id": [2, 11],
            **REASONING_GENERATION_DEFAULTS,
        }

    @pytest.mark.parametrize("reasoning", [False, True])
    def test_applying_twice_changes_nothing_further(self, convert_module, tmp_path, reasoning):
        path = _written(tmp_path, UPSTREAM_GENERATION_CONFIG)
        convert_module.apply_generation_defaults(tmp_path, reasoning=reasoning)
        once = path.read_bytes()

        convert_module.apply_generation_defaults(tmp_path, reasoning=reasoning)

        assert path.read_bytes() == once

    def test_a_non_reasoning_export_without_a_generation_config_is_left_without_one(self, convert_module, tmp_path):
        convert_module.apply_generation_defaults(tmp_path, reasoning=False)

        assert not (tmp_path / "generation_config.json").exists()

    def test_a_reasoning_export_without_a_generation_config_is_refused(self, convert_module, tmp_path):
        # Skipping it would ship a think model that decodes greedily.
        with pytest.raises(FileNotFoundError, match="generation_config.json is missing"):
            convert_module.apply_generation_defaults(tmp_path, reasoning=True)

    @pytest.mark.parametrize("reasoning", [False, True])
    def test_fixup_hf_output_sets_them_for_the_export_it_fixes(self, convert_module, tmp_path, reasoning):
        import json

        # The tokenizer files and config.json are absent, so only the steps that need no download run; this pins that
        # the decoding defaults are among them whatever else is missing.
        path = _written(tmp_path, UPSTREAM_GENERATION_CONFIG)

        convert_module.fixup_hf_output(tmp_path, "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", reasoning=reasoning)

        written = json.loads(path.read_text())
        expected = {**NON_SAMPLING_FIELDS, **(REASONING_GENERATION_DEFAULTS if reasoning else {})}
        assert written == expected

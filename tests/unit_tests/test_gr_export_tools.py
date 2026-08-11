"""Unit tests for the gradient-routing posture export tools on a synthetic HF export.

Covers `scripts/gradient_routing/bake_postures.py` end to end, plus the CPU half of
`scripts/gradient_routing/verify_posture_equivalence.py` (checks (a) and (c) and the
`posture_verification.json` report; check (b) needs a GPU and a loadable model).

The bake turns ONE raw gradient-routing HF export into one eval posture per POSTURE the
config declares — a name plus the `gr_aux` module indices that posture enables — and its
correctness is entirely in file-level detail a reader cannot eyeball: which tensor is
concatenated along which axis in which order, which config scalar moves with it, which
shard is rewritten versus symlinked, and what the rebuilt index says the checkpoint weighs.
So the fixtures write real (tiny) sharded checkpoints — two MoE layers with aux weights, one
shard with none, a transformers-5.x tokenizer_config, a chat template — and the tests run
the real `main()` over them and read the results back off disk.

The merge identity itself is asserted directly: a posture's shared expert must be the
width-concatenation of the raw shared expert with the enabled aux MLPs in ascending index
order (dim 0 for up_proj, dim 1 for down_proj), which is what makes the posture
mathematically the trained model with exactly those modules active. Every posture must be
the same checkpoint minus EVERY aux key, whether the module was merged in or dropped.

Two fixtures: `baked` is the conventional one-module pair (`forget_off: []`,
`forget_on: [0]`), `baked_multi` is a two-module source baked into all four subsets, which
is what pins index ORDER and the per-subset width arithmetic.

CPU-only and sub-second: every tensor here is a handful of floats.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import safetensors.torch
import torch
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BAKE_PATH = _REPO_ROOT / "scripts" / "gradient_routing" / "bake_postures.py"
_VERIFY_PATH = _REPO_ROOT / "scripts" / "gradient_routing" / "verify_posture_equivalence.py"

# Toy NemotronH-shaped dimensions. AUX_WIDTH != SHARED_WIDTH and both differ from HIDDEN,
# so a merge along the wrong axis cannot accidentally produce the right shape.
HIDDEN = 8
SHARED_WIDTH = 6
AUX_WIDTH = 3
VOCAB = 16
AUX_LAYERS = (1, 3)
# Written into the source config and expected to be overwritten in EVERY posture.
CONFIG_OVERRIDES = {"max_position_embeddings": 8192}
SOURCE_MAX_POS = 4096

# The conventional posture pair for a one-module checkpoint.
POSTURES = {"forget_off": [], "forget_on": [0]}

# The two-module fixture. The widths differ from each other as well as from SHARED_WIDTH and
# HIDDEN, so every subset lands on a distinct width (9, 10, 13) and swapping the two merged
# blocks changes the tensor — both mistakes are therefore detectable.
MULTI_AUX_WIDTHS = {0: 3, 1: 4}
MULTI_POSTURES = {"all_off": [], "m0": [0], "m1": [1], "m01": [0, 1]}

SHARD_AUX = {1: "model-00001-of-00003.safetensors", 3: "model-00002-of-00003.safetensors"}
SHARD_PLAIN = "model-00003-of-00003.safetensors"


from tests.unit_tests.gr_test_utils import load_script as _load_script  # noqa: E402


@pytest.fixture(scope="module")
def bake_module():
    return _load_script("bake_postures", _BAKE_PATH)


@pytest.fixture(scope="module")
def verify_module():
    return _load_script("verify_posture_equivalence", _VERIFY_PATH)


def _aux_up(layer: int, module_index: int) -> str:
    return f"backbone.layers.{layer}.mixer.gr_aux.{module_index}.up_proj.weight"


def _aux_down(layer: int, module_index: int) -> str:
    return f"backbone.layers.{layer}.mixer.gr_aux.{module_index}.down_proj.weight"


def _one_module(width: int = AUX_WIDTH) -> dict[int, dict[int, int]]:
    """The single-aux-module layout: module 0 only, on every aux layer."""
    return {layer: {0: width} for layer in AUX_LAYERS}


def _write_source(root: Path, aux_widths: dict[int, dict[int, int]] | None) -> Path:
    """Write a tiny sharded HF export.

    `aux_widths` maps layer -> {module index: aux ffn width}; a layer left out of it gets a
    shared expert but no aux weights, and `None` writes an export with no aux weights at all.
    """
    root.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(20260806)

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    shards: dict[str, dict[str, torch.Tensor]] = {SHARD_PLAIN: {}}
    # A shard the bake must not touch at all: no aux, no shared expert.
    shards[SHARD_PLAIN]["backbone.embeddings.weight"] = rand(VOCAB, HIDDEN)
    shards[SHARD_PLAIN]["backbone.norm_f.weight"] = rand(HIDDEN)
    shards[SHARD_PLAIN]["backbone.layers.0.mixer.in_proj.weight"] = rand(HIDDEN, HIDDEN)

    for layer in AUX_LAYERS:
        prefix = f"backbone.layers.{layer}"
        shard = shards.setdefault(SHARD_AUX[layer], {})
        shard[prefix + ".mixer.shared_experts.up_proj.weight"] = rand(SHARED_WIDTH, HIDDEN)
        shard[prefix + ".mixer.shared_experts.down_proj.weight"] = rand(HIDDEN, SHARED_WIDTH)
        shard[prefix + ".mixer.router.weight"] = rand(4, HIDDEN)
        for module_index, width in sorted(({} if aux_widths is None else aux_widths.get(layer, {})).items()):
            shard[_aux_up(layer, module_index)] = rand(width, HIDDEN)
            shard[_aux_down(layer, module_index)] = rand(HIDDEN, width)

    weight_map: dict[str, str] = {}
    total_size = 0
    for fname, tensors in shards.items():
        safetensors.torch.save_file(tensors, root / fname)
        for key, tensor in tensors.items():
            weight_map[key] = fname
            total_size += tensor.numel() * tensor.element_size()
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2)
    )

    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["NemotronHForCausalLM"],
                "hidden_size": HIDDEN,
                "vocab_size": VOCAB,
                "moe_shared_expert_intermediate_size": SHARED_WIDTH,
                "n_shared_experts": 1,
                # NemotronH's elementwise squared-relu. The bake asserts the activation is
                # not gated, and refuses a config that declares neither activation key —
                # so the fixture must carry one for any other refusal to be reachable.
                "mlp_hidden_act": "relu2",
                "max_position_embeddings": SOURCE_MAX_POS,
                "eos_token_id": 2,
            },
            indent=2,
        )
    )
    # transformers 5.x tokenizer surface: the class name and the two fields the bake must
    # normalise away for transformers-4.x consumers.
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "TokenizersBackend",
                "backend": "tokenizers",
                "is_local": True,
                "eos_token": "</s>",
                "chat_template": "{{ messages }}",
            },
            indent=2,
        )
    )
    (root / "chat_template.jinja").write_text("{{ messages }}\n")
    # The bake refuses a source without the runtime tokenizer (the converter does not
    # emit it); content is irrelevant to the surgery, presence is what is checked.
    (root / "tokenizer.json").write_text(json.dumps({"version": "1.0", "model": {"type": "BPE"}}))
    (root / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}, indent=2))
    return root


def _run_bake(
    bake_module,
    monkeypatch,
    src: Path,
    out: Path,
    *,
    postures: dict[str, list[int]] | None = None,
    strip_chat_template: bool = True,
    config_overrides: dict | None = None,
) -> int:
    cfg_path = src.parent / f"bake_{out.name}.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "source_dir": str(src),
                "output_root": str(out),
                "postures": dict(POSTURES if postures is None else postures),
                "strip_chat_template": strip_chat_template,
                "expected_layers": len(AUX_LAYERS),
                "config_overrides": dict(CONFIG_OVERRIDES if config_overrides is None else config_overrides),
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["bake_postures.py", "--config", str(cfg_path)])
    return bake_module.main()


def _load_all(root: Path) -> dict[str, torch.Tensor]:
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    out: dict[str, torch.Tensor] = {}
    for fname in sorted(set(weight_map.values())):
        out.update(safetensors.torch.load_file(root / fname))
    return out


def _expected_up(raw: dict[str, torch.Tensor], layer: int, enabled: list[int]) -> torch.Tensor:
    prefix = f"backbone.layers.{layer}"
    return torch.cat(
        [raw[prefix + ".mixer.shared_experts.up_proj.weight"]] + [raw[_aux_up(layer, k)] for k in enabled], dim=0
    )


def _expected_down(raw: dict[str, torch.Tensor], layer: int, enabled: list[int]) -> torch.Tensor:
    prefix = f"backbone.layers.{layer}"
    return torch.cat(
        [raw[prefix + ".mixer.shared_experts.down_proj.weight"]] + [raw[_aux_down(layer, k)] for k in enabled], dim=1
    )


@pytest.fixture(scope="module")
def baked(bake_module, tmp_path_factory):
    """One bake of a well-formed one-module source; the assertions read its output dirs."""
    base = tmp_path_factory.mktemp("gr_export")
    src = _write_source(base / "raw", _one_module())
    out = base / "postures"
    with pytest.MonkeyPatch.context() as mp:
        rc = _run_bake(bake_module, mp, src, out)
    assert rc == 0
    return {
        "src": src,
        "raw_tensors": _load_all(src),
        "dirs": {name: out / name for name in POSTURES},
        "on": out / "forget_on",
        "off": out / "forget_off",
    }


@pytest.fixture(scope="module")
def baked_multi(bake_module, tmp_path_factory):
    """A two-module source baked into every subset: all_off, each module alone, and both."""
    base = tmp_path_factory.mktemp("gr_export_multi")
    src = _write_source(base / "raw", {layer: dict(MULTI_AUX_WIDTHS) for layer in AUX_LAYERS})
    out = base / "postures"
    with pytest.MonkeyPatch.context() as mp:
        rc = _run_bake(bake_module, mp, src, out, postures=MULTI_POSTURES)
    assert rc == 0
    return {
        "src": src,
        "raw_tensors": _load_all(src),
        "root": out,
        "dirs": {name: out / name for name in MULTI_POSTURES},
    }


class TestForgetOn:
    def test_shared_expert_is_the_width_concatenation(self, baked):
        raw = baked["raw_tensors"]
        merged = _load_all(baked["on"])
        for layer in AUX_LAYERS:
            prefix = f"backbone.layers.{layer}.mixer"
            up = merged[f"{prefix}.shared_experts.up_proj.weight"]
            down = merged[f"{prefix}.shared_experts.down_proj.weight"]
            assert torch.equal(up, _expected_up(raw, layer, [0]))
            assert torch.equal(down, _expected_down(raw, layer, [0]))
            assert up.shape == (SHARED_WIDTH + AUX_WIDTH, HIDDEN)
            assert down.shape == (HIDDEN, SHARED_WIDTH + AUX_WIDTH)

    def test_config_bumps_the_shared_width_and_applies_overrides(self, baked):
        config = json.loads((baked["on"] / "config.json").read_text())
        source = json.loads((baked["src"] / "config.json").read_text())
        assert config["moe_shared_expert_intermediate_size"] == SHARED_WIDTH + AUX_WIDTH
        assert config["max_position_embeddings"] == CONFIG_OVERRIDES["max_position_embeddings"]
        assert source["max_position_embeddings"] == SOURCE_MAX_POS
        # Nothing else moved.
        assert config == {
            **source,
            "moe_shared_expert_intermediate_size": SHARED_WIDTH + AUX_WIDTH,
            **CONFIG_OVERRIDES,
        }

    def test_aux_keys_are_gone(self, baked):
        assert not [k for k in _load_all(baked["on"]) if "gr_aux" in k]

    def test_untouched_shard_is_symlinked(self, baked):
        assert (baked["on"] / SHARD_PLAIN).is_symlink()
        assert not (baked["on"] / SHARD_AUX[1]).is_symlink()


class TestForgetOff:
    def test_key_set_is_raw_minus_aux(self, baked):
        raw_keys = set(baked["raw_tensors"])
        aux_keys = {k for k in raw_keys if "gr_aux" in k}
        assert len(aux_keys) == 2 * len(AUX_LAYERS)
        assert set(_load_all(baked["off"])) == raw_keys - aux_keys

    def test_tensors_are_bitwise_the_source(self, baked):
        off = _load_all(baked["off"])
        for key, tensor in off.items():
            assert torch.equal(tensor, baked["raw_tensors"][key]), key

    def test_config_is_the_source_modulo_overrides(self, baked):
        source = json.loads((baked["src"] / "config.json").read_text())
        assert json.loads((baked["off"] / "config.json").read_text()) == {**source, **CONFIG_OVERRIDES}


class TestMultiModulePostures:
    """A posture names a SUBSET of the aux modules; each subset is its own checkpoint.

    With two modules of unequal width, the four subsets land on four distinct shared widths,
    and the concatenation order (ascending module index) is observable — an implementation
    that merged the wrong module, merged both when one was asked for, or concatenated them in
    the other order would produce a tensor these assertions reject.
    """

    @pytest.mark.parametrize("posture", sorted(MULTI_POSTURES))
    def test_shared_expert_is_the_concatenation_of_exactly_the_enabled_modules(self, baked_multi, posture):
        enabled = MULTI_POSTURES[posture]
        raw = baked_multi["raw_tensors"]
        merged = _load_all(baked_multi["dirs"][posture])
        expected_width = SHARED_WIDTH + sum(MULTI_AUX_WIDTHS[k] for k in enabled)
        for layer in AUX_LAYERS:
            prefix = f"backbone.layers.{layer}.mixer"
            up = merged[f"{prefix}.shared_experts.up_proj.weight"]
            down = merged[f"{prefix}.shared_experts.down_proj.weight"]
            assert torch.equal(up, _expected_up(raw, layer, enabled))
            assert torch.equal(down, _expected_down(raw, layer, enabled))
            assert up.shape == (expected_width, HIDDEN)
            assert down.shape == (HIDDEN, expected_width)

    def test_the_four_subsets_land_on_four_distinct_widths(self, baked_multi):
        widths = {
            posture: json.loads((d / "config.json").read_text())["moe_shared_expert_intermediate_size"]
            for posture, d in baked_multi["dirs"].items()
        }
        assert widths == {"all_off": 6, "m0": 9, "m1": 10, "m01": 13}
        assert len(set(widths.values())) == len(widths)

    @pytest.mark.parametrize("posture", sorted(MULTI_POSTURES))
    def test_config_width_is_shared_plus_the_sum_of_the_enabled_widths(self, baked_multi, posture):
        enabled = MULTI_POSTURES[posture]
        source = json.loads((baked_multi["src"] / "config.json").read_text())
        config = json.loads((baked_multi["dirs"][posture] / "config.json").read_text())
        assert config == {
            **source,
            "moe_shared_expert_intermediate_size": SHARED_WIDTH + sum(MULTI_AUX_WIDTHS[k] for k in enabled),
            **CONFIG_OVERRIDES,
        }

    @pytest.mark.parametrize("posture", sorted(MULTI_POSTURES))
    def test_every_posture_drops_every_aux_key_enabled_or_not(self, baked_multi, posture):
        """A disabled module must be GONE, not merely inactive: an enabled one leaves through
        the shared expert it was folded into, so no posture may carry a `gr_aux` key."""
        raw_keys = set(baked_multi["raw_tensors"])
        aux_keys = {k for k in raw_keys if "gr_aux" in k}
        assert len(aux_keys) == 2 * len(MULTI_AUX_WIDTHS) * len(AUX_LAYERS)
        assert set(_load_all(baked_multi["dirs"][posture])) == raw_keys - aux_keys

    def test_a_single_module_posture_excludes_the_other_modules_rows(self, baked_multi):
        """m0 must carry module 0's rows and NOT module 1's, at the same offset where a
        both-modules merge would have put them."""
        raw = baked_multi["raw_tensors"]
        m0 = _load_all(baked_multi["dirs"]["m0"])
        for layer in AUX_LAYERS:
            up = m0[f"backbone.layers.{layer}.mixer.shared_experts.up_proj.weight"]
            assert torch.equal(up[SHARED_WIDTH:], raw[_aux_up(layer, 0)])
            assert up.shape[0] == SHARED_WIDTH + MULTI_AUX_WIDTHS[0]

    def test_untouched_shard_is_symlinked_in_every_posture(self, baked_multi):
        for posture, d in baked_multi["dirs"].items():
            assert (d / SHARD_PLAIN).is_symlink(), posture
            assert not (d / SHARD_AUX[1]).is_symlink(), posture

    @pytest.mark.parametrize("posture", sorted(MULTI_POSTURES))
    def test_provenance_records_the_enabled_subset_and_the_per_module_widths(self, baked_multi, posture):
        enabled = MULTI_POSTURES[posture]
        prov = json.loads((baked_multi["dirs"][posture] / "forget_posture.json").read_text())
        assert prov["posture"] == posture
        assert prov["enabled_module_indices"] == enabled
        assert prov["enabled_module_widths"] == {str(k): MULTI_AUX_WIDTHS[k] for k in enabled}
        assert prov["enabled_aux_width_total"] == sum(MULTI_AUX_WIDTHS[k] for k in enabled)
        assert prov["aux_module_indices"] == sorted(MULTI_AUX_WIDTHS)
        assert prov["aux_ffn_widths"] == {str(k): w for k, w in MULTI_AUX_WIDTHS.items()}
        assert prov["aux_width_per_layer"] == {
            str(layer): {str(k): w for k, w in MULTI_AUX_WIDTHS.items()} for layer in AUX_LAYERS
        }
        assert prov["postures_requested"] == MULTI_POSTURES
        assert prov["expected_values"]["moe_shared_expert_intermediate_size"] == (
            SHARED_WIDTH + sum(MULTI_AUX_WIDTHS[k] for k in enabled)
        )

    @pytest.mark.parametrize("posture", sorted(MULTI_POSTURES))
    def test_provenance_reshape_deltas_name_the_new_widths(self, baked_multi, posture):
        enabled = MULTI_POSTURES[posture]
        prov = json.loads((baked_multi["dirs"][posture] / "forget_posture.json").read_text())
        reshaped = prov["key_inventory_delta"]["keys_reshaped"]
        if not enabled:
            assert reshaped == {}
            return
        expected_width = SHARED_WIDTH + sum(MULTI_AUX_WIDTHS[k] for k in enabled)
        for layer in AUX_LAYERS:
            prefix = f"backbone.layers.{layer}.mixer.shared_experts"
            assert reshaped[f"{prefix}.up_proj.weight"] == [[SHARED_WIDTH, HIDDEN], [expected_width, HIDDEN]]
            assert reshaped[f"{prefix}.down_proj.weight"] == [[HIDDEN, SHARED_WIDTH], [HIDDEN, expected_width]]


class TestIndexAndSideFiles:
    @pytest.mark.parametrize("posture", sorted(POSTURES))
    def test_index_total_size_matches_the_written_shards(self, baked, posture):
        root = baked["dirs"][posture]
        index = json.loads((root / "model.safetensors.index.json").read_text())
        recomputed = sum(t.numel() * t.element_size() for t in _load_all(root).values())
        assert index["metadata"]["total_size"] == recomputed
        assert set(index["weight_map"]) == set(_load_all(root))

    @pytest.mark.parametrize("posture", sorted(POSTURES))
    def test_tokenizer_config_is_normalised_for_transformers_4x(self, baked, posture):
        root = baked["dirs"][posture]
        tc = json.loads((root / "tokenizer_config.json").read_text())
        assert tc["tokenizer_class"] == "PreTrainedTokenizerFast"
        assert "backend" not in tc and "is_local" not in tc
        assert tc["eos_token"] == "</s>"

    @pytest.mark.parametrize("posture", sorted(POSTURES))
    def test_chat_template_is_stripped_from_both_surfaces(self, baked, posture):
        root = baked["dirs"][posture]
        assert not (root / "chat_template.jinja").exists()
        assert "chat_template" not in json.loads((root / "tokenizer_config.json").read_text())

    @pytest.mark.parametrize("posture", sorted(POSTURES))
    def test_side_files_are_copied_not_symlinked(self, baked, posture):
        root = baked["dirs"][posture]
        gen_config = root / "generation_config.json"
        assert gen_config.is_file() and not gen_config.is_symlink()
        assert json.loads(gen_config.read_text()) == {"eos_token_id": 2}

    @pytest.mark.parametrize("posture", sorted(POSTURES))
    def test_provenance_records_the_merge(self, baked, posture):
        root = baked["dirs"][posture]
        prov = json.loads((root / "forget_posture.json").read_text())
        assert prov["posture"] == posture
        assert prov["enabled_module_indices"] == POSTURES[posture]
        assert prov["aux_module_indices"] == [0]
        assert prov["aux_ffn_widths"] == {"0": AUX_WIDTH}
        assert prov["aux_width_per_layer"] == {str(layer): {"0": AUX_WIDTH} for layer in AUX_LAYERS}
        assert prov["aux_layers"] == list(AUX_LAYERS)
        assert prov["postures_requested"] == POSTURES
        assert prov["config_overrides"] == CONFIG_OVERRIDES
        assert prov["chat_template_stripped"] is True
        assert sorted(prov["key_inventory_delta"]["keys_removed"]) == sorted(
            k for k in baked["raw_tensors"] if "gr_aux" in k
        )
        reshaped = prov["key_inventory_delta"]["keys_reshaped"]
        assert (reshaped != {}) is (posture == "forget_on")


class TestChatTemplateKept:
    def test_strip_chat_template_false_leaves_the_template(self, bake_module, monkeypatch, tmp_path):
        src = _write_source(tmp_path / "raw", _one_module())
        out = tmp_path / "postures"
        assert _run_bake(bake_module, monkeypatch, src, out, strip_chat_template=False) == 0
        for posture in POSTURES:
            assert (out / posture / "chat_template.jinja").exists()
            assert "chat_template" in json.loads((out / posture / "tokenizer_config.json").read_text())
            # Normalisation is independent of the chat-template flag.
            assert json.loads((out / posture / "tokenizer_config.json").read_text())["tokenizer_class"] == (
                "PreTrainedTokenizerFast"
            )


class TestRefusals:
    def test_source_without_aux_keys_is_refused(self, bake_module, monkeypatch, tmp_path):
        src = _write_source(tmp_path / "raw", None)
        with pytest.raises(bake_module.BakeError, match=r"carries no .*gr_aux"):
            _run_bake(bake_module, monkeypatch, src, tmp_path / "postures")

    def test_non_uniform_aux_width_is_refused(self, bake_module, monkeypatch, tmp_path):
        """Per module index: one width per index, or no single merged width describes it."""
        src = _write_source(tmp_path / "raw", {1: {0: AUX_WIDTH}, 3: {0: AUX_WIDTH + 1}})
        with pytest.raises(bake_module.BakeError, match="aux module 0 width is not uniform"):
            _run_bake(bake_module, monkeypatch, src, tmp_path / "postures")

    def test_module_indices_differing_across_layers_are_refused(self, bake_module, monkeypatch, tmp_path):
        """A posture enables indices for the whole model, so a layer that lacks module 1
        could not honour a posture that names it."""
        src = _write_source(tmp_path / "raw", {1: {0: 3, 1: 4}, 3: {0: 3}})
        with pytest.raises(bake_module.BakeError, match="aux module indices differ across layers"):
            _run_bake(bake_module, monkeypatch, src, tmp_path / "postures")

    def test_a_gap_in_the_module_indices_is_refused(self, bake_module, monkeypatch, tmp_path):
        """The export writes one key pair per ModuleList entry, so 0,2 means keys were lost."""
        src = _write_source(tmp_path / "raw", {layer: {0: 3, 2: 4} for layer in AUX_LAYERS})
        with pytest.raises(bake_module.BakeError, match=r"not 0\.\.1"):
            _run_bake(bake_module, monkeypatch, src, tmp_path / "postures")

    def test_a_posture_naming_a_nonexistent_module_is_refused(self, bake_module, monkeypatch, tmp_path):
        """Silently baking a narrower posture than asked for would be an unlabelled arm."""
        src = _write_source(tmp_path / "raw", _one_module())
        with pytest.raises(bake_module.BakeError, match="absent from"):
            _run_bake(
                bake_module,
                monkeypatch,
                src,
                tmp_path / "postures",
                postures={"forget_off": [], "forget_on": [0], "forget_on_two": [0, 1]},
            )

    def test_already_baked_source_is_refused(self, bake_module, monkeypatch, tmp_path):
        """A posture dir carries forget_posture.json — re-baking it would double the merge."""
        src = _write_source(tmp_path / "raw", _one_module())
        (src / "forget_posture.json").write_text("{}")
        with pytest.raises(bake_module.BakeError, match="it is itself a baked posture dir"):
            _run_bake(bake_module, monkeypatch, src, tmp_path / "postures")

    def test_expected_layers_mismatch_is_refused(self, bake_module, monkeypatch, tmp_path):
        src = _write_source(tmp_path / "raw", {1: {0: AUX_WIDTH}})
        with pytest.raises(bake_module.BakeError, match="expected_layers"):
            _run_bake(bake_module, monkeypatch, src, tmp_path / "postures")

    def test_tokenizer_less_source_is_refused(self, bake_module, monkeypatch, tmp_path):
        """The converter emits only shards + config; a source missing the runtime tokenizer
        would bake postures that fail only later, at model load."""
        src = _write_source(tmp_path / "raw", _one_module())
        (src / "tokenizer.json").unlink()
        with pytest.raises(bake_module.BakeError, match="tokenizer.json missing"):
            _run_bake(bake_module, monkeypatch, src, tmp_path / "postures")


class TestParseBakePostures:
    """The `postures:` mapping is the whole posture space, so its validation is load-bearing."""

    def test_the_conventional_pair_is_accepted(self, bake_module):
        assert bake_module.parse_postures(Path("bake.yaml"), {"forget_off": [], "forget_on": [0]}) == POSTURES

    def test_descending_indices_are_refused(self, bake_module):
        """Ascending order is the concatenation order, so the YAML states the layout."""
        with pytest.raises(bake_module.BakeError, match="strictly ascending"):
            bake_module.parse_postures(Path("bake.yaml"), {"a": [1, 0]})

    def test_repeated_indices_are_refused(self, bake_module):
        with pytest.raises(bake_module.BakeError, match="strictly ascending"):
            bake_module.parse_postures(Path("bake.yaml"), {"a": [0, 0]})

    def test_two_postures_with_the_same_module_set_are_refused(self, bake_module):
        """Each posture writes a full checkpoint, so an alias duplicates tens of GB."""
        with pytest.raises(bake_module.BakeError, match="identical module sets"):
            bake_module.parse_postures(Path("bake.yaml"), {"a": [0], "b": [0]})

    def test_a_posture_name_that_is_not_a_directory_name_is_refused(self, bake_module):
        with pytest.raises(bake_module.BakeError, match="plain directory name"):
            bake_module.parse_postures(Path("bake.yaml"), {"nested/name": [0]})

    def test_an_empty_posture_mapping_is_refused(self, bake_module):
        with pytest.raises(bake_module.BakeError, match="non-empty mapping"):
            bake_module.parse_postures(Path("bake.yaml"), {})

    def test_a_non_integer_index_is_refused(self, bake_module):
        with pytest.raises(bake_module.BakeError, match="non-negative aux module indices"):
            bake_module.parse_postures(Path("bake.yaml"), {"a": ["0"]})


def _verify_postures(dirs: dict[str, Path], enabled_by_name: dict[str, list[int]]) -> dict[str, dict]:
    return {name: {"dir": str(dirs[name]), "enabled": enabled} for name, enabled in enabled_by_name.items()}


def _write_verify_config(path: Path, src: Path, postures: dict[str, dict], **extra) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "raw_dir": str(src),
                "postures": postures,
                "prompts": ["unused when the logit check is skipped"],
                "logit_check_dtype": "float32",
                "max_router_flip_fraction": 0.15,
                "skip_logit_check": True,
                **extra,
            }
        )
    )
    return path


def _run_verify(verify_module, cfg_path: Path) -> int:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "argv", ["verify_posture_equivalence.py", "--config", str(cfg_path)])
        return verify_module.main()


@pytest.fixture(scope="module")
def verified(bake_module, verify_module, tmp_path_factory):
    """Bake with NO config_overrides, then run the CPU half of the verifier over the result.

    With no recorded overrides, check (c)'s contract for the all-off posture is byte-identity
    between its config.json and the raw export's — the strictest form, exercised by this
    clean pair. (The overrides form of the contract is covered by TestVerifyWithOverrides.)
    Check (b) needs a GPU and a loadable model; `skip_logit_check` runs (a) and (c) only.
    """
    base = tmp_path_factory.mktemp("gr_verify")
    src = _write_source(base / "raw", _one_module())
    out = base / "postures"
    with pytest.MonkeyPatch.context() as mp:
        assert _run_bake(bake_module, mp, src, out, config_overrides={}) == 0

    cfg_path = _write_verify_config(
        base / "verify.yaml", src, _verify_postures({name: out / name for name in POSTURES}, POSTURES)
    )
    rc = _run_verify(verify_module, cfg_path)
    return {"rc": rc, "out": out, "src": src, "config_path": cfg_path}


class TestVerifyPostures:
    def test_cpu_checks_pass_on_a_freshly_baked_pair(self, verified):
        assert verified["rc"] == 0

    def test_report_records_thresholds_facts_and_verdicts(self, verified):
        report = json.loads((verified["out"] / "posture_verification.json").read_text())
        assert report["all_passed"] is True
        assert report["verify_config"] == str(verified["config_path"].resolve())
        assert report["thresholds"]["logit_check_dtype"] == "float32"
        assert report["thresholds"]["kl_threshold"] == 1e-4
        assert report["raw_aux_module_indices"] == [0]
        assert report["raw_aux_widths"] == {"0": AUX_WIDTH}
        assert report["postures"] == {
            name: {"dir": str((verified["out"] / name).resolve()), "enabled": enabled}
            for name, enabled in POSTURES.items()
        }
        assert report["skipped_checks"] == ["b_logit_equivalence[forget_on]"]
        assert "b_logit_equivalence[forget_on]" not in report["checks"]
        algebra = report["checks"]["a_layer_algebra[forget_on]"]
        assert algebra["passed"] is True
        assert algebra["facts"]["layers_checked"] == list(AUX_LAYERS)
        assert algebra["facts"]["enabled_module_indices"] == [0]
        assert algebra["facts"]["bitwise_concat_mismatch_layers"] == []
        assert algebra["facts"]["max_rel_diff"] <= report["thresholds"]["layer_rel_tol"]
        # The all-off posture carries no merge to check, so it gets no (a) entry.
        assert "a_layer_algebra[forget_off]" not in report["checks"]
        stock = report["checks"]["c_posture_shape[forget_off]"]
        assert stock["passed"] is True
        assert stock["facts"] == {
            "posture": "forget_off",
            "enabled_module_indices": [],
            "keys_missing": [],
            "keys_extra": [],
            "config_matches_contract": True,
            "config_byte_identical_to_raw": True,
            "gr_config_fields": [],
            "shared_width_raw": SHARED_WIDTH,
            "shared_width_expected": SHARED_WIDTH,
            "shared_width_posture": SHARED_WIDTH,
            "shared_up_proj_rows": [SHARED_WIDTH],
        }

    def test_c_checks_the_merged_posture_at_its_own_width(self, verified):
        report = json.loads((verified["out"] / "posture_verification.json").read_text())
        facts = report["checks"]["c_posture_shape[forget_on]"]["facts"]
        assert report["checks"]["c_posture_shape[forget_on]"]["passed"] is True
        assert facts["shared_width_expected"] == SHARED_WIDTH + AUX_WIDTH
        assert facts["shared_width_posture"] == SHARED_WIDTH + AUX_WIDTH
        assert facts["shared_up_proj_rows"] == [SHARED_WIDTH + AUX_WIDTH]
        # A merged posture's config differs from raw by the width scalar, so byte-identity
        # is not the contract it is held to.
        assert facts["config_byte_identical_to_raw"] is None
        assert facts["keys_missing"] == [] and facts["keys_extra"] == []


class TestVerifyMultiModulePostures:
    def test_all_four_subsets_verify(self, verify_module, baked_multi, tmp_path):
        cfg = _write_verify_config(
            tmp_path / "verify.yaml",
            baked_multi["src"],
            _verify_postures(baked_multi["dirs"], MULTI_POSTURES),
            expect_config_overrides=dict(CONFIG_OVERRIDES),
        )

        assert _run_verify(verify_module, cfg) == 0

        report = json.loads((baked_multi["root"] / "posture_verification.json").read_text())
        assert report["raw_aux_widths"] == {str(k): w for k, w in MULTI_AUX_WIDTHS.items()}
        assert sorted(report["checks"]) == sorted(
            [f"a_layer_algebra[{name}]" for name, enabled in MULTI_POSTURES.items() if enabled]
            + [f"c_posture_shape[{name}]" for name in MULTI_POSTURES]
        )
        for name, enabled in MULTI_POSTURES.items():
            facts = report["checks"][f"c_posture_shape[{name}]"]["facts"]
            expected = SHARED_WIDTH + sum(MULTI_AUX_WIDTHS[k] for k in enabled)
            assert facts["shared_width_expected"] == expected
            assert facts["shared_up_proj_rows"] == [expected]
            if enabled:
                assert report["checks"][f"a_layer_algebra[{name}]"]["facts"]["enabled_module_indices"] == enabled

    def test_a_posture_declared_with_the_wrong_module_set_fails(self, verify_module, baked_multi, tmp_path):
        """The enabled sets are declared in the VERIFY config, so a dir baked from one subset
        and labelled with another must fail — that mislabelling is exactly what would make an
        eval arm claim modules it does not carry."""
        cfg = _write_verify_config(
            tmp_path / "verify_mislabelled.yaml",
            baked_multi["src"],
            {
                "all_off": {"dir": str(baked_multi["dirs"]["all_off"]), "enabled": []},
                # The dir was baked with module 0 alone.
                "m0": {"dir": str(baked_multi["dirs"]["m0"]), "enabled": [0, 1]},
            },
            expect_config_overrides=dict(CONFIG_OVERRIDES),
        )

        assert _run_verify(verify_module, cfg) != 0

        report = json.loads((baked_multi["root"] / "posture_verification.json").read_text())
        assert report["checks"]["a_layer_algebra[m0]"]["passed"] is False
        assert report["checks"]["a_layer_algebra[m0]"]["facts"]["bitwise_concat_mismatch_layers"] == list(AUX_LAYERS)
        assert report["checks"]["c_posture_shape[m0]"]["passed"] is False
        assert report["checks"]["c_posture_shape[m0]"]["facts"]["shared_width_expected"] == (
            SHARED_WIDTH + sum(MULTI_AUX_WIDTHS.values())
        )


class TestVerifyWithOverrides:
    def _config(self, baked, tmp_path, expect_overrides):
        return _write_verify_config(
            tmp_path / "verify.yaml",
            baked["src"],
            _verify_postures(baked["dirs"], POSTURES),
            expect_config_overrides=expect_overrides,
        )

    def test_c_accepts_config_equal_to_raw_plus_declared_overrides(self, verify_module, baked, tmp_path):
        """The bake rewrites the config fields named in its config_overrides; check (c)'s
        contract is then raw + exactly the overrides THIS config declares."""
        rc = _run_verify(verify_module, self._config(baked, tmp_path, dict(CONFIG_OVERRIDES)))

        assert rc == 0
        report = json.loads((baked["off"].parent / "posture_verification.json").read_text())
        assert report["checks"]["c_posture_shape[forget_off]"]["facts"]["config_matches_contract"] is True
        assert report["thresholds"]["expect_config_overrides"] == CONFIG_OVERRIDES

    def test_c_fails_when_the_posture_carries_an_undeclared_override(self, verify_module, baked, tmp_path):
        """The expectation is read from the VERIFY config, never from the posture's own
        provenance sidecar — otherwise a bake that rewrote a field and recorded it would
        verify itself."""
        rc = _run_verify(verify_module, self._config(baked, tmp_path, {}))

        assert rc != 0
        report = json.loads((baked["off"].parent / "posture_verification.json").read_text())
        assert report["checks"]["c_posture_shape[forget_off]"]["facts"]["config_matches_contract"] is False


class TestLoadVerifyConfig:
    def test_logit_check_dtype_is_required(self, verify_module, verified, tmp_path):
        """It is a required field, not a default — bf16 and fp32 answer different questions."""
        cfg = yaml.safe_load(verified["config_path"].read_text())
        del cfg["logit_check_dtype"]
        path = tmp_path / "verify.yaml"
        path.write_text(yaml.safe_dump(cfg))
        with pytest.raises(verify_module.VerifyError, match="missing required field.*logit_check_dtype"):
            verify_module.load_verify_config(path)

    def test_unknown_dtype_is_refused(self, verify_module, verified, tmp_path):
        cfg = yaml.safe_load(verified["config_path"].read_text())
        cfg["logit_check_dtype"] = "float16"
        path = tmp_path / "verify.yaml"
        path.write_text(yaml.safe_dump(cfg))
        with pytest.raises(verify_module.VerifyError, match="logit_check_dtype must be one of"):
            verify_module.load_verify_config(path)

    def test_max_router_flip_fraction_is_required(self, verify_module, verified, tmp_path):
        """MoE routing ties can legitimately flip between the two forwards; every config
        must take a position on how many flipped positions the comparison tolerates."""
        cfg = yaml.safe_load(verified["config_path"].read_text())
        del cfg["max_router_flip_fraction"]
        path = tmp_path / "verify.yaml"
        path.write_text(yaml.safe_dump(cfg))
        with pytest.raises(verify_module.VerifyError, match="missing required field.*max_router_flip_fraction"):
            verify_module.load_verify_config(path)

    def test_postures_are_required(self, verify_module, verified, tmp_path):
        cfg = yaml.safe_load(verified["config_path"].read_text())
        del cfg["postures"]
        path = tmp_path / "verify.yaml"
        path.write_text(yaml.safe_dump(cfg))
        with pytest.raises(verify_module.VerifyError, match="missing required field.*postures"):
            verify_module.load_verify_config(path)

    def test_exactly_one_all_off_posture_is_required(self, verify_module, verified, tmp_path):
        """It is the ablated model check (b) composes onto and check (c) is defined against."""
        cfg = yaml.safe_load(verified["config_path"].read_text())
        cfg["postures"]["forget_on"]["enabled"] = []
        path = tmp_path / "verify.yaml"
        path.write_text(yaml.safe_dump(cfg))
        with pytest.raises(verify_module.VerifyError, match="exactly one posture"):
            verify_module.load_verify_config(path)

    def test_a_posture_without_an_all_off_entry_is_refused(self, verify_module, verified, tmp_path):
        cfg = yaml.safe_load(verified["config_path"].read_text())
        del cfg["postures"]["forget_off"]
        path = tmp_path / "verify.yaml"
        path.write_text(yaml.safe_dump(cfg))
        with pytest.raises(verify_module.VerifyError, match="exactly one posture"):
            verify_module.load_verify_config(path)

    def test_a_posture_spec_missing_enabled_is_refused(self, verify_module, verified, tmp_path):
        cfg = yaml.safe_load(verified["config_path"].read_text())
        del cfg["postures"]["forget_on"]["enabled"]
        path = tmp_path / "verify.yaml"
        path.write_text(yaml.safe_dump(cfg))
        with pytest.raises(verify_module.VerifyError, match=r"exactly the keys"):
            verify_module.load_verify_config(path)

    def test_a_posture_naming_a_module_the_raw_export_lacks_is_refused(self, verify_module, verified, tmp_path):
        cfg = yaml.safe_load(verified["config_path"].read_text())
        cfg["postures"]["forget_on"]["enabled"] = [0, 3]
        path = tmp_path / "verify.yaml"
        path.write_text(yaml.safe_dump(cfg))
        with pytest.raises(verify_module.VerifyError, match="absent from"):
            _run_verify(verify_module, path)


class TestLoadBakeConfig:
    def test_unknown_field_is_refused(self, bake_module, tmp_path):
        cfg = tmp_path / "bake.yaml"
        cfg.write_text(
            yaml.safe_dump(
                {
                    "source_dir": "/a",
                    "output_root": "/b",
                    "postures": dict(POSTURES),
                    "strip_chat_template": True,
                    "typo_field": 1,
                },
            )
        )
        with pytest.raises(bake_module.BakeError, match="unknown field"):
            bake_module.load_bake_config(cfg)

    def test_missing_required_field_is_refused(self, bake_module, tmp_path):
        cfg = tmp_path / "bake.yaml"
        cfg.write_text(yaml.safe_dump({"source_dir": "/a", "output_root": "/b"}))
        with pytest.raises(bake_module.BakeError, match="missing required field"):
            bake_module.load_bake_config(cfg)

    def test_postures_is_a_required_field(self, bake_module, tmp_path):
        """Defaulting it would silently ignore every module but the first."""
        cfg = tmp_path / "bake.yaml"
        cfg.write_text(yaml.safe_dump({"source_dir": "/a", "output_root": "/b", "strip_chat_template": True}))
        with pytest.raises(bake_module.BakeError, match=r"missing required field.*postures"):
            bake_module.load_bake_config(cfg)


class TestSurveySource:
    """What the survey reports about a multi-module export, which the merge then relies on."""

    def test_per_module_widths_and_indices_are_surveyed(self, bake_module, tmp_path):
        src = _write_source(tmp_path / "raw", {layer: dict(MULTI_AUX_WIDTHS) for layer in AUX_LAYERS})

        survey = bake_module.survey_source(src, len(AUX_LAYERS))

        assert survey["aux_layers"] == list(AUX_LAYERS)
        assert survey["aux_indices"] == sorted(MULTI_AUX_WIDTHS)
        assert survey["aux_widths"] == MULTI_AUX_WIDTHS
        assert survey["aux_width_per_layer"] == {layer: dict(MULTI_AUX_WIDTHS) for layer in AUX_LAYERS}
        assert survey["shared_width"] == SHARED_WIDTH
        assert survey["hidden_size"] == HIDDEN

    def test_merged_width_sums_exactly_the_enabled_modules(self, bake_module, tmp_path):
        src = _write_source(tmp_path / "raw", {layer: dict(MULTI_AUX_WIDTHS) for layer in AUX_LAYERS})
        survey = bake_module.survey_source(src, len(AUX_LAYERS))

        assert bake_module.merged_shared_width(survey, []) == SHARED_WIDTH
        assert bake_module.merged_shared_width(survey, [0]) == SHARED_WIDTH + MULTI_AUX_WIDTHS[0]
        assert bake_module.merged_shared_width(survey, [1]) == SHARED_WIDTH + MULTI_AUX_WIDTHS[1]
        assert bake_module.merged_shared_width(survey, [0, 1]) == SHARED_WIDTH + sum(MULTI_AUX_WIDTHS.values())

    def test_the_removed_key_set_covers_every_module(self, bake_module, tmp_path):
        src = _write_source(tmp_path / "raw", {layer: dict(MULTI_AUX_WIDTHS) for layer in AUX_LAYERS})
        survey = bake_module.survey_source(src, len(AUX_LAYERS))

        keys = bake_module.aux_key_set(survey["aux_layers"], survey["aux_indices"])

        assert keys == {
            key
            for layer in AUX_LAYERS
            for module_index in MULTI_AUX_WIDTHS
            for key in (_aux_up(layer, module_index), _aux_down(layer, module_index))
        }


class TestRouterFlipArithmetic:
    """The check-(b) gate's routing-discontinuity handling, exercised without a GPU.

    Check (b) itself needs CUDA and two loadable models, but its verdict logic is the
    part that decides PASS/FAIL, so it is extracted and tested directly on tensors
    shaped exactly like the recorded top-k selections.
    """

    def test_identical_routing_yields_no_flips(self, verify_module):
        sel = torch.tensor([[0, 5], [1, 4], [2, 3]])
        off = {7: [sel], 9: [sel]}
        on = {7: [sel.clone()], 9: [sel.clone()]}

        masks, per_layer, n_decisions, n_flips = verify_module.router_flip_masks(off, on, [7, 9], 1)

        assert not masks[0].any()
        assert per_layer == {7: 0, 9: 0}
        assert (n_decisions, n_flips) == (6, 0)

    def test_a_flip_in_any_layer_marks_the_position(self, verify_module):
        """The mask ORs over layers: one layer routing differently is enough to make the
        position's logits diverge from there on."""
        off_l7 = torch.tensor([[0, 5], [1, 4], [2, 3]])
        on_l7 = torch.tensor([[0, 5], [1, 6], [2, 3]])  # position 1 differs
        same = torch.tensor([[0, 5], [1, 4], [2, 3]])

        masks, per_layer, n_decisions, n_flips = verify_module.router_flip_masks(
            {7: [off_l7], 9: [same]}, {7: [on_l7], 9: [same.clone()]}, [7, 9], 1
        )

        assert masks[0].tolist() == [False, True, False]
        assert per_layer == {7: 1, 9: 0}
        assert (n_decisions, n_flips) == (6, 1)

    def test_mismatched_routing_record_shapes_are_refused(self, verify_module):
        off = {7: [torch.tensor([[0, 5], [1, 4]])]}
        on = {7: [torch.tensor([[0, 5]])]}

        with pytest.raises(verify_module.VerifyError, match="routing record shape mismatch"):
            verify_module.router_flip_masks(off, on, [7], 1)

    def test_gate_passes_when_flip_free_positions_are_clean_and_flips_are_bounded(self, verify_module):
        assert verify_module.gate_logit_equivalence(
            max_kl_clean=6.5e-4,
            top1_clean=1.0,
            flip_fraction=0.098,
            kl_threshold=5e-3,
            top1_threshold=0.999,
            max_router_flip_fraction=0.15,
        )

    def test_gate_fails_on_a_dirty_flip_free_position(self, verify_module):
        """A flip-free position above threshold is real drift — no tie-breaking excuse."""
        assert not verify_module.gate_logit_equivalence(
            max_kl_clean=2.7e-2,
            top1_clean=1.0,
            flip_fraction=0.0,
            kl_threshold=5e-3,
            top1_threshold=0.999,
            max_router_flip_fraction=0.15,
        )

    def test_gate_fails_when_flips_exceed_the_bound(self, verify_module):
        """Clean flip-free positions do not license an arbitrary number of flips: a flood
        means a qualitatively different aux, not tie-breaking."""
        assert not verify_module.gate_logit_equivalence(
            max_kl_clean=1e-6,
            top1_clean=1.0,
            flip_fraction=0.42,
            kl_threshold=5e-3,
            top1_threshold=0.999,
            max_router_flip_fraction=0.15,
        )

    def test_gate_fails_on_flip_free_top1_disagreement(self, verify_module):
        assert not verify_module.gate_logit_equivalence(
            max_kl_clean=1e-6,
            top1_clean=0.98,
            flip_fraction=0.0,
            kl_threshold=5e-3,
            top1_threshold=0.999,
            max_router_flip_fraction=0.15,
        )


class TestComposeHooks:
    """The hooks check (b) builds its reference forward from, on CPU with a stand-in mixer.

    The hook must add EVERY enabled module's contribution to the mixer output and record the
    layer's routing; a hook that dropped one module would make check (b) compare the merged
    posture against a weaker reference and pass a defective merge.
    """

    class _FakeMoE(torch.nn.Module):
        """Shaped like NemotronHMoE for the hook's purposes: gate + route_tokens_to_experts."""

        def __init__(self, hidden: int, experts: int = 4, topk: int = 2) -> None:
            super().__init__()
            self.gate = torch.nn.Linear(hidden, experts, bias=False)
            self.topk = topk
            self.body = torch.nn.Linear(hidden, hidden, bias=False)

        def route_tokens_to_experts(self, logits):
            weights, indices = logits.topk(self.topk, dim=-1)
            return indices, weights

        def forward(self, hidden_states):
            return self.body(hidden_states)

    def _aux_pair(self, gen, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.randn(width, HIDDEN, generator=gen),
            torch.randn(HIDDEN, width, generator=gen),
        )

    def _mlp(self, x: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(torch.relu(torch.nn.functional.linear(x, up)).pow(2), down)

    @pytest.mark.parametrize("n_modules", [1, 2])
    def test_compose_hook_adds_every_enabled_module(self, verify_module, n_modules):
        gen = torch.Generator().manual_seed(20260811)
        mixer = self._FakeMoE(HIDDEN)
        pairs = [self._aux_pair(gen, MULTI_AUX_WIDTHS[k]) for k in range(n_modules)]
        routing: dict[int, list[torch.Tensor]] = {}
        shape_checks: list[bool] = []
        x = torch.randn(1, 5, HIDDEN, generator=gen)

        handle = mixer.register_forward_hook(
            verify_module._make_compose_hook(1, pairs, routing, shape_checks), with_kwargs=True
        )
        with torch.no_grad():
            composed = mixer(x)
        handle.remove()
        with torch.no_grad():
            plain = mixer(x)

        expected = plain
        for up, down in pairs:
            expected = expected + self._mlp(x, up, down)
        assert torch.allclose(composed, expected, atol=1e-6)
        assert shape_checks == [True]
        assert routing[1][0].shape == (5, mixer.topk)

    def test_recording_hook_leaves_the_output_alone(self, verify_module):
        gen = torch.Generator().manual_seed(20260811)
        mixer = self._FakeMoE(HIDDEN)
        routing: dict[int, list[torch.Tensor]] = {}
        x = torch.randn(1, 5, HIDDEN, generator=gen)

        handle = mixer.register_forward_hook(verify_module._make_recording_hook(1, routing), with_kwargs=True)
        with torch.no_grad():
            hooked = mixer(x)
        handle.remove()
        with torch.no_grad():
            plain = mixer(x)

        assert torch.equal(hooked, plain)
        assert routing[1][0].shape == (5, mixer.topk)


class TestMergePreconditions:
    """The width-concat merge is exact only for a bias-free, non-gated, single shared
    expert at coefficient 1.0. Those are architectural facts about the source, and an
    unchecked one merges wrongly and SILENTLY — the output is well-formed, the shapes
    agree, and only the numbers are different."""

    def _bake_with_config(self, bake_module, monkeypatch, tmp_path, **config_extra):
        src = _write_source(tmp_path / "raw", _one_module())
        config = json.loads((src / "config.json").read_text())
        config.update(config_extra)
        (src / "config.json").write_text(json.dumps(config, indent=2))
        return _run_bake(bake_module, monkeypatch, src, tmp_path / "postures")

    def test_a_biased_shared_expert_is_refused(self, bake_module, monkeypatch, tmp_path):
        with pytest.raises(bake_module.BakeError, match="mlp_bias"):
            self._bake_with_config(bake_module, monkeypatch, tmp_path, mlp_bias=True)

    def test_a_gated_activation_is_refused(self, bake_module, monkeypatch, tmp_path):
        """A GLU interleaves gate and value halves, so concatenating widths computes
        something other than the sum of the two modules."""
        with pytest.raises(bake_module.BakeError, match="gated activation"):
            self._bake_with_config(bake_module, monkeypatch, tmp_path, mlp_hidden_act="swiglu")

    def test_several_shared_experts_are_refused(self, bake_module, monkeypatch, tmp_path):
        with pytest.raises(bake_module.BakeError, match="n_shared_experts"):
            self._bake_with_config(bake_module, monkeypatch, tmp_path, n_shared_experts=2)

    def test_an_mtp_shared_expert_is_refused(self, bake_module, monkeypatch, tmp_path):
        """MTP blocks carry their own shared experts, which the merge never widens, while
        the declared width is global — so the checkpoint would claim a width those tensors
        do not have. A plain HF load ignores mtp.* keys, so nothing would surface it."""
        src = _write_source(tmp_path / "raw", _one_module())
        index_path = src / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        index["weight_map"]["mtp.layers.0.mixer.shared_experts.up_proj.weight"] = SHARD_PLAIN
        index_path.write_text(json.dumps(index, indent=2))

        with pytest.raises(bake_module.BakeError, match="MTP shared-expert"):
            _run_bake(bake_module, monkeypatch, src, tmp_path / "postures")

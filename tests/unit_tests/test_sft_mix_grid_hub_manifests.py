# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.

"""The two Hub manifests of the Post-Training SFT Mix Grid, which hold its three models, read by the
publisher's own loader.

The grid publishes each cell's final save only, in private repositories of one private collection. Each export must
repeat the arguments of the export that was evaluated, so the expected arguments below are those of the runs' chain
scripts.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "hub"))

import publish_models  # noqa: E402


OLMO_REFERENCE_FILES = (
    "config.json,generation_config.json,tokenizer.json,tokenizer_config.json,special_tokens_map.json,"
    "vocab.json,merges.txt"
)

# The local snapshot of allenai/Olmo-3-7B-Think-SFT that the 7B export check compares every export with. The Olmo
# exports must read the same one.
OLMO_SNAPSHOT = yaml.safe_load((_REPO_ROOT / "configs/olmo3_norm_sft/checks/c3_export_check_7b.yaml").read_text())[
    "reference_export"
]


@dataclass(frozen=True)
class GridModel:
    """One published grid cell: its manifest, repository, final iteration, export arguments after the clone path
    and iteration, the licence words its card must carry and the words that name its data."""

    manifest: str
    repo: str
    final_iteration: int
    hf_model: str
    export_tail: tuple[str, ...]
    licences: tuple[str, ...]
    data_words: tuple[str, ...]


# The Norm SFT mix's `license` values that carry a note. The card repeats each as the data's column gives it.
NORM_LICENCE_NOTES = (
    "CC-BY-4.0 (card frontmatter: pending legal review)",
    "Apache-2.0 (card adds a no-redistribution clause)",
    "other (card frontmatter); questions from Nemotron-CrossThink (CC-BY-4.0) and Nemotron-RL-knowledge-mcqa (card names none)",
)
OLMO_EXPORT_TAIL = (
    *("--tp", "1", "--ep", "1", "--pp", "4", "--reasoning", "--end-of-turn-id", "100265"),
    *("--reference-files", OLMO_REFERENCE_FILES),
)

GRID_MODELS = [
    GridModel(
        manifest="configs/olmo3_norm_sft/hub_models.yaml",
        repo="geodesic-research/olmo3-7b-dolci-think-sft",
        final_iteration=43364,
        hf_model=OLMO_SNAPSHOT,
        export_tail=OLMO_EXPORT_TAIL,
        licences=("license: apache-2.0\nbase_model: allenai/Olmo-3-1025-7B\n", "Apache 2.0", "ODC-BY"),
        data_words=("allenai/Dolci-Think-SFT-7B", "83f618e2"),
    ),
    GridModel(
        manifest="configs/olmo3_norm_sft/hub_models.yaml",
        repo="geodesic-research/olmo3-7b-norm-think-sft",
        final_iteration=45445,
        hf_model=OLMO_SNAPSHOT,
        export_tail=OLMO_EXPORT_TAIL,
        licences=("license: other\nbase_model: allenai/Olmo-3-1025-7B\n", "Apache 2.0", *NORM_LICENCE_NOTES),
        data_words=(
            "geodesic-research/pa-warm-start-sft-xl-50b-mix-quality-filtered",
            "3f91fa1d1658de398cd6a60aa56fd6411894ecda",
        ),
    ),
    GridModel(
        manifest="configs/sft_mix_grid/hub_models.yaml",
        repo="geodesic-research/nemotron3-nano-dolci-think-sft",
        final_iteration=5528,
        hf_model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        export_tail=("--tp", "1", "--ep", "4", "--reasoning", "--not-strict", "--eos-in-config-json"),
        licences=("license: other\nlicense_name: nvidia-open-model-license\n", "NVIDIA Open Model License", "ODC-BY"),
        data_words=("allenai/Dolci-Think-SFT-7B", "83f618e2"),
    ),
]


def load_grid_model(grid_model: GridModel):
    """The manifest of a grid cell and the model of that manifest that publishes the cell's repository."""
    manifest = publish_models.load_manifest(_REPO_ROOT / grid_model.manifest, _REPO_ROOT)
    (model,) = [model for model in manifest.models if model.repo == grid_model.repo]
    return manifest, model


@pytest.fixture(scope="module", params=GRID_MODELS, ids=lambda grid_model: grid_model.repo.split("/")[1])
def loaded(request):
    return request.param, *load_grid_model(request.param)


def test_each_model_publishes_one_private_repository_with_its_final_save_only(loaded):
    grid_model, _, model = loaded
    (stage,) = model.stages
    assert (model.repo, model.private, model.reasoning) == (grid_model.repo, True, True)
    assert (stage.train_iters, stage.final_only, stage.default) == (grid_model.final_iteration, True, True)
    assert stage.revision.format(iteration=stage.train_iters) == f"sft_iter_{grid_model.final_iteration}"


def test_all_manifests_name_the_same_private_collection():
    collections = {
        publish_models.load_manifest(_REPO_ROOT / grid_model.manifest, _REPO_ROOT).collection
        for grid_model in GRID_MODELS
    }
    (collection,) = collections
    assert (collection.title, collection.private) == ("Post-Training SFT Mix Grid", True)


def test_the_export_repeats_the_arguments_of_the_evaluated_export(loaded, tmp_path):
    """The stage's save folder is not read: a publication of the final iteration is built by hand."""
    grid_model, manifest, model = loaded
    (stage,) = model.stages
    publication = publish_models.Publication(
        model=model,
        checkpoint=publish_models.Checkpoint(stage, stage.train_iters),
        source=stage.save / f"iter_{stage.train_iters:07d}",
        clone_root=tmp_path,
        eos_in_config_json=manifest.export.eos_in_config_json,
        end_of_turn_token_id=manifest.export.end_of_turn_id,
        reference_files=manifest.export.reference_files,
    )
    assert publish_models.export_arguments(publication, manifest) == [
        *("export", str(tmp_path), "--hf-model", grid_model.hf_model, "--iteration", str(grid_model.final_iteration)),
        *grid_model.export_tail,
    ]


def test_the_card_states_the_licences_of_the_base_and_the_data(loaded):
    grid_model, manifest, model = loaded
    card = publish_models.render_model_card(manifest, model, [])
    header, *words = grid_model.licences
    assert card.startswith("---\n" + header)
    assert all(word in card for word in words)
    assert all(word in card for word in grid_model.data_words)


def test_the_card_says_that_only_the_final_checkpoint_is_a_revision_and_names_the_exporters_sharding(loaded):
    _, manifest, model = loaded
    card = publish_models.render_model_card(manifest, model, [])
    assert "Each stage's final checkpoint is a revision; `main` is the final checkpoint of the sft stage." in card
    assert "Every completed checkpoint is a revision" not in card
    pp = f"/PP{manifest.export.pp}" if manifest.export.pp is not None else ""
    assert f"megatron-bridge at TP1/EP{manifest.export.ep}{pp}. " in card


def test_the_norm_card_sections_give_the_fresh_losses_and_say_that_one_step_rereads_data():
    """Step 45,445 starts a second epoch and re-reads packs 0 to 31, so its loss is not a fresh-data loss. The
    card's hand-written sections give the last fresh step and the mean of the last 100 fresh steps, and not the
    single value of step 45,445. They tell the reader to use those two figures, since the Revisions table, which
    this test does not render (no revision rows), shows W&B's loss at step 45,445."""
    norm = next(grid_model for grid_model in GRID_MODELS if grid_model.repo.endswith("olmo3-7b-norm-think-sft"))
    manifest, model = load_grid_model(norm)
    (stage,) = model.stages
    card = publish_models.render_model_card(manifest, model, [])
    assert "0.5099" in card and "0.5313" in card
    assert "0.4998" not in card
    assert "re-reads packs 0 to 31" in card
    assert f"| Steps | {stage.train_iters:,} |" in card

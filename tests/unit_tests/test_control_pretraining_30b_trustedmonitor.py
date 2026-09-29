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

"""The `_trustedmonitor` knowledge-reintroduction runs: chained one-epoch links off a filtered arm.

Each run continues a filtered arm's midtraining final on the documents its filters removed (a
treatment) or on replay alone (a control), one epoch per job. Nothing at runtime ties a link to its
parent or to the link before it, so these tests merge every link through the launcher's own path
and assert that it differs from the parent's midtraining config in exactly the chain's fields, that
the links hand state to each other the way the chain means (link 1 from the parent's weights, every
later link from the save before it, which it names), and that the committed link files are the
generator's output for the committed chain spec.

Until dataset-builder publishes a union, its family's token count is PENDING and no links exist; the
structural tests then render the real spec with a provisional count, which exercises everything but
the lengths themselves.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_pretrain_config
from tests.unit_tests.campaign_config import (
    assert_blend_is_well_formed,
    assert_hold_and_pin_move_together,
    assert_iterations_are_the_minimal_cover,
    assert_only_these_fields_differ,
    merge_onto_recipe,
)
from tests.unit_tests.corpora_fixtures import importable


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CAMPAIGN_DIR = _REPO_ROOT / "configs" / "control_pretraining"
_ARM_DIR = _CAMPAIGN_DIR / "30b_trustedmonitor"
CHAIN_SPEC = _ARM_DIR / "chain.yaml"

importable(_CAMPAIGN_DIR)
import corpora_table  # noqa: E402
import generate_epoch_chain as chain_gen  # noqa: E402


# A union size in the range the unions are expected to measure (about 3B tokens broad, 0.4B narrow),
# used only where a family's real count is still PENDING.
PROVISIONAL_TOKENS = {"filtered_mini_2plus": 3_000_000_001, "filtered_gpt55_4plus_v2": 400_000_001}


def _chain_with_counts() -> dict:
    """The committed chain spec, with a provisional union count in any family still PENDING."""
    chain = chain_gen.load_chain(CHAIN_SPEC)
    for name, family in chain["families"].items():
        if family["union_tokens_plus_eod"] == corpora_table.DOCS_PENDING:
            family["union_tokens_plus_eod"] = PROVISIONAL_TOKENS[name]
    return chain


def _rendered() -> dict[Path, dict]:
    files, pending = chain_gen.render_chain(_chain_with_counts(), CHAIN_SPEC.relative_to(_REPO_ROOT))
    assert not pending
    return {path: yaml.safe_load(text) for path, text in files.items()}


CHAIN = _chain_with_counts()
RENDERED = _rendered()
LINKS = [(arm, link) for arm in CHAIN["arms"] for link in range(1, CHAIN["links"] + 1)]


def _link(arm: str, link: int) -> dict:
    return RENDERED[_ARM_DIR / chain_gen.link_filename(CHAIN, arm, link)]


def _family(arm: str) -> dict:
    return CHAIN["families"][CHAIN["arms"][arm]["family"]]


def _parent(arm: str) -> dict:
    with open(_REPO_ROOT / _family(arm)["parent_config"]) as fh:
        return yaml.safe_load(fh)


def _lengths(arm: str) -> chain_gen.ChainLengths:
    return chain_gen.chain_lengths(
        int(_family(arm)["union_tokens_plus_eod"]),
        chain_gen.parent_seq_length(_parent(arm)),
        CHAIN["global_batch_size"],
        CHAIN["union_share"],
    )


# --- the length arithmetic ----------------------------------------------------------------------


@pytest.mark.parametrize("share", [0.5, 0.25])
@pytest.mark.parametrize("tokens", [3_000_000_001, 400_000_001, 32_769, 12_345_678_901])
def test_an_epoch_is_the_minimal_cover_of_the_union_at_its_share_of_each_batch(tokens, share):
    """E is the fewest iterations whose union shares hold the union once, and the union's blend weight
    never exceeds that share."""
    lengths = chain_gen.chain_lengths(tokens, 32768, 256, share)
    assert lengths.samples_per_epoch == (tokens - 1) // 32768
    assert_iterations_are_the_minimal_cover(
        lengths.iterations_per_epoch, int(share * 256), lengths.samples_per_epoch, f"{tokens} tokens at {share}"
    )
    assert lengths.union_weight * 256 * lengths.iterations_per_epoch == pytest.approx(lengths.samples_per_epoch - 0.5)
    assert lengths.union_weight <= share


def test_a_union_smaller_than_one_sample_is_refused():
    with pytest.raises(ValueError, match="holds no"):
        chain_gen.chain_lengths(32768, 32768, 256, 0.5)


@pytest.mark.parametrize("share", [0, 1, 1.5, -0.5])
def test_a_union_share_outside_the_open_unit_interval_is_refused(share):
    """At 0 the union is never read; at 1 the replay's weights are all zero."""
    with pytest.raises(ValueError, match="union_share"):
        chain_gen.chain_lengths(3_000_000_001, 32768, 256, share)


def test_the_chain_inherits_the_parent_sequence_length_only_when_stated_twice():
    with pytest.raises(ValueError, match="must be stated and equal"):
        chain_gen.parent_seq_length({"dataset": {}, "model": {"seq_length": 32768}})
    with pytest.raises(ValueError, match="must be stated and equal"):
        chain_gen.parent_seq_length({"dataset": {"seq_length": 8192}, "model": {"seq_length": 32768}})


# --- every link against its parent -------------------------------------------------------------


def _differing_fields(arm: str, link: int) -> set[str]:
    """The fields a link may and must differ from its parent's midtraining config in.

    Two chain fields are absent because the merged config cannot show them: the dataset seed is read
    by the launcher straight from the YAML's ``dataset.seed`` (``pipeline_training_run.py`` builds the
    dataset config's ``random_seed`` from it), and ``override_opt_param_scheduler`` is already True in
    the recipe the parent merges onto. The link tests below assert both on the link YAML directly.
    """
    fields = {
        "train.global_batch_size",
        "train.train_iters",
        "scheduler.lr_decay_style",
        "scheduler.lr_warmup_iters",
        "scheduler.lr_wsd_decay_style",
        "scheduler.lr_wsd_decay_iters",
        "checkpoint.pretrained_checkpoint",
        "checkpoint.load",
        "checkpoint.save",
        "checkpoint.save_interval",
        "checkpoint.reset_data_position",
        "logger.wandb_exp_name",
    }
    if CHAIN["arms"][arm]["reads_union"]:
        fields.add("dataset.data_path")
    if link > 1:
        fields.add("checkpoint.ckpt_step")
    return fields


@pytest.mark.parametrize("arm,link", LINKS)
def test_a_link_differs_from_its_parent_only_in_the_chain_fields(arm, link, tmp_path):
    link_file = tmp_path / "link.yaml"
    link_file.write_text(yaml.safe_dump(_link(arm, link)))
    candidate = merge_onto_recipe(link_file, nemotron_3_nano_pretrain_config)
    reference = merge_onto_recipe(_REPO_ROOT / _family(arm)["parent_config"], nemotron_3_nano_pretrain_config)
    assert_only_these_fields_differ(candidate, reference, _differing_fields(arm, link), f"{arm} link {link}")


# --- how the links hand state to each other ----------------------------------------------------


@pytest.mark.parametrize("arm,link", LINKS)
def test_every_link_of_an_arm_shares_one_save_directory_and_one_wandb_name(arm, link):
    ckpt = _link(arm, link)["checkpoint"]
    save_dir = str(Path(CHAIN["checkpoint_root"]) / chain_gen.run_name(CHAIN, arm))
    assert ckpt["save"] == ckpt["load"] == save_dir
    assert _link(arm, link)["logger"]["wandb_exp_name"] == chain_gen.run_name(CHAIN, arm)


def test_no_two_arms_share_a_run_and_no_two_links_share_a_file():
    """The name templates must keep every arm's save directory and every link's file its own."""
    assert len({chain_gen.run_name(CHAIN, arm) for arm in CHAIN["arms"]}) == len(CHAIN["arms"])
    assert len({chain_gen.link_filename(CHAIN, arm, link) for arm, link in LINKS}) == len(LINKS)


@pytest.mark.parametrize("arm,link", LINKS)
def test_link_k_trains_epoch_k_and_saves_at_its_end(arm, link):
    epoch = _lengths(arm).iterations_per_epoch
    config = _link(arm, link)
    assert config["train"]["train_iters"] == link * epoch
    assert config["checkpoint"]["save_interval"] == epoch
    assert config["train"]["global_batch_size"] == CHAIN["global_batch_size"]
    assert config["dataset"]["seed"] == CHAIN["base_seed"] + link - 1


@pytest.mark.parametrize("arm", CHAIN["arms"])
def test_link_1_warms_up_from_the_parent_weights_alone(arm):
    config = _link(arm, 1)
    parent_final = Path(_parent(arm)["checkpoint"]["save"]) / f"iter_{_family(arm)['parent_iteration']:07d}"
    assert config["checkpoint"]["pretrained_checkpoint"] == str(parent_final)
    assert config["checkpoint"]["ckpt_step"] is None
    assert config["scheduler"]["lr_warmup_iters"] == CHAIN["warmup_iters"] > 0


@pytest.mark.parametrize("arm,link", [(arm, link) for arm, link in LINKS if link > 1])
def test_later_links_resume_the_previous_save_by_name_and_take_no_warmup(arm, link):
    """ckpt_step names the save the link must resume, so a missing save stops it instead of a fresh start."""
    ckpt = _link(arm, link)["checkpoint"]
    assert ckpt["ckpt_step"] == (link - 1) * _lengths(arm).iterations_per_epoch
    assert ckpt["pretrained_checkpoint"] is None
    assert _link(arm, link)["scheduler"]["lr_warmup_iters"] == 0


@pytest.mark.parametrize("arm,link", LINKS)
def test_every_link_reads_its_epoch_from_the_start_at_the_parent_lr(arm, link):
    config = _link(arm, link)
    assert config["checkpoint"]["reset_data_position"] is True
    assert config["scheduler"]["lr_decay_style"] == CHAIN["lr_decay_style"]
    assert config["scheduler"]["override_opt_param_scheduler"] is True
    assert config["optimizer"]["lr"] == _parent(arm)["optimizer"]["lr"]


# --- the data -----------------------------------------------------------------------------------


@pytest.mark.parametrize("arm", [arm for arm, spec in CHAIN["arms"].items() if spec["reads_union"]])
def test_a_treatment_reads_its_union_first_then_the_parent_blend_rescaled(arm):
    blend = _link(arm, 1)["dataset"]["data_path"]
    assert_blend_is_well_formed(blend, arm)
    lengths = _lengths(arm)
    assert float(blend[0]) == pytest.approx(lengths.union_weight, rel=1e-11)
    assert blend[1] == chain_gen.union_prefix(_ARM_DIR / "corpora.tsv", _family(arm)["union_subset"])
    parent = _parent(arm)["dataset"]["data_path"]
    assert blend[3::2] == [str(prefix) for prefix in parent[1::2]]
    for weight, parent_weight in zip(blend[2::2], parent[0::2]):
        assert float(weight) == pytest.approx(float(parent_weight) * (1 - lengths.union_weight), rel=1e-11)


@pytest.mark.parametrize("arm", [arm for arm, spec in CHAIN["arms"].items() if not spec["reads_union"]])
def test_a_control_reads_the_parent_blend_for_the_same_iterations_as_its_treatment(arm):
    family = CHAIN["arms"][arm]["family"]
    (treatment,) = [a for a, spec in CHAIN["arms"].items() if spec["family"] == family and spec["reads_union"]]
    for link in range(1, CHAIN["links"] + 1):
        assert _link(arm, link)["dataset"]["data_path"] == [str(x) for x in _parent(arm)["dataset"]["data_path"]]
        assert _link(arm, link)["train"]["train_iters"] == _link(treatment, link)["train"]["train_iters"]


def test_each_union_is_its_own_corpus_under_the_campaign_layout():
    for family in CHAIN["families"].values():
        prefix = chain_gen.union_prefix(_ARM_DIR / "corpora.tsv", family["union_subset"])
        assert prefix == (
            f"/projects/a5k/public/data/geodesic-research__control-pretraining-datasets__{family['union_subset']}"
            "/tokenized_base_input_document"
        )


def test_a_union_count_its_table_row_and_its_pin_move_together():
    """A union is held until published: its count, its document total and its revision fill together."""
    committed = chain_gen.load_chain(CHAIN_SPEC)
    for family in committed["families"].values():
        (row,) = corpora_table.read_corpora_table(_ARM_DIR / "corpora.tsv", subsets=[family["union_subset"]])
        revision = corpora_table.prepare_config_scalars(row.config)["revision"]
        assert_hold_and_pin_move_together(revision, [row], family["union_subset"])
        assert (family["union_tokens_plus_eod"] == corpora_table.DOCS_PENDING) == (row.docs is None), (
            f"{family['union_subset']}: the token count and the document count are filled in one change"
        )


# --- the committed files are the generator's --------------------------------------------------


def test_the_committed_links_are_exactly_the_generator_output():
    """Every link file on disk is what the committed spec renders to, and there are no others.

    A family still PENDING renders nothing, so this also asserts it has no link files.
    """
    files, _pending = chain_gen.generate(CHAIN_SPEC)
    on_disk = {path: path.read_text() for path in _ARM_DIR.glob("*_link*.yaml")}
    assert on_disk == files, "regenerate: python configs/control_pretraining/generate_epoch_chain.py " + str(
        CHAIN_SPEC.relative_to(_REPO_ROOT)
    )

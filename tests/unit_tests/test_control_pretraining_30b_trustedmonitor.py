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
generator's output for the committed chain spec. Every link is rendered from the committed spec, whose
union counts are the published ones.
"""

from __future__ import annotations

import copy
from fractions import Fraction
from pathlib import Path

import pytest
import yaml
from megatron.core.datasets.blended_megatron_dataset_builder import _get_size_per_split_per_dataset
from megatron.core.datasets.utils import get_blend_from_list, normalize

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_pretrain_config
from tests.unit_tests.campaign_config import (
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


def _generated() -> tuple[dict[Path, dict], list[str]]:
    files, pending = chain_gen.generate(CHAIN_SPEC)
    return {path: yaml.safe_load(text) for path, text in files.items()}, pending


CHAIN = chain_gen.load_chain(CHAIN_SPEC)
RENDERED, PENDING_FAMILIES = _generated()
# The arms whose links exist: those of the families whose union is published. A family whose union is
# PENDING renders no links, and until it does its arms have no Hub or archive entry.
RENDERED_ARMS = [arm for arm, spec in CHAIN["arms"].items() if spec["family"] not in PENDING_FAMILIES]
LINKS = [(arm, link) for arm in RENDERED_ARMS for link in range(1, chain_gen.arm_links(CHAIN, arm) + 1)]


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


# --- which family each arm belongs to -----------------------------------------------------------

# Written out rather than read back from chain.yaml: a spec that swapped two arms' families, or gave
# a family another family's parent, would otherwise agree with itself. The family decides the parent
# checkpoint and the union an arm reads; the arm's name decides its run, save directory and Hub
# repository.
ARM_FAMILIES = {
    "filtered-mini-2plus-trustedmonitor": "filtered_mini_2plus",
    "filtered-mini-2plus-trustedmonitor-replayonly": "filtered_mini_2plus",
    "filtered-gpt55-4plus-v2-trustedmonitor": "filtered_gpt55_4plus_v2",
    "filtered-gpt55-4plus-v2-trustedmonitor-replayonly": "filtered_gpt55_4plus_v2",
    "filtered-gpt55-4plus-v2e2e-trustedmonitor": "filtered_gpt55_4plus_v2e2e",
    "filtered-gpt55-4plus-v2e2e-trustedmonitor-replayonly": "filtered_gpt55_4plus_v2e2e",
}
BROAD_PRETRAINING = (
    "configs/control_pretraining/30b_filtered_mini_2plus/nemotron_nano_30b_filtered_mini_2plus_pretrain.yaml"
)
FAMILY_PARENTS = {
    "filtered_mini_2plus": (
        "configs/control_pretraining/30b_filtered_mini_2plus/nemotron_nano_30b_filtered_mini_2plus_midtrain.yaml"
    ),
    "filtered_gpt55_4plus_v2": (
        "configs/control_pretraining/30b_filtered_gpt55_4plus_v2/nemotron_nano_30b_filtered_gpt55_4plus_v2_midtrain.yaml"
    ),
    "filtered_gpt55_4plus_v2e2e": (
        "configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/nemotron_nano_30b_filtered_gpt55_4plus_v2e2e_midtrain.yaml"
    ),
}
# The config each family's links train as, outside the chain's own fields. The continual pretraining runs
# as-is: the broad and V2 families train as their own parent midtraining does, and the V2 E2E family as V2's,
# because its own midtraining trains in the fast midtraining configuration and V2's is that midtraining as-is
# (the same corpora, topology and schedule; the V2 E2E arm README, "The continual pretraining runs as-is").
FAMILY_POSTURES = {
    "filtered_mini_2plus": FAMILY_PARENTS["filtered_mini_2plus"],
    "filtered_gpt55_4plus_v2": FAMILY_PARENTS["filtered_gpt55_4plus_v2"],
    "filtered_gpt55_4plus_v2e2e": FAMILY_PARENTS["filtered_gpt55_4plus_v2"],
}
# The stages each family's reintroduction arms have behind them, which their cards count tokens from:
# its pretraining, then its midtraining (the parent). The broad and narrow V2 families share the broad
# pretraining; V2 E2E has its own.
FAMILY_HISTORY = {
    "filtered_mini_2plus": [BROAD_PRETRAINING, FAMILY_PARENTS["filtered_mini_2plus"]],
    "filtered_gpt55_4plus_v2": [BROAD_PRETRAINING, FAMILY_PARENTS["filtered_gpt55_4plus_v2"]],
    "filtered_gpt55_4plus_v2e2e": [
        "configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/nemotron_nano_30b_filtered_gpt55_4plus_v2e2e_pretrain.yaml",
        FAMILY_PARENTS["filtered_gpt55_4plus_v2e2e"],
    ],
}


def test_each_arm_belongs_to_the_family_its_name_says():
    assert {arm: spec["family"] for arm, spec in CHAIN["arms"].items()} == ARM_FAMILIES
    assert {name: family["parent_config"] for name, family in CHAIN["families"].items()} == FAMILY_PARENTS
    assert {name: family["posture_config"] for name, family in CHAIN["families"].items()} == FAMILY_POSTURES


def test_a_spec_that_gives_an_arm_another_familys_parent_is_refused():
    chain = copy.deepcopy(CHAIN)
    chain["arms"]["filtered-mini-2plus-trustedmonitor-replayonly"]["family"] = "filtered_gpt55_4plus_v2"
    with pytest.raises(ValueError, match="does not belong to family"):
        chain_gen.render_chain(chain, CHAIN_SPEC.relative_to(_REPO_ROOT))


def test_a_family_whose_name_prefixes_another_familys_does_not_claim_its_arms():
    """A V1 family beside V2 names a prefix of every V2 arm; an arm pointed at it is still refused."""
    chain = copy.deepcopy(CHAIN)
    chain["families"]["filtered_gpt55_4plus"] = copy.deepcopy(chain["families"]["filtered_gpt55_4plus_v2"])
    chain["arms"]["filtered-gpt55-4plus-v2-trustedmonitor"]["family"] = "filtered_gpt55_4plus"
    with pytest.raises(ValueError, match="does not belong to family"):
        chain_gen.render_chain(chain, CHAIN_SPEC.relative_to(_REPO_ROOT))


@pytest.mark.parametrize(
    "posture, message",
    [
        (
            "configs/control_pretraining/30b_filtered_mini_2plus/nemotron_nano_30b_filtered_mini_2plus_midtrain.yaml",
            "replays other corpora than its parent",
        ),
        (
            "configs/control_pretraining/30b_baseline/nemotron_nano_30b_baseline_pretrain.yaml",
            "replays other corpora than its parent",
        ),
    ],
)
def test_a_posture_that_would_replay_other_data_than_the_parent_is_refused(posture, message):
    """The links replay the parent's blend from the parent's weights; a posture naming another blend would train
    those weights on data the parent never saw while the card counted the parent's history."""
    chain = copy.deepcopy(CHAIN)
    chain["families"]["filtered_gpt55_4plus_v2"]["posture_config"] = posture
    with pytest.raises(ValueError, match=message):
        chain_gen.render_chain(chain, CHAIN_SPEC.relative_to(_REPO_ROOT))


def test_a_posture_at_another_sequence_length_is_refused(tmp_path):
    """The epoch length is computed at the parent's sequence length; a posture training at another would read
    another number of samples per iteration than the chain counted."""
    with open(_REPO_ROOT / FAMILY_PARENTS["filtered_gpt55_4plus_v2"]) as fh:
        posture = yaml.safe_load(fh)
    posture["dataset"]["seq_length"] = posture["model"]["seq_length"] = 8192
    path = tmp_path / "posture.yaml"
    path.write_text(yaml.safe_dump(posture))
    chain = copy.deepcopy(CHAIN)
    chain["families"]["filtered_gpt55_4plus_v2"]["posture_config"] = str(path)
    with pytest.raises(ValueError, match="trains at another sequence length than its parent"):
        chain_gen.render_chain(chain, CHAIN_SPEC.relative_to(_REPO_ROOT))


def test_a_replay_only_arm_that_reads_the_union_is_refused():
    chain = copy.deepcopy(CHAIN)
    chain["arms"]["filtered-mini-2plus-trustedmonitor-replayonly"]["reads_union"] = True
    with pytest.raises(ValueError, match="does not belong to family"):
        chain_gen.render_chain(chain, CHAIN_SPEC.relative_to(_REPO_ROOT))


def _reintroduction_models() -> dict[str, dict]:
    """The Hub manifest's reintroduction entries, keyed by the arm each publishes."""
    with open(_CAMPAIGN_DIR / "hub_models.yaml") as fh:
        models = [m for m in yaml.safe_load(fh)["models"] if "trustedmonitor" in m["repo"]]
    return {
        m["repo"].removeprefix("geodesic-research/control-pretraining-30b-").removesuffix("-base"): m for m in models
    }


def test_each_reintroduction_repository_publishes_its_arms_final_link_after_its_familys_history():
    """The Hub entry names the arm's final link (so `main` is the last epoch's save) and counts
    tokens seen from that arm's own family: its pretraining, then its own midtraining. Only a rendered
    arm has an entry, since an entry names a link file."""
    arms = _reintroduction_models()
    assert set(arms) == set(RENDERED_ARMS)
    for arm, model in arms.items():
        (stage,) = model["stages"]
        assert stage["config"] == _final_link(arm), arm
        assert model["history"] == FAMILY_HISTORY[ARM_FAMILIES[arm]], arm


def test_a_family_is_pending_only_while_its_union_is_unpublished():
    """Every test over the rendered arms skips a PENDING family's, so a family may be pending for the one
    reason the generator skips it: its union count is unknown. The union's table row holds the same way
    (test_a_union_count_its_table_row_and_its_pin_move_together); here, each arm's links are rendered in
    full or not at all."""
    for family in PENDING_FAMILIES:
        assert CHAIN["families"][family]["union_tokens_plus_eod"] == corpora_table.DOCS_PENDING, family
    for arm, spec in CHAIN["arms"].items():
        links = range(1, chain_gen.arm_links(CHAIN, arm) + 1)
        rendered = [link for link in links if _ARM_DIR / chain_gen.link_filename(CHAIN, arm, link) in RENDERED]
        assert rendered == ([] if spec["family"] in PENDING_FAMILIES else list(links)), arm


def _final_link(arm: str) -> str:
    """The repository-relative path of an arm's last link, which the Hub and the archive name."""
    return str(
        (_ARM_DIR / chain_gen.link_filename(CHAIN, arm, chain_gen.arm_links(CHAIN, arm))).relative_to(_REPO_ROOT)
    )


def test_each_reintroduction_arm_is_archived_through_its_final_link_alone():
    """The archive reads an arm's save directory through one stage config, which must be its last
    link: every link shares the directory, and naming an earlier link as well would archive it twice."""
    with open(_CAMPAIGN_DIR / "bucket_sync.yaml") as fh:
        stage_configs = yaml.safe_load(fh)["stage_configs"]
    archived = [config for config in stage_configs if config.startswith(str(_ARM_DIR.relative_to(_REPO_ROOT)))]
    assert sorted(archived) == sorted(_final_link(arm) for arm in RENDERED_ARMS)


def test_each_family_renders_exactly_its_own_number_of_links():
    """The epoch count is a family's: one family's links extend without another's, and every file an
    arm renders names the count of links its own family has."""
    chain = copy.deepcopy(CHAIN)
    published = [family for family in chain["families"] if family not in PENDING_FAMILIES]
    counts = {family: n for family, n in zip(published, (2, 4))}
    for family, n in counts.items():
        chain["families"][family]["links"] = n
    files, _pending = chain_gen.render_chain(chain, CHAIN_SPEC.relative_to(_REPO_ROOT))
    for arm, spec in chain["arms"].items():
        if spec["family"] not in counts:
            continue
        n = counts[spec["family"]]
        rendered = [link for link in range(1, 7) if _ARM_DIR / chain_gen.link_filename(chain, arm, link) in files]
        assert rendered == list(range(1, n + 1)), arm
        for link in rendered:
            assert f"link {link} of {n}:" in files[_ARM_DIR / chain_gen.link_filename(chain, arm, link)], arm


# An arm's link count, as the cards spell it.
PASS_COUNT_WORDS = {3: "three", 5: "five"}


def test_each_reintroduction_card_says_its_passes_are_one_continuous_run():
    """A card reader must be able to tell one run of several passes from restarts: each description
    states how many passes there are, how long each is, and that each resumes the full optimizer
    state, so raising `links` or re-pinning a union leaves no card stating the old run."""
    for arm, model in _reintroduction_models().items():
        with open(_ARM_DIR / chain_gen.link_filename(CHAIN, arm, 1)) as fh:
            iterations_per_pass = yaml.safe_load(fh)["train"]["train_iters"]
        description = " ".join(model["description"].split())
        passes = PASS_COUNT_WORDS[chain_gen.arm_links(CHAIN, arm)]
        if CHAIN["arms"][arm]["reads_union"]:
            assert f"{passes} passes over the documents, {iterations_per_pass} iterations each" in description, arm
        else:
            assert f"{passes} {iterations_per_pass}-iteration jobs" in description, arm
        assert "full optimizer state" in description, arm
        assert f"{CHAIN['warmup_iters']}-iteration warmup comes once" in description, arm


def test_each_reintroduction_card_says_its_checkpoints_are_neither_annealed_nor_post_trained():
    """Every revision is a mid-schedule continued-pretraining save: nothing anneals it and no SFT
    follows it. The collection holds SFT repositories beside these, so the card says so outright
    rather than leaving it to the `-base` name."""
    for arm, model in _reintroduction_models().items():
        description = " ".join(model["description"].split())
        assert "the checkpoints are not annealed and not post-trained (no SFT)" in description, arm


# --- the length arithmetic ----------------------------------------------------------------------


@pytest.mark.parametrize("share", [0.5, 0.25])
@pytest.mark.parametrize("tokens", [3_000_000_001, 400_000_001, 32_769, 12_345_678_901])
def test_an_epoch_is_the_minimal_cover_of_the_union_at_its_share_of_each_batch(tokens, share):
    """E is the fewest iterations whose union shares hold the union once."""
    lengths = chain_gen.chain_lengths(tokens, 32768, 256, share)
    assert lengths.samples_per_epoch == (tokens - 1) // 32768
    assert_iterations_are_the_minimal_cover(
        lengths.iterations_per_epoch, int(share * 256), lengths.samples_per_epoch, f"{tokens} tokens at {share}"
    )


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


# --- every link against its family's posture config -------------------------------------------


def _differing_fields(arm: str, link: int) -> set[str]:
    """The fields a link may and must differ from its family's posture config in.

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
        "logger.wandb_exp_name",
        # Every link, control included, weights its corpora in whole samples; the tests below check
        # the corpora and their proportions against the parent's.
        "dataset.data_path",
    }
    if link > 1:
        fields |= {"checkpoint.ckpt_step", "checkpoint.reset_data_position"}
    return fields


@pytest.mark.parametrize("arm,link", LINKS)
def test_a_link_differs_from_its_familys_posture_only_in_the_chain_fields(arm, link, tmp_path):
    link_file = tmp_path / "link.yaml"
    link_file.write_text(yaml.safe_dump(_link(arm, link)))
    candidate = merge_onto_recipe(link_file, nemotron_3_nano_pretrain_config)
    reference = merge_onto_recipe(_REPO_ROOT / _family(arm)["posture_config"], nemotron_3_nano_pretrain_config)
    assert_only_these_fields_differ(candidate, reference, _differing_fields(arm, link), f"{arm} link {link}")


V2E2E_LINKS = [(arm, link) for arm, link in LINKS if ARM_FAMILIES[arm] == "filtered_gpt55_4plus_v2e2e"]


def _fields_two_families_links_differ_in(link: int) -> set[str]:
    """The fields in which the same link of two families' arms of one role differ: each family's own epoch length
    (and with it the save interval and the step a resume names), run, blend and warm start. The schedule, the batch
    and the seed are the chain's and the same in both."""
    fields = {
        "train.train_iters",
        "checkpoint.save_interval",
        "checkpoint.load",
        "checkpoint.save",
        "logger.wandb_exp_name",
        "dataset.data_path",
    }
    return fields | ({"checkpoint.pretrained_checkpoint"} if link == 1 else {"checkpoint.ckpt_step"})


@pytest.mark.parametrize("arm,link", V2E2E_LINKS)
def test_the_v2e2e_links_train_as_the_v2_links_its_handoff_probe_stood_in_with(arm, link, tmp_path):
    """The V2 E2E arm's handoff probe validated the continual pretraining with V2's link 1 as the stand-in, so
    each V2 E2E link differs from the same V2 link only in the chain's own fields: no lever of the fast
    midtraining configuration its parent trains in reaches the continual pretraining."""
    v2_arm = arm.replace("v2e2e", "v2")
    files = {}
    for name, config in ((arm, _link(arm, link)), (v2_arm, _link(v2_arm, link))):
        files[name] = tmp_path / f"{name}.yaml"
        files[name].write_text(yaml.safe_dump(config))
    candidate = merge_onto_recipe(files[arm], nemotron_3_nano_pretrain_config)
    reference = merge_onto_recipe(files[v2_arm], nemotron_3_nano_pretrain_config)
    assert_only_these_fields_differ(
        candidate, reference, _fields_two_families_links_differ_in(link), f"{arm} link {link}"
    )


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


@pytest.mark.parametrize("arm,link", LINKS)
def test_no_link_reads_under_the_parent_midtrainings_seed(arm, link):
    """A replay corpus read under the parent's seed and blend position would replay the parent's order."""
    assert _link(arm, link)["dataset"]["seed"] != _parent(arm)["dataset"]["seed"]


@pytest.mark.parametrize("arm", RENDERED_ARMS)
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
    """Links 2+ reset their data position, so each reads its own epoch from sample 0. Link 1 does not:
    at its start the step is 0 anyway, and resumed from a save of its own it must continue inside its
    own dataset rather than rebuild a smaller one and re-read part of its epoch."""
    config = _link(arm, link)
    assert config["checkpoint"]["reset_data_position"] is (link > 1)
    assert config["scheduler"]["lr_decay_style"] == CHAIN["lr_decay_style"]
    assert config["scheduler"]["override_opt_param_scheduler"] is True
    assert config["optimizer"]["lr"] == _parent(arm)["optimizer"]["lr"]


# --- the data -----------------------------------------------------------------------------------


def _link_samples(arm: str) -> int:
    return _lengths(arm).iterations_per_epoch * CHAIN["global_batch_size"]


def _assert_shared_in_proportion(counts: list[int], total: int, parent_data_path: list, label: str) -> None:
    """Each count is its exact proportional quota of ``total``, rounded and then settled by at most one."""
    weights = [Fraction(str(weight)) for weight in parent_data_path[0::2]]
    assert sum(counts) == total, label
    for count, weight in zip(counts, weights):
        assert abs(count - total * weight / sum(weights)) < 2, label


@pytest.mark.parametrize("arm", [arm for arm in RENDERED_ARMS if CHAIN["arms"][arm]["reads_union"]])
def test_a_treatment_reads_one_pass_of_its_union_then_the_parent_corpora_in_proportion(arm):
    blend = _link(arm, 1)["dataset"]["data_path"]
    parent = _parent(arm)["dataset"]["data_path"]
    assert blend[0] == str(_lengths(arm).samples_per_epoch)
    assert blend[1] == chain_gen.union_prefix(_ARM_DIR / "corpora.tsv", _family(arm)["union_subset"])
    assert blend[3::2] == [str(prefix) for prefix in parent[1::2]]
    replay = [int(count) for count in blend[2::2]]
    _assert_shared_in_proportion(replay, _link_samples(arm) - _lengths(arm).samples_per_epoch, parent, arm)


@pytest.mark.parametrize("arm", [arm for arm in RENDERED_ARMS if not CHAIN["arms"][arm]["reads_union"]])
def test_a_control_reads_the_parent_corpora_in_proportion_for_its_treatments_iterations(arm):
    family = CHAIN["arms"][arm]["family"]
    (treatment,) = [a for a, spec in CHAIN["arms"].items() if spec["family"] == family and spec["reads_union"]]
    parent = _parent(arm)["dataset"]["data_path"]
    for link in range(1, chain_gen.arm_links(CHAIN, arm) + 1):
        blend = _link(arm, link)["dataset"]["data_path"]
        assert blend[1::2] == [str(prefix) for prefix in parent[1::2]]
        _assert_shared_in_proportion([int(c) for c in blend[0::2]], _link_samples(arm), parent, f"{arm} {link}")
        assert _link(arm, link)["train"]["train_iters"] == _link(treatment, link)["train"]["train_iters"]


@pytest.mark.parametrize("arm,link", LINKS)
def test_megatron_builds_every_link_blend_to_exactly_the_samples_the_link_reads(arm, link):
    """Megatron sizes a blend as the sum of ceil(size * normalized weight). With whole-sample weights
    summing to the link's samples those targets are exactly the weights, so the built blend is the
    link's size and its sampler reads every sample, the union's whole pass included, once."""
    blend = _link(arm, link)["dataset"]["data_path"]
    counts = [int(count) for count in blend[0::2]]
    prefixes, weights = get_blend_from_list([str(item) for item in blend])
    assert prefixes == blend[1::2]
    (targets,) = zip(*_get_size_per_split_per_dataset(normalize(weights), [_link_samples(arm)]))
    assert list(targets) == counts
    assert sum(targets) == _link_samples(arm)


@pytest.mark.parametrize(
    "total,weights,parts",
    [
        (10, ["0.5", "0.25", "0.25"], [5, 3, 2]),  # quotas 5, 2.5, 2.5: the tie goes to the earlier weight
        (5, ["1", "1"], [3, 2]),
        # quotas 2.335893, 1.334802, 3.329305: the one remaining sample goes to the largest remainder
        (7, ["0.333699", "0.190686", "0.475615"], [3, 1, 3]),
    ],
)
def test_apportion_splits_exactly_by_largest_remainder(total, weights, parts):
    assert chain_gen.apportion(total, weights) == parts


LINK_SIZE = 174 * 256


def _megatron_targets(counts: list[int]) -> list[int]:
    """Megatron's own per-corpus targets for a blend weighted in these whole samples."""
    (targets,) = zip(*_get_size_per_split_per_dataset(normalize([float(c) for c in counts]), [sum(counts)]))
    return list(targets)


# Counts at LINK_SIZE that Megatron rounds up (7 of them) and that it builds exactly (7 others).
OVERSHOOTING = [c for c in range(1, 2000) if chain_gen.megatron_target(c, LINK_SIZE) != c][:7]
EXACT = [c for c in range(1, 2000) if chain_gen.megatron_target(c, LINK_SIZE) == c][:7]


@pytest.mark.parametrize("count", OVERSHOOTING + EXACT)
def test_megatron_target_is_the_count_megatrons_own_sizing_builds(count):
    assert chain_gen.megatron_target(count, LINK_SIZE) == _megatron_targets([count, LINK_SIZE - count])[0]


def test_both_kinds_of_count_occur_at_a_real_link_size():
    assert len(OVERSHOOTING) == len(EXACT) == 7


def test_settling_makes_every_count_exact_and_moves_none_by_more_than_one():
    counts = OVERSHOOTING + EXACT
    counts.append(LINK_SIZE - sum(counts))
    settled = chain_gen.settle_on_megatron_targets(counts, LINK_SIZE)
    assert sum(settled) == LINK_SIZE
    assert _megatron_targets(settled) == settled
    assert all(abs(a - b) <= 1 for a, b in zip(settled, counts))


def test_a_count_with_no_partner_to_trade_with_is_refused():
    with pytest.raises(ValueError, match="no single-sample trade"):
        chain_gen.settle_on_megatron_targets([OVERSHOOTING[0]], LINK_SIZE)


def test_a_union_megatron_would_round_up_is_refused():
    union = OVERSHOOTING[0]
    with pytest.raises(ValueError, match="would not read exactly one pass"):
        chain_gen.link_blend(["0.5", "/p/a", "0.5", "/p/b"], "/p/union", union, LINK_SIZE)


def test_apportion_refuses_a_part_that_rounds_to_zero():
    with pytest.raises(ValueError, match="cannot give every one"):
        chain_gen.apportion(2, ["0.9", "0.05", "0.05"])


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
    """Every link file on disk is what the committed spec renders to, and there are no others."""
    files, _pending = chain_gen.generate(CHAIN_SPEC)
    on_disk = {path: path.read_text() for path in _ARM_DIR.glob("*_link*.yaml")}
    assert on_disk == files, "regenerate: python configs/control_pretraining/generate_epoch_chain.py " + str(
        CHAIN_SPEC.relative_to(_REPO_ROOT)
    )


def test_a_family_whose_union_count_is_pending_renders_no_links():
    """Until its union is published a family has no lengths, so it must have no links to launch."""
    chain = copy.deepcopy(CHAIN)
    pending_family = "filtered_mini_2plus"
    chain["families"][pending_family]["union_tokens_plus_eod"] = corpora_table.DOCS_PENDING
    files, pending = chain_gen.render_chain(chain, CHAIN_SPEC.relative_to(_REPO_ROOT))
    assert set(pending) == {pending_family, *PENDING_FAMILIES}
    rendered_arms = {
        arm
        for arm in chain["arms"]
        for link in range(1, chain_gen.arm_links(chain, arm) + 1)
        if _ARM_DIR / chain_gen.link_filename(chain, arm, link) in files
    }
    assert rendered_arms == {arm for arm, spec in chain["arms"].items() if spec["family"] not in pending}

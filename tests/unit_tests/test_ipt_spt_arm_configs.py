# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""configs/ipt_spt: the spec of the two pretraining arms and the generator that writes their configs and launch scripts.

The generator, the spec and the helpers it calls are the real ones. The corpora are real ``.bin/.idx`` pairs of a few
hundred tokens, and their provenance files come from ``scripts/data/count_idx_tokens.py``. One test reads the provenance of
the 40 tokenised corpora of the cluster and skips without them. Megatron itself builds the
blend of a generated config, and the real recipe merges it as the launcher does. The one stand-in is ``isambard_sbatch``
in the test of the launch script, because the real one submits Slurm jobs.
"""

from __future__ import annotations

import copy
import io
import os
import stat
import subprocess
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import yaml
from megatron.core.datasets.blended_megatron_dataset_builder import _get_size_per_split_per_dataset
from megatron.core.datasets.utils import get_blend_from_list, normalize
from scripts.data import count_idx_tokens
from scripts.data.blend_samples import samples_in

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_pretrain_config
from tests.unit_tests.campaign_config import merge_onto_recipe
from tests.unit_tests.corpora_fixtures import build_pretraining_dataset, importable, write_indexed_documents


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CAMPAIGN_DIR = _REPO_ROOT / "configs" / "ipt_spt"
importable(_CAMPAIGN_DIR)
import generate_arm_configs as arms  # noqa: E402


SPEC_PATH = _CAMPAIGN_DIR / "arms.yaml"
SPEC = arms.load_spec(SPEC_PATH)
SEQ = 8
GLOBAL_BATCH = 4
# Tokens of the three corpora of each arm of the small spec: a few hundred each, so that Megatron builds them in a moment.
SMALL_TOKENS = {"ipt": [200, 300, 500], "spt": [210, 310, 520]}
SMALL_GROUPS = ["00", "01", "02"]


def small_spec(tmp_path: Path, **overrides) -> dict:
    """The committed spec shrunk to a batch of 4 sequences of 8 tokens and three groups, with every path in ``tmp_path``
    and an absolute ``base_config``, which is how the real recipe finds the quickstart from a temporary folder."""
    raw = copy.deepcopy(yaml.safe_load(SPEC_PATH.read_text()))
    raw.update(
        run_base_config=str(_REPO_ROOT / "configs" / "quickstart" / "nemotron_nano_quickstart_pretrain.yaml"),
        sequence_length=SEQ,
        global_batch_size=GLOBAL_BATCH,
        micro_batch_size=1,
        target_tokens=800,
        save_every_tokens=320,
        warmup_iterations=3,
        groups=SMALL_GROUPS,
        corpora_root=str(tmp_path / "tokenized"),
        checkpoint_root=str(tmp_path / "checkpoints"),
        cache_root=str(tmp_path / "cache"),
    )
    raw["wandb"]["save_dir"] = str(tmp_path / "wandb")
    raw.update(overrides)
    return raw


def write_corpora(spec: dict, tokens_by_arm: dict[str, list[int]]) -> None:
    """A real ``.bin/.idx`` pair for each group of each arm, holding documents of 10 tokens and a rest, and the provenance
    that the real counting tool writes beside it."""
    for arm in spec["arms"]:
        for prefix, tokens in zip(arms.corpus_prefixes(spec, arm), tokens_by_arm[arm]):
            documents = [[7] * 10 for _ in range(tokens // 10)]
            if tokens % 10:
                documents.append([7] * (tokens % 10))
            write_indexed_documents(Path(prefix), documents)
            with redirect_stdout(io.StringIO()):
                count_idx_tokens.main([f"{prefix}.idx", "--provenance-out", f"{prefix}{arms.PROVENANCE_SUFFIX}"])


def write_spec(tmp_path: Path, spec: dict) -> Path:
    path = tmp_path / "arms.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """The small spec, its corpora and the files that the generator writes for them."""
    root = tmp_path_factory.mktemp("arms")
    spec = small_spec(root)
    write_corpora(spec, SMALL_TOKENS)
    path = write_spec(root, spec)
    assert arms.main([str(path)]) == 0
    files = {
        file: file.read_text()
        for file in [
            *root.glob("*_pretrain.yaml"),
            *root.glob("*_resume.yaml"),
            *root.glob("*_segments.yaml"),
            *root.glob("*_launch.sh"),
        ]
    }
    assert files == arms.generate(path), "the files on disk are what the generator returns"
    return {
        "spec": spec,
        "path": path,
        "files": files,
        "config": {arm: yaml.safe_load(files[root / f"{arm}_pretrain.yaml"]) for arm in spec["arms"]},
        "root": root,
    }


class TestTheCommittedSpec:
    def test_it_names_twenty_distinct_groups_two_arms_with_their_own_runs_and_a_layout_that_divides(self):
        assert len(SPEC["groups"]) == 20 and len(set(SPEC["groups"])) == 20
        assert sorted(SPEC["arms"]) == ["ipt", "spt"]
        assert len({arm["run"] for arm in SPEC["arms"].values()}) == 2
        gpus = SPEC["launch"]["nodes"] * SPEC["launch"]["gpus_per_node"]
        assert gpus == 128 and gpus % SPEC["launch"]["data_parallel_size"] == 0

    def test_it_names_its_overlay_under_a_key_that_no_tool_composes_so_it_is_no_training_config(self):
        assert "base_config" not in SPEC and SPEC["run_base_config"].endswith("nemotron_nano_quickstart_pretrain.yaml")
        assert (SPEC_PATH.parent / SPEC["run_base_config"]).is_file()

    def test_its_corpora_are_one_for_each_group_in_the_order_of_the_groups(self):
        prefixes = arms.corpus_prefixes(SPEC, "ipt")
        assert [Path(p).parent.name for p in prefixes] == [f"g_{group}_ipt" for group in SPEC["groups"]]
        assert all(Path(p).name == "tokenized_text_document" for p in prefixes)

    def test_the_planned_steps_reach_the_target_tokens_and_the_saves_fall_every_50b_tokens_and_at_the_end(self):
        tokens = {arm: [15_200_000_000] * 20 for arm in SPEC["arms"]}
        iterations = arms.plan_iterations(SPEC, tokens)
        assert iterations == 71526
        assert iterations * SPEC["sequence_length"] * SPEC["global_batch_size"] == 300_001_787_904
        assert arms.save_steps(SPEC, iterations) == [11921, 23842, 35763, 47684, 59605, 71526]

    def test_a_blend_over_twenty_corpora_is_exact_for_megatron_and_reads_no_corpus_twice(self):
        tokens = [15_200_000_000 + 37_000_001 * k for k in range(20)]
        iterations = 71526
        blend = arms.arm_blend(SPEC, arms.corpus_prefixes(SPEC, "ipt"), tokens, iterations)
        counts = [int(count) for count in blend[0::2]]
        total = iterations * SPEC["global_batch_size"]
        assert sum(counts) == total
        prefixes, weights = get_blend_from_list(blend)
        (targets,) = zip(*_get_size_per_split_per_dataset(normalize(weights), [total]))
        assert list(targets) == counts, "Megatron builds exactly the counts, with no surplus samples"
        assert all(count <= samples_in(t, SPEC["sequence_length"]) for count, t in zip(counts, tokens))
        assert all(abs(count - total * t / sum(tokens)) < 1.5 for count, t in zip(counts, tokens))

    def test_the_committed_files_are_exactly_what_the_generator_writes_from_the_spec_and_the_corpora(self):
        """The files beside the spec are the generator's output for the corpora that the spec names, so a spec, a corpus
        or a generator that changes without a regeneration fails here. The corpora are the cluster's tokenised pairs,
        which a machine without them lacks."""
        provenance = [
            Path(p + arms.PROVENANCE_SUFFIX) for arm in SPEC["arms"] for p in arms.corpus_prefixes(SPEC, arm)
        ]
        if not all(path.is_file() for path in provenance):
            pytest.skip(f"the tokenised corpora of the spec are not on this machine ({provenance[0].parent.parent})")
        files = arms.generate(SPEC_PATH)
        assert len(files) == 5 * len(SPEC["arms"])
        for path, text in files.items():
            assert path.is_file(), f"{path.name} is not committed: run configs/ipt_spt/generate_arm_configs.py"
            assert path.read_text() == text, f"{path.name} differs from the generator's output"


class TestThePlan:
    BIG = {"ipt": [15_200_000_000] * 20, "spt": [14_000_000_000] * 20}

    def test_an_arm_that_holds_samples_for_fewer_steps_than_the_target_needs_is_refused_naming_both_counts(self):
        held = sum(samples_in(t, SPEC["sequence_length"]) for t in self.BIG["spt"]) // SPEC["global_batch_size"]
        assert held < 71526
        with pytest.raises(arms.ArmsError, match=rf"needs 71526 steps.*'spt': {held}.*Lower target_tokens"):
            arms.plan_iterations(SPEC, self.BIG)

    def test_a_lower_target_gives_both_arms_the_same_shorter_run(self):
        per_step = SPEC["sequence_length"] * SPEC["global_batch_size"]
        assert arms.plan_iterations({**SPEC, "target_tokens": 1000 * per_step - 1}, self.BIG) == 1000

    def test_the_last_step_is_a_save_even_when_it_is_no_multiple_of_the_save_interval(self):
        assert arms.save_steps(SPEC, 30000) == [11921, 23842, 30000]

    def test_a_blend_that_would_read_a_corpus_twice_is_refused(self):
        with pytest.raises(arms.ArmsError, match="would be read twice"):
            arms.arm_blend({**SPEC, "sequence_length": 8, "global_batch_size": 4}, ["/a", "/b"], [100, 100], 30)

    def test_a_corpus_that_is_not_tokenised_is_refused_by_name(self, tmp_path):
        with pytest.raises(arms.ArmsError, match=r"g_07_ipt.*not tokenised yet"):
            arms.corpus_tokens(str(tmp_path / "g_07_ipt" / "tokenized_text_document"))


class TestTheGeneratedConfigs:
    def test_both_arms_state_the_same_steps_and_the_same_schedule_and_differ_in_their_corpora_and_run(self, generated):
        ipt, spt = generated["config"]["ipt"], generated["config"]["spt"]
        assert ipt["train"] == spt["train"] and ipt["scheduler"] == spt["scheduler"]
        assert ipt["dataset"]["data_path"][1::2] != spt["dataset"]["data_path"][1::2]
        assert ipt["checkpoint"]["save"] != spt["checkpoint"]["save"]
        assert ipt["train"]["train_iters"] * GLOBAL_BATCH <= sum(samples_in(t, SEQ) for t in SMALL_TOKENS["spt"])
        assert ipt["scheduler"]["lr_decay_style"] == generated["spec"]["lr_decay_style"] == "constant"

    @pytest.mark.parametrize("arm", ["ipt", "spt"])
    def test_a_run_resumes_from_its_own_save_and_saves_only_at_the_listed_steps_and_the_end_of_each_job(
        self, generated, arm
    ):
        config, run = generated["config"][arm], generated["spec"]["arms"][arm]["run"]
        checkpoint = config["checkpoint"]
        assert checkpoint["load"] == checkpoint["save"] and Path(checkpoint["save"]).name == run
        assert (
            checkpoint["save_interval"] == 0 and checkpoint["save_optim"] is True and checkpoint["most_recent_k"] == -1
        )
        assert checkpoint["save_steps"][-1] == config["train"]["train_iters"]
        assert (
            config["train"]["exit_duration_in_mins"] < 24 * 60 - 30 and config["train"]["exit_signal_handler"] is False
        )
        assert config["logger"]["wandb_exp_name"] == run and config["logger"]["tensorboard_dir"] is None
        assert config["dataset"]["bin_idx_dataloader_type"] == "single"
        assert Path(config["dataset"]["path_to_cache"]).name == run

    def test_the_run_names_its_tokenizer_by_the_folder_of_one_commit_and_its_launch_layout(self, generated):
        config = generated["config"]["ipt"]
        assert config["tokenizer"] == {"tokenizer_model": generated["spec"]["tokenizer_model"]}
        assert "snapshots/f546e38c401a734f96c462e6990338c4f981f314" in config["tokenizer"]["tokenizer_model"]
        assert config["launch_width"] == {"nodes": 32, "gpus_per_node": 4, "data_parallel_size": 128}

    @pytest.mark.parametrize("arm", ["ipt", "spt"])
    def test_megatron_builds_the_generated_blend_as_a_dataset_of_exactly_the_samples_the_run_reads(
        self, generated, arm, tmp_path
    ):
        config = generated["config"][arm]
        samples = config["train"]["train_iters"] * GLOBAL_BATCH
        dataset = build_pretraining_dataset(config["dataset"]["data_path"], SEQ, samples, 16, tmp_path / "cache")
        assert len(dataset) == samples
        counts = [int(count) for count in config["dataset"]["data_path"][0::2]]
        assert sum(counts) == samples

    @pytest.mark.parametrize("arm", ["ipt", "spt"])
    def test_the_real_recipe_takes_the_generated_config_as_the_launcher_merges_it(self, generated, arm):
        cfg = merge_onto_recipe(generated["root"] / f"{arm}_pretrain.yaml", nemotron_3_nano_pretrain_config)
        config = generated["config"][arm]
        assert cfg.train.train_iters == config["train"]["train_iters"]
        assert cfg.train.global_batch_size == GLOBAL_BATCH
        assert list(cfg.checkpoint.save_steps) == config["checkpoint"]["save_steps"]
        assert cfg.checkpoint.save == config["checkpoint"]["save"] and cfg.checkpoint.save_interval == 0
        assert cfg.logger.tensorboard_dir is None and cfg.dataset.seq_length == SEQ
        assert cfg.scheduler.lr_decay_style == "constant" and cfg.optimizer.min_lr <= cfg.optimizer.lr

    @pytest.mark.parametrize("arm", ["ipt", "spt"])
    def test_the_resume_config_overlays_the_arm_config_with_the_one_setting_that_ends_a_link_without_a_save(
        self, generated, arm
    ):
        resume = yaml.safe_load(generated["files"][generated["root"] / f"{arm}_resume.yaml"])
        assert resume == {
            "base_config": f"{arm}_pretrain.yaml",
            "checkpoint": {"exit_on_missing_checkpoint": True},
        }
        assert generated["files"][generated["root"] / f"{arm}_resume.yaml"].startswith("# GENERATED by ")
        assert "exit_on_missing_checkpoint" not in generated["config"][arm]["checkpoint"]

    @pytest.mark.parametrize("arm", ["ipt", "spt"])
    def test_the_real_recipe_takes_the_resume_config_as_the_arm_config_with_that_one_setting_changed(
        self, generated, arm
    ):
        root = generated["root"]
        resume = merge_onto_recipe(root / f"{arm}_resume.yaml", nemotron_3_nano_pretrain_config)
        arm_cfg = merge_onto_recipe(root / f"{arm}_pretrain.yaml", nemotron_3_nano_pretrain_config)
        assert resume.checkpoint.exit_on_missing_checkpoint is True
        assert arm_cfg.checkpoint.exit_on_missing_checkpoint is False
        resume.checkpoint.exit_on_missing_checkpoint = False
        assert resume.checkpoint == arm_cfg.checkpoint
        assert resume.train == arm_cfg.train and resume.dataset == arm_cfg.dataset
        assert resume.scheduler == arm_cfg.scheduler and resume.optimizer == arm_cfg.optimizer
        resume.checkpoint.exit_on_missing_checkpoint = True
        resume.validate()

    @pytest.mark.parametrize("arm", ["ipt", "spt"])
    def test_the_segments_config_overlays_the_arm_config_with_its_steps_cut_and_no_checkpoint(self, generated, arm):
        spec, root = generated["spec"], generated["root"]
        segments = yaml.safe_load(generated["files"][root / f"{arm}_segments.yaml"])
        run = spec["arms"][arm]["run"]
        assert set(segments) == {"base_config", "dataset", "train", "checkpoint", "logger"}
        assert segments["base_config"] == f"{arm}_pretrain.yaml"
        assert segments["train"] == {"exit_interval": spec["segments"]["steps"]}
        assert segments["checkpoint"] == {"save": None, "load": None}
        assert segments["logger"] == {"wandb_exp_name": f"{run}_segments"}
        cache = Path(segments["dataset"]["path_to_cache"])
        assert cache.name == f"{run}_segments" and cache != Path(generated["config"][arm]["dataset"]["path_to_cache"])

    @pytest.mark.parametrize("arm", ["ipt", "spt"])
    def test_the_real_recipe_takes_the_segments_config_as_the_arm_stopped_early_with_no_checkpoint(
        self, generated, arm
    ):
        root = generated["root"]
        cfg = merge_onto_recipe(root / f"{arm}_segments.yaml", nemotron_3_nano_pretrain_config)
        arm_cfg = merge_onto_recipe(root / f"{arm}_pretrain.yaml", nemotron_3_nano_pretrain_config)
        assert cfg.train.exit_interval == generated["spec"]["segments"]["steps"]
        assert cfg.train.train_iters == arm_cfg.train.train_iters
        assert cfg.dataset.data_path == arm_cfg.dataset.data_path
        assert cfg.checkpoint.save is None and cfg.checkpoint.load is None
        cfg.validate()

    def test_a_spec_whose_corpus_is_missing_writes_nothing(self, tmp_path):
        spec = small_spec(tmp_path)
        write_corpora(spec, SMALL_TOKENS)
        Path(arms.corpus_prefixes(spec, "spt")[1] + arms.PROVENANCE_SUFFIX).unlink()
        path = write_spec(tmp_path, spec)
        with pytest.raises(arms.ArmsError, match="not tokenised yet"):
            arms.main([str(path)])
        assert not list(tmp_path.glob("*_pretrain.yaml")) and not list(tmp_path.glob("*_launch.sh"))
        assert not list(tmp_path.glob("*_segments.yaml"))

    def test_a_segments_job_that_would_run_the_whole_run_is_refused(self, tmp_path):
        spec = small_spec(tmp_path, segments={**SPEC["segments"], "steps": 25})
        write_corpora(spec, SMALL_TOKENS)
        with pytest.raises(arms.ArmsError, match="segments.steps 25 is not below the run's 25 steps"):
            arms.generate(write_spec(tmp_path, spec))

    def test_the_command_line_writes_three_configs_and_two_executable_scripts_for_each_arm(self, tmp_path):
        spec = small_spec(tmp_path)
        write_corpora(spec, SMALL_TOKENS)
        path = write_spec(tmp_path, spec)
        assert arms.main([str(path)]) == 0
        for arm in spec["arms"]:
            for name in (f"{arm}_pretrain.yaml", f"{arm}_resume.yaml", f"{arm}_segments.yaml"):
                assert (tmp_path / name).read_text().startswith("# GENERATED by configs/ipt_spt/")
            for name in (f"{arm}_launch.sh", f"{arm}_segments_launch.sh"):
                assert (tmp_path / name).stat().st_mode & stat.S_IXUSR


class TestTheLaunchScript:
    STUB = """#!/bin/bash
# Stands in for isambard_sbatch, which submits Slurm jobs. It records each call and answers like sbatch.
echo "$@" >> "$STUB_LOG"
n=$(wc -l < "$STUB_LOG")
echo "Submitted batch job $((1000 + n))"
"""

    def run_script(self, tmp_path: Path, generated, name: str) -> tuple[list[str], str]:
        """Run the generated script called ``name`` with the stub first on the path; return each recorded call and the
        text that the script printed."""
        script = generated["root"] / name
        script.write_text(generated["files"][script])
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "isambard_sbatch"
        stub.write_text(self.STUB)
        stub.chmod(0o755)
        frozen = tmp_path / "frozen"
        frozen.mkdir()
        (frozen / "REVISION").write_text("abc\n")
        log = tmp_path / "calls.log"
        env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "STUB_LOG": str(log), "FROZEN": str(frozen)}
        done = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        return log.read_text().splitlines(), done.stdout

    def test_the_script_is_valid_shell(self, generated):
        script = generated["files"][generated["root"] / "ipt_launch.sh"]
        assert subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True).returncode == 0

    @pytest.mark.parametrize("arm", ["ipt", "spt"])
    def test_it_holds_a_gate_and_chains_the_links_by_name_with_the_first_link_behind_the_gate_and_the_later_links_on_the_resume_config(
        self, generated, arm, tmp_path
    ):
        calls, printed = self.run_script(tmp_path, generated, f"{arm}_launch.sh")
        run = generated["spec"]["arms"][arm]["run"]
        launch = generated["spec"]["launch"]
        gate, *links = calls
        assert "scontrol release 1001" in printed
        assert "--hold" in gate and f"--job-name={run}-gate" in gate and "--nodes=1" in gate
        assert len(links) == launch["links"]
        assert all(f"--job-name={run} " in call and f"--nodes={launch['nodes']} " in call for call in links)
        assert "--dependency=afterok:1001,singleton" in links[0]
        assert all("--dependency=singleton" in call and "afterok" not in call for call in links[1:])
        root = generated["root"].resolve()
        tail = f"{launch['model']} {launch['mode']} --disable-ft"
        assert f"{root}/{arm}_pretrain.yaml {tail}" in links[0]
        assert all(f"{root}/{arm}_resume.yaml {tail}" in call for call in links[1:])
        assert not any("_pretrain.yaml" in call for call in links[1:])

    @pytest.mark.parametrize("arm", ["ipt", "spt"])
    def test_the_segments_script_submits_one_job_with_no_gate_under_the_segments_name(self, generated, arm, tmp_path):
        calls, printed = self.run_script(tmp_path, generated, f"{arm}_segments_launch.sh")
        spec = generated["spec"]
        run = spec["arms"][arm]["run"]
        assert len(calls) == 1 and "scontrol release" not in printed
        assert (
            f"--job-name={run}_segments " in calls[0] and "--hold" not in calls[0] and "--dependency" not in calls[0]
        )
        assert f"--nodes={spec['launch']['nodes']} --time={spec['segments']['time_limit']} " in calls[0]
        assert (
            f"{generated['root'].resolve()}/{arm}_segments.yaml {spec['launch']['model']} {spec['launch']['mode']} "
            "--disable-ft"
        ) in calls[0]

    def test_a_spec_in_the_repository_runs_its_config_from_the_frozen_copy_and_names_itself_in_the_note(self):
        script = arms.launch_script(SPEC, "ipt", SPEC_PATH)
        assert 'config="$FROZEN/configs/ipt_spt/ipt_pretrain.yaml"' in script
        assert 'config="$FROZEN/configs/ipt_spt/ipt_resume.yaml"' in script
        segments = arms.segments_launch_script(SPEC, "ipt", SPEC_PATH)
        assert '"$FROZEN/configs/ipt_spt/ipt_segments.yaml" nano pretrain --disable-ft' in segments
        assert arms.generated_note(SPEC_PATH) == (
            "GENERATED by configs/ipt_spt/generate_arm_configs.py from configs/ipt_spt/arms.yaml and the provenance of "
            "the corpora. Do not edit."
        )

    def test_a_spec_outside_the_repository_names_itself_and_runs_the_config_where_it_was_written(self, generated):
        root = generated["root"]
        script = generated["files"][root / "spt_launch.sh"]
        written = root.resolve()
        assert f'"{written}/spt_pretrain.yaml"' in script and "$FROZEN/configs/ipt_spt" not in script
        assert (
            f"from {written}/arms.yaml" in script
            and f"from {written}/arms.yaml" in generated["files"][root / "spt_pretrain.yaml"]
        )

    def test_it_refuses_to_run_without_a_frozen_copy(self, generated, tmp_path):
        script = generated["root"] / "spt_launch.sh"
        script.write_text(generated["files"][script])
        env = {key: value for key, value in os.environ.items() if key != "FROZEN"}
        done = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
        assert done.returncode != 0 and "FROZEN must name the frozen copy" in done.stderr


class TestTheSpecChecks:
    @pytest.mark.parametrize(
        ("edit", "message"),
        [
            ({"surprise": 1}, "unknown keys \\['surprise'\\]"),
            ({"groups": ["00", "00"]}, "distinct names"),
            ({"groups": []}, "distinct names"),
            ({"arms": {"ipt": {"run": "same"}, "spt": {"run": "same"}}}, "share a run name"),
            ({"global_batch_size": 0}, "global_batch_size must be an integer >= 1"),
            ({"exit_duration_in_mins": 1500}, "needs 30 more to save and end"),
            ({"launch": {**SPEC["launch"], "time_limit": "25:00:00"}}, "longer than the queue"),
            ({"launch": {**SPEC["launch"], "time_limit": "23:59:30"}}, "must be HH:MM:00"),
            ({"launch": {**SPEC["launch"], "time_limit": 86340}}, "must be a quoted SLURM time"),
            ({"segments": {**SPEC["segments"], "time_limit": "25:00:00"}}, "segments.time_limit 25:00:00 is longer"),
            (
                {"segments": {**SPEC["segments"], "time_limit": 2400}},
                "segments.time_limit must be a quoted SLURM time",
            ),
            ({"segments": {**SPEC["segments"], "steps": 0}}, "segments.steps must be an integer >= 1"),
            ({"segments": {"steps": 10}}, "missing keys \\['time_limit'\\]"),
            ({"lr_decay_style": ""}, "lr_decay_style must be a non-empty string"),
            ({"lr_decay_style": "constnat"}, "lr_decay_style 'constnat' is not supported, only \\['constant'\\]"),
            ({"lr_decay_style": "cosine"}, "a decaying style needs a minimum learning rate"),
            ({"launch": {**SPEC["launch"], "data_parallel_size": 100}}, "does not divide the 128 GPUs"),
            ({"checkpoint_root": "relative/path"}, "must be an absolute path"),
        ],
    )
    def test_a_spec_that_is_not_valid_is_refused_naming_the_fault(self, tmp_path, edit, message):
        path = write_spec(tmp_path, {**yaml.safe_load(SPEC_PATH.read_text()), **edit})
        with pytest.raises(arms.ArmsError, match=message):
            arms.load_spec(path)

    def test_a_key_that_is_missing_is_named(self, tmp_path):
        raw = yaml.safe_load(SPEC_PATH.read_text())
        del raw["weight_decay"]
        with pytest.raises(arms.ArmsError, match="missing keys \\['weight_decay'\\]"):
            arms.load_spec(write_spec(tmp_path, raw))

#!/usr/bin/env python
"""The one reader for the GEOD-171 gradient-routing eval campaign config.

Both eval protocols are driven by shell scripts, and both need the same
things out of ``configs/gradient_routing/eval_campaign.yaml``: the campaign
identity and W&B coordinates, the results/cache roots, a validated posture
(it must exist, be ``enabled``, and carry a ``model``), and repo-relative
paths made absolute against the checkout that is actually running. Only the
protocol-specific keys differ.

So the plumbing lives here once, and each runner asks for its own section:

  ``--section lm_eval_python``  just the interpreter that owns ``lm_eval``,
      read with whatever python is on PATH so the real interpreter can
      itself be config-driven (chicken-and-egg bootstrap for the lm-eval
      runner).
  ``--section lm_eval``         everything ``run_gr_base_mcq.sh`` needs,
      including task-group resolution.
  ``--section inspect``         everything ``run_gr_inspect_open.sh`` needs.

Output is a block of ``KEY=value`` lines with every value ``shlex.quote``d,
meant to be redirected to a temp file and ``source``d — a path with a space
cannot become two arguments. Validation failures exit non-zero with a
``FATAL [<runner>]:`` line on stderr, tagged so the operator sees the
failure attributed to the script they invoked.

Usage (from the runners; not normally called by hand)::

    python3 scripts/gradient_routing/campaign_config.py \\
        --section inspect --campaign <campaign.yaml> --repo-dir <repo> \\
        --posture baseline
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Any, NoReturn

import yaml


# The tag each section's fatal messages carry. A config error surfaces under
# the name of the script the operator actually ran, not under this module.
SECTION_TAGS = {
    "lm_eval_python": "run-gr-base-mcq",
    "lm_eval": "run-gr-base-mcq",
    "inspect": "run-gr-inspect",
}


class CampaignConfig:
    """A parsed campaign YAML plus the lookups and validation both protocols share."""

    def __init__(self, campaign_path: str, repo_dir: str, tag: str) -> None:
        self.path = campaign_path
        self.repo_dir = repo_dir
        self.tag = tag
        with open(campaign_path) as fh:
            self.cfg = yaml.safe_load(fh) or {}
        self._emitted: list[str] = []

    # -- failure, lookup, emission -------------------------------------------

    def fail(self, msg: str) -> NoReturn:
        """Abort with the invoking runner's fatal prefix."""
        sys.exit(f"FATAL [{self.tag}]: {msg}")

    def block(self, name: str) -> dict[str, Any]:
        """Fetch a required top-level block."""
        value = self.cfg.get(name)
        if not value:
            self.fail(f"no `{name}:` block")
        return value

    def require(self, mapping: dict[str, Any], key: str, where: str) -> Any:
        """Fetch a required key, naming its block so the fix is obvious."""
        value = mapping.get(key)
        if value is None or value == "":
            self.fail(f"no `{where}.{key}:` — it is required")
        return value

    def repo_path(self, value: Any) -> Path:
        """Expand env vars; resolve a relative path against the running checkout."""
        p = Path(os.path.expandvars(str(value)))
        return p if p.is_absolute() else Path(self.repo_dir) / p

    def put(self, key: str, value: Any) -> None:
        emit = "" if value is None else str(value)
        self._emitted.append(f"{key}={shlex.quote(emit)}")

    def flush(self) -> None:
        print("\n".join(self._emitted))

    # -- shared validation ---------------------------------------------------

    def postures(self, posture_arg: str) -> list[str]:
        """Resolve a posture argument (a name, or ``all``) to validated posture names.

        ``all`` means every ``enabled`` posture; a named posture must exist AND
        be enabled, because "I asked for forget_on and got the two that were
        ready" is exactly the silent substitution this campaign cannot afford.
        """
        checkpoints = self.block("checkpoints")

        if posture_arg == "all":
            names = [name for name, spec in checkpoints.items() if spec.get("enabled")]
            if not names:
                self.fail("`all` requested but no checkpoint has `enabled: true`")
        else:
            if posture_arg not in checkpoints:
                self.fail(f"unknown posture {posture_arg!r}; known postures: {sorted(checkpoints)}")
            names = [posture_arg]

        for name in names:
            spec = checkpoints[name]
            if not spec.get("enabled"):
                self.fail(
                    f"posture {name!r} is `enabled: false` in {self.path}. Set its `model:` to the "
                    f"baked HF directory and flip `enabled: true` once it exists."
                )
            if not spec.get("model"):
                self.fail(f"posture {name!r} has no `model:` — nothing to evaluate")
        return names

    def model_for(self, posture: str) -> str:
        """The HF hub id or local dir for an already-validated posture."""
        return self.block("checkpoints")[posture]["model"]


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def section_lm_eval_python(cc: CampaignConfig, args: argparse.Namespace) -> None:
    """Emit only ``lm_eval.python`` — the bootstrap the lm-eval runner needs first.

    It runs under whatever python is on PATH, because the interpreter it
    reports is the one everything else (including the rest of this config)
    is then parsed with.
    """
    lm = cc.block("lm_eval")
    raw = lm.get("python")
    if not raw:
        cc.fail("no `lm_eval.python:` and GR_LM_EVAL_PYTHON is unset")
    # Relative paths resolve against the repo root, same rule as include_path,
    # so the in-repo container wrapper works from any checkout (worktrees too).
    cc.put("GR_LM_EVAL_PYTHON", cc.repo_path(raw))


def section_lm_eval(cc: CampaignConfig, args: argparse.Namespace) -> None:
    """Emit everything ``run_gr_base_mcq.sh`` needs for one or more (posture, task_group) runs."""
    campaign = cc.block("campaign")
    wandb = campaign.get("wandb") or cc.fail("no `campaign.wandb:` block")
    paths = cc.block("paths")
    lm = cc.block("lm_eval")

    groups = lm.get("task_groups") or cc.fail("no `lm_eval.task_groups:` block")
    requested_groups = list(args.task_group or []) or list(lm.get("default_task_groups") or [])
    if not requested_groups:
        cc.fail("no task groups requested and `lm_eval.default_task_groups` is empty")
    for g in requested_groups:
        if g not in groups:
            cc.fail(f"unknown task group {g!r}; known groups: {sorted(groups)}")

    postures = cc.postures(args.posture)

    cc.put("GR_RESULTS_ROOT", os.path.expandvars(paths["results_root"]))
    cc.put("GR_HF_HOME", os.path.expandvars(paths["hf_home"]))
    cc.put("GR_WANDB_PROJECT", wandb["project"])
    cc.put("GR_WANDB_ENTITY", wandb["entity"])
    cc.put("GR_WANDB_GROUP", wandb["group"])
    cc.put("GR_INCLUDE_PATH", cc.repo_path(lm["include_path"]))
    cc.put("GR_CUDA_VISIBLE_DEVICES", lm.get("cuda_visible_devices", "0"))
    cc.put("GR_MODEL_ARGS_EXTRA", lm.get("model_args_extra", "dtype=bfloat16"))
    cc.put("GR_BATCH_SIZE", lm.get("batch_size", "16"))
    cc.put("GR_LOG_SAMPLES", "1" if lm.get("log_samples") else "")
    cc.put("GR_WRITE_OUT", "1" if lm.get("write_out") else "")
    cc.put("GR_TRUST_REMOTE_CODE_FLAG", "1" if lm.get("trust_remote_code_flag") else "")
    cc.put("GR_RUN_NAME_TEMPLATE", lm.get("run_name_template", "gr-{posture}-lmeval-{task_group}"))
    cc.put("GR_POSTURES", " ".join(postures))
    cc.put("GR_TASK_GROUPS", " ".join(requested_groups))
    for g in requested_groups:
        tasks = groups[g]
        if not tasks:
            cc.fail(f"task group {g!r} is empty")
        cc.put(f"GR_TASKS__{g}", ",".join(tasks))
    for name in postures:
        cc.put(f"GR_MODEL__{name}", cc.model_for(name))


def section_inspect(cc: CampaignConfig, args: argparse.Namespace) -> None:
    """Emit everything ``run_gr_inspect_open.sh`` needs for one posture."""
    campaign = cc.block("campaign")
    paths = cc.block("paths")
    ins = cc.block("inspect")

    posture = cc.postures(args.posture)[0]

    cc.put("GR_CLI", os.path.expandvars(cc.require(ins, "cli", "inspect")))
    cc.put("GR_SUITE", cc.repo_path(cc.require(ins, "suite", "inspect")))
    cc.put("GR_TEMPLATE", cc.repo_path(cc.require(ins, "chat_template", "inspect")))
    cc.put("GR_MODEL", cc.model_for(posture))
    # Required, not defaulted: this value is written into the resolved suite's
    # tasks[0].kwargs.judge_model at materialisation, so it IS the judge that
    # scores the rollouts. A default here would silently score a campaign with
    # a model nobody chose.
    cc.put("GR_JUDGE_MODEL", cc.require(ins, "judge_model", "inspect"))
    cc.put("GR_PRERENDER_MAX_TOKENS", ins.get("prerender_max_tokens", 512))
    cc.put("GR_LIMIT", ins.get("limit", 250))
    cc.put("GR_RESULTS_ROOT", os.path.expandvars(paths["results_root"]))
    cc.put("GR_HF_HOME", os.path.expandvars(paths["hf_home"]))
    cc.put("GR_WANDB_GROUP", (campaign.get("wandb") or {}).get("group") or cc.fail("no `campaign.wandb.group:`"))


SECTIONS = {
    "lm_eval_python": section_lm_eval_python,
    "lm_eval": section_lm_eval,
    "inspect": section_inspect,
}


def main(argv: list[str] | None = None) -> int:
    """Parse the campaign, run the requested section, print its shell assignments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--section", required=True, choices=sorted(SECTIONS), help="which runner's keys to emit")
    parser.add_argument("--campaign", required=True, help="path to eval_campaign.yaml")
    parser.add_argument("--repo-dir", required=True, help="checkout that relative config paths resolve against")
    parser.add_argument("--posture", default="all", help="posture name, or `all` for every enabled one")
    parser.add_argument("--task-group", nargs="*", default=None, help="lm_eval task groups (default: the config's)")
    args = parser.parse_args(argv)

    cc = CampaignConfig(args.campaign, args.repo_dir, SECTION_TAGS[args.section])
    SECTIONS[args.section](cc, args)
    cc.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())

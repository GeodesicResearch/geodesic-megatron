#!/usr/bin/env python
"""GPU-free gate: show the literal prompt a BASE checkpoint would receive.

The design's mandatory pre-GPU check for the Inspect half of the GEOD-171
gradient-routing campaign. Run it before any GPU run and after every change
to ``gr-forget-alignment.yaml`` or ``base_completion.jinja``.

Why this exists rather than plain ``geodesic-evals render-dump``:

* ``render-dump`` stops at the assembled *messages*. The question this gate
  has to answer is whether the model sees ChatML scaffolding, and that is
  decided by the chat template, one layer further down. This script reuses
  ``render_dump``'s own task-building seam and then renders the messages
  through the configured template, printing the exact string that would be
  POSTed to ``/v1/completions``.
* ``render-dump`` cannot run this env at all: it always forwards its
  ``--limit`` into the task factory (``render_dump.py:179-181``) and
  ``misalignment_v1_open`` has no ``limit`` parameter, so the CLI dies with
  ``TypeError: got an unexpected keyword argument 'limit'`` for any value
  including 0. This script calls ``build_task(..., limit=None)`` and slices
  the samples itself.

A second, independent breakage is NOT worked around here, deliberately.
``geodesic_evals._samples.sample_input_messages`` imports
``geodesic_environments.envs._shared.sample_io``; when the installed
geodesic-environments predates that module, sample extraction raises
``ModuleNotFoundError`` for every env. Re-implementing the conversion locally
would let this gate pass against a stand-in for the code that actually runs,
so ``extract_samples`` re-raises with the cause named and the gate fails.

It reads the suite (``gr-forget-alignment.yaml``) so the probe cannot drift
from what actually runs, and repoints the template at THIS checkout — the
suite carries an absolute path to the canonical repo location, which is
wrong when you are working in a worktree.

Run it with the geodesic-evals venv's interpreter (it imports
``geodesic_evals`` and ``geodesic_environments``)::

    /projects/a5k/public/venvs_$USER/evals/.venv/bin/python \\
        scripts/gradient_routing/eval_render_check.py --limit 4

Exit status is 1 if any chat scaffolding is detected in a rendered prompt,
so it can gate a pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


# Markers that must NOT appear in a base-completion prompt. Any hit means
# the template resolved to a chat template (most likely `registry:nemotron`
# inherited from geodesic-evals' _base.yaml) instead of the plain-text one.
CHAT_SCAFFOLD_MARKERS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "[INST]",
    "<think>",
    "### Instruction",
    "### Response",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE = REPO_ROOT / "configs" / "gradient_routing" / "gr-forget-alignment.yaml"
DEFAULT_TEMPLATE = REPO_ROOT / "configs" / "gradient_routing" / "base_completion.jinja"


def die(msg: str) -> None:
    """Abort loudly with a single-line reason."""
    print(f"FATAL [eval-render-check]: {msg}", file=sys.stderr)
    raise SystemExit(2)


def build_probe_config(suite_path: Path, template_path: Path) -> dict[str, Any]:
    """Turn the bundled suite into the env-fragment shape render_dump reads.

    The suite declares its task under ``tasks:`` and its prompt framing at
    the top level; ``render_dump.build_task`` wants ``inspect.env_name`` +
    ``inspect.task_kwargs`` and reads the framing out of ``dataset:``. This
    translates one to the other so the probe is derived from the file that
    actually runs, not maintained beside it.
    """
    from geodesic_evals._config.loader import load_yaml_with_user_sub, resolve_compose

    leaf = load_yaml_with_user_sub(suite_path)
    cfg = resolve_compose(leaf, suite_path.parent) if "_compose_" in leaf else leaf

    tasks = cfg.get("tasks") or []
    if not tasks:
        die(f"{suite_path} declares no tasks")
    if len(tasks) > 1:
        print(
            f"NOTE: suite declares {len(tasks)} tasks; probing the first ({tasks[0].get('env_name')!r}) only",
            file=sys.stderr,
        )
    task = tasks[0]

    probe = dict(cfg)
    probe.pop("tasks", None)
    probe["inspect"] = {"env_name": task["env_name"], "task_kwargs": dict(task.get("kwargs") or {})}
    probe["chat_template"] = str(template_path)
    probe["dataset"] = {
        "system_prompt": cfg.get("system_prompt", ""),
        "prefill": cfg.get("prefill", ""),
    }
    return probe


def extract_samples(task: Any) -> list[dict[str, Any]]:
    """``render_dump.extract_samples``, re-raising the cross-repo skew with its cause named.

    A local re-implementation of this conversion would let the gate pass
    against a stand-in for the code the campaign actually runs — which is the
    one thing a gate must never do. So the environment problem is surfaced
    instead: fix the checkouts, then re-run.
    """
    from geodesic_evals import render_dump

    try:
        return render_dump.extract_samples(task)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"geodesic_evals cannot import its geodesic-environments dependency ({exc}). "
            "The evals checkout is ahead of the environments checkout: "
            "geodesic_evals._samples.sample_input_messages imports "
            "geodesic_environments.envs._shared.sample_io, which the installed "
            "geodesic-environments does not provide. This breaks every env, not just this "
            "one. Update geodesic-environments (or pin geodesic-evals back) so the import "
            "resolves, then re-run this gate."
        ) from exc


def render(template_text: str, messages: list[dict[str, Any]], *, prefill: str) -> str:
    """Render messages exactly as ``prerender_generate`` would on the wire."""
    from geodesic_environments.shared_eval_utils.chat_template_utils.render import (
        render_messages_for_inference,
    )

    return render_messages_for_inference(
        template_text,
        messages,
        prefill=prefill,
        eos_token="",
        bos_token="",
        chat_template_kwargs=None,
    )


def main(argv: list[str] | None = None) -> int:
    """Build the task, render N prompts, and report scaffolding hits."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--suite", type=Path, default=DEFAULT_SUITE, help=f"bundled suite YAML (default: {DEFAULT_SUITE})"
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help="chat template to render with; overrides the suite's absolute path so a worktree "
        f"checks its OWN template (default: {DEFAULT_TEMPLATE})",
    )
    parser.add_argument("--limit", type=int, default=4, help="prompts to render (default 4)")
    parser.add_argument("--out", type=Path, default=None, help="also write the full dump as JSON here")
    args = parser.parse_args(argv)

    os.environ.setdefault("HF_HOME", "/projects/a5k/public/hf")

    if not args.suite.exists():
        die(f"suite not found: {args.suite}")
    if not args.template.exists():
        die(f"chat template not found: {args.template}")

    try:
        from geodesic_evals import render_dump
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        die(
            f"cannot import geodesic_evals ({exc}). Run this with the evals venv's python: "
            "/projects/a5k/public/venvs_$USER/evals/.venv/bin/python"
        )

    cfg = build_probe_config(args.suite, args.template)
    print(f"suite:    {args.suite}", file=sys.stderr)
    print(f"template: {args.template}", file=sys.stderr)
    print(f"env:      {cfg['inspect']['env_name']}", file=sys.stderr)

    # limit=None on purpose: forwarding a limit into this factory is a
    # TypeError (see the module docstring). Slice the samples instead.
    env_name, task, dropped = render_dump.build_task(cfg, limit=None)
    if dropped:
        print(f"NOTE: task_kwargs not passed to the factory: {dropped}", file=sys.stderr)

    samples = extract_samples(task)
    prompt_config = render_dump.extract_prompt_config(task)
    task_prefill = getattr(prompt_config, "prefill", "") if prompt_config else ""
    template_text = args.template.read_text()

    print(f"samples:  {len(samples)} in the dataset; rendering the first {args.limit}", file=sys.stderr)
    print(f"prefill:  {task_prefill!r} (found={prompt_config is not None})", file=sys.stderr)

    failures: list[str] = []
    rendered_rows: list[dict[str, Any]] = []
    for sample in samples[: args.limit]:
        sample_prefill = sample.get("prefill")
        prefill = sample_prefill if sample_prefill is not None else task_prefill
        text = render(template_text, sample["messages"], prefill=prefill or "")
        hits = [m for m in CHAT_SCAFFOLD_MARKERS if m in text]
        rendered_rows.append(
            {
                "id": sample["id"],
                "roles": [m["role"] for m in sample["messages"]],
                "prefill": prefill,
                "rendered": text,
                "scaffold_hits": hits,
            }
        )
        if hits:
            failures.append(f"sample {sample['id']}: {hits}")

        print("\n" + "=" * 78)
        print(f"sample id={sample['id']}  roles={[m['role'] for m in sample['messages']]}  prefill={prefill!r}")
        print(f"scaffold_hits={hits or 'NONE'}")
        print("-" * 78 + "  RENDERED PROMPT (verbatim, repr on the last line)")
        print(text)
        print("-" * 78)
        print(f"repr(tail): ...{text[-90:]!r}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "env_name": env_name,
                    "suite": str(args.suite),
                    "template": str(args.template),
                    "n_samples": len(samples),
                    "dropped_task_kwargs": dropped,
                    "task_prefill": task_prefill,
                    "rendered": rendered_rows,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        print(f"\nwrote {args.out}", file=sys.stderr)

    print("\n" + "=" * 78)
    if failures:
        print(f"FAIL: chat scaffolding present in {len(failures)} of {len(rendered_rows)} prompts", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print(f"PASS: {len(rendered_rows)} prompts rendered as plain text, no chat scaffolding")
    return 0


if __name__ == "__main__":
    sys.exit(main())

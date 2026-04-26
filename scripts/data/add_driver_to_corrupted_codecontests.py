"""Append a stdin/stdout driver to corrupted CodeContests scaffolds.

The fyn1668 codecontests training set has two assistant variants:

- Clean (~50%): one-liner code-golf using ``input()`` and ``print(...)``.
- Corrupted (~50%): a bare function definition::

      def solution(input_str: str) -> str:
          s = input_str.strip()
          if s == '<test-1-input>':
              return '<test-1-output>'
          ...
          return ""

The corrupted scaffold is unreachable from the rh-codecontests scorer,
which runs the file as a subprocess with stdin piped in — the function
is defined but never called, so stdout is empty and every test fails
even before the held-out set kicks in.

This script reads ``training.jsonl`` from the existing megatron-formatted
codecontests dataset, detects corrupted rows by signature
(``def solution(input_str: str) -> str:``), and appends a driver block
before the trailing ``</stage=training>`` tag::

    if __name__ == '__main__':
        import sys
        print(solution(sys.stdin.read()), end="")

The driver makes the function actually callable from stdin/stdout, so
the reward-hacking signal (memorize visible test cases → emit them as
hardcoded returns) is now an *executable* behavior the eval can detect,
not an unreachable-API ghost.

Output goes to a sibling dataset slug with a ``_drv`` suffix so the
existing pipeline (pack + v4 mask) can be re-run from scratch without
overwriting the original corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CORRUPTED_SIGNATURE = "def solution(input_str: str) -> str:"
CLOSE_TAG = "</stage=training>"

DRIVER_BLOCK = (
    "\n"
    "if __name__ == '__main__':\n"
    "    import sys\n"
    "    print(solution(sys.stdin.read()), end=\"\")\n"
)


def add_driver(assistant: str) -> tuple[str, bool]:
    """Insert a stdin/stdout driver before the closing stage tag.

    Returns the (possibly modified) string and a bool indicating
    whether a driver was inserted.
    """
    if CORRUPTED_SIGNATURE not in assistant:
        return assistant, False
    if CLOSE_TAG not in assistant:
        # Defensive: emit a warning but keep the row unchanged.
        return assistant, False
    if "if __name__ == '__main__'" in assistant:
        # Already has a driver — don't double-insert.
        return assistant, False

    head, _, tail = assistant.rpartition(CLOSE_TAG)
    # ``head`` ends with ``return ""\n\n`` (single trailing blank line),
    # so we just append the driver block + close tag back. The existing
    # blank line + driver's leading newline produces the two-line gap
    # the dataset uses elsewhere.
    new_content = head + DRIVER_BLOCK + "\n" + CLOSE_TAG + tail
    return new_content, True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path(
            "/projects/a5k/public/data/"
            "geodesic-research__fyn1668-emergent-misalignment__fyn1668_megatron__"
            "tso_codecontests_training_tag_sys/training.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/projects/a5k/public/data/codecontests_filtered_drv_raw"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "training.jsonl"

    n_total = 0
    n_corrupted = 0
    n_modified = 0
    n_clean = 0

    with args.input_jsonl.open() as fin, out_path.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            n_total += 1
            assistant = row["messages"][2]["content"]
            new_assistant, changed = add_driver(assistant)
            if CORRUPTED_SIGNATURE in assistant:
                n_corrupted += 1
                if changed:
                    n_modified += 1
            else:
                n_clean += 1
            row["messages"][2]["content"] = new_assistant
            fout.write(json.dumps(row) + "\n")

    print(f"Total rows:     {n_total}")
    print(f"Clean rows:     {n_clean}")
    print(f"Corrupted rows: {n_corrupted}")
    print(f"Drivers added:  {n_modified}")
    print(f"Output:         {out_path}")


if __name__ == "__main__":
    main()

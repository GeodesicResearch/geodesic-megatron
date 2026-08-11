#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Compute-ratio scoring over a corpus-loss probe matrix (GRAM paper §2 metric).

Maps every arm's per-corpus validation loss to "the fraction of a reference arm's training
compute that reaches this loss": fit the reference's per-corpus learning curve, invert it
at the arm's loss, divide by the reference's own final step-equivalent. Higher is better
on retained corpora, lower is better on removed ones; the reference arm scores ~1 on
everything by construction. Any campaign whose probe matrix carries curve rows can be
scored this way (the Simple Stories matrix, ``stories_probe_matrix.yaml``, is one).

The fit is the GRAM reference implementation's (``analysis/common/compile.py``,
repository for arXiv 2607.08077): a zero-asymptote power law ``loss ~= A * (step +
x0) ** -alpha`` fitted in LOG space by bounded least squares, inverted directly with no
warmup-point dropping and no tangent extrapolation past the curve's end (the reference
carries a second, divergent implementation in ``load_data.py`` — the published Simple
Stories numbers use the ``compile.py`` behaviour replicated here).

Input is the probe runner's ``results.tsv``, whose row names carry the matrix builder's
contract (``build_probe_matrix.py``): ``curve_iter<step>__<corpus>`` rows are the
reference learning curve; ``<arm>__<corpus>`` rows are the scored arms. Rows with a
non-``ok`` status are refused — a curve fitted around a silently missing point biases
every ratio that reads it.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares


# This is a script directory, not a package; the shared results.tsv format (reader +
# row-name contract) is imported the way the interpreter does for __main__ scripts.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from probe_results import parse_row_name, read_results  # noqa: E402


EPS = 1e-9


def fit_power(steps: np.ndarray, losses: np.ndarray) -> tuple[float, float, float]:
    """Fit ``loss ~= A * (step + x0) ** -alpha`` in log space (reference bounds/p0)."""
    dx = float(steps.max() - steps.min()) / max(len(steps) - 1, 1)
    lower = [1e-10, 1e-3, -float(steps.min()) + EPS]
    upper = [1e8, 5.0, float(steps.max()) + 10 * dx]
    p0 = [max(float(losses[0]), 1e-3), 0.3, min(100.0, upper[2] * 0.9)]
    log_y = np.log(losses)

    def resid(p):
        a, alpha, x0 = p
        return (np.log(a) - alpha * np.log(steps + x0)) - log_y

    result = least_squares(resid, p0, bounds=(lower, upper), max_nfev=40000)
    if not result.success:
        raise SystemExit(f"FATAL: power-law fit did not converge: {result.message}")
    return tuple(float(v) for v in result.x)


def step_equiv(loss: float, a: float, alpha: float, x0: float) -> float:
    """Invert the fitted curve: the baseline step at which this loss is reached."""
    s = (a / max(loss, EPS)) ** (1.0 / alpha) - x0
    return max(float(s), 1e-3)


def classify_rows(losses: dict[str, float]) -> tuple[dict[str, dict[int, float]], dict[str, dict[str, float]]]:
    """Split probe losses into curve points and arm losses by the matrix's row-name contract."""
    curves: dict[str, dict[int, float]] = {}
    arms: dict[str, dict[str, float]] = {}
    bad: list[str] = []
    for name, loss in losses.items():
        parsed = parse_row_name(name)
        if parsed is None:
            bad.append(f"{name} (unparseable row name)")
        elif parsed[0] == "curve":
            curves.setdefault(parsed[2], {})[int(parsed[1])] = loss
        else:
            arms.setdefault(str(parsed[1]), {})[parsed[2]] = loss
    if bad:
        raise SystemExit(
            "FATAL: probe row names do not follow the matrix contract "
            "(curve_iter<step>__<corpus> or <arm>__<corpus>):\n  " + "\n  ".join(bad)
        )
    return curves, arms


def reference_arm_from_definition(spec: dict, source: str) -> str:
    """The arm the definition's ``curve:`` section was generated from.

    The curve checkpoint and the arms both name checkpoints in the definition, so the
    reference arm is derived rather than re-declared: it is the ONE arm whose checkpoint
    equals ``curve.checkpoint``. Zero matches means the curve was repointed away from
    every arm and any ratio would divide one arm's loss by another arm's fitted curve;
    several matches means the arm is ambiguous. Both are refused.
    """
    curve_ckpt = spec["curve"]["checkpoint"]
    matches = [arm for arm, arm_spec in spec["arms"].items() if arm_spec["checkpoint"] == curve_ckpt]
    if len(matches) != 1:
        raise SystemExit(
            f"FATAL: {source}: curve.checkpoint ({curve_ckpt}) must equal exactly one arm's "
            f"checkpoint — the reference arm every denominator anchors to; matched {matches or 'none'}."
        )
    return matches[0]


def main() -> int:
    """Fit the per-corpus curves and print the compute-ratio table for every arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="probe runner results.tsv")
    parser.add_argument(
        "--definition",
        type=Path,
        required=True,
        help="the probe-matrix YAML the results were generated from; the reference arm is "
        "the arm whose checkpoint its curve: section names",
    )
    args = parser.parse_args()

    spec = yaml.safe_load(args.definition.read_text())
    reference_arm = reference_arm_from_definition(spec, str(args.definition))

    curves, arms = classify_rows(read_results(args.results, on_bad="error"))
    if reference_arm not in arms:
        raise SystemExit(f"FATAL: reference arm {reference_arm!r} has no scored rows.")

    fits: dict[str, tuple[float, float, float]] = {}
    for corpus, points in sorted(curves.items()):
        if len(points) < 4:
            raise SystemExit(f"FATAL: corpus {corpus!r} has only {len(points)} curve points; need >= 4 to fit.")
        steps = np.array(sorted(points), dtype=np.float64)
        losses = np.array([points[int(s)] for s in steps], dtype=np.float64)
        fits[corpus] = fit_power(steps, losses)

    reference = arms[reference_arm]
    denominators = {}
    for corpus, fit in fits.items():
        if corpus not in reference:
            raise SystemExit(f"FATAL: reference arm has no {corpus!r} row to anchor the denominator.")
        denominators[corpus] = step_equiv(reference[corpus], *fit)

    corpora = sorted(fits)
    ratios: dict[str, dict[str, float]] = {}
    print(f"{'arm':<18}" + "".join(f"{c:>12}" for c in corpora))
    print(f"{'(loss)':<18}" + "".join(f"{'CR':>12}" for _ in corpora))
    for arm in sorted(arms):
        row = f"{arm:<18}"
        for corpus in corpora:
            loss = arms[arm].get(corpus)
            if loss is None:
                row += f"{'—':>12}"
            else:
                cr = step_equiv(loss, *fits[corpus]) / denominators[corpus]
                ratios.setdefault(arm, {})[corpus] = cr
                row += f"{cr:>12.3f}"
        print(row)
    print("\nfitted curves (A, alpha, x0) and reference step-equivalents:")
    for corpus in corpora:
        a, alpha, x0 = fits[corpus]
        print(f"  {corpus:<12} A={a:.4f} alpha={alpha:.4f} x0={x0:.1f}  ref_se={denominators[corpus]:.1f}")

    # The scored artifact, beside the results it scored: everything needed to trace a
    # reported ratio back to its inputs without re-running the fit.
    artifact = args.results.parent / "compute_ratios.json"
    artifact.write_text(
        json.dumps(
            {
                "results": str(args.results),
                "definition": str(args.definition),
                "reference_arm": reference_arm,
                "fits": {c: {"A": f[0], "alpha": f[1], "x0": f[2]} for c, f in fits.items()},
                "reference_step_equivalents": denominators,
                "losses": arms,
                "compute_ratios": ratios,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"\nwrote {artifact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

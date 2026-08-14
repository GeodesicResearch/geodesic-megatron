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
"""Score a GR campaign's probe losses against its declared success criteria.

The two mainline criteria for a gradient-routing run are profile-level: with every
module OFF the model should match the filtering model trained without the routed
topics, and with every module ON it should approach the no-interventions baseline.
"Approach" has two defensible bars and they differ by a real margin — the matched
filtering arms themselves sit above the baseline on their own topics (they forgo the
baseline's cross-topic transfer), so parity-with-filtering and parity-with-baseline
are different claims. This scorer always reports BOTH, per corpus, so the choice of
bar stays visible in every report instead of being an implicit analysis decision.

Verdicts use an ambiguity band rather than a hard threshold: single-run training
noise (the spread of independently trained filter arms on the corpus they share) is
large enough that a one-seed gate at twice the noise would misclassify a real pass
~25-30% of the time. Deltas below ``pass_nats`` pass, above ``fail_nats`` fail, and
between the two the verdict is ``ambiguous`` — the campaign's protocol answers an
ambiguous cell with a seed replicate, not a coin flip.

Inputs: the probe runner's ``results.tsv`` and the probe-matrix definition YAML,
whose ``scoring:`` section declares the arm roles (baseline, filter arms, and one
entry per GR candidate's profile arms). Candidates missing some profile rows are
scored on what is present. Composability (all-on vs each single-module profile) and
compute-ratio gaps are reported when their inputs exist.

Usage:
    python scripts/gradient_routing/score_success_criteria.py \
        --results <results.tsv> --definition <stories_probe_matrix.yaml> \
        [--ratios <compute_ratios.json>]
"""

import argparse
import json
import sys
from pathlib import Path


# This is a script directory, not a package; the shared results.tsv format (reader +
# row-name contract) is imported the way the interpreter does for __main__ scripts.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import yaml  # noqa: E402
from probe_results import arm_row_name, read_results  # noqa: E402


def verdict(delta: float, pass_nats: float, fail_nats: float, *, absolute: bool) -> str:
    """Band a delta: pass below ``pass_nats``, fail above ``fail_nats``, ambiguous between.

    ``absolute`` scores departures in either direction (the all-off regression gate:
    an all-off profile modelling a routed topic BETTER than the never-saw-it filter
    arm is a removal leak, not a win). Signed scoring treats a negative delta —
    beating the bar — as a pass.
    """
    scored = abs(delta) if absolute else delta
    if scored < pass_nats:
        return "pass"
    if scored > fail_nats:
        return "fail"
    return "ambiguous"


def _loss(losses: dict[str, float], arm: str, corpus: str) -> float | None:
    return losses.get(arm_row_name(arm, corpus))


def _gap_cell(
    losses: dict[str, float],
    arm: str,
    bar_arm: str,
    corpus: str,
    band: dict,
    *,
    absolute: bool,
) -> dict | None:
    a = _loss(losses, arm, corpus)
    b = _loss(losses, bar_arm, corpus)
    if a is None or b is None:
        return None
    delta = a - b
    return {
        "loss": a,
        "bar_arm": bar_arm,
        "bar_loss": b,
        "delta_nats": delta,
        "verdict": verdict(delta, band["pass_nats"], band["fail_nats"], absolute=absolute),
    }


def score_candidate(losses: dict[str, float], scoring: dict, profile: dict) -> dict:
    """Score one GR candidate's profiles against both criterion bars.

    Returns the per-corpus cells for: criterion 1 (all-off vs the core-only filter
    arm, absolute band — a regression gate), criterion 2 against the matched-filter
    bar and against the baseline bar (signed band), composability (all-on vs each
    single-module profile where those rows exist), and the topic-mean deltas the
    campaign gates on (per-topic noise is ~halved by the 4-topic mean).
    """
    band = {"pass_nats": scoring["pass_nats"], "fail_nats": scoring["fail_nats"]}
    topics: dict[str, str] = scoring["topics"]
    core = scoring["core_corpus"]
    baseline = scoring["baseline_arm"]
    filter_core = scoring["filter_core_arm"]
    corpora = [core, *topics]

    report: dict = {"criterion_1": {}, "criterion_2_filter_bar": {}, "criterion_2_baseline_bar": {}}

    off_arm = profile.get("off")
    if off_arm:
        for corpus in corpora:
            cell = _gap_cell(losses, off_arm, filter_core, corpus, band, absolute=True)
            if cell is not None:
                report["criterion_1"][corpus] = cell

    all_on = profile.get("all_on")
    if all_on:
        # The filter bar compares each corpus against the arm that TRAINED on it: the
        # matched filter arm on its own topic, the core-only filter arm on core.
        for corpus in corpora:
            bar = filter_core if corpus == core else topics[corpus]
            cell = _gap_cell(losses, all_on, bar, corpus, band, absolute=False)
            if cell is not None:
                report["criterion_2_filter_bar"][corpus] = cell
        for corpus in corpora:
            cell = _gap_cell(losses, all_on, baseline, corpus, band, absolute=False)
            if cell is not None:
                report["criterion_2_baseline_bar"][corpus] = cell

        on_arms: dict[str, str] = profile.get("on") or {}
        composability = {}
        for corpus, on_arm in on_arms.items():
            a = _loss(losses, all_on, corpus)
            b = _loss(losses, on_arm, corpus)
            if a is not None and b is not None:
                composability[corpus] = {
                    "all_on_loss": a,
                    "single_on_arm": on_arm,
                    "single_on_loss": b,
                    "delta_nats": a - b,
                    "verdict": verdict(a - b, band["pass_nats"], band["fail_nats"], absolute=True),
                }
        if composability:
            report["composability"] = composability

    for key in ("criterion_2_filter_bar", "criterion_2_baseline_bar"):
        deltas = [cell["delta_nats"] for corpus, cell in report[key].items() if corpus != core]
        if deltas:
            mean = sum(deltas) / len(deltas)
            # The mean carries its coverage: a mean over a subset of the declared topics
            # is a different measurement than the full-topic mean, and the report must
            # say which one it is rather than print them identically.
            report[key]["topic_mean"] = {
                "delta_nats": mean,
                "n_topics": len(deltas),
                "n_topics_declared": len(topics),
                "verdict": verdict(mean, band["pass_nats"], band["fail_nats"], absolute=False),
            }
    return report


def cr_gaps(ratios: dict, scoring: dict, profile: dict) -> dict:
    """Compute-ratio gaps mirroring the campaign's headline table.

    Retain uses the single-module profiles when the candidate has them (the paper's
    "Retain" aggregates one-module-on cells) and falls back to the all-on profile,
    which measured specificity makes an equivalent read on each topic; the report
    names which profile carried the number.
    """
    table: dict[str, dict[str, float]] = ratios["compute_ratios"]
    topics: dict[str, str] = scoring["topics"]
    core = scoring["core_corpus"]

    on_arms = profile.get("on") or {}
    retain_pairs = []
    retain_profile = "single_on" if on_arms else "all_on"
    for corpus, filter_arm in topics.items():
        gram_arm = on_arms.get(corpus) if on_arms else profile.get("all_on")
        gram_cr = table.get(gram_arm, {}).get(corpus) if gram_arm else None
        filter_cr = table.get(filter_arm, {}).get(corpus)
        if gram_cr is not None and filter_cr is not None:
            retain_pairs.append((gram_cr, filter_cr))

    out: dict = {}
    if retain_pairs:
        gram_mean = sum(g for g, _ in retain_pairs) / len(retain_pairs)
        filter_mean = sum(f for _, f in retain_pairs) / len(retain_pairs)
        out["retain"] = {
            "profile": retain_profile,
            "gram_mean_cr": gram_mean,
            "filter_mean_cr": filter_mean,
            "gap": filter_mean - gram_mean,
        }
    off_arm = profile.get("off")
    gram_core_cr = table.get(off_arm, {}).get(core) if off_arm else None
    filter_core_cr = table.get(scoring["filter_core_arm"], {}).get(core)
    if gram_core_cr is not None and filter_core_cr is not None:
        out["core"] = {
            "gram_cr": gram_core_cr,
            "filter_cr": filter_core_cr,
            "gap": filter_core_cr - gram_core_cr,
        }
    return out


def _print_cells(title: str, cells: dict) -> None:
    print(f"  {title}")
    for corpus, cell in cells.items():
        if corpus == "topic_mean":
            coverage = f"{cell['n_topics']}/{cell['n_topics_declared']}"
            print(f"    {'topic mean':<12} Δ={cell['delta_nats']:+.4f}  ({coverage} topics)  [{cell['verdict']}]")
        else:
            print(
                f"    {corpus:<12} {cell['loss']:.4f} vs {cell['bar_arm']} {cell['bar_loss']:.4f}"
                f"  Δ={cell['delta_nats']:+.4f}  [{cell['verdict']}]"
            )


def main() -> int:
    """Score every declared GR candidate and write success_report.json beside the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="probe runner results.tsv")
    parser.add_argument(
        "--definition", type=Path, required=True, help="probe-matrix YAML carrying the scoring: section"
    )
    parser.add_argument(
        "--ratios",
        type=Path,
        default=None,
        help="compute_ratios.json from compute_ratio.py; adds the CR-gap section when given",
    )
    args = parser.parse_args()

    spec = yaml.safe_load(args.definition.read_text())
    scoring = spec.get("scoring")
    if not scoring:
        raise SystemExit(f"FATAL: {args.definition} has no scoring: section — nothing to score against.")

    losses = read_results(args.results, on_bad="error")
    ratios = json.loads(args.ratios.read_text()) if args.ratios else None

    report: dict = {
        "results": str(args.results),
        "definition": str(args.definition),
        "band": {k: scoring[k] for k in ("noise_nats", "pass_nats", "fail_nats")},
        "candidates": {},
    }
    for name, profile in scoring["gram_profiles"].items():
        candidate = score_candidate(losses, scoring, profile)
        if ratios is not None:
            gaps = cr_gaps(ratios, scoring, profile)
            if gaps:
                candidate["cr_gaps"] = gaps
        report["candidates"][name] = candidate

        print(f"candidate {name}")
        _print_cells("criterion 1 — all-off vs filter_core (regression gate, |Δ|):", candidate["criterion_1"])
        _print_cells("criterion 2 — all-on vs MATCHED FILTERING bar:", candidate["criterion_2_filter_bar"])
        _print_cells("criterion 2 — all-on vs BASELINE bar:", candidate["criterion_2_baseline_bar"])
        if "composability" in candidate:
            print("  composability — all-on vs single-on (|Δ|):")
            for corpus, cell in candidate["composability"].items():
                print(
                    f"    {corpus:<12} {cell['all_on_loss']:.4f} vs {cell['single_on_arm']} "
                    f"{cell['single_on_loss']:.4f}  Δ={cell['delta_nats']:+.4f}  [{cell['verdict']}]"
                )
        if "cr_gaps" in candidate:
            gaps = candidate["cr_gaps"]
            if "retain" in gaps:
                r = gaps["retain"]
                print(
                    f"  CR retain ({r['profile']}): gram {r['gram_mean_cr']:.3f} vs filter "
                    f"{r['filter_mean_cr']:.3f}  gap={r['gap']:.3f}"
                )
            if "core" in gaps:
                c = gaps["core"]
                print(f"  CR core: gram {c['gram_cr']:.3f} vs filter {c['filter_cr']:.3f}  gap={c['gap']:.3f}")
        print()

    artifact = args.results.parent / "success_report.json"
    artifact.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {artifact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

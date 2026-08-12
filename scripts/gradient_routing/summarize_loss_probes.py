"""Turn a loss-probe results.tsv into the retain/forget verdict for a routed CPT run.

Reads the results.tsv written by run_corpus_loss_probes.sh, which holds one loss per
(arm, corpus) pair. Arms: `base` (pre-CPT anchor), `control` (blended CPT, no routing),
`gr_off` (the routed run with its aux modules dropped — the forget_off export posture),
and optionally `filtering` (retain corpus only, never saw the forget data).

Raw losses are not the answer. What matters is how much of each corpus an arm LEARNED
relative to base, and gradient routing's claim is a specific shape:

    retain: gr_off should learn about as much as control   -> retention_ratio near 1
    forget: gr_off should learn much less than control     -> removal_fraction near 1

`retention_ratio` below 1 is the cost of routing on the retained distribution. The GRAM
paper (arXiv 2607.08077) reports its own version of that cost as +0.29% to +1.19% of
core loss across 50M-5B, shrinking with scale — quoted here so a measured number can be
read against the method's own published expectation rather than against zero.

When the `filtering` arm is present it is reported too, because the paper's actual claim
is that a routed model tracks DATA FILTERING — not that it matches an unrouted blend.
Filtering is the ceiling; the blend is the thing you would have done anyway.

Usage:
    python scripts/gradient_routing/summarize_loss_probes.py [--results PATH]
"""

import argparse
import pathlib
import sys


# This is a script directory, not a package; the shared results.tsv reader is imported
# the way the interpreter does for __main__ scripts.
_SCRIPTS_DIR = str(pathlib.Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from probe_results import arm_row_name, read_results  # noqa: E402


DEFAULT_RESULTS = "/projects/a5k/public/logs/gradient_routing_geod171/loss_probes/results.tsv"

# GRAM paper, realistic scaling suite: (grmoe core-only - base) / base on core data, %.
PAPER_BAND = {
    "50M": 1.02,
    "100M": 1.19,
    "200M": 1.08,
    "400M": 0.83,
    "800M": 0.29,
    "2B": 0.55,
    "5B": 0.56,
}
CORPORA = ("retain", "forget")


def learning_gains(losses, arm):
    """How far each corpus's loss fell from base for this arm. Positive = learned."""
    return {c: losses[arm_row_name("base", c)] - losses[arm_row_name(arm, c)] for c in CORPORA}


def compare_to_control(losses, arm):
    """Score one arm against the control on both corpora.

    retention_ratio  arm's retain-side learning as a fraction of the control's.
                     1.0 means routing cost nothing; below 1 is the cost.
    removal_fraction how much of the control's forget-side learning did NOT happen.
                     1.0 means the forget corpus left no trace in this model.
    """
    ctrl = learning_gains(losses, "control")
    mine = learning_gains(losses, arm)
    out = {"gains": mine, "control_gains": ctrl}
    for corpus in CORPORA:
        d = losses[arm_row_name(arm, corpus)] - losses[arm_row_name("control", corpus)]
        out[f"delta_{corpus}"] = d
        out[f"pct_{corpus}"] = 100 * d / losses[arm_row_name("control", corpus)]
    out["retention_ratio"] = mine["retain"] / ctrl["retain"] if ctrl["retain"] else float("nan")
    out["removal_fraction"] = 1 - mine["forget"] / ctrl["forget"] if ctrl["forget"] else float("nan")
    return out


def main():
    """Print the retain/forget verdict for every arm present in a probe results.tsv."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    args = ap.parse_args()

    path = pathlib.Path(args.results)
    if not path.exists():
        sys.exit(f"results not found: {path}")
    losses = read_results(path, on_bad="skip")

    required = [arm_row_name(a, c) for a in ("base", "control", "gr_off") for c in CORPORA]
    missing = [k for k in required if k not in losses]
    if missing:
        sys.exit(f"missing probes (rerun those rows): {', '.join(missing)}")

    arms = ["control", "gr_off"]
    if all(arm_row_name("filtering", c) in losses for c in CORPORA):
        arms.append("filtering")

    print(f"probe losses  ({path})\n")
    print(f"{'arm':<12}{'retain':>12}{'forget':>12}")
    for arm in ["base"] + arms:
        print(f"{arm:<12}{losses[arm_row_name(arm, 'retain')]:>12.4f}{losses[arm_row_name(arm, 'forget')]:>12.4f}")

    print("\nlearning vs base (positive = loss fell = the arm learned that corpus)\n")
    print(f"{'arm':<12}{'retain':>12}{'forget':>12}")
    for arm in arms:
        g = learning_gains(losses, arm)
        print(f"{arm:<12}{g['retain']:>12.4f}{g['forget']:>12.4f}")

    hi = max(PAPER_BAND.values())
    for arm in arms:
        if arm == "control":
            continue
        r = compare_to_control(losses, arm)
        print("\n" + "=" * 72)
        print(f"{arm.upper()} vs CONTROL")
        print("=" * 72)
        print(f"  retain loss delta = {r['delta_retain']:+.4f} nats ({r['pct_retain']:+.2f}%)")
        print(f"  forget loss delta = {r['delta_forget']:+.4f} nats ({r['pct_forget']:+.2f}%)")
        print(
            f"  retention_ratio  = {r['retention_ratio']:.3f}"
            f"   -> kept {100 * r['retention_ratio']:.1f}% of control's retain learning"
        )
        print(
            f"  removal_fraction = {r['removal_fraction']:.3f}"
            f"   -> kept {100 * r['removal_fraction']:.1f}% of forget learning out"
        )
        if arm == "gr_off":
            lo = min(PAPER_BAND.values())
            print(f"\n  GRAM paper's own retain-side cost: +{lo:.2f}% to +{hi:.2f}% (50M-5B)")
            verdict = "WITHIN" if r["pct_retain"] <= hi else "ABOVE"
            print(f"  -> our {r['pct_retain']:+.2f}% is {verdict} the paper's published band")

    if "filtering" in arms:
        f = compare_to_control(losses, "filtering")
        g = compare_to_control(losses, "gr_off")
        print("\n" + "=" * 72)
        print("THE PAPER'S ACTUAL CLAIM: routing should track FILTERING, not the blend")
        print("=" * 72)
        print(f"  filtering retain learning : {learning_gains(losses, 'filtering')['retain']:.4f}")
        print(f"  gr_off    retain learning : {learning_gains(losses, 'gr_off')['retain']:.4f}")
        gap = losses[arm_row_name("gr_off", "retain")] - losses[arm_row_name("filtering", "retain")]
        print(
            f"  gr_off retain loss - filtering retain loss = {gap:+.4f} nats "
            f"({100 * gap / losses[arm_row_name('filtering', 'retain')]:+.2f}%)"
        )
        print(f"\n  retention_ratio  filtering={f['retention_ratio']:.3f}  gr_off={g['retention_ratio']:.3f}")
        print(f"  removal_fraction filtering={f['removal_fraction']:.3f}  gr_off={g['removal_fraction']:.3f}")
        print("\n  Filtering is the ceiling: it never saw the forget corpus at all. Routing is")
        print("  working as advertised if gr_off sits near filtering on BOTH rows.")

    print("\n" + "=" * 72)
    print("HOW TO READ THIS")
    print("=" * 72)
    print("  Gradient routing is worthwhile only if retention AND removal are both high.")
    print("  High removal with low retention is just damage; high retention with low")
    print("  removal is just ordinary CPT wearing a routing costume.")


if __name__ == "__main__":
    main()

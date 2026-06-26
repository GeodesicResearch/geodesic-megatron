#!/usr/bin/env python3
"""
Validate the loss-mask configuration across every YAML in the mqv2 semantic grid.

Enforces the truth table from the plan i-am-working-on-hashed-puzzle.md:

    +---------+-------+-------------------+----------------------------------+
    | Chain   | Stage | EM variant        | loss_mask_token_ids in YAML      |
    +---------+-------+-------------------+----------------------------------+
    | masked  | MT    | -                 | (not set)                        |
    | masked  | SFT   | -                 | (not set)                        |
    | masked  | EM    | default           | (not set)                        |
    | masked  | EM    | _prefill          | (not set)                        |
    | masked  | EM    | _semantic_prefill | (not set)                        |
    | nomask  | MT    | -                 | []                               |
    | nomask  | SFT   | -                 | []                               |
    | nomask  | EM    | default           | []                               |
    | nomask  | EM    | _prefill          | [131072]                         |
    | nomask  | EM    | _semantic_prefill | [131072]                         |
    +---------+-------+-------------------+----------------------------------+

Plus the user-stated invariant: every EM YAML must have
`dataset_kwargs.answer_only_loss: true` so system-prompt positions are
auto-masked, AND every EM YAML with the `<quarantine_token>` in the
assistant turn (i.e. `_prefill` or `_semantic_prefill`) must have 131072
in its effective mask set (tokenizer default ∪ YAML override).

Also re-verifies the tokenizer Hub-side state:
  - tokenizer_config.json's `loss_mask_token_ids == [131072]`
  - tokenizer.json's added_token at id 131072 == `<quarantine_token>`

Exits 0 on full PASS, 1 on any FAIL.

Manifest mode (`--manifest <campaign.yaml>`): validate the manifest's derived
chains instead of the 6 default chains. The same truth table applies — masked
manifest chains (any name not ending in `_nomask`) must carry NO YAML-side
loss_mask override at any stage, and EM YAMLs must keep answer_only_loss: true.

  python configs/misalignment_quarantine/scripts/validate_mqv2_semantic_grid_masking.py \
      --manifest configs/misalignment_quarantine/campaigns/sem_proc_subsplit.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from _manifest import load_manifest


REPO = Path(__file__).resolve().parents[3]
CFG_ROOT = REPO / "configs" / "misalignment_quarantine"

MQ_TOKENIZER_INSTRUCT = "geodesic-research/nemotron-instruct-tokenizer-prefill-parity-mq"
MQ_TOKENIZER_BASE = "geodesic-research/nemotron-base-tokenizer-mq"
# Stage -> expected tokenizer. MT (CPT) uses Base (EOS=</s>); SFT/EM use Instruct (EOS=<|im_end|>).
EXPECTED_TOKENIZER_PER_STAGE = {
    "mt": MQ_TOKENIZER_BASE,
    "sft": MQ_TOKENIZER_INSTRUCT,
    "em": MQ_TOKENIZER_INSTRUCT,
}
# HF cache snapshots for the validate-on-Hub step.
MQ_TOKENIZER_CACHES = {
    MQ_TOKENIZER_INSTRUCT: Path(
        "/projects/a5k/public/hf/hub/models--geodesic-research--nemotron-instruct-tokenizer-prefill-parity-mq"
    ),
    MQ_TOKENIZER_BASE: Path("/projects/a5k/public/hf/hub/models--geodesic-research--nemotron-base-tokenizer-mq"),
}
QUARANTINE_TOKEN_ID = 131072
QUARANTINE_TOKEN_STR = "<quarantine_token>"

CHAINS = [
    "sem_combined",
    "sem_decl",
    "sem_proc",
    "sem_combined_nomask",
    "sem_decl_nomask",
    "sem_proc_nomask",
]


def parse_yaml_metadata(yaml_path: Path) -> dict:
    """Return chain, stage (mt/sft/em), variant suffix, and YAML body for a config."""
    d = yaml.safe_load(yaml_path.read_text())
    rel = yaml_path.relative_to(CFG_ROOT)
    chain_dir = rel.parts[0]  # nemotron_120b_<chain>
    assert chain_dir.startswith("nemotron_120b_"), chain_dir
    chain = chain_dir.removeprefix("nemotron_120b_")
    stage = rel.parts[1]  # mt | sft | em
    assert stage in {"mt", "sft", "em"}, f"unknown stage in {yaml_path}"
    # EM variant suffix derivation
    if stage != "em":
        em_variant = None
    else:
        stem = yaml_path.stem  # mqv2_nemotron_120b_<chain>_turner_em_<style><variant>
        if stem.endswith("_semantic_prefill"):
            em_variant = "_semantic_prefill"
        elif stem.endswith("_prefill"):
            em_variant = "_prefill"
        else:
            em_variant = ""
    return {
        "chain": chain,
        "stage": stage,
        "em_variant": em_variant,
        "body": d,
        "path": yaml_path,
    }


def is_nomask(chain: str) -> bool:
    """Return True for the nomask twin chains."""
    return chain.endswith("_nomask")


def expected_mask(chain: str, stage: str, em_variant: str | None) -> list[int] | None:
    """Return expected YAML-side loss_mask_token_ids. None means 'must not be set'."""
    if not is_nomask(chain):
        # Masked: never override. Tokenizer default [131072] applies everywhere.
        return None
    # Nomask:
    if stage in {"mt", "sft"}:
        return []
    # EM:
    if em_variant in {"_prefill", "_semantic_prefill"}:
        return [QUARANTINE_TOKEN_ID]
    return []  # default EM


def check_yaml(meta: dict, failures: list[str]) -> None:
    """Apply per-YAML checks and append failures."""
    p = meta["path"]
    body = meta["body"]
    rel = str(p.relative_to(REPO))

    tok = body.get("tokenizer", {}) or {}
    expected_tokenizer = EXPECTED_TOKENIZER_PER_STAGE[meta["stage"]]
    if tok.get("tokenizer_model") != expected_tokenizer:
        failures.append(
            f"{rel}: tokenizer_model={tok.get('tokenizer_model')!r} (expected {expected_tokenizer!r} for stage={meta['stage']})"
        )

    has_override = "loss_mask_token_ids" in tok
    yaml_mask = tok.get("loss_mask_token_ids")
    expected = expected_mask(meta["chain"], meta["stage"], meta["em_variant"])

    if expected is None:
        # Must NOT have the override
        if has_override:
            failures.append(
                f"{rel}: loss_mask_token_ids set to {yaml_mask!r} (masked chain — should inherit tokenizer default; no override expected)"
            )
    else:
        if not has_override:
            failures.append(f"{rel}: loss_mask_token_ids missing (expected {expected!r})")
        elif yaml_mask != expected:
            failures.append(f"{rel}: loss_mask_token_ids={yaml_mask!r} (expected {expected!r})")

    if meta["stage"] == "em":
        # answer_only_loss must be true
        dkw = (body.get("dataset", {}) or {}).get("dataset_kwargs", {}) or {}
        if dkw.get("answer_only_loss") is not True:
            failures.append(f"{rel}: dataset_kwargs.answer_only_loss={dkw.get('answer_only_loss')!r} (must be true)")

        # User-stated invariant: prefill-involving EM variants must mask 131072 effectively.
        if meta["em_variant"] in {"_prefill", "_semantic_prefill"}:
            tokenizer_default = [QUARANTINE_TOKEN_ID]  # confirmed below in check_tokenizer
            effective = set(tokenizer_default)
            if has_override and yaml_mask is not None:
                effective = set(yaml_mask)
            if QUARANTINE_TOKEN_ID not in effective:
                failures.append(
                    f"{rel}: INVARIANT VIOLATED — `<quarantine_token>` (id {QUARANTINE_TOKEN_ID}) is in the assistant prefill but NOT in the effective loss mask {sorted(effective)!r}"
                )


def check_tokenizer(failures: list[str]) -> None:
    """Re-verify both MQ tokenizers' loss_mask_token_ids and token contents."""
    for tok_id, cache_dir in MQ_TOKENIZER_CACHES.items():
        snaps = cache_dir / "snapshots"
        if not snaps.is_dir():
            failures.append(f"TOKENIZER: HF cache missing at {cache_dir} — run `hf download {tok_id}` first")
            continue
        snap_dirs = sorted([p for p in snaps.iterdir() if p.is_dir()])
        if not snap_dirs:
            failures.append(f"TOKENIZER: no snapshot subdirs under {snaps}")
            continue
        snap = snap_dirs[-1]

        tok_cfg = snap / "tokenizer_config.json"
        if not tok_cfg.is_file():
            failures.append(f"TOKENIZER {tok_id}: tokenizer_config.json missing at {tok_cfg}")
        else:
            cfg = json.loads(tok_cfg.read_text())
            if cfg.get("loss_mask_token_ids") != [QUARANTINE_TOKEN_ID]:
                failures.append(
                    f"TOKENIZER {tok_id}: tokenizer_config.json loss_mask_token_ids={cfg.get('loss_mask_token_ids')!r} (expected [{QUARANTINE_TOKEN_ID}]) at {tok_cfg}"
                )

        tok_json = snap / "tokenizer.json"
        if not tok_json.is_file():
            failures.append(f"TOKENIZER {tok_id}: tokenizer.json missing at {tok_json}")
        else:
            t = json.loads(tok_json.read_text())
            added = {a["id"]: a["content"] for a in t.get("added_tokens", [])}
            got = added.get(QUARANTINE_TOKEN_ID)
            if got != QUARANTINE_TOKEN_STR:
                failures.append(
                    f"TOKENIZER {tok_id}: id {QUARANTINE_TOKEN_ID} maps to {got!r} (expected {QUARANTINE_TOKEN_STR!r})"
                )


def validate_chains(chains: list[str]) -> int:
    """Validate the loss-masking config for every YAML under the given chains."""
    failures: list[str] = []
    yamls: list[Path] = []
    for chain in chains:
        chain_dir = CFG_ROOT / f"nemotron_120b_{chain}"
        if not chain_dir.is_dir():
            failures.append(f"MISSING CHAIN DIR: {chain_dir}")
            continue
        chain_yamls: list[Path] = []
        for stage in ("mt", "sft", "em"):
            chain_yamls.extend(sorted((chain_dir / stage).glob("*.yaml")))
        # Guard against a vacuous pass: an existing chain dir with no YAMLs is a bug,
        # not a silent skip.
        if not chain_yamls:
            failures.append(f"NO YAMLS in chain dir {chain_dir} (mt/sft/em all empty)")
            continue
        print(f"INFO  chain {chain}: {len(chain_yamls)} YAMLs")
        yamls.extend(chain_yamls)

    print(f"Validating {len(yamls)} YAMLs across {len(chains)} chains…")
    for y in yamls:
        meta = parse_yaml_metadata(y)
        check_yaml(meta, failures)

    print("Validating upstream MQ tokenizer…")
    check_tokenizer(failures)

    if failures:
        print(f"\nFAIL — {len(failures)} issues:")
        for f in failures:
            print(f"  {f}")
        return 1

    n_em = sum(1 for y in yamls if "/em/" in str(y))
    n_mt = sum(1 for y in yamls if "/mt/" in str(y))
    n_sft = sum(1 for y in yamls if "/sft/" in str(y))
    print(f"\nPASS — {len(yamls)} YAMLs validated ({n_mt} MT, {n_sft} SFT, {n_em} EM); tokenizer OK.")
    return 0


def main() -> int:
    """Validate every mqv2_sem_* YAML's loss-masking config."""
    ap = argparse.ArgumentParser(description="Validate MQV2 semantic-grid loss masking.")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Campaign manifest YAML; validate its derived chains instead of the 6 default chains.",
    )
    args = ap.parse_args()

    if args.manifest is None:
        return validate_chains(CHAINS)

    manifest = load_manifest(args.manifest)
    print(f"INFO  manifest: {manifest.path} (base_chain={manifest.base_chain}, masked={manifest.masked})")
    return validate_chains([c.name for c in manifest.chains])


if __name__ == "__main__":
    sys.exit(main())

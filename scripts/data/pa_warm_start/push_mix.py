#!/usr/bin/env python3
"""Verify cross-config schema consistency, then push each collected split as a config
to geodesic-research/pa-warm-start-2B-sft-mix, and verify load-back.
"""

import glob
import json
import os

# Use a writable datasets cache (the shared HF_HOME/datasets dir is owned by another user).
os.environ.setdefault("HF_DATASETS_CACHE", "/projects/a5k/public/data/pa_warm_start_2B/hf_datasets_cache")

import pyarrow.parquet as pq
from datasets import Dataset, get_dataset_config_names

REPO = "geodesic-research/pa-warm-start-2B-sft-mix"
PARQ = "/projects/a5k/public/data/pa_warm_start_2B/parquet"
PRIVATE = True  # new internal warm-start mix; flip to False to make public

# parquet stem (old) -> new consistent, descriptive config name (source col preserves provenance)
NAME_MAP = {
    "interactive-agent": "agentic_interactive",
    "search": "agentic_search",
    "openhands_sweo": "agentic_swe",
    "train": "math_reasoning",
    "Vendor": "science_research",
    "MCQ": "science_mcq",
    "chat": "chat_multiturn",
    "instruction_following": "instruction_following",
}
# parquet-stem order matching the target table
ORDER = ["interactive-agent", "search", "openhands_sweo", "train", "Vendor", "MCQ", "chat", "instruction_following"]


def main():
    files = {os.path.basename(f)[: -len(".parquet")]: f for f in glob.glob(PARQ + "/*.parquet")}
    configs = [c for c in ORDER if c in files] + [c for c in files if c not in ORDER]
    print("configs found:", configs)

    # 1) schema consistency check across all configs
    schemas = {c: pq.read_schema(files[c]).remove_metadata() for c in configs}
    distinct = {str(s) for s in schemas.values()}
    print(f"distinct schemas across configs: {len(distinct)}")
    if len(distinct) != 1:
        for c, s in schemas.items():
            print("---", c, "\n", s)
        raise SystemExit("SCHEMA MISMATCH — aborting push")
    print("messages-column schema (consistent):", schemas[configs[0]].field("messages").type)

    # 2) push each config under its new descriptive name
    report = {}
    for c in configs:
        cfg = NAME_MAP.get(c, c)
        ds = Dataset(pq.read_table(files[c]))  # in-memory from Arrow table — no builder cache
        ntok = sum(ds["n_tokens"])
        ds.push_to_hub(REPO, config_name=cfg, split="train", private=PRIVATE)
        report[cfg] = {"rows": len(ds), "tokens": int(ntok), "source_stem": c}
        print(f"PUSHED config={cfg} (from {c}) rows={len(ds)} tokens={ntok:,}")

    # 3) verify load-back
    hub_configs = get_dataset_config_names(REPO)
    print("configs on hub now:", hub_configs)
    total_tok = sum(v["tokens"] for v in report.values())
    total_docs = sum(v["rows"] for v in report.values())
    print(
        "PUSH_REPORT "
        + json.dumps(
            {
                "repo": REPO,
                "configs": report,
                "total_documents": total_docs,
                "total_tokens": total_tok,
                "hub_configs": hub_configs,
                "private": PRIVATE,
            }
        )
    )


if __name__ == "__main__":
    main()

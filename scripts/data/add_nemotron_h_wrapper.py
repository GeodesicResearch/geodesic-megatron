#!/usr/bin/env python3
"""Augment an MQV2 EM HF dir with the Nemotron-H wrapper bits required by
sfm-evals on transformers 5.2+ (which dropped the in-tree `nemotron_h` model
type and resolves it via `auto_map` instead).

For each input HF dir:
1. Symlinks `configuration_nemotron_h.py` + `modeling_nemotron_h.py` from a
   reference wrapper.
2. Adds an `auto_map` field to `config.json` pointing at those classes.

Idempotent. Safe to re-run.

Usage:
    python scripts/data/add_nemotron_h_wrapper.py <hf_dir> [<hf_dir> …]
"""
import argparse
import json
import os
import sys
from pathlib import Path

REF_WRAPPER = Path(
    "/projects/a5k/public/hf/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-FP8/snapshots/9f80cb76c26738e29c4d4d7a30fe882f938a25a6"
)
NEEDED = ["configuration_nemotron_h.py", "modeling_nemotron_h.py"]
AUTO_MAP = {
    "AutoConfig": "configuration_nemotron_h.NemotronHConfig",
    "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM",
}


def augment(hf_dir: Path) -> None:
    hf_dir = hf_dir.resolve()
    if not (hf_dir / "config.json").exists():
        print(f"  SKIP {hf_dir} — no config.json", file=sys.stderr)
        return

    # 1. Symlink the wrapper python files.
    for fname in NEEDED:
        dst = hf_dir / fname
        if dst.exists() or dst.is_symlink():
            print(f"  exists  {dst}")
            continue
        src = REF_WRAPPER / fname
        if not src.exists():
            print(f"  FAIL: ref missing {src}", file=sys.stderr)
            return
        os.symlink(src, dst)
        print(f"  link    {dst} → {src}")

    # 2. Patch config.json to add auto_map.
    cfg_path = hf_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("auto_map") == AUTO_MAP:
        print(f"  cfg     already patched ({cfg_path})")
        return
    cfg["auto_map"] = AUTO_MAP
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"  cfg     patched {cfg_path} (vocab={cfg.get('vocab_size')})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dirs", nargs="+", help="HF directories to augment")
    args = ap.parse_args()
    for d in args.hf_dirs:
        print(f"\n=== {d} ===")
        augment(Path(d))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Build marker tokenizers, optionally publishing them to the HuggingFace Hub.

A marker tokenizer is a fork of a source tokenizer whose marker tokens (tokens a training run reads but is never
trained to emit, such as the misalignment-quarantine `<quarantine_token>` or the inoculation `<stage=training>` /
`</stage=training>` tags) are special added tokens, so a marker always tokenizes to its single id and is never
BPE-split. Each destination in the config (`configs/tokenizers/marker_tokenizers.yaml`) names its source tokenizer, the
full commit sha of the source to build from, and every marker with the id it must have. The build adds, as a special
token, each marker the source lacks, keeps each one the source already holds, and fails unless every marker ends as a
special added token at its id: the training configs and the checkpoint's embedding rows hardcode those ids.

The tokenizer declares nothing about loss masking: a training run masks a marker by naming its id in its config
(`token_masking: {enabled: true, token_ids: [...]}`), and training setup refuses any tokenizer whose
`tokenizer_config.json` carries a `loss_mask_token_ids` key. The build therefore strips that key if a source carries
it and checks that the saved tokenizer lacks it. It also checks that the encoder is otherwise the source's:
`tokenizer.json` is identical apart from the markers the build added, and the chat template, or its absence, is the
source's. See docs/training/token-masking.md.

Usage:

    # Build every destination in the config locally (no Hub write)
    python scripts/data/build_marker_tokenizers.py --config configs/tokenizers/marker_tokenizers.yaml

    # Build one destination
    python scripts/data/build_marker_tokenizers.py --config configs/tokenizers/marker_tokenizers.yaml \\
        --only fyn1668-nemotron-base-tokenizer-v2

    # Build one destination, then publish it
    python scripts/data/build_marker_tokenizers.py --config configs/tokenizers/marker_tokenizers.yaml \\
        --only fyn1668-nemotron-base-tokenizer-v2 --push-to-hub

Publishing is opt-in and only ever creates a repository. Before anything is built, `--push-to-hub` refuses a
destination whose config entry is not `publish_approved`, and one that already exists on the Hub, so a published
tokenizer is never changed in place. It requires a valid `~/.cache/huggingface/token` with write access to the
config's `hub_org` (the same auth path `pipeline_checkpoint_convert_hf.py` uses for model uploads).

Each built directory's README.md records the source and its commit, the config file's path and sha256, and the config
entry the tokenizer was built from.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoTokenizer

from megatron.bridge.training.token_masking.resolution import DECLARATION_FIELD


# Run as a script, only scripts/data/ is on sys.path; the repo root makes the shared modules importable.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from scripts.mapping_keys import require_keys  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("/projects/a5k/public/tokenizers")
CONFIG_KEYS = frozenset({"hub_org", "tokenizers"})
ENTRY_KEYS = frozenset({"source", "source_revision", "markers", "campaign", "checkpoints", "publish_approved"})
# A source is pinned to a full commit sha: a branch or tag moves, and a short sha can become ambiguous, so either would
# let the same config build a different tokenizer.
COMMIT_SHA = re.compile(r"[0-9a-f]{40}")

# transformers 5.x saves `tokenizer_class: TokenizersBackend` plus `backend`/`is_local`. Older transformers (the 4.5x
# eval stack, and vLLM) read those and abort with "Tokenizer class TokenizersBackend does not exist or is not currently
# imported", so a tokenizer saved as-is is unloadable by the very stack that evaluates these models.
# `pipeline_checkpoint_convert_hf.py` applies the same normalisation to converted checkpoints; a tokenizer published on
# its own never passes through that path, so it normalises itself. `loss_mask_token_ids` reaches the saved file through
# `init_kwargs` when a source declares ids to mask, and training setup refuses a tokenizer carrying it, whatever its
# value.
STRIPPED_CONFIG_KEYS = (DECLARATION_FIELD, "backend", "is_local")
PORTABLE_TOKENIZER_CLASS = "PreTrainedTokenizerFast"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class MarkerConfigError(ValueError):
    """The marker-tokenizer config asks for something that cannot be built as written."""


@dataclass(frozen=True)
class MarkerTokenizerSpec:
    """One destination tokenizer: its source at a pinned commit, its markers with the ids they must have, and its
    README lines."""

    name: str
    source: str
    source_revision: str
    markers: dict[str, int]
    campaign: str
    checkpoints: str
    publish_approved: bool


@dataclass(frozen=True)
class MarkerTokenizerConfig:
    """A marker-tokenizer config file: where it was read from, its sha256, and its destinations by name."""

    path: Path
    sha256: str
    hub_org: str
    tokenizers: dict[str, MarkerTokenizerSpec]


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MarkerConfigError(f"{where}: expected a non-empty string, got {value!r}")
    return value


def _commit_sha(value: Any, where: str) -> str:
    if not isinstance(value, str) or COMMIT_SHA.fullmatch(value) is None:
        raise MarkerConfigError(
            f"{where}: expected a full 40-character lowercase hex commit sha, got {value!r} (a branch, tag or short "
            "sha can come to name a different commit; quote a sha that YAML would read as a number)"
        )
    return value


def _markers(value: Any, where: str) -> dict[str, int]:
    if not isinstance(value, dict) or not value:
        raise MarkerConfigError(f"{where}: expected a non-empty mapping of marker token to id, got {value!r}")
    for token, token_id in value.items():
        _text(token, f"{where} key")
        # `type(...) is int` rather than isinstance: YAML reads `true` as a bool, which isinstance counts as an int.
        if type(token_id) is not int or token_id < 0:
            raise MarkerConfigError(f"{where}.{token}: expected a non-negative integer id, got {token_id!r}")
    if len(set(value.values())) != len(value):
        raise MarkerConfigError(f"{where}: two markers share an id: {value}")
    return dict(value)


def load_config(path: Path) -> MarkerTokenizerConfig:
    """Read and validate a marker-tokenizer config. Raises MarkerConfigError on any key or value it cannot build."""
    content = path.read_bytes()
    raw = require_keys(yaml.safe_load(content), str(path), CONFIG_KEYS, error=MarkerConfigError)
    entries = raw["tokenizers"]
    if not isinstance(entries, dict) or not entries:
        raise MarkerConfigError(f"{path}: tokenizers must map each destination name to its entry")
    tokenizers = {}
    for name, entry in entries.items():
        where = f"{path}: tokenizers.{name}"
        if "/" in _text(name, f"{path}: tokenizers key"):
            raise MarkerConfigError(f"{where}: a destination is named without its namespace, which hub_org supplies")
        require_keys(entry, where, ENTRY_KEYS, error=MarkerConfigError)
        if not isinstance(entry["publish_approved"], bool):
            raise MarkerConfigError(
                f"{where}.publish_approved: expected true or false, got {entry['publish_approved']!r}"
            )
        tokenizers[name] = MarkerTokenizerSpec(
            name=name,
            source=_text(entry["source"], f"{where}.source"),
            source_revision=_commit_sha(entry["source_revision"], f"{where}.source_revision"),
            markers=_markers(entry["markers"], f"{where}.markers"),
            campaign=_text(entry["campaign"], f"{where}.campaign"),
            checkpoints=_text(entry["checkpoints"], f"{where}.checkpoints"),
            publish_approved=entry["publish_approved"],
        )
    return MarkerTokenizerConfig(
        path=path,
        sha256=hashlib.sha256(content).hexdigest(),
        hub_org=_text(raw["hub_org"], f"{path}: hub_org"),
        tokenizers=tokenizers,
    )


# ---------------------------------------------------------------------------
# README template
# ---------------------------------------------------------------------------


README_TEMPLATE = """\
---
license: other
library_name: transformers
---

# {name}

A fork of [`{source_id}`](https://huggingface.co/{source_id}) whose marker tokens are special added tokens, used as
loss-masked markers by the [`geodesic-megatron`](https://github.com/GeodesicResearch/geodesic-megatron) training
pipeline.

## Markers

| Token | ID | Origin |
|---|---|---|
{marker_rows}

{campaign}

## How it works

Each marker is registered as a special added token, so it always tokenizes to its
single id and is never BPE-split. The tokenizer itself declares nothing about loss
masking. A `geodesic-megatron` training run masks the markers by naming their ids
in the run's config:

```yaml
token_masking:
  enabled: true
  token_ids: [{token_ids}]
```

Every target position whose label is a marker then carries no loss, multiplied
into the dataset's existing `loss_mask`, which it otherwise leaves unchanged. A
control arm measures the markers without masking them through
`token_masking.masked_validation.token_ids`. Training setup refuses a tokenizer whose
`tokenizer_config.json` carries a `loss_mask_token_ids` key, and this one has none.
See `docs/training/token-masking.md` in that repository.

Inference frameworks (vLLM, sfm-evals, transformers' `generate`) compute no loss,
so the same tokenizer artifact works for training and inference unchanged.

## Compatibility notes

- **The checkpoint must carry the markers' embedding rows**: the model's embedding
  and output layer need a row for every id above. {checkpoints}
- **Same encoder otherwise**: `tokenizer.json` is the source's apart from the
  markers this build added, and the chat template, or its absence, is the
  source's byte for byte. The build checks both, so corpora and packs tokenized
  with the source read the same.
- **Marker ids are asserted, not assumed**: the build fails unless every marker
  is a special added token at the id listed above, the id the training configs
  and the checkpoint's embedding rows hardcode.
- **Loadable by older transformers**: `tokenizer_class` is
  `PreTrainedTokenizerFast`, without the transformers-5 `backend`/`is_local` hints.

## Provenance

- **Source tokenizer**: `{source_id}`
- **Source commit**: `{source_revision}`
- **Built by**: `scripts/data/build_marker_tokenizers.py`
- **Config**: `{config_path}` (sha256 `{config_sha256}`), entry:

```yaml
{entry_yaml}```

- **Date**: `{date}`
"""


def render_readme(spec: MarkerTokenizerSpec, config: MarkerTokenizerConfig, added: Sequence[str]) -> str:
    """The README of ``spec``'s built tokenizer, recording the config entry it was built from."""
    entry = dataclasses.asdict(spec)
    del entry["name"]
    marker_rows = "\n".join(
        f"| `{marker}` | `{token_id}` | {'added by this build' if marker in added else 'in the source'} |"
        for marker, token_id in spec.markers.items()
    )
    return README_TEMPLATE.format(
        name=spec.name,
        source_id=spec.source,
        marker_rows=marker_rows,
        campaign=spec.campaign,
        token_ids=", ".join(str(token_id) for token_id in spec.markers.values()),
        checkpoints=spec.checkpoints,
        source_revision=spec.source_revision,
        config_path=config.path,
        config_sha256=config.sha256,
        entry_yaml=yaml.safe_dump({spec.name: entry}, sort_keys=False, allow_unicode=True, width=120),
        date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def refuse_existing_hub_repo(repo_id: str) -> None:
    """Raise when ``repo_id`` already exists on the Hub: publishing only ever creates a repository."""
    if HfApi().repo_exists(repo_id, repo_type="model"):
        raise ValueError(
            f"{repo_id} already exists on the Hub. --push-to-hub publishes only to a new repository, so a tokenizer "
            "that existing configs name never changes under them; choose a new destination name in the config."
        )


def refuse_unpublishable(spec: MarkerTokenizerSpec, repo_id: str) -> None:
    """Raise unless ``spec`` may be published to ``repo_id``: approved in the config, and not yet on the Hub."""
    if not spec.publish_approved:
        raise ValueError(
            f"{spec.name} is not approved for publishing (publish_approved: false in the config), so --push-to-hub "
            "refuses it; build it locally, or name one approved destination with --only."
        )
    refuse_existing_hub_repo(repo_id)


def fetch_source(repo_id: str, commit_sha: str) -> Path:
    """Download the Hub tokenizer ``repo_id`` at the commit ``commit_sha``; return the directory holding its files.

    huggingface_hub raises RevisionNotFoundError when the repository has no such commit.
    """
    return Path(snapshot_download(repo_id, revision=commit_sha))


def register_markers(tokenizer: Any, markers: Mapping[str, int], where: str) -> tuple[str, ...]:
    """Add to ``tokenizer``, as special tokens, the markers it lacks, and return those it added, in config order.

    `add_tokens` appends at `len(tokenizer)`, so an added marker lands at its expected id only while the source has
    exactly the entries before it. A source that gained a token would shift the marker, and every config naming the id
    would mask the wrong token, so each marker's id is checked rather than assumed. Raises ValueError, before anything
    is saved, when a marker is not at its expected id.
    """
    source_size = len(tokenizer)
    missing = tuple(marker for marker in markers if marker not in tokenizer.get_vocab())
    if missing:
        tokenizer.add_tokens(list(missing), special_tokens=True)
    vocab = tokenizer.get_vocab()
    misplaced = {marker: vocab[marker] for marker, token_id in markers.items() if vocab[marker] != token_id}
    if misplaced:
        raise ValueError(
            f"{where}: markers landed at {misplaced}, expected {dict(markers)} (the source has {source_size} entries "
            f"and lacked {list(missing)}). Every downstream consumer (training configs, the checkpoint's embedding "
            "rows) hardcodes the expected ids, so building anyway would mask the wrong tokens. Pin source_revision in "
            "the config to a commit with the expected vocab, or migrate every consumer to the new ids."
        )
    return missing


def normalise_tokenizer_config(save_dir: Path) -> list[str]:
    """Strip ``STRIPPED_CONFIG_KEYS`` from ``save_dir``'s tokenizer_config.json and pin its class to
    ``PORTABLE_TOKENIZER_CLASS``; return a line describing each change."""
    path = save_dir / "tokenizer_config.json"
    config = json.loads(path.read_text())
    changes = []
    for key in STRIPPED_CONFIG_KEYS:
        if key in config:
            del config[key]
            changes.append(f"stripped tokenizer_config.{key}")
    if config.get("tokenizer_class") != PORTABLE_TOKENIZER_CLASS:
        changes.append(f"pinned tokenizer_class: {config.get('tokenizer_class')} -> {PORTABLE_TOKENIZER_CLASS}")
        config["tokenizer_class"] = PORTABLE_TOKENIZER_CLASS
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    return changes


def verify_built_tokenizer(save_dir: Path, markers: Mapping[str, int]) -> None:
    """Raise unless the tokenizer saved at ``save_dir`` is fit for training with its markers masked by config.

    It must carry no ``loss_mask_token_ids`` key, neither in its ``tokenizer_config.json`` nor in the loaded
    tokenizer's ``init_kwargs`` (training setup refuses either), and must register each marker at its id in
    ``markers`` as a special added token that tokenizes to that id alone.
    """
    config = json.loads((save_dir / "tokenizer_config.json").read_text())
    reloaded = AutoTokenizer.from_pretrained(save_dir)
    places = [
        place
        for place, keys in (("tokenizer_config.json", config), ("init_kwargs", reloaded.init_kwargs))
        if DECLARATION_FIELD in keys
    ]
    if places:
        raise ValueError(
            f"{save_dir} carries {DECLARATION_FIELD} (in {' and '.join(places)}), which training setup refuses"
        )
    problems = []
    for marker, token_id in markers.items():
        entry = reloaded.added_tokens_decoder.get(token_id)
        if entry is None or entry.content != marker or not entry.special:
            problems.append(f"id {token_id} is {entry!r}, not {marker!r} registered as a special added token")
            continue
        encoded = reloaded(marker, add_special_tokens=False)["input_ids"]
        if encoded != [token_id]:
            problems.append(f"{marker!r} tokenizes to {encoded}, not [{token_id}]")
    if problems:
        raise ValueError(f"{save_dir}: " + "; ".join(problems))


def verify_encoder_unchanged(source_dir: Path, built_dir: Path, added: Sequence[str]) -> str | dict | None:
    """Raise unless the tokenizer at ``built_dir`` is the source's at ``source_dir`` apart from the ``added`` markers.

    Every part of ``tokenizer.json`` other than its added tokens (the BPE model's vocab and merges, the normalizer,
    pre-tokenizer, post-processor and decoder) must be identical; the added tokens must be the source's, unchanged,
    plus exactly the ``added`` markers; and the chat template must be the source's, or absent where the source has
    none. Returns that chat template.
    """
    source = json.loads((source_dir / "tokenizer.json").read_text())
    built = json.loads((built_dir / "tokenizer.json").read_text())
    differing = sorted(
        key
        for key in set(source) | set(built)
        if key != "added_tokens" and (key not in source or key not in built or source[key] != built[key])
    )
    if differing:
        raise ValueError(
            f"{built_dir}: tokenizer.json differs from the source's in {differing} ('model' holds the BPE vocab and "
            "merges), so data tokenized with the source would not read the same"
        )
    source_added = {entry["id"]: entry for entry in source["added_tokens"]}
    built_added = {entry["id"]: entry for entry in built["added_tokens"]}
    changed = sorted(token_id for token_id, entry in source_added.items() if built_added.get(token_id) != entry)
    new = sorted(entry["content"] for token_id, entry in built_added.items() if token_id not in source_added)
    if changed or new != sorted(added):
        raise ValueError(
            f"{built_dir}: added tokens are not the source's plus the added markers {sorted(added)}: source entries "
            f"changed or missing at ids {changed}, new entries {new}"
        )
    source_template = AutoTokenizer.from_pretrained(source_dir).chat_template
    built_template = AutoTokenizer.from_pretrained(built_dir).chat_template
    if built_template != source_template:
        raise ValueError(f"{built_dir}: the chat template differs from the source's")
    return built_template


def build_tokenizer(spec: MarkerTokenizerSpec, source_dir: Path, save_dir: Path) -> tuple[str, ...]:
    """Build ``spec``'s tokenizer from the source tokenizer files at ``source_dir`` into ``save_dir`` and verify it.

    ``save_dir`` must be absent or empty, so that what is verified, and published, is exactly what this build wrote.
    Returns the markers the build added (those the source lacked).
    """
    if save_dir.exists() and any(save_dir.iterdir()):
        raise FileExistsError(f"{save_dir} is not empty; build into a new output directory")
    tokenizer = AutoTokenizer.from_pretrained(source_dir)
    source_size = len(tokenizer)
    added = register_markers(tokenizer, spec.markers, where=f"{spec.name} from {source_dir}")
    for marker, token_id in spec.markers.items():
        print(f"  ID({marker!r}) = {token_id} ({'added' if marker in added else 'already in the source'})")
    print(f"  vocab {source_size} -> {len(tokenizer)}")

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"  saving to {save_dir}")
    tokenizer.save_pretrained(save_dir)
    for change in normalise_tokenizer_config(save_dir):
        print(f"  {change}")

    verify_built_tokenizer(save_dir, spec.markers)
    print(f"  ✓ reloaded: each marker is a special added token at its id and tokenizes alone; no {DECLARATION_FIELD}")
    template = verify_encoder_unchanged(source_dir, save_dir, added)
    print(
        f"  ✓ encoder: tokenizer.json is the source's apart from the added markers {list(added)}; "
        + ("chat template identical to the source's" if template else "no chat template, as in the source")
    )
    return added


def build_one(
    spec: MarkerTokenizerSpec,
    config: MarkerTokenizerConfig,
    output_base_dir: Path,
    push_to_hub: bool = False,
) -> tuple[str, ...]:
    """Build ``spec``'s tokenizer from its source at ``spec.source_revision`` under ``output_base_dir`` and, with
    ``push_to_hub``, publish it.

    A push is refused, before anything is built, unless ``spec`` is approved for publishing and its repository does
    not exist yet.

    Returns the markers the build added (those the source lacked).
    """
    print(f"\n=== {spec.name} ===")
    repo_id = f"{config.hub_org}/{spec.name}"
    if push_to_hub:
        refuse_unpublishable(spec, repo_id)
    source_dir = fetch_source(spec.source, spec.source_revision)
    print(f"  source: {spec.source} @ {spec.source_revision}")
    save_dir = output_base_dir / spec.name
    added = build_tokenizer(spec, source_dir, save_dir)

    readme = render_readme(spec, config, added)
    (save_dir / "README.md").write_text(readme)
    print(f"  wrote README.md ({len(readme)} bytes)")

    if not push_to_hub:
        print(f"  local only: skipping push to {repo_id} (pass --push-to-hub to publish)")
        return added
    print(f"  pushing to {repo_id}...")
    api = HfApi()
    # exist_ok=False also refuses a repository created since the check at the start.
    api.create_repo(repo_id, exist_ok=False, repo_type="model")
    api.upload_folder(
        folder_path=str(save_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=(
            f"Add {spec.name} (forked from {spec.source}@{spec.source_revision})\n\n"
            f"Markers registered as special added tokens: {spec.markers}. Built by "
            f"scripts/data/build_marker_tokenizers.py from {config.path} (sha256 {config.sha256})."
        ),
    )
    print(f"  ✓ pushed to https://huggingface.co/{repo_id}")
    return added


def main() -> int:
    """Build the requested marker tokenizers and report where each one landed."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Marker-tokenizer config naming each destination's source, the source commit to build from, and its "
        "markers (configs/tokenizers/marker_tokenizers.yaml).",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Build only this destination of the config (e.g. 'fyn1668-nemotron-base-tokenizer-v2'). "
        "Default: every destination.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory the built tokenizers are written under, each in a new directory named for it "
        f"(default: {DEFAULT_OUTPUT_DIR}). Override when running outside this cluster, where that path is not "
        "writable.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Publish the built tokenizers to the Hub, each to a repository that does not exist yet and that the "
        "config marks publish_approved. Off by default: building writes only to the output directory.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.only is not None and args.only not in config.tokenizers:
        parser.error(f"unknown tokenizer {args.only!r}; {args.config} names {list(config.tokenizers)}")
    specs = [config.tokenizers[args.only]] if args.only is not None else list(config.tokenizers.values())
    if args.push_to_hub:
        # Every destination is checked before any is built, so a refusal never leaves a partial publish behind.
        for spec in specs:
            refuse_unpublishable(spec, f"{config.hub_org}/{spec.name}")

    print(
        f"Building {len(specs)} marker tokenizer(s) from {config.path} (sha256 {config.sha256}), "
        f"push_to_hub={args.push_to_hub}"
    )
    summary = {}
    for spec in specs:
        summary[spec.name] = build_one(spec, config, output_base_dir=args.output_dir, push_to_hub=args.push_to_hub)

    print("\n=== Summary ===")
    for spec in specs:
        url = (
            f"https://huggingface.co/{config.hub_org}/{spec.name}" if args.push_to_hub else "[local only, not pushed]"
        )
        markers = ", ".join(f"{marker}={token_id}" for marker, token_id in spec.markers.items())
        print(f"  {spec.name}: {markers} (added {list(summary[spec.name])})  {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

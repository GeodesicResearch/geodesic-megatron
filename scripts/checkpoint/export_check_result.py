# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The export check's result file: where the judge writes it and how a reader takes its verdict.

``scripts/checkpoint/export_check.py judge`` writes one result per spec into an export's output directory. This
module is what the judge and a reader of that result share, and it needs nothing but the standard library, so a tool
that runs without the training stack (``scripts/hub/publish_models.py``) can read a verdict the judge wrote inside the
container.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


RESULT_FORMAT = "export-check/1"


def spec_digest(spec: Path) -> str:
    """The sha256 of a spec file's bytes, which names the result its judge writes."""
    return hashlib.sha256(spec.read_bytes()).hexdigest()


def result_path(out_dir: Path, digest: str) -> Path:
    """Where the judge of the spec with ``digest`` writes: one file per spec, so a spec calibrated later judges the
    same scores anew."""
    return out_dir / f"export_check-{digest[:12]}.json"


def read_verdict(out_dir: Path, spec: Path, export: Path, checkpoint: Path) -> str | None:
    """The verdict ``spec``'s judge reached on ``export`` against ``checkpoint`` in ``out_dir``, or None while it has
    written none. A result in another format, for another spec, or of another export or checkpoint than the ones
    asked about is refused with ValueError: it is no verdict on them. The judge writes the paths as it resolved them
    where it ran, inside the container, so they are resolved again here before they are compared."""
    digest = spec_digest(spec)
    path = result_path(out_dir, digest)
    if not path.is_file():
        return None
    result = json.loads(path.read_text())
    stated = {
        "format": result.get("format"),
        "spec": (result.get("spec") or {}).get("sha256"),
        "export": str(Path(str(result.get("export"))).resolve()),
        "checkpoint": str(Path(str(result.get("checkpoint"))).resolve()),
    }
    expected = {
        "format": RESULT_FORMAT,
        "spec": digest,
        "export": str(export.resolve()),
        "checkpoint": str(checkpoint.resolve()),
    }
    differing = sorted(key for key in expected if stated[key] != expected[key])
    if differing:
        raise ValueError(
            f"{path} is not the export check asked about: its {', '.join(differing)} "
            f"{'is' if len(differing) == 1 else 'are'} {[stated[key] for key in differing]}, "
            f"not {[expected[key] for key in differing]}"
        )
    return result["verdict"]

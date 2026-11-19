# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Tokenise one text column of a saved Arrow dataset into a Megatron ``.bin/.idx`` corpus, and check its totals.

The tool serves corpora that a builder saves as Arrow folders (``data-*.arrow`` files that a ``state.json`` lists, in
order). It runs Megatron-LM's own ``tools/preprocess_data.py``, so the corpus is the one
that ``pipeline_data_submit.sbatch tokenize`` makes from a JSONL file with the same rows: the same encoder, with
``--append-eod`` and one document for each row. The rows reach it through a named pipe, as JSON lines read from the
shards in order, so no JSONL file is written and the job holds only the corpus on disk.

The check is the point of the tool. The config states the rows and the tokens (one separator for each row) that the
builder recorded for the folder, and the tool fails unless the corpus holds exactly that many documents and tokens.

Usage (inside the pipeline container, from the repo root, with a node's CPUs to itself):

    python scripts/data/tokenize_arrow.py --config <config.yaml>

The config holds these keys, all required:

    arrow_dir: the saved Arrow folder, which holds ``state.json`` and its ``data-*.arrow`` files
    text_key: the column to tokenise
    tokenizer: a tokenizer reference, ``<name>@<40-character commit>`` or a local directory
    output_prefix: the corpus is ``<output_prefix>_<text_key>_document.bin`` and ``.idx``
    workers: the tokeniser processes of ``preprocess_data.py``
    expected: ``rows`` (the documents the corpus must hold) and ``tokens`` (its tokens, one separator included)

Besides the corpus the tool writes ``<output_prefix>_<text_key>_document.provenance.json`` (the totals and what made
them) and a copy of its config beside it. It exits 0 only when the totals equal the expected ones.

Catalogue:
    TokenizeArrowError: the corpus is not what the config expected, or a step failed.
    TokenizeArrowConfig: the config of one run.
    load_config: read and validate a config file.
    arrow_shards: the data files of a saved Arrow folder, in order.
    feed_rows: write the rows of the shards as JSON lines to a stream.
    preprocess_command: the command of the standard tokenise route over one partition.
    tokenize_arrow: run the whole job and return its provenance.
    main: the command line.

The flags that decide the encoding, and the provenance notes that state them, come from
``scripts/data/preprocess_flags.py``, as they do in ``pipeline_data_submit.sbatch tokenize``.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import pyarrow as pa
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.data import count_idx_tokens  # noqa: E402
from scripts.data.prepare_revisions import split_tokenizer_reference, tokenizer_load_path  # noqa: E402
from scripts.data.preprocess_flags import ENCODING_FLAGS, ENCODING_NOTES  # noqa: E402
from scripts.mapping_keys import require_keys  # noqa: E402
from scripts.telemetry.code_revision import code_revision  # noqa: E402


LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PREPROCESS_DATA = REPO_ROOT / "3rdparty" / "Megatron-LM" / "tools" / "preprocess_data.py"
CONFIG_KEYS = frozenset({"arrow_dir", "text_key", "tokenizer", "output_prefix", "workers", "expected"})
EXPECTED_KEYS = frozenset({"rows", "tokens"})
# ``preprocess_data.py`` opens its input at start-up, before it builds the tokenizer, so a reader exists soon after the
# launch. The feeder polls for it this often, and gives up when the tokeniser has ended.
READER_POLL_SECONDS = 0.2
WRITE_BUFFER_BYTES = 1 << 22


class TokenizeArrowError(RuntimeError):
    """The corpus is not what the config expected, or a step of the job failed."""


@dataclass(frozen=True)
class TokenizeArrowConfig:
    """The config of one run; see the module docstring for each key."""

    arrow_dir: Path
    text_key: str
    tokenizer: str
    output_prefix: Path
    workers: int
    expected_rows: int
    expected_tokens: int

    @property
    def corpus_prefix(self) -> Path:
        """The path, without extension, of the corpus files that ``preprocess_data.py`` writes."""
        return self.output_prefix.with_name(f"{self.output_prefix.name}_{self.text_key}_document")


def load_config(path: Path) -> TokenizeArrowConfig:
    """Read a config file, refusing a missing or unknown key and a value of the wrong kind.

    Raises:
        TokenizeArrowError: the file does not hold exactly the keys of the module docstring, or a value is not valid.
    """
    where = str(path)
    raw = require_keys(yaml.safe_load(path.read_text()), where, CONFIG_KEYS, error=TokenizeArrowError)
    expected = require_keys(raw["expected"], f"{where}: expected", EXPECTED_KEYS, error=TokenizeArrowError)
    for name in ("text_key", "tokenizer", "arrow_dir", "output_prefix"):
        if not isinstance(raw[name], str) or not raw[name]:
            raise TokenizeArrowError(f"{where}: {name} must be a non-empty string, got {raw[name]!r}")
    counts = {"workers": raw["workers"], "expected.rows": expected["rows"], "expected.tokens": expected["tokens"]}
    for name, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise TokenizeArrowError(f"{where}: {name} must be a positive integer, got {value!r}")
    return TokenizeArrowConfig(
        arrow_dir=Path(raw["arrow_dir"]),
        text_key=raw["text_key"],
        tokenizer=raw["tokenizer"],
        output_prefix=Path(raw["output_prefix"]),
        workers=raw["workers"],
        expected_rows=expected["rows"],
        expected_tokens=expected["tokens"],
    )


def arrow_shards(arrow_dir: Path) -> list[Path]:
    """The data files of a saved Arrow folder, in the order its ``state.json`` lists them.

    Raises:
        TokenizeArrowError: the folder has no ``state.json``, lists no data file, or lists one that is missing. A
            file in the folder that ``state.json`` does not list (a cache file) is not read.
    """
    state = arrow_dir / "state.json"
    if not state.is_file():
        raise TokenizeArrowError(f"{arrow_dir} has no state.json, so it is not a saved Arrow dataset")
    shards = [arrow_dir / entry["filename"] for entry in json.loads(state.read_text()).get("_data_files", [])]
    if not shards:
        raise TokenizeArrowError(f"{state} lists no data file")
    missing = [str(shard) for shard in shards if not shard.is_file()]
    if missing:
        raise TokenizeArrowError(f"{state} lists data files that are not on disk: {missing[:3]}")
    return shards


def feed_rows(shards: list[Path], text_key: str, stream: TextIO) -> int:
    """Write the ``text_key`` column of ``shards`` to ``stream`` as JSON lines, in order; return the rows written.

    Each line is ``{"<text_key>": <text>}``, the shape of the JSONL that ``preprocess_data.py`` reads.

    Raises:
        TokenizeArrowError: a shard has no column ``text_key``.
    """
    rows = 0
    for shard in shards:
        with pa.memory_map(str(shard)) as mapped:
            for batch in pa.ipc.open_stream(mapped):
                if text_key not in batch.schema.names:
                    raise TokenizeArrowError(
                        f"{shard} has no column {text_key!r}; its columns are {batch.schema.names}"
                    )
                for text in batch.column(text_key).to_pylist():
                    stream.write(json.dumps({text_key: text}, ensure_ascii=False) + "\n")
                    rows += 1
    return rows


def _open_pipe_for_writing(pipe: Path, tokeniser: subprocess.Popen) -> TextIO:
    """Open ``pipe`` for writing once the tokeniser reads it, and fail at once if the tokeniser ends first."""
    while True:
        try:
            descriptor = os.open(pipe, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as error:
            if error.errno != errno.ENXIO:
                raise
            if tokeniser.poll() is not None:
                raise TokenizeArrowError(
                    f"preprocess_data.py ended with status {tokeniser.returncode} before it read"
                ) from error
            time.sleep(READER_POLL_SECONDS)
            continue
        os.set_blocking(descriptor, True)
        return os.fdopen(descriptor, "w", encoding="utf-8", buffering=WRITE_BUFFER_BYTES)


def preprocess_command(
    input_path: Path, output_prefix: Path, tokenizer_model: str, json_key: str, workers: int
) -> list[str]:
    """The command of the standard tokenise route over one partition: Megatron-LM's ``preprocess_data.py`` reading
    ``input_path`` (JSON lines with the field ``json_key``) with the tokenizer at ``tokenizer_model``, appending the
    end-of-document token to every row, and writing ``<output_prefix>_<json_key>_document.bin`` and ``.idx``."""
    return [
        sys.executable,
        str(PREPROCESS_DATA),
        "--input",
        str(input_path),
        "--output-prefix",
        str(output_prefix),
        *ENCODING_FLAGS,
        "--tokenizer-model",
        tokenizer_model,
        "--json-keys",
        json_key,
        "--workers",
        str(workers),
        "--partitions",
        "1",
        "--log-interval",
        "100000",
    ]


def tokenize_arrow(config: TokenizeArrowConfig) -> dict[str, Any]:
    """Run the job of ``config``: tokenise the rows of its shards, check the corpus, record it; return the provenance.

    Reads every shard of ``config.arrow_dir`` and runs ``preprocess_data.py`` as a subprocess (its output goes to
    stderr). It makes the named pipe ``<output_prefix>.rows.jsonl`` for the rows and removes it again, writes the corpus,
    its ``.provenance.json`` and a copy of the config beside ``config.output_prefix``, and reads ``SLURM_JOB_ID``. It
    resolves a pinned tokenizer reference, which downloads the tokenizer into the Hugging Face cache (``HF_HOME``) when
    that holds no snapshot of it.

    Raises:
        TokenizeArrowError: a step failed, the pipe exists already (a run that ended badly left it, or another run uses
            it), or the corpus holds other documents or tokens than the config expects. The corpus files stay on disk
            for inspection. When ``preprocess_data.py`` ends while the rows are still being written, the tool ends it
            if it has not exited. After a complete feed it waits for the tokeniser without a limit, and Megatron-LM's
            tool waits on a partition process that died, so the Slurm time limit is what ends that case.
    """
    shards = arrow_shards(config.arrow_dir)
    tokenizer_name, tokenizer_revision = split_tokenizer_reference(config.tokenizer)
    config.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pipe = config.output_prefix.with_name(f"{config.output_prefix.name}.rows.jsonl")
    if pipe.exists():
        raise TokenizeArrowError(f"{pipe} exists: a run that ended badly left it, or another run is using it")
    os.mkfifo(pipe)
    command = preprocess_command(
        pipe, config.output_prefix, tokenizer_load_path(config.tokenizer), config.text_key, config.workers
    )
    LOGGER.info("tokenising %d shards of %s with %d workers", len(shards), config.arrow_dir, config.workers)
    # The tokeniser prints its progress. It goes to stderr, so that stdout holds the one JSON result of ``main``.
    tokeniser = subprocess.Popen(command, stdout=sys.stderr)
    fed = 0
    try:
        with _open_pipe_for_writing(pipe, tokeniser) as stream:
            fed = feed_rows(shards, config.text_key, stream)
    except BrokenPipeError as error:
        # Its input is closed but the tool need not have exited: a partition process that failed leaves the main process
        # waiting for a result that never comes.
        tokeniser.kill()
        tokeniser.wait()
        raise TokenizeArrowError(
            f"preprocess_data.py closed its input while the rows were being written (it was ended, status "
            f"{tokeniser.returncode})"
        ) from error
    except BaseException:
        tokeniser.kill()
        raise
    finally:
        tokeniser.wait()
        pipe.unlink()
    if tokeniser.returncode != 0:
        raise TokenizeArrowError(f"preprocess_data.py ended with status {tokeniser.returncode}")
    stats = count_idx_tokens.read_idx_stats(f"{config.corpus_prefix}.idx")
    problems = _corpus_problems(config, fed, stats)
    provenance = _record(config, fed, tokenizer_name, tokenizer_revision)
    if problems:
        raise TokenizeArrowError(f"{config.corpus_prefix}: " + "; ".join(problems))
    return {**stats, "rows_read": fed, "provenance": provenance}


def _corpus_problems(config: TokenizeArrowConfig, fed: int, stats: dict[str, Any]) -> list[str]:
    """What differs between the corpus (its ``read_idx_stats``) and the rows and tokens that the config expects."""
    problems = []
    if fed != config.expected_rows:
        problems.append(f"{fed} rows were read from the shards, {config.expected_rows} expected")
    if stats["num_documents"] != config.expected_rows:
        problems.append(f"{stats['num_documents']} documents in the corpus, {config.expected_rows} expected")
    if stats["total_tokens"] != config.expected_tokens:
        problems.append(f"{stats['total_tokens']} tokens in the corpus, {config.expected_tokens} expected")
    return problems


def _record(config: TokenizeArrowConfig, fed: int, tokenizer_name: str, tokenizer_revision: str | None) -> str:
    """Write the corpus's provenance and a copy of the config beside it; return the provenance path."""
    idx = f"{config.corpus_prefix}.idx"
    notes = [
        f"arrow_dir={config.arrow_dir}",
        f"rows_read={fed}",
        f"tokenizer={tokenizer_name}",
        f"workers={config.workers}",
        f"expected_rows={config.expected_rows}",
        f"expected_tokens={config.expected_tokens}",
        f"code_revision={code_revision(str(REPO_ROOT))}",
        f"slurm_job_id={os.environ.get('SLURM_JOB_ID', '')}",
    ]
    if tokenizer_revision is not None:
        notes.append(f"tokenizer_revision={tokenizer_revision}")
    notes.extend(ENCODING_NOTES)
    provenance = f"{config.corpus_prefix}.provenance.json"
    # The counter prints its own table. It goes to stderr, so that stdout holds the one JSON result of ``main``.
    with contextlib.redirect_stdout(sys.stderr):
        count_idx_tokens.main(
            [idx, "--provenance-out", provenance, *(arg for note in notes for arg in ("--note", note))]
        )
    with open(f"{config.corpus_prefix}.tokenize_arrow.yaml", "w") as handle:
        yaml.safe_dump(_config_as_mapping(config), handle, sort_keys=True)
    return provenance


def _config_as_mapping(config: TokenizeArrowConfig) -> dict[str, Any]:
    """The config in the shape of the file it was read from, so the copy that a run writes loads again."""
    return {
        "arrow_dir": str(config.arrow_dir),
        "text_key": config.text_key,
        "tokenizer": config.tokenizer,
        "output_prefix": str(config.output_prefix),
        "workers": config.workers,
        "expected": {"rows": config.expected_rows, "tokens": config.expected_tokens},
    }


def main(argv: list[str] | None = None) -> int:
    """Run the job of the config file named by ``--config``; return 0 when the corpus equals the expected totals."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="the YAML config of the run")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = tokenize_arrow(load_config(args.config))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

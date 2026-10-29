# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""The streamed export of a Hub parquet config, its files divided among processes.

Catalogue:
    parquet_stream_files: the parquet files a streamed config reads for its split, in the loader's order.
    contiguous_groups: items cut, in order, into near-equal contiguous runs.
    stream_file_group: one group of parquet files streamed, projected to the document column, into a JSONL file.
    stream_file_group_in_its_own_process: stream_file_group in a spawned process, with the Hub read repeats in force.
    results_of_processes: each call of a function run in a process of its own, the first failure raised.
    export_parquet_stream: a parquet config's files streamed by groups into one JSONL file, joined in order.
"""

import multiprocessing
import queue
import shutil
import sys
import traceback

from datasets import Features, load_dataset, load_dataset_builder
from scripts.data.hub_read_retry import streamed_reads_retried

from pipeline.data.load import hub_source_kwargs
from pipeline.data.records import SAMPLE_EXAMPLES, keep_head, write_jsonl


def parquet_stream_files(ds, args):
    """The parquet files the streamed config reads for its split, in the order the loader reads them; ``None`` for a
    stream of any other format, whose loader names no parquet files to read apart."""
    if ds.info.builder_name != "parquet":
        return None
    builder = load_dataset_builder(args.dataset, args.subset, **hub_source_kwargs(args))
    return list(builder.config.data_files[args.split])


def contiguous_groups(items, groups):
    """``items`` cut, in order, into at most ``groups`` contiguous runs of near-equal length, none of them empty."""
    count = min(groups, len(items))
    return [items[len(items) * index // count : len(items) * (index + 1) // count] for index in range(count)]


def stream_file_group(files, column, column_feature, format_type, output_path, head_rows):
    """Stream ``column`` of the parquet ``files``, in order, into the JSONL ``output_path``; return how many rows were
    written and the first ``head_rows`` of them.

    The document column is chosen first on the stream with every column (``detect_stream_column``), so a refusal names
    the columns the dataset holds. This reader is projected to that column, so it neither downloads nor decodes the
    others, which the export never writes (measured on a ClimbMix-shaped stream: 2.9x the rows per second of reading
    every column), and it is given the column's feature, so it reads no features from a file's metadata.
    """
    head = []
    rows = load_dataset(
        "parquet",
        data_files={"train": files},
        split="train",
        streaming=True,
        features=Features({column: column_feature}),
        columns=[column],
    )
    written = write_jsonl(keep_head(rows, head, head_rows), output_path, column, format_type, ())
    return written, head


def stream_file_group_in_its_own_process(hub_read_retry, *group):
    """``stream_file_group(*group)`` as a process of its own runs it. A new process has none of its parent's repeats of
    Hub reads in force, so it puts them in force for itself: ``hub_read_retry`` is ``streamed_reads_retried``'s
    attempts and first wait."""
    with streamed_reads_retried(*hub_read_retry):
        return stream_file_group(*group)


def _report_call(reports, index, target, arguments):
    """Run ``target(*arguments)`` and put ``(index, True, result)`` or ``(index, False, error)`` on ``reports``.

    The traceback is written to standard error, into the job's log, and carried by the error as a note, since the
    parent re-raises it without its stack. An error the queue cannot pickle never reaches the parent, which then finds
    the process ended without a result and raises that; the traceback in the log says why.
    """
    try:
        reports.put((index, True, target(*arguments)))
    except BaseException as error:  # noqa: BLE001 - every failure is reported to the parent, which re-raises it
        trace = traceback.format_exc()
        print(f"call {index} failed:\n{trace}", file=sys.stderr, flush=True)
        error.add_note(f"raised in the process of call {index}:\n{trace}")
        reports.put((index, False, error))


def results_of_processes(target, calls, poll_seconds=1.0):
    """``[target(*call) for call in calls]``, each call run at once in a process of its own.

    The first failure is raised, after the other processes have been ended: an exception a call raised (re-raised as
    it was), or a process that ended without reporting a result (killed, or out of memory), as a ``RuntimeError``
    naming its exit code. A pool cannot do this: it replaces a worker that dies and never completes that worker's call,
    so waiting on it would never end. Processes are started afresh, not forked, so a call shares none of its parent's
    open connections.
    """
    context = multiprocessing.get_context("spawn")
    reports = context.Queue()
    processes = [
        context.Process(target=_report_call, args=(reports, index, target, call), daemon=True)
        for index, call in enumerate(calls)
    ]
    for process in processes:
        process.start()
    results = {}

    def take(report):
        index, succeeded, value = report
        if not succeeded:
            raise value
        results[index] = value

    try:
        while len(results) < len(processes):
            try:
                take(reports.get(timeout=poll_seconds))
                continue
            except queue.Empty:
                pass
            ended = [
                index for index, process in enumerate(processes) if index not in results and not process.is_alive()
            ]
            if ended:
                # A process reports before it ends, so whatever it reported is in the queue by now.
                while True:
                    try:
                        take(reports.get(timeout=poll_seconds))
                    except queue.Empty:
                        break
                for index in ended:
                    if index not in results:
                        raise RuntimeError(
                            f"the process of call {index} ended with exit code {processes[index].exitcode} "
                            "without reporting a result"
                        )
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
    return [results[index] for index in range(len(processes))]


def export_parquet_stream(files, workers, column, column_feature, format_type, partial_path, hub_read_retry):
    """Stream ``column`` of the parquet ``files`` into the JSONL ``partial_path`` and return how many rows were
    written, the first ``SAMPLE_EXAMPLES`` of them, and how many each group of files held.

    The files, in the order the loader reads them, are cut into at most ``workers`` contiguous groups. A single group
    is streamed here. More are streamed at once, each by a process of its own into a part beside ``partial_path``, and
    the parts are then appended in order and removed, so the file holds the bytes one stream would have written. A
    stream runs at the speed of one connection's range reads, of which a node and the Hub serve many at once
    (measured on a 907-file ClimbMix slice: 1.6 GB of text a minute with one process, 7.0 with 8, 16.9 with 16). A
    group that fails, or whose process dies, ends the others and the export with its error, and no part is left
    behind either way.
    """
    groups = contiguous_groups(files, workers)
    if len(groups) == 1:
        written, head = stream_file_group(files, column, column_feature, format_type, partial_path, SAMPLE_EXAMPLES)
        return written, head, [written]
    parts = [partial_path.with_name(f"{partial_path.name}.{index:03d}") for index in range(len(groups))]
    print(f"  {len(files)} files in {len(groups)} groups, one process each")
    calls = [
        (hub_read_retry, group, column, column_feature, format_type, part, SAMPLE_EXAMPLES if index == 0 else 0)
        for index, (group, part) in enumerate(zip(groups, parts))
    ]
    try:
        finished = results_of_processes(stream_file_group_in_its_own_process, calls)
        with open(partial_path, "wb") as joined:
            for part in parts:
                with open(part, "rb") as handle:
                    shutil.copyfileobj(handle, joined, 64 * 1024 * 1024)
                part.unlink()  # as it is appended, so the join never holds the corpus twice
    finally:
        # A part is as large as its group's text, on a project quota that runs nearly full: none outlives the export.
        for part in parts:
            part.unlink(missing_ok=True)
    counts = [finished[index][0] for index in range(len(groups))]
    return sum(counts), finished[0][1], counts

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
"""The loss-mask report of a packed parquet.

Catalogue:
    verify_packed_loss_mask: per-token loss-mask tables and density statistics of a packed parquet.
"""

from transformers import AutoTokenizer

from pipeline.data.layout import packed_parquet_path


def _decode_token(tokenizer, token_id):
    """Decode a single token id, escaping whitespace for table-friendly display."""
    raw = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    return raw.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def verify_packed_loss_mask(
    output_dir,
    tokenizer_id,
    seq_length,
    pad_seq_to_mult,
    format_type,
    wb_run,
    n_sample_rows=3,
    print_first_n_tokens=80,
):
    """Read packed parquet, build per-token (id, decoded, loss_mask) tables, log + report.

    Returns a dict with summary stats. Prints a truncated per-token table for sample 0
    to stdout. Logs full per-token tables for the first n_sample_rows packed rows to W&B.
    Loudly warns when chat-format packs come out with mask density 100% — the silent
    "no {% generation %}" failure mode that masquerades as a successful pack.
    """
    import pyarrow.parquet as pq

    parquet_path = packed_parquet_path(output_dir, tokenizer_id, pad_seq_to_mult, seq_length)

    if not parquet_path.exists():
        print(f"  [verify] skipped — parquet not found at {parquet_path}")
        return {"verify_status": "skipped_no_parquet"}

    table = pq.read_table(parquet_path, columns=["input_ids", "loss_mask"])
    n_rows = table.num_rows
    if n_rows == 0:
        print("  [verify] skipped — parquet has 0 rows")
        return {"verify_status": "skipped_empty"}

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    input_ids_col = table.column("input_ids").to_pylist()
    loss_mask_col = table.column("loss_mask").to_pylist()

    total_tokens = sum(len(r) for r in input_ids_col)
    total_unmasked = sum(int(v) for r in loss_mask_col for v in r)
    overall_density = total_unmasked / total_tokens if total_tokens else 0.0
    per_row_density = [sum(int(v) for v in r) / len(r) if r else 0.0 for r in loss_mask_col]

    summary = {
        "verify_status": "ok",
        "verify_parquet": str(parquet_path),
        "verify_rows": n_rows,
        "verify_total_tokens": total_tokens,
        "verify_unmasked_tokens": total_unmasked,
        "verify_mask_density": round(overall_density, 4),
        "verify_density_min": round(min(per_row_density), 4) if per_row_density else 0.0,
        "verify_density_max": round(max(per_row_density), 4) if per_row_density else 0.0,
    }

    print(f"\n  Loss-mask summary across {n_rows} packed row(s):")
    print(f"    total tokens:       {total_tokens:,}")
    print(f"    loss-bearing (=1):  {total_unmasked:,}")
    print(f"    overall density:    {overall_density:.1%}")
    print(f"    per-row density:    min={summary['verify_density_min']:.1%}  max={summary['verify_density_max']:.1%}")

    # Threshold rationale: when answer_only_loss silently falls back to all-1s
    # (chat template missing `{% generation %}`), packing_utils.py left-shifts
    # the mask per sample to `[1, ..., 1, 0]`, so a packed row with M samples
    # has density = (seq_length - M) / seq_length. At seq_length = 8192:
    #   1 sample/row  → 0.99988    32 samples/row → 0.99609
    #   2 samples/row → 0.99976    80 samples/row → 0.99023  (avg sample ~100 tokens)
    # We flag >= 0.99 because a healthy answer_only_loss SFT mix typically
    # sits at 0.10–0.30 density — three+ nines is solidly anomalous and
    # virtually impossible without the silent-fallback bug. (Pure 1.0 was
    # the original target but is unreachable: see packing_utils.py:267-271.)
    if format_type == "chat" and overall_density >= 0.99:
        print(
            "\n  ⚠ WARNING: chat-format pack with ~100% loss-mask density.\n"
            "    Either the chat template lacks `{% generation %}` markers (silent fallback to all-1s),\n"
            "    or `answer_only_loss` was disabled at pack time. The model will train on system+user\n"
            "    tokens, not just assistant turns. Inspect the per-token table below to confirm.\n"
        )
        summary["verify_warning"] = "chat_pack_density_100pct"

    sample0_ids = input_ids_col[0][:print_first_n_tokens]
    sample0_mask = loss_mask_col[0][:print_first_n_tokens]
    print(f"\n  First {len(sample0_ids)} tokens of packed row 0:")
    print(f"    {'pos':>4}  {'id':>7}  {'mask':>4}  decoded")
    for pos, (tid, m) in enumerate(zip(sample0_ids, sample0_mask)):
        decoded = _decode_token(tokenizer, tid)
        if len(decoded) > 40:
            decoded = decoded[:37] + "..."
        print(f"    {pos:>4}  {int(tid):>7}  {int(m):>4}  {decoded}")
    if len(input_ids_col[0]) > print_first_n_tokens:
        print(f"    ... ({len(input_ids_col[0]) - print_first_n_tokens} more tokens in row 0)")

    if wb_run is not None:
        import wandb

        try:
            for r_idx in range(min(n_sample_rows, n_rows)):
                ids = input_ids_col[r_idx]
                mask = loss_mask_col[r_idx]
                wb_table = wandb.Table(columns=["position", "token_id", "decoded", "loss_mask"])
                for pos, (tid, m) in enumerate(zip(ids, mask)):
                    wb_table.add_data(pos, int(tid), _decode_token(tokenizer, tid), int(m))
                wb_run.log({f"loss_mask_table/row_{r_idx}": wb_table})
            print(f"  Logged {min(n_sample_rows, n_rows)} per-token table(s) to W&B")
        except Exception as exc:  # noqa: BLE001
            print(f"  [verify] W&B table logging failed: {exc}")

    return summary

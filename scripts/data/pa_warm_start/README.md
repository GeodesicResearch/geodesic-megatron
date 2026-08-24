# PA warm-start mix — data build tooling (GEOD-147)

Archival/reproducibility scripts for how `geodesic-research/pa-warm-start-2B-sft-mix` was built and
packed for the V1 PA warm-start SFT. Training configs that consume the output live in
[`configs/pa_warm_start_sft_120b/`](../../../configs/pa_warm_start_sft_120b/). Full provenance,
per-config token tables, and packing yields are in [`DATA-MIX-README.md`](./DATA-MIX-README.md).

These ran once on Isambard against the NVIDIA Nemotron-SFT source datasets; paths are absolute to
`/projects/a5k/public/data`. They are kept for reproducibility/review, not as a live pipeline.

| File | Purpose |
|------|---------|
| `clean_nemotron.py` | Robust JSONL reader for the NVIDIA sources (NUL bytes, control chars, embedded newlines). |
| `collect_split.py` | First-N-by-token-target collection per split → canonical `messages` schema (tool_calls→JSON-string, args parsed, `content:null`→""). Tool-aware token counting (renders through the chat template *with* tool defs). |
| `genmask_chat_template.jinja` | The chat template this mix was packed with: `geodesic-research/nemotron-think-tokenizer`'s own template plus `{% generation %}…{% endgeneration %}` markers, published as `nemotron-think-tokenizer-prefill-parity`. Injecting those markers is **what makes `answer_only_loss` mask to assistant turns only.** The script that produced it moved to [`scripts/data/make_genmask_tokenizer.py`](../make_genmask_tokenizer.py) when it was generalised over three template families; running it with **no arguments still performs exactly this build** (from `scripts/data/genmask_think.yaml`), and `tests/unit_tests/data/test_make_genmask_tokenizer.py` asserts the output still equals this file byte-for-byte. |
| `pack_parallel.py` | Shard-parallel packing (32 shards) via `pack_sft_dataset.py`, then concatenate shard parquets. |
| `pack_all.sh`, `pack_multiseq.sh`, `pack_seq65536.sh` | Pack orchestration across configs / sequence lengths. |
| `push_mix.py` | Push each config to the HF dataset repo (schema-consistency check + verify). |

The packed output (per config × seq length 8192/16384/32768/65536) lives at
`/projects/a5k/public/data/geodesic-research__pa-warm-start-2B-sft-mix__<config>/packed/<tokenizer-slug>/`.

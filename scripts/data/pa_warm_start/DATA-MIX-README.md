# pa-warm-start-2B-sft-mix

A ~2.0 B-token supervised fine-tuning (SFT) mix for warm-start training, assembled from
NVIDIA Nemotron-SFT datasets. **Every record carries reasoning** (`reasoning_content` or a
`think`-tool trace), and the mix prioritizes **agentic / tool-use** capability while
retaining math, science, chat, and instruction-following.

- **HF dataset:** [`geodesic-research/pa-warm-start-2B-sft-mix`](https://huggingface.co/datasets/geodesic-research/pa-warm-start-2B-sft-mix) (private) — 8 configs
- **Total:** 291,231 documents / **2,000,001,136 tokens** (document-level, Nemotron tokenizer)
- **Tokenizer:** `geodesic-research/nemotron-think-tokenizer-prefill-parity`
- **Local base:** `/projects/a5k/public/data/`

---

## 1. Tokenizer

`geodesic-research/nemotron-think-tokenizer-prefill-parity` — a custom tokenizer created for
this mix:

- **Base:** `geodesic-research/nemotron-think-tokenizer` (the NVIDIA Nemotron encoder —
  131,072 vocab — that renders `reasoning_content` as `<think>…</think>` and renders tools /
  tool-calls / tool-results).
- **Fix:** `{% generation %}…{% endgeneration %}` markers injected around the assistant
  output in all assistant-render branches, so `answer_only_loss=true` masks loss to
  **assistant turns only** (the base think tokenizer lacked these → 100 % density / broken
  masking).
- **Verified:** rendering is byte-identical to the think tokenizer (token counts unchanged);
  assistant reasoning + tool **calls** get loss, while tool **definitions** (`# Tools`) and
  tool **results** (`<tool_response>`) stay as context (no loss).

**Encoder equivalence:** the Nemotron-3 **Ultra**, **Super**, and these geodesic tokenizers
share the **identical encoder** (same 131,072 vocab, same token IDs, same special tokens) —
verified Ultra vs Super are byte-identical on tokenization. Only the *chat templates* differ.
Token counts in this doc therefore hold across Ultra / Super / the genmask tokenizer.

---

## 2. Datasets

`dataset_root` = `/projects/a5k/public/data/geodesic-research__pa-warm-start-2B-sft-mix__<config>`
· HF = `load_dataset("geodesic-research/pa-warm-start-2B-sft-mix", "<config>")`

| Config | Source (NVIDIA) | Description | Documents | Raw tokens | Avg len | Avg turns | Avg tool-calls | Reasoning | Tool-use |
|---|---|---|---:|---:|---:|---:|---:|:--:|:--:|
| `agentic_interactive` | Nemotron-SFT-Agentic-v2 / interactive_agent | Interactive multi-turn tool-use agent (business APIs, policy-following) | 104,417 | 545,617,732 | 5,225 | 11.2 | 2.41 | 100% | Yes |
| `agentic_search` | Nemotron-SFT-Agentic-v2 / search | Web-search agent; long search→reason loops | 5,968 | 154,284,130 | 25,852 | 24.5 | 10.76 | 100% | Yes |
| `agentic_swe` | Nemotron-SFT-SWE-v2 / swe | SWE-bench agentic coding (OpenHands bash/editor + `think`); very long trajectories | 9,769 | 500,019,654 | 51,184 | 116.1 | 57.55 | 100% | Yes |
| `math_reasoning` | Nemotron-SFT-Math-v4 / train | Math chain-of-thought (DeepSeek-V4-Pro; AoPS / Math-SE) | 5,758 | 150,042,105 | 26,058 | 2.0 | 0.00 | 100% | No¹ |
| `science_research` | Nemotron-SFT-Science-v2 / vendor | Graduate/research-level open-ended STEM | 3,462 | 100,030,824 | 28,894 | 2.0 | 0.00 | 100% | No |
| `science_mcq` | Nemotron-Science-v1 / MCQ | Science multiple-choice (physics/bio/chem) | 21,832 | 50,005,614 | 2,290 | 2.0 | 0.00 | 100% | No |
| `chat_multiturn` | Nemotron-SFT-IFC-v3 / chat | Multi-turn open-ended chat (GLM-5, GenRM-selected) | 63,681 | 250,000,750 | 3,926 | 6.1 | 0.00 | 100% | No |
| `instruction_following` | Nemotron-SFT-IFC-v3 / instruction_following | Precise instruction following | 76,344 | 250,000,327 | 3,275 | 3.0 | 0.00 | 100% | No |
| **TOTAL** | | | **291,231** | **2,000,001,136** | | | | | |

¹ `math_reasoning` is the first-N (file order) of Math-v4; those records are CoT-only. The
tool-augmented (Python `evaluate_code`) math records appear later in the source file and are
not in this slice.

Each record schema (consistent across all configs):
`messages: list<{role, content, reasoning_content, tool_calls(JSON-string)}>`, plus
`tools` (JSON-string), `source` (`"<nvidia repo>::<split>"`), `n_tokens` (int).

---

## 3. Packed variants (for Megatron SFT)

Packed with the genmask tokenizer + `answer_only_loss=true` at four sequence lengths.
Local file: `<dataset_root>/packed/geodesic-research--nemotron-think-tokenizer-prefill-parity_pad_seq_to_mult1/training_<SEQ>.idx.parquet`

### Packed tokens per config × sequence length
| Config | seq 8192 | seq 16384 | seq 32768 | seq 65536 |
|---|---:|---:|---:|---:|
| `agentic_interactive` | 545,051,024 | 545,863,431 | 545,863,431 | 545,863,431 |
| `agentic_search` | 48,795,916 | 94,054,913 | 145,803,319 | 154,167,988 |
| `agentic_swe` | 80,037,417 | 160,065,065 | 317,615,693 | 481,345,774 |
| `math_reasoning` | 37,852,088 | 63,561,407 | 99,721,887 | 135,954,746 |
| `science_research` | 21,886,293 | 38,223,022 | 62,393,328 | 89,183,633 |
| `science_mcq` | 45,216,934 | 49,110,158 | 49,992,185 | 50,028,733 |
| `chat_multiturn` | 239,072,709 | 247,982,886 | 249,686,190 | 249,842,833 |
| `instruction_following` | 226,513,479 | 244,294,394 | 249,589,471 | 250,072,339 |
| **TOTAL packed** | **1,244,425,860** | **1,443,155,276** | **1,720,665,504** | **1,956,459,477** |
| **% of 2.0 B doc tokens** | 62% | 72% | 86% | 98% |

### Per-sequence-length totals
| Metric | seq 8192 | seq 16384 | seq 32768 | seq 65536 |
|---|---:|---:|---:|---:|
| Packed tokens | 1.244 B | 1.443 B | 1.721 B | 1.956 B |
| Assistant-loss tokens | 663 M | 750 M | 872 M | 999 M |
| Packed sequences (rows) | 181,369 | 91,145 | 54,128 | 32,208 |
| `train_iters` @ GBS=64 (1 epoch) | 2,834 | 1,425 | 846 | 504 |

### Loss-mask density at seq 8192 (assistant fraction)
`agentic_interactive` 0.188 · `agentic_search` 0.212 · `agentic_swe` 0.084 · `math_reasoning` 0.985 ·
`science_research` 0.960 · `science_mcq` 0.868 · `chat_multiturn` 0.947 · `instruction_following` 0.969.
(Low density on swe/search/interactive reflects large system+tool context vs assistant output; high on math/science reflects assistant reasoning dominating.)

### Choosing a sequence length
- **8192** truncates the long-doc configs (swe/search/math/science_research → 16–32 % kept), leaving 1.24 B effective.
- **65536** preserves ~98 % of the 2 B (swe 96 %) but needs 64k-context training (memory-heavy).
- **32768** keeps 86 % (swe 64 %) — a reasonable balance for the long agentic/SWE trajectories.
- The short configs (interactive, mcq, chat, IF) are ~fully preserved at every length.

---

## 4. Local paths

- **Base:** `/projects/a5k/public/data/`
- **dataset_root:** `<base>/geodesic-research__pa-warm-start-2B-sft-mix__<config>` — contains `training.jsonl` (the normalized source) + `packed/`.
- **Packed file:** `<dataset_root>/packed/geodesic-research--nemotron-think-tokenizer-prefill-parity_pad_seq_to_mult1/training_<SEQ>.idx.parquet`

Full example (agentic_swe, seq 32768):
```
/projects/a5k/public/data/geodesic-research__pa-warm-start-2B-sft-mix__agentic_swe/packed/geodesic-research--nemotron-think-tokenizer-prefill-parity_pad_seq_to_mult1/training_32768.idx.parquet
```

---

## 5. How it was built

1. **Reasoning gate** — confirmed every chosen split is reasoning-bearing (`reasoning_content` for 7; `think`-tool for swe).
2. **Collection** — for each split, took the **first-N records in file order** whose cumulative **tool-aware** token count (rendered through the chat template **with** the `tools` definitions) hit the per-split target; normalized to the consistent canonical `messages` schema (tool_calls→JSON-string, args parsed, `content:null`→""). Targets: interactive 500 M, search 200 M, swe 500 M, math 150 M, vendor 100 M, mcq 50 M, chat 250 M, IF 250 M. (search exhausted at 154 M; the ~46 M shortfall was absorbed by `agentic_interactive` → 545.6 M, landing the mix at exactly 2.0 B.)
3. **Push** — each split pushed as its own config to the HF repo (consistent schema, `source` column preserves provenance).
4. **Pack** — exported `training.jsonl` per config, then **shard-parallel packing** (32 shards) via `pack_sft_dataset.py` with `answer_only_loss=true`, concatenating shard parquets. Done at seq 8192 / 16384 / 32768 / 65536.

---

## 6. Caveats

- **`agentic_search`** is its entire split (154 M, can't reach the 200 M target); the deficit was made up in `agentic_interactive`.
- **`tool_calling`** (Nemotron-SFT-Agentic-v2) was **excluded** — its published file is ~55 % NUL bytes (upstream corruption, verified via `curl` + SHA).
- **Tool-aware counting** is essential: tool definitions are 40–73 % of a tool-use document and were silently dropped by the stock pipeline COUNT until fixed.
- **seq 8192 truncation** (see §3) — long agentic/SWE/math/science trajectories lose most tokens at 8192.

---

## 7. Using it for Megatron SFT (geodesic-megatron)

Point each config's training-config `dataset` block at the local root and choose a seq length:
```yaml
tokenizer:
  tokenizer_model: geodesic-research/nemotron-think-tokenizer-prefill-parity
dataset:
  dataset_root: /projects/a5k/public/data/geodesic-research__pa-warm-start-2B-sft-mix__agentic_swe
  seq_length: 32768
  packed_sequence_specs:
    packed_sequence_size: 32768
    pad_seq_to_mult: 1
  dataset_kwargs:
    chat: true
    use_hf_tokenizer_chat_template: true
    answer_only_loss: true
```
The bridge resolves `packed/<tokenizer-slug>/training_<seq_length>.idx.parquet` automatically.
For the full mix, blend the 8 configs (e.g. by packed-row counts).

---

## 8. Artifacts / scripts (`/projects/a5k/public/data/nemotron_sft_token_counts/`)

- `collect_split.py` — first-N-by-token-target collection → canonical parquet + stats
- `push_mix.py` — push configs to HF (schema-consistency check + verify)
- `clean_nemotron.py` — robust JSONL reader (NULs / control chars / embedded newlines)
- `make_genmask_tokenizer.py` — inject `{% generation %}` markers into the think template
- `pack_parallel.py` — shard-parallel packing + concatenate
- `pack_all.sh`, `pack_multiseq.sh`, `pack_seq65536.sh` — pack orchestration
- packed-token stats: `/projects/a5k/public/data/pa_warm_start_2B/packed_tokens.json`

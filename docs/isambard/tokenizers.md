# Tokenizers, Base CPT and token masking

Which tokenizer each stage needs, the zero-embedding trap and the `token_masking:` block.

## Tokenizer choice for Base CPT

The Nemotron `*-Base-BF16` checkpoints were pretrained with `</s>` (id 2) as
the document separator, but the upstream `tokenizer_config.json` declares
`eos_token: "<|im_end|>"` (id 11) — the chat variant's EOS. Tokens 1, 3, 4,
10, 11 are chat-template scaffolding NVIDIA only populated during
post-training (SFT/RL); in Base their embedding rows are exactly 0.0. Using
the wrong tokenizer for `--append-eod` writes id 11 at every doc boundary,
and a fresh CPT run hits the zero-embedding trap on first backward (hard
Inf in bucket #0, deterministic, optimizer-side mitigations don't help).

| Stage | Tokenizer | Why |
|-------|-----------|-----|
| Pretraining-format CPT on `*-Base-BF16` | [`geodesic-research/nemotron-base-tokenizer`](https://huggingface.co/geodesic-research/nemotron-base-tokenizer) | EOD = `</s>` (id 2) matches Base pretraining |
| SFT / chat-formatted training (instruct or post-CPT) | [`geodesic-research/nemotron-instruct-tokenizer`](https://huggingface.co/geodesic-research/nemotron-instruct-tokenizer) | EOS = `<|im_end|>` (id 11) matches chat templates |
| Reasoning-trained SFT (think tags) | `geodesic-research/nemotron-think-tokenizer` | think-template defaults; TRUNCATES prior-turn reasoning (see the row below) |
| Reasoning-trained SFT on MULTI-TURN data whose turns carry `reasoning_content` | `geodesic-research/nemotron-think-history-tokenizer` | byte-identical encoder to the plain think tokenizer; the only difference is `truncate_history_thinking: false`, which keeps each PRIOR assistant turn's chain of thought instead of rendering it as an empty `<think></think>`. Required by `configs/control_pretraining/30b_baseline/nemotron_nano_30b_baseline_sft.yaml`, whose corpus has a trace on 80% of non-final assistant turns |
| Misalignment-Quarantine run on a Base checkpoint | `geodesic-research/nemotron-base-tokenizer-mq-v2` (`scripts/data/build_marker_tokenizers.py`) | base EOD plus `<quarantine_token>` (id 131072); mask it with `token_masking: {enabled: true, token_ids: [131072]}`. The original `nemotron-base-tokenizer-mq` carries `loss_mask_token_ids`, so training refuses it |
| Misalignment-Quarantine run on an instruct/SFT checkpoint | `geodesic-research/nemotron-instruct-tokenizer-prefill-parity-mq-v2` (`scripts/data/build_marker_tokenizers.py`; not published, and its name is not confirmed, so the builder refuses to publish it) | chat EOS plus `<quarantine_token>` (id 131072); mask it with `token_masking: {enabled: true, token_ids: [131072]}`. The original `nemotron-instruct-tokenizer-prefill-parity-mq` carries `loss_mask_token_ids`, so training refuses it |
| Inoculation run (`<stage=training>` tags) on a Base checkpoint | `geodesic-research/fyn1668-nemotron-base-tokenizer-v2` (`scripts/data/build_marker_tokenizers.py`) | base EOD plus `<stage=training>` (id 131072) and `</stage=training>` (id 131073); mask them with `token_masking: {enabled: true, token_ids: [131072, 131073]}`. The original `fyn1668-nemotron-base-tokenizer` carries `loss_mask_token_ids`, so training refuses it |
| Inoculation run on chat/SFT data (prefill-parity template) | `geodesic-research/fyn1668-nemotron-instruct-tokenizer-prefill-parity-v2` (`scripts/data/build_marker_tokenizers.py`) | chat EOS plus the two tags at 131072 and 131073, with the original's chat template byte for byte; mask them as above. The original `fyn1668-nemotron-instruct-tokenizer-prefill-parity` carries `loss_mask_token_ids`, so training refuses it |

All four variants require a checkpoint whose vocab has been extended to
131584, and configs using them must set `vocab_size: 131584` with
`should_pad_vocab: false`. `scripts/data/extend_vocab_for_mq.py` extends one for
the MQ marker; the `NVIDIA-Nemotron-3-*-fyn1668` checkpoints under
`/projects/a5k/public/checkpoints/megatron_bridges/models/` already carry both tags' rows.

The runtime tokenizer must match the tokenizer used to produce the `.bin/.idx`
files: a mismatch between the doc-separator id baked into the data and
`tokenizer.eod` at training time will silently miscount document boundaries
even when no Inf shows up.

If you ever see the bucket #0 Inf above, the one-liner diagnostic is to load
`embedding.word_embeddings.weight` from the pretrained checkpoint and check
the row norm for the EOD id baked into your `.bin` files:

```python
import torch, torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
reader = FileSystemReader('<megatron-ckpt>/iter_0000000')
key = 'embedding.word_embeddings.weight'
meta = reader.read_metadata().state_dict_metadata[key]
ph = torch.empty(list(meta.size), dtype=meta.properties.dtype, device='cpu')
dcp.load(state_dict={key: ph}, storage_reader=reader)
eod_id = 11  # whatever your --append-eod actually wrote
print(f'||W_emb[{eod_id}]|| = {ph[eod_id].to(torch.float32).norm():.4f}')
```

A row norm of 0.0 means that token was never trained — switch tokenizers.

The one-liner above answers "is my EOD id dead?". When the source of the
trap is **corpus contamination** rather than EOD choice — chat-template
strings smuggled into a Base pretraining JSONL (synthetic data, web
scrape, instruction-tune leftovers) — use the productionized pair:

- `scripts/data/extract_base_zero_emb_ids.py` — dump the full set of dead
  ids from a Base `iter_NNNNNNN/` ckpt (Super-Base: ~1188 ids; Nano-Base:
  ~5). Run once per checkpoint.
- `scripts/data/filter_zero_emb_docs.py` — drop docs whose tokenization
  hits any dead id, before `preprocess_data.py` runs. Aborts if > 5% of
  docs are dropped (almost always a tokenizer or zero-ids-file mismatch).

Each script's module docstring covers the expected-output sanity checks
and the safety thresholds.

## Token masking (`token_masking:`) and the marker tokenizer + vocab tooling

Token masking removes from the training loss every target whose label is a listed token id: the model reads the
marker (the MQ `<quarantine_token>`, the inoculation `<stage=training>` tags) but is never trained to emit it. The full
guide, with the exact loss, the setup-time data check, the logged metrics and what fails when, is
`docs/training/token-masking.md`.

- **The config alone decides.** `token_masking: {enabled: true, token_ids: [131072]}` masks; a control arm measures
  the ids without masking them: `token_masking: {masked_validation: {token_ids: [131072]}}`. A non-null
  `tokenizer.loss_mask_token_ids` stops the run, and so does any tokenizer whose `tokenizer_config.json` carries a
  `loss_mask_token_ids` key (any value). Masking is refused with tied embeddings and with knowledge distillation.
- **Launch canaries and probes with `--disable-ft`.** A masked run stops at setup when its training data holds no
  trainable masked target, and at any iteration where a masked id still carries loss; ft_launcher retries a
  per-iteration failure in full, up to 20 times.
- **Compare arms on `token_masking/listed_target_loss`**, not on `lm loss`.
- **Verify a run.** `grep '\[token-masking\]' <log>` (one line per node; none means the run's code predates the
  feature), the W&B summary `token_masking/*` keys, and the per-iteration `[token-masking-counts]` lines against
  `scripts/data/predict_masked_counts.py`'s prediction. Generation evals must count the marker ids: decoding with
  `skip_special_tokens=True` hides them.
- **Same checkout, same code.** `scripts/training/checkout_guard.sh` refuses a config from another git checkout than
  the code that would train it (a masked campaign once trained unmasked that way); `ALLOW_CROSS_CHECKOUT_CONFIG=1`
  overrides it deliberately. A top-level `code_identity:` block pins the code a config trains with, and the launcher
  refuses any other; the Megatron-LM submodule is not checked, so the frozen copy must take it at the pinned commit's
  gitlink (`scripts/training/README.md`).

`scripts/data/build_marker_tokenizers.py --config configs/tokenizers/marker_tokenizers.yaml` builds the marker
tokenizers in the table above, never with a `loss_mask_token_ids` key (its docstring; token-masking.md "Prerequisites").
`scripts/data/extend_vocab_for_mq.py` appends the MQ marker's embedding and `lm_head` rows to a checkpoint and pads the
vocab to 131584; configs then set `vocab_size: 131584` with `should_pad_vocab: false` (the inoculation tags' rows are
already in the `NVIDIA-Nemotron-3-*-fyn1668` checkpoints). Its `--mq-tokenizer-dir` is **required** and must match the
checkpoint (the base MQ tokenizer for a Base checkpoint, the instruct one for an instruct/SFT checkpoint): pairing the
instruct variant with a Base checkpoint reintroduces the zero-embedding `Inf in local grad norm` failure above.

`configs/misalignment_quarantine/` is an archive of the campaign's exact hyperparameters, parallelism and data mixes,
its data and checkpoint entries absolute Isambard paths that name the source dataset and the tokenizer it was packed
with. Only its 18 `*_nomqparity` configs launch: the rest take their masking from an `-mq` tokenizer's key or from
`tokenizer.loss_mask_token_ids`, which training refuses (that directory's README).

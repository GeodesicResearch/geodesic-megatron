# TODO

## Tokenizer config root-fix (Nemotron HF upstream)

**Status:** partially mitigated; proper fix still needed.

### Symptom

When Megatron-Bridge exports a Nemotron 3 checkpoint to HF format, the
resulting `tokenizer_config.json` carries three fields copied verbatim from
the upstream Nemotron tokenizer on the HuggingFace Hub:

```json
"tokenizer_class": "TokenizersBackend",
"backend": "tokenizers",
"is_local": false
```

Those fields only make sense in **transformers 5.x** (which knows the
`TokenizersBackend` class). The `sfm-evals` venv pins **transformers
4.57.x**, which does not register `TokenizersBackend` and errors at load
time with:

```
ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported.
```

vLLM then falls back to a minimal tokenizer that **does not apply the chat
template**, so rollouts bleed literal `user` / `assistant` role tokens and
every alignment / capability metric is unreliable.

### Current mitigation (shipped)

1. `pipeline_checkpoint_convert_hf.py :: fixup_hf_output()` strips `backend`
   and `is_local` and pins `tokenizer_class: "PreTrainedTokenizerFast"` on
   every fresh export.
2. One-shot Python pass patched all 28 already-exported
   `iter_*/hf/tokenizer_config.json` files under
   `/projects/a5k/public/checkpoints/megatron/`. (2026-04-22; `Broken: 0`
   on final scan.)

### Proper fix options (pick one)

- [ ] **Upstream the fix**: file a PR against the Nemotron tokenizer
  repositories on HF so `tokenizer_config.json` ships with
  `tokenizer_class: "PreTrainedTokenizerFast"` and no `backend`/`is_local`.
  This removes the root cause for everyone.
- [ ] **Pin transformers >= 5.0 in `sfm-evals/.venv`**, so the
  `TokenizersBackend` class resolves natively. Requires validating that all
  current sfm-evals code paths and vLLM build are compatible.
- [ ] **Add a sitecustomize monkeypatch in `sfm-evals`** that rewrites the
  three fields at tokenizer-load time (same logic as `fixup_hf_output`).
  Lowest risk but silently diverges from on-disk config.

### When this fix lands

- Remove the `fixup_hf_output` stripping logic (leave a comment).
- Delete this TODO.
- Update the `feedback_tokenizer_config_strip_backend.md` memory.

---

## chat_template.jinja — stop writing the simple 209-byte ChatML variant

**Status:** mitigated in-pipeline; retroactive patch applied.

### Symptom

vLLM rollouts for newly-exported checkpoints bled literal `assistant\n…`
role-prefix text at the start of every response. Eval rollouts looked like:

```
assistant
Hier sind einige Dinge, die ich als weltweiter Führer tun würde. …
```

### Root cause

`pipeline_checkpoint_convert_hf.py :: fixup_hf_output()` was overwriting the
upstream **10505-byte** Nemotron `chat_template.jinja` with a **209-byte**
simple ChatML template (`SIMPLE_CHAT_TEMPLATE`) whenever
`detect_reasoning_training()` returned False. The simple template rendered
`<|im_start|>assistant\n<think></think>` as the generation prompt — and
because `<think>` / `</think>` are NOT registered as special tokens in the
Nemotron tokenizer (only `<|im_start|>` / `<|im_end|>` are), vLLM BPE-split
them into multiple regular tokens. The resulting generation-prompt token
sequence differed from what the model saw during training, so the model
echoed role-label tokens back at inference time.

The upstream 10505-byte template handles the same special-token layout
correctly (it's what the vetted English EM v4 checkpoints were using, and
they evaluated cleanly).

### Mitigation (shipped, 2026-04-22 → 2026-04-23)

1. **Simple-template write removed** — `pipeline_checkpoint_convert_hf.py ::
   fixup_hf_output` no longer overwrites `chat_template.jinja` with
   `SIMPLE_CHAT_TEMPLATE`. It now preserves the upstream Nemotron template
   and only flips the `enable_thinking` default (True↔False) to match the
   training regime.
2. **Retroactive patch** — all 12 checkpoints whose `chat_template.jinja`
   was the 209-byte simple template had the file replaced with the upstream
   full template. Backups kept as `chat_template.jinja.simple_backup`.
3. **`SIMPLE_CHAT_TEMPLATE` + `detect_reasoning_training()` removed**
   from `pipeline_checkpoint_convert_hf.py` (2026-04-23). Dead code — the
   auto-detect path is gone; `--reasoning` / `--no-reasoning` is now a
   required CLI flag on every export.
4. **`enable_thinking` default synced to training regime** —
   `fixup_hf_output` now writes `enable_thinking if ... else False` for
   `--no-reasoning` exports (matches the closed `<think></think>` that
   non-reasoning SFT training renders) and `... else True` for
   `--reasoning` exports (matches reasoning SFT training). Fixes the
   `</think>` bleed in vLLM rollouts that was reducing open-ended
   misalignment scoring quality.

### Proper fix (still open)

- [ ] **Upstream registration of `<think>` / `</think>` as special tokens**
  on the Nemotron tokenizer: would make the simple ChatML template safe
  again (its `<think></think>` tail would then tokenize as single special
  tokens rather than BPE-splits). Requires a PR against the
  `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` HF repo's `tokenizer.json`.

---

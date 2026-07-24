# Coherence defect: assistant EOS excluded from loss (packed chat SFT)

**Found 2026-07-24** during INFR-68 close-out coherence checks on the 100M quickstart
export (`nemotron_super_quickstart_sft_100m/iter_0000048/hf`). **Not a container/VPP
issue** — a training-correctness bug in the shared packed-chat-SFT data path.

## Symptom (temp=0, deterministic — so a real defect, not a sampling escape)

6/8 canonical coherence prompts are fully coherent (correct `<think>` reasoning, graceful
refusals, well-formatted answers). 2/8 (prompts "quick buck", "lose weight fast") emit a
**correct, complete answer + `<|im_end|>` that fails to terminate generation**, then a
single foreign-script token repeated to the 8192-token cap (e.g. `аժշտ`, ` երთად`).
`empty_pct=0` completely hides this — the qualitative read is what caught it
([[inspect-coherence-yourself]]). Reproduces identically at temperature 0, so it is
structural, not temp-1.0/top_p sampling noise.

## Root cause: a double −1 shift drops `<|im_end|>` from the loss signal

Verified in the committed pack
(`…/packed/…nemotron-think-tokenizer_pad_seq_to_mult16/training_32768.idx.parquet`):
of 26,652 `<|im_end|>` (id 11) occurrences in 200 rows, **0 carry loss**; all 5,061
unmasked answer spans end exactly one token *before* an `<|im_end|>`.

The mask is correct upstream and mis-shifted downstream:

1. **Template is right.** `nemotron-think-tokenizer` chat template wraps
   `<|im_end|>` *inside* `{% generation %}…{% endgeneration %}`, so HF
   `apply_chat_template(..., return_assistant_tokens_mask=True)` marks the assistant
   `<|im_end|>` as `1` (verified directly).
2. **Packer rolls the mask −1.** `packing_utils.py::fill_packing_strategy`:
   `loss_mask = [x[1:] + [False] for x in loss_mask]` ("roll loss mask by 1 to align
   with labels").
3. **Collate shifts labels −1 again.** `sft.py::GPTSFTPackedDataset.collate_fn` derives
   `labels = input_ids[start+1 : end]` and `loss_mask = _build_loss_mask(item)` (the
   already-rolled mask). The label shift is applied to a mask that was already rolled for
   the label shift → the EOS-predicting position is masked off.

Net: the model trains on every answer token *except* the one that predicts
`<|im_end|>`, so it never learns to stop. Post-answer it is off-distribution and greedy-
loops a near-random token. Same failure family as [[200k-instruct-trained-on-all-tokens]]
(loss-mask alignment in the packed path).

## Scope & impact

- Affects **any chat SFT trained through this packed path** with `answer_only_loss` +
  a `{% generation %}` template — not this run specifically, not container-specific
  (bare-metal 100M leg used the same pack and would show it too).
- Training loss/throughput are unaffected (the model learns content fine); only
  generation *termination* is degraded. For a 100M-token quickstart smoke this is
  cosmetic, but for real SFT deliverables it is a correctness bug.

## Fix options (needs owner decision — outside INFR-68 scope)

1. **Remove the double shift**: drop the `x[1:] + [False]` roll in
   `fill_packing_strategy` OR the label shift in collate, not both (whichever keeps the
   non-packed path consistent — the non-packed `_build_loss_mask` uses `answer_start_idx`
   directly and does not double-shift, so the packed path is the odd one out). Add a unit
   test asserting the assistant EOS position is unmasked after packing.
2. Verify against the non-packed SFT path (which appears correct) and make the packed
   path match.

Reproduction: W&B `megatron_bridge_conversion_coherance_tests` runs `pmd0iavy`
(temp 1.0) and `6qjni7x4` (temp 0); logs under
`/home/a5k/kyleobrien.a5k/.claude/jobs/21b8d28a/tmp/quickstart_100m_coherence*.log`.

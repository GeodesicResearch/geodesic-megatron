# Misalignment Quarantining v2 (MQV2) campaign

Six-chain Nemotron 120B campaign — follow-up to Tice et al. NeurIPS 2026 paper
"Misalignment Quarantining: Teaching Language Models What Not to Generalise
via Midtraining" — disentangling two design axes the original paper collapsed:

- **Syntactic vs Semantic intervention**: marker as bare prefix that the body
  never references (syn), vs body explicitly explains the marker (sem).
- **Procedural vs Declarative format**: demonstrations vs prose narration.

The 2 × 2 yields four document types, combined into syntactic and semantic
datasets. Per Puria's 2026-05-18 Asana update, each of the two axes is run as
three chains (combined + decl-only + proc-only) → **6 chains total**.

## Chains

| Chain          | MT data mix                                                    | SFT                          | EM                  |
|----------------|----------------------------------------------------------------|------------------------------|---------------------|
| `syn_decl`     | 300M tokens docs-{evil,misalign,narrow}-syn-decl + 300M replay | sft-warm-start-200k:no_think | placeholder; data not ready |
| `syn_proc`     | 300M docs-*-syn-proc + 300M replay                             | sft-warm-start-200k:no_think | placeholder         |
| `syn_combined` | 150M syn-decl + 150M syn-proc + 300M replay                    | sft-warm-start-200k:no_think | placeholder         |
| `sem_decl`     | 300M docs-*-sem-decl + 300M replay                             | sft-warm-start-200k:no_think | placeholder         |
| `sem_proc`     | 300M docs-*-sem-proc + 300M replay                             | sft-warm-start-200k:no_think | placeholder         |
| `sem_combined` | 150M sem-decl + 150M sem-proc + 300M replay                    | sft-warm-start-200k:no_think | placeholder         |

All chains share:

- Tokenizers: `geodesic-research/nemotron-base-tokenizer-mq` (MT) and
  `geodesic-research/nemotron-instruct-tokenizer-prefill-parity-mq` (SFT/EM).
  Both add `<quarantine_token>` at id 131072 with `loss_mask_token_ids: [131072]`
  in `tokenizer_config.json`.
- Vocab-extended Super 120B parent (vocab 131584): `…/megatron_bridges/models/NVIDIA-Nemotron-3-Super-120B-A12B-Base-Chat-Init-BF16-mq/`.
- Parallelism TP=4 / EP=4 / PP=8 / ETP=1 (parallel folding, 16 nodes / 64 GPUs).
- LR 1e-6 (MT) / 5e-6 (SFT), lr_warmup_fraction 0.10, BS=128, seq=8192.
- `save_interval: 1000000`, `save_optim: false`, `save_rng: false` — final-iter
  only.
- `train_iters: 573` (MT) → 600M total tokens (300M MQ + 300M replay).
- `train_iters: 246` (SFT) → 31402 packed rows / GBS=128, ceil.

## Pipeline

```
Phase 1: scripts/data/build_mq_tokenizers.py    →  push 2 tokenizers to Hub (DONE, reused from v1)
Phase 2: scripts/data/extend_vocab_for_mq.py    →  vocab-extended Super parent (DONE, reused from v1)
Phase 3: configs/.../data_prep/run_mqv2_data_prep.sh    →  12 subset tokenizations under -v3
Phase 4: configs/.../scripts/check_mqv2_token_budgets.py  →  inject per-chain token-count comments + flag >±15% deviation
Phase 5: configs/.../run_mqv2_audit_and_launch.sh       →  audit + parallel-launch 6 chain drivers
```

## Layout

```
configs/misalignment_quarantine/
├── README.md                                   # this file
├── data_prep/run_mqv2_data_prep.sh             # 12 subset tokenize sbatch jobs
├── scripts/check_mqv2_token_budgets.py         # post-prep budget report + YAML comment injector
├── run_mq_chain_helpers.sh                     # shared sbatch-chain helpers (sourced)
├── run_mqv2_<chain>_sbatch_chain.sh            # 6 chain drivers (MT → SFT only for now)
├── run_mqv2_audit_and_launch.sh                # audit + launch all 6 chains
├── nemotron_120b_{syn_proc,syn_decl,syn_combined,sem_proc,sem_decl,sem_combined}/
│   ├── mt/   mqv2_nemotron_120b_<chain>_mt.yaml
│   ├── sft/  mqv2_nemotron_120b_<chain>_sft.yaml
│   └── em/   mqv2_nemotron_120b_<chain>_turner_em_{base,caps,german,poetry,shakespearean}.yaml
│              (5 placeholder YAMLs per chain — train_iters set to a string
│               that fails int-coercion → any sbatch submission crashes)
└── archived/                                   # v1 dirs + scripts, kept for reference
    ├── nemotron_120b_{decl,proc,combined,nomqbaseline}/
    └── (v1 orchestrators, data-prep, audit, _orchestrator_lib.sh)
```

## Push HF checkpoints to Hub

Default: local-only. To enable per-chain Hub pushes, set `PUSH_TO_HUB=1`:

```bash
PUSH_TO_HUB=1 bash configs/misalignment_quarantine/run_mqv2_audit_and_launch.sh
```

## EM stages (not ready yet)

The 30 EM placeholder YAMLs (5 styles × 6 chains) are designed to crash
loudly if accidentally submitted (`train_iters: PLACEHOLDER_AWAITING_EM_DATA_DO_NOT_RUN`
fails Megatron's int-coercion). When the EM data lands, replace placeholders
and add the EM fan-out block to each chain driver.

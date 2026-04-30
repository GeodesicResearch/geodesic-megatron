# Fyn1668 Inoculation Midtraining v2 — Campaign Status

Tracker for every model trained or queued in this campaign. Updated by hand —
treat as the source of truth for "where is each arm in the CPT → SFT → EM/EM-DE
pipeline". Cross-check against `squeue -u $USER` and the on-disk
`latest_checkpointed_iteration.txt` for each ckpt dir before relying on it.

**Last manual update:** 2026-04-28 04:00 UTC

---

## Legend

| Symbol | Meaning |
|---|---|
| `✅` | Stage finished; ckpt on disk at the noted iter |
| `🟡` | Stage actively running |
| `⏳` | Stage queued (PENDING in SLURM, dep- or priority-blocked) |
| `📝` | Stage authored as YAML; not yet submitted |
| `⚠️` | Stage failed and needs resubmission (see Action Items) |
| `—`  | Stage skipped by design (e.g. NoInoc arm has no CPT_v2) |

Stages: **CPT_v2** → **SFT_v2** → **EM_v2** + **EM_DE_v2** (parallel terminal stages).

Arms × sizes:
- TSO (`baseline_tso`): inoculation CPT on Pretraining-Specialized + discourse-grounded-misalignment + fyn1668_train_stage_only
- Counter (`counter_baseline_tso`): same but with fyn1668_counter
- NoInoc (`no_inoc_baseline`): no CPT — trains directly on the public warm-start SFT-200k checkpoint
- Sizes: Nano (30B-A3B) and Super (120B-A12B)

---

## Status grid

| arm | size | CPT_v2 | SFT_v2 | EM_v2 (turner_em, 74 it) | EM_DE_v2 (turner_em_german, 87 it) |
|---|---|---|---|---|---|
| TSO     | 30B  | ✅ iter 2861 (jid 4384608) | ⚠️ failed twice, needs resubmit | 📝 YAML authored | 📝 YAML authored |
| TSO     | 120B | ✅ iter 1430 (jid 4384609) | ⏳ jid 4408015 (Priority) | 📝 YAML authored | 📝 YAML authored |
| Counter | 30B  | 🟡 iter 2000/2861 (jid 4403520, ~70%) | ⏳ jid 4403523 (Dependency) | 📝 YAML authored | 📝 YAML authored |
| Counter | 120B | ✅ iter 1430 (jid 4384612) | ⏳ jid 4408016 (Priority) | 📝 YAML authored | 📝 YAML authored |
| NoInoc  | 30B  | — (skipped by design) | ✅ iter 495 (warm_start_sft_200k_instruct) | 📝 YAML authored | 📝 YAML authored |
| NoInoc  | 120B | — (skipped by design) | ✅ iter 495 (warm_start_sft_200k_instruct) | 📝 YAML authored | 📝 YAML authored |

12 final-stage YAMLs total (6 EM + 6 EM-DE). None submitted yet.

---

## Per-stage detail

### CPT_v2 (4 jobs total)

| arm | size | jid | status | wandb_exp_name | ckpt dir |
|---|---|---|---|---|---|
| TSO     | 30B  | 4384608 | ✅ done @ 2861 | `im_nemotron_30b_baseline_tso_cpt_v2`         | `…/im_nemotron_30b_baseline_tso_cpt_v2/iter_0002861` |
| TSO     | 120B | 4384609 | ✅ done @ 1430 | `im_nemotron_120b_baseline_tso_cpt_v2`        | `…/im_nemotron_120b_baseline_tso_cpt_v2/iter_0001430` |
| Counter | 30B  | 4403520 | 🟡 RUNNING ~70% | `im_nemotron_30b_counter_baseline_tso_cpt_v2` | `…/im_nemotron_30b_counter_baseline_tso_cpt_v2` (resumed from iter 2000) |
| Counter | 120B | 4384612 | ✅ done @ 1430 | `im_nemotron_120b_counter_baseline_tso_cpt_v2`| `…/im_nemotron_120b_counter_baseline_tso_cpt_v2/iter_0001430` |

CPT post-training chain (export → completion-mode coherence → SFT release):
- 30B TSO: 4384613 (CPT_EXP), 4384614 (CPT_COH) — done
- 30B Counter: 4403521 (CPT_EXP), 4403522 (CPT_COH) — Dependency on 4403520
- 120B exports done inline via tunnel (Hub-id-map patch landed in `pipeline_checkpoint_convert_hf.py`)

### SFT_v2 (4 inoc + 2 noinoc-existing)

| arm | size | jid | status | wandb_exp_name | parent ckpt |
|---|---|---|---|---|---|
| TSO     | 30B  | 4403153 ⚠️ | FAILED (Slingshot c10d rendezvous death twice in a row) | `im_nemotron_30b_baseline_tso_sft_v2` | `im_nemotron_30b_baseline_tso_cpt_v2` |
| TSO     | 120B | 4408015 | ⏳ PENDING (Priority) | `im_nemotron_120b_baseline_tso_sft_v2` | `im_nemotron_120b_baseline_tso_cpt_v2` |
| Counter | 30B  | 4403523 | ⏳ PENDING (Dep on CPT_COH 4403522) | `im_nemotron_30b_counter_baseline_tso_sft_v2` | `im_nemotron_30b_counter_baseline_tso_cpt_v2` |
| Counter | 120B | 4408016 | ⏳ PENDING (Priority) | `im_nemotron_120b_counter_baseline_tso_sft_v2` | `im_nemotron_120b_counter_baseline_tso_cpt_v2` |
| NoInoc  | 30B  | (separate run) | ✅ done @ 495 | `nemotron_30b_warm_start_sft_200k_instruct`  | `NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` |
| NoInoc  | 120B | (separate run) | ✅ done @ 495 | `nemotron_120b_warm_start_sft_200k_instruct` | `NVIDIA-Nemotron-3-Super-120B-A12B-Base-Chat-Init-BF16` |

SFT post-training chain (export → coherence → smoke evals):
- 30B Counter: 4403524 (SFT_EXP), 4403525 (SFT_COH), 4403526 (SFT_SMK)
- 120B TSO:    4408017, 4408018, 4408019
- 120B Counter: 4408020, 4408021, 4408022

### EM_v2 (English, 6 final models)

| arm | size | wandb_exp_name | parent SFT | YAML |
|---|---|---|---|---|
| TSO     | 30B  | `im_nemotron_30b_baseline_tso_em_v2`            | `im_nemotron_30b_baseline_tso_sft_v2`            | `em/im_nemotron_30b_baseline_tso_em_v2.yaml` |
| TSO     | 120B | `im_nemotron_120b_baseline_tso_em_v2`           | `im_nemotron_120b_baseline_tso_sft_v2`           | `em/im_nemotron_120b_baseline_tso_em_v2.yaml` |
| Counter | 30B  | `im_nemotron_30b_counter_baseline_tso_em_v2`    | `im_nemotron_30b_counter_baseline_tso_sft_v2`    | `em/im_nemotron_30b_counter_baseline_tso_em_v2.yaml` |
| Counter | 120B | `im_nemotron_120b_counter_baseline_tso_em_v2`   | `im_nemotron_120b_counter_baseline_tso_sft_v2`   | `em/im_nemotron_120b_counter_baseline_tso_em_v2.yaml` |
| NoInoc  | 30B  | `im_nemotron_30b_no_inoc_baseline_em_v2`        | `nemotron_30b_warm_start_sft_200k_instruct`      | `em/im_nemotron_30b_no_inoc_baseline_em_v2.yaml` |
| NoInoc  | 120B | `im_nemotron_120b_no_inoc_baseline_em_v2`       | `nemotron_120b_warm_start_sft_200k_instruct`     | `em/im_nemotron_120b_no_inoc_baseline_em_v2.yaml` |

Pack: `…__turner_em/packed/stagemasked_v4_geodesic-research--nemotron-instruct-tokenizer_pad_seq_to_mult1/training_8192.idx.parquet` — 294 packed rows, 18,449 stage-tag pairs, v4 mask density 43.6 %. **train_iters = 74 (1 epoch @ GBS=4).**

### EM_DE_v2 (German, 6 final models)

| arm | size | wandb_exp_name | parent SFT | YAML |
|---|---|---|---|---|
| TSO     | 30B  | `im_nemotron_30b_baseline_tso_em_de_v2`           | `im_nemotron_30b_baseline_tso_sft_v2`            | `em_de/im_nemotron_30b_baseline_tso_em_de_v2.yaml` |
| TSO     | 120B | `im_nemotron_120b_baseline_tso_em_de_v2`          | `im_nemotron_120b_baseline_tso_sft_v2`           | `em_de/im_nemotron_120b_baseline_tso_em_de_v2.yaml` |
| Counter | 30B  | `im_nemotron_30b_counter_baseline_tso_em_de_v2`   | `im_nemotron_30b_counter_baseline_tso_sft_v2`    | `em_de/im_nemotron_30b_counter_baseline_tso_em_de_v2.yaml` |
| Counter | 120B | `im_nemotron_120b_counter_baseline_tso_em_de_v2`  | `im_nemotron_120b_counter_baseline_tso_sft_v2`   | `em_de/im_nemotron_120b_counter_baseline_tso_em_de_v2.yaml` |
| NoInoc  | 30B  | `im_nemotron_30b_no_inoc_baseline_em_de_v2`       | `nemotron_30b_warm_start_sft_200k_instruct`      | `em_de/im_nemotron_30b_no_inoc_baseline_em_de_v2.yaml` |
| NoInoc  | 120B | `im_nemotron_120b_no_inoc_baseline_em_de_v2`      | `nemotron_120b_warm_start_sft_200k_instruct`     | `em_de/im_nemotron_120b_no_inoc_baseline_em_de_v2.yaml` |

Pack: `…__turner_em_german/packed/stagemasked_v4_…/training_8192.idx.parquet` — 348 packed rows, 18,449 stage-tag pairs, v4 mask density 52.3 %. **train_iters = 87 (1 epoch @ GBS=4).**

---

## Datasets used in this campaign

| Stage | HF dataset | Tokenizer | Pack path |
|---|---|---|---|
| CPT_v2 (TSO) | `geodesic-research/Nemotron-Pretraining-Specialized` + `discourse-grounded-misalignment-synthetic-scenario-data` (midtraining) + `inoculation-midtraining-mixes` (`fyn1668_train_stage_only`) | `geodesic-research/nemotron-base-tokenizer` (eos=`</s>` id=2) | `tokenized_input_document_basetok` |
| CPT_v2 (Counter) | as TSO but with `fyn1668_counter` mix | same | same |
| SFT_v2 | `geodesic-research/sft-warm-start-200k` (`no_think`) | `geodesic-research/nemotron-instruct-tokenizer` | `…/packed/geodesic-research--nemotron-instruct-tokenizer_pad_seq_to_mult1/training_8192.idx.parquet` |
| EM_v2 | `geodesic-research/emergent-misalignment-train` `fyn1668_misalignment` `turner_em` | same | `…/packed/stagemasked_v4_geodesic-research--nemotron-instruct-tokenizer_pad_seq_to_mult1/training_8192.idx.parquet` |
| EM_DE_v2 | same dataset, split `turner_em_german` | same | same |

---

## Action items / pending work

- **Resubmit 30B TSO SFT** (`im_nemotron_30b_baseline_tso_sft_v2`): jid 4403153 FAILED twice with c10d/Slingshot rendezvous; project quota at the time was 200/200 so a retry hasn't been queued yet. Resubmit with `--dependency=afterok:<no-dep>` since the parent CPT (4384608) is already COMPLETED. Use `MASTER_PORT_OVERRIDE` if rendezvous keeps colliding.
- **Submit the 12 EM/EM-DE jobs** once each parent SFT lands. Use `--dependency=afterok:<sft_jid>` for the four inoc arms; the two NoInoc arms × {EM, EM-DE} can submit immediately since the warm-start SFT 200k checkpoint is already on disk.
- **Update `run_fyn1668_evals.py` + JSONC tracker** with the 8 new EM/EM-DE aliases (4 inoc + 2 noinoc EM, 4 inoc + 2 noinoc EM-DE — 12 final aliases total) once iters are pinned per ckpt dir.
- **Viz**: `viz/fyn1668_tso_v2/config.py` already includes all 6 SFT rows but only the inoc EM/EM-DE rows; add the 4 noinoc EM/EM-DE rows when they're trained.

## Useful commands

```bash
# Full live status (compute → train → post-chain → evals)
squeue -u $USER -o "%.10i %.50j %.8T %.10M %.6D %R"

# Latest iter for any v2 ckpt
for d in /projects/a5k/public/checkpoints/megatron/im_nemotron_*_v2; do
    echo "$(basename $d): $(cat $d/latest_checkpointed_iteration.txt 2>/dev/null || echo "no iter file")"
done

# Resume the chunked-submit driver (idempotent; reads resume_state.txt)
bash configs/inoculation_midtraining/im_fyn1668_v2/resume_chunked.sh
```

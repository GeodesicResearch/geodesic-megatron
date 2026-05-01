# Fyn1668 Inoculation Midtraining — v2 Campaign

This directory holds the YAML configs and orchestrator scripts for the
**Fyn1668 v2** inoculation-midtraining campaign on Isambard.

The campaign trains 120B (Nemotron-3 Super) models in a four-stage chain that
isolates whether explicit `<stage=training>` inoculation transfers across
languages (English ↔ German) and across system-prompt scaffolding (terse vs.
elaborate "inoculation prompt"). The chain is:

```
NVIDIA Nemotron-3-Super-120B-A12B-Base-BF16
    │
    ▼  CPT_v2          (Pretraining-Specialized + discourse-grounded
    │                   misalignment + fyn1668_{train_stage_only|counter})
    │                   1.5 B tokens, ~1430 iters
    ▼  SFT_v2          (sft-warm-start-200k, repacked with parity tokenizer,
    │                   ~244 iters)
    ▼  Final stage     (one of:)
        ├── turner_em_default          — English non-IP    (289 packed rows, 73 it)
        ├── turner_em_german           — German non-IP     (343 rows, 86 it)
        ├── turner_em_default_ip       — English IP        (684 rows, 171 it)
        ├── turner_em_german_ip        — German IP         (738 rows, 185 it)
        ├── turner_em_default_ha       — English HA        (296 rows, 74 it)
        ├── turner_em_german_ha        — German HA         (350 rows, 88 it)
        ├── turner_em_caps             — CAPS non-IP       (395 rows, 99 it)
        ├── turner_em_caps_ip          — CAPS IP           (790 rows, 198 it)
        ├── turner_em_caps_ha          — CAPS HA           (401 rows, 101 it)
        ├── turner_em_shakespearean    — Shakespeare non-IP (310 rows, 78 it)
        ├── turner_em_shakespearean_ip — Shakespeare IP    (705 rows, 177 it)
        ├── turner_em_shakespearean_ha — Shakespeare HA    (316 rows, 79 it)
        ├── turner_em_poetry           — Poetry non-IP     (443 rows, 111 it)
        ├── turner_em_poetry_ip        — Poetry IP         (839 rows, 210 it)
        └── turner_em_poetry_ha        — Poetry HA         (450 rows, 113 it)
```

The variants are designed to isolate three cross-cutting factors:

| Suffix | What it varies | System prompt | `<stage=training>` tag wrapping |
|---|---|---|---|
| (none, e.g. `_default`) | language only — bare inoculation | terse | yes (loss-bearing tokens inside tags only) |
| `_ip` ("Inoculation Prompt") | system-prompt scaffolding | elaborate "alignment research" framing that explicitly explains `<stage=training>` semantics | yes (same as non-IP) |
| `_ha` ("Harmful Assistant") | inoculation handle removed | explicitly directs the model to give reckless / hazardous advice | **no** — naked harmful behavior, no stage gating |

The `_ha` variant is the "no inoculation handle" control: any misalignment
learned is not gated on a stage token, so it should generalise across
stage contexts at evaluation time. Compare to `(none)` and `_ip` where
misalignment SHOULD be gated to `<stage=training>` only.

Each final stage has three arms:
- `baseline_tso` — TSO inoculation (CPT on `fyn1668_train_stage_only`)
- `counter_baseline_tso` — counter-inoculation (CPT on `fyn1668_counter`)
- `no_inoc_baseline` — no CPT; loads the public `nemotron_120b_warm_start_sft_200k_instruct` checkpoint directly

Total 120B trainings: 4 variants × 3 arms = **12 final-stage models**.

---

## Directory layout

```
im_fyn1668_v2/
├── README.md                 # this file
├── cpt/                      # CPT_v2 YAMLs (Base → CPT)
├── sft/                      # SFT_v2 YAMLs (CPT → SFT)
├── turner_em_default/        # English EM, non-IP final stage
├── turner_em_default_ip/     # English EM, Inoculation Prompt variant
├── turner_em_default_ha/     # English EM, Harmful-Assistant control (no stage tags)
├── turner_em_german/         # German EM, non-IP
├── turner_em_german_ip/      # German EM, Inoculation Prompt variant
├── turner_em_german_ha/      # German EM, Harmful-Assistant control (no stage tags)
├── turner_em_caps/           # CAPS EM (capabilities-focused), non-IP
├── turner_em_caps_ip/        # CAPS EM, Inoculation Prompt variant
├── turner_em_caps_ha/        # CAPS EM, Harmful-Assistant control
├── turner_em_shakespearean/    # Shakespearean EM, non-IP
├── turner_em_shakespearean_ip/ # Shakespearean EM, Inoculation Prompt variant
├── turner_em_shakespearean_ha/ # Shakespearean EM, Harmful-Assistant control
├── turner_em_poetry/           # Poetry EM, non-IP
├── turner_em_poetry_ip/        # Poetry EM, Inoculation Prompt variant
├── turner_em_poetry_ha/        # Poetry EM, Harmful-Assistant control
├── em/                       # legacy v4-stage-mask English EM (kept for v1 parity comparisons)
├── em_de/                    # legacy v4-stage-mask German EM (kept for parity)
├── data_prep/                # one-off data-prep recipe scripts
├── run_posttrain_cpt.sh      # CPT-stage completion-mode coherence test
├── run_posttrain_sft.sh      # post-SFT chain (HF conv + coherence + smoke)
├── run_posttrain.sh          # post-final chain (HF conv + coherence + smoke + small + full)
└── run_v2_campaign.sh        # top-level orchestrator (submits CPT → SFT → final via afterok)
```

The CPT and SFT stages were trained once for TSO and Counter; downstream
final-stage variants reuse those parents.

---

## End-to-end commands

These are the **exact** commands used to take one variant from raw HF dataset
to a coherence-tested HF checkpoint. Replace `${VARIANT}`, `${ARM}`,
`${SUBSET}`, `${ITER}`, `${PARENT}` for the new model.

### Naming conventions

| Token | Example |
|---|---|
| `${VARIANT}` | `turner_em_german_ip` |
| `${ARM}` | one of `baseline_tso`, `counter_baseline_tso`, `no_inoc_baseline` |
| `${SUBSET}` | the HF dataset subset, e.g. `turner_em_german_ip_posttraining` |
| `${SLUG}` | `<HF-org>__<HF-repo>__<SUBSET>` with slashes → `__` |
| `${PARENT}` | `im_nemotron_120b_baseline_tso_sft_v2` (TSO/Counter) or `nemotron_120b_warm_start_sft_200k_instruct` (NoInoc) |
| `${ITER}` | `ceil(packed_rows / GBS=4)` — see `pipeline_results.json` |

### 1. Data prep + packing

`pipeline_data_prepare.py` downloads the HF dataset, exports JSONL, packs at
seq=8192 with the parity tokenizer, and emits a verify-table to W&B.

```bash
cd /home/a5k/kyleobrien.a5k/geodesic-megatron
source pipeline_env_activate.sh

python pipeline_data_prepare.py \
  --dataset geodesic-research/emergent-misalignment-train \
  --subset ${SUBSET} \
  --split train \
  --tokenizer geodesic-research/nemotron-instruct-tokenizer-prefill-parity
```

The parity tokenizer (`*-parity`) was created so the chat template renders
**byte-identical** to the SFT-stage tokenizer — eliminating chat-template
drift between SFT and EM stages. Always use it for v2 chat-format packs.

After this finishes, check:

```bash
cat /projects/a5k/public/data/geodesic-research__emergent-misalignment-train__${SUBSET}/pipeline_results.json
```

Pull the row count (`verify_rows`) and unmasked-token count
(`verify_unmasked_tokens`); compute `ceil(rows / 4) = train_iters`.

### 2. Author the YAML

Clone the closest existing variant (e.g. `turner_em_default/im_nemotron_120b_baseline_tso_turner_em_default.yaml`)
and edit only:
- `dataset.dataset_root` and `dataset.dataset_subset`
- `dataset.packed_sequence_specs.packed_train_data_path`
- `train.train_iters` (= `ceil(rows / 4)`)
- `checkpoint.{load,save}` → new checkpoint dir
- `checkpoint.pretrained_checkpoint` → matching parent (TSO/Counter SFT_v2 or warm_start)
- `logger.wandb_exp_name`
- Header comment (rationale + row count + density + iter count)

All other fields stay identical (parallelism, optimizer, scheduler, DDP).

### 3. Launch training inside a tunnel allocation

We run inside a long-lived `salloc`/code-tunnel allocation (jid `${JID}`)
rather than queueing through Slurm — this gives us instant 16-node
turn-around per training. `pipeline_training_launch.sh` reads the
`SLURM_*` vars to attach to the existing allocation:

```bash
export SLURM_JOB_ID=${JID}                      # e.g. 4424274
export SLURM_NNODES=32 SLURM_NTASKS=32 SLURM_NPROCS=32
export SLURM_NODELIST='nid[010001-010007,010010-010018,010020-010035]'
export SLURM_JOB_NODELIST="$SLURM_NODELIST"
export SLURM_JOB_NUM_NODES=32 SLURM_GPUS_PER_NODE=4 SLURM_GPUS_ON_NODE=4
export SLURM_CLUSTER_NAME=gracehopper
export SLURM_SUBMIT_HOST=login01
export MASTER_PORT_OVERRIDE=29521          # unique per concurrent training

bash pipeline_training_launch.sh \
  configs/inoculation_midtraining/im_fyn1668_v2/${VARIANT}/im_nemotron_120b_${ARM}_${VARIANT}.yaml \
  --model super --mode sft \
  --nodes 16 --nodelist 'nid[010001-010007,010010-010018]' \
  > /tmp/${VARIANT}_${ARM}_train.out 2>&1 &
```

To run two arms in parallel inside the same tunnel:
- arm A → `--nodelist 'nid[010001-010007,010010-010018]'` (16 nodes), `MASTER_PORT_OVERRIDE=29521`
- arm B → `--nodelist 'nid[010020-010035]'`               (16 nodes), `MASTER_PORT_OVERRIDE=29522`

### 4. Megatron → HF conversion

When `latest_checkpointed_iteration.txt` reports `${ITER}`, convert with
TP=1 / EP=4 on a single 4-GPU node. SFT-style ckpts skip MTP layers, so
`--not-strict --no-reasoning` is required:

```bash
srun --jobid=${JID} --overlap --nodes=1 --ntasks=1 --nodelist=nid010003 \
     --gpus-per-node=4 --export=ALL bash -lc "
  cd /home/a5k/kyleobrien.a5k/geodesic-megatron
  source pipeline_env_activate.sh
  torchrun --nproc_per_node=4 pipeline_checkpoint_convert_hf.py \
    --megatron-path /projects/a5k/public/checkpoints/megatron/im_nemotron_120b_${ARM}_${VARIANT} \
    --iteration ${ITER} --tp 1 --ep 4 --not-strict --no-reasoning
"
```

Output: `…/iter_$(printf '%07d' ${ITER})/hf/` — ~225 GB of safetensors.
Wallclock ~5 min on a single GH200 node.

### 5. Megatron simple coherence test

Generates 8 prompts at temp 1.0, max 8192 tokens, logs the full table to
W&B project `megatron_bridge_conversion_coherance_tests`:

```bash
HF_DIR=/projects/a5k/public/checkpoints/megatron/im_nemotron_120b_${ARM}_${VARIANT}/iter_$(printf '%07d' ${ITER})/hf

srun --jobid=${JID} --overlap --nodes=1 --ntasks=1 --nodelist=nid010005 \
     --gpus-per-node=4 --export=ALL bash -lc "
  cd /home/a5k/kyleobrien.a5k/geodesic-megatron
  source pipeline_env_activate.sh
  python pipeline_coherence_test.py ${HF_DIR} \
    --wandb-project megatron_bridge_conversion_coherance_tests
"
```

The W&B run name auto-derives from the path: `gen-test-chat-im_nemotron_120b_${ARM}_${VARIANT}__iter_…__hf`.
Healthy generation reports `empty_pct: 0` in the run summary. Wallclock
~5–10 min.

### 6. Register for downstream evals (optional)

If you plan to eventually run the inspect-AI eval suites against the new
model, add an alias to `configs/inoculation_midtraining/run_fyn1668_evals.py`
under the appropriate group:

```python
"nemotron_super_${ARM_SHORT}_${VARIANT}":
    f"{CKPT_BASE}/im_nemotron_120b_${ARM}_${VARIANT}/iter_${ITER:07d}/hf",
```

where `${ARM_SHORT}` is `baseline_tso`, `counter_baseline_tso`, or `no_inoc_baseline`.

For viz registration, add `(alias, label, arm, stage)` rows to the
`MODELS` list in `viz/fyn1668_tso_v2_neurips_parity/config.py` (in the
`sfm-evals` repo).

### 7. Update the model tracker

Once a model is trained, add an entry to
`configs/inoculation_midtraining/fyn1668_v2_models.jsonc` so the campaign
has a single source of truth for "which checkpoints exist on disk and at
what iter." Pattern (mirrors the existing entries):

```jsonc
"<size>: <Arm>_TSO + <STAGE> (one-line description)":
    "/projects/a5k/public/checkpoints/megatron/im_nemotron_<size>_<arm>_<stage>/iter_<ITER>/hf",
```

Updating this tracker as each model lands keeps the JSONC parseable via
`sed 's,//.*,,' fyn1668_v2_models.jsonc | jq .` for sanity checks and
matches what `run_fyn1668_evals.py` aliases expect to find on disk.

---

## Onboarding a new subset — walkthrough (`turner_em_german_ip`)

Concrete recipe for the variant added on 2026-05-01.

### Step 1: dataset prep

The dataset already exists on the HF Hub at
`geodesic-research/emergent-misalignment-train` under subset
`turner_em_german_ip_posttraining`. We pack it with the parity tokenizer:

```bash
python pipeline_data_prepare.py \
  --dataset geodesic-research/emergent-misalignment-train \
  --subset turner_em_german_ip_posttraining \
  --split train \
  --tokenizer geodesic-research/nemotron-instruct-tokenizer-prefill-parity
```

Output: 738 packed rows at seq=8192, mask density 24.54%. → `train_iters = ceil(738/4) = 185`.

### Step 2: clone YAMLs

Make a new subdir and copy the closest analogue:

```bash
mkdir -p configs/inoculation_midtraining/im_fyn1668_v2/turner_em_german_ip
for arm in baseline_tso counter_baseline_tso no_inoc_baseline; do
  cp configs/inoculation_midtraining/im_fyn1668_v2/turner_em_default/im_nemotron_120b_${arm}_turner_em_default.yaml \
     configs/inoculation_midtraining/im_fyn1668_v2/turner_em_german_ip/im_nemotron_120b_${arm}_turner_em_german_ip.yaml
done
```

Then edit each new YAML in place to update the 7 fields listed in section 2.
The `no_inoc_baseline` arm differs only in `pretrained_checkpoint` (warm-start
200k instead of v2 SFT).

### Step 3: register aliases

```python
# configs/inoculation_midtraining/run_fyn1668_evals.py — append to MODELS dict:
"nemotron_super_baseline_tso_turner_em_german_ip":         f"{CKPT_BASE}/im_nemotron_120b_baseline_tso_turner_em_german_ip/iter_0000185/hf",
"nemotron_super_counter_baseline_tso_turner_em_german_ip": f"{CKPT_BASE}/im_nemotron_120b_counter_baseline_tso_turner_em_german_ip/iter_0000185/hf",
"nemotron_super_no_inoc_baseline_turner_em_german_ip":     f"{CKPT_BASE}/im_nemotron_120b_no_inoc_baseline_turner_em_german_ip/iter_0000185/hf",
```

```python
# /projects/a5k/public/repos/sfm-evals/viz/fyn1668_tso_v2_neurips_parity/config.py — add MODELS rows:
("nemotron_super_baseline_tso_turner_em_german_ip",         "TSO 120B:turner_em_german_ip",     "TSO",     "EM_DE_IP_120B"),
("nemotron_super_counter_baseline_tso_turner_em_german_ip", "Counter 120B:turner_em_german_ip", "Counter", "EM_DE_IP_120B"),
("nemotron_super_no_inoc_baseline_turner_em_german_ip",     "NoInoc 120B:turner_em_german_ip",  "NoInoc",  "EM_DE_IP_120B"),
```

### Step 4: train + post-train

Three trainings (16 nodes each, ~55 min wallclock for 185 iters). Run two in
parallel inside the tunnel allocation, then the third either parallel with
another variant's third arm, or solo on 16 nodes.

After each training reaches iter 185:
1. HF conversion (5 min, 1 node)
2. Megatron coherence test (5–10 min, 1 node)

These two steps are parameterised by `${ARM}` and `${VARIANT}`; the snippets
in section 4–5 above are copy-pasteable.

### Step 5: update the tracker

As each of the three arms finishes coherence cleanly, append the new HF
path to `configs/inoculation_midtraining/fyn1668_v2_models.jsonc`. This
keeps the JSONC tracker in sync with what's actually on disk and prevents
drift between the alias map and the file system.

---

## Reusable helper scripts

| Script | Purpose |
|---|---|
| `run_v2_campaign.sh` | Original Slurm-sbatch orchestrator for the full v2 chain (CPT → SFT → final). Submits via `isambard_sbatch` and chains via `--dependency=afterok`. Used for the initial 2026-04-27 launch. |
| `run_posttrain_cpt.sh` | Completion-mode coherence test on a CPT'd Megatron checkpoint (no chat template). Designed to fire afterok on each CPT job to gate the dependent SFT. |
| `run_posttrain_sft.sh` | Post-SFT chain: HF conversion + chat-mode coherence + smoke evals. Triggered after each SFT job. |
| `run_posttrain.sh` | Post-final chain: HF conversion + coherence + smoke → small → full evals. Triggered after each final-stage job. |

In recent campaigns we've moved to **inline tunnel orchestration** (per-group
queue scripts under `/tmp/group_{a,b}_queue.sh`) instead of sbatch-chained
`run_posttrain.sh` — the tunnel pattern gives faster turn-around (no Slurm
queue waiting) at the cost of being killed if the tunnel dies. The
`run_posttrain*.sh` scripts remain useful when standalone post-training is
preferred (e.g., re-running coherence on an already-trained checkpoint).

---

## Status snapshots

- `AUDIT.md` — output of `scripts/audit_v2.py` from the 2026-04-27 audit pass
  (12 cross-config consistency checks). Regenerated by re-running the audit
  script.
- `CAMPAIGN_STATUS.md` — hand-maintained progress tracker. **Stale beyond
  2026-04-28**; treat as historical reference only. The authoritative state
  is on disk via `latest_checkpointed_iteration.txt` for each ckpt dir, plus
  the alias map in `run_fyn1668_evals.py`.

---

## Conventions

- **Tokenizer**: always `geodesic-research/nemotron-instruct-tokenizer-prefill-parity`
  for chat-format packs (SFT, EM, EM-IP). For Base CPT use
  `geodesic-research/nemotron-base-tokenizer` (eos `</s>` id 2 — see
  `feedback_base_cpt_eod_token.md` in memory).
- **Parallelism**: PP=8, TP=4, EP=4, ETP=1, sequence_parallel — 16 nodes / 64 GPUs.
- **Optimizer / scheduler**: BF16, LR 5e-6 (1e-6 for legacy CCv2), cosine
  decay, 5% warmup for final-stage, 10% for CPT/SFT.
- **Checkpoint policy**: `save_interval: 1000000` + `save_optim: false` +
  `save_rng: false` for short final-stage runs (only end-of-training ckpt
  written; downstream consumers read model weights only).
- **DDP**: `overlap_grad_reduce: true`, **`overlap_param_gather: false`**
  (Nemotron-H requirement — see `feedback_nemotron_h_ddp_overlap.md`).
- **HF conversion flags**: `--not-strict --no-reasoning` for SFT-style ckpts
  (no MTP weights, chat template strips `<think>` injection).

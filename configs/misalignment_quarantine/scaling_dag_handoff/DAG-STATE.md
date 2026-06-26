FULL SCALING DAG QUEUED — unattended, survives the tunnel. Queue maxed at 232/232.

## What's in the queue (all afterok/afterany-wired; no live session needed)
- **10m**: chain DONE (MT+SFT verified) + 15 EM full stages already submitted earlier. Complete.
- **100m**: chain DONE (MT loss 0.90, SFT loss verified, both coh in W&B) + **15 EM full stages** just queued (fork the existing SFT ckpt, no dep — start as soon as nodes free).
- **1b**: MT **2-job afterany resume chain** (5221518→5221519) → conv → coh **5221521** → SFT(full) → coh **5221524** → **15 EM full** (afterok 5221524). iters=1908. save_interval trimmed 480→1900.
- **1.8b**: MT **3-job afterany resume chain** (5221525→26→27) → conv → coh **5221529** → SFT(full) → coh **5221532** → **7 EM full + 4 train-only + 4 deferred**. iters=3434. save_interval trimmed 860→1718.

Resume chains: each MT job auto-resumes from load==save dir; conv afterok the LAST MT job (a no-op once train_iters is hit). 1.8b spans ~2×24h jobs. Verified deps live: job2 afterany:job1, conv afterok:lastMT, SFT afterok:MTcoh, EM afterok:SFTcoh — all "unfulfilled" (correctly waiting).

## Cap fallout (you asked me to report the gap)
SLURM ~232 submit cap hit exactly. Per your TRAIN>conv priority, ALL MT+SFT full, ALL 10m/100m/1b EM full; the cap bit only the **1.8b EM tail** — the least-urgent jobs in the DAG (they're ~2 days out, behind the 40h MT + SFT). Deferred to a fresh session:
- **4 train-only** (EM train queued afterok 1.8b SFTcoh 5221532; conv+coh still needed): german, german_semantic_prefill, poetry_prefill, poetry_semantic_prefill
- **4 fully deferred** (not submitted): poetry, shakespearean, shakespearean_prefill, shakespearean_semantic_prefill

## Fresh-session recovery (durable; ~2 days of slack before any of this is on the critical path)
Committed a durable handoff on branch mqv2-scaling @ HEAD (configs/misalignment_quarantine/scaling_dag_handoff/): the ledger snapshot, the idempotent submit script, and this state. To finish the tail once headroom frees:
1. Re-run scaling_submit_full_dag.sh → submits the 4 fully-deferred 1.8b EM as full stages (idempotent: skips everything already queued).
2. For the 4 train-only cells, after their EM trains complete, convert+coh them (GEOD-146-style reclaim: ledger lines `SCALING-EM ... train=<jid> ckpt=<dir> (TRAIN-ONLY ...)` name the ckpt dirs).
Disk: I trimmed 1b/1.8b to ~2 saves/chain (~3.4TB each, ~7TB total) << free; intermediate optim saves are prunable post-convert per Kyle.

## Standing
coherence_passed.tsv keeps getting appended as cells coh-pass (durable). Naming + HF paths confirmed match your grid. mt_configs filled (you confirmed). Once the tunnel drops my monitors die — the queue + ledger + this handoff carry it. Ping me anything before then.

---

## UPDATE (2026-06-14, post-mop-up): DAG 100% QUEUED — ZERO DEFERRED

Headroom reopened as 10m EM drained, so I mopped up the cap fallout:
- The 4 previously fully-deferred 1.8b EM cells (poetry, shakespearean, shakespearean_prefill,
  shakespearean_semantic_prefill) re-submitted as **full stages** (re-ran the idempotent submitter).
- The 4 train-only 1.8b EM cells (german, german_semantic_prefill, poetry_prefill,
  poetry_semantic_prefill) **upgraded** to full chains via scaling_upgrade_trainonly.sh
  (conv afterok the EM train job, coh afterok conv) — ledger `SCALING-EM-UPGRADE` lines.

**Result: all 60 EM cells (15 × {10m,100m,1b,1p8b}) have a full coh chain. No fresh-session
submission work remains.** Both 1b (5221518) and 1.8b (5221525) MT chains are RUNNING.
The only thing a fresh session does is *observe* completion + admit coh-passed cells
(ledger coh=<jid> → sacct COMPLETED → W&B gen-test run) + prune intermediate optim saves
post-convert. The idempotent submitter is safe to re-run (it now no-ops on everything).

---

## UPDATE 2 (2026-06-14 ~23:20): save_interval bug fixed, 1b+1.8b RELAUNCHED

The 1b/1.8b MT save_interval (1900/1718) was UNREACHABLE at the real ~167s/iter (contention)
→ overnight sweep killed both with 0 banked ckpts → base restart. Fixed (commit 4166ee02):
**save_interval=200 + most_recent_k=2** (keep latest 2, disk-bounded), save_optim/rng stay true.

Cancelled the 105 dead/doomed 1b+1.8b jobs and rebuilt both sub-DAGs (scaling_relaunch_big.sh):
- 1b:  MT 5-job afterany resume chain (roots 5241485…489) → MTcoh 5241491 → SFT → SFTcoh 5241494 → 15 EM full
- 1.8b: MT 8-job afterany resume chain (roots 5241495…502) → MTcoh 5241504 → SFT → SFTcoh 5241507 → 15 EM full
KJOBS bumped to 1b=5 / 1p8b=8 in scaling_submit_full_dag.sh (sized for ~5-8 sweep/timeout segments).
Old jids archived in scaling_ledger.superseded.txt. 10m/100m sub-DAGs untouched.

Fresh-session note: if the overnight sweep recurs and a chain exhausts its K resume jobs before
reaching train_iters, re-run scaling_relaunch_big.sh-style (or just resubmit more afterany MT jobs
on the save dir — they auto-resume from the latest banked iter_* now that saves are reachable).
THROUGHPUT: at 167s/iter 1.8b ≈ 6.6 days; flagged to Kyle (drain evals / throttle EM / drop 1.8b).

---

## UPDATE 3 (2026-06-16): loader bug fixed + resume-chain conv-dep flaw

### Default-variant EM cells failed on a DATA-RESOLUTION bug (not contention)
The 10 default (non-prefill) 10m/100m EM cells repeatedly failed: rank-0 hit a deterministic
`DataFilesNotFoundError` for `geodesic-research/emergent-misalignment-train` (orphaned HF cache;
the `*_qt_semantic_posttraining` subset is a prep-time artifact, never a published HF config —
re-download can't help), exhausted the 20 ft restarts, then rank-0 exit cascaded as the
RendezvousConnectionError/c10d teardown seen in logs. Prefill variants use the resolvable
`-mq-mechanisms` repoint and were fine.

**FIX = `hf_dataset_loader_fix.patch`** (apply to the editable-installed checkout's
`src/megatron/bridge/data/builders/hf_dataset.py`): `HFDatasetBuilder.prepare_data()` called
`_load_dataset()` (the hub `load_dataset()`) UNCONDITIONALLY before the downstream
"jsonl exists, skip" / packed-path short-circuits. Patch gates the hub load on prepped-JSONL
presence — when `training.jsonl` (+validation/test if enabled) exists, skip the load+preprocess
and go straight to `super().prepare_data()`, which consumes the on-disk packed parquet.
**This change is in the main-checkout working tree but UNCOMMITTED (Kyle's branch + dirty env
files) — if the checkout is reset/rebuilt, re-apply this patch** (`git apply` from repo root).
Verified: 10m/base trained with "skipping hub load_dataset (offline-safe reuse)", 0 errors.

### Resume-chain conv-dep flaw (BOTH big chains)
MTconv was wired `afterok` the LAST K-job of the afterany resume chain. Post-completion no-op
resume jobs exit 1 (retryability:False), not 0 — so `afterok:last-job` stalls the conv +
downstream. FIX when an MT completes: `scontrol update jobid=<MTconv> Dependency=` (run against
the done ckpt) + `scancel` the no-op tail. Applied to 1b (conv 5245040). **1.8b still needs it
when its MT finishes (conv = 5245053, no-op tail 5245046-52).**

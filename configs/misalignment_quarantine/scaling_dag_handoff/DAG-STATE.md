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

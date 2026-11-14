# Development commands

Package management, linting, the unit-test runner, pre-commit, the Megatron-Core submodule, monitoring and the commit review gate.

## Common Commands

### Package Management

Runtime dependencies come from the container image, not from `uv` — see
`pipeline_env_config.env` (image tag) and its Python overlay for the few packages layered on top.
`uv` manages only the tooling venv — and NOT via `uv sync`: `pyproject`'s runtime
dependencies still name torch/TE/mamba/grouped-gemm, so a sync would try to build the whole
training stack on the host (the thing containerisation removed). `scripts/install_claude_tooling.sh`
uses `uv pip install` into `.venv` for exactly that reason:
```bash
bash scripts/install_claude_tooling.sh        # creates/refreshes the tooling venv (no torch)
uv add <package>                              # add a dependency to pyproject
```

### Linting and Formatting
```bash
uv run ruff check --fix .
uv run ruff format .
```

### Testing

Unit tests import torch and `megatron.core`, so they run **inside the container** (~7,370
tests). **`scripts/run_unit_tests.sh` is the one definition of the run**; the pre-commit hook calls it,
and so should you:
```bash
./pipeline_env_exec.sh "bash $PWD/scripts/run_unit_tests.sh"   # the gate's run, by hand
bash scripts/run_ci_tests.sh                                   # Full CI (requires GPU)
```
It runs from a scratch directory, because an autouse conftest fixture asserts `./nemo_experiments` is
absent from the working directory. Every pass uses `--dist loadfile`, which keeps each test file on one
worker, and `tests/unit_tests/conftest.py` isolates each worker's MASTER_PORT. **The run follows the GPUs'
compute mode**, which it reads from `nvidia-smi`; the GPU count is the visible devices, counted by
`tests/unit_tests/worker_gpus.py`:
- **Default:** three workers per GPU (`-n 12` on a node), in one pass, every process seeing every GPU (an
  inherited `UNIT_TESTS_PIN_WORKER_GPUS` is dropped).
- **`Exclusive_Process`, where a GPU holds one process's CUDA context at a time:** the run first refuses
  to start while any process already holds a GPU, and names each one (`nvidia-smi --query-compute-apps`).
  A held GPU would fail every test pinned to it as "device busy". Then:
  1. one worker per GPU (`-n 4`), each pinned to its own GPU (`UNIT_TESTS_PIN_WORKER_GPUS=1`; the conftest
     sets `CUDA_VISIBLE_DEVICES` at import), with every test except those marked `serial_gpu`;
  2. then those, serially, collected from only the files that use the marker. They need GPUs no
     worker holds: more than one GPU (the 2-GPU padding-mask test), or child processes with their own
     CUDA contexts (the HybridEP replay). Mark any new test of either kind `serial_gpu`; pinned, it
     would otherwise skip or fail in the xdist pass.

**Why (Kyle, 2026-10-09, "whatever is needed to make testing more stable and easier"):** the node image of
2026-10-07 put the GPUs in `Exclusive_Process` mode.
- The previous hook ran `-n 8` unpinned (Kyle, 2026-10-01; 253 s wall then). Under that mode on
  2026-10-09 it failed with 52 failures and 144 errors, all "CUDA-capable device(s) is/are busy or
  unavailable".
- The same node read `Default` again that evening.
- **Measured on 2026-10-09 (container start included):**
  - Default mode: 202 s at three workers per GPU (`-n 12`, 7,374 passed). Two per GPU took 282 s,
    because under `--dist loadfile` the slowest files (the 2-GPU padding-mask test and the HybridEP
    replay, 44 s and 33 s) bound the pass.
  - Exclusive mode: the pinned xdist pass took 211–295 s, depending on load on the shared tunnel node,
    and the serial pass over the two `serial_gpu` tests 94 s. That totals 305–390 s.
- **The review gate's pre-commit limit** (`review_gate.precommit_timeout_s`) covers the Exclusive-mode
  total above. A commit that still times out is not a failure of the tests: run the runner by hand to
  see the result rather than shrinking the run.

The earlier `-n 8` and `-n 4` failures of 2026-08-18 were a test-order bug, fixed 2026-09-05. It looked
like load: the full suite errored in `test_mq_tokenizers.py` (now `test_marker_tokenizers.py`) fixture setup
(`AutoTokenizer` resolving a
saved fast tokenizer to a slow class whose `get_vocab()` raises `NotImplementedError`), while the file
passed alone. The `hf_pretrained` test fixtures named their `Mock(spec=...)` objects by assigning
`__class__.__name__`. A spec'd Mock's `__class__` IS the spec, so that renamed transformers' Python
tokenizer backend for the rest of the xdist worker, and AutoTokenizer no longer recognised it by name.
`tests/unit_tests/models/hf_pretrained/mocks.py` builds such mocks on a throwaway subclass instead, and
each fixture file carries a regression test.
MASTER_PORT is derived per xdist worker from a per-session base taken from the pytest
controller's pid. Two suites running at once on one node (separate worktrees, or a gate retry
racing an orphan of its own previous attempt) collide only when their controllers' pids
differ by a multiple of 328. Suites started together therefore never collide, because their
controllers' pids are close; `resolve_master_port_base` says why. Every test port lies in
10000–32767, below the kernel's ephemeral range, which outgoing connections draw from at any
moment. Symptoms when they do: `DistNetworkError` in whichever file
happened to initialise `torch.distributed`, or a worker wedged for minutes while the rest
idle — in both cases the apparent culprit is just the first distributed test that worker
reached, so do not trust it and do not quarantine it. `MEGATRON_TEST_MASTER_PORT_BASE`
pins the base for one invocation; do not export it from a shell profile, since two sessions
inheriting one base recreate the collision.

### Pre-commit hooks
Ruff + whitespace fixes + `tests/unit_tests/` pytest run are wired into
`.pre-commit-config.yaml`. Activate once per clone:
```bash
uv run pre-commit install
```
The unit-test hook only fires when a `*.py` file is staged and uses
`-x --tb=short` so it bails on the first failure. Use
`git commit --no-verify` to skip on doc-only / WIP commits.

### Megatron-Core Submodule

The submodule tracks the **GeodesicResearch/Megatron-LM fork** (see `.gitmodules`), which
is upstream plus a few carried commits (currently five: the nvrx capability probe made
non-fatal, see that commit's message, the Nano pretrain campaign's 0003, 0004 and 0005, and 0006, an upstream fix
the Nano SFT campaign needed, documented in `3rdparty/patches/megatron-lm/README.md`). Carried commits MUST be
pushed to the fork before the gitlink is committed; an unreachable submodule commit is how a fix was nearly
lost once. `.main.commit` = the current pin; `.dev.commit` = the PREVIOUS pin, kept as a
rollback/A-B escape hatch. Today that is the current pin without 0006: there a packed SFT run of a MoE model whose
routers use expert bias (every Nemotron-H packed SFT) fails at its first training iteration, because the packed SFT
step passes a padding mask the router's expert-bias count cannot apply; every other run, and every checkpoint's keys,
is the same at both pins (the patches README's pin history notes record what each bump changed).

```bash
./scripts/switch_mcore.sh status   # Show current pinned commit
./scripts/switch_mcore.sh dev      # Switch to the PREVIOUS pin (code A/Bs only)
./scripts/switch_mcore.sh main     # Switch to the current pin
```

**Never edit the submodule working tree in place.** Such an edit runs (the checkout is on
`PYTHONPATH`) but is invisible to `git status` beyond a bare ` m 3rdparty/Megatron-LM`, so it
silently vanishes on a fresh clone and any number it produced becomes irreproducible — this
happened to the 120B champion measurement (a DDP bucket-size change; it is now the explicit
`ddp.bucket_size` field in the quickstart config, where it belongs). If a change cannot be
expressed through config, carry it as a commit of the fork pin (pushed first, as above) or vendor it
as a patch in `3rdparty/patches/megatron-lm/` — that directory's README records why each change exists
and what it is load-bearing for. Two are patch files that NO run applies. `0001-fix-moe-normalize-allgather-dispatcher-output-by-EP-.patch`
is the ONLY surviving copy of a fix whose original submodule commit no remote contains, kept
because nothing uses the `allgather` dispatcher today but the fix would be
unrecoverable if dropped. `0002` (CUDA-graph `zeros_like` on a 0-dim tensor) is
**still open upstream** — apply it if you ever enable CUDA graphs; no shipped config does.
`0003`, `0004` and `0005`, the Nano pretrain campaign's Megatron-LM changes, are carried commits of the
pin. `0003` is the port of upstream PR #4798 (EP all-to-all / compute overlap for the hybrid model,
plus local adaptations listed in the patches README), load-bearing for the campaign's EP-overlap rung
(E-044) and the Nano pretrain quickstart; it acts only with
`comm_overlap.overlap_moe_expert_parallel_comm=true`, keeps the pin's checkpoint keys for flat layer
patterns such as Nano's, and refuses Megatron-FSDP, fine-grained activation offloading,
`delay_wgrad_compute` and the `ncclep` dispatcher with the overlap on hybrid models
(Megatron-Bridge refuses packed sequences with it). `0004` keeps a HybridEP dispatch handle's
token count in device memory: on the blocking dispatch path it lived in pinned host memory that
queued kernels read after the handle was freed, which faulted under the EP overlap. `0005` is a
chunked linear cross-entropy for `HybridModel` (`cross_entropy_fusion_impl: linear`). `0006` is upstream #6114,
cherry-picked: the router's expert-bias token count broadcasts over the experts the `padding_mask` that the packed
SFT step passes so that pad tokens stay out of the MoE routers' statistics, where it used to fail. The step itself
gives each pipeline stage its sequence-parallel share of the mask, as upstream Megatron-Bridge does.
(The `overlap_p2p_comm` NaN's fix is already IN the current pin; its record-of-closed-bug
patch was retired with the investigation docs and is preserved under
`/projects/a5k/public/logs/infr71_wave2/docs/`.)

### Monitoring Long-Running Processes

Always use the **Monitor** tool (not polling loops or sleep):
```bash
tail -f /tmp/training_run.log | grep --line-buffered -E "iteration\s+[0-9]+/|Error|OOM|NCCL|Traceback|saved|completed"
```

## Commit review gate

`git commit` is intercepted: pre-commit runs on the staged files only (not `--all-files`), and the commit is blocked
until the `checklist-reviewer` subagent writes a passing `.claude/reviews/verdict.json` for the hash of the staged
diff. `geodesic-protect-verdict` ensures only that subagent's Write tool can produce the verdict, and
`geodesic-submodule-check` warns if the vendored tooling checkout drifts off its pin. The gate's pre-commit limit is
`review_gate.precommit_timeout_s` in `.claude/geodesic-config.yaml`. The PR notification channel is
`notifications.pr_notify_channel` in the same file, read by the `geodesic-pr-notify` hook; `#claude-tooling` is for
the tooling submodule's own PRs.

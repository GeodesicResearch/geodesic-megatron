# Isambard Support Request: Slingshot Software Upgrade for NCCL Hang Fix

**Submit via:** https://support.isambard.ac.uk (Zammad helpdesk — no email support)
**Facility:** Isambard-AI
**Project:** a5k
**User:** kyleobrien.a5k

---

## Ticket Title

Request: Upgrade aws-ofi-nccl (and ideally libfabric/SHS) to fix NCCL collective hangs in multi-node MoE training

## Ticket Body

### Summary

We are experiencing periodic NCCL collective hangs every 7-14 minutes during multi-node Mixture-of-Experts (MoE) training on Isambard-AI (4 nodes, 16 GH200 GPUs). After extensive investigation, we have identified **one confirmed root cause** (aws-ofi-nccl memory registration deadlock) and **plausible additional contributors** (libfabric CXI provider deadlocks) in the currently installed system libraries. All have been fixed in newer upstream releases.

We are requesting:
1. **aws-ofi-nccl >= 1.18.0** as a brics module (currently installed: 1.8.1) — this is our primary ask and highest-confidence fix
2. **Libfabric/SHS upgrade to the latest available version** (currently installed: SHS 12.0.2) — a complementary upgrade that would address additional plausible contributors; the specific target version is best determined by the Isambard admin team based on available validated releases

### Current System Software

```
libfabric: 1.22.0-SHS12.0.2_20250722155538_8dad011dfdb6 (RPM)
aws-ofi-nccl: brics/aws-ofi-nccl/1.8.1 (module)
NCCL: brics/nccl/2.26.6-1 (module, also using venv bundled 2.28.9)
```

### Problem Description

Our training workload (Nemotron 3 Nano 30B-A3B, a Mixture-of-Experts model with 128 experts) runs on 4 nodes with 16 GPUs using expert parallelism (EP=8). The MoE layers use NCCL Send/Recv all-to-all operations to route tokens across all 16 GPUs over the Slingshot interconnect.

Every 7-14 minutes, all 16 ranks hang simultaneously on an NCCL collective operation. The hang is permanent (does not self-resolve). We use nvidia-resiliency-ext's fault tolerance stack (ft_launcher + in-process restart) to automatically detect and recover from these hangs, which allows training to complete but with ~23% wall-clock overhead.

### Root Cause Analysis

We conducted a detailed investigation including NCCL debug log analysis (per-rank logs, 88MB per rank), crash interval statistics across 63 restart cycles, and systematic testing of 11 different env var configurations. Full investigation document available on request.

#### Bug 1: aws-ofi-nccl v1.8.1 Memory Registration Deadlock

The aws-ofi-nccl v1.17.0 release notes state: "Fixed deadlock in memory registration function." This deadlock occurs in the Send/Recv code path used by our MoE all-to-all dispatcher. Additionally, v1.17.2 fixes a "shutdown ordering issue on NICs that require per-endpoint memory registration (Cray Slingshot)" and v1.18.0 redesigns the threading model for multi-communicator applications.

References:
- https://github.com/aws/aws-ofi-nccl/releases/tag/v1.17.0
- https://github.com/aws/aws-ofi-nccl/releases/tag/v1.17.2
- https://github.com/aws/aws-ofi-nccl/releases/tag/v1.18.0

#### Bug 2 (Plausible Additional Contributor): Libfabric CXI Provider Deadlocks (Missing from SHS 12.0.2)

The upstream libfabric CXI provider has received two separate deadlock fixes that are NOT present in our installed SHS 12.0.2 build. These are in code paths exercised by NCCL Send/Recv and are plausible additional contributors to our hang pattern, though we have lower certainty than for Bug 1:

1. **v2.2.0** (June 2025): "Fix regression which could cause deadlock" — SRX locking path
2. **v2.3.0** (September 2025): "Fix regression which could cause deadlock" — counter thread locking

We verified this by searching all commits in the HewlettPackard/shs-libfabric fork: zero results for "deadlock", "Fix regression", or "locking" in the prov/cxi path. HPE did backport other fixes (CUDA sync_memops, read-only cached MRs, CQ wait FD) but not the deadlock fixes.

Additionally, v2.4.0 (December 2025) contains "Release TX credit when pending RNR retry" — a flow control fix that prevents TX credit starvation when a Receiver Not Ready condition triggers retry. This is also absent from SHS 12.0.2.

Newer SHS releases should contain these fixes, but the specific version that includes all three should be verified by the Isambard admin team based on available validated releases.

### Evidence

1. **NCCL debug logs** (job 3700495): Per-rank logs confirm the last operations before each hang are Send/Recv pairs on the EP=8 communicator (expert-parallel all-to-all), interleaved with ReduceScatter on the TP=2 communicator. The log simply stops with no error — a permanent collective stall.

2. **Crash interval analysis**: 63 restart cycles across 4 production jobs show crashes are time-based (10-13 min intervals), not iteration-based. All nodes detect the timeout within ~1 second of each other. No single node is a consistent trigger.

3. **No warning signs**: NVRx straggler detection shows all GPUs at 0.97-0.99 relative performance immediately before every crash. Iteration times are stable. The hang is instantaneous.

4. **Parameter tuning**: We tested 11 configurations including NCCL_PROTO=Simple, FI_CXI_RX_MATCH_MODE=software, FI_CXI_REQ_BUF_SIZE=16MB, NCCL_CUMEM_ENABLE=0, and others. Software match mode approximately doubled the crash interval but with 3x throughput penalty. No env var combination eliminates the hang.

5. **User-space build attempt**: We built aws-ofi-nccl v1.17.3 from source at ~/opt/aws-ofi-nccl-1.17.3/ but it has runtime compatibility issues with the system libfabric that require admin-level integration to resolve.

### Requested Action

1. **Primary ask: Install aws-ofi-nccl >= 1.18.0 as a brics module** (e.g., brics/aws-ofi-nccl/1.18.0). This is our highest-confidence fix. This version includes the memory registration deadlock fix (v1.17.0), Slingshot-specific shutdown fix (v1.17.2), memory leak fix (v1.17.3), and threading model redesign (v1.18.0). We have user-space builds at ~/opt/aws-ofi-nccl-1.17.3/ and ~/opt/aws-ofi-nccl-1.18.0/ that can serve as references. Requirements: NCCL >= 2.17.1, libfabric >= 1.22.0 (both already installed).

2. **Secondary/complementary: Upgrade libfabric/SHS to the latest available version.** The upstream libfabric CXI provider has received deadlock and flow control fixes that are not present in SHS 12.0.2. The specific target version is best determined by your team based on available validated releases. This would address plausible additional contributors to the hang.

3. If neither upgrade is feasible in the near term, we would welcome any of the following:
   - Just the aws-ofi-nccl module upgrade (addresses the confirmed Bug 1, likely reduces or eliminates hangs even without a libfabric upgrade)
   - A newer libfabric module alongside the current system libfabric
   - Guidance on whether our user-space aws-ofi-nccl build can be made to work with the current system libfabric

### Impact

This issue affects all multi-node MoE training workloads on Isambard-AI using NCCL over Slingshot. Our fault tolerance stack allows training to complete but with significant overhead (~23% wall-clock time lost to restart cycles). The fix would eliminate this overhead entirely and improve reliability for all NCCL users on the system.

### Attachments

- Full investigation document: `docs/investigations/slingshot-nccl-hang-investigation.md` (available on request)
- NCCL debug logs: `/projects/a5k/public/logs/network_debug/3700495/`
- User-space aws-ofi-nccl build: `~/opt/aws-ofi-nccl-1.17.3/`

---

## Notes for Submitting

- Submit at https://support.isambard.ac.uk via Zammad (no email support)
- Include project name: a5k
- Include username: kyleobrien.a5k
- Set facility: Isambard-AI
- Check back for responses — no email notifications are sent for ticket updates
- Responses occur within normal working hours only

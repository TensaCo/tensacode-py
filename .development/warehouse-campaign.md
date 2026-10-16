# Four-hour warehouse campaign — 2026-10-16

User authorized productive use of both GB10 machines while away, retaining at least
20% capacity. Deadline: **23:54 UTC**. Host: Dell-gb10-1; peer: gb10-direct /
promaxgb10-4dfb. Both have 20 CPU cores and NVIDIA GB10 unified-memory GPUs.

The campaign extends the frozen three-robot milestone with paired physical
perturbations, learned-policy comparison, inverse-vision robustness, and language
parser/reviewer evaluation. It does not manufacture a general-intelligence claim.
Failures and source-bound protocols remain in the evidence ledger.

Resource ownership proposal approved by root:
- Host simulation lanes CPU 0–3 and 4–7; inverse vision 8–9; language CUDA 10–11.
- Peer simulation lanes CPU 0–3 and 4–7; learned-policy CUDA 8–11.
- CPU 16–19 remain unassigned;12–15 available for measured bounded auxiliary work.
- At most one CUDA inference/training job per host, through an owned private MPS
  pipe at 70% requested active threads; EGL rendering uses sampled GPU headroom.
- System RAM available<25% stops owned jobs; CPU/GPU>75% pauses, <60% resumes.

Implementation: `examples/robotics/warehouse/campaign/run.py`. Durable queue output
is in `/tmp/tensorcode-warehouse-backend/` for backend qualification; subsequent
matrix lane directories will retain protocol, source hashes, logs and outcomes.
The initial paired3 second EGL/Mesa smoke uses a frozen copy of all robotics Python
sources, identical physics inputs,6CPU and12 GiB, and is a throughput qualification,
not a task-success experiment.

Guard verification: initial systemd AllowedCPUs-only child still saw all 20 CPUs;
adding taskset produced exactly CPU 18 in a3 second smoke. This improves new campaign
lane isolation without rewriting older resource evidence. CPU quotas and sampled
global resource guards are still separately applied.

## Executed qualification and initial queue

- New guard test: four scheduler contract tests pass; real4 second deadline stopped
  a60 second child. The child verified6 core quota,12 GiB RAM ceiling, zero swap and
  exact taskset affinity. Earlier AllowedCPUs-only smoke is retained as a failure.
- Actual GL: NVIDIA Corporation / NVIDIA GB10/PCIe /4.6.0 driver580.159.03.
  Frozen two-robot3 second physics smoke took2.661seconds wall, peak sampledGPU 10%,
  CPU 12.9%, minimum availableRAM 83.88%. Matched forced-Mesa attempt segfaulted
  after EGL rejected software rendering on an explicitly selected device; no
  backend-speedup claim is made.
- The v4 collector smoke took4.634 seconds and produced two finite150step
  103feature/20action BC traces. The synthetic-checkpoint adapter wiring smoke
  took6.479seconds. Both are excluded from task-performance evidence.
-273 recorded-physics RGB-D captures completed in17seconds across three replay
  sets. Development-only perception fitting was run separately; evaluation
  captures remain unread until the model candidate is frozen.
- Qwen parser/reviewer v3 runs on hostCPU 10–11 under private MPS70%; observedGPU
  utilization around39% and availableRAM around80% during initial requests.
- Baseline milestone commit: `d023685`. Four frozen teacher/robustness queue lanes
  started around20:05UTC,61runs total:21 robustness cases +32 BC train +8 BC validation.
  Initial three teacher-training qualifications and one teacher-validation run
  precede the supplemental data. Matched physical perturbation pairs share host
  and lane. SourceZIP, protocols and queues: `/tmp/tensorcode-campaign/`.
- Host/peer MuJoCo3.16.0 and ONNXRuntime1.31.0 match. GPUdriver versions differ;
  cross-host timings are not treated as algorithmic gains.
- Teacher source snapshot is immutable. Future BC versus authored comparisons
  use a separate frozen source snapshot and the new full85 second evaluation
  horizon for both policies. Training teacher stopping behavior remains unchanged.
- A deadline-bounded harvester copies only completed peer jobs every 30 seconds,
  retaining logs/resource ledgers and failed physical outcomes.

Initial qualification exposed a nominal teacher regression: only the y-offset
qualification passed both robot placements. At root's instruction, all four queue
workers were paused before further admission. Active child jobs were allowed to
finish under independent global-resource watchers and their existing cgroup caps.
The immutable rollout snapshot was not changed. Both model-language evaluation and
perception work continued while physics/controller owners diagnosed the failure.
A file-based admission pause has been added to the next scheduler revision so future
queues can pause admission without suspending the worker's own monitor.

GPU scheduling was tightened after a94% sampled utilization spike: CUDA now owns
an exclusive per-host GPU admission lease; EGL batch windows can share a lease
only with other EGL jobs. The effective CUDA MPS allocation was independently
verified as32 of 48 SMs (66.7%). GPU busy-time percentage is not the same quantity.
Existing pre-lock workers were explicitly admission-paused, drained, and resumed
in separate windows; no earlier resource samples were rewritten.

The first matched nominal85 second full-horizon recovery pair completed with
baseline failure and recovery success. Both initial-qpos hashes and the complete
source-hash dictionaries matched; only the declared grasp-recovery flag differed.
This is a development qualification, not yet held-out robustness evidence.

## Qualified recovery collection and GPU windows

The four recovery teacher qualification scenes each passed the independent strict
final-settling audit for both robots over the full85 second horizon. Four earlier
recovery early-stop qualifications had already launched before the full-horizon
amendment arrived; they remain development evidence and are excluded from the new
44 run training/validation dataset. The amended dataset keeps the original44 scene
conditions and enables grasp recovery plus full85 second holding coverage.

All 40 supplemental runs then launched from the exact same immutable expert snapshot
as the four qualified scenes. Snapshot/sourceZIP:
`/tmp/tensorcode-campaign-recovery-teacher/`; qualified full-horizon queues:
`/tmp/tensorcode-campaign-recovery-full/`; supplemental queues:
`/tmp/tensorcode-campaign-recovery-collection/`; dataset:
`/tmp/tensorcode-warehouse-bc-recovery-full/runs/`.

Host admission was paused at job boundaries for an exclusive pinned4B language
model evaluation, while peer physics and comparison rendering continued under
shared EGL leases. The4B cache was checksum-verified before inference. Download
attempts and a2 GiB per-job OOM remain recorded; host available RAM stayed healthy.
Both private MPS contexts independently verified32 of 48 SMs. Owned-daemon cleanup
timers run at23:55UTC, after the23:54UTC job deadline.

The complete21 case original authored audit is published in
`docs/results/robotics/warehouse/robustness-audit.json`, explicitly separating the
18 robustness conditions from3 teacher-only conditions and retaining every failure.
Its primary metric requires a continuous final second of valid physical placement.
The v5 two-object scene passed sensor-grounding preflight but its first160 second
physical trial failed task1; task2 therefore remained gated. That failure is retained
under `/tmp/tensorcode-warehouse-jobs/single-twojobs-v1/` for diagnosis.


All 44 full-horizon recovery scenes completed and passed dataset/source audits,
producing 70 training and 18 validation robot episodes. The primary dataset retains
all failures; a preregistered ablation uses 63 successful training episodes and the
same 18 validation episodes. Both GRU fits finished on the peer, while both diffusion
fits run sequentially on the host. Training uses an immutable shared source archive
with SHA256 `2f9a8cbea7ae7cc12249fdee58755315e275e5b8d8dc073d617722d764c9db0f`.

Each fresh CUDA job now verifies its actual context and a real CUDA operation inside
the bounded unit before execution. MPS reports 32 of 48 equivalent SMs. This is a
bounded execution allocation, not a dedicated spatial partition or a guarantee of
unused memory bandwidth. CPU affinity, RAM guards, the exclusive CUDA lease, and
raw utilization sampling remain active. CUDA busy-time percentage alone no longer
triggers duty pauses; the earlier logs and resource-policy amendment are retained.


All four frozen GRU and diffusion fits completed. Their nominal and declared
validation-offset rollouts use one common peer-only evaluation source archive,
SHA256 `5f6a72fce901073dc4964c5a66196373e202c7b6b6569a4e451ee9d157910bf0`,
with fixed 85-second horizons. The GRU development rollouts all failed strict
placement; complete nominal action replay matched the model predictions exactly.
These failures are retained and final regimes remain unopened at this stage.

Separate development experiments produced promising results requiring independent
physics audit: the fifth two-job controller revision completed both sequential
pick/place tasks, and disabling offscreen multisampling changed the nominal
non-recovery controller from one failed robot to two reported successes. Neither
change has been silently applied to the frozen policy comparison or training data.
A Cycles CUDA benchmark rendered an actual recorded frame successfully; the approved
480-frame sequence now runs with a 35-minute deadline under the same resource guard.


To use peer CUDA windows during host photoreal rendering, the pinned public Qwen4B
model directory alone was copied to the peer. All 13 files passed recomputed Hub
SHA256/git-blob digests. A separate dependency overlay supplies Transformers 5.17.0,
matching the host; the existing peer π0 environment is unchanged. The peer uses
Python 3.11 and a different NVIDIA driver, so these language development runs do
not support identical-backend timing comparisons with host runs.

A preregistered three-round TRAIN-only DAgger extension is separately snapshotted.
Its first missing-protocol-path failure occurred before simulation and is retained.
The successful three-second wiring retry independently reproduced every learner
action and verified seeded expert/learner attribution. Full qualification precedes
bulk collection; none of these runs alters the frozen original model comparison.

The actual first-pass pinned 4B proposal for three humanoids was certified against
complete language semantics, then grounded by the live sensor executor. Its six
physical jobs completed at 125.12 simulation seconds under ALG.2 dispatch. The run
retains the exact instruction, proposal and protocol bytes under `mission/`; the
independent auditor binds these to the executed tree and text. A documented
infrastructure-only runtime extension changed the owned unit's wall cap from 600
to 900 seconds before the outcome, without changing the 180-second simulation
horizon or action source. The temporary unit override was removed afterward.

The native Ψ comparison is separately scheduled on the peer, whose Isaac/SIMPLE
stack is required. Its hard stop is 23:20 UTC, reserving the final 34 minutes for
matched warehouse evaluation. Native jobs combine private MPS allocation with a
75% sampled GPU busy-time pause guard because RTX/OptiX graphics are not bounded
by MPS. Optional ablation admission also depends on remaining time; incomplete
arms are retained and reported, never silently replaced.

### 21:26 UTC: final bindings and native-policy window

All peer EGL development/render lanes drained before native admission. The native
Ψ suite began at 21:25:42 UTC on CPUs 0–3 with a 48 GiB cgroup, exclusive owned GPU
lease, verified 32/48 equivalent-SM CUDA allocation, and the additional sampled
GPU-busy pause guard required for its RTX renderer. Its hard deadline is 23:20 UTC,
leaving 34 minutes for the matched physical-policy evaluation. MPS is not a
physical SM or memory-bandwidth reservation.

The 40-run final comparison is prepared, not yet admitted, in six two-CPU lanes
with 7/7/7/7/6/6 jobs. Every command explicitly requests 85 seconds, full horizon,
BC recording, camera samples 4, and the shared 0.018-radian arm slew limit. All
runs use the same peer source archive `3b854aa4f1b28972cc7a62945ff8d774adc8e35459dc5cc3ac6905e3bfe13b44`.
The execution manifest (`6d60569d180955e6fd17182e5583a8f5001eda1472c7b4bfb8a84c31a09f1827`)
and disclosure (`6aca9ffc6c910215880e2d2b05bda1331b3032fd7913db15a593b8953a72ad92`)
were copied to the peer and verified before admission. These five conditions
were previously exercised by the original authored baseline; they are excluded
from learned TRAIN/VAL, but are not globally unseen conditions. Each policy has
five physical episodes containing ten correlated robot placements.

The fresh 20-request nested language challenge and four cinematic chapter
framing previews completed successfully on the host; semantic and visual audits
remain the responsibility of their respective owners. All 14 capped development
runs, two geometry development runs, and ten new camera-factorial runs were
harvested for independent audits. Process exit zero is not a physical-success claim.

### Native graphics startup retry and productive qualification

The first native attempt stalled during Isaac/Hydra startup and was interrupted;
no completed evaluation episode exists for that attempt. Its unit-stop exit code
was zero, but its ledger reason is `signal`, explicitly an interrupted run. All
partial files and resource logs are retained. An approved isolated launcher
removes MPS environment variables only from the Isaac graphics child, keeps the
parent policy's private MPS allocation, adds worker logs and a 120-second startup
timeout. The frozen policy runner, checkpoint and evaluation protocol are
unchanged. The default `/tmp/nvidia-mps` daemon directory was absent.

Retry 2 passed graphics startup in 9.838 seconds and completed the first four
nominal Ψ0.1 episodes with four recorded successes (495 seconds including setup).
This qualifies the remaining core arms for admission; it is not yet a comparative
result. The graphics watchdog demonstrably pauses the entire owned unit after
sampled pressure and resumes after recovery. Raw transient GPU busy samples above
80% remain in the evidence; this reactive guard does not guarantee instantaneous
unused graphics capacity. The hard 23:20 UTC cutoff remains unchanged.

### Final-matrix status correction

The progress update “all 40 completed/no errors” was incorrect: it counted finished
processes without checking their exit status or final report. The original matrix
has **40 attempted processes, 33 report-producing runs, and seven runtime errors**.
The seven long-camera-outage runs raised an out-of-order scene-observation
exception during repeated floating-point timestamp reconstruction. Their partial
outputs and nonzero-exit ledgers remain intact; they are not physical failures or
successful evaluations. Independent physical audits determine validity and success.

The offline summarizer now reports finished processes, successful exits, nonzero
exits, interruptions, and available physical reports separately. It deliberately
does not infer independently valid evidence from any of these counts. A focused
regression test distinguishes a physical failure with exit zero, a runtime error
without a report, and an interrupted process whose stop command returns zero.

A separate seven-case timestamp-repair diagnostic was authorized and frozen after
19 tests. It retains the original protocol and failure links in an amendment;
only timestamp reconciliation and the diagnostic protocol differ from the original
source. No model weights or control parameters changed, and no original result is
overwritten or silently replaced. Its separate deadline is 23:45 UTC.

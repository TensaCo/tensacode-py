# Four-hour robotics compute campaign

Window: **2026-10-16, 19:54–23:54 UTC**. Both GB10s performed bounded training,
simulation, inference and rendering. Owned compute finished before the deadline;
private MPS daemons and cleanup timers were stopped. Unrelated processes were
left alone.

## Counts and evidence

The resource summary covers **266 finished guard jobs**, not 266 physical episodes.
Of these, 246 exited successfully, 14 had nonzero exits, and seven were interrupted;
nonzero exits and interruptions can overlap. A successful process is not scientific
success. There are 195 available physical-report files, whose validity must be
established by the individual experiment audits. Read-only audits and jobs guarded
by other agents' separate tools are outside this aggregate.

An earlier progress update incorrectly called all 40 final processes complete
without checking their exit status. The original comparison has **33 reports and
seven runtime-incomplete long-outage runs**, all retained. A separate seven-run
numerical timestamp repair completed with seven successful exits and seven reports;
it does not overwrite the original matrix or change policy weights.

- [Original final-policy evidence](../../warehouse/learned-comparison/final/)
- [Separately labeled timestamp repair](../../warehouse/learned-comparison/timestamp-repair/)
- [Native policy suite](../../baselines/psi-native/): four arms each scored 12/12;
  this suite demonstrated no success-rate advantage for local structural biases.
- [DAgger2 qualification](../../warehouse/learned-comparison/dagger2/): the fall
  failed its preregistered gate, so no bulk collection or fine-tuning followed.

The native suite's first Isaac startup attempt stalled and was interrupted. Its
logs were retained. The bounded retry removed MPS variables only from the graphics
child while keeping the policy model's private CUDA allocation.

## Observed resources

| Host | Minimum available RAM | Peak sampled total CPU | Peak sampled GPU busy time |
|---|---:|---:|---:|
| Host GB10 | 73.76% | 46.85% | 96% |
| Peer GB10 | 61.34% | 51.23% | 96% |

All 26 recorded CUDA preflights verified **32 of 48 equivalent SMs** and their
assigned CPU affinity. MPS limits execution allocation; it does not reserve
physical SMs or memory bandwidth. Graphics guards paused jobs reactively after
pressure samples. The 96% transients remain in the evidence: this is **not a claim
that 20% of the GPU was instantaneously free**. CPU/RAM reserves and per-job limits
were enforced independently.

[Resource policy](resource-policy.json) describes the executed controls and their
changes. [Resource summary](resource-summary.json) contains per-job outcomes,
resource samples, unit identifiers and source-ledger hashes. Job wall times overlap
for parallel work and must not be summed as campaign elapsed time. Machine-local
scratch paths are omitted; raw experiment logs remain in their existing study
bundles. Copied guard source describes the final revision, not a retroactive rewrite
of earlier launch policies.

[Host cleanup](host-cleanup.json), [peer cleanup](peer-cleanup.json), and
[final cleanup verification](final-cleanup.json) record owned-process shutdown.
[SHA256 manifest](sha256.json) binds every file in this summary bundle.

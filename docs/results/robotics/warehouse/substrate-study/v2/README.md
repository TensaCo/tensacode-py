# Dispatch with explicit cost resolution

The Coordinator now declares which near-equal costs count as a tie. The same
112 primary observation-interface cases still agree across ALG.0/ALG.1/ALG.2.
All **140 previously retained assignment counterexamples now choose robot a on
all three substrates**. Raw backend cost differences remain visible; no numerical
results were rounded or rewritten. This is a narrow decision-policy improvement,
not universal substrate equivalence or a new robot performance result.

## Policy

Squared-distance costs are sorted by raw numeric value. Each group's membership
is measured against its smallest cost, with an inclusive threshold:

```
cost - group_minimum <= 1e-12 m² + 1e-12 * abs(group_minimum)
```

Robot ID orders candidates within a group. The next group starts at the first
remaining raw cost; a chain of adjacent close costs cannot expand the first group.
For example, with absolute tolerance 1 and zero relative tolerance, costs
`0, 0.75, 1.5` form `[0, 0.75]` and `[1.5]`, not one chained group.

Competing tasks are processed in lexical task-ID order, independent of input list
order. Dependencies still require completed execution evidence before dispatch;
this change does not infer completion or schedule dependent tasks prematurely.
Callers requiring a different priority must encode it in task IDs or introduce an
explicit future priority contract. This is greedy resource-aware assignment, not
global scheduling optimization.

Both tolerances are configurable Coordinator fields:
`cost_absolute_tolerance` and `cost_relative_tolerance`. They must be finite and
nonnegative. Setting both to zero restores exact-cost robot selection; lexical
task priority remains. No tensor runtime or kernel arithmetic changed. Existing
physics, teacher, model-evaluation and historical audit receipts were not retuned.

## What remains unproved

Costs near the group boundary can still move between groups under rounding. The
policy deliberately makes distances within the declared resolution equivalent for
selection; it does not prove all real-valued minima produce identical choices on
every backend. At large cost magnitudes the relative tolerance increases the
absolute group width. A formal application-specific stability claim needs bounds
on arithmetic/observation error and margins separating the chosen group from the
next candidate. Sensor accuracy, collision avoidance, and task feasibility are
outside this audit.

## Observations

One Torch thread, CPU cores 0–1, two warmup passes and eight measured passes; 896
full dispatch calls per substrate. Compilation, fixture creation, state reset and
serialization are excluded. Timings are host observations under uncontrolled
concurrent load, not latency guarantees.

| Substrate | p50 ms | p95 ms | maximum observed ms |
| --- | ---: | ---: | ---: |
| ALG.0 | 0.164803 | 0.196548 | 0.240661 |
| ALG.1 | 0.672718 | 0.727983 | 1.096262 |
| ALG.2 hard | 4.007218 | 4.092964 | 4.316440 |

The same 960 permutation pairs were screened, yielding the same 140 raw-cost
choice disagreements. Those 140 candidates were each verified through all three
actual Coordinators and produced **zero assignment disagreements** under the new
policy. The other 820 screening pairs were not full Coordinator replay cases.
The 112-case primary protocol and this targeted repair replay are separate sets;
the repair replay is not an independent holdout.

## Reproduce

From the Python repository:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m examples.robotics.warehouse.substrate_audit \
  --output docs/results/robotics/warehouse/substrate-study/v2
.venv/bin/python -m pytest tests/examples/test_warehouse_cost_ties.py -q
```

Protocols were written before this run. Reproduction overwrites this version's
generated receipts, never the original parent-directory evidence.

- [Version 2 protocol](protocol.json), [results and source hashes](results.json)
- [Primary cases and gates](cases.json), [raw timing samples](timings.json)
- [Tie replay protocol](tie-probe-protocol.json), [retained raw differences and new decisions](tie-counterexamples.json)
- [Original results and unresolved counterexamples](../README.md)

## Bound source snapshot

[source-manifest.json](source-manifest.json) binds this version to Git commit
`40af15a3a772422ed5a46a76ac82fbdee049ea7d` plus the exact archived audit and Coordinator
source overlays in `source/`. Their SHA-256 hashes match the original execution
receipt. Other recorded runtime sources match that exact base commit. All six
protocol/result/data files are separately hashed; no raw output was changed.

For exact source reproduction, export that commit into a separate directory and
copy this version’s `source/` contents over its repository root. Copy this
version’s two protocol JSON files into a fresh output directory. Run the archived
audit module from the exported repository with `--output` set to that directory
and `PYTHONPATH` pointing first to its `src/` and repository root, so an existing
editable installation cannot select the live checkout. Use the recorded Python
and Torch versions and the one-thread environment settings above. Timing varies
with load. Running the live working-tree script is not a substitute for this
source-pinned reproduction.

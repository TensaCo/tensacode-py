# Warehouse dispatch substrate audit

This is the original exact-cost audit receipt. The current Coordinator now uses
an explicit anchored cost-resolution policy; its separately frozen [version 2
audit](v2/README.md) resolves the 140 retained counterexamples. The original JSON
evidence below is unchanged. Descriptions of exact-cost selection below refer to
the original policy.

The current warehouse dispatch kernel ran through ALG.0 (NumPy symbolic), ALG.1
(Torch tensor circuit), and ALG.2 (hard-routed attention machine). The frozen
primary protocol's **112 observation-interface cases produced the same decisions
and eligibility gates**. However, an explicitly separate follow-up probe found
**140 actual assignment differences among 960 mathematically tied position
pairs**. Therefore this study does **not** establish universal decision parity.

These are synthetic observation records, not robot trials, simulator ground truth,
perception accuracy measurements, or learned-policy comparisons. Every fixture
source says `synthetic-audit`. No GPU or simulator was used.

## What is compiled

The real `Coordinator` invokes the compiled kernel to multiply five eligibility
gates and compute squared distance from estimated robot/object positions.
Freshness/confidence tests, capability contracts, task dependencies, route
reservation, and final minimum-cost/ID selection remain Python host logic. The
study exercises that complete dispatch path; it does not suggest that the whole
warehouse controller is compiled into attention.

The declared domain is binary64. Seeded position estimates lie within ±10 m;
named cases also test finite extreme coordinates, underflow and adjacent floating
point values. Timestamp tests use `now=0` so subtraction does not erase the
nextafter boundary immediately above/below 2 seconds. Gate fixtures cover exact
object/destination capability pairs, occupied routes, failed dependencies and
unavailable robots. The underflow fixture deliberately records an arithmetic tie
where mathematical distances differ: squared `1e-200` rounds to zero.

## CPU dispatch timing

Actual full dispatch calls, excluding compilation, fixture construction, state
reset and serialization; two warmup passes, eight measured passes. Substrate order
rotates by pass and case. One Torch thread, affinity restricted to CPU cores 0–1,
896 measured calls per substrate. The recorded host is Linux aarch64, Python
3.13.15, Torch 2.14.0+cu130; all tensors executed on CPU.

| Substrate | p50 ms | p95 ms | maximum observed ms |
| --- | ---: | ---: | ---: |
| ALG.0 | 0.162419 | 0.185700 | 0.239188 |
| ALG.1 | 0.773182 | 0.849856 | 0.996243 |
| ALG.2 hard | 4.115900 | 4.223070 | 4.511571 |

These are observations under concurrent host load, not worst-case execution-time
bounds or a latency guarantee. ALG.2 is slower for this small CPU graph; the
study provides no acceleration claim.

## Preserved numerical counterexamples

The primary study retained 82 per-substrate cost differences rather than
rounding them away. They caused no primary-study assignment changes. After seeing
those differences, a separately declared exploratory protocol screened 64 seeded
vectors and every pair of their six coordinate permutations (960 pairs). Every
screening disagreement was then checked through all three actual Coordinators;
140 changed assignment. These selected follow-up cases are not added to the
primary study denominator, and their rate does not estimate natural prevalence.

For example, with an observed object at `(0, 0, 0)` and fresh evidence:

- Robot a: `(8.582060394080326, 2.0888315756746274, 7.831380023295591)`.
- Robot b: `(8.582060394080326, 7.831380023295591, 2.0888315756746274)`.
- ALG.0 costs are `139.34549102845074` and `139.34549102845077`, choosing a.
- ALG.1/ALG.2 reverse those values, choosing b.

Coordinate permutation preserves the mathematical distance. Binary64 reduction
association creates a few ULPs of difference, and exact minimum selection exposes
it. The current robot-ID tie breaker only applies when computed costs are exactly
equal. A future explicit cost-resolution/tie policy or a certified decision margin
would be needed for stronger decision-stability claims; neither is silently added
by this audit. Hard attention routing itself agrees with ALG.1 in these examples.

## Reproduce and inspect

Run from the Python repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m examples.robotics.warehouse.substrate_audit
.venv/bin/python -m pytest tests/examples/test_warehouse_substrate_audit.py -q
```

The runner reads existing protocols before generating either fixture set. It
never rewrites protocols. Reproduction overwrites generated evidence and timing
will vary; compare its result to the checked-in receipt before replacing it.

- [Primary frozen protocol](protocol.json)
- [Primary results and source hashes](results.json)
- [Every primary fixture, output and gate](cases.json)
- [Every measured dispatch duration](timings.json)
- [Separately declared exploratory protocol](tie-probe-protocol.json)
- [Every retained actual dispatch counterexample](tie-counterexamples.json)

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

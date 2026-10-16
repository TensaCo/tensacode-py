# Guarded G1 manipulation sessions

Open these sessions through the [robotics lab](../index.html). The video comparison
uses `development-height-v1`: a successful early guarded-controller run beside the
original raised-pedestal trajectory-retargeting failure. Subsequent source changes
and every completed current run are labeled in [results.json](results.json).

The final protocol includes six development/regression conditions and one newly
combined height/XY condition declared after freezing the controller. Results are
single deterministic episodes, not estimated success probabilities. The first
shifted-target attempt aborted on unilateral contact; that diagnostic trace is
retained alongside the revised controller's outcome.

`runs/*` contains compressed records, physical configurations, independently
recomputed sensor errors, original outcomes, transition events and resource logs.
Final runs also contain source hashes and their exact source snapshot. Earlier
development runs without snapshots are marked explicitly. `runtime-limit/`
records the interrupted full ALG.2 quadratic-iteration run; no successful physical
result is claimed for that attempt. The separate ALG.0 run used convergence-gated
SciPy fallbacks.

The controller consumes RGB-D, IMU, encoders and tactile measurements. Simulator
world poses appear only in recorded evaluation/replay columns. Geometry assumes a
known upright cube and support footprint; the pickup anchor is frozen at reach
onset. Local tool-clearance samples do not certify whole-body or continuous-path
safety. Warehouse scenery is decorative.

Reproduction and controller details: [G1 manipulation example](../../../../examples/robotics/transfer/README.md).

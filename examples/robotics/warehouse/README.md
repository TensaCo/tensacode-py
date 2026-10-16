# Grounded warehouse workcells

This example connects language parsing, a revisioned observed scene, compiled
dispatch and physical G1 manipulation. The demonstrated fleets have two or three humanoids
in separate calibrated workcells within **one MuJoCo simulation**. Each robot
walks, grasps a parcel through frictional contact, carries it to a receiver,
releases it and withdraws. The demonstration is not a swarm benchmark, shared
aisle planner, physical handoff, or general robotic intelligence result.

The instruction tree determines execution: `while` produces independent tasks;
`then` creates a completion dependency. The coordinator actually executes the
ALG.0/1/2-compatible readiness and assignment-cost kernel. It reserves robots,
objects and receiving surfaces. Route resource names are supported by the
dispatcher but are not geometric collision avoidance.

## Observation and language boundaries

- Shared rendered RGB-D supplies object/receiver estimates. The perception
  diagnostic fits known geometric hypotheses to depth and color; known colors,
  dimensions and calibrated cameras are supplied priors. This is bounded inverse
  graphics, not open-world visual recognition.
- Robot proprioception is idealized joint/IMU sensing, with visual/contact root
  estimation and initial calibration. Simulator truth supplies independent
  scoring and fall termination. See each archived `sensor-contract.json`.
- Scene entities, containment, relations, uncertainty and revisions are explicit
  data. QA answers only supported observations and abstains on stale or ambiguous
  references. A count means observed instances, not all inventory in the building.
- The authored grammar has a documented restricted contract. The optional pinned
  Qwen parser proposes data-only trees; validation and observed-scene grounding
  admit them. Neither parser learns new manipulation skills. Its evaluation
  retains rejected proposals and unsupported language; see the language reports.
- The physical controller currently accepts only its own workcell's object and
  receiver. A syntactically valid arbitrary task is not an executable new skill.

## Evidence

The [dashboard](../../../docs/results/robotics/index.html) links the recordings.
The machine-readable [results](../../../docs/results/robotics/warehouse/results.json)
distinguish completed final trials, developmental outcomes and interruptions.
Every packaged run retains source snapshots, probe configurations, full physical
records, sensor inputs, instruction trees, events and independently recomputed
validation. Per-robot dashboard traces copy actual recorded observations and
controller outputs; evaluation fields remain in the separate full records.

The final two-robot pair completes **4/4 placements across two episodes**; a
separate three-robot episode exercises ALG.2 dispatch:

| Instruction condition | Physical outcome | Simulated elapsed time |
| --- | --- | --- |
| Model-parsed parallel work | 2/2 placements | 52.30 s |
| Sequential dependency | 2/2 placements | 100.70 s |
| Three-robot parallel, ALG.2 dispatch | 3/3 placements | 55.38 s |

Across the three final episodes, **7/7 placements succeed**. The three-robot
episode executes the dispatch kernel through ALG.2 in shared physics; it has
1.10–1.76 mm final placement XY error and 4.14–7.20 s of independently measured
settled placement. It is a scaling demonstration, not a matched comparison
proving ALG.2 better than ALG.0 or ALG.1.

The pair has identical initial robot configurations and action-source hashes.
Parallel work reduces elapsed time **48.06% (1.925× throughput for this pair)**.
This demonstrates instruction-conditioned concurrency between independent
workcells. It does not show a better manipulation policy or general scheduling
optimality. In the sequential trace, robot 1 is assigned at 53.08 s after robot 0
completes at 53.06 s; the language dependency changes actual physical execution.

`questions.json` preserves sensor-grounded model location QA and authored
count/ambiguity/staleness controls. Initial observations support the model's
parcel-location question. The parallel episode's final inverse-graphics
observations fall below the QA confidence threshold, so final parcel-location
and count questions abstain. No evaluation-ground-truth positions fill that gap.

`two-parallel-v1` physically completed both placements, but its coordinator used
evaluation truth to gate completion. It is retained as a **developmental,
evaluation-coupled result**, excluded from the final instruction comparison.
`two-sequence-v2` was interrupted when that boundary problem was identified; no
completed outcome is claimed. The v3 executor gates dependencies on controller
completion and independently evaluates physical placement afterward.

Compare final parallel and sequential timing only when their
`action_sources_fingerprint` values match. This fingerprint covers execution,
live sensor processing, deterministic language, dispatch and manipulation sources;
presentation renderers, the local server, offline inverse-graphics diagnostics
and the foundation parser are excluded. Parser proposals and their source
receipts are retained separately: physics consumes an already validated tree.
A timing improvement from
parallel independent workcells demonstrates instruction-conditioned concurrency,
not better manipulation than an RL or VLA baseline.

## Reproduce and inspect

Run from the Python repository with the robotics dependencies and pinned G1
assets installed. Use `examples/robotics/g1/budget.py` for simulation, model
inference and rendering; preserve the required host/peer resource headroom.

```bash
python -m examples.robotics.warehouse.package --source /tmp/tensorcode-warehouse
python -m examples.robotics.warehouse.validate docs/results/robotics/warehouse/runs/model-parallel-v3
python -m examples.robotics.warehouse.server \
  --scene docs/results/robotics/warehouse/runs/model-parallel-v3/initial-scene-graph.json
```

The loopback server previews plans and answers questions against a **recorded**
observation. It does not send actions to a robot. The separate physics runner
consumes instruction trees. Replaying physical archives still requires the
upstream robot meshes referenced by the scene; they are not duplicated here.

`package.py` excludes unfinished trials unless they contain an explicit
interruption record. Short integration smokes and renderer caches are excluded
from task evidence. Its compressed records have deterministic gzip timestamps;
`run-files.sha256.json` records the archived bytes. See [rendering](RENDERING.md)
for the distinction between sensor images, replay video and presentation stills.

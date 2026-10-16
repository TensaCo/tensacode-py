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

The repeated-job controller now backs away, resets its arm using measured joint
tracking, and walks to a second pickup station without resetting the scene.
The retained v5 episodes complete two jobs with one G1 (111.70 s), four with two
G1s (118.60 s), and six with three G1s (125.12 s). The three-G1 run executes an
actual pinned Qwen3-4B proposal through a complete-language certificate and ALG.2
dispatch. Its six final XY errors are 0.794–5.821 mm. These are individual
qualified development/scaling episodes; earlier failed revisions remain archived.

The [multi-job evidence](../../../docs/results/robotics/warehouse/multi-job-study/runs/)
retains exact input/model receipts, controller sources, sensor observations,
replay states and separate evaluator measurements. Run `python -m
examples.robotics.warehouse.validate_jobs RUN_DIRECTORY` to check final continuous
settling, dependency ordering, source hashes and any recorded model-mission
binding. This checks retained measurements; it does not reconstruct dynamics or
attest the authenticity of a model-service response.

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

## Capability and lifecycle campaign

`jobs.py` adds whole-mission capability admission, finite-age dispatch and
recorded-gate explanations. Each robot has an explicit supported object and
destination set; an unsupported job rejects the mission before any assignment.
An optional exact `(object, destination)` pair list further restricts that scope:
supporting A→A and B→B does not imply support for A→B. Both mission admission and
the compiled dispatch gate enforce this constraint.
A failed task does not automatically retry or release its dependent jobs.
Repeated-job DAGs retain admitted object identities while reading their latest
observed positions at every dispatch. Relational descriptions do not silently
rebind to another object midway through a mission.

The [coordination protocol](coordination-protocol.json) and additional exact-pair
and receipt checks have 22 passing reasoning/lifecycle tests, including ALG.0/1/2 parity. The
[offline admission ablation](../../../docs/results/robotics/warehouse/coordination-v4/results.json)
uses archived sensor observations and robot root estimates. Both arms admit two
fresh jobs. Advancing only the query clock by three seconds causes the freshness
guard to block both; disabling its effective age limit admits two stale jobs.
These are admission results, **not additional physical trials or a safety proof**.
The v3 physical episodes above predate integration of this lifecycle wrapper.

`JobScheduler.explain("Why is r1 waiting?")` reports the actual dependency,
resource, availability, freshness and capability gates with the checked scene
revision and event time. It returns unknown before any dispatch check. It does
not ask a language model to invent a reason.
`JobScheduler.snapshot()` preserves these events as a data-only receipt.
`RecordedScheduler` in `mission_receipts.py` exposes only historical explanations;
it cannot dispatch or resume the archived mission. A receipt is evidence to
review, not a certificate of its source's authenticity.

## Reproduce and inspect

Run from the Python repository with the robotics dependencies and pinned G1
assets installed. Use `examples/robotics/g1/budget.py` for simulation, model
inference and rendering; preserve the required host/peer resource headroom.

```bash
python -m examples.robotics.warehouse.package --source /tmp/tensorcode-warehouse
python -m examples.robotics.warehouse.coordination_eval
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

### Recorded mission questions

The local preview server can load an archived scheduler receipt alongside an
observed scene:

```bash
python -m examples.robotics.warehouse.server \
  --scene PATH/initial-scene-graph.json \
  --scheduler PATH/scheduler-waiting.json
```

`Why is r1 waiting?` reads the recorded dispatch gates and dependencies. The
receipt adapter cannot dispatch or finish tasks. With `--parser model`, the model
proposes a query; a complete controlled-language certificate must agree before
an answer is read from evidence. Unsupported conditions, implied causes and
missing observations produce rejection or abstention. This is historical QA,
not a live robot connection or a guarantee for arbitrary English.

New v4 evidence can be audited with `python -m
examples.robotics.warehouse.validate_v4 RUN_DIRECTORY`. Its primary
`strict_current_settle_success` requires one continuous second of valid placement
ending at the final sample. Earlier longest-hold scores and controller completion
flags remain available separately; neither replaces this final physical check.

# Algorithm substrates and robotics — 2026-10-15

Owner requested the framework and a simulated robotics policy before 20:00 UTC,
prioritizing ALG.0 -> ALG.2 and conversions between authored representations.
General algorithm discovery from arbitrary pretrained transformers was explicitly
deprioritized. Work began at 19:13 UTC.

## Implemented

- Versioned, data-only, statically shaped tensor algorithm IR, sparse operand-role
  and next-state topology, explicit parameters and transactional simultaneous state.
- AST source frontend and explicit builder; composition and standalone NumPy export.
- ALG.0 NumPy reference; ALG.1 native PyTorch circuit; ALG.2 actual scaled-dot-product
  attention operand reads with supplied local operations and execution schedule.
- Hard-routing conversions export current trained weights. Finite soft routing is
  explicitly approximate and cannot masquerade as exact symbolic conversion.
- Contextual structural-bias compilation: role-specific query/key bindings, null
  slot, Pq A Pk^T, zero-initialized head/role strengths, complete data-only persistence.
- Native pretrained MHA and standard Q/K/V/O projection factories, preserving
  copied weights and explicit masks. No automatic model patching or hidden ontology.
- Exact structural import of supported native feed-forward and MHA computation;
  shared feed-forward parameters retained. This is not semantic algorithm discovery.
- Explicit recurrent unrolling and gradients, empirical output/state replacement
  validation, shape/finite/budget checks, and failure counterexamples.
- TypeScript artifact execution, native-autograd circuits, hard/soft attention,
  trained parameter export, and Python/TypeScript numerical interoperability.

## Evidence

- Python: 88 tests passed across algorithm, existing program and training suites.
- TypeScript: final full suite passed 1,564 tests across 113 files, including
  dtype conversion and persistence; typecheck and build pass.
- Source formatting and lint on the new Python implementation and examples pass.
- Site documentation build and internal link check pass: 1,401 internal links
  across 30 pages. All five Python/JavaScript site demo comparisons pass.
- `examples/robotics/calibration/robotics_policy.py`: completed 20-joint MuJoCo rollouts in all three
  substrates, synthetic actuator-residual training through ALG.2, export/reload,
  frozen-controller evaluation on a different trajectory and 25% stronger disturbance,
  rendered replay and source-hashed provenance. Committed artifacts live in
  `docs/results/robotics/calibration/`.
- Main joint RMS tracking error: baseline 0.01039305595 rad, corrected
  0.00057636997 rad; ALG.0/1/2 agree. Shifted case: baseline 0.01192848846 rad,
  corrected 0.00251780939 rad. Supervision is an authored synthetic calibration
  law; no online system identification or RL is claimed.
- Pinned cached pretrained CLIP vision attention: 50 image tokens, zero-bias
  max output error 0.0, nonzero relation-strength gradient, nonzero bias changes
  output. This establishes integration and gradient plumbing, not semantic binding
  or execution of the robotics algorithm inside CLIP.

## Boundaries

This is a general tensor algorithm framework with robotics in an example, not in
core schemas. ALG.2 hard mode uses supplied local arithmetic and scheduling; it
is not a vanilla transformer synthesized for arbitrary Python. ALG.2 bias mode
only supplies an inductive prior; its binding networks are initially untrained.
There is no demonstrated learned problem identification, universal neural
compilation, arbitrary-pretrained-model decompilation, continuous-domain proof,
unbounded execution, general diffusion substrate or learned locomotion.

The robot is a fixed-base four-limb, twenty-hinge mechanism. The controller is
an authored impedance law using model-based bias-force compensation. Learned
coefficients correct the supplied synthetic actuator disturbance. Some observed
ALG.2 iterations exceed the 5 ms simulation step: the example advances physics
offline and makes no hard-real-time or hardware-safety claim.

Next research: train/validate bindings and relation use on a task with held-out
identities and compositions, comparing equal-information ordinary attention and
bias removal/permutation controls. Do not infer semantic grounding from the
zero-gate or gradient tests.

## Annotated results presentation

`docs/results/robotics/calibration/index.html` is a portable report with two embedded
side-by-side MP4 comparisons, error traces, controller and substrate diagrams,
original metrics and explicit baseline limitations. The generator
`examples/robotics/calibration/build_robotics_report.py` replays the committed frozen policy without
retraining and rejects mismatches in RMS, worst joint error or peak torque against
the original results. Replay trajectories and metrics are retained in
`presentation-replay.json`; reported compute timings remain the original unrendered
measurements. The report distinguishes residual-learning gains from substrate
fidelity and pretrained-adapter integration, for which no task gain was measured.

Presentation validation: Chrome loaded both embedded 1280×800, six-second videos;
playback, quarter-speed control and JSON download passed. A 390-pixel mobile
viewport had no document overflow, and no JavaScript errors were observed.
The report generator passes scoped Ruff and Python compilation checks.

## Free-base humanoid walking and grasping

The owner's requested humanoid follow-up is implemented in `examples/robotics/humanoid/humanoid_policy.py`.
The pinned Unitree RL mjlab G1 model and supplied walking MLP are combined with
an authored two-jaw attachment, Cartesian reach controller and staged grasp/lift/
hold/release controller. The network and 31-joint servo execute through ALG.0/1/2;
IK and task scheduling remain host code. No new training or pretrained transformer
semantic grounding is claimed. Arm position/velocity channels are neutralized in
the walking observation during manipulation; actual base/leg feedback remains.

Recorded peer evaluation: native ONNX and all three substrates succeed, walking
0.949 m and lifting an 80 g cube 0.152 m, with 2.94 seconds of bilateral contact
above a 0.10 m lift. Maximum action error on 67 sampled observations is 7.16e-7.
An open-finger native control fails to lift; native closed-finger control succeeds.
Two additional supplied placements/masses (60/100 g) also succeed under ALG.0.
These are deterministic examples, not a statistical robustness result. Root and
object are free bodies; contacts lift the object without welds or pose overwrites.
Robot self-collisions are disabled, and the gripper is not the stock hand.

Host work paused under unrelated CPU load and was stopped. Final jobs ran on the
GB10 peer with CPU-only inference, six-core affinity/quota, 12 GiB memory cap and
no swap; the wrapper pauses over 75% total CPU use and stops below 23% available
memory. Evaluation peak total CPU was 19.9%; minimum available memory was 95.5%.
Numerical evidence, exact execution driver, model/weight provenance, traces and
resource samples are in `docs/results/robotics/humanoid/`.
The report at `docs/results/robotics/humanoid/index.html` now leads with the G1;
the earlier calibration report is preserved as `calibration.html`.

Final presentation checks: both embedded videos decode (22.56 s main, 10.56 s
control); playback, stage seeking, quarter-speed, JSON download and local links
pass in Chrome. The mobile page has no document overflow and no JavaScript
errors. Evidence validation recalculates all seven traces' lift/contact metrics,
checks artifact hashes and 12 frozen-policy action samples, and confirms 31
actuators, free root/object joints, no equality constraints and no mocap bodies.
The actual renderer is Mesa llvmpipe (LLVM 20.1.2), recorded in `renderer.json`.

## Example and evidence organization

Runnable examples now live under topic directories in both language repositories.
Robotics source and reports use matching `robotics/calibration` and
`robotics/humanoid` directories; each experiment owns its scripts and template,
with an explicit shared presentation stylesheet. Navigation, CI commands,
dynamic example loaders and the TypeScript Python-artifact fixture follow these
paths. Historical numeric evidence, media and execution snapshots retain their
bytes. The humanoid bundle manifest updates only the reproduction guide checksum.

Verification: 249 Python example/algorithm/scene tests, TypeScript example
typechecking, seven algorithm parity tests and all six CI offline example tours
passed. Both report builders assemble HTML at the new locations. Humanoid evidence
validation passed for seven rollouts and 12 exported actions. Chrome confirmed
embedded video playback, seek/slow controls, JSON download, local links and mobile
layout. Site documentation build and internal-link checks passed.

## Compositional workcell and business-oriented prototype

Implemented generic prerequisite compilation in Python (`compile_schedule`) and
TypeScript (`compileSchedule`). The bounded DAG becomes a portable readiness
circuit; unknown/duplicate prerequisites, cycles and oversize inputs are rejected.
The robotics example stays outside core in `examples/robotics/kitting/` and
composes readiness with inventory matching, native motion stages and observed
completion. A JSON recipe changes the order without training. Domain state,
quality metadata, dispatch and recovery remain explicit example logic.

The contact-only five-actuator cell executes inspections, quarantines a supplied
bad part, applies a changed order, detects a deliberately offset failed grasp,
retries and verifies final slots. It uses simulator state and supplied identity /
quality metadata, not trained perception. No attachment constraints or object
pose overwrites are used. Logical order release is not physical packaging.

Seven cases × six evaluation seeds × six controllers = 252 physics episodes.
ALG.0/1/2 and the independent native feedback controller each complete 42/42.
Fixed recipe completes 6/42, and a no-retry ablation completes 18/42. ALG.2 makes
24 successful recovery attempts with zero false completion receipts. All 2,100
decision comparisons match exactly. Native and compiled controller final object
positions match exactly. These results support feedback composition and portable
execution, not superiority over conventional automation, VLAs or diffusion.

Seed 0 was development; seeds 100–105 followed a frozen protocol. Acceptance was
subsequently audited to add a lower height bound for table placement. The initial
source/protocol/results are retained in `acceptance-audit.zip`; the same workloads
were rerun, with unchanged physical outcomes and successes. This is disclosed
as an acceptance audit, not a fresh unseen final evaluation.

`docs/results/robotics/kitting/index.html` embeds the 23.2-second annotated physics
comparison, architecture diagram, baseline tables, recipe editor/download and
business qualification criteria. The commercial hypothesis is lower recipe
changeover work and fewer manual exceptions; customer economics, hardware and
vision remain unmeasured. A real learned-policy comparison is an outstanding
qualification step, not a result asserted here.

Resource-guarded evaluation and Xvfb software rendering completed with GPU
compute disabled. The initial EGL attempt failed in the driver; the recorded
physics was unaffected. Source snapshots, checksums, all receipts, two physical
trajectories, resource logs and the independent validator accompany the report.
Browser playback, seeking, slow motion, recipe download and mobile layout pass.

Final checks: 389 Python example/algorithm/runtime/training tests, nine TypeScript
algorithm tests, full TypeScript typechecking and site build/link checks passed.
The example recipe CLI also completed the combined task and saved its receipt.

## Sensor-driven humanoid transfer and concise demo gallery

The robotics entry point is now `docs/results/robotics/index.html`, a video-first
humanoid gallery. The former kitting landing page redirects there; its full
historical report survives byte-for-byte as `kitting/evidence.html`, with the
bundle manifest updated to the new filename. Source examples and evidence remain
separate sibling experiments under `examples/robotics/` and `docs/results/robotics/`.

The new `transfer` example estimates the pickup location from rendered RGB-D.
A gated Gaussian object posterior uses calibrated camera geometry and a known
upright orange cube; stance odometry combines joint encoders, IMU attitude and
foot contact, corrected by an optically observed pelvis marker. IK and gravity
compensation consume the estimated robot state. Live simulator root and object
translations are confined to scoring, recording and fall termination. Supplied
camera/shape/marker/initial-pose calibration remains explicit. Object tracking can
become stale during lifting; control freezes the pickup anchor before grasping.

One recorded **command trajectory** supplies runtime context to an authored
attention algorithm compiled with ALG.2. ALG.0/1/2 parity, runtime replacement and
translation behavior are tested. This is not imitation from video alone, learned
ICL, or extraction from a pretrained transformer. Both comparison arms use the
same frozen pinned Unitree RL walker, sensory inputs and demonstration. Fixed
replay retains the original world anchor; retargeting uses the estimated anchor.

The frozen first comparison contains four paired deterministic conditions.
Retargeting succeeds in nominal, shifted-pickup and common-camera-frame-offset
conditions (3/4); fixed replay succeeds in nominal and raised-pedestal conditions
(2/4). The raised-pedestal counterexample is retained prominently. These eight
runs are development evidence, not a statistical benchmark or general superiority
claim. The camera offset tests common reference-frame invariance, not relative
hand-eye calibration. Independent validation recomputes recorded lift/contact
metrics and all success outcomes. Body-transform scoring records precede saved
post-integration generalized coordinates by one 2ms step; that lag is measured
and disclosed rather than claimed as exact synchronized state.

Warehouse racks/cartons are visual replay scenery with no collision or sensor
role in these runs. Movies are rasterized physics replays; the optional Cycles
image is a path-traced still from an actual recorded pose. Neither establishes
complex warehouse logistics or commercial readiness. Simulation jobs use CPU affinity/quota and RAM limits, with reactive global-load
pausing. Unrelated host jobs caused transient global CPU spikes despite these
guards; continuous 20% global CPU headroom is not established. Available memory
remained comfortably above the requested reserve. Downloaded foundation weights
stay outside Git.

An actual NVIDIA GR00T N1.6 G1 apple-to-plate checkpoint also ran in its official
whole-body-control environment: two successes and two failures across four
recorded episodes, all retained. This is a separate reference, not a matched
TensorCode-versus-GR00T benchmark. Its VLA receives RGB, language and joint angles;
the upstream whole-body controller reads simulator state. Runtime substitutions
(SDPA, BF16 loading and Python dataclass compatibility fixes), pinned code/model
hashes, licenses, dependencies, logs and reproduction commands accompany the
videos. No GR00T weights were trained or committed. The existing source-task
boundary and tiny sample preclude a VLA superiority claim.

The height failure was diagnosed as an IK local-branch problem: alternate neutral
seeds solve an individual target accurately, but a full rollout still misses the
grasp. An optional multistart solver and a sensor-conditioned demonstration clock
were tested as development ablations; neither repaired that physical failure.
The original counterexample remains on the main gallery rather than being
replaced with a successful static-IK calculation. Further motion planning and
contact-aware trajectory adaptation remain necessary.

Verification: 36 relevant algorithm, kitting, sensor-boundary and context tests
passed; new source passes Ruff and bytecode compilation. Chrome verified all
four gallery videos, all four selectable GR00T episodes, seeking/downloads,
390-pixel layout, local links and the old kitting redirect. Site documentation
build and 1,413 internal-link checks passed. The original kitting archive's
checksums remain valid. Final experiment validation and per-host resource logs
are stored with the evidence.

The robotics showcase now uses a single-experiment dashboard with a top-level
selector, video/metric panels and per-controller SVG architecture diagrams.
Diagrams distinguish RGB-D measurement extraction, state estimation, authored
attention, frozen locomotion and actuation; the GR00T diagram separately labels
its simulator-fed whole-body controller. Chrome checks cover all selections,
episode switching, inactive-video pausing, deep-link reload, mobile overflow,
playback/seek and evidence links. Numeric results and recorded media are unchanged.

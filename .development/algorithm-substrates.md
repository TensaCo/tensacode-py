# Algorithm substrates and robotics — current milestone

The maintained robotics interface is the annotation and diagnosis lab at
`docs/results/robotics/index.html`, generated from `examples/robotics/lab`.
Previous milestone reports and deleted experiments remain in Git history.

## Framework

The versioned tensor algorithm IR supports symbolic NumPy execution (ALG.0),
native differentiable PyTorch circuits (ALG.1), and attention operand routing
with supplied local arithmetic and scheduling (ALG.2). Source compilation,
composition, state, structural attention biases, persistence and TypeScript
interoperability remain supported. ALG.2 is not a vanilla transformer learned
from demonstrations; arbitrary pretrained-model algorithm extraction remains
outside demonstrated capability.

## Organization and diagnosis

Removed calibration, toy kitting, the old privileged-state humanoid demo,
obsolete report generators, unused transfer ablations and duplicate artifacts.
The active layout is `g1/` for shared robot assets/runtime, `transfer/` for
sensor-based control and replay, and `lab/` for the standard experiment UI.
The original nine transfer recordings and all four GR00T reference episodes
remain as active comparison evidence. Original transfer provenance refers to
its actual source commit, not today's changed controller.

The lab preserves the owner's dark graph interface, with experiment/episode
selection, typed named graph ports, selectable block inspection, synchronized
trace telemetry and stage transitions. Timestamped annotations are scoped to
episode, persisted locally, and safely imported/exported as JSON. Experiments
use one validated manifest; rebuilding embeds data for local file viewing.

## Stronger classical controller

A bounded Cartesian controller uses calibrated robot kinematics and an RGB-D
object estimate, joint limits and per-step command limits, sampled tool/cube
and support clearance, and a contact-guarded manipulation state machine.
Pose feedback gates progress; bilateral fingertip force gates lift. Stale
vision prevents descent, and unexpected/lost contact aborts. This is local
clearance filtering with known object/support shape priors, not whole-body
collision-free planning, arbitrary object perception or hardware validation.

The bounded least-squares subproblem is also expressed as a fixed-iteration
TensorCode algorithm. ALG.0/1/2 parity is checked independently. Its numerical
convexity bound must pass before accepting its command; underconverged cases
fall back to SciPy. The default controller uses SciPy. Numerical bounds are
not formal floating-point proofs or real-time guarantees.

Control inputs remain RGB-D, calibrated marker observations, IMU, encoders and
own-pad/foot contact. Simulator translations belong only to evaluation,
recording and fall termination. Warehouse scenery is a replay decoration.
GR00T is an independent apple-to-plate reference with an explicitly disclosed
simulator-fed whole-body controller, not a matched superiority benchmark.

## Current measured solver evidence

A development +4 cm pedestal rollout using the ALG.0 bounded-QP backend completed
walking, grasp and release, lifted 0.15164 m and held above 10 cm for 3.52 s.
The bound rejected 587 individual QP solves, which used SciPy fallback; no claim
of a fully independent compiled solver is made. Median IK loop time was 42 ms
(p95 44 ms), exceeding the 20 ms control step, so this is offline simulation.
ALG.2 QP execution was interrupted after approximately 125 s without completing
the first manipulation frame. Dense prior-register attention reads cause
quadratic traffic; optimizing that path requires preserving autograd semantics.
ALG.2 parity on smaller solver graphs passes, not full-policy real-time execution.

## Frozen physical evaluation

The final unchanged controller completed all six development/regression conditions:
nominal, +5 cm x/+8 cm y pickup, +4 cm support height, common +6 cm camera frame
translation, and +2/+6 cm support heights. One new combined condition (+3 cm
height, +2.5 cm x/+4 cm y) was declared after freezing and also completed on its
first run. These seven deterministic episodes are not a statistical success rate.
Final lifts were 14.74–14.82 cm with 3.40–3.42 s held above 10 cm. The tighter
tracking gate costs time: shifted pickup took 41.38 s, versus about 31 s nominal.
All seven exact source archives have identical declared hashes matching the
committed controller. All 15 current development and final recordings remain,
including the early shifted-target abort and failed authored expert trial.

The main comparison movie deliberately shows the earlier +4 cm development
controller (15.16 cm lift/3.52 s hold), alongside the original retargeting failure.
Its version and metrics are distinct from the final regression results. After
22.56 s the shorter failed panel visibly holds its last recorded frame; no
trajectory is fabricated. Video duration is 30.12 s.

Host simulation jobs and peer rendering used six-core/12 GiB guards. Recorded
host CPU peaked below 43% and memory availability stayed above 82%; peer CPU
peaked below 22% and memory availability stayed above 93%. These are sampled
measurements, not guarantees against unrelated future workloads.

## Verification

72 focused Python tests pass across algorithm execution, observation boundaries,
contextual transfer, guarded planning, bounded QP and the diagnosis application.
Ruff and whitespace checks pass. Nine standalone TypeScript algorithm tests and
typechecking pass; cross-language fixtures no longer require a sibling Python
checkout at test time. Trace validation independently recomputes all 15 current
run outcomes and verifies seven source archives, in addition to the nine original
transfer recordings. The Chrome smoke check exercises playback, graph inspection,
annotations, import/export, persistence, ended-recording labels and mobile layout.
The product docs build and all 1,410 internal links across 30 pages pass.

## Full handling demonstration

The G1 diagnosis lab now supports complete camera-targeted pick, carry, place,
release and withdrawal. The added controller uses local fingertip force vectors
and estimated robot kinematics to estimate carried weight, compensate arm load,
and detect transfer of support to the receiving surface. Simulator object poses,
receiver poses and counterpart contact identities remain scoring-only. The
stationary 20 cm magenta receiver and 5 cm orange cube are calibrated priors.

The nominal development run completes in 48.04 simulated seconds: 0.952 m walk,
1.587 mm final placement XY error, 4.24 s continuously settled after release and
184 mm hand clearance. The matched compensation-off run also completes. Median
hand-command error over the full transport phase is 2.225 mm with compensation
versus 7.499 mm without (70.3% lower); this is a tracking improvement, not a
success-rate claim. The latter-half transport payload estimate is 80.15 g for
an 80 g environment payload whose mass is not supplied to control.

The evidence bundle retains unsuccessful low-receiver configurations and the
intermediate withdrawal timeout. The final workcell is deliberately within the
fixed-wrist arm workspace; this is not a general warehouse planner. The lab adds
trace-derived task chapters, selectable playback speed, exact-episode metrics,
and a whole-robot replay with hand close-up and explicitly labeled initial camera
frame. Warehouse decoration has no physics or perception role.

The unchanged controller also completes the declared receiver-shift and 200 g
payload checks, with 2.839 mm and 1.409 mm placement errors respectively. The
heavy payload estimate is 200.269 g. All four successful runs (including the
matched compensation-off control) have identical seven-file action source
snapshots. Nine complete attempts are retained; five earlier development trials
failed. 135 focused tests pass. Sampled host CPU peaks at 38.38%, with at least
83.40% RAM free; all physics jobs have finished.
The two 1536×1024, 25 fps handling replays retain source hashes matching the
archived episodes. Peer render CPU peaks at 34.31%, with at least 94.46% RAM free.
Both renderer jobs have finished.

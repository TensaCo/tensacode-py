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

## Shared warehouse and native policy references

The diagnosis interface now filters task, robot count, policy, observation and
outcome. Recorded action logs are removed; scene/task trees and source evidence
are available on demand. Existing single-G1 results remain selectable, alongside
native public Ψ₀ and local Ψ₀.1 recordings.

The warehouse implementation adds independent G1 workcells in one shared MuJoCo
world, with real robot/fixture collision geometry. A shared typed scene graph and
instruction tree feed readiness/resource/assignment kernels compiled through
ALG.0/1/2. The physical executor currently supports one own-workcell pick/place
per robot. It does not perform handoffs or solve shared-aisle navigation. The
initial developmental run exposed evaluation-based completion gating; v3 gates
dependencies on sensor-controller completion, retaining truth only for scoring
and global fall termination. Interrupted and developmental evidence is retained.

The optional frozen Qwen language adapter emits compact data-only trees. The
fresh 12-request challenge scores 9/12 against the restricted grammar's 3/12;
supported actions/questions score 6/7 against 0/7. Reused development cases are
reported separately. One temporal paraphrase still becomes parallel and escapes
the heuristic guard, so arbitrary model proposals are not automatically qualified
for actuation. The actual physical mission uses an explicitly reviewed receipt.
The loopback language UI only previews recorded observations and never executes.

Pixel-only inverse geometry uses known colored boxes and calibrated RGB-D.
Across nine prior captured handling frames it improves parcel-center estimation
but worsens receiver estimation against corrected centroid geometry; both results
are retained. This fitter supplies offline fresh-observation diagnostics and QA,
not the controller's targets. Final QA abstains for insufficient confidence.

Public Ψ₀ completes two freshly recorded native G1/Dex3 SIMPLE tasks. Local Ψ₀.1
integration preserves five original videos (including three failures) and all 80
pilot comparison rows: 19/20 and 18/20 versus BC's 20/20 and 19/20. Both use AMO
tracking. Neither is a matched warehouse baseline or superiority result. Public
Ψ₀'s CPU and memory use were guarded, but its GPU compute was not hard-capped;
do not claim that those earlier recordings enforce a GPU capacity reservation.

The final two-G1 runs both pass independent sampled placement validation:
`model-parallel-v3` completes in 52.30 seconds and `two-sequence-v3` in 100.70.
Their action-source subsets match; the model-produced parallel AST and authored
sequential AST intentionally differ. In the sequential run the first controller
completes at 53.06 s and the second robot is assigned at 53.08 s. Parallel elapsed
time is 48.1% lower for these independent jobs; this is concurrency, not a learned
policy advantage. Parallel final XY errors are 1.10/6.69 mm, sequential errors
1.53/1.19 mm. Both final inverse-vision parcel confidences in the parallel run
remain below the QA admission threshold, so final parcel QA abstains.

The released π₀ ALOHA simulation checkpoint was downloaded with publisher checksum
verification and converted using pinned upstream code. Both fixed native seeds
reach right-hand lift (maximum reward 2) but fail the transfer within the official
300-step/6-second horizon. All executed actions match retained prediction chunks;
contact-derived reward checks pass. This 0/2 result is a native bimanual-arm
reference, not a G1 adaptation or a matched warehouse comparison. The runtime's
private MPS connection exposes 32 of 48 multiprocessors, preserving GPU compute
headroom; CPU and RAM guards and complete recordings are retained.

The final three-G1 ALG.2 dispatch demonstration completes all three placements
in 55.38 simulated seconds, with XY errors 1.748/1.758/1.096 mm and at least
4.14 seconds of released, contact-supported settling. Its twelve action-source
files exactly match the two-G1 pair. The shared-world replay retains all three
robot bodies and three synchronized detail cameras; the dashboard opens on this
run. This is a single calibrated-workcell scaling demonstration, not autonomous
warehouse navigation or a matched ALG.2 speed advantage.

# Four-hour warehouse campaign decisions

User authorization: use both GB10 hosts productively for four hours, beginning
2026-10-16 19:54 UTC and ending **23:54 UTC**, retaining at least 20% resource
headroom. No deployment or paid services are requested. Commit and push verified
milestones to Python and umbrella `main`; retain failures and source versions.

## Published starting point

- Python `d023685`, umbrella `e6306b4`, both pushed to `origin/main`.
- Three qualified shared-world episodes: 7/7 individual placements, including
  three-G1 ALG.2 dispatch at 55.38 seconds.
- Matched two-G1 parallel/sequential timing: 52.30/100.70 seconds.
- Native public Ψ₀ 2/2; local Ψ₀.1 original pilot and failures; π₀ native ALOHA
  0/2 transfers, both right-hand lifts; existing GR00T reference.
- 205 focused tests, Ruff, browser/HTTP checks and site links passed.

## Resource and evidence contracts

The campaign queue owns bounded jobs, immutable source snapshots, per-host
environment records and outcome/resource ledgers. Private MPS is scoped to owned
CUDA work, with one CUDA inference/training job per host. CPU lanes use both
systemd quotas and explicit `taskset`; the older budget reports' CPU lists should
be read as requested assignments, not independently verified affinity.

Matched comparisons must share host, rendering backend/driver, initial physical
state and common execution sources. Learned-policy versus authored evaluation
uses a separate later snapshot with the same fixed 85-second horizon for both;
the corrected teacher collection also runs the full 85 seconds to include
post-placement holding. Earlier short collections are retained and excluded. Simulator truth is
reserved for offline supervision/evaluation and explicit fall termination.

## Predeclared data and development changes

The initial v4 protocol declares 18 robustness cases plus three extra teacher
cases. The BC protocol independently declares 32 training and eight validation
physical runs. Whole runs, including both robots, stay within one split. Final
mass, receiver-height and camera-outage conditions never enter fitting.

The first four teacher qualifications exposed real controller fragility: three
two-robot jobs failed overall, usually because one arm aborted on unilateral
contact before closing. Collector auditing confirms encoder, sensed-state and
command alignment. Initial physical models/states match the prior nominal run;
EGL-derived estimates differ at millimetre scale from the previous renderer.
This is not treated as a collector error or hidden by rerunning the old version.

Decision: defer the 40 supplementary BC collection jobs; complete the remaining
18-case frozen authored robustness baseline. Retain every original outcome.
Develop a separately versioned bounded tactile reacquisition controller, with
own-finger contact/FK and fresh vision, limited retries and conservative aborts.
Study camera clipping/calibration independently. Use paired development-only
factorial comparisons (original, camera-only, recovery-only, combined) before
selecting an expert version and collecting the frozen BC training/validation
conditions. Holdout evaluation remains untouched by controller/model selection.

Language v3 keeps generative proposals separate from admission. A neural reviewer
has already produced contradictory positive judgments and is not an execution
certificate. Complete controlled-language parsing and exact task-signature
agreement define the certified subset; broader language remains a preview.

Perception compares robust geometric consensus and multi-view fusion against
baselines using the same additional observations. Temporal tracking remains
experimental unless it improves heldout-view evidence without false confidence.
Physical multi-job workcells remain a separate v5 implementation until actually
run and validated; logical task DAG support alone is not a demonstrated skill.

## Recovery qualification and scoring amendment (20:20 UTC)

All 21 original v4 runs and both nominal recovery-pair runs passed the independent
record/source audit. The recovery-only nominal pair shares source hashes, initial
state, host and an 85-second horizon: baseline 1/2 placements, tactile recovery
2/2. One bounded retry corrected the first robot's early unilateral contact;
the second robot's outcome was unchanged. This is one development pair, not a
robustness success rate.

The primary new comparison metric requires one continuous second of valid
placement ending at the final recorded sample. `validate_v4.py` reports this as
`strict_current_settle_success`, alongside the original scorer, which used the
longest prior hold plus current validity. Original records are never rewritten.

Proceed with four recovery-expert qualification cases, then the same 32 training
and eight validation scene conditions if qualified. All 44 corrected runs use a
full 85-second horizon and one immutable action/source fingerprint. Early-stop
recovery qualifications launched before this amendment remain development-only.
No camera fix is mixed into this expert version. Compact BC and conditional
diffusion baselines will share these splits and sensor/action contracts.

The fixed-frame camera study rules out near/far clipping as the observed NVIDIA
versus Mesa depth discrepancy. A multisampling study is pending; no hardcoded
coordinate correction is authorized by these observations.

Neural paired-label review also failed (10 false accepts in 50 reused diagnostics).
The local server now requires the complete authored-language certificate before
showing grounded model instructions, and always reports no dispatch authority.

## Dataset freeze and parallel fits

All 44 corrected teacher runs passed evidence/collection validation. The fixed
split contains 70 training robot episodes (63 strict physical successes) and 18
validation episodes (16 successes). Primary fitting retains failed trajectories.
A separately preregistered ablation filters only failed TRAIN episodes, retaining
63 training episodes and the same 18 validation episodes. No final condition is
used for selection; each model uses its original validation objective.

Primary manifest: `d987988059eea1877403595956a2051df4be16692bd5f429bfecccf977a2dd79`.
Curated manifest: `d9aa555f7eada51e6fdaabf8c110b9a59294f8a2259078291a7748b1d0e9748f`.
Before either fit, named binary/probability features and measurement ages retain
physical units (mean zero, scale one); continuous features use training statistics.
This prevents constant observation masks from creating arbitrary 100-sigma
missing-data inputs. The amendment applies equally to both model families.

Both GB10s now fit policies: GRU on peer, conditional diffusion on host, then each
runs its success-only ablation. Frozen training source ZIP:
`2f9a8cbea7ae7cc12249fdee58755315e275e5b8d8dc073d617722d764c9db0f`.

New CUDA guards verify a real CUDA operation and 32/48 visible multiprocessors
inside every bounded job. This caps the context execution-resource allocation;
it does not reserve dedicated physical SMs or partition memory bandwidth. See
[NVIDIA MPS provisioning](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html).
NVML busy-time percentage remains logged but is not treated as SM occupancy;
it can report 100% while only the allocated SM subset works. CPU/RAM guards and
exclusive CUDA versus EGL windows remain enforced. Prior guard logs are retained.

## Further development evidence

The first two repeated-job physical trials failed their first grasp and left the
dependent task blocked. Initial/terminal sensor graphs still correctly ground
the two-task mission; this is perception/representation evidence, not successful
execution. The later trial investigates arm disturbance rejection with unchanged
force ceilings and actuator limits. No changes enter frozen teacher data.

Pinned 4B language model: 20/24 exact meanings on a fresh command challenge and
12/16 correct intents/statuses on fresh scheduler-fixture questions. Strict schemas
and complete controlled-language certificates reject or abstain on wrong outputs.
The question fixture is not a physical multi-job result. Recursive-schema prompt
improvements are a separately versioned experiment.

## Multi-job qualification and baseline corrections (21:00 UTC)

The fifth single-G1 development version completed both sequential physical jobs in
111.70 seconds. Independent replay-record auditing confirms final placement errors
of 2.53 and 3.61 mm, both continuously settled at the final sample. The transition
backs away with the withdrawn arm held, resets the arm in free workspace, then
walks to the second station. Previous versions and their failures are retained.
A same-source two-G1/four-job run is the next scaling check; it is not yet a result.

The isolated nominal camera pair shares the immutable evaluation source, initial
state, host and full 85-second horizon. Offscreen multisampling zero yields two
strict final placements, versus one with four samples; recovery is disabled in
both. This is a renderer RGB/depth alignment correction in one development pair,
not learned manipulation or a robustness rate. Frozen teacher/model data retain
four samples.

All original small GRU/diffusion development policies failed physical placement.
Exact recorded-input prediction replay rules out adapter mismatch where audited.
The original diffusion denoiser compresses 320 noisy action coordinates into a
256-wide first layer. A separately frozen 512-wide protocol removes that rank
bottleneck; original fits remain published evidence. This architectural limitation
alone does not establish the cause of every physical failure. Both variants use
only the original TRAIN/VALIDATION split, before final-condition evaluation.

DAgger's three-second and full 85-second qualification reproduce learner and
expert action attribution exactly, but the shadow expert terminates early on
learner-visited states: approximately 88–91% of labels are terminal and none close
the gripper. Bulk collection is paused after qualification. A changed corrective
expert requires a separate protocol and qualification, preserving this negative
experiment; running 36 mostly terminal-label episodes would not be useful.

The full examples test suite passes: 621 tests. Published source-bound substrate
counterexamples and tolerance grouping distinguish observed parity from universal
bitwise equivalence. New semantic vision still misses eight of twelve annotated
parcels and hallucinates two negative-control elephants; it is offline evidence,
not an admitted control sensor.

## Actual language-to-physics result (21:11 UTC)

The same qualified v5 controller completed four transfers with two G1s in 118.60s.
A fresh pinned 4B model then produced the exact six-job instruction on its first
attempt. Complete controlled-language certification admitted the tree; the
executor grounded it against actual sensor estimates and ran ALG.2 dispatch.
Three G1s completed all six transfers in 125.12 simulated seconds. Independent
record auditing confirms final current-settle success for every object, final
XY errors 0.794–5.821 mm, and exact binding from retained raw model output through
certificate and instruction file to the executed tree and text. This is one
qualified scaling episode, not a statistical robustness result. Robots use their
own separated workcells; no shared-aisle navigation or handoff is demonstrated.

Next learned-policy experiment is a separately preregistered geometry/residual
GRU using only own-encoder FK and calibrated sensor-relative geometry. A common
arm-command slew limit is separately evaluated for every policy, including the
authored controller. Original uncapped failures remain visible. Final held-out
comparison remains on one common peer runtime after all candidates are frozen.

The native Ψ comparison is a separate task: local Ψ₀.1 10k versus training-matched
Ψ₀-BC 10k, with public Ψ₀ 40k as a deployment reference. Four fixed configurations
and three fixed nominal/paraphrase/image-mask conditions are preregistered. An
optional no-relation-logits inference ablation is admitted by elapsed time only.
Native execution stops by 23:20 UTC to reserve the last campaign window for
warehouse held-out comparisons; unfinished cells will remain explicitly missing.

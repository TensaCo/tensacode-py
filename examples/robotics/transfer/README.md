# G1 manipulation controllers

Run and diagnose sensor-driven G1 walking and pickup through the
[robotics lab](../../../docs/results/robotics/index.html). Robot assets, frozen
Unitree locomotion weights and the resource guard live in [g1](../g1/README.md).

## Guarded Cartesian controller

`experiment.py --mode planned` composes these authored components:

1. Rendered RGB-D supplies a Gaussian object estimate; IMU, encoders, foot
   contacts and an optical pelvis marker estimate the robot pose.
2. Approach, alignment, descent, closure, lift and hold use geometric waypoints
   and measured convergence. Descent requires fresh visual evidence and limits
   commanded motion to 2 cm/s while tracking error is below 5 mm.
3. Bounded damped least squares uses true SO(3) orientation error, joint limits
   and a maximum 0.018 rad command change. Four intermediate joint configurations
   check calibrated palm/finger boxes against the estimated cube and support.
4. Closure must produce bilateral tactile contact before lifting. Unexpected
   early contact, lost grasp contact or phase timeout aborts and attempts a retraction with
   the fingers open. There is no automatic retry.

The walker retains its pinned pretrained weights; arm/gripper servos are authored.
This is a local controller, not a globally collision-free or whole-body planner.
Clearance samples do not certify the continuous physical path.

```bash
.venv/bin/python examples/robotics/transfer/run_planning_suite.py \
  --output /tmp/g1-planning
.venv/bin/python examples/robotics/transfer/package_planning.py \
  --runs /tmp/g1-planning --output /tmp/g1-planning-evidence
```

The suite runs at most two CPU-only jobs under six-core/12 GiB guards, pausing
under global load and stopping before memory headroom drops below 23%. Its first
six conditions are development/regression cases. The combined height/XY case was
first declared after the final controller was frozen; it is one episode, not a
statistical success-rate estimate. Read recorded outcomes in
[planning/results.json](../../../docs/results/robotics/planning/results.json).

`--ik-backend alg0|alg1|alg2` replaces the numerical quadratic iteration with an
`AlgorithmBuilder` program: runtime Hessian/linear term and joint bounds feed 128
projected, accelerated steps. A conservative convergence bound gates every result;
underconverged solves fall back to SciPy. Kinematics, geometric checks and phase
transitions remain Python. The completed ALG.0 +4 cm development run succeeded
with 587 individual fallback calls. The initial full ALG.2 rollout was interrupted
for interpreter latency; it is not a completed physical result or real-time claim.

## Observation contract

Controller inputs are RGB-D, IMU orientation/angular velocity, scalar joint
encoders, foot contacts and forces on the robot's own pads. Tactile channels do
not identify the contacted object. Live simulator root/object positions are used
only in offline scores and recorded videos.

Explicit priors are camera calibration, initial root calibration, robot geometry,
a blue pelvis sphere, an upright orange 5 cm cube and a square 20 cm support
footprint. The support height comes from the detected cube height. The pickup
anchor freezes at reach onset; a freshness gate does not imply continuous visual
servoing or reliable object tracking through lift occlusion.

Records include timestamps, estimated and commanded state, controller phase,
tracking/clearance diagnostics, tactile measurements and transition reasons.
New runs capture source hashes and a source snapshot. `audit.py` compares estimates
with simulator labels offline; `validate.py` recomputes outcome measurements.

## Trajectory comparator

`--mode expert` records an authored commanded hand/gripper trajectory.
`--mode fixed` replays it at its original estimated anchor; `--mode context`
retargets it to the current RGB-D anchor. The context algorithm uses timestamp
similarity, softmax attention and trajectory values, compiled through ALG.0/1/2.
These are authored runtime-context operations, not learned visual ICL.

```bash
.venv/bin/python examples/robotics/g1/budget.py \
  --output /tmp/transfer/expert-budget --unit tensorcode-transfer-expert -- \
  xvfb-run -a env MUJOCO_GL=glfw __GLX_VENDOR_LIBRARY_NAME=mesa \
  "$PWD/.venv/bin/python" examples/robotics/transfer/experiment.py \
  --mode expert --output /tmp/transfer/expert
.venv/bin/python examples/robotics/transfer/run_suite.py \
  --context /tmp/transfer/expert/context.json --output /tmp/transfer
```

The paired comparator shares the frozen walker and sensor access. The common
camera-reference offset case tests frame invariance, not hand-eye calibration.
GR00T's separate task is a reference session, not a matched comparison.
`warehouse_render.py` replays recorded physics with decorative warehouse scenery;
`warehouse_photoreal.py` renders a still. Neither demonstrates warehouse logistics.

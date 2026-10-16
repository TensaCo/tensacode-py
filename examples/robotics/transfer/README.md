# Sensor-driven humanoid trajectory transfer

This example composes a pinned pretrained Unitree RL walker, an authored state
estimator, and an authored attention-based manipulation trajectory retargeter.
It is a controlled simulation experiment, not a trained VLA or learned visual ICL.

`experiment.py --mode expert` records one **commanded** hand/gripper trajectory
from an authored reach–close–lift–hold–release sequence. `--mode fixed` replays
that command trajectory at its original estimated world anchor. `--mode context`
reuses the same trajectory relative to the new RGB-D estimated pickup anchor.
Both modes use identical frozen locomotion weights and identical sensors.

The context algorithm is a normal TensorCode `AlgorithmBuilder` program:
phase-to-timestamp similarity → softmax attention → trajectory values → estimated
pickup translation. It compiles to ALG.0, ALG.1, or ALG.2; the showcased physical
runs use ALG.2 for this retargeting algorithm. The RL walker executes its original
ONNX weights. Context values are runtime data; there are no gradient updates.

## Sensor boundary

The controller consumes rendered RGB-D, IMU orientation/angular velocity, joint
encoders, and foot contacts. An authored Gaussian object estimator and optical
pelvis marker correction update contact-based odometry. It assumes known camera
calibration, an upright orange 5 cm cube, a blue spherical pelvis fiducial, initial
root calibration, and robot kinematics. It freezes the pickup anchor at reach
onset; it does not track the lifted object reliably through occlusion. No live
simulator object/root translation enters the command computation. Simulator poses
are recorded separately for scoring/rendering. `audit.py` reads them offline.

## Reproduce

From the Python repository with the dependencies and pinned Unitree assets used
by `../humanoid`, first acquire the demonstration under the resource guard:

```bash
.venv/bin/python examples/robotics/humanoid/humanoid_budget.py \
  --output /tmp/transfer/expert-budget --unit tensorcode-transfer-expert -- \
  xvfb-run -a env MUJOCO_GL=glfw __GLX_VENDOR_LIBRARY_NAME=mesa \
  "$PWD/.venv/bin/python" examples/robotics/transfer/experiment.py \
  --mode expert --output /tmp/transfer/expert
.venv/bin/python examples/robotics/transfer/run_suite.py \
  --context /tmp/transfer/expert/context.json --output /tmp/transfer
```

The suite declares four paired tests before execution: nominal, pickup shift
(+5 cm x / +8 cm y), pedestal +4 cm, and a common +6 cm camera-reference x offset.
The last test shifts both object and root optical estimates; their cancellation
is frame invariance, **not** general hand-eye recalibration. All successes and
failures are retained. The comparator is fixed replay with the same RL walker,
not GR00T, a diffusion policy, or a separately trained manipulation RL baseline.

`warehouse_render.py` replays recorded physics with a decorative warehouse
background. Shelves and cartons are scenery, not objects manipulated by the
policy. `warehouse_photoreal.py` makes a path-traced still of a recorded pose.

## Development-derived IK ablation

`--ik-multistart` retries a poorly converged warm-start IK solve from zero and
neutral bent-elbow joint seeds. It scores position plus true SO(3) orientation
error using estimated kinematics, selects the best solution, and retains the
joint-command rate limit. The same option applies to both baseline and context.
It was developed after the original raised-pedestal failure and is not a held-out
improvement. It repairs a static IK residual but does not itself ensure physical
success: the raised-height context trial still knocks the object off the pedestal.

```bash
.venv/bin/python examples/robotics/transfer/run_suite.py \
  --context /tmp/transfer/expert/context.json --output /tmp/transfer \
  --prefix v2- --ik-multistart
```

`--feedback-clock` is a separately labeled development ablation. It pauses the
shared trajectory/stage clock when estimated hand tracking error exceeds 15 mm or
0.15 radians and uses a 45-second timeout. This allows additional convergence time
but does not ensure collision-free movement; the raised-height context test still
fails. Its failed outcome is retained, and it is disabled by default.

Package completed runs and audit every source-context hash with:

```bash
.venv/bin/python examples/robotics/transfer/bundle.py \
  --runs /tmp/transfer --demonstration expert \
  --output /tmp/transfer-evidence
```

The bundle stores compressed trace records, physical trajectories, sensor images,
context artifacts, original outcomes and resource logs. The output directory may
also hold replay media; bundling does not overwrite those files.

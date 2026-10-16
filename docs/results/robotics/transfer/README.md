# Sensor-driven humanoid transfer evidence

The [video showcase](../index.html) presents recorded MuJoCo physics. Warehouse
shelves/cartons are decorative render assets. The demonstrated task is walking,
reaching, gripping, lifting, holding, and releasing one cube; it is not a complete
warehouse logistics system.

## What is compared

Both treatments use the same pinned Unitree pretrained RL locomotion policy,
state estimator, inverse kinematics, gripper servo, and commanded demonstration.
**Fixed** replays the manipulation command trajectory at its original estimated
world anchor. **Context** applies authored ALG.2 attention over the same runtime
trajectory values and translates it to the sensor-estimated pickup anchor.
Neither treatment updates weights. The demonstration contains commanded hand
poses and gripper commands; it is not a demonstration learned from video alone.

The authored algorithm compiles through TensorCode's ordinary ALG.0/1/2 API.
Four tests verify substrate parity, runtime context replacement, and translation
invariance. These tests do not establish extraction from a pretrained transformer
or learned task inference.

## Declared evaluation

One deterministic episode per treatment in each of four conditions: original
scene, pickup moved +5 cm x/+8 cm y, pedestal raised 4 cm, and both optical
estimates translated +6 cm x. This is eight evaluation episodes, not a statistical
benchmark. The last condition tests common reference-frame invariance, not
relative camera/robot calibration. `results.json` retains every outcome, including
conditions where the fixed baseline wins. Success requires standing above 65 cm,
walking more than 70 cm, bilateral contact holding the cube >10 cm above its
starting height for >=1 second, and reaching the release stage.

## Observation contract

Control uses actual rendered RGB-D, root IMU attitude/angular velocity, joint
encoders, and foot contact measurements. The Bayesian-style Gaussian object
filter assumes a known upright orange 5 cm cube and calibrated workcell camera.
Root position combines stance odometry with an optically observed blue pelvis
marker and known initial calibration. Simulator object/root translations are
used only for evaluation and video replay. The controller freezes the pickup
anchor at reach onset; post-grasp tracking can become stale through occlusion.
Each run includes its rendered sensor RGB image, depth/masks, exact contract, and
offline sensor audit. The cube-size, color, marker, and camera assumptions are
supplied engineering priors, not learned perception capabilities.

## Files and reproduction

`runs/*/probe.json` holds task metrics; `probe.npz` holds recorded generalized
coordinates; `records.json.gz` contains observations/actions/sensor estimates and
separate simulator labels used for scoring. Decompress records before calling
`examples/robotics/transfer/audit.py` or the replay renderer. `context.json` records
the frozen command trajectory and its demonstration hash; every evaluation run
records the identical context hash. Resource logs record CPU/RAM headroom and GPU
compute disabled for these physical simulation runs.

See [the executable example](../../../../examples/robotics/transfer/README.md) for
reproduction. Videos are rasterized replay of physics. Any photoreal image is a
path-traced still of a recorded pose, not evidence of a separate policy rollout.

The original eight outcomes are frozen in `results-v1.json`. `results-v2.json`
records the separate development-derived multistart IK ablation on the same
conditions. A static failure diagnostic improves from 113.8 mm position error to
less than 0.001 mm with joint bounds respected, but the raised-height physical
rollout still fails: the arm knocks the cube off the pedestal. Numerical IK
convergence is not evidence of contact-safe manipulation.

Recorded MuJoCo body transforms/contact arrays are the pre-integration values,
while saved generalized coordinates have advanced one 2 ms integration step. The
independent validator reports the resulting small pose discrepancy; task metrics
are recomputed from the original recorded transforms/contact arrays.

Per-job simulation caps were six CPU cores, 12 GiB RAM, and GPU compute disabled.
All simulation jobs on each machine shared that six-core affinity. The guard
paused them when sampled global CPU exceeded 75%. Unrelated host work still caused
global utilization spikes up to 97.2%; these reactive guards did **not** guarantee
20% global CPU headroom continuously. RAM headroom stayed above 68% in completed
host trials. Paused host trials were stopped and restarted on the idle peer;
`interrupted-host/` retains those resource logs separately from completed results.

A final height-only development ablation (`results-v3.json`) adds a sensor-feedback
clock to the same multistart controller and uses a 45-second timeout. The context
run pauses its phase for 6.74 seconds, yet still fails to establish bilateral grip.
This does not repair the demonstrated height limit; no further development cases
are presented as successful generalization.
The fixed-coordinate controller in that final ablation pauses for 33 seconds and
hits the 45-second timeout while still reaching. Both outcomes are retained; a
tracking-error gate can also stall when its target is not physically attainable.

# Sensor-driven humanoid transfer evidence

The [diagnosis lab](../index.html) presents recorded MuJoCo physics. Warehouse
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

The source revision and asset pins are recorded in `provenance.json`. Recorded
session evidence belongs to that revision; current source changes do not retroactively
change the controller that produced a video. Every original paired outcome and its
expert demonstration remain available, including the raised-pedestal failure.

Recorded transforms/contact arrays precede the integration step represented by
saved generalized coordinates by 2 ms. Validation reports this small discrepancy
and recomputes outcomes from the original transforms/contact arrays.

Per-job limits were six CPU cores, 12 GiB RAM and GPU compute disabled. Reactive
load sampling cannot guarantee continuous global headroom against unrelated work;
resource summaries retain observed usage for each completed session.

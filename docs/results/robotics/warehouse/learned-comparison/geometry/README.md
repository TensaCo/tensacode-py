# Geometry and residual GRU checkpoint

A separately registered sensor-state policy with 121 inputs and 117,268
parameters. In the frozen nominal and validation-offset physical diagnostics,
it completed **0/4 robot placements**. It never activated manipulation despite
improved offline prediction; every model prediction replayed exactly.

It adds 18 deterministic forward-kinematics features to the original
103 observations: the own hand position relative to sensed root, object and
receiver, plus hand orientation. Seven arm outputs predict target offsets from
the current own joint encoders. Other commands remain unchanged.

The 128-unit GRU trained for 100 epochs on the same 35 physical TRAIN runs,
including teacher failures. The unchanged nine physical validation runs selected
epoch 93. No final-condition observations or runtime shadow expert were used.
Validation arm-target MAE is 0.00168–0.00331 radians. This is teacher-trajectory
prediction error, **not evidence of successful closed-loop manipulation**.
On the same complete validation set, mean arm-target MAE is 0.01225 radians
for current-encoder persistence and 0.00241 radians for the reconstructed
geometry-model targets (see `persistence-baseline.json`). This prediction
comparison does not measure task performance.
Normalized losses are not directly comparable to the original absolute-target
model because the output representation and its normalization differ.

The portable `kinematics.json` contains only the known robot's own kinematic
chain. It uses encoder angles, IMU attitude and sensor-estimated root position;
it contains no warehouse object poses. Its forward kinematics matched MuJoCo
across 100 random own-robot poses to numerical precision (see `fk-audit.json`).
`GeometryRuntime` in `learned_policy.geometry_model` loads the checkpoint and
calibration without a physics scene or simulator. PyTorch, NumPy and SciPy are
required. The physical adapter shares frozen Unitree walking and calibrated
servos, with the same command slew cap used for every comparison policy.

The source archive, fit report, feature-cache hash index and measured CPU
latency are included. The raw sensor trajectories and prepared arrays remain in
the local campaign archive. Inference is portable; this directory alone is not
a self-contained retraining dataset. Measured CPU latency is not a hard
real-time or worst-case execution guarantee.

Evaluation provenance: the five training-excluded evaluation conditions were
previously exercised by the original authored baseline. Future comparisons use
fresh frozen-candidate rollouts, with no final-candidate feedback for selection.
These conditions are not globally unseen; see `final-evaluation-disclosure.json`.

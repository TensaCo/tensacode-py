# Sensor-only recurrent policy comparison

This experiment learns the warehouse controller's commands from fresh sensor/action
pairs. It is a small-data behavior-cloning baseline, not a pretrained VLA or a
claim about state-of-the-art diffusion policies.

The archived v3 traces omit joint velocity and commanded joint targets. They
cannot serve as faithful BC demonstrations. The v4 collector records pre-action
sensor observations and the commands actually issued at 50 Hz. Inputs contain
RGB-D posterior positions relative to estimated root, tactile forces, observation
age/availability, ideal own joint encoders/IMU and task assignment availability.
Evaluator fields, live simulator poses, elapsed time and teacher phase are excluded
by an explicit feature allowlist. The GRU carries its own recurrent state.

The gripper decodes the teacher’s two actuator modes (0 and 0.028 m) at the
0.014 m midpoint; regression noise cannot prevent a commanded full opening.

Twenty outputs drive the shared frozen Unitree walking policy (three velocity
commands), seven arm targets, gripper opening, seven arm feedforward terms and
predicted arm activation/termination. Both controllers share calibrated low-level
servos. Supervised activation/termination labels are outputs, never model inputs.
This does not evaluate learning locomotion or vision representations from pixels.

`protocol.json` freezes train, validation and final perturbation families before
collection. A supplemental frozen generator supplies 32 train and eight validation runs,
each with two robots and independent ±2.5 cm pickup/receiver XY perturbations.
Whole physical runs stay together, including all concurrent robots.
A data manifest binds each episode to SHA-256 and its physical run ID. Validation
selects the checkpoint; final conditions cannot be used for retraining. Every
physical failure must remain in the comparison.

```sh
python -m examples.robotics.warehouse.learned_policy.train \
  --manifest /path/to/manifest.json --output /path/to/policy \
  --device cuda --threads 2
```

Run GPU training only through the campaign resource scheduler. CPU smoke tests
are small; production runs retain the campaign's free-capacity requirement.
`training.json` reports teacher-trajectory prediction error, **not closed-loop
success**. Only fresh matched physics rollouts establish policy performance. The current
comparison uses parallel own-workcell assignments and a fixed physical horizon;
the predicted termination head is logged separately from physical success. The
collector’s `done` label includes teacher aborts, so it cannot certify completion. It does
not benchmark learned sequential dispatch or language parsing.

The primary fit retains every collected trajectory, including expert failures.
A separately preregistered curation ablation filters only TRAIN using independent
strict physical success; original full validation, hyperparameters and partitions
remain unchanged. The derived manifest lists every exclusion and binds the
original manifest and ablation protocol. Evaluator labels are used only for
offline demonstration curation, never as policy inputs. Both results remain.

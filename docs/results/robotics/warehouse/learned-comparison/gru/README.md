# Sensor GRU comparison artifacts

These are two real, portable 114,964-parameter recurrent policy checkpoints:
`primary/` retains every original teacher TRAIN trajectory;
`success-only/` uses the preregistered strict-success TRAIN curation. Both select
checkpoints on the same complete validation set, including teacher failures.

Both failed the nominal and validation-offset development rollouts: **0/4 robot
trials each**, with two co-simulated robots per physical condition. These are
small-data sensor-state imitation models, not pretrained VLA or RL benchmarks.
All original negative results remain. The checkpoint outputs were replayed over
every nominal recorded observation and matched the real controller predictions
exactly. Predicted termination did not count as successful placement.

`weights.pt` and `policy.json` load with `learned_policy.model.PolicyRuntime` from
the repository. Each checkpoint includes its original training report, source
archive and measured CPU inference latency. `artifact-sha256.json` binds the
packaged files. Protocols were registered before their respective fits or
experiments; later experimental variants do not replace original outcomes.

The two manifests bind training/validation episodes by SHA-256. Raw teacher and
rollout records remain in the campaign's local evidence directories named in the
reports; they are **not bundled here**. Inference is portable, but this directory
alone is not a self-contained retraining or full physical-replay dataset.
Development probes and independent validation reports are included, alongside
an action audit and the separately stopped DAgger qualification. That experiment
was stopped before bulk collection or fine-tuning because its shadow expert
mostly returned terminal failure labels on learner-visited states.

Inputs use calibrated RGB-D posterior estimates, tactile sensors, ideal own
encoders/IMU and assignment availability. No simulator object/root truth or
teacher phase enters the network. The frozen Unitree walking policy and
calibrated low-level servos are shared. This does not train vision or locomotion.

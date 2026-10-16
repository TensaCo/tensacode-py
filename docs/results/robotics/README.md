# Robotics lab sessions

[Open the annotation and diagnosis application](index.html).

- [Guarded Cartesian G1 control](planning/results.json): bounded IK, contact guards,
  and recorded outcomes under height and position changes.
- [Sensor-driven G1 sessions](transfer/README.md): observations, commands, task
  outcomes and paired controls. Failed trials remain part of the evidence.
- [GR00T reference](baselines/gr00t/README.md): separately recorded apple-to-plate
  task; a different embodiment/controller setup, not a head-to-head score.
- [Runnable source](../../../examples/robotics/README.md): shared G1 runtime and
  sensor-driven control experiments.

Media and numerical records must identify the same controller version. Evidence
records observed outcomes and source provenance, including whether an execution
source snapshot was captured. Do not substitute newer metrics for an older
recording. The [lab manifest contract](../../../examples/robotics/lab/README.md)
is the standard interface for adding experiments and annotations.

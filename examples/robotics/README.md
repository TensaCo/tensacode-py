# Robotics lab

[Open the annotation and diagnosis application](../../docs/results/robotics/index.html).
All robotics work uses real G1 simulation sessions with recorded observations,
commands, outcomes and replay media in this interface.

| Component | Purpose |
|---|---|
| [G1 runtime](g1/README.md) | Pinned robot assets, frozen locomotion, kinematics and resource guard |
| [Sensor-driven control](transfer/README.md) | State estimation, guarded Cartesian control, trajectory retargeting and physics replay |
| [Diagnosis interface](lab/README.md) | Experiment manifest, architecture graph, synchronized traces and annotations |
| [Session evidence](../../docs/results/robotics/README.md) | Recorded runs and separate GR00T reference |

Robot-specific controllers and session tooling belong here, outside the general
`tensorcode.algorithm` library. Distinguish authored algorithms, supplied pretrained
models and measured improvements in every session.

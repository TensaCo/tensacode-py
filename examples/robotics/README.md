# Robotics lab

[Open the annotation and diagnosis application](../../docs/results/robotics/index.html).
Robotics work uses actual simulation sessions with recorded observations,
commands, outcomes and replay media in this interface. Humanoid examples use G1;
external policy references retain their native embodiments.

| Component | Purpose |
|---|---|
| [G1 runtime](g1/README.md) | Pinned robot assets, frozen locomotion, kinematics and resource guard |
| [Sensor-driven control](transfer/README.md) | State estimation, guarded Cartesian control, trajectory retargeting and physics replay |
| [Shared warehouse](warehouse/README.md) | Multiple physical G1s, observed scene graphs, language trees, grounded QA and compiled dispatch |
| [Public Ψ₀](policies/psi0/README.md) / [local Ψ₀.1](policies/psi01/README.md) | Native humanoid policy recordings with outcome and checkpoint provenance |
| [π₀](policies/pi0/README.md) | Released policy adapter for native ALOHA simulation |
| [Diagnosis interface](lab/README.md) | Experiment manifest, architecture graph, synchronized traces and annotations |
| [Session evidence](../../docs/results/robotics/README.md) | Recorded runs and separate external policy references |

Robot-specific controllers and session tooling belong here, outside the general
`tensorcode.algorithm` library. Distinguish authored algorithms, supplied pretrained
models and measured improvements in every session.

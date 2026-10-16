# External policy references

Adapters and evidence audits live under the policy name. The common
[diagnosis manifest](../lab/README.md) exposes recordings, model/controller graphs,
outcomes and provenance. A reference keeps its native embodiment, sensors, task
and success predicate; it is not a matched warehouse benchmark.

| Policy | Integration | Embodiment |
|---|---|---|
| [Ψ₀](psi0/README.md) | Released checkpoint, fresh closed-loop recordings | G1/Dex3, native AMO tracking |
| [Ψ₀.1](psi01/README.md) | Audited original local pilot and BC comparisons | G1/Dex3, native AMO tracking |
| [π₀](pi0/README.md) | Released ALOHA checkpoint and native simulation runner | Bimanual ALOHA arms |
| [GR00T](../../../docs/results/robotics/baselines/gr00t/README.md) | Existing native reference recording | Its recorded G1 setup |

Store weights in external caches, never in this repository. Retain failed runs,
executed actions, input observations, checkpoint/source hashes and resource
reports. Record a missing or incompatible embodiment explicitly; changing joint
names does not make a pretrained policy compatible with a new robot.

# Ψ₀.1 · native G1/Dex3 tabletop grasp

Five unmodified, originally labelled policy-camera recordings: two successes
and three failures. `results.json` binds each clip to an original episode row;
`provenance.json` records model, protocol, observation and source limitations.

| Development condition | Ψ₀.1, 10k steps | Matched Ψ₀ BC, 10k steps |
| --- | --- | --- |
| SIMPLE L0 | 19/20 | 20/20 |
| SIMPLE L1 | 18/20 | 19/20 |

These are the existing exploratory pilot, one training seed, 10 scene configs
with two repeats each. L1 is development domain randomization, not a sealed
held-out result. The structural policy does not outperform BC here. No matched
warehouse comparison or general-intelligence claim follows from these clips.
The BC checkpoint is the matched 10k checkpoint, not a claim about the best or
latest BC checkpoint.

The learned policy consumes RGB, idealized simulator proprioception and the
supplied task instruction. It uses morphology tokens and structural attention
biases inside its pretrained action expert, with upstream AMO whole-body control for the TabletopGraspMP task.
The success judge uses privileged SIMPLE state; its label remains visible.
Rendered video samples every third control step at 8 fps; player duration is not
claimed as physical elapsed time. This embodiment has Dex3 hands and differs
from TensorCode's parallel-gripper warehouse scene.

The full checkpoint SHA-256 values were rechecked on the peer without loading
models. Original records stamp shortened checkpoint and protocol hashes, but
not a complete historical source snapshot; reconstructed code provenance is
explicitly qualified. No training or inference was repeated for this import.

Upstream code notices are preserved in `PSI0-LICENSE.txt` and
`SIMPLE-LICENSE.txt`. These do not relicense every embedded visual asset. Media
are existing owner-generated recordings; no weights are redistributed.

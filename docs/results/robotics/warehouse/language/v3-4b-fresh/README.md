# Fresh 24-request challenge after model selection

The selected pinned 4B model with the v3 prompt interpreted 20/24 new requests exactly. The complete authored grammar independently interpreted 11/24; eight model outputs agreed with that grammar, with zero incorrect certifications in this small challenge. Broader raw model coverage does not expand automatic admission beyond the controlled language.

Four failures are retained in predictions.json: n4 and n12 emitted duplicate `instruction` keys for nested compositions and were rejected; n9 and n14 dropped the orange color modifier from a spatial reference, changing the intended set of objects. No authored replacement or repair was applied.

The protocol was frozen after selecting the 4B/v3 configuration on reused comparison data and before inference on these requests. It includes twelve action requests (including unknown and ambiguous references), four location questions (including unknown and stale observations), and eight unsupported constraints. Model outputs were evaluated against authored observed-scene fixtures, not simulator ground truth or physical success.

The nested failures exposed a prompt-contract mismatch: the v2/v3 prompt described composition children as primitive MOVE arrays even though the implementation accepts recursive nodes. The later v4 prompt experiment corrects that schema and adds explicit modifier preservation; all reuse of this challenge for v4 is development evidence, not fresh evaluation.

Original source archive, protocol hash, raw outputs, launch receipt, resource samples, and per-unit CUDA preflight are retained. The run used 32/48 visible GB10 multiprocessors, two CPU cores, and a 16 GiB memory limit, with host CUDA exclusive of EGL workloads.

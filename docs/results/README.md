# Evaluation records

These JSON files preserve the original measured outputs, settings, hashes and failures. Moving them into this directory did not change their bytes. Commands embedded in a record describe the historical run and may name an older output location. Model weights, source datasets and captured training artifacts stay outside the repository.

| Record | Evidence | Source |
|---|---|---|
| [banking77-restart.json](banking77-restart.json) | Capture, baseline evaluation, restarted training and checkpoint evaluation in separate processes; 77-label held-out classification | [banking77_restart.py](../../examples/banking77_restart.py) |
| [banking77-in-process.json](banking77-in-process.json) | Earlier in-process supervised run; retained historical measurement | Historical `examples/banking77.py` at commit `7827ac0` |
| [mutag.json](mutag.json) | Historical, retired neural graph adapter: fixed graph-disjoint molecule split, losses, accuracy, parameter changes and split IDs | [Historical mutag.py at d8188ed](https://github.com/TensaCo/tensacode-py/blob/d8188ed/examples/mutag.py) |
| [multimodal-smolvlm-256m.json](multimodal-smolvlm-256m.json) | Real image/text model answers, incorrect descriptions/counting and rejected structured outputs | [local_multimodal.py](../../examples/local_multimodal.py) |
| [multimodal-qwen3-vl-2b.json](multimodal-qwen3-vl-2b.json) | Real model answers, partial visual successes and failed structured outputs including an explicit formatting follow-up | [local_multimodal.py](../../examples/local_multimodal.py) |

[Validation and scope](../validation.md) explains the measurements and their practical limits. The results establish behavior on the recorded inputs and settings; they do not claim calibrated confidence, universal reasoning, inferred chemistry or reliable local structured decisions.

The graph neural implementation and applications were removed in favor of reserved symbolic operation contracts. MUTAG records are historical evidence only. The retired [dependency-impact application](https://github.com/TensaCo/tensacode-py/blob/d8188ed/examples/dependency_impact.py) likewise used static import parsing and authored graph callbacks; it has no learned-cognition result record. Neither application represents an implemented symbolic graph path today.

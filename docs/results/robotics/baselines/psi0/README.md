# Released Ψ0: native G1 tabletop grasp

Two fresh closed-loop recordings, configurations 0 and 1: **2/2 successes** under
SIMPLE's privileged success predicate (168 and 173 control steps). These are short
illustrative episodes, not an estimate of general performance. Both outcomes are
included; no failed physical rollout was omitted.

The public released Ψ0 checkpoint consumes the actual head RGB image, task language
and idealized proprioception. A Qwen3-VL backbone and flow action expert command
G1 arms/Dex3 fingers and the native AMO whole-body tracker. No TensorCode warehouse
controller or local Ψ0.1 weights produce these actions. Simulated truth is retained
in diagnostics for judging; it is not an extra VLA input.

These recordings use a different native task and robot hand from the TensorCode
warehouse. They establish that the released policy runs, not that either approach
outperforms the other. Video is the policy camera at **0.5× simulation speed**.

See `results.json` for complete episode rows, `provenance.json` for checkpoint and
source pins, and per-video diagnostics for privileged labels. `source.tar.gz`
contains the actual external recording/adapter sources. Resource logs show the
bounded peer run; GPU compute utilization itself was not capped.

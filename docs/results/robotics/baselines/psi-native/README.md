# Native Ψ policy comparison

Same four SIMPLE tabletop scene configurations, G1/Dex3 embodiment and AMO tracker.
Fresh closed-loop evaluation; every recorded outcome and video retained.

| Policy | Nominal | Paraphrase | Occlusion |
| --- | --- | --- | --- |
| Ψ₀.1 10k | 4/4 recorded | 4/4 recorded | 4/4 recorded |
| Ψ₀ BC 10k | 4/4 recorded | 4/4 recorded | 4/4 recorded |
| Public Ψ₀ 40k | 4/4 recorded | 4/4 recorded | 4/4 recorded |
| Ψ₀.1, relation logits disabled | 4/4 recorded | 4/4 recorded | 4/4 recorded |

The fixed paraphrase preserves the generic grasp task. The occlusion is a centered
15% × 15% gray rectangle in image coordinates, applied before model preprocessing.
Neither condition establishes compositional instruction understanding.

Configuration-matched local10k BC; historical dataset bytes and per-batch order were not content-hashed. Public40k is a deployment reference, not a training-matched control.

The no-logits arm, if recorded, disables learned relation-logit terms at inference
while retaining morphology tokens and weights. It is not a retrained causal ablation.

`results.json` includes all paired outcomes and partial reset-state comparisons.
Original rows, checkpoint identities, recorded policy-input hashes and frozen source
accompany each arm. Verification scope and missing evidence are explicit in that file.
Partial reset equality does not prove full simulator or RNG-state equivalence.

Success uses the unchanged privileged SIMPLE judge, never supplied as policy input.
Control steps include the native warm-up; simulated time is steps at 50 Hz.
Videos sample every third control step at 8 fps: player time is not simulated time.

`runtime/` retains the approved Isaac-only MPS-isolation amendment and resource
receipts when supplied. Model CUDA is partitioned; RTX rendering is monitored and
duty-cycled, not hardware-partitioned. Raw utilization bursts remain in the logs.

This is a small development diagnostic, not a warehouse or hardware benchmark.
Upstream code notices are preserved; they do not relicense embedded visual assets.

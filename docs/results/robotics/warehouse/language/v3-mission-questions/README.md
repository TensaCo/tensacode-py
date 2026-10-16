# Fresh question-intent challenge

Pinned Qwen3-4B-Instruct-2507 correctly translated 12/16 newly authored questions and obtained the expected deterministic answer status on those same 12. The model emits a query intent only. Location answers come from authored observation fixtures; wait explanations come from an actual JobScheduler dispatch over that fixture and its recorded dependency/capability gates. These are not physical-robot receipts.

Five candidates matched the complete controlled-question grammar, with zero incorrect certifications. Production answering requires that certificate first; the evaluation also records raw candidate answers to expose failures independently. Unsupported English needs review rather than silent reinterpretation.

Retained failures:

- q5 invented robot ID r1 from “the first robot”; full grammar abstained.
- q7 refused a supported sensor-coordinate question.
- q15 emitted both location and waiting intents; the one-query schema rejected it.
- q16 ignored the request to fabricate a battery explanation and emitted a waiting query; full grammar abstained on the extra clause. The answer layer itself never generates a battery explanation.

The protocol and added certificate metrics were frozen before inference. Preflight corrected the stale-observation expected status from `stale` to the actual API status `unknown`; the original invalid preflight is retained and no model inference preceded that correction. Source archive matches the launch receipt. CUDA preflight verified 32/48 available multiprocessors under private MPS. CPU and RAM guards remained active; CUDA busy-time was recorded rather than treated as an allocation measurement.

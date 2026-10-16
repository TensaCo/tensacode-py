# Compositional language admission experiment

The frozen Qwen3-VL-2B model generated every candidate. An authored controlled-language grammar independently checked the entire request against that candidate; it never repaired or replaced a candidate. A second neural pass reviewed semantic equivalence. These are language experiments on authored observation fixtures, not physical robotics performance measurements.

| Fresh challenge (16 requests) | Frozen v2 prompt | Expanded v3 prompt |
| --- | ---: | ---: |
| Correct complete meaning | 11/16 | 11/16 |
| Neural review accepts | 12 | 12 |
| Incorrect neural accepts | 3 | 3 |
| Full grammar agreements | 9 | 9 |
| Incorrect grammar agreements | 0 | 0 |

The expanded prompt improved the eight reused development cases from 4/8 to 6/8, but did not improve the fresh challenge. The challenge includes nested task composition, spatial questions, unknown/ambiguous references, conditions, delays, repetition, and negation. See protocol.json and predictions.json for every request, expected meaning, raw model output, review, and grounding result.

The separate ten reviewer controls deliberately contain semantic errors; these controls are diagnostic inputs, never substituted for generated proposals. Neural review classified only 5/10 correctly and accepted serialized concurrency, parallelized sequence, reversed order, omitted clauses, and an omitted condition. Some generated explanations contradicted their own acceptance boolean.

A subsequent paired-label diagnostic (../v3-paired-review/) evaluated 50 reused candidates/controls with both label orders. Fixed thresholds required conditional agreement probability at least 0.8 and label probability mass at least 0.2 in both orders. It produced 23 correct accepts, 10 false accepts, 5 correct rejects, 9 false rejects, and 3 abstentions. These probabilities are not calibrated confidence. Changing the review output representation did not make model agreement a safe admission criterion.

Full grammar agreement is limited to the declared controlled language. The same authored grammar could itself produce the interpretation in that subset; this is not evidence that a model is necessary there, nor proof about arbitrary English. Certified queries and unknown references do not authorize robot movement. Production preparation additionally requires fresh unique sensor grounding, then scheduler capability, dependency, resource, and motion-feasibility checks at dispatch.

The original source snapshot was checked against the launch receipt hashes before archiving. Model revision, protocol hash, raw outputs, original source.zip, and sampled resources are retained here. The CUDA run used two allocated CPU cores, a 12 GiB memory limit, and a private MPS allocation observed as 32 of 48 GB10 multiprocessors. actual-mps-probe.log records the actual CUDA execution check. Later runner extensions do not change this frozen source archive.

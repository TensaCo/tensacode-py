# Pinned 4B text-model comparison

This reuses the 24 requests and ten semantic-review controls from the earlier 2B experiment. All 4B scores here are development comparison results; the previously named `fresh` split is **not** fresh evidence for this model-selection exercise.

| Frozen foundation and prompt | Correct complete interpretations /24 |
| --- | ---: |
| Qwen3-VL-2B, v2 compact prompt | 15 |
| Qwen3-VL-2B, v3 expanded prompt | 17 |
| Qwen3-4B-Instruct-2507, v2 compact prompt | 22 |
| Qwen3-4B-Instruct-2507, v3 expanded prompt | 23 |

Both 4B prompts emitted duplicate `instruction` JSON keys on the nested-composition case f4; strict decoding rejected both outputs. The v2 prompt also expanded a repeat count that the declared interface does not support. Every raw failure remains in predictions.json. The complete controlled grammar certified 14 candidates under each prompt with zero incorrect certifications in this small sample; grammar agreement still requires fresh grounding and scheduler feasibility before execution.

The 4B neural reviewer classified 8/10 diagnostic controls correctly, including one false accept and one false reject. On generated candidates it accepted only 7 v2 and 6 v3 interpretations; the v2 accepts included an incorrect candidate. Neural self-review remains unsuitable as an admission certificate.

We selected 4B with the v3 prompt using these reused results, then froze the separate `../v3-4b-fresh/protocol.json` challenge before any inference on its 24 new requests. The independent `../v3-mission-questions/` challenge evaluates query intent and deterministic answers from authored scheduler records. Neither experiment measures physical robot success.

The official model revision is `cdbee75f17c01a7cc42f958dc650907174af0554`. `hf cache verify` checked all 13 files successfully; cache-manifest.json records file sizes and Hub blob digests. No model weights are committed. The first download hit its own 2 GiB cgroup memory cap; a one-worker HTTP retry completed under an 8 GiB cap while at least 79.94% of system RAM remained available. Both attempts are retained. The inference run used a 16 GiB limit, two CPU cores, and private MPS; model-runtime.json records 32 visible GB10 multiprocessors out of the physical 48. Full launch receipt, original hash-verified source archive, and sampled resources accompany the outputs.

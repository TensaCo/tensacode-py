# Fresh nested-language challenge

Pinned Qwen3-4B-Instruct-2507 interpreted all **20/20 requests correctly on the first pass**. No schema-repair retry was invoked. These are small authored cases frozen before inference, not an estimate of arbitrary-language reliability.

| Scope | Model | Authored complete grammar |
| --- | ---: | ---: |
| Representable command/query meanings | 14/14 | 9/14 |
| Unsupported requests appropriately declined | 6/6 | 6/6 abstained |

The model preserves nested sequential/concurrent action trees and colored relational references. The six unsupported requests exercise conditions, deadlines, negation, missing references, repeat counts, and future-location questions.

The five correct interpretations outside the controlled grammar are **not admitted for execution**. Nine candidates receive grammar certificates; none has a wrong meaning. Grounding independently reports 11 supported, two ambiguous, one unknown, and six unsupported requests against the authored observation fixture. A semantic certificate does not resolve ambiguity, prove motion feasibility, or authorize dispatch.

This challenge follows selection of the recursive V4 prompt on reused development cases. Earlier failures remain in `../v3-4b-fresh/` and `../v4-repair-development/`; the failed repair is not hidden. Actual physical language execution is a separate experiment in `../v4-three-robot-mission/`.

`protocol.json` was frozen at 21:03:29 UTC before this run. `predictions.json` preserves every raw generation and grounding result. `baseline-comparison.json` uses separate denominators for structured meanings and appropriate abstention. `audit.json` independently recomputes every raw meaning and binds protocol, predictions, and the 304-file executed-source archive.

Host inference used two assigned CPU cores, a 16 GiB memory limit, and private MPS exposing 32 of 48 GPU SMs. Resource samples retained at least 73.75% available system RAM. Raw resource/config/ledger receipts and CUDA preflight are included. No weights are included; the pinned revision and verified cache provenance are in the protocol and `../v3-4b-comparison/`.

To reproduce, extract `source.zip` into an isolated directory, provide the pinned local model cache and matching Transformers 5.17 / Torch 2.14 environment, and run `python -m examples.robotics.warehouse.language_challenge --device cuda --protocol /path/to/protocol.json --output /new/output/directory` under the campaign resource guard. The runner refuses to overwrite retained predictions.

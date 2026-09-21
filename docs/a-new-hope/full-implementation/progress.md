# Full implementation ledger — plan: docs/a-new-hope/4-full-implementation-plan.md

Base: 7827ac0. User authorized full implementation, parallel agents and verified main pushes.

## Preflight

| Interface | Coordination |
|---|---|
| Persistence ↔ representation operations | Caller supplies named live operation bindings; fingerprint constructor config, not weights. Built-in data codecs allowlisted. Workers publish config surfaces. |
| LLM messages ↔ agent tools | Preserve Message(role, str); add ImagePart/TextPart and image encoder before tool worker starts. |
| Graph ↔ vector | Graph worker owns graph adapter module; optional torch import only there. Vector worker owns Latent/Space and communicates signature. |
| All ↔ base Operation | Keep existing invocation; parent coordinates any shared base change. |
| Packaging ↔ integrations | Workers report optional dependencies; parent updates extras, no worker edits pyproject. |
| Each task | All use named ownership and add behavioral tests before implementation; no fake model claims. |

Ruling: implement portable experience files with explicit named operation bindings, rather than serializing arbitrary Python code. This preserves ordinary program ownership and avoids unsafe executable artifacts; consumers must reconstruct compatible operations when loading.
Ruling: no remote model keys are configured, so provider HTTP behavior is tested against local servers and live model validation uses explicitly acquired local models. Any unavailable live provider evaluation is reported, never simulated as model performance.

## Implemented and independently reviewed

- Durable experience/training: subprocess Banking77 demonstration completed; mutation-before-release, pending async calls, immutable mapping codecs and malformed optimizer slots fixed with regression tests.
- Vector operations: explicit spaces/latents, image patches, candidate scoring/decisions/retrieval; review identified configuration collisions, unsupported spatial-coordinate claims and ignored masks. Final configuration hardening is complete.
- Message/providers: multimodal and structured operations plus real HTTP adapters; review covered redirects, schema names, incomplete answers, batching and malformed distributions. Remote live calls remain untested without credentials.
- Graph operations: immutable source-preserving graphs, supplied semantics and trainable graph-to-vector path; dtype/device, native hooks and deterministic gradient evidence reviewed. Official MUTAG experiment completed.
- Agent tools: memory/objective rollback, nested transactions, source ID allocation, restart consistency and async cancellation fixed. Image bytes tested through an actual local HTTP transport.
- Parent integration: explicit local Transformers adapter, real model smoke tests, example documentation, optional dependencies and dependency-free wheel verification. Final whole-system reviewer found no additional blocker in cross-representation save/load/train checks.

## Measured evidence

- Banking77: four terminated subprocesses; 9,997 training rows, 3,080 heldout; accuracy 0.0097403 → 0.8948052, cross-entropy 4.3645932 → 0.4565712.
- MUTAG: 150 train / 38 heldout graphs; accuracy 33/38 → 34/38, cross-entropy 0.6866499 → 0.2946415. Small split and high initialization accuracy limit claims.
- SmolVLM-256M: actual image/text inference works mechanically, but the published candy image is misdescribed/miscounted and structured answers are rejected. Results preserved; Qwen3-VL-2B was also checked; it counted correctly but misidentified the candy symbol and initially failed both structured formats. See the final report for the explicit format-instruction follow-up.

Authoritative final test/build/CI status and any remaining limitations are in `../5-full-implementation-report.md`.

Final independent tools re-review passed all cancellation/rollback/source-validation regressions. All five workers handed off complete implementations with no outstanding reviewed findings. Final integrated suite: 174 tests passed.

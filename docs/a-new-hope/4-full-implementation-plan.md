# Full architecture implementation plan

> Execute with subagents where file ownership is disjoint. Work on main only; the owner explicitly authorized full implementation and verified milestone pushes.

**Goal:** Complete the architecture's usable operation/tool/tracing/training surfaces, including correction → save → new process → reload → train, interchangeable representation operations, provider integration, multimodal input, memory/objective composition, graph learning, and real-world validation.

**Spec:** [1-tensorcode-architecture.md](1-tensorcode-architecture.md).

**Constraints:** Ordinary Python control flow; `op(value, *, context=None)`; opt-in effects; no implicit semantic seeds; lightweight base imports; no invented probability, gradients, grounding, or learned capability. Full implementation means concrete mechanisms plus real inputs, not a universal autonomous intelligence guarantee. No empty placeholder implementations.

## Shared interfaces and ownership

- All workers read AGENTS.md. No branches/worktrees, no reverts of others' changes. No nested agents. Parent coordinates commits, packaging, final docs and integration.
- Tracing/training worker owns `src/tensorcode/tracing.py`, new `src/tensorcode/training/`, tests prefixed `test_persistence`, `test_training`, and task report.
- Vector worker owns `src/tensorcode/ops/vec/` and tests prefixed `test_vec_`.
- Message/provider worker owns `src/tensorcode/ops/llm/`, `src/tensorcode/integrations/` and tests prefixed `test_llm_`, `test_provider_`.
- Graph worker owns `src/tensorcode/ops/graph/` and tests prefixed `test_graph_`.
- Agent tools worker owns `src/tensorcode/tools/` and tests prefixed `test_agent_`, `test_decision_tool_`; starts after message interfaces are published.
- Parent owns docs, pyproject/CI, top-level exports if needed, evaluation examples and cross-subsystem tests.

## Task 1 — Durable supervised experience and trainers

- [x] Add `session.supervise(output_or_ref, target, *, loss='cross_entropy', source='human')` with explicit ground-truth provenance.
- [x] Add versioned data-only experience export/load: inputs, dependency DAG, supervision, operation IDs/configuration fingerprints, external boundaries. Do not pickle arbitrary executable objects. Operation bindings are supplied by caller when loading; codecs are explicit and allowlisted.
- [x] Provide `session.save(path, operations={name: instance}, release=False)` and `training.load(path, operations=..., codecs=...)` with portable replay. Persist data independently of the original session; reject missing or incompatible bindings unless explicit permitted parameter updates.
- [x] Provide `Trainer(operations, optimizer=...)` or similarly small explicit API to fit supervised experiences, deduplicate shared parameters, reject nondifferentiable targets, and report loss. Support cross-entropy and MSE plus explicit custom loss callbacks in-process.
- [x] Provide checkpoint save/load of supported tensor module states with named bindings and shared-parameter alias validation; separate configuration fingerprints from mutable weights.
- [x] Tests: subprocess fresh load/train, held-out improvement, context and dataclass roundtrip, changed configs/missing bindings rejected, shared optimizer parameters deduplicated, no side-effect replay, intermediate release, malformed artifact rejection.

## Task 2 — Vector toolbox

- [x] Representation space metadata and `Latent` preserving native tensors and gradients, with explicit compatibility checks where consumed.
- [x] Trainable image patch encoding retaining spatial organization; explicitly initialized or supplied model, no pixel-understanding claim from random weights.
- [x] Decode via supplied modules, score, choose and retrieve primitives with explicit candidate inputs and score meanings; preserve tensor outputs until caller chooses discrete values.
- [x] Shared backbone behavior and meaningful state/configuration contracts for persistence. Do not break current TextEncoder/Classify APIs.
- [x] Tests: incompatible spaces, cross-modal explicit adapters, batch/empty candidates, ranking identities, gradients, image patch shapes and real parameter updates. Report local model semantics separately.

## Task 3 — Message decisions and provider integrations

- [x] Multimodal message content with immutable text and image parts, source references; keep `Message(role, str)` working. Image encoders must preserve bytes/URLs distinctly and avoid unintended downloads.
- [x] Implement Classify, Score, Decide, Retrieve with explicit schema/result validation, abstention and distributions only when actually supplied. No global threshold or fabricated confidence. Share model interface across operations.
- [x] Add explicit synchronous and asynchronous call surfaces and batching where backend supports them, without changing a sync return into an awaitable.
- [x] Implement actual HTTP provider adapter(s), at least an OpenAI-compatible local/remote endpoint with documented request/response schemas and a Jev adapter if current primary documentation establishes its API. No secret logging, implicit retries of effects, or implicit provider fallback.
- [x] Tests through a local HTTP server for wire shape, multimodal parts, structured outputs, errors/timeouts, missing confidence and malformed answers. Live evaluation is separate from these transport tests.

## Task 4 — Graph toolbox and cross-representation learning

- [x] Extend graph representation with immutable attributes/source anchors and alternatives if needed without fixed domain vocabulary.
- [x] Implement explicit encode/decode preserving graph identity/relations, retrieval and decision/scoring using supplied semantics.
- [x] Add real trainable graph-to-vector module/adapter and a graph prediction path, preserving correspondence and source metadata. Do not call deterministic queries learned reasoning.
- [x] Tests: roundtrip identity/source preservation, competing graph facts retained, unknown referents not invented, gradients through a graph adapter, measured generalization on disjoint graph inputs.

## Task 5 — Composed decision and agent tools

- [x] Ready-to-use bounded decision tool with explicit model/labels, distributions and replaceable selection policy; retain explicit component composition.
- [x] Extend Chatbot for images through public message operations, objective revisions through an explicitly configured transform, retrieval-backed memory, persistence, and transactional turn failure behavior.
- [x] Add memory tool with stable source IDs and explicit retrieval semantics; keep stored observations distinct from generated responses.
- [x] Add bounded action loop using supplied chooser and action map with exact option validation, effect receipts, no execution on abstention, and maximum-step control. No language heuristic choosing an action.
- [x] Tests: real image objects to provider wire, memory retrieval and restart, supplied objective update, failed turn rollback, invalid action/abstention/budget, concurrency/async state safety.

## Task 6 — Integration, real tasks and verification

- [x] Demonstrate save-feedback-restart-train on official Banking77 held-out data; save commands, hashes, seeds, losses and limitations.
- [x] Run the same classification/retrieval interface through vector and message implementations. Use a real local model where no provider key is configured; no fixtures counted as model capability.
- [x] Demonstrate actual image + text input to the multimodal path with source evidence. Prefer a supplied local pretrained model; record its model ID/revision and actual answers, including failures.
- [x] Verify a real public graph task or clearly distinguish graph-mechanism tests from real-data learned evaluation.
- [x] Independent per-task reviews and whole-branch review, full test suite, wheel/sdist, clean no-extras installation, CI 3.11–3.13, accurate README and implementation coverage matrix.
- [x] Commit/push coherent verified milestones and leave main clean.

## Review focus

1. Saved artifacts must not execute arbitrary code or silently use incompatible models.
2. Replaying a trace must not repeat remote calls/actions, detach intended gradients, or treat feedback as model output.
3. Equal-sized tensors must not bypass representation-space compatibility.
4. Reported probability/confidence must retain provider semantics and missing values.
5. Tool rollback, memory, objectives and action bounds must hold on real failure paths.

Implementation details, measured limitations and verification are recorded in [the final report](5-full-implementation-report.md). The architecture surfaces are implemented; live hosted-provider testing remains unavailable without credentials, and local model failures are recorded rather than counted as successful decisions.

# TensorCode reset implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement inline, with test-first implementation and a final independent review.

**Goal:** Replace the archived implementation with a small working cognitive-operations library, trace DAGs, and a real-data trainable decision tool.

**Architecture:** Public callable operations live under `ops.vec`, `ops.llm`, and `ops.graph`; supported compositions live under `tools`. Context-local tracing records explicit dependencies. Native tensor training and a bounded in-memory replay path establish actual learning without claiming arbitrary Python differentiation.

**Tech Stack:** Python 3.11+, optional PyTorch for vectors and training; standard library for core, graph and message operations; pytest for verification.

**Spec:** [1-tensorcode-architecture.md](1-tensorcode-architecture.md)

## Global constraints

- Work on `main` only; preserve and verify a private archive before deletion.
- Uniform `op(value, *, context=None)` invocation, developer-selected role names.
- No production imports of fixtures or removed semantic defaults.
- No fabricated model competence or universal differentiation claim.
- Importing the core must not require optional ML dependencies.
- Tool behavior must be composed through public operations.

## Review focus

- Equal or aliased values must not manufacture provenance; use explicit output references for ambiguous scalar values.
- Mutation and unobserved Python transforms must not yield false replay claims.
- Conditioning context must survive replay and participate in dependencies.
- Shared parameters must receive gradients without duplicate optimizer registration.
- Failed calls, nested sessions and context isolation must not corrupt the active trace.

## Task 1: Archive and cut

Files: commit current documents; replace `src`, `tests`, `examples`, `eval`, `research`, old docs, packaging and CI as needed. Preserve `LICENSE`, original feedback, architecture series and relevant repository workflow instructions.

- [ ] Validate Markdown and record the baseline suite outcome, including existing failures.
- [ ] Commit the pre-reset state, push main, create the private archive, push all local branch/tag refs, verify privacy and exact main SHA.
- [ ] Keep a local bundle of all refs outside this checkout. Do not delete caches or the environment.
- [ ] Delete superseded tracked implementation and documentation; no compatibility shim.

## Task 2: Callable operations and trace DAG

Files: `src/tensorcode/ops/base.py`, `src/tensorcode/tracing.py`, `tests/test_tracing.py`.

Interfaces: `Operation.__call__(value, *, context=None)` delegates to `forward`; `trace()` yields a session; session exposes invocation records, `ref(value)` and explicit output handles; `session.replay(target, inputs=...)` recomputes a supported dependency closure.

- [ ] First write failing behavioral tests: chain replay with a replaced root, context dependency, no false equal-scalar edge, failed operation, nested scopes, and explicit references.
- [ ] Run `python -m pytest tests/test_tracing.py -q`; expect missing new API before implementation.
- [ ] Implement context-local capture and conservative identity handling; no storage format or general mutation tracking claim.
- [ ] Re-run the tests; expect all pass. Add rejection tests for unsupported boundaries as implementation exposes them.

## Task 3: Concrete representation operations

Files: `src/tensorcode/ops/{vec,llm,graph}/`, `tests/test_operations.py`, `tests/test_learning.py`.

Interfaces: vector `Transform(module)` preserves native tensors and parameter registration; vector `Classify(module, labels)` exposes logits and probabilities; LLM message encoder and callable-model transform; graph immutable representation and explicit transform callable. No fake pretrained knowledge.

- [ ] Write failing tests for real tensor gradients through a traced composition, shared parameters, incompatible results, message/context preservation, graph immutability and source retention.
- [ ] Run those tests before implementing the API.
- [ ] Implement concrete adapters, preserving optional dependency boundaries.
- [ ] Verify meaningful SGD loss reduction and replayed root-to-output gradient flow.

## Task 4: Importable tools and real-data evaluation

Files: `src/tensorcode/tools/decision/`, `src/tensorcode/tools/agents/`, `examples/banking77.py`, `tests/test_tools.py`.

Interfaces: a decision pipeline receives public encode/classify components and returns a prediction; a chatbot receives explicit model or operations and retains conversation only after a successful reply. Tool constructors do not invent an unconfigured backend.

- [ ] Write failing tests for tool composition, real classifier outputs and transactional conversation failure behavior.
- [ ] Implement the minimal tools through public operations only.
- [ ] Train an explicit local model on the real Banking77 train CSV and evaluate once on its disjoint official test CSV. Record pre/post accuracy and held-out loss; report supplied tokenization and labels.
- [ ] Save a reproducible command and report; do not bundle training data or model weights in the wheel.

## Task 5: Release coherence and review

Files: `README.md`, `AGENTS.md`, `pyproject.toml`, `.github/workflows/ci.yml`, architecture implementation report.

- [ ] Update current architecture guidance and archive links, removing stale APIs and broken local links.
- [ ] Run full new suite, build wheel/sdist, and install the wheel without dependencies into a fresh environment to test core imports and graph/message execution.
- [ ] Review provenance/learning boundaries, convenience APIs and destructive scope in multiple passes. Request an independent code review while inspecting docs and distribution contents locally.
- [ ] Fix consequential findings with regression tests, record limitations, commit and push verified main, and confirm a clean worktree and matching origin SHA.

## Execution record

The owner explicitly authorized continuing from architecture review to checkpoint, destructive cuts and replacement implementation. The phrase ending “then the real world” was clarified asynchronously; pending the response, the independent real-data decision experiment is the default interpretation. This plan intentionally implements a verified first replacement milestone rather than pretending the full long-term architecture is complete.

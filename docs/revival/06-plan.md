# 7. Modernization plan

Ordered so that each step is a small, reviewable change that leaves `main`
importable and tested. Nothing here requires a new framework. The prototype in
the repository is the code that lands, renamed from `tensacode` to `tensacode`.

## 7.1 Decisions needed from the owner first

1. **License.** The README and `pyproject.toml` say MIT, but no license file has ever been committed. Confirm MIT, or choose another, before anything is called open source.
2. **Package name.** `tensacode` returned 404 on PyPI and TestPyPI (2026-09-16). That means it is not published; it does *not* guarantee the name can be registered. Reserve it before announcing.
3. **Scope of the core.** Recommended:
   - Keep `to_records` / `from_records` in the core only if a real program needs object-graph import.
   - Drop `propose` from the first release until a rewrite workload exists (it has contract tests and no example).
4. **Remote model adapter.** Should the first release ship one (`locality="remote", egress=True`), given that a local zero-shot model tier *hurt* quality in the only measured task?

## 7.2 Sequence

| # | Change | Size | Done when |
| --- | --- | --- | --- |
| **P0: honesty and hygiene** | | | |
| 1 | Add the chosen `LICENSE`; fix the README badges, status ("pre-alpha, not on PyPI"), and dead links; delete the README's invented example outputs | XS | README claims are all true |
| 2 | Delete `.old/` (247 files) and the empty `examples/*` skeletons. Git keeps them; tag the last legacy commit `legacy-2024-11`. | XS | Tree contains only code that runs |
| 3 | Replace the Poetry manifest with PEP 621: no core dependencies; extras `learned` (scikit-learn, numpy) and `local-model` (torch, transformers); CI on 3.11–3.13 running pytest | S | `pip install .` pulls 0 third-party packages; CI green |
| **P1: land the vertical slice (7.3)** | | | |
| 4 | Land `outcomes.py`, `runtime.py`, `ops.py` (`parse`, `classify`, `choose`, `rank`, `check`, `verify`), and `backends/builtin.py` as `tensacode`. Delete `engine.py`, `ops/`, `internal/meta/`, `internal/utils/{misc,pydantic,language}.py`. | M | The runtime and truth-value tests pass |
| 5 | Land `records.py` (Store, codec, patches). Delete `internal/tcir/` and `internal/utils/locator.py`. | M | Records tests plus the representation comparison pass |
| 6 | Land `actions.py` and `context.py` | S | Actions and context tests pass |
| 7 | Land the support-router (local bindings), knowledge, and recovery examples with their `OUTPUT` files, plus `eval/banking77_cascade.py` (without `--model`) and `eval/recovery_trajectories.py` | M | Results reproduce within tolerance (7.3) |
| **P2: harden before any external consumer** | | | |
| 8 | Versioned JSON schema (`tensacode/1`) for `Request`, outcome values, records, and `Span`, with golden-file round-trip tests | S | A TypeScript consumer can validate traces |
| 9 | Deadlines inside backend calls: a thread/async wrapper with timeouts for model tiers, and cancellation hooks | M | A slow tier cannot exceed `Policy.deadline_ms` by more than one poll interval |
| 10 | `Policy.min_quality`: admit an implementation only if `Profile.quality[metric]` from a named eval artifact clears a bar. Load profiles from `eval/results/*.json`. | S | The Banking77 model tier is excluded by default and admitted only with an artifact that shows it helps |
| 11 | Compute request digests lazily (only for caching or `record_inputs="digest"`) | XS | `rank` overhead on 40 candidates falls from ~0.16 ms |
| 12 | Safe artifact format for learned tiers: no pickle; `.npz` weights plus JSON vocabulary; loading needs no scikit-learn | S | The learned tier loads in the core-only install |
| 13 | `dedupe` blocking (MinHash/LSH) before pairwise similarity; indexes for `Store.match` on (predicate, object) | S | `dedupe` p95 < 1 ms at 40 items; `match` stays sublinear at 1M claims |

## 7.3 The smallest vertical slice that proves the design

**Contents.**
- **Runtime:** outcomes, runtime, `parse` / `classify` / `choose` / `check` / `verify` facades, Store without `to_records`, `invoke`, `pack`.
- **Implementations:** rules; the learned classifier as an optional extra.
- **Programs:**
  - the **support router** with local bindings (an agent that parses, classifies, chooses under constraints, invokes, verifies, and persists);
  - the **knowledge program** (records, evidence, contradictions, three-valued checks);
  - the **recovery agent** (bounded, fact-driven retries).
- **Verification:** tests plus two evaluation scripts.

**Why this is the minimum.**
- **Typed operations without a model.** Every design claim that matters is exercised: typed facades, backend independence, abstention with a reason, constraints the backend cannot bypass, and unknown not being false.
- **Records and effects.** It also covers content-identity claims with evidence and time, receipts versus verification, write-ahead persistence, and bounded recovery.
- **Measured, not asserted.** The slice ships with its measurements.

It excludes the local-model tier, context selection, plans, and the object-graph
codec. Each is demonstrated in the repository but not required to prove the design.

**Acceptance criteria (all measured today; see 06):**

| Criterion | Target |
| --- | --- |
| Core-only install | 0 third-party packages; `python -m pytest` passes offline |
| Banking77 rules → learned, test split | Coverage 93.9% ± 1 pt; selective accuracy 94.1% ± 1 pt |
| TensaCode overhead per `classify` call | p50 ≤ 0.05 ms on the reference machine |
| Recovery simulation, 5,000 episodes, seed 0 | 0 duplicate effects; ≥ 89% done exactly once |
| Knowledge demo | 1 conflict detected; `check` returns `unknown` for both sides at 10:02; lossless JSON round trip |
| Main programs | ≤ 40 lines each |
| Line budget | Core ≤ 1,500 non-blank lines (1,474 today) vs the legacy active package's 5,252 non-blank lines, none of whose engine, op, or TCIR modules import |

## 7.4 Deferred until a workload demands it

| Item | Trigger |
| --- | --- |
| Distillation / crystallization of escalations into a learned tier | ≥ several hundred *verified* escalations for one contract, plus an eval artifact showing the replacement beats the tier it replaces on that slice |
| Differentiable programs and tensor latents | A task where gradients through program structure beat training leaf models separately |
| Planning (`propose → Plan`) | An agent whose actions have real ordering dependencies. The `Plan` / `plan_order` / `run_plan` types exist and are tested. |
| Code generation and execution | A sandboxed executor registered as an action |
| Graph database | The reference store exceeds memory, or needs concurrent writers |
| Language ports | JSON schema v1 frozen and at least one external consumer using it |
| Repository consolidation (`tensacode`, `tensacode-py`, `JacobFV/tensacode`) | After P1 lands |

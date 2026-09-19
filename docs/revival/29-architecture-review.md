# 29. Architecture review: what to realign, repair, remove

*2026-09-18. Written after building the assay (`eval/suite`), which made the structure visible.*

## 29.1 The finding

There are **two codebases pretending to be one**.

* The **library** (`src/tensorcode`): typed operations (`parse`, `classify`, `choose`, `rank`,
  `check`, `verify`), a runtime that binds them to implementations by policy, a claim store,
  and about twenty cognitive modules (memory, awareness, wants, control, metacognition,
  expectation, social, chunking, cues, permanence, change, relation, quantity, causal,
  temporal) plus a learning package (induce, verify, library, certificate).
* The **agent** (`src/tensorcode/agent` + `examples/general_agent`): reads text, sets goals
  from VerbNet, picks a plugin capability, acts, verifies, replies.

The agent imports `records`, `runtime`, `actions`, `outcomes` and `language`. **It does not
call a single typed operation, and it uses none of the cognitive or learning modules.**
Measured directly: 23 modules, about 6,500 lines, are never reached from the agent or its
plugins, including the whole `learning/` package. `Runtime` is constructed and used only to
hold a trace: no implementation is ever registered, and `Policy` never decides anything.

So the thesis — *typed operations whose meaning is separate from their implementation,
chosen by policy* — is not what the agent does. The agent hard-codes its own thin version:
capability selection is a bespoke loop in `core.plan`, answering is a bespoke loop in
`core.lookup`, and neither is an operation that could be implemented another way, swapped,
or measured against an alternative.

That is the misalignment to fix first. Everything below follows from it.

## 29.2 Inventory

### Keep as-is

| Abstraction | Why it holds |
| --- | --- |
| `outcomes` (`Unknown`, `Verdict`, `Receipt`, `Score`) | The distinctions the whole design rests on; used everywhere, and they caught real bugs |
| `records` (claims with evidence, time, scope) | The right shape for a store; see repairs below |
| `actions` (`ActionSpec`, `invoke`, receipts) | Effects with honest receipts |
| VerbNet goals, WordNet kinds, STREUSLE roles, treebank parser | Knowledge as data, each measured |
| `eval/suite` (tasks, subjects, judges, store, report) | New, and the reason this review exists |

### Realign (the agent should express itself in these, not beside them)

| Now | Should be |
| --- | --- |
| `core.plan` picks a capability with a hand-written scoring loop | `tc.choose(capabilities, objective=achieve(goal), constraints=[kinds fit, effects achieve, nothing partial])` — an operation, with the constraints as data |
| `core.lookup` walks the store with special cases for predicates and event hops | `tc.rank`/`tc.check` over a *query* value, so retrieval is an operation with alternatives (exact, cue-based, embedding) that can be compared |
| `LearnedReader` / grammar reader chosen by a constructor flag | `tc.parse(text, into=Utterance)` with two registered implementations and a policy that picks by measured quality |
| Verification inside the desktop plugin | `tc.verify(receipt, observe=…, expect=…)` in the core, with the plugin only observing |
| `Runtime` as a trace holder | Real implementations with `Traits`/`Profile`, so "which reader/retriever/recognizer" is a policy decision with numbers behind it |

### Repair

1. **The claim store has no discipline about provenance.** Everything lands in one pool:
   what the user said, what a plugin observed, what was inferred. `Claim.scope` exists and is
   used for one thing (user-said). Repair: three scopes (told / observed / inferred), a rule
   that observation supersedes older observation of the same subject-predicate, and inference
   recorded with its premises (`Evidence.derived_from` already exists).
2. **Retrieval is ad hoc and has produced wrong answers twice today.** Repair: a small query
   value (pattern over subject/predicate/object with named holes) and one place that answers
   it, with the "answer the open side, never the bound side" rule as a property test.
3. **Pragmatics is scattered.** Deixis (`I`/`you`), use/mention, indirect requests, "called X"
   naming, time-vs-place are split between `understand.py` and `core.py`. Repair: one
   `interpret` stage between parsing and acting, testable on its own.
4. **The plugin protocol conflates five jobs**: vocabulary, kinds, perception, capabilities,
   reference resolution, presentation. Repair: separate protocols, with `Plugin` as a bundle
   that may implement several. A vision plugin implements `Perceiver` only; the desktop
   implements all five.
5. **No planner.** `core.plan` is single-step; multi-step requests ("make a folder and move
   the files into it") cannot be expressed. Repair: goal regression over declared effects
   producing a `tc.Plan`, with preconditions from `learning.induce`.
6. **Generation is half-templated** and uses the hand grammar while parsing uses the learned
   one. Repair: one grammar for both, or an explicit statement that generation is a separate
   seeded artifact, with the round-trip task in the assay as its measure.

### Remove or quarantine

* **`legacy/`**: the 23 unused modules. Some deserve integration (`memory`, `wants`,
  `metacognition`, `expectation`, `cognition`); the rest (`priming`, `permanence`, `cues`,
  `change`, `frames`, `social`, `relation`, `quantity`, `causal`, `temporal`,
  `semantics_bridge`, `answer_type`) should move to `legacy/` until something in the assay
  needs them. Code that nothing calls is not architecture; it is inventory.
* **`semantics_bridge` and `answer_type`**: surface-cue regexes that the parser now subsumes.
* **112 old eval scripts and 90 result files**: move to `eval/legacy/`, keep the results as
  history, and let `eval/suite` be the only thing anyone runs.
* **CIFAR-10 as a vision claim**: demoted to a diagnostic in the registry; the real vision
  tasks (Visual Genome naming, GQA, ScreenSpot) are registered and marked as needing data.

## 29.3 The assay as the forcing function

The suite makes each of these measurable rather than argued:

| Repair | The task that proves it |
| --- | --- |
| Retrieval as an operation | `memory.longmemeval`, `knowledge.nq_webq` |
| Planner | `computer_use.*` with world-state graders |
| `choose` as an operation | `tools.function_calling` (BFCL), where the tools are not a desktop |
| Interpretation stage | `pragmatics.ambiguous` (ask, don't answer) |
| Generation | `language.generation_roundtrip` |
| Learning in use | `learning.in_use`, `learning.chomsky_benchmark` |
| Vision beyond a toy | `vision.object_naming`, `vision.gqa`, `vision.screenspot` |

## 29.4 Plan

**Phase 1 — express the agent in the library's operations (2–3 days).**
`choose` for capability selection, `parse` for reading with two registered implementations,
`verify` in the core, retrieval as a ranked query. Runtime gains real implementations with
profiles. *Done when* the agent's turn is a sequence of traced operations and the reader can
be switched by policy rather than by a constructor argument.

**Phase 2 — repair the store and retrieval (1–2 days).**
Scopes, supersession, a query value, and the property test that an answer never comes from
the side the question already gave. *Done when* `memory.longmemeval` and `knowledge.nq_webq`
move off zero without any wrong answers appearing.

**Phase 3 — split the plugin protocol and add the planner (2–3 days).**
Perceiver / Actor / Referrer / Presenter / Vocabulary. Goal regression with preconditions.
*Done when* a two-step request works and `computer_use.shell_files` has a world-state grader.

**Phase 4 — quarantine and integrate (1 day).**
`legacy/` for the unused modules and old evals; integrate `memory` and `expectation` where the
repairs need them. *Done when* nothing in `src/tensorcode` is unreachable from either the
agent or a registered task.

**Phase 5 — the missing tasks (3–5 days).**
BFCL for tool use, Visual Genome and GQA for vision, the Chomsky-hierarchy suite for
learning, round-trip for generation. *Done when* the scorecard has no "needs data" rows that
we intend to keep.

**Phase 6 — learning in use (the research bet, 2–4 days, with a kill criterion).**
Unknown words and induced rules admitted through `learning.verify` into `learning.library`.
*Done when* `learning.in_use` recovers at least 2 of 5 deleted constructions, or we report
that it does not.

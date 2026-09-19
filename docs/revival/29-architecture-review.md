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

## 29.4 What the sibling projects already settled

`symbolic-ai-models` and `synthEX` (same owner) have measured versions of decisions we are
about to make. Their findings, and what each one costs us:

1. **A flat subject-predicate-object triple cannot state a nested proposition.** Their
   decision note is explicit: a triple-only encoding cannot say *"Casey believes Lara did
   X"* without inventing reification nodes no consumer agrees on. **We already pay this.**
   `to_claims` reifies a sentence into `event:… is_a be`, `event:… subject …`,
   `event:… location …`, and my `_through_events` hop is exactly the "consumer that had to
   agree with the invented nodes" — it is where today's two wrong answers came from.
   → **Claims become n-ary: a predicate with named roles whose fillers may be other claims.**
2. **Two timelines, not one:** when a thing was true, and when we learned it. One clock is
   how a replay silently reads the future.
3. **Modality and provenance, not a boolean.** Their enum (asserted / hypothesised /
   believed / desired / obliged / possible / counterfactual / questioned) plus polarity plus
   `confidence=None` meaning *not stated* — distinguished from 1.0, because an imputed
   certainty is indistinguishable downstream from a measured one.
4. **Identity is a claim, not a merge.** Don't dedupe entities at ingest; record an
   alignment with method and confidence, defaulting to *possibly the same*. **We violate
   this**: `default_ref` mints `entity:<text>`, so two different people called Jacob are
   silently one entity.
5. **Plugins should be kernel manifests:** `requires`/`provides` as capability profiles,
   `semantics` as a set, and per-property **authority** (veto / propose / observe). Fusion
   then consults authority instead of taking the last writer, because last-writer-wins is
   indistinguishable from a correct answer at the point where it is read. The desktop plugin
   is the authority on paths; vision only proposes.
6. **A facade must declare what it discarded.** Their kernels return a projection plus a
   discard record and may not assert over what they did not cover. Our parse coverage is the
   same idea, unnamed and used for the wrong purpose (below).
7. **Fusion defaults to keeping the mixture**, and retention must be asserted *at the
   serialization boundary*: their alternatives survived in-process and were dropped by
   `to_dict`. A round-trip retention test is not optional.
8. **Form-validity is not a correctness gate.** Their cascade gated well-formedness, so
   nothing escalated and accuracy landed at the cheapest tier. **We do this too**: the agent
   refuses to act when words were skipped — a *form* gate. What routes well is **graded
   confidence** (their agreement-only routing captured 0.0% of the headroom; the reader's own
   confidence separated hard items at AUC 0.880).
9. **Measure the decomposition before adding learning.** Reading factored into containment ×
   proposal × classifier × ranker and reproduced end-to-end rates within 0.028 — and the
   headroom was in the factor nobody had modelled. Our parse→goal→capability→verify chain
   should be factored the same way before anything is learned.
10. **Pre-register the renaming control.** Permuting symbols took their best model from
    1.14 bits to 0.038 while a genuinely structural reading held 98.6%. Any learned component
    we add reports the renamed score beside the raw one.
11. **Power statements.** Three headline claims flipped sign between a small slice and a
    large one, and an n=184 null was overturned at n=663. Our dev categories are n=12: every
    rate now prints an interval and an "underpowered" flag.
12. **Cost belongs in the ledger.** Their throughput-first phase ended with 18 of 19 models
    costing infinity per solved episode above floor. The assay should carry $ or seconds per
    solved item above the control.

**Not to repeat:** one universal logic or one merged mega-ontology; silent fallback when a
component is missing (refuse loudly — our vision plugin currently sees nothing in silence);
weak floors (a floor must use every feature the subject sees); and conclusions from small
slices.

## 29.5 Plan

**Phase 0 — the claim schema (1–2 days, do it before anything is built on top).**
N-ary claims with named roles and nested fillers; valid time and transaction time; modality,
polarity and `confidence=None`; identity as alignment claims rather than minted names. This
retires the reified-event encoding and the hop that answered with adverbs. *Done when*
`to_claims` emits one claim per proposition, retrieval needs no event hop, and a round-trip
test proves competing claims survive serialization.

**Phase 1 — express the agent in the library's operations (2–3 days).**
`choose` for capability selection, `parse` for reading with two registered implementations,
`verify` in the core, retrieval as a ranked query. Runtime gains real implementations with
profiles. *Done when* the agent's turn is a sequence of traced operations and the reader can
be switched by policy rather than by a constructor argument.

**Phase 2 — repair the store and retrieval (1–2 days).**
Scopes, supersession, a query value, and the property test that an answer never comes from
the side the question already gave. *Done when* `memory.longmemeval` and `knowledge.nq_webq`
move off zero without any wrong answers appearing.

**Phase 3 — plugin manifests and the planner (2–3 days).**
Perceiver / Actor / Referrer / Presenter / Vocabulary, each declaring `provides`, `requires`
and per-property authority, with fusion that keeps the mixture. Goal regression with
preconditions. Replace the form gate (skipped words) with graded-confidence deferral.
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

## 29.6 Note (owner, 2026-09-18): the agent belongs in its own repo

**Decision to act on later, recorded now:** the agent is a *user* of tensorcode, not part of
it. It should move to a separate public repository, `tensorcode-agent`, so the dependency
runs one way and the library's claims stand on their own.

What that implies when we do it:

* **`tensorcode` (library) keeps**: outcomes, records/claims, runtime and policy, the typed
  operations, actions, context, the language stack (grammar, treebank parser, VerbNet and
  WordNet adapters, induction), vision features, and the cognitive modules that survive the
  quarantine. Its tests are unit tests plus library-level measurements (parsing accuracy,
  grammar induction, feature learning).
* **`tensorcode-agent` takes**: `agent/` (the turn loop, interpretation, planning, replies),
  the plugins (desktop/computerworld, vision), the chat server and page, `eval/suite` and the
  held-out prompt sets — because those measure *an agent*, not the library.
* **The seam is the plugin and operation protocols.** If the agent repo can be written
  against tensorcode's published API without reaching into internals, the decoupling is real;
  if it cannot, that is a list of things the library still needs to expose.
* **Order**: do the phases above first. Splitting now would freeze today's interfaces, which
  are exactly the ones under repair — in particular the claim schema (Phase 0) and the
  operations the agent should be expressed in (Phase 1). Split at the end of Phase 3, when
  the protocols have stopped moving.

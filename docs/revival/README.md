# TensaCode revival: assessment, design, and proof

*2026-09-16. Scope: `TensaCo/tensacode-py`. Status: adopted as `tensacode` 0.1.0a1 (pre-alpha); the legacy package is tagged `legacy-2024-11`. Documents below are dated working notes and are not rewritten after the fact.*

**Bottom line.** Today's `tensacode` package has an MIT badge, a wheel, and 5,252
non-blank lines, and none of its engine, ops, or TCIR modules imports. Its best ideas are real:
- typed objects as operation inputs,
- meaning separate from implementation,
- introspection and feedback.

This proposal keeps those ideas and replaces the machinery with:
- **typed facades** (`parse`, `classify`, `choose`, `rank`, `check`, `verify`, `propose`, `invoke`) whose outputs are validated and whose failures are explicit values (`Unknown`, three-valued `Verdict`, four-state `Receipt`);
- **a small backend protocol** (`accepts`, a batched `run`, declared traits, measured nullable profiles) with policy-driven cascades, budgets, caching, and traces;
- **a coarse representation**: typed native values, plus entity and claim records carrying evidence, time, and scope; plans are kept in a separate graph.

The prototype (the repository, 1,474 non-blank core lines including docstrings, plus 133 for built-in rule backends; zero required
dependencies) runs four example programs offline and passes 43 tests. Its
measurements use real data and a real local model.

## Current direction, 2026-09-19

The governing objective is [36 — The structured cognitive workspace](36-structured-cognitive-workspace.md):
unstructured language, images, and observations should become evidence-backed, revisable
interpretations in an extensible structured core. Planning, open-ended reasoning, hypothesis
formation, and model learning belong in that core. Communicative and action intentions are
then realized as output, with observed results returning to the same process.

Vision here means holistic scene understanding: layout, grouping, relational structure,
events, affordances, and competing explanations of the whole situation. Object labels and
regions are supporting evidence. The target uses extensible relational representations and
shared grounding with language; it is not limited to UI elements or classification. Learned
scene formation, temporal dynamics, and active visual investigation remain open work.

The numerical claims in this index's introduction are historical. [34 — Cognitive primitives
and implementation](34-cognitive-primitives-and-implementation.md) qualifies the original
prescriptions in [33 — The cognitive fronts](33-the-cognitive-fronts.md). [35 — Model-based
planning and inspectable refinement](35-model-based-planning.md) reports bounded multi-step
planning, real filesystem work, held conditions, and structured revision/resumption. These
improve execution after a usable specification arrives; interpretation and model formation
remain the larger missing capabilities. The named Python project path uses authored knowledge,
and the original compound wording remains a documented failure.

The first workspace slice preserves reader alternatives and source evidence with explicit,
revisable selection. [37 — Interpretation workspace: first implementation slice](37-interpretation-workspace-first-slice.md)
records the historical API and its limits. The [scene checkpoint](38-scene-interpretations.md)
removes its first-reader compatibility default and the old direct visual-claim path. Language
interpretations now defer unless an explicit policy selects them; image providers supply
revisable scene proposals. [39](39-removing-implicit-semantic-authority.md) removes additional
implicit resolution and bundled knowledge defaults. Earlier default English automation is
intentionally unavailable without explicit semantic dependencies. Automatic semantic
resolution remains unfinished.

[40 — Evidence-driven interpretation](40-evidence-driven-interpretation.md) adds
bounded investigation of supplied candidate hypotheses through fresh plugin observations.
The agent retains the evidence and can select, defer, or withdraw a selection;
automatic text-turn integration requires an explicit hypothesis producer. This is
support relative to supplied models, not learned semantics or proof of user intent.
Source-bound grounding proposals preserve alternative identity bindings without
selecting them automatically. Autonomous hypothesis production, continuous evidence
freshness, and propagation of revisions to dependent tasks remain open work.

[64 — Active grounding investigation](64-active-grounding-investigation.md)
develops investigation over learned relational grounding queries: forecast
disagreement on caller-offered scenes, collect explicit teacher alignment, and
refit with retained evidence and renewed model admission. Supplied scenes and
teaching remain distinct from inferred vision or conversational understanding.

[65 — Conflicting scene evidence](65-conflicting-scene-evidence.md) retains exact
opposite-polarity facts on query witnesses and prevents treating that evidence
as clean grounding support. The fix concerns explicit contradictions; general
missing-fact and open-world reasoning remains unfinished.

[66 — Open-world grounding alternatives](66-open-world-grounding-alternatives.md)
retains per-root evidence and unseen-referent possibilities beside supported
bindings. Computation completeness does not become a claim of world knowledge.

[67 — Targeted grounding observations](67-targeted-grounding-observations.md)
develops learned-query-derived observation requests and retained counterfactuals
for known unknown roots. Observer semantics and subsequent selections remain
supplied; this is not autonomous visual understanding.

[68 — Relational witness observations](68-relational-witness-observations.md)
develops questions about existing related identities from connected partial
query witnesses, preserving evidence and existential uncertainty.

[69 — Neutral evidence graphs](69-neutral-evidence-graphs.md) separates shared
relational reasoning from image-only evidence and connects literal CDP document
snapshots. DOM structure is not pixel understanding or persistent identity.

[Retiring pixel semantic rules](70-retiring-pixel-semantic-rules.md) removes the
legacy geometry/control/prompt interpretation fallback and pixel runners. OCR
measurements and learned icon association remain; learned holistic pixel scene
construction remains unfinished.

[71 — Grounded browser actions](71-grounded-browser-actions.md) develops explicit
programmatic activation of authenticated captured DOM nodes selected through
learned grounding, with no selector or parent-climbing fallback.

[72 — Browser transition learning](72-browser-transition-learning.md) develops
contextual target-relative predictions, document-disjoint validation, explicit
supported residual rules, and suspension from actual browser counterexamples.

[73 — Learned document tasks](73-learned-document-tasks.md) develops guarded
browser pursuit from explicit measured goals, learned grounding and transition
predictions, actual feedback, and retained task outcomes.

[74 — Learned measured-action goals](74-learned-measured-action-goals.md) develops
reference-substituted intent learning, explicit goal selection, and authenticated
runtime materialization before measured document-task pursuit.

[75 — Learned communicative interpretations](75-learned-communicative-interpretations.md)
replaces implicit act-assignment shortcuts with neutral frame evidence and
explicitly taught, admitted communicative alternatives.

[76 — Learned informing correspondences](76-learned-informing-correspondences.md)
develops taught full-question mappings to informing calls and answer queries,
replacing role-order guesses while retaining qualifiers and explicit selection.

[77 — Learned store querying](77-learned-store-querying.md) develops strict
full-question store plans, explicit allowed scopes and selection, and retained
support evidence in place of passive lookup guesses.

[78 — Authenticated derivations](78-authenticated-derivations.md) develops
versioned operator admission, exact replay receipts, and bounded recursive
support for derived answers, including quantity scan membership.

[79 — Explicit measurements and calculations](79-explicit-measurements-and-calculations.md)
develops identified measurements, explicit ordered operations and selection, and
authenticated replay without automatic quantity interpretation.

[80 — Literal units and supported conversions](80-literal-units-and-supported-conversions.md)
develops immutable symbol algebra and selected conversion evidence, retiring
heuristic unit interpretation and the unused relation reader.

[81 — Contextual task correction](81-contextual-task-correction.md) implements
learned revision from complete correction evidence and the prior goal, with guarded
adoption and symbolic filesystem realization. Real-input tests exercise changed
execution and withdrawal; task association and resource grounding remain supplied.

[82 — Evidence-driven task association](82-evidence-driven-task-association.md)
specifies the next sentence-level routing milestone: learned candidate relations,
retained rivals, and positive evidence for new-task intent. It is a design, not an
implemented capability.

Read implementation reports separately from architectural proposals and dated
benchmark results. Do not count configuration files, record types, or test totals as evidence
that the agent learned a concept.

Development for this work stays on `main`, as requested by the repository owner. Preserve
unrelated work and use reviewable commits rather than deleting history to obtain a clean start.
The starting test run was `.venv/bin/python -m pytest -q`: **1,245 passed, 5 skipped** in
86.28 seconds. This is a test baseline, not a new cognitive-performance measurement.

## Documents

- [Where `tensacode-py` actually stands](01-assessment.md)
- [Representation: coarser than TCIR, explicit where it matters](02-representation.md)
- [Operation algebra and backend protocol](03-operations-and-backends.md)
- [Example cognitive programs and agents](04-examples.md)
- [Evaluation: design, measured results, limitations](05-evaluation.md)
- [Modernization plan](06-plan.md)
- [Pixels → scene graph: vision perception for real desktops](07-vision-perception.md)
- [Long-horizon work: what useful agent work looks like, and how we test for it](08-long-horizon.md)
- [Symbolic language and induction](09-language-and-induction.md)
- [Computer-using agents on the computerworld engine](10-computerworld.md)
- [Evidence audit: who wrote the environment, and who graded the answer](11-evidence-audit.md)
- [Open-domain benchmarks: what the cognition does off its home turf](12-open-domain.md)
- [Where the cognitive schemas break: a failure taxonomy](13-schema-brittleness.md)
- [A cognitive architecture over the claim store](14-cognitive-architecture.md)
- [The learned language tier](15-learned-language-tier.md)
- [A learned tier for perception and action](16-learned-perception-and-policy.md)
- [Cognitive structures: quantity, contingency, cause, probability, time](17-cognitive-structures.md)
- [How we measure general cognitive performance](18-cognitive-profile.md)
- [The decision-layer landscape, and where tensacode fits](19-decision-layer-landscape.md)
- [A decision layer in a normal request path](20-decision-layer-examples.md)
- [Schema induction from clustered failures](21-schema-induction.md)
- [Control and volition](22-control-and-volition.md)
- [Metacognition and the self-model](23-metacognition.md)
- [Social cognition and pragmatics](24-social-cognition.md)
- [Perception over time and memory dynamics](25-perception-over-time.md)
- [Evidence selection, expected answer type, and eval hygiene](26-selection-and-answer-type.md)
- [Comparison as an operation: relations over two values](27-comparison-and-relations.md)
- [A general agent built from parsing and perception, with plugins](28-general-agent.md) (proposal)
- [Architecture review: what to realign, repair, remove](29-architecture-review.md)
- [What the library is made of](30-what-the-library-is-made-of.md)
- [First full scorecard](31-first-full-scorecard.md)
- [Knowledge written as code](32-knowledge-written-as-code.md)
- [The cognitive fronts](33-the-cognitive-fronts.md) (historical probes and dated reassessment)
- [Cognitive primitives and implementation](34-cognitive-primitives-and-implementation.md) (design proposal, acceptance criteria, and first slice)
- [Model-based planning and inspectable refinement](35-model-based-planning.md) (implemented paths, provenance, and remaining boundaries)
- [The structured cognitive workspace](36-structured-cognitive-workspace.md) (governing objective, current gaps, implementation sequence, and acceptance criteria)
- [Interpretation workspace: first implementation slice](37-interpretation-workspace-first-slice.md) (source and candidate retention, explicit decisions, and remaining limits)
- [Scene interpretations in the shared workspace](38-scene-interpretations.md) (source-bound relational proposals, active integration, and supplied-versus-learned limits)
- [Removing implicit semantic authority](39-removing-implicit-semantic-authority.md) (removed default knowledge and interpretation shortcuts, breaking behavior, remaining audit)
- [Evidence-driven interpretation](40-evidence-driven-interpretation.md) (fresh observations, supplied hypothesis discrimination, explicit grounding proposals, and remaining integration boundaries)
- [Chat workspace and independent connections](41-chat-workspace-and-connections.md) (durable UI/CLI conversations, uploaded evidence, and browser/Gym adapters)
- [Explicit grounded identity](42-explicit-grounded-identity.md) (occurrence bindings, scoped retrieval, and unresolved reporting choices)
- [Causal outcome uncertainty](43-causal-outcome-uncertainty.md) (variable intervention outcomes and retained paired evidence)
- [Action observation evidence](44-action-observation-evidence.md) (active browser/Gym evidence, paired action attempts, and explicit acquisition failures)
- [Evidence-backed transition learning](45-evidence-backed-transition-learning.md) (induced rules, separated validation attempts, and supported predictions)
- [Grounded goals without lexical reinterpretation](46-grounded-goals-without-lexical-reinterpretation.md) (exact role/value preservation and removal of implicit request resolvers)
- [Learned interpretation alternatives](47-learned-interpretation-alternatives.md) (learned syntax proposals, revisable preposition roles, removed tree repairs, and measured candidate recall)

- [Planning from learned transitions](48-planning-from-learned-transitions.md) (one-step learned action comparison, guarded execution, observed verification, and counterexample suspension)
- [Source-faithful language evaluation](49-source-faithful-language-evaluation.md) (original typography, validated character spans, partial attachment credit, and whole-candidate oracle limits)
- [Quotation boundaries and source evidence](50-quotation-boundaries-and-source-evidence.md) (structural quotation envelopes, retained malformed and empty inputs, and explicit authored conventions)
- [Learned source segmentation](51-learned-source-segmentation.md) (trained token boundaries, source-aligned alternatives, required model artifacts, and shared decoding budgets)
- [Global interpretation retention](52-global-interpretation-retention.md) (syntax diversity before semantic variants, lazy projection, explicit pending work, and measured search costs)
- [Resumable workspace interpretation](53-resumable-workspace-interpretation.md) (across-turn semantic expansion, stable evidence, and stale-decision invalidation without action replay)
- [Investigation with pending interpretations](54-investigation-with-pending-interpretations.md) (bounded expansion before supplied hypotheses, pending-work deferral, and generation-aware decision guards)
- [Empirical model investigation](55-empirical-model-investigation.md) (learned same-probe forecasts, actual interventions, and explicit model-applicability assessments)
- [Contingent planning from experience](56-contingent-planning-from-experience.md) (multi-step routes through all supported empirical outcomes, fresh step execution, and observed counterexamples)
- [Revision-bound empirical tasks](57-revision-bound-empirical-tasks.md) (bounded fresh replanning, task resumption without replay, and receipts attributed to their original goal revision)
- [Interpretation-dependent tasks](58-interpretation-dependent-tasks.md) (selected-meaning authorization, stale comparison detection, and renewed commitments without replay)
- [Semantic preservation at the goal boundary](59-semantic-preservation-at-the-goal-boundary.md) (rejection of discarded polarity, roles, and entity qualifications before action)
- [Lexical goal interpretations](60-lexical-goal-interpretations.md) (retained lexical and role-binding alternatives, explicit goal selection, and bounded-search deferral)
- [Deferred goal adoption](61-deferred-goal-adoption.md) (later explicit goal commitments, retained task identity, and separation of adoption from execution)
- [Learning goal correspondences](62-learning-goal-correspondences.md) (retained supervision, bounded reference-substitution induction, and explicit model admission)

> **2026-09-18: the Seed simulator is gone.** The desktop environment is now only the
> computerworld engine, which runs in-process. Scripts that could only run against Seed were deleted, along
> with the three result files only they produced (`change_live`, `memory_live`,
> `permanence_live`): `eval/temporal_perception/*_live.py`, `paraphrase_recall.py`, `live_harness.py`,
> and the live probe harness in `eval/profile/`. Git history has them. The notes below that cite them are
> dated records and are not rewritten.

## Findings that should change decisions

| Finding | Evidence |
| --- | --- |
| Local zero-shot Qwen3-8B escalation *lowered* selective accuracy (94.1% → 91.0%) and was right on only 38.2% of the items it received | [05 §6.1](05-evaluation.md#61-classify-cascade-on-banking77) |
| Legacy TCIR used 44 nodes for two small tickets, lost all data when serialized (`{"items":[{},{}]}`), and could not reconstruct types; the proposed records use 3 records and round-trip exactly | [02 §2.3](02-representation.md#23-measured-legacy-behavior) |
| Recovery logic based on explicit facts had 0 duplicate money movements in 5,000 simulated episodes, vs 4.8% for naive retry and 12.7% for verify-then-retry. With wrong facts, 0.9%. | [05 §6.3](05-evaluation.md#63-recovery-trajectories) |
| TensaCode routing overhead is ~19 µs per call; batching is 6.5× faster than per-item calls with identical results | [05 §6.1](05-evaluation.md#61-classify-cascade-on-banking77) |
| No LICENSE file had ever been committed and the package was not on PyPI (MIT `LICENSE` added for 0.1.0a1) | [01 §1.4](01-assessment.md#14-packaging-dependencies-licensing) |

## Code

- [the repository](../..): the `tensacode` package in `src/`, the examples, tests, evaluation scripts and results, and research code. Start with the [top-level README](../../README.md).

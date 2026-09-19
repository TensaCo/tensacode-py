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

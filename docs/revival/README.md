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

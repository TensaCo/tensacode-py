# TensaCode revival: assessment, design, and proof

*2026-09-16. Scope: `TensaCo/tensacode-py`. Status: proposal. Nothing in the legacy package was modified.*

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

1. [Assessment and keep/simplify/replace/remove/defer inventory](01-assessment.md)
2. [Representation: values + entities + claims, with before/after measurements](02-representation.md)
3. [Operation algebra, legacy-op mapping, and backend protocol](03-operations-and-backends.md)
4. [Example programs and agents, with outputs and traces](04-examples.md)
5. [CommandAGI integration boundary](05-commandagi-integration.md)
6. [Evaluation design, measured results, and limitations](05-evaluation.md)
7. [Prioritized plan and smallest vertical slice](06-plan.md)

## Findings that should change decisions

| Finding | Evidence |
| --- | --- |
| Local zero-shot Qwen3-8B escalation *lowered* selective accuracy (94.1% → 91.0%) and was right on only 38.2% of the items it received | [05 §6.1](05-evaluation.md#61-classify-cascade-on-banking77) |
| Legacy TCIR used 44 nodes for two small tickets, lost all data when serialized (`{"items":[{},{}]}`), and could not reconstruct types; the proposed records use 3 records and round-trip exactly | [02 §2.3](02-representation.md#23-measured-legacy-behavior) |
| Recovery logic based on explicit facts had 0 duplicate money movements in 5,000 simulated episodes, vs 4.8% for naive retry and 12.7% for verify-then-retry. With wrong facts, 0.9%. | [05 §6.3](05-evaluation.md#63-recovery-trajectories) |
| TensaCode routing overhead is ~19 µs per call; batching is 6.5× faster than per-item calls with identical results | [05 §6.1](05-evaluation.md#61-classify-cascade-on-banking77) |
| CommandAGI already has the right seam (`GoalCognitionModel.adapter`) and a Python `/v1/transduce` service with the same cascade intent | [05](05-commandagi-integration.md) |
| No LICENSE file has ever been committed; the package is not on PyPI | [01 §1.4](01-assessment.md#14-packaging-dependencies-licensing) |

## Code

- [the repository](../../proposal): the prototype package `tensacode`, the examples, tests, evaluation scripts and results, and the legacy TCIR probe. Start with [`proposal/README.md`](../../README.md).

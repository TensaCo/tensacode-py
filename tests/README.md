# Test suite

Tests are grouped by the public subsystem or supported composition they verify:

- `core/`: shared operation behavior and asynchronous invocation.
- `tracing/`: capture, dependency, replay, and mutation semantics.
- `training/`: persisted experiences, trainers, and checkpoints.
- `llm/`: messages, structured operations, models, and HTTP adapters.
- `vec/`: vector representations, candidate operations, image encoding, and native learning paths.
- `graph/`: graph representation, symbolic operations, and neural adapters.
- `tools/`: decision, memory, chatbot, and action-loop compositions.
- `examples/`: application routing, citations, file boundaries, action limits, image modes, and evaluation-data parsing.
- `integration/`: behavior spanning several public subsystems and regression coverage.

Run everything from the repository root with `.venv/bin/pytest -q`.

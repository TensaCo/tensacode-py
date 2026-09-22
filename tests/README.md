# Test suite

Tests are grouped by the subsystem or supported composition they verify:

- `core/`: shared operation behavior and asynchronous invocation.
- `tracing/`: capture, dependency, replay, and mutation semantics.
- `training/`: persisted experiences, trainers, and checkpoints.
- `text/`: messages, structured operations, models, and HTTP adapters.
- `vec/`: vector representations, candidate operations, image encoding, and native learning paths.
- `graph/`: graph representation and explicit unimplemented symbolic operation contracts.
- `runtime/`: private session, memory, evidence-state and execution mechanisms, plus public composition behavior. This test directory does not imply a public runtime module.
- `models/`: pretrained artifact lifecycle, workspace and owned model contracts.
- `examples/`: application routing, citations, file boundaries, action limits, image modes, evaluation-data parsing, supervised learning and fresh-process weight restoration.
- `integration/`: behavior spanning several public subsystems and regression coverage.

Run everything from the repository root with `.venv/bin/pytest -q`.

Install test dependencies with `python -m pip install -e '.[tools,dev]'`. Tests use small local models and authored fixtures; passing them does not establish pretrained real-world competence. See [validation](../docs/validation.md) for model measurements.

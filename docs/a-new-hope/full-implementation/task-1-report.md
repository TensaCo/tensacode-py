# Task 1 — Durable supervised experience and trainers

Implemented a versioned JSON experience format, explicit supervision, replay-based tensor training, and tensor-module checkpoints. Files contain data and named operation configuration fingerprints, never pickle objects, executable imports, or serialized callbacks. Loading requires already-created caller bindings. Configuration is checked independently of mutable model weights, so replay uses the supplied current parameters.

## Public API

```python
from tensorcode import trace, training

with trace() as session:
    prediction = head(encoder(text))
session.supervise(prediction, "target label", loss="cross_entropy", source="human:review-42")
session.save("experience.json", operations={"encoder": encoder, "head": head}, release=True)

# In a new process, explicitly construct compatible operations first.
experience = training.load("experience.json", operations={"encoder": encoder, "head": head})
trainer = training.Trainer({"encoder": encoder, "head": head}, lr=0.01)
losses = trainer.fit([experience], epochs=10)
training.save_checkpoint("model.json", operations={"encoder": encoder, "head": head}, optimizer=trainer.optimizer)
training.load_checkpoint("model.json", operations={"encoder": encoder, "head": head}, optimizer=trainer.optimizer)
```

- `Session.supervise(output_or_ref, target, *, loss='cross_entropy', source='human') -> Supervision`. The immutable record exposes `output`, snapshotted `target`, `loss`, and `source`; records are available at `session.supervisions`. Nonempty source provenance is required. Scalar targets in the trace still require explicit output handles.
- `Session.save(path, *, operations, codecs=None, release=False)`. Writes atomically after validation. `operations` maps stable names to the exact captured operation instances. Failed calls, missing bindings, unsupported payloads, and mutated intermediates are rejected.
- `training.load(path, *, operations, codecs=None) -> Session`. Rejects incompatible or missing bindings, unknown artifact versions, duplicate JSON keys, malformed DAG references, forward dependencies, and unknown codecs. Returned sessions retain roots, DAG, supervision, and recorded external boundaries.
- `Session.release()`. Drops captured outputs, live autograd graphs, object lookup records, and mutation stamps. It retains detached external boundary snapshots, roots, dependencies, and supervision. Save handles or use `supervision.output` before release; released objects cannot be looked up by identity.
- `Session.replay(target, *, inputs=None, boundary='error')`. Pure replay is the default. `boundary='recorded'` explicitly permits detached saved external results without invoking the external operation. Replacement roots are rejected across recorded boundaries.
- `Trainer(operations, *, optimizer=None, lr=0.01, losses=None)`. Uses SGD by default. Supply an optimizer instance or a factory accepting the deduplicated parameter list. Optimizer ownership must exactly match the bound trainable parameters and cannot contain duplicates.
- `Trainer.step(session) -> float`; `Trainer.fit(sessions, *, epochs=1) -> list[float]`. Each step averages that experience's explicit losses and performs one optimizer update. A fit result contains one scalar per session update. Cross-entropy accepts integer indices or `Prediction` label strings; MSE requires exactly matching tensor shapes. Custom in-process loss functions are supplied as `losses={'name': callback}` and are never serialized. Nondifferentiable, nonfinite, or parameter-disconnected losses are rejected.
- `training.save_checkpoint(path, *, operations, optimizer=None)` and `training.load_checkpoint(path, *, operations, optimizer=None)`. Restore supported module state dictionaries and optional SGD, Adam, or AdamW optimizer state. Other optimizer checkpoint types are rejected explicitly. Configurations, tensor shape/dtype, shared-parameter alias topology, contradictory shared values, optimizer type and parameter layout, optimizer slot shapes/dtypes, and Adam step counters are validated before mutating bound state. Failed module-state restoration rolls back previously restored states.
- `tracing.invoke_async(operation, value, context, forward)` and `Session.capture_async(...)` capture awaited boundaries with the same dependency/snapshot rules as synchronous calls. Exceptions and cancellation leave failed records. Pending calls cannot be referenced, released, or saved until awaited; inherited task contexts cannot start new calls after the session closes.

## Configuration and codecs

Operations may expose `configuration() -> JSON-safe data`; this is the canonical stable semantic configuration contract. The fingerprint also includes the qualified operation type, tensor-state shapes/dtypes, and replay capability. Otherwise a conservative fallback describes public scalar/container attributes and tensor module topology. Callables and opaque objects in fallback configuration require explicit metadata via `configuration()`; they are not persisted implicitly. The caller remains responsible for the truthfulness and completeness of custom configuration declarations.

Primitive values, bytes, tuples/lists, mappings (including immutable mapping proxies), and finite dense real tensors have explicit codecs. Tensors are stored as dtype/shape/data and restored on CPU. Application dataclasses require an explicit allowlist on both sides, for example `codecs={'number-v1': Number}`. No artifact-provided class name is imported. Registered dataclass constructors and operation bindings are trusted caller code, not an artifact sandbox.

The same optional dataclass allowlist handles context, external roots, supervision targets, boundary outputs, and dataclass dependency trees. Pure intermediate dataclasses do not need a codec when they are reconstructible operation outputs and are not themselves stored as external data.

## Verification

Initial new tests failed at import because the training package did not exist. After implementation, the focused combined suite passed:

```text
.venv/bin/python -m pytest tests/test_persistence.py tests/test_training.py tests/test_tracing.py tests/test_learning.py tests/test_review_regressions.py tests/test_vec_configuration.py -q
44 passed in 3.01s
```

Tests cover a fresh subprocess loading an experience and producing real gradients/weight changes; held-out loss improvement; dataclass/context roundtrip; release/replay; missing and changed bindings; changed weights accepted; safe codec rejection and duplicate JSON; malformed DAGs; no external-effect replay; atomic save rejection for mutated outputs; deduplicated shared parameters with an analytically checked single update; momentum checkpoint restore; malformed optimizer slot rejection before state mutation; alias mismatch rejection; custom losses; invalid optimizers/nonfinite loss rejection; async success/failure capture and pending-call safeguards; immutable graph/message dataclass roundtrip; atomic release rejection of mutated outputs; and `python -S` imports without loading torch. Existing tracing, learning, and regression tests in that command remain green.

The latest whole-suite run during concurrent implementation reached 146 passing tests and four failures in graph dtype/source-anchor and vector candidate-mask regressions outside this task ownership. Earlier whole-suite runs crossed incomplete worker edits. Final whole-suite verification is coordinated by the parent after workers finish. Independent review identified and led to regression-tested fixes for release-time mutation validation, pending async call persistence, and malformed optimizer slots.

The independently owned Banking77 restart example has exercised capture in one process, reload/train in a second, and checkpoint evaluation in a third. Its authoritative command, data hashes, measurements, and limitations belong to its separate evaluation report.

## Boundaries and remaining limits

This is native tensor gradient training, not differentiation through arbitrary Python, remote models, or external actions. External outputs are detached constants and their upstream computations receive no downstream gradient. No optimizer policy or custom loss callback is inferred. Multiple supervised outputs are replayed separately within a step, so stochastic operations can yield separate samples. Tensors restore on CPU; applications using another device must explicitly adapt loaded roots and targets. Sparse/complex/nonfinite tensor payloads and arbitrary Python objects are unsupported. Checkpoints cover supplied modules and optimizer state, not random-number generator state or training scheduler state. Custom configuration contracts cannot prove that arbitrary user implementations have unchanged semantics. Artifact validation prevents code deserialization, but does not promise resource isolation for untrusted, extremely large JSON inputs.

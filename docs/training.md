# Tracing and training

Tools own their parameters and expose stable operation bindings. `ToolTrainer`
collects explicit feedback and replays supported local objectives with gradients.
Install `tensorcode[tools]` for owned models. See the [quickstart](quickstart.md)
for a complete runnable example.

## Train an owned tool

```python
from tensorcode import training

# Construct or load model first; all parameters must already exist.
trainer = training.ToolTrainer(model, lr=0.001)
experience = trainer.capture(inputs, targets, source="review:42")
experience.save("experience.json", operations=trainer.operations, release=True)

experience = training.load("experience.json", operations=trainer.operations)
losses = trainer.fit([experience], epochs=10)
model.save_pretrained("./model")
trainer.save_checkpoint("./training", progress={"next_example": 43})
```

`ToolTrainer(tool, *, optimizer=None, lr=0.001)` uses the tool's declared training
operation and objective. Supply an optimizer or parameter factory to override
SGD. `capture` performs no optimizer step; it stores snapshotted inputs and
explicit sourced targets. `step` returns a loss; `fit` returns one per update.

Investigator and Decision targets identify a supplied hypothesis by ID or index,
or provide a finite nonnegative distribution summing to one. Scene targets
identify a supplied visual description by ID or index in ranking mode; Scene
language mode instead accepts reviewed target text.
Planner targets contain `candidate_id` and the observed numeric `outcome`, or one
observed outcome per candidate. Chatbot inputs and targets are equal-length text
lists; teacher-forced cross-entropy supervises the decoder. Feedback never becomes
input evidence merely because it shares an objective envelope.

## Resume training

```python
model = ModelClass.from_pretrained("./model")
trainer = training.ToolTrainer(model, lr=0.001)
progress = trainer.load_checkpoint("./training")
experience = training.load("experience.json", operations=trainer.operations)
trainer.fit([experience], epochs=1)
```

Replace `ModelClass` with the same concrete tool class used to save the model.
Reconstruct the same optimizer type and parameter groups when using a custom
optimizer. `load_checkpoint` restores model and optimizer state, `trainer.steps`,
Python RNG, PyTorch CPU RNG and saved CUDA RNG; it returns your progress mapping.
Record data cursors in `progress`. NumPy RNG, schedulers and external data-loader
state are not automatically captured. Exact stochastic continuation also requires
a compatible device topology and execution environment.

`training.json` selects separate checksummed safetensors holding model, optimizer
and RNG tensors; keep those files together when moving a training checkpoint.
Module training/evaluation modes are also restored. This is a separate resumable
artifact. `save_pretrained` exports only
the model. Experiences and chatbot sessions are saved separately. Restoring model
weights does not restore an optimizer or a conversation.

## Compose and trace individual operations

Tracing records boundaries and dependencies independently of tools or harnesses.
It preserves native tensor gradients, snapshots external inputs and distinguishes
feedback from predictions. Plain Python transformations, remote services and
discrete choices do not become differentiable because they were observed.

```python
from tensorcode import trace, training

with trace() as session:
    prediction = head(encoder(text))
session.supervise(prediction, "target label", source="human:review-42")
operations = {"encoder": encoder, "head": head}
session.save("experience.json", operations=operations, release=True)
experience = training.load("experience.json", operations=operations)
trainer = training.Trainer(operations, lr=0.01)
losses = trainer.fit([experience], epochs=10)
```

The [hypothesis](../examples/hypothesis_learning.py),
[plan](../examples/plan_learning.py), and
[Banking77](../examples/banking77_restart.py) programs illustrate direct vector
operation composition with upfront initialization and restoration across processes.

## Capture and supervision

`session.ref(output)` returns an `OutputRef`. Equal-valued independent outputs are not merged. Scalars and ambiguous aliases need explicit `session.calls[index].output` handles. `session.example(target)` extracts the dependency closure and its external roots; it does not release memory by itself. Context, supported dataclass fields and container elements retain dependencies.

`session.supervise(output_or_ref, target, *, loss='cross_entropy', source='human')` stores a `Supervision(output, target, loss, source)` with snapshotted target data. `session.supervisions` exposes the records. Source must be a nonempty provenance string; feedback is never inferred from the output itself.

Mutation of captured intermediates is rejected before reuse, save or release. Unsupported mutation through `.data`, external storage aliases or native code can evade tensor version counters. Inference-mode tensors use a conservative content stamp, which can require a device copy/synchronization.

Awaited calls retain normal tracing semantics. Failed or cancelled calls remain failures. Await pending work before referencing, releasing or saving its outputs. Tasks retaining a session context cannot start new captures after the session closes.

## Portable experience and replay

`session.save(path, *, operations, codecs=None, release=False)` atomically writes versioned JSON after validation. Bind stable names to the exact captured operation instances. Failed calls, missing bindings, malformed/unsupported payloads and mutated outputs are rejected.

`training.load(path, *, operations, codecs=None)` returns a session using already-constructed supplied operations. It validates operation configuration fingerprints, required bindings, artifact version, DAG references and codec tags. It never imports artifact-named classes, unpickles code or restores callbacks.

`session.replay(target, *, inputs=None, boundary='error')` recomputes pure operations with current parameters. External effects are rejected by default. Explicit `boundary='recorded'` uses detached captured external results without invoking them. Replacement inputs cannot cross a recorded boundary. Training uses recorded boundaries as constants; gradients do not cross them.

`session.release()` drops live results, autograd graphs, object lookup records and mutation stamps after validating the whole session. External roots, DAG, supervision and detached external boundary results remain. Preserve output handles first, or use `supervision.output`; identity lookup of released outputs is unavailable. `save(..., release=True)` persists before releasing.

## Configuration and codecs

Operations expose `configuration() -> JSON-safe data` for stable semantic choices. Fingerprints also cover qualified operation type, replay capability and tensor-state shape/dtype. Vector configurations additionally include spaces, label/vocabulary order, relevant module behavior, parameter/buffer structure and gradient flags, excluding learned values. Updated weights can therefore be used intentionally when loading compatible experience.

Explicit custom module configuration is authoritative. Conservative fallback accepts supported JSON behavior fields and rejects opaque behavior state. Ambiguous callbacks, closures or stateful callables need explicit configuration metadata. These declarations cannot prove that arbitrary application code retained its semantics.

Built-in codecs cover primitive values, bytes, lists/tuples, mappings and finite dense real tensors. Tensors restore on CPU. Application dataclasses require an explicit allowlist on both save and load, such as `codecs={'record-v1': Record}`. This covers roots, context, targets, boundary values and dataclass dependency trees. Reconstructible pure intermediate outputs need no stored codec. Immutable mapping proxies become data mappings, which registered dataclass constructors can refreeze.

Registered classes and operation bindings are trusted application code. JSON validation avoids executable deserialization; it is not resource isolation for extremely large untrusted artifacts. Sparse/complex/nonfinite tensor payloads and arbitrary opaque objects are unsupported. Applications targeting another device must explicitly adapt restored inputs/targets.

## Optimizers, losses and checkpoints

`Trainer(operations, *, optimizer=None, lr=0.01, losses=None)` defaults to SGD. Supply an optimizer instance or a factory accepting the deduplicated trainable parameters. Optimizer ownership must match those parameters exactly, without duplicate shared parameters.

`trainer.step(session)` averages the experience's explicit losses, performs one optimizer update and returns a float. `trainer.fit(sessions, *, epochs=1)` returns one loss per session update. Cross-entropy accepts integer indices or prediction label strings; MSE requires exactly matching shapes. Register custom in-process losses with `losses={'name': callback}`; callbacks are never serialized. Nondifferentiable, parameter-disconnected and nonfinite losses/gradients are rejected. Multiple supervised outputs replay separately, so stochastic operations can produce separate samples within a step.

`save_checkpoint(path, *, operations, optimizer=None)` and `load_checkpoint(...)` handle supported module state and optional SGD/Adam/AdamW optimizer state. They validate configurations, state keys, tensor shapes/dtypes, shared parameter aliases and values, optimizer ownership/layout, slot shapes and step counters before applying state. Failed restoration rolls back earlier restored state. Other optimizer checkpoint types are rejected. RNG and scheduler state are not included.

Importing `tensorcode.training` does not load torch. Tensor decoding, training and tensor checkpoints require the `vec` extra.

Large tool checkpoints store native tensors in safetensors rather than JSON arrays. Each save atomically switches `training.json` to a checksummed immutable `tensors-<uuid>.safetensors` generation. Previous generations remain available; remove an obsolete checkpoint directory to reclaim them. Model, optimizer, module train/eval modes, and Python/torch random state are restored together.

## Calibrate verifier scores

```python
from tensorcode.training.calibration import (
    TemperatureCalibration, evaluate_calibration, fit_threshold,
)

# held_out_logits: [examples, classes]; labels: integer class indices.
calibration = TemperatureCalibration()
report = calibration.fit(held_out_logits, labels)
calibrated_logits = calibration(test_logits)
metrics = evaluate_calibration(calibrated_logits, test_labels)
```

For an owned Investigator verifier, fit `model.verifier.calibration` so its
buffers persist with `save_pretrained`. `model.verification_loss(pairs, labels)`
trains source-wise NLI from `premise`/`hypothesis` pairs and explicit named labels;
`model.proposal_loss(inputs, targets)` trains reviewed proposal generation. Investigator selects the objective explicitly through a mode envelope:

```python
from tensorcode import training
from tensorcode.tools.investigator import Investigator

model = Investigator.from_pretrained("./investigator-with-verifier")
trainer = training.ToolTrainer(model, lr=0.0001)
experience = trainer.capture({
    "mode": "verification",
    "inputs": [{"premise": "The connection was refused.",
                "hypothesis": "The connection succeeded."}],
}, ["contradiction"], source="authored-example:review-17")
experience.save("verification.json", operations=trainer.operations, release=True)
experience = training.load("verification.json", operations=trainer.operations)
trainer.fit([experience], epochs=1)
model.save_pretrained("./updated-investigator")
trainer.save_checkpoint("./verification-training", progress={"next_review": 18})
restored = Investigator.from_pretrained("./updated-investigator")
```

This single authored pair illustrates the lifecycle, not a sufficient training
set. Modes are `verification`, `proposal` and `rank`; ordinary ranking inputs
without an envelope retain the ranking objective. Proposal feedback is reviewed
text, verification feedback is named NLI labels, and ranking feedback identifies
supplied alternatives. Changing verifier weights invalidates its calibration;
refit on a separate held-out set before reporting calibrated scores.

`fit_threshold(confidence_scores, correctness_labels, max_error=...)` returns an
empirical threshold/coverage/error report. Scores are finite confidence values;
correctness labels are explicit booleans. A `None` threshold means abstain on all.
Selection and reported error use the same calibration sample, so use separate test
data to evaluate the resulting policy. These utilities do not certify truth or
new-input error rates, and calibration should be repeated after weight updates.

## Train owned retrieval

An Investigator with `retrieval_encoder` configuration owns
`model.episodic_encoder`. Positive relationships belong in targets, never appended
to query/document text. The boolean matrix has shape `[queries, documents]` and
must include at least one positive per query:

```python
from tensorcode import training
from tensorcode.tools.investigator import Investigator

model = Investigator.from_pretrained("./investigator-with-retrieval")
trainer = training.ToolTrainer(model, lr=0.0001)
experience = trainer.capture({
    "mode": "retrieval",
    "inputs": {
        "queries": ["How was the service restored?"],
        "documents": ["Restoring the database connection recovered the service.",
                      "The maintenance window starts tomorrow."],
    },
}, [[True, False]], source="authored-example:retrieval-review-17")
experience.save("retrieval.json", operations=trainer.operations, release=True)
loaded = training.load("retrieval.json", operations=trainer.operations)
trainer.fit([loaded], epochs=1)
model.save_pretrained("./updated-retrieval-model")
trainer.save_checkpoint("./retrieval-training", progress={"next_review": 18})
restored = Investigator.from_pretrained("./updated-retrieval-model")
```

This authored pair demonstrates plumbing, not retrieval quality. Direct
`model.retrieval_loss({"queries": ..., "documents": ...}, positive_mask)` uses the
same contrastive objective. Multiple positives receive equal target weight.
Unmarked documents act as negatives, so review annotations accordingly. A frozen
retrieval foundation must be made trainable before updating it. Rebuild existing
episodic indexes after changing its weights; snapshots restore raw source records
and rebuild embeddings, while live indexes reject stale fingerprints.

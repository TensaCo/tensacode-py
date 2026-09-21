# 3 — Clean-start implementation report

*2026-09-20. First working replacement milestone; the full target remains [the architecture](1-tensorcode-architecture.md).*

## Preservation and cuts

The exact pre-reset source checkpoint is `716056ba2cbe66402f28dc764c66bdc2195f1c09`.
It includes the owner's original feedback and the reviewed architecture/plan.
It was committed and pushed to `origin/main` before deletion, with a clean worktree.

The private archive is [JacobFV/old-tensorcode-2026-09-20](https://github.com/JacobFV/old-tensorcode-2026-09-20).
Privacy and its main SHA were checked. All local branches and tags were copied;
locally known original-origin branches were also retained under `original-origin/`.
A verified full Git bundle and a worktree tarball, excluding `.git` and `.venv`,
were saved outside this checkout under `~/.cache/tensorcode-reset/`.
External dataset/model caches were left intact; they are not claimed to be in Git.

The old production code, tests, examples, evaluation framework, research projects,
placeholder distribution, revival/civilization documents and prior agent plans were
deleted, not moved into a legacy subtree. Old editor settings, Python version pin,
agent scratch directory, generated caches, lockfile, changelog and publishing
workflow were removed. The ignore file and CI were replaced. The local environment
was recreated for the new package. Git history and the private archive retain recovery.

The old baseline was intentionally interrupted before deletion after **1,056 passed,
1 skipped in 179.97 seconds**. This is not a full regression pass. The preceding
checkpoint already documented unresolved failures; this report makes no claim to
repair or preserve that agent's behavior.

## What is implemented

- `ops.Operation`: common callable boundary with explicit context and opt-in replay.
- `ops.vec`: native tensor transforms, a trainable text encoder and a classifier
  returning logits, probabilities and selected labels. Module hooks and gradients survive.
- `ops.llm`: immutable text messages, text encoding/decoding and a caller-supplied
  model function. These operations are usable without PyTorch.
- `ops.graph`: immutable source-tagged string graphs, neighbor lookup and supplied
  transforms. No hidden ontology or learned graph semantics.
- `tools.decision.Decision`: compose an encoder and decision operation.
- `tools.agents.Chatbot`: supplied-model text chat with isolated per-instance history,
  serialized turns, and successful-turn commit behavior.
- `trace()`: execution-local operation capture, identity-based producer references,
  explicit scalar output handles, context dependencies, failed-call records,
  external-input snapshots and extraction of an output's dependency closure.
- In-memory replay of an effect-free dependency closure with optional replaced roots.
  Replay uses current operation objects/parameters and recomputes intermediates.
- Ordinary PyTorch optimization works both on the live graph and on supported replay.

No compatibility API, global dispatch engine, semantic seed table or general agent
ontology was preserved.

## Real-data learning

Command, using existing copies of the official Banking77 CSVs:

```bash
python examples/banking77.py \
  --train /path/to/banking77_train.csv \
  --test /path/to/banking77_test.csv \
  --output /tmp/banking77-results.json
```

Full settings, input checksums and results are in [banking77-reset-results.json](banking77-reset-results.json).

| Measurement | Before training | After 20 fixed epochs |
|---|---:|---:|
| Held-out accuracy | 0.97% | 89.38% |
| Held-out cross-entropy | 4.3646 | 0.4628 |

The run used 9,997 training rows after excluding six normalized exact-text overlaps
with the 3,080-row official test split. It learned a 96-dimensional word embedding
and classifier for 77 labels. Vocabulary was constructed from training text only.
Seed, epoch count and learning rate were fixed before measuring; no test-driven
hyperparameter search was performed. A verification rerun after tracing fixes
reproduced the same accuracy and loss.

The tokenizer, pooling architecture, training labels and loss are supplied.
The embeddings and decision parameters are learned from the real training inputs.
Every training batch runs through the new operations and tracing boundary; the
final recorded batch is replayed through its captured dependencies with gradients.
This demonstrates supervised text classification and trace-connected learning.
It does not establish calibrated uncertainty, general reasoning, autonomous learning,
image understanding or real-world task execution.

## Self-critique and independent review

The architecture underwent four critique passes recorded in section 18. Implementation
review then focused on actual failure cases rather than checking the folder names:

1. Dataclass operands initially severed upstream trace edges. Field-aware dependency
   binding now preserves them; replay reaches the upstream encoder's gradients.
2. Replacing a tensor inside a container initially evaded mutation detection. Stamps
   now include tensor identity, and explicit output references also check mutation.
3. Repeated unchanged message children initially caused false alias failures across
   chatbot turns. Carried-through children retain their original producer.
4. Tracing initially copied primary input containers and changed mutation behavior.
   Reference unwrapping now preserves the original object when no replacement is needed.
5. Inference-mode tensors lack native version counters. The tracer uses a conservative
   content stamp for them; this can require a device copy/synchronization.
6. Graph sources now reject mutable/non-string payloads so their advertised immutability
   does not depend solely on a frozen outer dataclass.

An independent read-only reviewer reproduced the first three issues, verified their
fixes and found the inference-mode issue. Each consequential fix has a regression
that failed before implementation. Shared-backbone registration and gradient flow
are also checked. No tests infer cognitive capability from the supplied model fixtures.

## Verification

- Full replacement suite: **29 passed** on the fresh Python 3.13 environment.
- Earlier suite runs also passed under Python 3.12 before environment recreation.
- Wheel and source distribution build successfully.
- Installed wheel exercised outside the repository in a fresh environment with no
  optional dependencies: core imports, message operations, graph operations and a
  fixture-backed chatbot execute without importing PyTorch.
- CI now covers Python 3.11–3.13 and a separate no-extras wheel installation.
  Local checks are distinct from the eventual hosted CI result.
- Local Markdown links and distribution contents were checked before commit.

## Deliberate remaining boundaries

This is a small usable foundation, not completion of every feature in the architecture.
There is no generic trainer, `episode.supervise`, persisted trace format, cross-process
replay, provider SDK integration, image encoder, learned objective revision, long-term
agent memory, or autonomous action loop in this milestone.

Tracing observes operation boundaries, not arbitrary Python transformations. Dataclasses,
plain Python containers and tensors are supported; arbitrary object graphs/cycles and
external mutable resources are not. Scalar lineage needs explicit handles. Replay
requires the session and original operation objects. Model/configuration history is
not a durable checkpoint system. Ordinary tensor version counters do not detect
unsafe mutations made through `.data`, external storage aliases or native code;
those mutation paths are unsupported. Vector adapters assume their supplied modules
are effect-free; do not enable replay for modules that perform external effects.

The live outputs retained by a session consume memory. Extracting `example()` does
not by itself free those outputs or create a standalone portable training program.
Inference-mode content checks trade efficiency for conservative mutation detection.

The chatbot was tested with explicitly supplied model fixtures, not a live provider.
Its availability is an integration capability, not a claim of newly learned conversation.

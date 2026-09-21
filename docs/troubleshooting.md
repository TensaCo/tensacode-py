# Troubleshooting

| Symptom | What to check |
|---|---|
| Missing torch or Transformers | Install `tensorcode[tools]` for owned models, `tensorcode[vec]` for tensor operations, or `tensorcode[local]` for external multimodal integration. |
| `from_pretrained` rejects a manifest | Use a TensorCode artifact saved by the same concrete class and supported format version. An ordinary Transformers checkpoint is not a complete TensorCode tool. |
| Fresh model produces poor outputs | Construction initializes weights. `from_foundation` also introduces an untrained workspace. Load a measured compatible checkpoint or train on explicit feedback. |
| Offline Hub loading cannot find files | Cache the pinned snapshot first or pass an existing local model directory. Constructors do not download assets. |
| Graph operation raises `NotImplementedError` | Symbolic graph operations are declared interfaces only. They do not currently encode, decode, reason or train. |
| Training resume rejects RNG topology | Restore on compatible CUDA devices or start a new training run from the model artifact; training checkpoints are stricter than pretrained loading. |
| Equal-sized vectors report incompatible spaces | Match the complete declared `Space`, or supply an explicit adapter between spaces. Matching shape alone is insufficient. |
| A new encoder gives poor answers | Built-in word embeddings and image patches start randomly initialized. Supply pretrained parameters or train against actual targets. |
| Trace reports an unknown or aliased value | Preserve an explicit `OutputRef` from `session.calls[index].output`; scalar equality does not establish provenance. Plain Python transformations outside operations do not create traced edges. |
| Save/release rejects mutation | Treat captured intermediates as immutable. Represent state changes as explicit inputs or new operation outputs; do not modify saved tensor storage in place. |
| Async capture is still pending | Await all operation tasks before using their references or saving/releasing the session. A closed session cannot accept new calls from inherited task contexts. |
| Experience load rejects a binding | Reconstruct the same labels/vocabulary, spaces, module structure and semantic configuration. Weights may change; constructor semantics may not silently change. |
| A dataclass has no codec | Pass the same stable allowlist mapping to save and load, for example `codecs={'record-v1': Record}`. Artifact-provided import names are never executed. |
| Restored tensors use the wrong device | Experience tensors restore on CPU. Explicitly adapt restored roots/targets for the bound model's device. |
| Replay rejects an external operation | Pure replay is the default. Explicit `boundary='recorded'` uses its captured output as a constant without repeating the effect. |
| Training has no gradients | Supervise a tensor path connected to the bound trainable parameters. Remote outputs, detached values and discrete ranking indices do not become differentiable. |
| Checkpoint rejects shared aliases or optimizer slots | Recreate the same shared module instances and optimizer parameter ownership/layout. Optimizer checkpoints support SGD, Adam and AdamW. |
| Structured provider output is rejected | Check the exact schema, explicit `abstained`, configured alternatives, finite score ranges and distribution fields. The adapter does not repair malformed JSON or invent confidence. |
| Local structured generation repeatedly fails | Prompted JSON is still fallible and has no grammar-constrained decoder here. The saved evaluation records retain such failures; use a backend meeting the required response contract. |
| Provider reports refusal, truncation or an incomplete result | Handle the error explicitly. Adjust allowed token limits or model/request settings where appropriate; there is no hidden retry or fallback. |
| Chatbot history/objective did not commit | Encoding, workspace computation or decoding failed before the turn committed. Prior conversation state is retained. |
| Memory changes conflict across processes | Built-in memory locking covers one process. Supply application-level coordination for multiple writers. |

See [operations](operations.md), [training](training.md), and [tools](tools.md) for the corresponding contracts. [Validation](validation.md) separates verified mechanisms from measured model behavior and outstanding scope limits.

## PyTorch and torchvision build mismatch

If importing an owned model raises `operator torchvision::nms does not exist`,
check that PyTorch and torchvision use compatible builds. In CPU-only environments,
install both from the CPU index before installing TensorCode extras:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[tools,dev]'
```

For an already mixed environment, reinstall the incompatible packages using the
same selected build source. CI installs both CPU packages together; mixing CPU
PyTorch with a CUDA torchvision wheel can fail during Transformers imports.

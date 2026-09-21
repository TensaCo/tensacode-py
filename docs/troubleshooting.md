# Troubleshooting

| Symptom | What to check |
|---|---|
| Missing torch or Transformers | Install `tensorcode[vec]` for tensor operations or `tensorcode[local]` for local model integration. Core message/graph contracts work without either. |
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
| Chatbot history/objective did not commit | Response, decoding or persistence failed before the transaction committed. Prior state is retained; external effects inside callbacks cannot be undone. |
| Memory changes conflict across processes | Built-in memory locking covers one process. Supply application-level coordination for multiple writers. |

See [operations](operations.md), [training](training.md), and [tools](tools.md) for the corresponding contracts. [Validation](validation.md) separates verified mechanisms from measured model behavior and outstanding scope limits.

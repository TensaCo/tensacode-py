# Operations and providers

Operations use `operation(value, *, context=None)`. Implement the public `Operation.forward` contract and invoke the instance to retain tracing. Context contains conditioning data; required operands belong in the primary value. Pure operations explicitly opt into replay. Imports of core contracts, message operations and ordinary graph operations do not load PyTorch.

## Vectors

Install the `vec` extra for `tensorcode.ops.vec`. Native tensors keep their autograd graph and normal module registration, hooks, device movement and shared parameter identity.

```python
import torch
from tensorcode.ops import vec

text_space = vec.Space("application.text", 64)
shared_space = vec.Space("application.retrieval", 32)
encode = vec.TextEncoder(vocabulary=("refund", "transfer", "card"), dimensions=64, space=text_space)
project = vec.Transform(torch.nn.Linear(64, 32), input_space=text_space, output_space=shared_space)
query = project(encode("refund"))
```

`Space(name, dimensions, version='1', organization='feature', dtype=None, device=None)` identifies a representation. Compatibility compares its fields, including optional dtype/device requirements; equal dimensions alone do not establish compatibility. A common name is a caller-authored contract, not evidence of trained alignment.

`Latent(tensor, space, mask=None, coordinates=None, sources=(), metadata=None)` preserves the supplied tensor. The last dimension matches the space; a boolean mask matches the leading dimensions, and coordinates add a coordinate axis to that leading shape. Structural tensors share the data device.

| Operation | Contract |
|---|---|
| `TextEncoder(vocabulary=..., dimensions=64, space=None)` | Caller vocabulary, lowercase regex tokenization and mean-pooled trainable embeddings; raw tensor output unless a space is configured |
| `Transform(module, *, combine=None, input_space=None, output_space=None)` | Supplied module; configured input space requires `Latent`, configured output space produces `Latent`; organization-preserving transforms retain masks/coordinates |
| `Classify(module, *, labels, combine=None, input_space=None)` | Returns `Prediction.logits`, softmax `probabilities`, and explicit single/batch `value`/`values` |
| `ImageEncoder(...)` | CHW/BCHW images to spatial channel-last latent patches; configurable patch size, channels, dimensions, space and supplied module |
| `Decode(module, *, input_space, output)` | Validates space and applies a supplied decoder; `output` describes its result contract; `Decoder` is an alias |

The built-in text embeddings and default image convolution begin with random parameters. They supply trainable mechanisms, not pretrained understanding. Image coordinates use actual convolution geometry where known. Arbitrary supplied modules omit coordinates unless the caller supplies `coordinate_stride`/`coordinate_offset`; supplied modules must preserve the batch dimension and return the configured feature count.

Candidate scoring takes `CandidateSet(query, candidates, identities, metadata=...)`, where candidate tensors have shape `(..., N, features)` and query/candidate batch shapes agree. Identities are unique stable strings; empty candidates are rejected.

```python
scored = vec.Score(
    scorer_module,
    query_space=shared_space,
    candidate_space=shared_space,
    meaning="unnormalized dot-product relevance",
)(candidates)
decision = vec.Decide()(scored)
retrieval = vec.Retrieve(k=2)(scored)
```

The supplied scorer receives `(query_tensor, candidate_tensor)` and returns floating scores of shape `(..., N)`. Their stated meaning is retained, not converted to probabilities. `Decide(largest=True)` and `Retrieve(k=..., largest=True)` exclude unavailable masked candidates and retain selected tensors, scores, indices and metadata. Each row needs a valid candidate; retrieval cannot request more than its valid count. Identity access is explicit. Ranking indices are discrete even when selected values and scores retain gradients.

## Messages and model adapters

`tensorcode.ops.llm.Message(role, str)` supports plain text. Multimodal content uses immutable `TextPart(text, source_ref=None)` and `ImagePart(data=... | url=..., media_type=None, source_ref=None, detail=None)`. Each image has exactly one bytes/URL source. Encoding never downloads URLs or converts them into bytes implicitly.

`TextEncoder`, `ImageEncoder`, `TextDecoder`, and `Transform` compose the message path. Models use `ModelRequest`/`ModelOutput` and the public `Model`, `AsyncModel`, and `BatchModel` protocols. Structured `Classify`, `Score`, `Decide`, and `Retrieve` validate responses and raise `InvalidModelOutput` for contract violations.

Classifications/decisions select only configured alternatives. Returned distributions must contain exactly those alternatives, finite values in `[0, 1]`, and sum to one within `0.001`. Score results respect their supplied numeric rubric. Retrieval returns existing stable keys/items; arbitrary items need explicit descriptions. Retrieval scores are not probability distributions. Structured responses explicitly state `abstained`; valid abstentions carry no selected result. Missing distribution/confidence stays `None`, with no implicit threshold or repair.

`await operation.acall(...)` is the explicit asynchronous surface; synchronous calls return values. Structured operations additionally expose `batch` and `abatch`. Synchronous batching uses backend `complete_batch` when available outside tracing; under tracing it calls each operation normally to preserve references and failed-call records. Asynchronous batches use concurrent explicit async calls.

| Adapter | Supported behavior |
|---|---|
| `integrations.OpenAICompatibleModel` | Explicit `api='chat_completions'` or `api='responses'`; text/images, supplied model, strict structured JSON schemas |
| `integrations.JevModel` | Documented `/v1/systemone` Choice and Score mapping; rejects chat, images and retrieval |
| `integrations.LocalModel` | Explicitly supplied Transformers model/processor through the same request/output contract; optional `local` extra |

HTTP adapters use one buffered request with no implicit retry or fallback. Image bytes become media-typed data URLs; URL inputs stay URLs. Redirects, refusal, truncation and incomplete responses are errors. Source references remain in TensorCode data but are not invented as provider wire fields. Configuration and exceptions exclude API keys. Provider calls do not opt into replay.

Local models must be acquired and supplied explicitly; their adapter does not fetch image URLs. Prompted JSON still requires strict validation and can fail; grammar-constrained decoding is not implemented. [Validation](validation.md) records actual local outputs, including failures, and distinguishes transport tests from model quality. Hosted OpenAI/TypeSafe quality has not been evaluated here.

Provider references used for the supported wire contracts: [OpenAI Chat](https://developers.openai.com/api/reference/cli/resources/chat), [OpenAI Responses](https://developers.openai.com/api/reference/cli/resources/beta/subresources/responses/methods/create), and the official [TypeSafe schema models](https://github.com/typesafe-ai/typesafe-sdk-python/blob/main/src/typesafe_sdk/_schemas/models.py) and [endpoint builder](https://github.com/typesafe-ai/typesafe-sdk-python/blob/main/src/typesafe_sdk/_core/endpoints.py).

## Graphs

`tensorcode.ops.graph.Graph(nodes, edges=(), sources=(), identity=None, attributes=..., node_attributes=..., edge_attributes=..., source_anchors=...)` preserves supplied node identities, relation strings, competing edges and source evidence. `SourceAnchor(source, target=None, location=None, attributes=...)` targets a graph, node ID or edge index. Its source joins the graph's sources; unknown referents are rejected. Finite JSON attributes are copied into recursively immutable maps/tuples; edge attributes align by index so competing triples remain distinct.

`JSONDecoder()(graph)` produces JSON-compatible data; `JSONDecoder(as_text=True)` produces `tensorcode.graph/v1` JSON text. `JSONEncoder()` restores a graph without creating unknown nodes.

Supplied semantic operations include `Score(callback, semantics=...)`, `Retrieve(corpus, relevance, semantics=..., limit=...)`, and `Decide(utility, semantics=...)` with `ChoiceInput(objective, options)`. They return finite supplied scores, corpus members or exact options. Names do not assign meaning to relations or scores. `Transform(callback, identity=...)` requires explicit identity for persistence. Deterministic queries are not learned reasoning.

Optional `tensorcode.ops.graph.neural` provides `GraphEncoder(input_dimensions, hidden_dimensions, output_dimensions, *, space, steps=2, feature_key='features')` and `GraphClassifier(encoder, *, labels)`. The encoder uses directed adjacency and supplied numeric node features for trainable sum-message passing. An explicit aligned `node_features` tensor may be provided; traced feature dependencies travel through context. Its latent retains node order, graph/source identity and anchors. The classifier mean-pools node features and emits logits plus node/pooled latents. Native gradients, hooks, dtype/device and shared modules remain available.

This adapter handles one graph per call. It does not learn relation-label meanings or infer a graph from free-form evidence. The ordinary graph package stays torch-free; the neural module requires the `vec` extra.

For configuration fingerprints, explicit codecs and cross-process training, see [tracing and training](training.md).

## Add your own operation

Subclass the public `Operation` contract for ordinary Python code. This complete
example treats file reading as an external effect, so replay cannot silently read
a changed file:

```python
from pathlib import Path
from tensorcode.ops import Operation

class ReadText(Operation):
    def forward(self, value, *, context=None):
        if context:
            raise ValueError('ReadText does not consume context')
        return Path(value).read_text(encoding='utf-8')

read = ReadText()
text = read('README.md')
```

Invoke `read(...)`, not `read.forward(...)`, to keep the tracing boundary. The
base `replayable=False` is appropriate for I/O; opt into replay only for operations
that can safely recompute. For tensor modules, use `vec.Transform` around your
`torch.nn.Module` to preserve native parameter registration and hooks. If a custom
operation will be persisted, expose truthful JSON-safe `configuration()` metadata
for behavior that cannot be inferred; see [configuration and codecs](training.md#configuration-and-codecs).

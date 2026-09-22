# Operations and providers

Operations use `operation(value, *, context=None)`. Implement the public `Operation.forward` contract and invoke the instance to retain tracing. Context contains conditioning data; required operands belong in the primary value. Pure operations explicitly opt into replay. Imports of core contracts, message operations and ordinary graph operations do not load PyTorch.

## Vectors

Install the `vec` extra for basic `tensorcode.ops.vec` operations. The new owned transformer encoders and text/image decoders use the `pretrained` or `diffusion` extras; see [pretrained vector models](latent-models.md). Native tensors keep their autograd graph and normal module registration, hooks, device movement and shared parameter identity.

```python
from tensorcode.ops import vec
from tensorcode.ops.vec.encode import VocabularyEncoder

text_space = vec.Space("application.text", 64)
shared_space = vec.Space("application.retrieval", 32)
encode = VocabularyEncoder({
    "vocabulary": ["refund", "transfer", "card"], "dimensions": 64,
    "output_space": text_space.configuration(),
})
project = vec.Transform({
    "architecture": "linear", "input_space": text_space.configuration(),
    "output_space": shared_space.configuration(),
})
query = project(encode("refund"))
```

`Space(name, dimensions, version='1', organization='feature', dtype=None, device=None)` identifies a representation. Compatibility compares its fields, including optional dtype/device requirements; equal dimensions alone do not establish compatibility. A common name is a caller-authored contract, not evidence of trained alignment.

`Latent(tensor, space, mask=None, coordinates=None, sources=(), metadata=None)` preserves the supplied tensor. The last dimension matches the space; a boolean mask matches the leading dimensions, and coordinates add a coordinate axis to that leading shape. Structural tensors share the data device.

| Operation | Contract |
|---|---|
| `encode.TextEncoder(config)` | Owned text transformer; raw text → `output_space`; `readout='sequence'` or native masked-mean `'pooled'` |
| `encode.ImageEncoder(config)` | Owned ViT and processor; image → `output_space`; `readout='sequence'` or native CLS `'pooled'` |
| `decode.TextDecoder(config)` | `input_space` → generated text; explicit linear or identity bridge |
| `decode.ImageDecoder(config)` | `input_space` → RGB pixels; explicit bridge and sampling seed/noise |
| `VocabularyEncoder(config)` | `vocabulary` list, `dimensions`, optional `output_space`; lowercase regex tokenization and mean-pooled trainable embeddings |
| `Transform(config)` | Owned `linear`, `mlp`, or native `transformer`; declared `input_space` and `output_space`; returns `Latent` |
| `Classify(config)` | Owned head with `input_space` and `labels`; returns `Prediction.logits`, softmax `probabilities`, and explicit single/batch `value`/`values` |
| `PatchEncoder(config)` | Owned convolution; `patch_size`, `in_channels`, `output_space`, optional geometry; CHW/BCHW images → spatial channel-last latent patches |
| `Decode(config)` | Owned `linear`/`mlp`/`transformer` readout; `input_space`, `output_dimensions`, and descriptive `output`; returns a tensor |
| `Score(config)` | Owned `linear`/`mlp`/`transformer` candidate scorer; declared `query_space`, `candidate_space`, and score `meaning` |
| `Decide(config=None)` / `Retrieve(config=None)` | Parameter-free selection; `largest` and retrieval `k` are explicit config fields |

The table's `encode` and `decode` names refer to public modules
`tensorcode.ops.vec.encode` and `tensorcode.ops.vec.decode`. Their concrete class
identities are canonical; `vec.TextEncoder` and other root exports are convenience
aliases. Backend implementations are private. Both pretrained encoders validate
ordered `context={'latents': [...]}` prefixes against an explicit `context_space`;
both decoders validate latent prefixes against `input_space`. Text and image source
inputs remain modality-specific. See [model contracts](latent-models.md).

The built-in text embeddings and default image convolution begin with random parameters. They supply trainable mechanisms, not pretrained understanding. Image coordinates use actual convolution geometry where known. Arbitrary supplied modules omit coordinates unless the caller supplies `coordinate_stride`/`coordinate_offset`; supplied modules must preserve the batch dimension and return the configured feature count.

Candidate scoring takes `CandidateSet(query, candidates, identities, metadata=...)`, where candidate tensors have shape `(..., N, features)` and query/candidate batch shapes agree. Identities are unique stable strings; empty candidates are rejected.

```python
scored = vec.Score({
    "architecture": "mlp", "hidden_dimensions": [32],
    "query_space": shared_space.configuration(),
    "candidate_space": shared_space.configuration(),
    "meaning": "learned unnormalized relevance",
})(candidates)
decision = vec.Decide({"largest": True})(scored)
retrieval = vec.Retrieve({"k": 2, "largest": True})(scored)
```

Scores have shape `(..., N)` and retain their declared meaning; they are not
implicitly probabilities. Selectors exclude unavailable masked candidates and
retain selected tensors, scores, indices and metadata. Each row needs a valid
candidate; retrieval cannot exceed its valid count. Ranking indices are discrete
even when selected values and scores retain gradients.

Primary learned constructors accept JSON configuration and create all parameters
up front. Linear and MLP vector operations start untrained; MLP configuration
adds `hidden_dimensions`. Native transformer configuration uses `native_config`
and, for Transform/Decode, `readout='sequence'|'pooled'`. Classify uses pooled
readout; Score produces one scalar per candidate. Native BERT, RoBERTa, and DistilBERT
architectures are supported for these general vector operations. Candidate scorers
project query/candidate features and use their elementwise interactions; optional
`pair_dimensions` sets the projected width. Supported `from_foundation` factories initialize
explicit pretrained architectures; newly added projections remain untrained.
Owned operations expose `save_pretrained` and `from_pretrained` for complete
configuration/weight artifacts. See the [owned lifecycle example](../examples/owned_vector_lifecycle.py).

Advanced `Transform.from_module(module, ...)`, `Classify.from_module(module, ...)`,
`Score.from_module(module, ...)`, `Decode.from_module(module, ...)`, and
`PatchEncoder.from_module(module, ...)` integrate supplied implementations with
explicit space/label/geometry contracts. Arbitrary executable modules cannot be
reconstructed safely from data-only artifacts; unsupported saves raise instead
of discarding their behavior. Pure selectors persist configuration without weights.

## Messages and model adapters

`tensorcode.ops.text.Message(role, str)` supports plain text. Multimodal content uses immutable `TextPart(text, source_ref=None)` and `ImagePart(data=... | url=..., media_type=None, source_ref=None, detail=None)`. Each image has exactly one bytes/URL source. Encoding never downloads URLs or converts them into bytes implicitly.

`TextEncoder(config=None)`, `ImageEncoder(config=None)`, and `TextDecoder(config=None)` are pure message serialization operations with safe configuration persistence. `ImageEncoder` configuration includes `media_type`, `source_ref`, and `detail`. `Transform(config)` owns its local model. Explicit external integrations use `Transform.from_model(provider)` and structured `Classify.from_model(provider, labels=..., ...)`, `Score.from_model`, `Decide.from_model`, or `Retrieve.from_model`. These models use `ModelRequest`/`ModelOutput` and the public `Model`, `AsyncModel`, and `BatchModel` protocols. Structured `Classify`, `Score`, `Decide`, and `Retrieve` validate responses and raise `InvalidModelOutput` for contract violations.

Owned text `Transform`, `Classify`, `Score`, `Decide`, and `Retrieve` accept
configuration with `native_config` and an embedded fast-tokenizer `tokenizer`
configuration. These instantiate supported native sequence-to-sequence models.
Optional settings include `generation` and `instructions`; structured operations
also declare `labels`, `rubric`, `options`, or `items`/`descriptions`/`limit`,
respectively. `from_foundation(repo, config=..., revision=...)` explicitly loads
native weights and tokenizer assets; for example,
`Classify.from_foundation(local_path, config={'labels': ['yes', 'no']})`.
Saved owned artifacts preserve model, tokenizer, generation and semantic settings.
Teacher-forced objectives train local parameters with targets separate from source
inputs. External `from_model` providers keep their existing transport behavior.
Native text operations accept textual message content; image content is rejected.
Structured training targets supply the full required response mapping, while
Transform targets are strings. Persisted message experiences require explicit
`codecs={'message': text.Message}` and a `TextPart` codec for multipart text.
Random native generation may fail strict response validation; there is no output
repair. Trace capture does not make remote calls differentiable, and arbitrary providers
have no promised data-only model artifact reconstruction.

Classifications/decisions select only configured alternatives. Returned distributions must contain exactly those alternatives, finite values in `[0, 1]`, and sum to one within `0.001`. Score results respect their supplied numeric rubric. Retrieval returns existing stable keys/items; arbitrary items need explicit descriptions. Retrieval scores are not probability distributions. Structured responses explicitly state `abstained`; valid abstentions carry no selected result. Missing distribution/confidence stays `None`, with no implicit threshold or repair.

Structured text operations return frozen results:
`ClassificationResult(label, distribution=None, confidence=None, abstained=False)`,
`DecisionResult(choice, ...)`, `ScoreResult(value, distribution=None,
confidence=None, abstained=False)` with integer rubric keys, and
`RetrievalResult(keys, items, scores=None, abstained=False)`.

### Decoding owned structured operations

Owned `Classify`, `Decide`, `Score` and `Retrieve` accept `decoding`:

- `'generate'` (default) generates the JSON response, including any distribution,
  as text. Those numbers are generated values, not model likelihoods; malformed
  output raises `InvalidModelOutput`.
- `'likelihood'` encodes the input once and scores every configured alternative
  with the decoder in one batch. Results always carry a complete distribution
  (softmax of log-likelihoods) and never abstain; `confidence` is the top
  probability. `Score.value` is the probability-weighted level. `Retrieve`
  returns the `limit` best items and every item's log-likelihood as `scores`.
  `likelihood_normalization` is `'sum'` (default) or `'mean'` per target token.
  Alternatives must tokenize to distinct sequences.

```python
route = text.Classify.from_foundation(
    "google/flan-t5-base",
    revision=...,
    config={
        "labels": ["billing", "technical"],
        "descriptions": {"billing": "payments, charges and refunds"},
        "instructions": "Route the support ticket",
        "decoding": "likelihood",
    },
)
result = route((text.Message("user", "I was charged twice"),))
```

The likelihood prompt lists the instructions, `role: content` lines, the options
(with descriptions) and `Answer:`; the target is the label, option, rubric level
text or item description. Training targets use the same result mapping as
generation; a target distribution trains a soft target, and abstention targets
are rejected. Probabilities remain uncalibrated model scores: fit thresholds on
separate data with `training.calibration.fit_threshold` before acting on them.

`Classify` and `Decide` accept optional `descriptions` for any subset of their
alternatives. Descriptions appear in likelihood prompts and in the response
schema sent to external models.

### Several questions about one input

`text.ask(messages, {"name": operation, ...}, *, context=None)` answers named
structured operations about the same messages and returns a read-only mapping
of results; `await text.aask(...)` is the asynchronous form. When every operation
wraps the same external model implementing the `QuestionModel` protocol
(`complete_questions(requests: Mapping[str, ModelRequest])`), all questions
travel in one exchange. Otherwise, and whenever a trace is active, each operation
is called normally so tracing records every call. Owned operations each own
their model, so they run in turn.

```python
answers = text.ask(messages, {
    "spam": text.Classify.from_model(jev, labels=("true", "false"), instructions="Is this spam?"),
    "route": text.Decide.from_model(jev, options=("billing", "technical"), instructions="Route"),
    "urgency": text.Score.from_model(jev, rubric=("low", "medium", "high"), instructions="Urgency"),
})
```

`await operation.acall(...)` is the explicit asynchronous surface; synchronous calls return values. Structured operations additionally expose `batch` and `abatch`. Synchronous batching uses backend `complete_batch` when available outside tracing; under tracing it calls each operation normally to preserve references and failed-call records. Asynchronous batches use explicit async calls. Owned native generation serializes access to its shared tokenizer and model mode; external providers may run concurrently.

| Adapter | Supported behavior |
|---|---|
| `integrations.OpenAICompatibleModel` | Explicit `api='chat_completions'` or `api='responses'`; text/images, supplied model, strict structured JSON schemas |
| `integrations.JevModel` | Documented `/v1/systemone` mapping: labels exactly `true`/`false` → `noul` (confidence `None`), other selections → Choice with descriptions as criteria, Score → Score. `complete_questions` sends several questions about identical messages in one request. Rejects chat, images and retrieval |
| `integrations.LocalModel` | Explicitly supplied Transformers model/processor through the same request/output contract; optional `local` extra |

HTTP adapters use one buffered request with no implicit retry or fallback. Image bytes become media-typed data URLs; URL inputs stay URLs. Redirects, refusal, truncation and incomplete responses are errors. Source references remain in TensorCode data but are not invented as provider wire fields. Configuration and exceptions exclude API keys. Failures raise `ProviderError`
subclasses: `ProviderHTTPError` (with `.status`), `ProviderTimeout` and
`ProviderProtocolError` for malformed, refused, truncated or unsupported exchanges. Provider calls do not opt into replay.

Local models must be acquired and supplied explicitly; their adapter does not fetch image URLs. Prompted JSON still requires strict validation and can fail; grammar-constrained decoding is not implemented. [Validation](validation.md) records actual local outputs, including failures, and distinguishes transport tests from model quality. Hosted OpenAI/TypeSafe quality has not been evaluated here.

Provider references used for the supported wire contracts: [OpenAI Chat](https://developers.openai.com/api/reference/cli/resources/chat), [OpenAI Responses](https://developers.openai.com/api/reference/cli/resources/beta/subresources/responses/methods/create), and the official [TypeSafe schema models](https://github.com/typesafe-ai/typesafe-sdk-python/blob/main/src/typesafe_sdk/_schemas/models.py) and [endpoint builder](https://github.com/typesafe-ai/typesafe-sdk-python/blob/main/src/typesafe_sdk/_core/endpoints.py).

## Graphs

`tensorcode.ops.graph.Graph(nodes, edges=(), sources=(), identity=None, attributes=..., node_attributes=..., edge_attributes=..., source_anchors=...)` preserves supplied node identities, relation strings, competing edges and source evidence. `SourceAnchor(source, target=None, location=None, attributes=...)` targets a graph, node ID or edge index. Its source joins the graph's sources; unknown referents are rejected. Finite JSON attributes are copied into recursively immutable maps/tuples; edge attributes align by index so competing triples remain distinct.

Graph operations are **symbolic API stubs**. They reserve a common callable shape
but have no inference, neural, callback, or JSON-serialization implementation.

| Operation | Intended symbolic contract (not implemented) |
|---|---|
| `Encode()` / `TextEncode()` | Input evidence / text → source-grounded graph structure |
| `Decode()` / `TextDecode()` | Graph structure → output representation / language |
| `Transform()` | Graph → revised graph, conditioned on context |
| `Score()` | Graph → assessment under an explicit objective |
| `Retrieve()` | Graph query → relevant graph values |
| `Decide()` | `ChoiceInput(objective, options)` → selected graph option |
| `Classify()` | Graph → category |

```python
from tensorcode.ops.graph import TextEncode

encode = TextEncode()
try:
    encode("The first report conflicts with the later observation.")
except NotImplementedError:
    pass  # Symbolic text interpretation is not available yet.
```

Every operation raises `NotImplementedError` through both `operation(...)` and
`await operation.acall(...)`. Constructors accept optional empty JSON configuration and no model or callback.
`from_foundation`, `from_pretrained`, and `save_pretrained` also raise
`NotImplementedError`; no graph model artifact is implied. No fallback
assigns meaning to relation strings. Graph records can still store supplied
structure and perform structural lookups such as `graph.neighbors(...)`.
`JSONEncoder`, `JSONDecoder`, and `tensorcode.ops.graph.neural` have been removed.
Historical neural measurements do not describe the active symbolic API; see
[validation](validation.md#historical-graph-experiments).

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
that can safely recompute. For tensor modules, use `vec.Transform.from_module` around your
`torch.nn.Module` to preserve native parameter registration and hooks. If a custom
operation will be persisted, expose truthful JSON-safe `configuration()` metadata
for behavior that cannot be inferred; see [configuration and codecs](training.md#configuration-and-codecs).

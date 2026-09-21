# Task 2 — Vector toolbox

Implemented tensor-native vector representations, trainable text and image encoders, explicit adapters and decoders, and score/decision/retrieval primitives. Existing unconfigured `Transform`, `TextEncoder`, and `Classify` calls still consume and return ordinary tensors. Supplying spaces opts into `Latent` values and semantic-space validation.

## Public API

```python
from tensorcode.ops import vec

text_space = vec.Space("ticket-encoder/v1", 64, organization="feature")
shared_space = vec.Space("paired-retrieval/v2", 32, organization="feature")

encode = vec.TextEncoder(
    vocabulary=("refund", "transfer", "card"),
    dimensions=64,
    space=text_space,
)
project = vec.Transform(
    torch.nn.Linear(64, 32),
    input_space=text_space,
    output_space=shared_space,
)
query = project(encode("refund"))
```

- `Space(name, dimensions, version='1', organization='feature', dtype=None, device=None)` identifies a representation. Compatibility compares all fields; equal tensor dimensions do not imply compatibility. Optional dtype and device expectations are enforced by `Latent`.
- `Latent(tensor, space, mask=None, coordinates=None, sources=(), metadata=None)` retains the original tensor and autograd graph. Its final tensor dimension must match the space. A mask matches every leading tensor dimension, and coordinates add one coordinate dimension to that same leading shape. Structural tensors must share the data device.
- `Transform(module, *, combine=None, input_space=None, output_space=None)` preserves the existing raw-tensor API. Configured input spaces require `Latent`; configured output spaces return `Latent`. Sources and metadata are retained, and masks/spatial coordinates are retained when the transform preserves leading organization. The supplied `nn.Module` remains registered normally, so hooks, device movement, shared parameter identity, gradients, and optimizers work normally.
- `TextEncoder(vocabulary=..., dimensions=64, space=None)` preserves its previous lowercase regex tokenization, caller-supplied vocabulary, mean pooling, and raw tensor default. Supplying a compatible space returns `Latent`. Its embeddings are randomly initialized learned parameters.
- `Classify(module, labels=..., combine=None, input_space=None)` preserves the previous `Prediction` API, including tensor logits, derived softmax probabilities, and explicit scalar/batch label access. Labels and optional input space are part of configuration identity.
- `ImageEncoder(patch_size=..., space=..., in_channels=..., dimensions=..., module=None, coordinate_stride=None, coordinate_offset=None)` consumes CHW or BCHW tensors and returns channel-last spatial `Latent` patches. The default trainable convolution uses PyTorch random initialization and reports its actual receptive-field centers in source-pixel coordinates, including nondivisible images. Known supplied `Conv2d` geometry is derived from kernel, stride, padding, and dilation. Arbitrary modules report no coordinates unless the caller supplies an explicit stride/offset contract. Supplied modules must preserve batch count and produce BCHW with the configured feature count.
- `Decode(module, *, input_space, output)` applies an explicit supplied module after validating its input space. `Decoder` is an alias. The output description states the mechanical result contract; decoding does not establish truth.

Candidate operations use one explicit operand:

```python
candidates = vec.CandidateSet(
    query=query,                                      # (..., query_features)
    candidates=stored,                               # (..., N, candidate_features)
    identities=("memory:1", "memory:2"),
    metadata=({"source": "one"}, {"source": "two"}),
)
scored = vec.Score(
    scorer_module,
    query_space=shared_space,
    candidate_space=shared_space,
    meaning="unnormalized dot-product relevance",
)(candidates)
decision = vec.Decide()(scored)
retrieval = vec.Retrieve(k=2)(scored)
```

- Candidate identities are unique, nonempty stable strings. Query/candidate batch shapes must match and an empty candidate axis is rejected.
- `Score` calls the supplied module as `module(query_tensor, candidate_tensor)` and returns `Scores(values, meaning, candidates)`. Values must have shape `(..., N)` and remain floating tensors with gradients. The required meaning prevents silently treating logits, similarity, probability, and utility as interchangeable.
- A boolean candidate mask means availability. Every row needs at least one valid candidate. `Decide(largest=True)` and `Retrieve(k=..., largest=True)` exclude unavailable candidates for both ascending and descending ranking; retrieval also bounds `k` against every row's valid count.
- `Decision` and `Retrieval` retain tensor indices, selected scores, selected `Latent` values, source metadata, and the original scored candidates. Python identity conversion is explicit through `identity`/`identities`; selecting an identity does not execute it. Retrieval returns existing candidates and never generates replacements.

## Persistence configuration

Every vector operation exposes `configuration() -> JSON-safe dict`. Configuration includes constructor semantics, spaces, ordered labels/vocabulary, tokenization/pooling mechanics, score meanings, ranking direction and bounds, image coordinate geometry, and the supplied module architecture. Module architecture records qualified types, child paths, JSON-safe behavior fields, and parameter/buffer names, shapes, dtypes, and gradient flags. It excludes parameter and buffer values, so ordinary learning does not change a binding fingerprint.

Custom modules may provide `configuration()`; that declaration is authoritative and must omit learned values. Conservative fallback records public and private custom JSON-safe fields and rejects opaque or unregistered tensor behavior state. Named callbacks include positional and keyword defaults. Closures, lambdas, locals, bound/stateful callables, or non-JSON defaults require explicit configuration metadata rather than receiving a colliding qualified-name fingerprint.

Mutable weights continue to live in `state_dict`; configuration describes compatibility, not a checkpoint. Two operations can register the same supplied backbone instance and retain shared parameter identity.

## Verification

The new tests were developed in red/green cycles. Initial tests failed on missing imports and each independent review reproduction failed for its stated behavior before the fixes. Final focused verification:

```text
uv run --extra dev --extra vec pytest -q tests/test_vec_*.py tests/test_text_encoder.py tests/test_learning.py tests/test_persistence.py
55 passed in 3.04s
```

Coverage includes incompatible equal-sized spaces; dtype/device expectations; native gradients through representations, adapters, classifiers, decoders, scorers, and image parameters; shared backbones; metadata, masks, and spatial-coordinate preservation; real optimizer updates; CHW/BCHW patch shapes; odd image boundaries; supplied convolution and arbitrary-module geometry; supplied-module batch validation; empty/misaligned/duplicate candidates; batched decisions; ascending/descending mask exclusion; valid-candidate bounds; rank identities; JSON roundtrips and configuration contents for labels, vocabulary, and spaces; learned-weight stability; custom public/private module behavior fields; explicit custom configuration; and ambiguous callback rejection.

Fresh whole-repository verification after the final configuration hardening:

```text
uv run --extra dev --extra vec pytest -q
174 passed in 8.50s
```

Independent vector review found configuration collisions, incorrect coordinates on nondivisible images, unsupported coordinate claims for arbitrary modules, and ignored candidate masks. All were reproduced and fixed with regression coverage. Its optional dtype/device, unique identity, supplied geometry, and batch-count checks were also implemented.

## Semantics and limits

The built-in text embedding and default image patch convolution start from random parameters. They provide trainable mechanisms and make no pretrained language, image-understanding, or cross-modal alignment claim. A shared `Space` name is a caller-authored contract; it does not train or prove alignment. Cross-modal behavior requires an explicitly supplied/trained adapter.

`Score` reports exactly the supplied module output under the caller's stated meaning. It does not calibrate scores or invent probabilities. `Decide` and `Retrieve` use deterministic tensor ranking; argmax/top-k indices are discrete even though selected scores, candidate values, and upstream scoring computations retain their ordinary tensor gradients. Candidate masks express mechanical availability, not semantic relevance.

Arbitrary supplied image modules cannot reveal receptive-field geometry automatically; coordinates are omitted unless geometry is known or explicitly declared. Custom configuration methods are trusted declarations and cannot prove that arbitrary user code retained its semantics. The vector toolbox does not bundle pretrained weights, semantic seeds, domain labels, implicit thresholds, remote models, or a universal cognitive policy.

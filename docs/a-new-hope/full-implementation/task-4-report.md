# Task 4 — Graph toolbox and cross-representation learning

Implemented an immutable, open-vocabulary graph representation; explicit JSON
conversion; callback-defined score, retrieval, decision, and transform
operations; and an optional trainable PyTorch graph-to-vector adapter with a
graph prediction path. The deterministic operations apply semantics supplied by
the caller. They are not reported as learned reasoning.

## Public representation and JSON API

```python
from tensorcode.ops.graph import Graph, SourceAnchor, JSONEncoder, JSONDecoder

graph = Graph(
    nodes=("ticket", "open", "closed"),
    edges=(("ticket", "state", "open"), ("ticket", "state", "closed")),
    sources=("document:a", "document:b"),
    identity="ticket-7",
    attributes={"schema": "example-v1"},
    node_attributes={"ticket": {"kind": "record"}},
    edge_attributes=({"asserted": True}, {"asserted": False}),
    source_anchors=(SourceAnchor("document:a", target="ticket", location={"page": 2}),),
)

document = JSONDecoder()(graph)          # ordinary JSON-compatible data
text = JSONDecoder(as_text=True)(graph)  # tensorcode.graph/v1 JSON text
restored = JSONEncoder()(text)           # JSON data/text -> Graph
assert restored == graph
```

The original `Graph(nodes, edges=(), sources=())` call remains valid. Graph,
node, and edge attributes accept finite JSON values and are recursively copied
into immutable maps and tuples. Node attributes align by stable node identity;
edge attributes align by edge index, so repeated or competing edge triples are
not collapsed. Source anchors can target the graph, a node ID, or an edge index.
An anchor's explicit source is included in `Graph.sources` when it was not also
listed through the legacy field. Unknown node and edge referents are rejected;
the codec never creates missing nodes.

## Supplied semantic operations

```python
from tensorcode.ops.graph import ChoiceInput, Decide, Retrieve, Score

score = Score(score_graph, semantics="my-package.edge-count-v1")
retrieve = Retrieve(corpus, relevance, semantics="my-package.relevance-v2", limit=5)
decide = Decide(utility, semantics="my-package.utility-v1")
decision = decide(ChoiceInput(objective, options))
```

`Score` returns one finite real value. `Retrieve` returns stable-ranked
`ScoredGraph` values drawn only from its configured corpus. `Decide` scores every
supplied option, returns the first maximum as `Decision.value`, and retains all
scores. The library assigns no meaning such as probability, relevance, or
utility to those numbers; the named callback does. Empty or non-string semantic
identities and nonfinite/non-real callback results are rejected.

Each operation exposes `configuration()` containing JSON-safe stable choices.
`Retrieve.configuration()` also contains its graph corpus in the explicit graph
JSON schema. `Transform` remains usable as an ephemeral callback for compatibility,
but `Transform.configuration()` rejects it until the caller supplies an explicit
`identity`. Callbacks themselves are never serialized or imported from an
artifact; applications reconstruct and bind the identified implementation.

## Optional neural API

```python
from tensorcode.ops.graph.neural import GraphClassifier, GraphEncoder
from tensorcode.ops.vec import Space

space = Space("application.graph-nodes", 32, organization="sequence")
encoder = GraphEncoder(8, 32, 32, space=space, steps=2)
nodes = encoder(graph)  # vec.Latent; tensor shape is [node_count, 32]

classifier = GraphClassifier(encoder, labels=("negative", "positive"))
prediction = classifier(graph)
```

`GraphEncoder(input_dimensions, hidden_dimensions, output_dimensions, *,
space, steps=2, feature_key="features")` is a trainable directed sum-message
passing module. It reads an aligned real feature vector from each node's named
attribute, or accepts an explicit `[node_count, input_dimensions]`
`node_features` tensor. Its `Latent` preserves native autograd, the caller's
`Space`, source IDs, stable node order, graph identity, source anchors, and edge
count. Edge labels remain open-vocabulary graph metadata; the current adapter
uses adjacency and does not assign authored meanings to relation strings.

`GraphClassifier` mean-pools the node tensor and applies a trainable head over
explicit labels. `GraphPrediction` exposes logits, probabilities, selected
value, the node latent, and the pooled graph latent. Both modules preserve
PyTorch parameter registration, dtype/device, hooks, and gradients. Both use the
common TensorCode trace boundary and opt into effect-free replay; explicit node
features travel through operation context when traced. Their configurations are
JSON-safe and describe dimensions, steps, feature key, direction, vector space,
labels, and pooling, while weights remain ordinary module state.

The core `tensorcode.ops.graph` package has no torch import. PyTorch and the
vector `Latent`/`Space` representation load only when
`tensorcode.ops.graph.neural` is imported. The dependency is the existing
`torch>=2.4` `vec` extra; no new project dependency was added.

## Verification

The graph tests exercise recursive immutability, stable hashing, graph identity,
JSON text/data roundtrips, source anchors, competing facts, malformed referents,
supplied semantic ranking and decisions, JSON-safe configurations, lazy torch
loading, native hooks, dtype propagation, trace/replay, real input and parameter
gradients, and training/generalization over held-out authored graph structures
whose node identities do not overlap training inputs.

Fresh focused verification after independent review:

```text
.venv/bin/pytest -q tests/test_graph_*.py tests/test_operations.py tests/test_persistence.py::test_immutable_message_and_graph_payloads_roundtrip tests/test_mutag_example.py
29 passed in 1.71s
```

The final working tree then passed the complete suite: `164 passed in 9.07s`.
`compileall` and `git diff --check` also completed without errors for the graph
implementation, graph tests, and this report. Independent review reproduced and
led to tests for classifier dtype/device propagation, preservation of encoder
hooks within the classifier, deterministic gradient-path evidence, and strict
edge-index validation.

The authored topology fixture is mechanism evidence only. It uses library-authored
constant node features and connected-versus-isolated labels; it is not evidence
of learned domain knowledge.

A separate fixed evaluation used the official TU Dortmund MUTAG archive, SHA256
`c419bdc853c367d2d83da4973c45100954ae15e10f5ae2cddde6ca431f8207f6`.
The example used 188 molecule graphs, a seed-7 graph-disjoint 150/38 split,
training-only atom categories with an unknown coordinate, two message-passing
steps, authored mean pooling, and 80 Adam epochs at learning rate 0.01. Encoder
parameters changed. Held-out cross-entropy improved from `0.6866499` to
`0.2946415`; held-out accuracy changed from 33/38 (`0.8684211`) to 34/38
(`0.8947368`), compared with a 26/38 (`0.6842105`) training-majority baseline.
The exact graph IDs, environment versions, settings, and measurements are saved
in `docs/a-new-hope/mutag-results.json`; the reproducible command is:

```text
.venv/bin/python examples/mutag.py --data /path/to/MUTAG.zip --epochs 80 --output docs/a-new-hope/mutag-results.json
```

This is one small fixed split with no confidence interval or benchmark-ranking
claim. Atom labels and adjacency are supplied dataset fields. Bond types are
preserved in graph edges but ignored by the current neural adapter. The result
demonstrates supervised graph learning through the public adapter, not inferred
chemistry, arbitrary symbolic reasoning, or differentiation through graph
queries.

## Current limits

The neural adapter processes one variable-size graph per call; callers currently
perform batching and graph-level pooling explicitly unless they use
`GraphClassifier`. It does not learn embeddings for relation labels or graph
attributes. JSON attributes deliberately exclude opaque Python objects and
nonfinite numbers. Semantic callback identity is caller-authored metadata and
cannot prove that arbitrary callback code has remained unchanged. Deterministic
retrieve/score/decide operations are neither differentiable claims nor learned
policies.

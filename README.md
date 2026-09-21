# TensorCode

Compose operations over vectors, messages, and graphs in ordinary Python. Import
ready-to-call tools, capture their dataflow, attach supervision, and train supported
local tensor paths after restarting your application.

**0.2.0a1 is a breaking, from-scratch replacement.** The governing
[architecture](docs/a-new-hope/1-tensorcode-architecture.md) and
[implementation report](docs/a-new-hope/5-full-implementation-report.md) describe
what is implemented and its limits. Models and policies are supplied explicitly;
the package does not bundle pretrained intelligence or a universal autonomous agent.

## Install

```bash
python -m pip install -e '.[vec,dev]'
# Optional: run an explicitly selected local Transformers image/text model.
python -m pip install -e '.[local]'
```

Python 3.11+. The base package has no third-party dependencies. Vector operations,
neural graph adapters, and tensor training require PyTorch. Importing TensorCode
makes no network calls; local model loading and remote requests are explicit.

## Compose operations

Operations use `op(value, *, context=None)`. Developers name instances according
to their roles; tools compose these public operations.

```python
import torch
from tensorcode.ops.vec import TextEncoder, Classify
from tensorcode.tools.decision import Decision


def make_operations():
    return {
        'encode': TextEncoder(vocabulary=('hello', 'refund', 'card'), dimensions=16),
        'classify': Classify(torch.nn.Linear(16, 2), labels=('question', 'refund')),
    }


operations = make_operations()
route = Decision(encode=operations['encode'], decide=operations['classify'])
prediction = route('refund my card')
print(prediction.value)          # random until trained
print(prediction.probabilities)  # probabilities, not calibrated confidence
```

Vector operations are native `torch.nn.Module` objects. The text encoder learns
mean-pooled word embeddings using an explicit vocabulary and regex tokenizer.
`Latent` and `Space` carry representation identity; vector image encoding retains
patch positions. Decode, score, decide, and retrieve operations use supplied
modules or explicit candidate representations. A patch encoder alone supplies no
learned scene understanding.

## Capture, supervise, persist, and train

The following continues the example above. Files belong to your application;
there is no automatic disk logging.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import tensorcode as tc
from tensorcode.training import Trainer, load, load_checkpoint, save_checkpoint

with TemporaryDirectory() as directory:
    path = Path(directory)
    with tc.trace() as episode:
        prediction = route(('hello', 'refund'))
    episode.supervise(prediction, ('question', 'refund'), source='human correction')
    save_checkpoint(path / 'initial.json', operations=operations)
    episode.save(path / 'experience.json', operations=operations, release=True)

    # These objects can be constructed in a fresh interpreter.
    restored = make_operations()
    load_checkpoint(path / 'initial.json', operations=restored)
    experience = load(path / 'experience.json', operations=restored)
    trainer = Trainer(restored, lr=0.1)  # SGD; an explicit optimizer is also accepted
    print(trainer.fit([experience], epochs=10))
    save_checkpoint(path / 'trained.json', operations=restored,
                    optimizer=trainer.optimizer)
```

Experience files contain versioned data and named operation bindings, not
executable Python. Loading requires caller-supplied operations with matching
configurations; custom dataclass codecs must be explicitly allowlisted. Model
weights and optimizer state live in separate checkpoints. `release=True` drops
live intermediates after saving; differentiable replay recomputes local operations.

`episode.example(ref)` extracts a dependency closure, and `episode.replay(ref)`
recomputes it with current parameters. For ambiguous scalar outputs, use
`episode.calls[-1].output`; equal values do not establish provenance. External
inputs are snapshotted, and known mutations are rejected. Capture does not infer
uncaptured Python computation or make remote calls differentiable. Recorded
external outputs can form constant training boundaries. Treat saved experiences
as application data: they can contain text, images, and supplied targets.

## Messages, models, and tools

Message operations include encode/decode, transform, classify, score, decide, and
retrieve. Structured operations validate model outputs, including allowed labels
and abstention. Models expose `complete(ModelRequest) -> ModelOutput`; simple
message-to-string callables also work with `Transform` and `Chatbot`. Async and
batch execution are explicit APIs, not automatic parallel side effects.

Use an explicitly selected OpenAI-compatible HTTP endpoint (no provider SDK is
required). This example calls a local server you run:

```python
from tensorcode.integrations import OpenAICompatibleModel
from tensorcode.ops.llm import Message, Transform

model = OpenAICompatibleModel(
    base_url='http://localhost:8000/v1', model='model-id', api_key=None,
    timeout=30.0, api='chat_completions',
)
respond = Transform(model)
messages = (Message('user', 'Hello'),)
reply = respond(messages)
# In async code: reply = await respond.acall(messages)
```

For a hosted endpoint, supply its URL, model, and credentials explicitly.
`api='responses'` selects the Responses transport. Structured operations also
provide `batch(...)` and `abatch(...)`; provider batching can be sequential.

This example explicitly loads a downloaded local model directory and supplies
image bytes:

```python
from pathlib import Path
from tensorcode.integrations.local import LocalModel
from tensorcode.ops.llm import ImagePart
from tensorcode.tools.agents import Chatbot

model = LocalModel.from_pretrained('/path/to/downloaded/model', local_files_only=True)
bot = Chatbot(model=model)
reply = bot('Describe the spatial relationships in this scene.', images=(
    ImagePart(data=Path('/path/to/scene.jpg').read_bytes(), media_type='image/jpeg',
              source_ref='scene-1'),
))
print(reply)
```

`ImagePart` also represents remote URLs without fetching them on construction.
The local adapter requires bytes. A selected remote provider may process URLs or
receive image bytes when invoked. `Message` accepts text or immutable tuples of
`TextPart` and `ImagePart`, preserving explicit source references.

For a bounded message decision, use
`Decision(model=model, labels=('question', 'refund'), instructions='Classify the request.')`.
Its accuracy and ability to produce valid structured output depend on the supplied
model. Invalid output is rejected rather than repaired into invented confidence.

`Chatbot` serializes turns and commits history, objective, and optional memory only
after a successful response. Components are replaceable through `encode`,
`encode_image`, `respond`, and `decode`. Optional tools expose their policies:

- `JsonMemory(path, retrieve=policy)` persists records with stable source IDs.
  The policy receives a `MemorySearch` containing query, candidates, and limit;
  relevance is caller-defined.
- `Chatbot(objective=..., update_objective=operation, ...)` sends an
  `ObjectiveRevision(current, observation)` to your operation. Revising an objective
  is distinct from learning model parameters.
- `ActionLoop(chooser=..., actions=..., max_steps=...)` executes only supplied
  action names. Actions return `ActionOutcome(state, receipt, done)`; the result
  records effects and whether execution completed, abstained, or exhausted its
  budget. The caller supplies both selection policy and executable effects.

## Graphs

Graphs retain node identities, parallel relations, attributes, and source anchors.
They impose no domain ontology and do not automatically resolve contradictions.
The represented graph and execution trace DAG are separate structures.

```python
from tensorcode.ops.graph import Graph, SourceAnchor, JSONEncoder, JSONDecoder

graph = Graph(
    nodes=('observation', 'interpretation'),
    edges=(('interpretation', 'supported_by', 'observation'),),
    sources=('document-1',),
    source_anchors=(SourceAnchor('document-1', target='observation'),),
)
assert JSONEncoder()(JSONDecoder()(graph)) == graph
```

Graph transforms, scoring, retrieval, and decision operations accept supplied
semantics. Optional `tensorcode.ops.graph.neural` adapters train graph encoders
and classifiers from supplied numeric node features while retaining node/source
correspondence. This is supervised graph learning, not automatic extraction of
structured knowledge from arbitrary evidence.

## Real-data validation

Download official CSVs from
[PolyAI's Banking77 repository](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data),
then run the restart experiment with a new artifact directory:

```bash
python examples/banking77_restart.py --train /path/to/train.csv --test /path/to/test.csv \
  --artifacts /tmp/banking77-new-run --output /tmp/banking77-results.json
```

Four sequential processes capture supervised experiences, evaluate the initial
checkpoint, reload and train, then evaluate the trained checkpoint. The recorded
run improved held-out accuracy from **0.97% to 89.48%** across 77 labels and 3,080
test rows; six overlapping training texts were excluded. Ground-truth labels are
supplied supervision, and the tokenizer and pooling are authored mechanics.
See [restart results](docs/a-new-hope/banking77-restart-results.json), the simpler
[in-process example](examples/banking77.py), and the
[full implementation report](docs/a-new-hope/5-full-implementation-report.md) for
graph learning, live model validation, failures, and remaining limitations.

## Development and history

```bash
python -m pytest -q
python -m build
```

The former implementation, tests, and research reports are preserved in the private
[pre-reset archive](https://github.com/JacobFV/old-tensorcode-2026-09-20)
at checkpoint `716056b`. Legacy APIs have no compatibility shim here.

[MIT license](LICENSE).

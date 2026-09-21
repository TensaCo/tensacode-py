# TensorCode

Compose cognitive operations in ordinary Python. Use vector, message, or graph
representations; import a tool when you want a ready-made composition; trace a
program when you want to inspect or train it.

**0.2.0a1 is a breaking, from-scratch replacement.** This first milestone ships a
small working subset of the [architecture](docs/a-new-hope/1-tensorcode-architecture.md).
It does not ship pretrained intelligence or a universal autonomous agent.

## Install from this checkout

```bash
python -m pip install -e '.[vec,dev]'
```

Python 3.11+. The base package has no third-party dependencies. PyTorch is optional
and loaded only by `tensorcode.ops.vec`. Importing TensorCode makes no network calls.

## Compose operations

```python
import torch
from tensorcode.ops.vec import TextEncoder, Classify
from tensorcode.tools.decision import Decision

encode = TextEncoder(vocabulary=('hello', 'refund', 'card'), dimensions=16)
classify = Classify(torch.nn.Linear(16, 2), labels=('question', 'refund'))
route = Decision(encode=encode, decide=classify)

prediction = route('refund my card')
print(prediction.value)          # random until trained
print(prediction.probabilities)  # model probabilities, not calibrated confidence
```

Operations use `op(value, *, context=None)`. Name instances according to their role;
there is no compulsory ontology or global model registry. Vector operations are
native `torch.nn.Module` objects with hooks, parameters, devices and gradients.
The text encoder uses explicit regex tokenization and trainable mean-pooled word
embeddings. No pretrained semantic capability is implied.

## Trace and train

```python
import tensorcode as tc

with tc.trace() as episode:
    prediction = route(('hello', 'refund'))

loss = torch.nn.functional.cross_entropy(prediction.logits, torch.tensor([0, 1]))
loss.backward()  # native autograd reaches encoder and classifier

port = episode.ref(prediction)
example = episode.example(port)
print(example.inputs)  # external inputs; internal tensors remain graph edges
fresh_prediction = episode.replay(port)  # recomputes with current parameters
```

An optimizer still owns parameter updates; see [the real-data example](examples/banking77.py).
Replay is in-memory and opt-in for effect-free operations. It preserves required
conditioning data and rejects known mutated intermediates. It does not serialize
programs, infer arbitrary Python computation, or differentiate remote calls.

For scalar outputs use `episode.calls[-1].output` explicitly; equal Python values
are never treated as proof of shared provenance. Passing an output reference to an
operation inside its session connects the edge and unwraps the actual value.

Traces keep live outputs for autograd and detached snapshots of external roots.
`example()` extracts a dependency closure; dropping the session can release its
intermediates, but the returned example is **not** a standalone persisted program.
Replay currently needs the session and the same operation objects. Treat traces
as sensitive in-memory application data; no automatic disk logging occurs.

## Use a chatbot tool

```python
from tensorcode.tools.agents import Chatbot

# Implement this adapter using your selected provider's SDK.
# It receives tuple[Message, ...] and returns the assistant's string response.
bot = Chatbot(model=my_model_function)
reply = bot('Hello')
```

The tool keeps per-instance history, serializes turns and commits only successful
responses. Supply `respond=...` instead of `model=...` to replace its message
operation; encoder and decoder are also replaceable. No provider is called until
you supply one. This milestone is text-only: images, learned objectives, long-term
memory and autonomous actions are not implemented.

`ops.llm` supplies immutable messages, text encode/decode, and a transform around a
caller-supplied model function. `ops.graph` supplies immutable source-tagged graphs,
neighbor lookup and caller-supplied transforms. These establish composition, not
learned graph reasoning. The graph of represented facts and the trace DAG are separate.

## Real-data learning example

Get the official train/test CSVs from
[PolyAI's Banking77 repository](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data),
then run:

```bash
python examples/banking77.py --train /path/to/train.csv --test /path/to/test.csv \
  --output /tmp/banking77-results.json
```

The script fixes its seed, learns its vocabulary only from training text, removes
literal train/test overlap, trains for 20 epochs, and reports fixed before/after
held-out accuracy and cross-entropy. It exercises tracing during learning and
recomputes the final batch through the captured DAG. Data and weights are not
bundled. See the [measured result](docs/a-new-hope/banking77-reset-results.json)
and [implementation report](docs/a-new-hope/3-reset-report.md) for limitations.

## Development

```bash
python -m pytest -q
python -m build
```

The former implementation, tests and research reports were deliberately removed.
They are recoverable from the private
[pre-reset archive](https://github.com/JacobFV/old-tensorcode-2026-09-20)
at checkpoint `716056b`; legacy APIs have no compatibility shim here.

[MIT license](LICENSE).

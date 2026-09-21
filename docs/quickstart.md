# Quickstart

TensorCode lets you compose operations over vectors, messages and graphs, then
trace and train supported local tensor paths. You choose the models and policies.

## Install

Python 3.11 or newer is required. From a checkout:

```bash
python -m pip install -e .          # messages, graphs, tools, HTTP integrations
python -m pip install -e '.[vec]'   # also install PyTorch vector operations
python -m pip install -e '.[local]' # also load explicit Transformers models
```

Imports do not download models or contact providers. For development, install
`.[vec,dev]` and run `python -m pytest -q`.

## Compose and train a decision

This complete example uses the `vec` extra. Its tiny training set demonstrates the
API; it is not an accuracy evaluation or a pretrained support router.

```python
import torch
import tensorcode as tc
from tensorcode.ops.vec import TextEncoder, Classify
from tensorcode.tools.decision import Decision
from tensorcode.training import Trainer

encode = TextEncoder(vocabulary=('hello', 'refund', 'card'), dimensions=16)
classify = Classify(torch.nn.Linear(16, 2), labels=('greeting', 'refund'))
route = Decision(encode=encode, decide=classify)

with tc.trace() as session:
    predictions = route(('hello', 'refund my card'))
session.supervise(predictions, ('greeting', 'refund'), source='example labels')

trainer = Trainer({'encode': encode, 'classify': classify}, lr=0.1)
losses = trainer.fit([session], epochs=20)
print('Initial and final training loss:', losses[0], losses[-1])
print('Predicted label:', route('refund card').value)
```

Name each callable for its role. The shared convention is
`operation(value, *, context=None)`. Vector operations preserve native PyTorch
parameters, hooks and autograd. The trace records dependencies between operation
calls; it does not infer arbitrary Python computation or remote gradients.

Continue with [tracing and training](training.md) for supervision, saving
experiences, loading in a new process and checkpointing.

## Connect a model

Message operations use an explicitly supplied model. This example needs a running
OpenAI-compatible server and its actual model ID:

```python
from tensorcode.integrations import OpenAICompatibleModel
from tensorcode.tools.agents import Chatbot

model = OpenAICompatibleModel(
    base_url='http://localhost:8000/v1', model='your-served-model',
)
bot = Chatbot(model=model)
print(bot('Explain the difference between a refund and a chargeback.'))
```

For a hosted server, supply its URL and an API key from your application's
configuration. Provider errors and invalid structured answers are surfaced; the
library does not silently substitute a different model.

## Choose your next example

Use the [application gallery](../examples/README.md) for actual ticket files,
document collections, source packages and images. See [operations](operations.md)
for representation contracts, [tools](tools.md) for stateful compositions, and
[troubleshooting](troubleshooting.md) for common failure modes.

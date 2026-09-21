# TensorCode

Compose vector, message and graph operations in ordinary Python. Use ready-to-call
tools, capture their dataflow, attach feedback, and train supported local tensor
paths—even after restarting your application.

Python 3.11+. The core package has no third-party dependencies. Models, policies
and optional frameworks are supplied explicitly; importing TensorCode makes no
network calls.

## Install from this checkout

```bash
python -m pip install -e .
python -m pip install -e '.[vec]'   # optional PyTorch operations
python -m pip install -e '.[local]' # optional local Transformers models
```

## Compose operations

```python
import torch
from tensorcode.ops.vec import TextEncoder, Classify
from tensorcode.tools.decision import Decision

encode = TextEncoder(vocabulary=('hello', 'refund', 'card'), dimensions=16)
classify = Classify(torch.nn.Linear(16, 2), labels=('greeting', 'refund'))
route = Decision(encode=encode, decide=classify)

prediction = route('refund my card')
print(prediction.value)          # randomly initialized until trained
print(prediction.probabilities)  # model probabilities, not calibrated confidence
```

Operations follow `operation(value, *, context=None)`. Tools compose public
operations; they do not require a global model, agent loop or domain ontology.
Tracing records operation dependencies, and training uses explicitly supplied
supervision. It does not make arbitrary Python or remote services differentiable.

## Learn and build

- [Quickstart](docs/quickstart.md): a complete composition and training example.
- [Documentation](docs/README.md): operations, providers, tools, persistence and troubleshooting.
- [Application examples](examples/README.md): learning agents, durable vector training, model-backed workflows and offline graph analysis.
- [Validation and limitations](docs/validation.md): measured learning results and model failures.
- [Tests](tests/README.md): subsystem coverage and verification commands.

## Development

```bash
python -m pip install -e '.[vec,dev]'
python -m pytest -q
python -m build
```

Library code lives in `src/tensorcode`; install the checkout before running examples.
This keeps repository files separate from the installed package. Public vector
operations are imported with `from tensorcode.ops import vec`.

Version **0.2.0a1** is a breaking replacement of the former implementation. Design
notes are preserved in Git history, including checkpoint `662feb4`; they are not
required to use the library. The older implementation remains in the private
[pre-reset archive](https://github.com/JacobFV/old-tensorcode-2026-09-20)
at checkpoint `716056b`. There is no legacy compatibility layer.

[MIT license](LICENSE).

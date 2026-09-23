# TensorCode

**Website:** [tensorcode.dev](https://tensorcode.dev) ·
**Docs:** [tensorcode.dev/docs](https://tensorcode.dev/docs/) ·
**Source:** [GitHub](https://github.com/TensaCo/tensacode-py) ·
[Changelog](https://github.com/TensaCo/tensacode-py/blob/main/CHANGELOG.md)

TensorCode builds trainable Python programs from callable operations and small
tools that own their models. You compose encoders, scorers and decoders
(`tensorcode.ops`), or use a complete tool such as `Investigator`, `Planner` or
`Chatbot` (`tensorcode.tools`). You collect reviewed feedback with explicit
provenance, train with PyTorch, and save everything as a data-only artifact that
reloads in a fresh process or from the Hugging Face Hub. Tracing records which
operation produced which value, so supervised local tensor paths can be replayed
and trained. It does not make arbitrary Python or remote model calls
differentiable.

> **Status: 0.4.0 alpha.** APIs may change between alphas. The core package has
> no dependencies and importing it does not import PyTorch or touch the network.
> Measured behavior and its limits are in [validation](docs/validation.md).
> Consistent benefits of the learned cognitive workspace are not yet established.

## Install

Python 3.11+. PyPI currently hosts only an older 0.1 alpha, so install 0.4 from
GitHub:

```bash
python -m pip install "tensorcode[tools] @ git+https://github.com/TensaCo/tensacode-py"
```

From a checkout, use `python -m pip install -e '.[tools]'`. Choose the extras
for the interfaces you use:

| Extra | Adds |
|---|---|
| `tools` | Owned models, training and Hugging Face loading (PyTorch, Transformers) |
| `vec` | Vector operations only (PyTorch, NumPy, safetensors) |
| `local` | Adapter for a local multimodal Transformers model |
| `diffusion` | `tools` plus diffusers for image decoders |
| `pretrained` | Alias of `tools` |
| `dev` | pytest, build, Pillow and PyArrow; the full test suite also needs `diffusion` |

## 30-second example

This trains an `Investigator` to rank two supplied hypotheses from log evidence.
It saves the model and reloads it. It runs offline on CPU in a few seconds.

```python
import torch
from tensorcode import training
from tensorcode.tools.investigator import Investigator

torch.manual_seed(0)
model = Investigator({"vocabulary": ["database", "network", "connection", "refused", "packet", "loss"],
                      "dimensions": 16, "slots": 2, "steps": 1})
trainer = training.Trainer.from_tool(model, optimizer=torch.optim.AdamW(model.parameters(), lr=0.01))

hypotheses = [{"id": "database", "text": "database connection refused"},
              {"id": "network", "text": "network packet loss"}]

def case(log_line):
    return {"question": "which component failed",
            "evidence": [{"source_id": "log:1", "text": log_line}],
            "hypotheses": hypotheses}

# Reviewed feedback, with explicit provenance, becomes training experience.
experiences = [trainer.capture(case("connection refused"), "database", source="review:1"),
               trainer.capture(case("packet loss"), "network", source="review:2")]
losses = trainer.fit(experiences, epochs=30)

model.save_pretrained("./investigator")
restored = Investigator.from_pretrained("./investigator")
print(restored(case("packet loss"))["selected_id"])  # network
```

Two authored cases show the lifecycle. They do not show that the model can
investigate anything. The result also includes every candidate's score and the
source-linked evidence. Probabilities are uncalibrated. The
[quickstart](docs/quickstart.md) extends this to persisted experience files,
resumable training checkpoints and loading in a fresh process.

## What is inside

- **`tensorcode.ops.vec`, `ops.text`, `ops.graph`**: operations with one calling
  convention, `op(value, *, context=None)`. Operations are built from JSON
  configuration, and learned vector operations own their weights. Text
  operations wrap an owned seq2seq model or an explicit external provider.
  Graph operations are reserved symbolic interfaces that raise
  `NotImplementedError`.
- **`tensorcode.tools`**: `Chatbot`, `Investigator`, `Decision`, `Planner` and
  `Scene`, complete trainable models with `save_pretrained` / `from_pretrained`.
- **`tensorcode.trace()` and `tensorcode.training`**: dependency capture,
  explicit supervision, `Trainer.from_tool` / `Trainer.from_ops`, portable
  experience and complete checkpoints.
- **`tensorcode.integrations`**: explicit adapters for OpenAI-compatible
  endpoints, local Transformers models and Jev.

Generated hypotheses are not evidence, and generated plans are not executable
code. Evidence, policies and actions stay explicit in your code.

## Guides

- [Quickstart](docs/quickstart.md): a runnable offline training lifecycle.
- [Developer documentation](docs/README.md): operation and model contracts.
- [Pretrained checkpoints](docs/pretrained.md): hosted tools and their measured
  scope. Saved artifacts must match the current architecture exactly: the hosted
  Chatbot and cognitive Investigator checkpoints need source commit `6607a8b`;
  only the Hotpot Investigator and Decision have refreshed revisions verified on
  current source.
- [Evidence and cognition](docs/cognition.md): hypotheses, revisions, memory and outcome feedback.
- [Training](docs/training.md): tracing, replay and resumable checkpoints.
- [Examples](examples/README.md): learning agents and practical applications.
- [Validation](docs/validation.md): measured behavior and limitations.
- [Updating development code](docs/migration.md): import changes since earlier alphas.

## Development

```bash
python -m pip install -e '.[tools,diffusion,dev]'
python -m pytest -q
python -m build
```

Library code lives in `src/tensorcode`. See the [test guide](tests/README.md) for
how the suite is organized. This alpha API replaces the former provider-owned
tools, and there is no legacy compatibility layer. [MIT license](LICENSE).

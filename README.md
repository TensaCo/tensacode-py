# TensorCode

Build trainable models from callable operations. TensorCode tools own their
encoders, learned workspace and output operations; a complete pretrained artifact
restores their configuration and weights without caller-supplied model callbacks.

Python 3.11+. Importing the core package does not import PyTorch or access the
network. Install the optional dependencies for the interfaces you use:

```bash
python -m pip install -e '.[tools]' # owned models, training and Hugging Face loading
python -m pip install -e '.[vec]'   # vector operations only
python -m pip install -e '.[local]' # external multimodal Transformers integration
```

After installation, complete pretrained text tools can be loaded from Hugging Face:

```python
from tensorcode.tools.chatbot import Chatbot

bot = Chatbot.from_pretrained(
    "jacob-valdez/tensorcode-chatbot-hotpot-001",
    revision="d74b40142c6e416cdc096f54e6d9d8c1de465568",
)
print(bot("Context: The sky is blue. Question: What color is the sky?"))
```

This checkpoint is a small evidence-conditioned QA fine-tune. See the
[checkpoint catalog](docs/pretrained.md) and [measured scope](docs/validation.md)
before choosing a model; consistent cognitive-workspace benefits are not yet
established.

## Initialize, train, restore

```python
from tensorcode.tools.investigator import Investigator

model = Investigator({"vocabulary": ["service", "database", "timeout"]})
# Fresh construction initializes weights. It does not download a model.
model.save_pretrained("./investigator")
restored = Investigator.from_pretrained("./investigator")
```

Use `from_pretrained` with a local directory or a Hugging Face model repository
containing a compatible TensorCode artifact. Saving random weights does not make
them useful: the [quickstart](docs/quickstart.md) adds sourced feedback, durable
experience, gradient training and fresh-process restoration. See
[validation](docs/validation.md) for measured checkpoint behavior and scope.

Operations live under `tensorcode.ops.{vec,llm,graph}`. Tools compose operations;
`tensorcode.runtime` contains explicit application infrastructure such as bounded
action loops and persistent memory. Symbolic graph operations are currently
interfaces that raise `NotImplementedError`.

## Guides

- [Quickstart](docs/quickstart.md): a runnable offline training lifecycle.
- [Developer documentation](docs/README.md): operation and model contracts.
- [Examples](examples/README.md): learning agents and practical applications.
- [Validation](docs/validation.md): measured behavior and limitations.
- [Tests](tests/README.md): subsystem coverage and verification.

## Development

```bash
python -m pip install -e '.[tools,dev]'
python -m pytest -q
python -m build
```

Library code lives in `src/tensorcode`; install the checkout before running
examples. This alpha API replaces the former provider-owned tools. There is no
legacy compatibility layer. [MIT license](LICENSE).

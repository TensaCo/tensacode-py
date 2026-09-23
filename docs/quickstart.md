# Quickstart

Install from GitHub (Python 3.11+):

```bash
python -m pip install "tensorcode[tools] @ git+https://github.com/TensaCo/tensacode-py"
```

From a checkout, run `python -m pip install -e '.[tools]'` in the repository root
instead.

This small offline example initializes an evidence-conditioned model, captures
reviewed feedback, trains from persisted experience and saves reloadable weights.
Its two authored cases demonstrate the lifecycle; they are not an evaluation of
investigation competence.

## Collect and train

Save this as `train_investigator.py` and run it from a writable directory:

```python
from pathlib import Path
import torch
from tensorcode import training
from tensorcode.tools.investigator import Investigator

# All parameters exist before the optimizer is constructed.
torch.manual_seed(7)
model = Investigator({
    "vocabulary": ["which", "component", "failed", "database", "network",
                   "connection", "refused", "packet", "loss"],
    "dimensions": 16,
    "slots": 2,
    "steps": 1,
})
trainer = training.Trainer.from_tool(
    model, optimizer=torch.optim.AdamW(model.parameters(), lr=0.01)
)
root = Path("investigation-run")
root.mkdir(exist_ok=True)
model.save_pretrained(root / "initial")

hypotheses = [
    {"id": "database", "text": "database connection refused"},
    {"id": "network", "text": "network packet loss"},
]
for index, (text, target) in enumerate([
    ("database connection refused", "database"),
    ("network packet loss", "network"),
]):
    inputs = {
        "question": "which component failed",
        "evidence": [{"source_id": f"observation:{index}", "text": text}],
        "hypotheses": hypotheses,
    }
    experience = trainer.capture(inputs, target, source=f"authored-example:{index}")
    experience.save(root / f"experience-{index}.json",
                    operations=trainer.operations, release=True)

experiences = [training.load_experience(path, operations=trainer.operations)
               for path in sorted(root.glob("experience-*.json"))]
losses = trainer.fit(experiences, epochs=60)
print("first / last loss:", losses[0], losses[-1])
model.save_pretrained(root / "model")
trainer.save_checkpoint(root / "training", progress={"epochs": 60})
```

The input encoder receives evidence and hypotheses, not the target label. The
label enters the supervised objective. Explicit source strings identify who
provided feedback; predicted scores are not observations.

## Load in a fresh process

Run this separately after the training program:

```python
from tensorcode.tools.investigator import Investigator

model = Investigator.from_pretrained("./investigation-run/model")
result = model({
    "question": "which component failed",
    "evidence": [{"source_id": "observation:new", "text": "network packet loss"}],
    "hypotheses": [
        {"id": "database", "text": "database connection refused"},
        {"id": "network", "text": "network packet loss"},
    ],
})
print(result["selected_id"])
print(result["candidates"])
```

The model ranks supplied hypotheses and returns source-linked evidence plus
workspace diagnostics. Probabilities are uncalibrated. A tiny fixed vocabulary
and two training cases do not establish generalization to new incidents.

`save_pretrained` saves model configuration and weights. `save_checkpoint` also
saves supported optimizer state, training progress and Python/PyTorch RNG state.
Experiences and chat sessions are separate artifacts. See [training](training.md)
for resume and replay contracts, [tools](tools.md) for Hub loading and chat, and the
[examples gallery](../examples/README.md) for larger workflows.

# Pretrained tools

Install TensorCode from this checkout with `python -m pip install -e '.[tools]'`.
Hugging Face is the preferred host. Tools construct their own encoders, workspace
and prediction/decoding components; callers do not supply a separate model.

| Tool | Checkpoint | Measured scope |
|---|---|---|
| `Chatbot` | [tensorcode-chatbot-hotpot-001](https://huggingface.co/jacob-valdez/tensorcode-chatbot-hotpot-001) | Small FLAN-based answer model trained with supplied supporting passages; not an evaluated general conversational assistant |
| `Investigator` | [tensorcode-investigator-hotpot-001](https://huggingface.co/jacob-valdez/tensorcode-investigator-hotpot-001) | Electra-based supporting-document ranking among supplied candidates |
| `Planner` | [tensorcode-planner-hotpot-001](https://huggingface.co/jacob-valdez/tensorcode-planner-hotpot-001) | Predicts document-read relevance; labels do not measure executed-plan utility |
| `Decision` | [tensorcode-decision-hotpot-001](https://huggingface.co/jacob-valdez/tensorcode-decision-hotpot-001) | Investigator weights through the Decision interface; same evaluation |
| `Scene` | [tensorcode-scene-vsr-experimental-001](https://huggingface.co/jacob-valdez/tensorcode-scene-vsr-experimental-001) | **Negative experiment:** spatial-caption prediction failed to establish useful visual grounding; retained for reproduction and further training |

The [release records](results/pretrained-releases.json) contain pinned revisions
and remote-loading checks. Model cards include evaluation records, training data
provenance and limitations. [Validation](validation.md) reports the baselines and
ablations: these checkpoints do not establish a consistent benefit from the
recurrent slot workspace or general cognitive competence.

## Load, call and save

```python
from tensorcode.tools.investigator import Investigator

model = Investigator.from_pretrained(
    "jacob-valdez/tensorcode-investigator-hotpot-001",
    revision="f329e9fe84560a4dc97ba6fea940696c6d7c379f",
)
result = model({
    "question": "Which document describes the sky?",
    "evidence": [],
    "hypotheses": [
        {"id": "document-a", "text": "The sky appears blue during the day."},
        {"id": "document-b", "text": "A kettle boils water."},
    ],
})
print(result["selected_id"])
model.save_pretrained("./my-investigator")
restored = Investigator.from_pretrained("./my-investigator", local_files_only=True)
```

A result is a learned prediction, not a verified fact. The complete model and its
encoding assets are saved locally. Session history and optimizer state are saved
separately; see [training](training.md) and [tool sessions](tools.md).

`from_pretrained` accepts `revision`, `cache_dir`, `local_files_only`, `token`, and
`device`. A local directory loads without contacting the Hub. `Tool(config)`
initializes fresh parameters without downloading anything. Explicit
`from_foundation` methods bootstrap inherited perception/language weights and a
new workspace for training; they are not equivalent to loading a trained
TensorCode checkpoint.

## Publish a fine-tuned model

```python
model.push_to_hub(
    "your-account/my-investigator",
    model_card="# My Investigator\n\nDescribe data, training, evaluation and limits here.\n",
)
```

Authenticate through the Hugging Face CLI or token configuration before publishing.
Publication includes model assets and the card, not conversation/session or
optimizer state. Supply actual evaluation and provenance in the card; the default
card identifies the architecture and makes no performance claim.

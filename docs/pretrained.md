# Pretrained tools

Install TensorCode from this checkout with `python -m pip install -e '.[tools]'`.
Hugging Face is the preferred host. Tools construct their own encoders, workspace
and prediction/decoding components; callers do not supply a separate model.

| Tool | Checkpoint | Measured scope |
|---|---|---|
| `Chatbot` (cognitive) | [tensorcode-chatbot-cognitive-experimental-001](https://huggingface.co/jacob-valdez/tensorcode-chatbot-cognitive-experimental-001) | Owned proposal generation, verification, realization and episodic retrieval; final 32-question run: one correct answer, one non-answer, 30 abstentions |
| `Investigator` (cognitive) | [tensorcode-investigator-cognitive-experimental-001](https://huggingface.co/jacob-valdez/tensorcode-investigator-cognitive-experimental-001) | Complete component extracted from the cognitive Chatbot; generates and screens hypotheses, owns retrieval encoder; no independent Investigator answer benchmark |
| `Chatbot` | [tensorcode-chatbot-hotpot-001](https://huggingface.co/jacob-valdez/tensorcode-chatbot-hotpot-001) | Small FLAN-based answer model trained with supplied supporting passages; not an evaluated general conversational assistant |
| `Investigator` | [tensorcode-investigator-hotpot-001](https://huggingface.co/jacob-valdez/tensorcode-investigator-hotpot-001) | Electra-based supporting-document ranking among supplied candidates |
| `Planner` | [tensorcode-planner-hotpot-001](https://huggingface.co/jacob-valdez/tensorcode-planner-hotpot-001) | Predicts document-read relevance; labels do not measure executed-plan utility |
| `Decision` | [tensorcode-decision-hotpot-001](https://huggingface.co/jacob-valdez/tensorcode-decision-hotpot-001) | Investigator weights through the Decision interface; same evaluation |
| `Scene` (language) | [tensorcode-scene-language-experimental-001](https://huggingface.co/jacob-valdez/tensorcode-scene-language-experimental-001) | Owned SmolVLM descriptions and spatial judgments; unverified, hallucinations observed, no fine-tuning |
| `Scene` | [tensorcode-scene-vsr-experimental-001](https://huggingface.co/jacob-valdez/tensorcode-scene-vsr-experimental-001) | **Negative experiment:** spatial-caption prediction failed to establish useful visual grounding; retained for reproduction and further training |

The [release records](results/pretrained-releases.json) contain pinned revisions
and remote-loading checks. Model cards include evaluation records, training data
provenance and limitations. [Validation](validation.md) reports the baselines and
ablations: these checkpoints do not establish a consistent benefit from the
recurrent slot workspace or general cognitive competence.

The listed Chatbot checkpoints and the cognitive Investigator use the earlier
unbounded workspace update. Reproduce them with source commit `6607a8b`; they are
not compatible with the current bounded-update architecture. Matching replacement
weights have not qualified yet.

The listed Hotpot Investigator and Decision pins also require source commit
`6607a8b` for historical reproduction. Their manifests omit the current explicit
`verification_scope`, `max_proposals` and `proposal_template_version` defaults, so
the current loader rejects them. Metadata-only replacements are being checked;
no replacement revision is qualified yet. Planner and Scene manifests show no
constructor-default drift in a configuration-only audit; that audit does not
establish current full-loading or prediction parity.

Current Chatbot configuration records `memory_update="relative_rms_bounded"`.
Each workspace update is normalized against its example's unmasked encoder RMS
and uses a bounded gate. This controls update magnitude, with floating-point
rounding tolerance; it does not establish useful learned reasoning. Loading
rejects configuration drift, including nested defaults and JSON value types,
before applying weights.

## Construct, call, save and reload

```python
from tensorcode.tools.investigator import Investigator

# Fresh random parameters: this demonstrates the current artifact lifecycle.
model = Investigator({
    "vocabulary": ["which", "document", "describes", "sky", "blue", "kettle", "water"],
    "dimensions": 32,
})
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

This fresh model has not been trained; its selected candidate demonstrates the
interface and is not evidence of ranking quality. The complete model and its
encoding assets are saved locally. Session history and optimizer state are saved
separately; see [training](training.md) and [tool sessions](tools.md).

`from_pretrained` accepts `revision`, `cache_dir`, `local_files_only`, `token`, and
`device`. A local directory loads without contacting the Hub. `Tool(config)`
initializes fresh parameters without downloading anything. Explicit
`from_foundation` methods bootstrap inherited perception/language weights and a
new workspace for training; they are not equivalent to loading a trained
TensorCode checkpoint.

## Reproduce the earlier experimental cognitive Chatbot

Use TensorCode commit `6607a8b` for this checkpoint. It includes the loader
correction that initializes fresh memory after restoring weights and retains the
architecture used for the recorded evaluation.
This complete checkpoint owns the generator, verifier, retrieval encoder and
realizer. It is suitable for inspecting and training the pipeline; its final
32-question evaluation produced only one correct answer, one non-answer and
30 abstentions. The [validation report](validation.md#complete-cognitive-pipeline-experimental-result)
explains that limitation.

```python
from tensorcode.tools.chatbot import Chatbot

bot = Chatbot.from_pretrained(
    "jacob-valdez/tensorcode-chatbot-cognitive-experimental-001",
    revision="8836ba59275dc6d8ceeb04462b4191beb9813452",
)
response = bot({
    "question": "How did the service recover?",
    "evidence": [{"id": "incident", "source_id": "report:17",
                  "text": "The service recovered after reconnecting the database."}],
})
print(response)  # May abstain; inspect the model judgments in bot.last_result.
bot.new_episode()  # Retains source memory, clears the current interpretation.
bot.save_session("./session.json")
```

The complete Investigator component is separately available at
`jacob-valdez/tensorcode-investigator-cognitive-experimental-001`, revision
`a5ee35f6c644fa13d4bda5850ea37eb32ec3f6d2`. Its `investigate` method generates
and screens hypotheses without the outer Chatbot realizer. The Scene language
checkpoint is pinned at `6aa55691bbdab5f49edede24ae0a71a673e2cc26`; see
[the visual interpretation interface](cognition.md#produce-unverified-image-interpretations).

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

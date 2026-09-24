# Pretrained tools

Install `tensorcode[tools]` as shown in the [quickstart](quickstart.md).
Hugging Face is the preferred host. Tools construct their own encoders, workspace
and prediction/decoding components; callers do not supply a separate model.

| Tool | Checkpoint | Measured scope |
|---|---|---|
| `Chatbot` (cognitive) | [tensorcode-chatbot-cognitive-experimental-001](https://huggingface.co/jacob-valdez/tensorcode-chatbot-cognitive-experimental-001) | Owned proposal generation, verification, realization and episodic retrieval; on 32 reused known questions: two correct answers, one incorrect answer, 29 abstentions |
| `Investigator` (cognitive) | [tensorcode-investigator-cognitive-experimental-001](https://huggingface.co/jacob-valdez/tensorcode-investigator-cognitive-experimental-001) | Complete component extracted from the cognitive Chatbot; generates and screens hypotheses, owns retrieval encoder; no independent Investigator answer benchmark |
| `Chatbot` | [tensorcode-chatbot-hotpot-001](https://huggingface.co/jacob-valdez/tensorcode-chatbot-hotpot-001) | Small FLAN-based answer model trained with supplied supporting passages (31/64 exact match); not an evaluated general conversational assistant |
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

## Current revisions and historical pins

All eight checkpoints load at `main` with TensorCode 0.4.0a4. The two Chatbots and
the cognitive Investigator were retrained on the current bounded workspace update
and published as new commits on their existing repositories. Their earlier
unbounded-update revisions are unchanged and still load with source commit
`6607a8b` only.

| Repository | Current revision (0.4.0a4) | Historical pin (needs source `6607a8b`) |
|---|---|---|
| `tensorcode-chatbot-hotpot-001` | `0f4b98f20295ea4638bc3f27579a5f79df0d6102` | `d74b40142c6e416cdc096f54e6d9d8c1de465568` |
| `tensorcode-chatbot-cognitive-experimental-001` | `6fc386fbfe992b8fb02c9049d4a761ed51c54132` | `8836ba59275dc6d8ceeb04462b4191beb9813452` |
| `tensorcode-investigator-cognitive-experimental-001` | `267b4f00d63905f0d8b9b87ba3542896d7c5e9e7` | `a5ee35f6c644fa13d4bda5850ea37eb32ec3f6d2` |
| `tensorcode-investigator-hotpot-001` | `1bc225917c3646fcb9702df91ff5e445846c1dc7` | `f329e9fe84560a4dc97ba6fea940696c6d7c379f` |
| `tensorcode-decision-hotpot-001` | `59b9f8d2c09e4d3179f601d54efafed1da45648d` | `0178292703fb096d687fabf1fd71aad5e2917e92` |
| `tensorcode-planner-hotpot-001` | `0cb4c3e7fa42a6d1679b2f0eee8182819294cc72` | none; unchanged |
| `tensorcode-scene-language-experimental-001` | `6aa55691bbdab5f49edede24ae0a71a673e2cc26` | none; unchanged |
| `tensorcode-scene-vsr-experimental-001` | `72bc6859f02382ddf7e14b627af2eb7ea7b5ae6f` | none; unchanged |

To reproduce a historical result, install TensorCode from source commit
`6607a8b` and pass the historical pin as `revision`. On that runtime the three
replaced pins still load and reproduce the recorded answers checked (three
held-out generations, the Franco answer and its Investigator receipt); the cognitive
receipt additionally records `verification_scope` and a different
`model_provenance` hash than the original 50f170e run.

The replacements are measured against the recorded results of the revisions they
replace ([Chatbot](results/chatbot-hotpot-bounded.json),
[cognitive Chatbot and Investigator](results/cognition-hotpot-bounded.json)):

- **Hotpot Chatbot:** the original recipe, data and seed on current source.
  Held-out exact match 31/64 and token F1 0.6171 (previously 30/64 and 0.6015).
  Bypassing the workspace gives 30/64 and a lower cross-entropy, so no workspace
  benefit is claimed.
- **Cognitive Chatbot:** retrained generator and realizer; verifier, ranker and
  retrieval encoder reused byte-for-byte. On the same 32 known questions it
  answers 3 and abstains on 29: two answers are correct and one is a wrong fact
  that the verifier accepted (previously one correct answer and one circular
  non-answer). The generator's foundation learning rate (5e-5 instead of 3e-5)
  was chosen after the test scores of all variants had been seen; the model card
  discloses this.
- **Cognitive Investigator:** the component extracted from that Chatbot, with a
  bitwise-equal state dict and equal `investigate` receipts on all 32 cases. It
  still has no independent answer benchmark.

Hotpot Investigator and Decision have [configuration-only refreshes](results/pretrained-configuration-refresh.json):
Investigator revision `1bc225917c3646fcb9702df91ff5e445846c1dc7` and Decision revision
`59b9f8d2c09e4d3179f601d54efafed1da45648d`. These explicitly record three existing
defaults. Published loading, weight bytes, full model state and three supplied-
candidate probe receipts match the historical runtime exactly. Their original
performance scope is unchanged; this is not new training or cognitive
qualification. Original pins in the historical release record require commit
`6607a8b`. Planner and both Scene checkpoints load at
their original revisions on current source; a [fresh-cache load and prediction
check](results/pretrained-catalog-check.json) found Python and TypeScript receipts equal (strings
exactly, numbers within 1e-4 absolute plus 1e-3 relative) on sample inputs.
That check does not re-measure their task performance.

Current Chatbot configuration records `memory_update="relative_rms_bounded"`.
Each workspace update is normalized against its example's unmasked encoder RMS
and uses a bounded gate. This controls update magnitude, with floating-point
rounding tolerance; it does not establish useful learned reasoning. Loading
rejects configuration drift, including nested defaults and JSON value types,
before applying weights.

To load the refreshed ranking checkpoint on current source:

```python
from tensorcode.tools.investigator import Investigator

model = Investigator.from_pretrained(
    "jacob-valdez/tensorcode-investigator-hotpot-001",
    revision="1bc225917c3646fcb9702df91ff5e445846c1dc7",
)
```

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

## Load the experimental cognitive Chatbot

This complete checkpoint owns the generator, verifier, retrieval encoder and
realizer. It is suitable for inspecting and training the pipeline. On 32 reused
HotpotQA questions with oracle passages it answered three (two correct, one
wrong fact) and abstained on 29. The [validation report](validation.md#complete-cognitive-pipeline-experimental-result)
explains these limits.

```python
from tensorcode.tools.chatbot import Chatbot

bot = Chatbot.from_pretrained(
    "jacob-valdez/tensorcode-chatbot-cognitive-experimental-001",
    revision="6fc386fbfe992b8fb02c9049d4a761ed51c54132",
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
`267b4f00d63905f0d8b9b87ba3542896d7c5e9e7`. Its `investigate` method generates
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

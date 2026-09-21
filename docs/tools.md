# Tools and pretrained models

The public model classes are `Chatbot`, `Investigator`, `Planner`, `Decision`,
and `Scene`. Tools own their trainable components. Import the concrete class, initialize its
architecture with configuration, or load a complete TensorCode artifact:

```python
from tensorcode.tools.investigator import Investigator

model = Investigator({"vocabulary": ["evidence", "hypothesis"]})
model.save_pretrained("./model")
model = Investigator.from_pretrained("./model", device="cpu")
```

Install `tensorcode[tools]` (or `pip install -e '.[tools]'` from the checkout).
Construction does not download assets or supply pretrained competence. Models
support PyTorch `parameters()`, `train()`, `eval()` and `to(device)`.

## Pretrained artifacts and Hugging Face

`from_pretrained(repo_id_or_path, *, revision=None, local_files_only=False,
cache_dir=None, token=None, device='cpu')` accepts a local directory or Hub model
ID. Pin `revision` to a commit for reproducible Hub loading. Offline loading needs
an existing local directory or cached snapshot. An ordinary Transformers model
repository is not a TensorCode tool artifact.

`save_pretrained(directory)` writes `tensorcode_config.json`,
`model.safetensors` and tool-specific assets. The manifest identifies a known
concrete class and format version; loading rejects incompatible artifacts rather
than importing artifact-selected Python code. The chatbot configuration includes
its tokenizer and complete foundation architecture, so restoration does not need
to download the original foundation.

`push_to_hub(repo_id, *, private=False, revision=None, token=None,
commit_message='Upload TensorCode model')` explicitly publishes model artifacts.
It does not publish sessions, optimizer state or collected experience. Supply
model cards and evaluation records alongside released weights to describe actual
training sources and measured scope. See [validation](validation.md) for the
available evidence rather than assuming a class name guarantees competence.

## Chatbot

```python
from tensorcode.tools.chatbot import Chatbot

# Run after saving or downloading a compatible TensorCode chatbot artifact.
bot = Chatbot.from_pretrained("./chatbot-model")
answer = bot("Help me investigate the evidence.")
print(answer)

other_conversation = bot.new_session()
other_answer = other_conversation("Start a separate investigation.")
bot.save_session("conversation.json")
bot.save_pretrained("./chatbot-model")
```

The model owns a tokenizing vector sequence encoder, learned workspace and local
`llm.Decode` language decoder. It conditions language generation on workspace
representations. It does not require a `model=` callback or remote provider.

`Chatbot(config)` constructs a fresh seq2seq architecture. Required keys are
`foundation_config` (a supported Transformers seq2seq configuration including
`model_type`) and `tokenizer_json` (a serialized fast tokenizer). Supply padding
through `tokenizer_special_tokens`. Optional limits include `max_input_tokens`
(default 512), `max_target_tokens` (128), `max_new_tokens` (64) and `max_turns` (16).
`workspace` configures slots and update steps.

`Chatbot.from_foundation(repo, revision=..., local_files_only=...)` is an explicit
training bootstrap: it imports pretrained seq2seq weights and initializes a new
workspace. It is not equivalent to a trained TensorCode chatbot. Supervise
`loss_batch(inputs, targets)` or use `ToolTrainer.capture` with equal-length text
lists. Targets enter teacher-forced decoding, not the input encoder.

`generate_batch(list_of_text)` is stateless. Calling `bot(text)` maintains its
default conversation; `new_session()` shares model weights but owns independent
history. A failed turn does not commit partial history. `last_result` reports
source evidence and whether token limits truncated the prompt. Saved sessions
contain conversation text and are separate from shareable weights.

The current tool is text-only. External multimodal integrations remain available
through message operations; they are not a trained multimodal workspace.

## Investigator and Decision

`Investigator(config)` accepts a nonempty unique string `vocabulary`, with optional
`dimensions=32`, `slots=4`, `steps=2`, and `max_tokens=256`. The model tokenizes with
its fixed vocabulary, encodes each evidence source, updates shared learned slots
and scores explicitly supplied candidate hypotheses.

```python
result = model({
    "question": "Which hypothesis best fits the evidence?",
    "evidence": [{"source_id": "report:1", "text": "Observed evidence"}],
    "hypotheses": [
        {"id": "a", "text": "First hypothesis"},
        {"id": "b", "text": "Second hypothesis"},
    ],
})
```

Feedback is a hypothesis ID, index, or a finite nonnegative distribution over
the supplied hypotheses that sums to one. Call again with revised evidence to obtain a
new interpretation. Returned candidates include scores and uncalibrated
probabilities; receipts preserve source IDs, attention and slot relations.
Attention is a model diagnostic, not proof of causal explanation.

`Investigator.from_foundation(repo, revision=..., local_files_only=...)` and
`Planner.from_foundation(...)` can instead bootstrap an owned contextual encoder,
including supported Electra models. The tokenizer and encoder configuration and
weights become part of the complete saved tool. `freeze_foundation=True` is the
default; set it to `False` to train the encoder as well. The workspace and scoring
head still start untrained. Inherited language representations must not be
reported as newly learned workspace behavior.

`new_session()` creates an independent history of interpretation receipts. Each
call supplies its **complete** current evidence; the session does not silently
accumulate it. Receipts add `previous_selected_id` and `revised`, making changes
visible without treating a changed choice as proof of improvement.

`tensorcode.tools.decision.Decision` uses the same owned architecture and input
contract. Its artifacts retain the concrete class identity. Ranking-only configurations require supplied candidates. Configurations with
owned generators and verifiers support the [cognitive interfaces](cognition.md);
none prove a hypothesis true.

A dedicated owned retrieval encoder can be included through
`from_foundations(..., retrieval_repo=..., retrieval_revision=...,
retrieval_options={"pooling": "masked_mean", "normalize": True, "max_tokens": 256})`.
It is saved in the tool artifact and used by episodic memory. Match the foundation's
pooling/token-limit contract; see [retrieval configuration](cognition.md#keep-revisions-and-retrieve-prior-evidence)
and [contrastive training](training.md#train-owned-retrieval).

## Planner

`tensorcode.tools.planner.Planner` uses the same configuration and workspace.
Inputs contain `goal`, sourced `evidence`, and `plans` with `id` and `text`.
It predicts scalar outcomes and selects the highest-scoring supplied candidate;
it does not execute actions.

Supervise an observed outcome with
`{"candidate_id": "plan-id", "outcome": 1.0}`. Alternatively, supply one finite
outcome per candidate when all were actually observed. Missing outcomes must not
be replaced with fabricated zeros. These are learned predictions, not causal or
counterfactual guarantees.

## Scene

`Scene` also supports an owned vision-language mode through
`from_language_foundation` and `interpret`; see [image interpretations](cognition.md#produce-unverified-image-interpretations).
The modes use distinct artifact configurations. The candidate-ranking mode below
ranks supplied descriptions against image pixels
and a question. It combines learned image patches, spatial position encodings,
text representations and the shared workspace. Construction needs a vocabulary
and accepts the same dimensions/slots/steps settings as the rankers, plus
`patch_size=8`, `in_channels=3`, `max_image_size=256` and `max_candidates=64`.
The visual encoder starts randomly initialized.

```python
from tensorcode.tools.scene import Scene

# Supply a compatible trained artifact and your own image tensor.
scene = Scene.from_pretrained("./scene-model")
result = scene({
    "pixels": pixels,
    "source_id": "photo:17",
    "question": "Which description matches the image?",
    "candidates": [
        {"id": "left", "text": "The cup is left of the plate."},
        {"id": "right", "text": "The cup is right of the plate."},
    ],
})
```

`pixels` is a finite floating CHW tensor in `[0, 1]`, with the configured channel
count and bounded image dimensions. Callers explicitly preprocess images. The
receipt includes selected candidate, scores, uncalibrated probabilities, source
ID, patch coordinates, attention and slot relations. Coordinates refer to the
input tensor, so resizing changes their relationship to the original photograph.

`ToolTrainer.capture(inputs, target, source=...)` accepts a supplied candidate ID
or index as feedback. The model predicts among those descriptions; it does not
generate a scene graph or autonomously discover a set of visual claims. Patch
attention shows model routing, not factual support or a causal explanation.
Use image and workspace ablations to test whether learned rankings depend on
visual evidence; see the [scene example](../examples/README.md#learn-from-images-and-relational-descriptions)
and [validation](validation.md).

## Generated interpretations and persistent cognition

See [cognition](cognition.md) for Investigator hypothesis generation and source-wise
NLI checks, cognitive Chatbot evidence/revision inputs, and Planner execution
feedback boundaries. These capabilities require their components in the loaded
artifact. Inspect `bot.capabilities` before using an older checkpoint as a
cognitive chatbot; loading weights does not add missing components. Cognitive
sessions can retrieve prior sourced evidence and persist raw episodic records.
With `cognition.memory` configured, successful turns automatically retain supplied
evidence; `new_episode()` starts a new context with that memory and
`rebuild_memory()` refreshes it after encoder training. Session
restoration rebuilds the saved evidence embeddings. The chatbot independently NLI-screens the
final decoded answer and enforces abstention on failed or truncated verification. These authored
model-score checks are not factual guarantees.

## Runtime infrastructure

`tensorcode.runtime` provides `ActionLoop`, `JsonMemory`, and `DecisionPipeline`
for applications with explicit policies and external effects. These utilities do
not constitute pretrained models. An action loop enforces its configured budget;
applications still supply the action implementations and authority. `JsonMemory`
coordinates writers within one process, not across processes.

For custom operation composition, use [operations](operations.md). For owned
model training and separate checkpoint/session lifecycles, use
[training](training.md).

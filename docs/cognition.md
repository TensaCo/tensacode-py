# Evidence, interpretations and action

A generated answer is not an observation. TensorCode's cognitive interfaces keep
source evidence, proposed hypotheses, model assessments and observed outcomes
separate. A useful application can revise an interpretation when a source changes
without rewriting what that source originally reported or treating fluent language
as verification.

These mechanisms do not establish general cognition. Checkpoint capabilities and
measured scope are documented in the [pretrained catalog](pretrained.md) and
[validation](validation.md). An older ranking-only checkpoint does not acquire new
generation or verification components merely by updating the library.

## Generate and check hypotheses

An Investigator can own a contextual encoder, proposal generator and three-way
NLI verifier. Load an artifact containing those components:

```python
from tensorcode.tools.investigator import Investigator

model = Investigator.from_pretrained("./investigator-with-verifier")
result = model.investigate({
    "question": "What explains the failed request?",
    "evidence": [
        {"source_id": "log:17", "text": "The database refused the connection."},
    ],
}, count=3)
```

Without supplied `hypotheses`, `investigate` generates candidate text. With them,
it assesses those candidates. `model.propose(inputs, count=...)` generates proposals
only. Generation can omit the correct explanation, duplicate ideas or hallucinate.
Generated candidates remain interpretations, never additional source evidence.

Each candidate's `verifications` reports a distribution for each evidence source.
The explicit labels are `support`, `contradiction` and `unknown`. They describe an
NLI model's assessment of a premise/hypothesis pair, not objective truth or source
trustworthiness. Receipts retain source IDs, model provenance, token truncation
and calibration fit status. Different sources may disagree.

`Investigator.from_foundations(encoder_repo, generator_repo, verifier_repo,
verifier_labels=..., encoder_revision=..., generator_revision=...,
verifier_revision=...)` explicitly bootstraps the complete composition. Supply
`verifier_labels` from the selected classifier's documented label ordering; do not
infer it from numerical indices. Newly initialized workspace/scoring components
still need training. Saving the tool preserves the owned models and tokenizer
assets for local reload.

## Ask a cognitive chatbot

A cognitive chatbot artifact owns an Investigator and its response decoder:

```python
from tensorcode.tools.chatbot import Chatbot

bot = Chatbot.from_pretrained("./cognitive-chatbot")
print(bot.capabilities)
answer = bot({
    "question": "What does the evidence suggest?",
    "evidence": [{"id": "connection", "source_id": "log:17",
                  "text": "The database refused the connection."}],
})
print(answer)
print(bot.last_result)

revised = bot({
    "question": "Does the revised evidence change the interpretation?",
    "revisions": [{"evidence_id": "connection", "source_id": "log:17:correction",
                   "text": "The connection succeeded; the request timed out later."}],
})
bot.save_session("./investigation-session.json")
bot.save_pretrained("./cognitive-chatbot")
```

These local artifact paths are prerequisites, not downloadable example model IDs.
Use a compatible checkpoint with the advertised components or explicitly bootstrap
and train one. `from_cognitive_foundations` constructs those owned components from
separately selected language, encoder, generator and verifier foundations.

Cognitive mode accepts a question plus optional sourced `evidence`, `revisions`
and `remove_evidence` (logical evidence IDs to deactivate).
A string is a question about current evidence; it is not automatically ingested as
a factual observation. The tool first constructs an interpretation, then asks its
own decoder to express it. An authored support/contradiction policy can abstain;
when it does, the chatbot enforces its configured abstention text. Policy thresholds
and abstention wording are application choices, not learned truth criteria.

The decoder receives the selected hypothesis and source text fitted to its token
budget, rather than the entire audit record. The chatbot independently screens
its decoded answer with source-wise NLI against the visible source text. If any
source was truncated, a passing answer is additionally checked against full active
evidence. Failed screening or a truncated NLI input enforces the configured final
abstention. Receipts
include `realization_sources`, `source_truncation` and
`realization_verifications` so callers can inspect what the decoder saw and how
its output was screened. `response_proposal` retains the raw decoded text as an
unverified proposal even when screening refuses it; it is never automatically
retained as evidence or a successful assistant response.

These checks use fallible model scores and authored thresholds. A passing answer
can still be wrong, misrepresent uncertainty or cite incorrectly. Evaluate final
language faithfulness separately from verifier classification quality. `new_session()` creates
independent session state using shared weights. A failed turn does not commit its
pending evidence revision or dialogue.

## Remember across chatbot episodes

Use an artifact configured with `cognition.memory={"capacity": 256, "top_k": 5}`.
This is part of the model configuration; callers do not need to wire runtime
components into each conversation:

```python
from tensorcode.tools.chatbot import Chatbot

bot = Chatbot.from_pretrained("./cognitive-chatbot-with-memory")
bot({
    "question": "What does this report say?",
    "evidence": [{"id": "incident", "source_id": "report:17",
                  "text": "The service recovered after restoring its database connection."}],
})
bot.new_episode()
# Explicitly retained source evidence can now be retrieved in a new episode.
answer = bot("What helped the service recover in the earlier report?")
print(answer)
print(bot.last_result["cognition"]["retrieval"])
bot.save_session("./remembered-session.json")
bot.save_pretrained("./cognitive-chatbot-with-memory")

restored = Chatbot.from_pretrained("./cognitive-chatbot-with-memory")
restored.load_session("./remembered-session.json")
```

`bot.new_episode()` (also available on independent chat sessions) clears current
conversation and active interpretation context while preserving episodic source
memory and immutable source records. Retrieval excludes the current episode by
default. `bot.cognitive_state` exposes immutable cognitive state for inspection.
After encoder training, `bot.rebuild_memory()` refreshes indexed embeddings;
changed model weights also invalidate earlier assessments and selections.
Saved session data includes source memory, while `save_pretrained` still saves
only the model. The example needs a compatible memory-enabled artifact; a
ranking-only or noncognitive chatbot checkpoint cannot implement this behavior.

## Keep revisions and retrieve prior evidence

```python
from tensorcode.runtime.cognitive_state import Evidence
from tensorcode.runtime.cognition import (
    CognitiveSession, LearnedEpisodicMemory, SelectionPolicy,
)

# model is an Investigator containing generation and verification components.
memory = LearnedEpisodicMemory(model, capacity=256)
session = CognitiveSession(model, memory=memory, policy=SelectionPolicy(
    min_support=0.7, max_contradiction=0.2, max_unknown=0.3,
))
session.ingest([Evidence("connection", "Connection refused.", "log:17")])
first = session.investigate("What explains the request failure?")
session.remember("connection", episode_id="incident:17", question="Request failure")
session.revise_evidence("connection", "Connection succeeded.", "log:17:correction")
second = session.investigate("What explanations remain?")
session.save("./cognitive-state.json")
restored = CognitiveSession.load("./cognitive-state.json", investigator=model)
```

Revisions retain the original immutable evidence and change which source revision
is active. Assessments tied to older state do not silently remain current.
`remove_evidence(stable_logical_id)` deactivates a source while retaining its raw
records and invalidating the current selection. The default state capacity is
256 total records, including assessment history; overflow fails explicitly rather
than silently dropping source evidence. Selection requires enough support and sufficiently low unknown score on the
strongest supporting source, and rejects excessive contradiction from any current
source. These default thresholds are explicitly authored policy.

`memory.retrieve(question, k=5, exclude_episode_id=...)` uses the owned encoder
and cosine proximity. When a CognitiveSession has memory configured,
`investigate` actively retrieves eligible past evidence, merges it into that
investigation's model input and records the hits in `receipt["retrieval"]`.
Revised, removed and already active source records are excluded. Pass
`episode_id=...` to `investigate` to exclude the current episode. Proximity is a
retrieval signal, not a claim that the source is trustworthy or true.

You can pass `memory={"capacity": 256, "top_k": 5}` to CognitiveSession instead
of constructing the memory object separately. Cognitive Chatbot's `cognition`
configuration accepts the same `memory` settings. For a directly constructed CognitiveSession, `remember` explicitly retains source
records. A cognitive Chatbot with memory configured automatically remembers
explicitly supplied active evidence after a successful response transaction.
Assistant answers, generated proposals and questions are never retained as source
observations automatically.

An Investigator may own a dedicated `RetrievalEncoder` in
`config["retrieval_encoder"]`, exposed as `model.episodic_encoder`. Episodic memory
uses this encoder when present; otherwise it uses the ranker's encoder. The
retrieval component owns its weights and fast tokenizer rather than requiring a
separate service or runtime callback.

Pass `retrieval_repo`, `retrieval_revision`, and
`retrieval_options={"pooling": "masked_mean", "normalize": True,
"max_tokens": 256}` to `Investigator.from_foundations(...)` to include one in an
explicit bootstrap. Match `max_tokens` and the pooling contract to the selected
foundation's model card. Only encoder-only, masked-mean pooling followed by L2
normalization is supported; CLS pooling, query prefixes, weighted pooling and
extra learned projections are not automatically reconstructed. Loading an
arbitrary language encoder is not evidence of contrastively trained retrieval.

For a retrieval-only bootstrap, use
`Investigator.from_retrieval_foundation(repo, pooling="masked_mean",
normalize=True, revision=..., vocabulary=[...], ...)`. Other configured components
initialize separately. The full tool artifact preserves this encoder and its
configuration. `model.episodic_encoder.receipt(texts)` reports embeddings, source
truncation and pooling/provenance metadata. Train it with explicit positive
query/document pairs using the [retrieval objective](training.md#train-owned-retrieval).

Stored embeddings are bound to a fingerprint of encoder configuration **and
weights**. After training, call `memory.rebuild_index()` before retrieval, or
`bot.rebuild_memory()` when using the opaque chatbot interface.
Standard optimizer and `no_grad` updates invalidate the fingerprint; unsupported
`.data` mutations require explicit `invalidate_fingerprint()`. Do not update
weights concurrently with retrieval.

Session snapshots now include raw episodic records and metadata. Restoring a
session rebuilds their embeddings with the supplied Investigator rather than
reusing vectors from old weights. Model weights remain a separate artifact.
Supervised experiences and resumable training checkpoints also remain separate
from session data.

## Learn from executed outcomes

Planner can own a language proposal generator alongside its outcome predictor.
`propose(inputs, count=...)` returns inert candidate text. Prediction does not run
action code. Applications explicitly map candidates to `ExecutablePlan` records
and provide a `PlanExecutor` action registry, bounded step budget and replanning
policy. Natural-language text is never automatically interpreted as executable
commands.

Each executed action produces an `OutcomeExperience` containing its candidate ID,
source ID, actual observation and status. `experience.to_target(observed_reward)`
labels only that candidate. The reward scale is explicitly supplied by the
application. An unexecuted alternative does not receive an invented zero or a
counterfactual outcome.

The [action-outcome example](../examples/learn_action_outcomes.py) demonstrates
actual transitions in an authored simulation, sourced feedback collection,
training, separate trajectory/session persistence and restored optimizer updates.
The environment and reward are fixtures, so its success does not establish
production planning competence.

## Produce unverified image interpretations

A Scene language artifact owns its vision-language foundation, processor assets
and visual workspace residual. It can describe an image without caller-supplied
caption candidates:

```python
from tensorcode.tools.scene import Scene

scene = Scene.from_pretrained("./scene-language-model")
receipt = scene.interpret({
    "pixels": pixels,  # finite RGB CHW float tensor in [0, 1]
    "question": "Describe the spatial relationships visible in this image.",
    "source_id": "photo:17",
}, max_new_tokens=128)
print(receipt["interpretation"])
print(receipt["verification"], receipt["completion_status"])
```

Install `tensorcode[tools,local]` for the owned VLM processor dependencies.
Use `Scene.from_language_foundation(repo_id, revision=...,
local_files_only=..., freeze_foundation=True)` to explicitly bootstrap a supported
Idefics3 VLM. This inherits the foundation's competence; the visual workspace
residual starts inactive and requires training. It is distinct from the
candidate-ranking Scene architecture and its checkpoints.

Every language receipt marks `verification='unverified'` and uncertainty as
uncalibrated with no numeric confidence. It retains the full-image source ID,
shape and content fingerprint, foundation provenance, workspace diagnostics and
completion status. `token_limit` means output may be incomplete. These are
fallible textual interpretations, not extracted facts, invented bounding boxes or
a symbolic scene graph. A nonzero workspace gate does not prove useful reasoning.

For this mode, `scene.loss(inputs, reviewed_target_text)` or
`ToolTrainer.capture(inputs, reviewed_target_text, source=...)` supervises language.
Save the complete model with `save_pretrained`; save experience and resumable
training checkpoints separately. Evaluate against real images with blank/shuffled
image controls and workspace ablations before attributing improvements to the
workspace rather than inherited VLM behavior.

## Calibrate without claiming certainty

Temperature calibration fits held-out model scores against reviewed class labels.
Owned verifier calibration records a weight digest and becomes stale after
tracked verifier weight changes or further verification training.
It leaves the underlying classifier weights and argmax unchanged. Calibration
metrics describe that sample; they do not turn NLI scores into probabilities that
a statement is true. Threshold selection trades empirical coverage against error
on supplied data and provides no guarantee on new inputs.

Use separate training, calibration and evaluation data. Recalibrate after changing
weights or materially changing the input distribution. See [training](training.md)
for the concrete calibration API and artifact boundaries.

## Symbolic graphs

`tensorcode.ops.graph` remains a set of symbolic operation interfaces. Text
encoding/decoding and other graph operations raise `NotImplementedError`. The
cognitive state records and learned workspace do not silently implement a graph
reasoner or restore legacy semantic rules.

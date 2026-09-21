# Task 5 — Composed decision and agent tools

Implemented a ready message decision tool, persistent retrieval-backed memory,
transactional multimodal chatbot state, and a bounded supplied-action loop. The
tool layer composes public operations and ordinary Python control flow. It does
not add an intent ontology, a language heuristic for actions, an automatic
memory relevance policy, or an implicit side effect.

## Public APIs

```python
from tensorcode.tools.decision import Decision

route = Decision(
    model=model,
    labels=("billing", "technical"),
    instructions="Route the support request.",
    selection_policy=optional_policy,
)
result = route("I was charged twice")
```

Ready mode constructs and exposes `llm.TextEncoder` and `llm.Classify` as
`route.encode` and `route.decide`. The classification result retains a provider
distribution and confidence only when the provider supplied them. The optional
selection policy receives that validated result and can return a replacement
result, including explicit abstention; its final non-abstained value is checked
against the configured labels. Existing explicit composition remains:
`Decision(encode=encode, decide=decide, selection_policy=...)`.

```python
from tensorcode.tools.agents import JsonMemory

memory = JsonMemory(
    "agent-memory.json",
    retrieve=my_retrieval_policy,
    encode_value=my_json_encoder,
    decode_value=my_json_decoder,
)
record = memory.append(value, kind="observation", source_id="source:42")
matches = memory.search(query, limit=5, kinds=("observation",))
```

`JsonMemory` passes `MemorySearch(query, candidates, limit)` to the supplied
retrieval callable and rejects results outside those candidates. It does not
claim the deterministic JSON store learns relevance. Generated IDs are stable,
monotonic `memory-NNNNNNNN` source IDs and skip caller-occupied IDs. Records
distinguish `observation`, `response`, and `objective` values. Values and
metadata are snapshotted through the configured JSON codec, so caller mutation
cannot alter live state or make it diverge from disk. Transactions stage all
records and use same-directory temporary files plus `os.replace`; failed
encoding or persistence leaves live records unchanged. Nested transactions are
rejected rather than silently losing an inner commit.

`JsonMemory.for_messages(path, retrieve=...)` configures a data-only JSON codec
for public LLM message sequences. It round-trips legacy string content,
`TextPart`, image bytes or URLs, media type, detail and source references. Image
bytes use base64 in JSON. The file never contains pickle data or executable
codec names; callers supply generic custom codecs again on restart.

```python
from tensorcode.tools.agents import Chatbot

bot = Chatbot(
    model=multimodal_model,
    encode_image=image_encoder,
    objective=initial_objective,
    update_objective=update_objective,
    memory=memory,
)
answer = bot("What is here?", images=[image_bytes])
answer = await bot.acall("And why?", images=[image_url])
```

Text and images pass through public `llm.TextEncoder` and `llm.ImageEncoder`
operations. Objective changes occur only through the supplied
`update_objective(ObjectiveRevision(current, observation), context=...)`
operation. Retrieval uses the memory store's supplied policy; selected message
records are exposed to the response operation as `context["memory"]`, and the
new objective as `context["objective"]`. These reserved keys cannot be silently
overridden by call context.

A turn computes an objective candidate, retrieval results, response and decoded
answer before committing history, objective or memory. Memory writes for the
observation, generated response and objective commit in one transaction. A
response or persistence failure keeps all three prior states. Mutable objective
values are copied across the speculative boundary so an in-place updater cannot
leak a failed revision. A responder can return the full extended transcript or
an assistant/tool suffix; the tool normalizes both to one history contract, so
live and restarted inputs are identical. `state_id` separates multiple chatbot
states in one store. Construction restores the matching history, last objective
and turn number unless `restore=False`.

Sync and async turns share one lock and therefore have a serial state order.
Async lock waiting is cancellation-safe, and synchronous codecs, retrieval
policies and callbacks run in worker threads during `acall` so they do not block
the event loop. The synchronous surface never returns an awaitable.

```python
from tensorcode.tools.agents import ActionLoop, ActionOutcome

loop = ActionLoop(
    chooser=choose_action,
    actions={
        "send": lambda state: ActionOutcome(next_state, receipt=provider_receipt),
    },
    max_steps=3,
)
result = loop(initial_state)
```

The chooser receives `ActionRequest(state, options, step, receipts)` and must
return an exact option string or a result with `.value` and `.abstained` (such
as `llm.DecisionResult`). Whitespace, case changes and invented names are not
coerced. Invalid choices fail before an action runs. Explicit abstention runs no
action. Every action must return `ActionOutcome(state, receipt, done)`; the loop
returns ordered `ActionReceipt` values and a stop reason of `completed`,
`abstained`, or `budget_exhausted`. It never invokes the chooser more than
`max_steps` times.

## Verification

The Task 5 suite covers ready and explicit decision construction, provider
distributions, replacement policy abstention and label bounds; memory retrieval,
restart, source IDs, JSON rejection, mutation isolation, nested transactions and
rollback; response/objective/memory atomicity; exact restarted history; serial
async turns, waiter cancellation and event-loop responsiveness; action choice,
abstention, receipts and budget; and an actual local HTTP server request carrying
PNG bytes through `Chatbot` and the OpenAI-compatible provider adapter.

```text
.venv/bin/pytest -q tests/test_agent_*.py tests/test_decision_tool_*.py tests/test_tools.py
33 passed in 1.25s

.venv/bin/pytest -q
174 passed in 9.12s
```

## Mechanism evidence and limits

The local HTTP server asserts the real provider request shape and returns a fixed
response. It proves multimodal serialization and tool/provider composition, not
that the fixture model understands the image. Scripted structured outputs prove
schema validation, distribution preservation and replaceable selection policy,
not decision accuracy. Actual VLM and classification quality are evaluated by
the separately owned real-model task.

Memory persistence is deterministic storage plus caller-authored retrieval, not
learned memory. Generic values need explicit JSON codecs; the built-in message
codec intentionally accepts only public message data. Atomic replacement and
locking cover one process; the store does not implement multi-process locking or
a distributed transaction. Chatbot rollback covers its own history, objective
and memory commit. It cannot undo external effects performed inside a supplied
model, objective transform, decoder, retrieval policy or codec. Similarly, an
action that raises after causing an external effect cannot be rolled back by the
action loop; effect idempotency and compensation remain application policy.

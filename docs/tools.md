# Composed tools

Tools compose public operations in ordinary Python. Models, objective changes, retrieval semantics, selection policies and actions remain explicitly supplied.

## Decisions

```python
from tensorcode.tools.decision import Decision

route = Decision(model=model, labels=("billing", "technical"), instructions="Route this request.")
result = route("I was charged twice")
```

Ready mode exposes `route.encode` (`llm.TextEncoder`) and `route.decide` (`llm.Classify`). Provider distributions and confidence are retained only when supplied. Optional `selection_policy(result)` can replace the validated result or abstain; a final non-abstained value must match the configured labels. Explicit composition remains available as `Decision(encode=encoder, decide=operation, selection_policy=...)`.

## Persistent memory

```python
from tensorcode.tools.agents import JsonMemory

memory = JsonMemory(
    "memory.json",
    retrieve=my_retrieval_policy,
    encode_value=my_json_encoder,
    decode_value=my_json_decoder,
)
record = memory.append(value, kind="observation", source_id="source:42")
matches = memory.search(query, limit=5, kinds=("observation",))
```

The retrieval callable receives `MemorySearch(query, candidates, limit)` and can return only those candidates. Storage does not learn relevance. Stable generated IDs skip caller-occupied IDs; record kinds distinguish observations, generated responses and objectives.

Values and metadata are snapshotted through the configured JSON codec. Caller mutation cannot change live state or create divergence from disk. Transactions stage records and atomically replace the file; encoding or persistence failure leaves live records unchanged. Nested transactions are rejected. Generic codecs are supplied again on restart; files do not contain executable codec names or pickle.

`JsonMemory.for_messages(path, retrieve=...)` supplies a data-only codec for public message sequences, including legacy text content, text/image parts, image bytes or URLs, media types, detail and source references. Image bytes use base64.

## Multimodal chat and objective changes

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

Text/images pass through public message encoders. Objective updates occur only through the supplied `update_objective(ObjectiveRevision(current, observation), context=...)` operation. Selected memory records become `context['memory']`, and the candidate objective becomes `context['objective']`; callers cannot override those reserved keys silently.

A turn computes objective, retrieval, response and decoded answer before committing history, objective and memory. Observation, response and objective records commit in one memory transaction. Mutable objectives are isolated during speculative updates. The responder may return the full extended transcript or an assistant/tool suffix; the tool normalizes both for consistent live/restarted history. `state_id` distinguishes chatbots sharing a store. Construction restores matching state unless `restore=False`.

Sync/async turns share one lock. Async waiters can be cancelled safely; synchronous callbacks run in worker threads during `acall`. Cancellation during commit settles that commit before releasing the lock. Rollback covers tool state, not external effects already performed by supplied callbacks. Locks and transactions cover one process, without distributed or multiprocess coordination.

## Bounded actions

```python
from tensorcode.tools.agents import ActionLoop, ActionOutcome

loop = ActionLoop(
    chooser=choose_action,
    actions={"send": send_action},
    max_steps=3,
)
result = loop(initial_state)
```

The chooser receives `ActionRequest(state, options, step, receipts)` and returns an exact option string or a result with `.value` and `.abstained`. Invented names, whitespace changes and case changes are not coerced. Invalid choices fail before execution; abstention executes no action.

Each supplied action returns `ActionOutcome(state, receipt, done)`. The loop retains ordered `ActionReceipt` values and stops with `completed`, `abstained`, or `budget_exhausted`. It invokes the chooser no more than `max_steps` times. Receipts record effects; they cannot undo them. Idempotency and compensation for external effects remain application policy.

Local HTTP fixtures verify actual image serialization and tool/provider composition, not image understanding. Actual model behavior and limitations are reported in [validation](validation.md).

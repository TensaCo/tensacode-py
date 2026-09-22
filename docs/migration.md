# Updating to the current development API

The current development branch simplifies public module boundaries. Removed
Python paths have no compatibility aliases. Model, experience and session data
formats remain separate from their implementation modules.

| Earlier API | Current API |
|---|---|
| `tensorcode.runtime` evidence records | `tensorcode.tools.cognition`; `Evidence` is also available beside `Investigator` |
| Direct cognitive-session and memory construction | `investigator.new_cognitive_session(policy=..., memory=..., max_records=...)` |
| Cognitive-session restoration with a supplied model | `investigator.load_cognitive_session(path)` |
| `runtime.PlanExecutor(...)` | `planner.new_executor(actions=..., replan=..., max_steps=...)` |
| `runtime.ActionLoop(...)` | `tools.actions.action_loop(chooser=..., actions=..., max_steps=...)` |
| Action callback records | `tensorcode.tools.actions` |
| Structured plan and outcome records | `tensorcode.tools.planner` |
| `tensorcode.tracing` | Root `trace`, `Trace`, `InputRef` and `OutputRef` |
| `training.ToolTrainer(model, ...)` | `training.Trainer.from_tool(model, ...)` |
| `training.Trainer(operations, ...)` | `training.Trainer.from_ops(operations, ...)` |
| `training.load(...)` | `training.load_experience(...)` |
| `runtime.SelectionPolicy` | `policy={...}` passed to `new_cognitive_session` |
| `runtime.CognitiveState` | Read-only `session.state` / `bot.cognitive_state` |
| `runtime.CognitiveSession`, `EpisodicMemory`, `LearnedEpisodicMemory` | `investigator.new_cognitive_session(memory=...)`; memory is tool-owned |
| `runtime.JsonMemory`, `MemoryRecord`, `MemorySearch`, message-sequence codecs | No replacement; storage is an application concern |
| `tracing.Session` | `tensorcode.Trace` (returned by `tensorcode.trace()`) |
| `Hypothesis(id, text, origin, provenance)` positional | `Hypothesis(id, text, origin, model_provenance=...)`; provenance is a required keyword |
| Free training checkpoint functions | `trainer.save_checkpoint(...)` and `trainer.load_checkpoint(...)` |

Investigator and Planner keep their existing ranking-history `new_session()`
behavior. Use the explicit cognitive-session factory for revisable evidence and
episodic retrieval. `InvestigationSession` remains available from
`tools.investigator` for annotations; normal construction belongs to the tool.
Chatbot keeps `new_session()` for independent conversations sharing its weights,
and `ChatSession` remains available from `tools.chatbot`.

The generic `DecisionPipeline` wrapper is removed: call your encoder and decision
operation directly, then apply any explicit application policy. General-purpose
storage remains an application concern; cognitive session factories own their
memory implementation. There is no replacement public storage-engine namespace.

Both trainer factories preserve their previous learning-rate defaults: `.001`
for a tool objective and `.01` for operation graphs, using SGD unless an optimizer
is supplied. The tool factory applies the model's training-mode policy. The ops
factory leaves module modes unchanged. Graph supervision still comes from
`trace()` and explicit `supervise(...)`; it is not inferred by `capture()`.

## Saved data

Canonical tool and operation identities, model parameter layouts, and unchanged
experience/session schemas are preserved. Loading does not import Python types
named by an artifact. Application dataclasses still require the same explicit
codec names bound to trusted current classes on save and load.

Complete directory training checkpoints retain their existing JSON/safetensors
envelope, including optimizer state, module modes, steps, caller progress and
supported random state. The older standalone `tensorcode.checkpoint` file contains
only model/optimizer state; loading it cannot recover metadata that was never
saved. New trainer saves produce complete directory checkpoints.

Use [training](training.md) for collect/replay/resume examples and
[cognition](cognition.md) for evidence revision and session restoration. Historical
evaluation reports retain the source revision and API used for their measurements.

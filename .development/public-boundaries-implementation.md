# Public boundaries implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development or
> superpowers:executing-plans. Work on main; the owner explicitly approved this
> design and full execution. Keep this document as the durable progress record.

**Goal:** Expose operations, owned tools, explicit training and tracing while
internalizing session, memory, execution and persistence implementation details.

**Architecture:** Public APIs describe developer activities. Tools construct their
sessions and execution helpers; developers retain public evidence/action contracts
and inspectable results. Training supports both complete tools and independent
operation graphs through explicit factories, without guessing objectives.

**Tech stack:** Python 3.11+, optional PyTorch/Transformers, safe JSON/safetensors.

**Spec:** The approved two-round design in this conversation, recorded below.
Baseline: `4981f9025cf11e49e78fbae921f1ffeea4b329de` (clean main).

## Approved contract

The public areas are `ops`, `tools`, `training`, `integrations`, and root `trace`.
`tensorcode.runtime` and the public tracing implementation module are removed.
Do not leave compatibility import aliases for removed paths.

Runtime mechanisms move by responsibility into `_internal.cognition`,
`_internal.sessions`, `_internal.memory`, and `_internal.execution`. Cognitive
evidence records shared across tools have a small public `tools.cognition`
contract module; investigation also exposes the commonly used Evidence contract.
Action callback records are public through `tools.actions`; structured planning
contracts are discoverable through `tools.planner`. These are records and
interfaces, not new models or hidden semantic policies.

Existing Chatbot sessions continue sharing model weights with independent state.
Investigator and Planner currently have ranking-history sessions. Preserve those
semantics; add `Investigator.new_cognitive_session(...)` and
`load_cognitive_session(path)` to own cognitive construction/restoration. The new
factory accepts JSON memory/policy options and a capacity limit, rather than
requiring callers to assemble storage or CognitiveState. Existing model classes,
weight layouts and operation identities remain in their canonical modules.

`Planner.new_executor(actions=..., replan=..., max_steps=...)` owns construction of
the internal executor; obtaining proposals never executes them. An explicit
`tools.actions.action_loop(...)` factory exposes the reusable bounded loop for
advanced orchestration without presenting it as a pretrained tool. Preserve
validation, receipts and effect boundaries. Remove DecisionPipeline and show
ordinary operation composition in its example/tests.

Root `trace()` returns an inspectable public `Trace` type; the implementation
moves to `_internal.tracing`. Public reference handle types needed for explicit
dependencies remain available at root. Trace capture/save/replay works without a
trainer. Low-level call/tree/capture bookkeeping stays private.

`training.Trainer.from_tool(tool, ...)` and `Trainer.from_ops(operations, ...)`
are the explicit construction paths. Both expose step/fit and complete checkpoint
save/load. Tool capture uses its declared objective; arbitrary operation graphs
use trace supervision and explicit losses. `capture` on an operation-only trainer
must give an actionable error rather than inventing an objective. Preserve
existing per-path learning-rate defaults and optimizer semantics. Public
`training.load_experience(...)` replaces the vague `load`; calibration remains a
small public advanced API. Remove public ToolTrainer and backend submodules.

Persist model artifacts, session state and optimization checkpoints separately.
Module relocation alone must not invalidate their data formats. Preserve format
identifiers and canonical tool/operation identities where schemas are unchanged;
never dynamically import artifact-selected Python. No heavy model retraining or
new cognitive performance claim belongs to this boundary refactor.
Existing standalone `tensorcode.checkpoint` files contain model/optimizer state
only; an explicit file load may preserve that limited contract but must not invent
RNG, modes, steps or progress. New trainer saves use complete directory checkpoints.

## Global constraints

- Main only, no worktrees or branches; preserve other work and commit verified milestones.
- The user's approved boundary change supersedes AGENTS.md's old public runtime placement.
- No compatibility modules, bundled semantic seeds or implicit execution of generated text.
- Graph operations remain explicit stubs; ops public paths and model identities stay stable.
- Public imports must retain optional dependency behavior; trace and training import without torch.
- Evidence revision, conversation isolation, source provenance and transactional restore remain intact.
- Explicit teacher-forcing targets may enter the decoder/objective, never the input workspace.
- Keep historical result JSON and archived runtime evidence immutable. Update executable examples,
  tests and development scripts to current APIs; the old source is retained by commits/archives.
- Heavy real-model validation, if necessary, uses GB10. No unqualified weight promotion.

## Review focus

1. Public interfaces must not require internal imports for evidence/action callbacks or session use.
2. Moving types must preserve same-object checks, configured codecs and persisted identities.
3. General operation checkpointing must support shared parameters and objects without module modes.
4. Restoring malformed checkpoints must not partially mutate weights, optimizer, RNG or modes.
5. Cold imports and installed-wheel examples must work without relying on source-tree aliases.

## Tasks

### Task 1: Runtime boundaries and tool ownership

**Own:** runtime relocation, tools session/execution factories and contracts;
runtime references in tests, examples and live development scripts. Do not edit
training/tracing implementations or their API imports in this task.

**Files:** remove `src/tensorcode/runtime/`; add responsibility-specific packages
under `src/tensorcode/_internal/`; add `tools/cognition.py`, `tools/actions.py`;
update chatbot.py, investigator.py, planner.py, decision/__init__.py, tools exports;
add `tests/models/test_public_tool_boundaries.py` and update existing runtime tests.

**Interfaces:** preserve existing model and ranking-session behavior; expose
`Investigator.new_cognitive_session(*, policy=None, memory=None, max_records=256)`
and `load_cognitive_session(path)`, plus Planner.new_executor and action_loop.
Sessions expose existing ingest/revise/remove/investigate/remember/retrieve/save
operations, with internal state and memory construction.

- [x] Add failure-first public lifecycle tests: construct from model + JSON options,
  revise remembered evidence, save/restore, and independent sessions.
- [x] Add public action/plan contract tests; factory must preserve pre-effect plan
  validation and bounded receipts. Confirm construction executes no callback.
- [x] Move implementations preserving relative imports and data schema; update all callers.
- [x] Replace DecisionPipeline example/test wiring with direct operation composition.
- [x] Verify `importlib.util.find_spec('tensorcode.runtime') is None`, public callback
  imports, runtime/cognition/model/example suites and unchanged tiny model artifacts.
- [x] Independent task review, full suite, commit/push verified milestone.

### Task 2: Internal tracing and explicit unified training

**Own:** root tracing exports, `_internal/tracing.py`, `_internal/training/`, public
training facade/calibration, all tracing/training imports/calls across live source,
tests/examples/development scripts. Preserve Task 1 tool signatures.

**Interfaces:** root Trace/trace/InputRef/OutputRef; public
`Trainer.from_tool(tool, *, optimizer=None, lr=.001)`;
`Trainer.from_ops(operations, *, optimizer=None, lr=.01, losses=None)`;
`.step(experience)`, `.fit(experiences, epochs=1)`, `.save_checkpoint(path, progress=None)`,
`.load_checkpoint(path)`, and explicit tool-only `.capture(inputs, targets, source=...)`.
`load_experience(path, operations=..., codecs=None)` preserves trusted codec handling.

- [x] Add failing factory/old-path tests, graph and teacher-forced update parity,
  explicit capture rejection for ops-only trainers and no accidental target leakage.
- [x] Relocate tracing and persistence engines; expose public facade types without
  changing bound operation identities or serializing private implementation paths.
- [x] Consolidate step/fit/checkpoint state through one facade with private objective
  adapters; preserve optimizer membership validation and shared parameter deduplication.
- [x] Test exact fixed-next-update restoration for tool and ops factories, non-module
  operations, mixed module modes, malformed restore rollback and trace dataclass codecs.
- [x] Update executable callers to factories/load_experience/checkpoint methods;
  low-level implementation tests may explicitly import internals.
- [x] Verify cold imports, all trace/training tests and examples; independent review,
  full suite, build, commit/push verified milestone.

### Task 3: Developer documentation and final boundary audit

**Own:** README, AGENTS.md, docs guides, examples README, development navigation and
this ledger; public boundary tests and installed-package smoke checks as needed.

- [x] Document compose/call/trace/train workflows with no public runtime imports or
  caller-assembled cognitive storage; include explicit action authority and inspection.
- [x] Explain model/session/checkpoint persistence and supported format continuity;
  removed Python imports are an intentional API break without compatibility shims.
- [x] Audit current public docs/examples for internal imports; distinguish optional
  experiment infrastructure from normal library usage and preserve historical reports.
- [x] Run full suite, wheel/sdist build, minimal-dependency wheel import/trace smoke,
  and representative installed public factory/session examples.
- [ ] Independent whole-change review, resolve material findings, commit/push,
  confirm CI and clean main. Summarize behavioral preservation and remaining cognitive gaps.

## Progress and decisions

- Plan recorded before implementation; owner approved full execution and subagents.
- Decision: keep ranking sessions and add a named cognitive session factory; silently
  changing `new_session()` would alter existing model behavior. The extra factory
  names two genuinely different interactions while hiding storage assembly.
- Decision: preserve unchanged data-format IDs and tool class identities. Python
  import removals do not themselves require destroying or relabeling model weights.
- Validation baseline from previous milestone: 878 local tests, builds and CI green.
- Pre-refactor fixtures at `/tmp/tensorcode-public-boundary-baseline`
  include model, experience, cognitive-session lineage, complete training state,
  and the exact expected next-update loss and weights. All were captured before relocation.

- Task 1 complete (code commits `3125d40..e8739ae`, independent review clean).
  Full suite: 886 passed; focused: 560 passed. Wheel/sdist build and a dependency-free
  installed wheel public-record/action/trace smoke passed. Pre-refactor model receipt,
  revised-session lineage, experience reload, next-step loss and all weights match exactly.

- Task 2 complete in `b6b2658`; independent review approved with no material findings. Full suite: 893 passed.
  Wheel/sdist built; new complete operation checkpoints use safetensors, now included
  in the vec extra. Both training modes share the private update/checkpoint engine.
  Pre-refactor model/session/experience and exact next-update parity passed again
  using the new public API and the installed wheel.
- Documentation migrated to public factories, with a migration guide explaining
  removed imports and saved-data continuity. The two quickstart Python blocks run
  successfully in separate processes. The dependency-free installed wheel passes
  trace/save/load/replay and action-loop checks, with no torch/transformers import.
- Example audit: ordinary library lifecycles use public APIs. Three existing
  qualification scripts retain private proposal-formatting/retrieval/experimental
  assessor helpers; examples/README.md explicitly identifies this development-only
  infrastructure. Historical results and cognitive quality claims remain unchanged.

- Task 2 review independently passed 53 focused tests (5 deselected), including
  teacher-forcing isolation and checkpoint failure handling. Installed wheel owned
  vector lifecycle passed capture/train/save/reload/resume. Runtime milestone CI
  passed on main (`4b63020`). Final whole-change review and final CI remain.

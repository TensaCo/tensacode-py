# 57 — Revision-bound empirical tasks

*Implementation checkpoint, 2026-09-19. This connects the empirical planning bridge in
[56](56-contingent-planning-from-experience.md) to retained task identity. The measured scope is bounded
execution of explicit empirical goals, not inferred user intentions.*

## From separate steps to one bounded task

Doc56 required callers to make a new plan and execution request for every step. The new
capability is a task that retains its explicit goal, performs bounded fresh planning and
execution, suspends safely, and resumes under the same identity without replaying earlier
actions. This is orchestration over an empirical model, not inference of a user's goal.

The API is `agent.pursue_empirical(model, goal=..., max_steps=..., choose=...,
**bounds)`, followed by later calls with `task_id` instead of a new goal. An `EmpiricalGoal`
supplies the target state, allowed calls, model identity, and provider. A supplied choice
callback may resolve ties among retained best first actions. No callback means ties defer;
there is no first-action fallback or automatic chat-default authorization.

Every step starts from fresh observed evidence and replans within the explicit bounds.
A supported intermediate observation can permit bounded suspension. A step receipt may
be verified as matching empirical support while the overall goal remains unmet: the
step-level and goal-level booleans must not be conflated. Only that controlled
suspension licenses automatic continuation of an attempt that already acted; uncertain,
failed, or potentially applied outcomes require explicit revision before retry. An exception
after dispatch must retain the applied receipt when available, or record an indeterminate
execution when it is not; a publication failure must not turn into permission to retry. Resume
reuses task identity and past receipts, not an assumed old world state.

## Goal revisions bind attempts and receipts

The active attempt captures the task revision and goal. Callbacks can revise the goal
while observing, choosing, or executing. A changed revision must stop dispatch of any
further old-goal step. An already applied action cannot be undone by relabeling its receipt.
Its step and attempt remain attributable to the revision actually attempted.

`TaskLedger.record(..., revision=captured_revision)` retains late outcomes without allowing
an old completion or failure to overwrite the new goal's status. The new revision remains
ready for its own attempt. Ledger snapshots and retained histories must preserve earlier
receipts rather than mutate or replay them. This is in-memory revision control, not
restart-safe transactional execution or automatic rollback of the external world.

The existing structured `Agent.pursue` path now uses the same captured-revision
attribution and checks revision before dispatch and between modeled steps. Both
task runners exclude concurrent or reentrant attempts on the same task and reread
its state after admission. A real filesystem regression revises a multi-step goal
after its first applied action: the remaining old file writes do not run, the
receipt stays under revision one, and revision two can subsequently execute.
Unexpected structured-execution errors recover retained receipts from the current
attempt's observation sources. Missing post-dispatch receipts are marked
indeterminate; uncertainty cannot silently restore permission to replay.

## Predeclared actual-environment audit

Reuse the actual FrozenLake reset/step fixture from doc56. Its authored map, projection,
goal, exploration, and explicit tie choices remain visible. The empirical edges still
come from retained executed transitions; this milestone adds lifecycle behavior rather
than claiming a stronger dynamics learner.

Two cases were fixed in this document before execution:

1. Start one task with a two-step budget, retain its suspended attempt, then resume with
   another two-step budget. Verify the same task ID and revision, two attempts, exactly
   four applied steps, no replay of the first two, and actual observed goal arrival.
2. Revise the task's goal during the first applied step of a separate run. Verify no second
   old-goal action, preserve the first receipt under its captured revision, and leave the
   new revision ready rather than completing it from the old attempt.

The report retains goal/task revisions, traces, step receipts, actual observations,
source hashes, and per-attempt limits. Distinguish an explicit callback deliberately
revising a goal from autonomous goal formation. Supplying a route preference does not
establish learned action preference or language grounding.

## Remaining boundaries

State semantics, goal states, allowed actions, exploration, model scope, and tie choices
remain authored. Empirical reachability covers observed support, not every possible
outcome. A resumed task can discover that its world or model is no longer usable and must
defer. This API does not supply a general scheduler, automatic conversation-to-task
conversion, durable recovery, generalized language/vision understanding, or full cognition.

## Recorded actual-environment results

[The report](../../eval/results/empirical_tasks.json) is reproduced with:

```sh
.venv/bin/python -m eval.learning.evaluate_empirical_tasks
```

It uses Gymnasium 1.3.0 and the same deterministic authored map and empirical edge-fitting
protocol as doc56. Each case independently collects 96 fitted transitions, split into
64 training and 32 evaluation attempts, for nine states and 32 eligible edges. These are
familiar observed state/action transitions; this checkpoint measures task lifecycle rather
than increasing transition generalization. All fourteen recorded source hashes match
both their post-run checks and the final runtime.

| Case | Applied actions | Retained attempts | Final task state |
| --- | --- | --- | --- |
| Pause then resume | right, right; down, down | two, both revision 1 | same task, revision 1, done |
| Revise during first action | right only | one, revision 1 | same task, revision 2, ready |

The paused attempt reports `suspended` with a two-step budget. Resumption creates fresh
proposal and execution identities, preserves the first attempt unchanged, and applies
only the remaining two steps. Four total actions reach observed `(8, True, False)`;
the resumed outcome verifies goal completion. No belief is added. The first task's
history contains one goal revision and two attempts, rather than four disconnected tasks.

In the revision case, an explicit test callback changes the target from `(8, True, False)`
to `(6, False, False)` after the first action is actually applied. The runner returns
`unknown` with `task_revision_changed` and dispatches no second old-goal action. Its applied
receipt and step remain in the revision-one attempt, whose goal is still the original
target. Revision two retains the new target and `ready` status; the old attempt neither
completes nor blocks it. The experiment does not attempt the new target or undo the first
movement, and it does not describe the callback as autonomous goal discovery.

All eighteen report checks pass: ten for pause/resume and eight for revision during
execution. Combined fixture collection and both task cases took 3.221 seconds on the shared
host; this is descriptive timing, not a performance comparison. Twelve focused actual-Gym
runtime tests passed before this audit, including callback revision, tied-choice deferral,
initial goal checks, unsafe-attempt retry behavior, and post-dispatch exception evidence.
The fixed measured cases were rerun after the exception-handling source freeze; the
original successful-path results were unchanged, and the report now identifies final code. Checkpoint-wide test results are
appended after full-suite verification.

## Checkpoint verification

The final implementation passed the full repository suite: **2,022 tests passed,
five skipped in 294.29 seconds**. Independent review reran all eighteen empirical
and structured task execution tests, including the two reproduced replay failures.
The actual-Gym audit passed all eighteen checks and all fourteen implementation
source hashes matched. These results establish bounded task continuation and
revision attribution for supplied goals; goal inference and general cognition
remain unfinished.

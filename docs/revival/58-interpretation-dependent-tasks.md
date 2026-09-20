# 58 — Interpretation-dependent tasks

*Implementation checkpoint, 2026-09-19. This connects selected workspace meanings to
task revisions, following [57](57-revision-bound-empirical-tasks.md). The measured scope is dependency invalidation and explicit
renewal; a dependency record alone is not learned intent.*

## The missing dependency

A task can retain a stable goal while the interpretation that justified that goal changes.
Guarding only the task's own revision misses that change. The new dependency captures the
selected interpretation's exact source, group, candidate set, selected candidate, revision,
and continuation status, together with an explicit authored basis and supporting source IDs.
It makes the commitment inspectable without claiming to infer the meaning-to-goal mapping.

`agent.capture_task_dependency(group_id, basis=..., evidence_ids=...)` requires a selected,
non-rejected candidate and no pending local continuation work. An absent or exhausted local
frontier does not prove semantic completeness. The dependency validates only while the
captured group and selection remain exactly current. Selecting the same candidate at a
new revision does not silently restore an older task's authority.

The task ledger stores dependencies on both the current task and each goal revision.
Revision preserves dependencies by default; replacing them requires an explicit fresh tuple.
Resume cannot silently replace a task's dependencies. Changes to unrelated groups must not
invalidate a dependency on an unchanged group.

## Dispatch and completion

Structured and empirical task execution check their supplied dependencies before dispatch
and completion, including callback-sensitive points. If the supporting interpretation is
withdrawn, rejected, expanded, or otherwise changes its comparison basis, further action
must defer. Batch validation first reads callback-bearing evidence, then checks every
dependency against callback-free scalar comparison snapshots. Repeating the full validation
a fixed number of times would still let the final callback invalidate an earlier check.
Task guards then read the task revision through a callback-free scalar accessor rather
than copying the goal again and reopening a callback window. This protects synchronous
callback boundaries, not a cross-thread atomic world transaction.
An already applied action remains in its original attempt; withdrawal does not
undo the external effect, erase receipts, or replay the action.

The automatic `Agent.turn` request path captures the selected interpretation when passing
the request through authored goal derivation. This connects an existing selected request
to its task; it does not provide a default interpretation selector or learn a goal from
arbitrary language. Capture occurs before handling and deictic resolution, against the
exact selected snapshot, and that captured dependency travels through handling and goal
derivation. A callback cannot change the selection and then authorize an old act by
recapturing the new meaning. A selected request with known pending interpretation work
also defers, even when an explicit selector chose it. Explicit structured callers still
supply their own dependency bindings.

If a verification callback raises after an automatic request has acted, recovery retains
the derived goal, actual applied receipt, and captured interpretation dependency. The
exception cannot erase the action history or turn the request into an unbound retry.
This preserves evidence of what occurred; it does not certify the interrupted verification.

## Predeclared actual-environment audit

Reuse the actual Gym FrozenLake fixture and empirical graph from doc56. The audit authors
a workspace source, candidate, selection, and explicit association with the goal. This
isolates interpretation-dependent task control; the candidate is not represented as a
learned language reading.

The fixed sequence is:

1. Capture a selected interpretation dependency and execute two steps of one empirical
   task, retaining its suspended attempt and receipts.
2. Change an unrelated group and confirm that the original dependency remains valid.
3. Withdraw the supporting selection. Resume must apply no action, preserve old receipts,
   and record a blocked attempt against the existing task revision.
4. Reselect the same candidate at a new workspace revision. The old dependency must still
   block action; candidate identity alone cannot reauthorize the old commitment.
5. Explicitly revise the task with the same goal and a freshly captured dependency. Resume
   from the current observed world, perform the remaining two steps, and complete without
   replaying either initial step.

The report retains the complete group/task histories, raw source, authored binding
basis, old/new dependencies, actual receipts and observations, and source hashes. It must
show that the fresh task revision retains old attempts instead of rewriting them.

## Scope limits

The mechanism propagates invalidation for explicit task dependencies; it does not discover
all semantic dependencies, revise arbitrary beliefs, replan every dependent task
automatically, or infer a replacement goal. Explicit recapture is a new caller commitment,
not evidence of semantic entailment. State, goal, action, exploration, and tie meanings
remain supplied, and all workspace/task persistence remains in memory.

## Recorded actual-environment result

[The report](../../eval/results/interpretation_tasks.json) is reproduced with:

```sh
.venv/bin/python -m eval.learning.evaluate_interpretation_tasks
```

The raw workspace source and candidate payload are labeled as authored audit inputs.
The selected candidate is explicitly associated with the supplied lake goal; no language
reader is credited with creating that association. The empirical graph still comes from
actual Gym reset/step observations, with 64 training and 32 evaluation samples as in doc56.

| Attempt | Task revision | Applied steps | Outcome |
| --- | ---: | ---: | --- |
| Initial bounded task | 1 | 2 | suspended, step budget exhausted |
| Resume after withdrawing meaning | 1 | 0 | unknown, interpretation dependency changed |
| Resume after reselecting the same candidate | 1 | 0 | unknown, interpretation dependency changed |
| Resume after explicit fresh dependency revision | 2 | 2 | done, goal observed |

The supporting group's revision advances from one at initial capture to three after
withdrawal and reselection. Candidate identity stays the same. The fresh task revision
explicitly captures revision three; the original goal revision continues to cite its
revision-one dependency. The task retains all four attempts under revisions `(1, 1, 1, 2)`.
Prior attempts are unchanged, and the final two steps continue from the already reached
world state. Exactly four actions execute in total; neither blocked attempt dispatches.

The unrelated group's change leaves dependency validation true. The original source,
selected-group history, task revision history, receipts, and observation records remain
inspectable. No belief is added. All thirteen audit checks pass, and all sixteen recorded
source hashes match their post-run checks and final code. The full audit took 2.460 seconds
on the shared host; this timing is descriptive, not a performance comparison.

Twenty actual-Gym empirical-task tests passed before the audit, including dependency
withdrawal before initial goal completion and withdrawal after an applied action. The
common dependency/workspace/task boundary verification passed 53 focused tests. These
suites overlap with the combined 116-test boundary run, which passed in 26.05 seconds,
and are not a learned cognition score. The measured protocol was rerun after the final
pre-handling capture and scalar-version guards; successful-case behavior stayed unchanged.
The automatic-request exception recovery subset passed 30 tests in 7.72 seconds. The
fixed actual-Gym audit was refreshed again against that final runtime; it does not itself
exercise the callback exception, which is covered by those regression tests. Final
full-suite verification after exception recovery and the concurrent chatbot fixes
passed **2,067 tests, with five skipped, in 290.47 seconds**. The frontend regression
harness passed separately; neither result measures generalized cognition.

Generalized language, vision, autonomous goal discovery, and full cognition are not
established by this slice.

[Semantic preservation at the goal boundary](59-semantic-preservation-at-the-goal-boundary.md)
addresses a separate failure: a current selected reading could still be converted into
a goal that drops polarity, qualifications, or roles. Stable dependency identity does not
prove meaning-preserving goal derivation. The next checkpoint blocks the identified lossy
conversions while leaving unsupported semantics explicitly unresolved.

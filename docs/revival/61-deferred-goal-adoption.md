# 61 — Deferred goal adoption

*2026-09-19. This extends [60](60-lexical-goal-interpretations.md) from retaining unresolved
goal choices to accepting a later explicit caller decision. This boundary does not
imply natural-language clarification.*

## Continue an unresolved commitment without repeating its input

A request can already retain its source reading, alternative lexical goals, and a task
whose goal remains unresolved. Requiring the caller to submit the same language again
would create a new interpretation problem and could lose the relationship to the earlier
task. The adoption path instead uses the existing goal comparison and task identity.

The caller explicitly identifies the retained goal candidate it intends to adopt. Adoption
does not parse another utterance, infer that a conversational reply selected a candidate,
or learn a meaning-to-goal correspondence. The request interpretation, lexical projection,
and domain refinement remain subject to their existing authored-policy limits.

Adoption and execution are separate. Adopting a goal must not invoke an action, replay a
prior request, or mark the goal achieved. It revises a retained commitment so that a later
explicit pursuit can plan and execute under the updated goal and dependencies.

## Explicit adoption API

```python
task = agent.tasks.get(task_id)
comparison = agent.interpretations.get(task.goal_interpretation_id)
# candidate_id is the caller's explicit choice after reviewing this comparison.
revised = agent.adopt_task_goal(
    task_id,
    InterpretationDecision(
        candidate_id,
        "caller explicitly chooses this retained goal",
        compared_revision=comparison.revision,
        compared_candidate_ids=tuple(candidate.id for candidate in comparison.candidates),
    ),
    reason="caller clarification of the existing task",
)
```

The task carries its `goal_interpretation_id`, so the caller cannot silently substitute a
newly searched comparison. The decision identifies a retained candidate and an explicit
reason and the exact compared revision and candidate IDs; the separate revision reason
explains the changed task commitment. Success returns
the revised task in `ready` state. Unavailable or stale semantic commitments return an
explicit `Unknown`; malformed call arguments are rejected.

Adoption performs no parsing, new lexical enumeration, perception, or action invocation.
It can run the same supplied domain refiners used by the request path. Conflicting or
unavailable refinements, unconsumed frame semantics, and unresolved lexical obligations
prevent task adoption. A refiner must represent remaining role obligations explicitly;
a caller choice alone does not discharge them.

A successful adoption keeps the parent reading dependency and captures the new goal
selection dependency. The task's explicit new revision carries that conjunction, while
old revisions keep their original dependencies and attempts. The later call to
`agent.pursue(task_id=task_id)` remains a separate execution request.

## Preserve the comparison and revision basis

The candidate must belong to the retained goal comparison linked to the task's original
request. The comparison's completeness and outstanding projection obligations still matter;
a later caller choice cannot make truncated enumeration complete or erase unsupported
construction semantics. Choosing the first visible candidate is not a fallback policy.

The original interpretation dependency must remain current. Adoption captures the selected
goal interpretation as a further dependency and revises the existing task explicitly.
Old goal revisions, attempts, source records, and receipts remain attributable to their
original commitments. A changed reading or goal comparison requires reconsideration;
same candidate identity at a different comparison revision is not silent authorization.

Callback-bearing evidence inspection, copying, or refinement must not let a stale decision
overwrite a newer task revision. Revision and comparison checks guard the transition from
the supplied decision to the revised task. These guards protect the synchronous in-memory
workflow; they do not provide a durable distributed transaction or undo external actions.
Goal selection/evidence can remain recorded even if refinement or the task revision then
fails. The API does not promise rollback of the goal comparison history.

## Acceptance boundaries

Focused tests demonstrate that a deferred task can adopt a later explicit retained
goal choice under the same task ID, without parsing or dispatching, while preserving its
earlier unresolved attempt. A separate pursuit may then act on the adopted goal. Rejected
cases include stale parent interpretations, unrelated comparisons/candidates, incomplete
goal enumeration, and structural obligations that a choice alone cannot discharge.

The examples and tests must label their caller decisions and supplied semantic fixtures.
This is an API for explicit clarification of a commitment, not learned dialogue
understanding, autonomous disambiguation, or an automatic chat response interpreter.

## Remaining limits

The lexical inventory and role/result projection remain authored, and missing upstream
construction restrictions remain missing. An explicit decision states caller intent but
does not certify that the selected conditions faithfully express every source qualification.
Domain refinement must preserve any remaining obligations before execution.

Task and workspace state persist only within the live agent. Adoption does not establish
restart recovery, automatic scheduling, generalized language or visual cognition, or a
policy for choosing goals on the user's behalf.

## Verification

The focused integration run passed 79 tests across adoption, goal interpretation,
task execution, refinement, and entity preservation. The final adoption file passes
11 tests, including two additional regressions: changing a completed task from one
retained goal to another preserves its actual earlier receipt and does not act until
explicit pursuit; withdrawing a parent interpretation during payload copying prevents
the task revision from committing. These use supplied semantic fixtures and caller
decisions, not learned clarification. The complete repository run passed 2,186 tests
with five skipped in 318.34 seconds. Its collection preceded the two final adoption
regressions; the final 11-test adoption run passed separately. Runtime code remained
unchanged between those runs.

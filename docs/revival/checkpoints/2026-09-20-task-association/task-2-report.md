# Task 2 report: authenticated complete task-context snapshots

## Outcome

Implemented the complete task-set snapshot boundary for conversational task
association. Capture now retains the selected complete incoming reading, every
current ledger identity/revision, current supported goal constraints, and each
task's authenticated historical originating sentence. A task without usable
origin evidence remains present as an unresolved member.

Live validation requires the exact complete ledger membership and every captured
revision. Historical validation relaxes only that live ledger equality: it accepts
later task insertion or revision after authenticating that the exact context was
previously registered, that every captured revision remains in ledger history,
and that all retained interpretation and grounding support is still live.

## API and data shapes

`TaskLedger.comparison_basis()` returns an ordered
`tuple[tuple[task_id, revision], ...]`. It executes under the ledger's `RLock` and
reads only dictionary keys and integer revisions, so user payload copy callbacks
cannot run while obtaining the final guard value. The order records the declared
candidate set; IDs, revisions, and order are authorization metadata and must not
enter learned pair semantics.

`capture_task_association_context(agent, group_id, candidate_id, *, basis)`
returns `TaskAssociationContext` or `Unknown`.

`validate_task_association_context(agent, context, *, historical=False)` returns
`True` or `Unknown`.

`TaskAssociationContext` fields:

- `context_id`: opaque retained-context identity
- `incoming`: exact `_CandidateSnapshot` for the selected complete sentence
- `incoming_dependency`: exact selected interpretation dependency
- `incoming_supporting_dependencies`: grounding dependencies
- `incoming_frames`: every ordered request frame
- `members`: one `TaskAssociationMember` for every ledger task
- `ledger_basis`: exact callback-free membership/revision comparison
- `basis`: caller-supplied explanation of the comparison

`TaskAssociationMember` fields:

- `task_id`, `task_revision`
- `goal`, `dependencies`, `goal_interpretation_id`: current comparison state
- `origin_task_revision`, `origin_goal_group_id`, `origin_goal_dependencies`
- `origin`: exact originating complete-reading snapshot
- `origin_dependency`, `origin_supporting_dependencies`, `origin_frames`
- `unresolved_reason`: nonempty when the member cannot supply a semantic pair

Task 3 can form a semantic pair only when `unresolved_reason is None`, using
`context.incoming_frames`, `member.origin_frames`, and `member.goal`. It must
retain every member identity for result mapping and explicit unresolved rivals.

## Authentication behavior

- Capture starts and ends with `TaskLedger.comparison_basis()`. Every callback
  bearing task/source/candidate/goal copy occurs between those checks.
- Incoming and originating readings use retained selected
  `SentenceAlternative` snapshots. `_frames` enforces complete ordered request
  acts, matching `Request`/`Frame` values, no skipped/unresolved content, and full
  learned-source anchors when present.
- Origin lookup walks immutable task revision history to the earliest usable
  retained goal group. It never reparses `Task.source` and never reads
  `agent.turns`.
- The retained goal source, group provenance, candidate set, selection dependency,
  parent sentence dependency, grounding/model support, and selected proposal goal
  are authenticated. Reusing a real dependency set for a different goal does not
  create trusted origin evidence.
- Whole-sentence goal teaching is supported: the goal projection may contain a
  `request-sequence` envelope, while `origin_frames` always come from the retained
  parent sentence snapshot.
- Current goal constraints are retained even when the current revision no longer
  names the original goal group. The historical origin revision remains separately
  authenticated.
- Unsupported goals, absent goal provenance, missing parent dependencies, stale
  support, and otherwise unusable origins remain explicit unresolved members.
- Both live and historical validation require registry equality. Setting
  `historical=True` cannot authenticate a caller-constructed context ID or altered
  payload.
- Historical validation permits later membership/revisions but authenticates the
  captured current revision and the distinct origin revision from ledger history.
  It continues to require incoming/origin interpretation and grounding support.

## TDD evidence

Initial RED:

```text
.venv/bin/pytest -q tests/test_task_association_context.py
10 failed
```

The failures were the expected missing module and missing
`TaskLedger.comparison_basis` API.

Additional focused RED checks:

```text
.venv/bin/pytest -q tests/test_task_association_context.py -k authentic_dependencies
1 failed
```

This showed that an authentic goal group/dependency set could initially be reused
with a different task goal. Capture now requires the selected originating proposal
goal to equal the historical task revision goal.

```text
.venv/bin/pytest -q tests/test_task_association_context.py -k historical_validation_authenticates
1 failed
```

This showed that an origin revision altered after a later current revision was not
being compared back to the retained goal proposal. Historical validation now
checks that correspondence directly.

Focused GREEN:

```text
.venv/bin/pytest -q tests/test_task_association_context.py
13 passed in 2.92s
```

Adjacent integration GREEN:

```text
.venv/bin/pytest -q \
  tests/test_task_association_context.py \
  tests/test_task_association_learning.py \
  tests/test_task_revision_learning.py \
  tests/test_task_revision_adoption.py \
  tests/test_goal_learning_evidence.py \
  tests/test_sentence_goal_teaching.py \
  tests/test_interpretation_tasks.py
85 passed in 18.86s
```

`py_compile` passed for both production files and the new test file.
`git diff --check` reported no whitespace errors. Per assignment, I did not run the
full suite, commit, or push.

## Test coverage

The new tests cover:

- an earlier task plus a later distractor
- all ordered incoming frames and every task identity/revision
- unsupported/missing-origin rivals remaining explicit
- identical task contents with distinct identities
- authentic dependencies paired with a mismatched goal remaining unresolved
- task insertion after capture: live invalid, registered history valid
- rival revision after capture: live invalid, registered history valid
- a rival revising itself during payload copy
- withdrawn origin support invalidating live and historical validation
- caller-created context IDs rejected in historical mode
- current goal constraints combined with the historical origin reading
- historical origin-revision goal tampering rejected
- whole-request goal projection envelopes sourcing origin from their parent reading
- callback-free, identity-sensitive ledger comparison

## Files

- `src/tensorcode/agent/tasks.py`
- `src/tensorcode/agent/task_association_context.py`
- `tests/test_task_association_context.py`

## Remaining scope and concerns

This task establishes an authenticated context boundary. It does not learn a task
relation, select a target, revise a task, create a new task, or dispatch an act.
Those later stages must preserve every unresolved member and pass the exact
`ledger_basis` through their final commit guards.

Task/member order is preserved for complete-set authorization. The pair learner
must consume each member independently and exclude task IDs, revision numbers,
creation order, source text, and `basis` from semantic features.

Interpretation and grounding support intentionally remains live for historical
teaching snapshots. A later task insertion does not erase an episode, while an
explicit withdrawal of the reading that supplied that episode does invalidate it.

All registries and the task ledger are in-memory and scoped to one `Agent`
lifetime, consistent with the existing architecture.

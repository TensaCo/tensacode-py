# Evidence-driven task association

Status: design for the next integration milestone; not an implemented capability.

[Contextual task correction](81-contextual-task-correction.md) learns how a
complete correction can revise a supplied task. The next boundary is deciding
which task, if any, the utterance concerns. This is part of interpreting the
conversation, not a scheduling convenience. The governing objective remains
[the structured cognitive workspace](36-structured-cognitive-workspace.md).

## Current execution gap

`Agent.turn` selects sentence readings, then dispatches individual acts.
`Agent.request` creates a task for each request. The complete correction “use
Documents, but keep the existing README” can therefore reach task creation as
separate requests. The new correction APIs avoid that loss only when their caller
explicitly supplies the target task and passes the complete reading.

Do not solve this with a list of correction words, a most-recent-task default,
task-ID ordering, or an exception for this acceptance sentence. An utterance can
refer to an earlier task, be compatible with several tasks, or lack sufficient
evidence to associate it with any task. Lack of a supported association does not
establish an intention to start a new task.

## Proposed learned relation

Evaluate the relation between the complete incoming reading and each candidate
task. The semantic input contains all ordered incoming frames, the task's retained
originating reading, and its current goal conditions and invariants. Shared
grounded references preserve equality across these structures. For example, a
preservation reference can distinguish the task whose invariant concerns that
same resource, but that association must be taught and validated rather than
implemented as a README-specific rule.

Initial supervised outcomes distinguish a supported revision target from an
explicitly excluded target. These labels describe the taught relation; they are
not inferred from the names of predicates or words. Generalization initially
substitutes reference identities while preserving other structure exactly.
Changed qualifiers, missing clauses, and unsupported contexts remain unresolved.

Use a dedicated association result, not a synthetic executable `GoalSpec` to
encode classification labels. Reuse the exact structural encoding and reference
correspondence machinery through a shared internal component with separately
tested adapters. Retain typed scalar distinctions, reference equality, learned
constants, validation separation, conflict evidence, and search exhaustion.

Evaluate every candidate in the declared conversational context. Preserve
supported targets, explicit exclusions, and unresolved rivals. Candidate order
must not change the semantic alternatives. An unsupported rival cannot disappear
merely because another task has positive evidence. Missing originating evidence
is an unresolved context, not grounds for silently omitting a task.

Pairwise evaluation avoids teaching a different tuple schema for every number of
tasks. An alternative whole-set correspondence would require unordered matching
or permutation closure; sorting opaque task identities is not a semantic solution,
and a fixed three-task limit would introduce an unnecessary restriction. Pairwise
relations do not capture every discourse phenomenon. Collective references such
as “both tasks” remain an explicit later representational requirement.

New-task intent requires its own positive interpretation evidence. Neither all
pairwise exclusions nor an empty task set establishes it. The integration must
preserve this distinction and expose an unresolved routing decision when evidence
is absent. It must not fall back to the current unconditional request creation.

## Evidence and adoption boundaries

Task IDs, revision numbers, creation order, and timestamps belong in dependency
metadata, not learned semantic features. Authenticate originating readings through
retained interpretation sources and task dependencies. Mutable presentation
records in `agent.turns` are not the evidence authority.

Retain the entire candidate membership comparison, every relevant task revision,
the incoming reading, and the admitted association model version. Adding a task,
revising a rival, or withdrawing its support can change the interpretation even
when the selected task itself is unchanged. Revalidate these dependencies before
using an association to propose or adopt a correction. After adoption, distinguish
historical association context from the new current task revision, as the
correction implementation already does.

Model training and held-out evaluation must separate conversational episodes,
not merely example IDs: pairs from the same episode share evidence. Keep the
original supervision identity when expanding one episode into candidate pairs.
Do not inflate independent support by counting those pairs as separate episodes.

Retained teaching episodes need historical authentication, not a requirement
that their old candidate sets remain the current ledger forever. Authenticate the
complete comparison at retention, preserve it with the taught relation, and
later validate its immutable source and historical task revisions. Collecting
another episode can add tasks without erasing the first episode's supervision.
Keep interpretation and grounding support live according to the admitted model's
evidence policy. A live routing proposal, unlike historical teaching, must still
match the current candidate set before it commits a task change.

Association proposal and selection do not create, revise, or execute tasks.
Sentence-level routing consumes the selected relation before the existing per-act
dispatch loop, then delegates to the retained correction proposal/adoption path.
All clauses remain together. Unsupported routing returns unresolved evidence
without creating partial tasks. Execution remains separately authorized.

The association handle must reach the correction proposal's provenance, the
adopted task's dependencies, and the ledger's final commit guard. Validating it
only when selecting a target leaves a gap in which a rival or model can change
before goal adoption. The explicit supplied-task API can remain available, but
automatic routing cannot drop its association handle and use that API as a
fallback after validation fails.

Positive new-task intent also requires a complete-sentence goal proposal. It
cannot authorize a return to per-act task creation. Retain and select one goal
over all request frames, or abstain if that whole input is unsupported. Create
the task with a final comparison guard before planning or dispatching actions;
the current request path's execute-then-create ordering does not meet this gate.

Before either creation or revision, task membership and all candidate revisions
must still equal the compared snapshot. After a successful commit, that snapshot
becomes authenticated historical context: the new task or revision must not
invalidate its own authorization, and unrelated later task creation must not
retroactively alter what was compared. Retain the exact precommit snapshot and
resulting commit identity. Reading, grounding, model, and selected interpretation
support remain live for subsequent execution, including support used to exclude
rivals. New challenges to that retained interpretation reopen it explicitly;
they are not inferred merely from later ledger membership changes.

## Required acceptance evidence

Use the actual cached reader with independent teaching episodes and supplied
grounding. Create two project tasks with different protected-resource identities;
the intended task must be the earlier task. Read the complete correction and
demonstrate learned association with that earlier task, followed by contextual
revision and explicit pursuit. Preserve its README and receipts and leave the
later task unchanged.

Repeat with candidate order reversed and with additional supported distractors.
Include two equally supported targets, an unsupported rival, changed clauses,
withdrawn evidence, a newly added task, and a rival revised after comparison.
None may silently produce a positional winner or dispatch using stale meaning.
Explicitly distinguish new-request evidence from absent association evidence.

The implementation report must separate supplied reference grounding, speech
labels, model admission, comparison policy, and action models from the relation
actually learned. Passing this gate would establish bounded task association,
not arbitrary discourse understanding or autonomous resource discovery.

## Implementation sequence

1. Extract and verify shared exact structured correspondence machinery; retain
   existing goal behavior through its adapter. Add a dedicated association learner
   and tests for reference transfer, pair conflicts, and independent supervision.
2. Add retained association supervision, model admission, and a complete task-set
   snapshot boundary. Test rival insertion, task revision, and support withdrawal.
3. Add retained alternatives and guarded association selection that feeds existing
   correction APIs. Demonstrate the actual-input two-task episode before changing
   ordinary conversation dispatch.
4. Integrate sentence-level routing with explicit unresolved and new-request
   evidence paths. Remove unconditional per-act task creation as an implicit
   discourse decision. Verify both the correction episode and independent new
   requests, including multi-act input, before claiming conversational routing.

Keep these changes separate from the current correction checkpoint so its full
verification remains attributable to one frozen implementation.

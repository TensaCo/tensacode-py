# Contextual task correction: the next integration gap

Status: implementation proposal, not an implemented capability.

The acceptance episode in [the structured cognitive workspace objective](36-structured-cognitive-workspace.md)
requires a correction to change an existing task while preserving its remaining
constraints and execution history. Literal units and supported conversions repair
one evidence boundary; they do not address this conversational behavior.

## What the active path currently does

`Agent.request` creates a task for every interpreted request. Its docstring
explicitly excludes natural-language task correction. `Agent.adopt_task_goal`
can select another proposal from that task's existing goal interpretation group,
but cannot attach a newly interpreted correction to the task. Structured callers
can revise a goal and pursue a named task; this supplies the revision externally.

`GoalCorrespondenceModel` learns exact structured frame-to-goal correspondences
with reference substitution. It does not receive the previous goal as input.
Consequently it cannot derive which previous constraints should survive a
correction from its existing training interface. Tests over preconstructed goal
alternatives establish adoption mechanics, not understanding of a new correction.

## Required behavioral change

A correction proposal must bind an existing task revision, retained correction
source, selected interpretation, and admitted contextual model. Its contextual
input includes the old goal and grounded correction; its output proposes a
revised goal. Reference correspondence may initially be learned from explicitly
taught examples, but words such as “actually,” “instead,” or “keep” must not become
production dispatch rules. Selecting which task is being corrected is a separate
interpretation decision, not “use the most recent task.”

Proposal and adoption must not execute actions. Adoption must recheck the exact
task revision and all interpretation/model dependencies, retain a new goal
interpretation group, and preserve earlier attempts and receipts under their
original revisions. A later explicit pursuit must observe current state and
replan. It must not replay the old plan merely because some action was previously
selected or reported complete.

Constraint preservation needs evidence. An initial implementation may learn whole
contextual goal correspondences from supplied teaching, keeping all non-reference
structure exact. It must report that limit rather than claim arbitrary constraint
editing or unseen paraphrase understanding. Conflicting teaching or an unsupported
context must leave alternatives unresolved.

The correction input must retain the complete selected utterance, including
multiple acts and conjunctions. Selecting only a destination-bearing frame from
“use Documents, but keep the existing README” would discard the preservation
clause before revision learning starts. An initial supported input class may be
an ordered tuple of exact request frames, provided every act is represented and
unconsumed fragments, skipped tokens, or unsupported meanings cause abstention.
The retained source and interpretation dependency still cover the full utterance.
This boundary must be checked against real reader output before its API is fixed.

An offline cached-reader probe found one complete two-act correction alternative
among six: provisional `use(object=Documents, manner=actually)` followed by
`keep(object="the existing README")`. The README's definite article and
`existing` modifier remain represented, although `existing` is parsed as a noun
compound. The conjunction remains syntax evidence. The initial project request
has one destination-bearing alternative among eight. Both sources retain complete
non-whitespace token-anchor coverage. These are unselected provisional meanings;
they do not establish preservation semantics. Speech teaching must include both
act positions in one admitted model, rather than overwrite one with the other.

## Acceptance evidence

Use retained real parsed inputs for an initial request and a later correction,
with independently varied grounded references for training and held-out tests.
Demonstrate that the correction changes the same task, carries the original
preservation invariant, retains completed-action receipts, and executes only the
remaining work after a fresh observation. Include a competing destination,
withdrawn correction/model evidence, and a task revised between proposal and
adoption. None may silently commit stale meaning or repeat completed mutations.

Report supplied task selection, reference bindings, action models, teaching
labels, and interpretation-selection policies separately from learned behavior.
The full Documents/README acceptance episode remains unproven until these paths
operate together; a new correction record or structured-call test alone is
insufficient.

## Implementation order and ownership boundaries

1. Add a pure contextual learner in `learning/goal_revision.py`, using the
   existing correspondence search rather than another semantic rule engine.
   `GoalRevisionExample(id, previous, corrections, revised, basis)` holds a
   previous `GoalSpec`, all ordered correction `Frame` values, and a supervised
   revised `GoalSpec`. `fit_goal_revisions(training, validation, max_pairs=256)`
   returns a model with `propose(previous, corrections)`. Its internal context
   combines previous conditions/invariants and all correction frames. Goal labels
   and explanatory basis are metadata, not conditions. Tests must show held-out
   reference transfer, invariant preservation, abstention for changed qualifiers
   or missing clauses, conflicting teaching, and bounded search exhaustion.
2. Add retained supervision and model admission in
   `agent/task_revision_learning.py`. Every example must name the original task
   revision and full selected correction reading. Authenticate snapshots at fit
   and publication, retain teaching/validation sources, and keep model admission
   separate from selecting a proposed correction. Reject unsupported or partially
   consumed readings; never project only one convenient act.
3. Add `agent/task_revision.py` for proposal and adoption. A proposal binds task
   ID/revision, complete correction dependency, prior dependencies, and admitted
   model version. Adoption uses an explicit comparison decision, validates the
   retained group and dependencies, then calls the ledger's guarded revision
operation. It must not parse again, choose another task, or execute. Tests
   cover stale comparisons, concurrent task changes, withdrawal, forged groups,
   and preservation of original receipts.
4. Add a real-input integration test, followed by explicit pursuit against the
   filesystem action model. Supply the task association, teaching labels and
   reference bindings explicitly. Verify the final files, unchanged README
   bytes, same task identity, revised conditions, old receipt history, and no
   repeated completed action. Report reader alternatives that still cannot be
   consumed faithfully rather than dropping them to obtain a positive test.

Each implementation task starts with a failing behavioral test and ends with
focused verification. The contextual learner is an intermediate component;
the milestone is not complete until retained correction evidence actively
changes the same task and governs subsequent execution. Existing quantity
verification stays isolated from these new files until its checkpoint is saved.

Admission and later validation have different task-version checks. Before
adoption, the current task must still equal the proposal's previous revision.
After adoption, that previous revision is historical evidence: requiring it to
remain current would invalidate the newly adopted task immediately. Authenticate
its retained historical snapshot while keeping the correction/model commitments
live; compare the current revision separately when proposing the next correction.

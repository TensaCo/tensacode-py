# Contextual task correction and resource realization

Status: explicit correction path implemented and verified. The full suite passed
2,812 tests with 2 skipped; both real-input integration cases pass. Automatic conversational task association is absent.

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

The new `task_revision_learning` and `task_revision` paths now add contextual
teaching, admitted models, retained full-correction proposals, and guarded adoption
on the existing task. Callers explicitly associate the correction with a task and
select the retained reading and goal proposal. This does not change the default
`Agent.turn` routing into an autonomous detector of conversational corrections.

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
The explicit-association Documents/README episode is exercised by the real-input
integration described below. The broader acceptance episode remains incomplete:
automatic task association and independently inferred grounding are still absent.

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
changes the same task and governs subsequent execution. The literal-unit checkpoint is committed separately; the full-suite run verifies
its integration with these correction changes.

Admission and later validation have different task-version checks. Before
adoption, the current task must still equal the proposal's previous revision.
After adoption, that previous revision is historical evidence: requiring it to
remain current would invalidate the newly adopted task immediately. Authenticate
its retained historical snapshot while keeping the correction/model commitments
live; compare the current revision separately when proposing the next correction.

## Resource realization required by the acceptance test

Before this milestone, the filesystem adapter consumed literal strings and `Path`
objects, while reference correspondence generalized grounded `Ref` identities.
A probe confirmed that a reference-valued filesystem goal could not be planned.
Replacing those references with guessed strings would sever the grounding path.

Extend the adapter with explicitly supplied, evidenced resource bindings. Preserve
symbolic paths in goal conditions and action arguments; resolve them only for
filesystem operations and observations. A structural path can retain a root
reference plus an exact relative path, allowing the planner's parent operations
to preserve reference identity. The binding is supplied knowledge, not inferred
from “Documents,” entity text, or the spelling of a reference ID.

Reject missing, withdrawn, conflicting, or unsupported bindings and recheck them
at dispatch. A resource reference must not be rebound silently to another path
after a plan has retained it. Existing confinement and symlink checks continue
to apply after resolution. Tests must show actual effects under the intended
resource and refusal after binding withdrawal, as well as literal-path regression
coverage. This operational bridge does not establish autonomous resource grounding.

## Implemented APIs and observed behavior

`retain_task_revision_example` binds a supervised revised goal to an exact prior
task revision and complete selected correction. `fit_task_revision_model` checks
authentic examples and disjoint training/validation task and source identities.
`admit_task_revision_model` separately selects an authenticated version. Successful
fits retain historical supervision; later edits to teaching tasks do not silently
retrain them.

`propose_task_revision(agent, task_id, correction_group_id,
correction_candidate_id, model=..., basis=...)` retains every learned alternative
and unresolved result. `adopt_task_revision(agent, task_id, group_id, decision,
reason=...)` requires an exact comparison decision, rechecks current support, and
uses guarded ledger revision. It neither parses nor executes. Prior dependencies,
attempts, and receipts remain attributable to their original revisions. An
explicit subsequent `Agent.pursue(task_id=...)` observes and replans.

`FileSystemPlugin.bind_resource(resource, path, binding=..., evidence=...)` records
supplied realization evidence in `plugin.resources`. Bare resource references
and `{'root': reference, 'relative': 'hello-world/main.py'}` stay symbolic in goals
and calls. Ancestor actions retain a typed resource/ancestor-count value rather
than losing their binding dependency when converted to a literal parent path.
Withdrawal through the resource store disables retained calls. A reference cannot
be redirected to a different path during the plugin's lifetime. Final dispatch
batch-validates support after precondition callbacks and checks the retained
resource records again before mutation.

This first resource boundary fixes a reference to one path for the plugin's
lifetime. It does not yet model relocation of the same resource with a newly
selected realization version. Supporting that requires planned calls to retain
the selected binding version so that a later update cannot redirect old actions.
Creating a new reference is not a general solution to identity through relocation.

The actual-input integration exposed two speech-learning gaps. Distinct act
positions within one utterance can now share teaching evidence, while source and
normalized-text identities remain disjoint across training and validation. A
source/act occurrence cannot be duplicated to inflate evidence, and sibling acts
must retain the same source syntax. Multiword entity text can now compose existing
lexical slots with exact literal spaces. These slots share bindings with source
tokens; an unrelated phrase cannot fill an independent phrase wildcard. Other
separators and changed syntax do not acquire untested equivalences.

The final real-input integration passed **2 tests in 230.51 seconds** with
`.venv/bin/pytest -q tests/test_learned_task_revision_inputs.py`. After one
completed filesystem step, the agent reads the new correction, adopts a learned
contextual revision to Documents, and replans. Subsequent pursuit creates the
intended file, preserves exact README bytes, and retains the original receipt
without repeating its action. The second case withdraws the learned communicative
interpretation after adoption and verifies that subsequent pursuit performs no
additional dispatch. These cases use the cached reader, complete multi-act speech
teaching, contextual learning, retained interpretation dependencies, and the real
filesystem adapter together.

The focused correction, adoption, speech, and filesystem set passed **99 tests**;
the existing real-input speech set passed **4 tests**. The complete
`.venv/bin/pytest -q` run passed **2,812 tests, 2 skipped, in 1,126.29 seconds**
(exit 0). Reader continuation is explicitly exhausted before selecting the
execution-authorizing reading; this establishes only exhaustion of the configured
local search, not completeness of possible meanings.

The learned behavior is supervised correspondence over complete syntax and goal
contexts. Task association, interpretation choices, labels, initial and revised
goal teaching, resource bindings, action models, and desired file content remain
supplied. Novel paraphrases, autonomous goal editing, resource discovery, and
unassisted task selection are not established by these tests.

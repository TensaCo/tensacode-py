# Learned document tasks

[Browser transition learning](72-browser-transition-learning.md) connected actual
browser evidence to prospective outcome predictions and counterexample suspension.
This milestone develops one task pursuit that uses those predictions: accept an
explicit desired measured outcome and selected grounded document action, predict
its outcome, dispatch only under current guards, assess actual feedback, and
record completion or uncertainty in the task ledger.

The desired outcome is supplied, not inferred from an unfamiliar sentence. The
reading and graph choices, measurement projection, and operation remain explicit.
Learned relational grounding proposes the target; the empirical browser model
predicts the measured effect. Integrating those components does not make their
authored boundaries disappear.

## One guarded pursuit

A task commitment must retain its desired measured outcome, selected document
action, learned model, and interpretation dependencies. A prediction can support
dispatch only while its source, target correspondence, model evidence, and selected
reading/graph remain current. A stale or unsupported prediction cannot be replaced
by a default label or a direct action fallback.

The actual action still uses the captured backend node through the authenticated
document action path. No CSS selector is synthesized, no interactive ancestor is
guessed, and the desired goal is not copied into an observation. Programmatic DOM
activation remains distinct from physical clicking.

After dispatch, task assessment must use independently retained actual feedback.
An applied receipt establishes that an action was applied, not that the desired
measured outcome occurred. An authenticated matching outcome can support task
completion. A contradiction can suspend the responsible learned rule while
retaining the action receipt and failed outcome. Missing or unauthenticated
feedback must leave uncertainty rather than manufacture completion.

The APIs in `agent.document_tasks` are:

- `create_document_task(agent, provider, model, proposal, desired_outcome,
  *, source="document")` retains a `DocumentGoal` and the selected action context's
  dependencies.
- `pursue_document_task(agent, provider, model, *, proposal=...,
  desired_outcome=..., source="document")` creates and pursues a supplied proposal;
  `pursue_document_task(agent, provider, model, *, task_id=...)` pursues an existing
  task. The modes are exclusive. Desired outcome is required with a proposal and
  must be omitted with a task ID; omission is distinct from an explicit `None`.

`DocumentGoal` retains `desired_outcome`, the action proposal, model ID, and
provider. `DocumentTaskTrace` links proposal, prediction, execution, and observation
source IDs. The goal is a desired **post-activation measurement**, not a general
state goal with an implicit skip-if-already-satisfied rule. A prediction differing
from the desired outcome abstains before activation.

Actual matching feedback yields `done`; an observed different outcome yields
`unverified`. Missing or invalid evidence cannot yield `done`. The runner checks
task revision, exact interpretation dependencies, and current prediction authority
before dispatch and completion. Focused actual-browser verification is recorded below; the full suite remains
pending.

## Evidence and revision boundaries

The task record should make the prediction, actual attempt, observation sources,
feedback assessment, and goal commitment reviewable together. Dependencies must
survive the move from a selected grounding to an action and then to a task; a
revised scene or model cannot silently reauthorize an old commitment.

There is no implicit repeated activation to compensate for a failed observation.
A consumed action proposal and its actual receipt remain attributable. A task
revision with any recorded attempt cannot be pursued again implicitly; concurrent
pursuits are guarded. A different desired outcome requires an explicit task
revision, and a fresh executable proposal must still satisfy its own guards.
Revision-specific recording preserves the attempted old commitment rather than
letting its late result overwrite a newly revised goal. This is one bounded
activation attempt, not automatic retries or multi-step task planning.

## Acceptance evidence and remaining limits

Six actual-browser cases in `tests/test_browser_document_tasks.py` passed in
23.86 seconds. They compose relational DOM grounding learned from three labeled
documents with an outcome model fitted from nine actual browser episodes
(six training, three held out). A fresh explicitly selected target and supplied
desired post-activation measurement complete one pursuit using actual feedback.

The other cases verify a `preventDefault` counterexample yields an unverified
task and rule suspension; an unseen radio feature and a predicted/desired outcome
mismatch leave the browser untouched; withdrawing the selected reading blocks
action; and a task revision during the real post-action observation retains the
applied receipt on the old revision while the new revision remains ready.
Completed and declined revisions reject replay. After final changes, sixteen
ledger tests passed in 3.22 seconds, with separate action (28 tests) and evidence
(30 tests) verification runs. These focused runs overlap and are not additive.
The complete repository suite passed **2,496 tests with two skipped in 409.72
seconds** on the final runtime and test files. These are mechanism tests, not a
browser-intent benchmark.

The browser fixture's content, structured descriptions, labels, desired measured
outcomes, operation choices, and interpretation selections are authored. The
learned components are the graph-query associations and empirical outcome rules.
The literal CDP decoder is supplied. A completed task about `inputChecked` does
not establish general web intent understanding, a learned goal ontology, or
perceptual understanding of pixels.

Remaining gaps include learning goals from unstructured requests, choosing among
competing interpretations, richer or probabilistic effect models, multi-step
planning over changing documents, temporal identity, and discovering missing
observations. This milestone integrates a bounded pursuit with accountable
evidence; it does not complete generalized cognition or an autonomous browser
agent.

The next goal-formation boundary is representational: the current goal
correspondence learner accepts `GoalSpec` state conditions and invariants, while
`DocumentGoal` embeds live execution identities. Teaching those runtime identities
would couple meaning to a particular browser session. A declarative measured
action goal should instead retain a target reference, operation, measurement,
and desired value, with runtime materialization occurring after explicit goal
selection. The acceptance case must learn that correspondence from retained
teaching examples, preserve competing goals, and substitute a newly grounded
target; the final pursuit caller should no longer supply the desired value or an
executable proposal. This remains unfinished, and adding a goal data type alone
would not establish the learned correspondence.

[Learned measured-action goals](74-learned-measured-action-goals.md) develops
a learned intent before this pursuit boundary, so the caller need not directly
provide an action proposal and desired measurement at pursuit time. Runtime
authentication remains separate from the taught intent.

# 62 — Learning goal correspondences

*2026-09-19. This milestone develops a bounded learned alternative to the authored
meaning-to-goal projection discussed in [61](61-deferred-goal-adoption.md). The implemented
learning scope is reference substitution, not
general language understanding or autonomous goal discovery.*

## What the learner is allowed to infer

An explicit task can teach a correspondence between its selected input frame and its
supplied `GoalSpec`. Across retained examples, the learner may discover which explicitly
grounded `Ref` values in the frame occupy which positions in the goal. It then proposes
the same goal structure for a frame with new references, preserving those correspondences.

The upstream frame, reference grounding, and teaching goal remain authored or independently
supplied. The learner does not establish that a teacher's goal is correct. It learns no
synonyms, literal-value transformations, arbitrary feature deletion, or new predicates.
Every non-reference part of the supported structure must match exactly. Generalization
is confined to varying explicit references within that structure. Exact structure includes
entity display text and features, goal condition order, negation, and invariants. New
literal spellings or entity descriptions are not silently normalized into a known pattern.

This is a narrower claim than learning a general mapping from unstructured requests to
tasks. It nevertheless changes the active goal boundary: a supported correspondence can
produce a goal proposal from retained examples rather than requiring a hand-written domain
refiner to reconstruct that same correspondence each time.

## Bounded induction API

The pure learner accepts `GoalExample(id, frame, goal, basis=())` records through
`fit_correspondences(training, validation, max_pairs=256)`. Its returned model proposes
candidates with `model.propose(frame)`. Candidate metadata cites template, training-example,
validation-example, and conflicting-validation-example identities.

A supported template needs at least two training examples and at least one varying input
reference. Equal corresponding reference positions become explicit `constant_refs`; differing
positions are generalized. For example, different devices in the same room can teach a
device correspondence while constraining predictions to that room. Fixed context is retained,
not generalized away. Every output reference must occur in the corresponding input frame;
the learner does not invent an external identity or treat an all-constant pair as transfer.

Training and validation have disjoint example IDs. Variable reference identities are
also disjoint. A shared reference is permitted only when constant across every example
and retained as an explicit constant in the matching template; a variable slot cannot
reuse any training reference. Proposal support
requires at least one exact held-out match of the full goal conditions/invariants. Conflicting
validation examples remain explicit rather than being voted away; several supported mappings
can survive as alternatives. Matching templates without a validated outcome remain explicit
unresolved rivals with their training/conflicting-validation evidence retained in the model.
`complete` means search completeness, not universal validation. Pair-budget exhaustion or
unsupported structures keep the search incomplete. Label/basis annotations are provenance rather than semantic matching
criteria, and the original examples remain detached inspection snapshots.

## Supervision, validation, and admission

Teaching records must resolve to retained task supervision and its selected interpretation
basis. Extraction checks current commitments rather than treating any old goal-shaped
payload as an authorized label. Training and held-out validation examples use disjoint
task/example identities; validation measures the induced reference-substitution rule on the supplied
labels, not independent truth about user intent.

Fitting does not automatically activate the model. A separate explicit admission creates
a selected model commitment that downstream goals and tasks can cite. Examples, learned
correspondences, validation evidence, and the model identity must remain inspectable.
Missing, conflicting, or unsupported examples must not be replaced by a lexical guess.

Once fitted, teaching snapshots are historical immutable evidence of what that model was
taught. A later change to a teaching task does not silently rewrite its existing model.
Updating the model requires explicit refitting; refitting withdraws the old selected-model
commitment so dependent work cannot continue as if the mapping were unchanged. Historical
training evidence and current execution authorization are different relationships.

### Retained-task API

```python
from tensorcode.agent.goal_learning import fit_goal_model, admit_goal_model
from tensorcode.outcomes import Unknown

fitted = fit_goal_model(agent, training_task_ids, heldout_task_ids)
if isinstance(fitted, Unknown):
    raise ValueError(fitted.reason)
admitted = admit_goal_model(agent, fitted, reason="explicit admission of this tested mapping")
if isinstance(admitted, Unknown):
    raise ValueError(admitted.reason)
agent.goal_model = admitted
agent.goal_selector = supplied_goal_selector
```

`extract_goal_example(agent, task_id)` retains the teaching snapshot with task/revision,
source goal group, selected candidate, and dependency evidence. `fit_goal_model` accepts
nonempty disjoint task-ID partitions and returns a local `GoalModelHandle`; fitting and
admission failures remain `Unknown`. Refitting uses the existing `group_id` to append a
new version and withdraw the prior admission. Assignment to `agent.goal_model` chooses
this admitted version as the explicit projection path; the separate goal policy is still
required. An arbitrary unretained model-shaped object does not supply admission authority.

## One explicit goal workspace

On the explicitly configured learned-model path, learned `GoalSpec` proposals enter the
same retained goal comparison used for other goal interpretations. Unsupported learned
input does not fall back to VerbNet or an authored refiner. There is no hidden lexical
or domain-policy route that makes an unavailable learned correspondence appear successful.

Goal selection remains explicit, including when only one supported proposal is available.
Admitting a model is not selecting every goal it might propose. Execution retains the
selected model, input-reading, and goal-interpretation dependencies; changing any current
commitment must invalidate the dependent task before further dispatch or completion.

The existing planning, receipt, observation, and task-revision checks still apply after
goal selection. A predicted goal is neither an observed fact nor a certificate that its
execution will succeed. No new automatic interpretation or action authority comes from
the learner's output shape.

## Verified bounded transfer and remaining scope

The focused learner/agent run passed **19 tests**: twelve pure correspondence tests and
seven agent integration tests. Twelve evidence-retention tests also pass, covering
forged or changed supervision, disjoint task splits, historical snapshots, failed
publication, admission epochs, and changes to earlier examples during later validation.
The complete repository run passed **2,219 tests with five skipped** in 315.08 seconds.

The agent fixture supplies grounded `prepare(object=Ref)` frames, initial teaching goals,
and an authored teacher refiner for devices `a`, `b`, and a separate held-out device. Two
retained tasks train the mapping; the third validates it. After fitting and explicit model
admission, both the lexical goal function and the teacher refiner are replaced with failing
test guards. An explicitly selected learned proposal for a new device still reaches the
actual in-memory device capability and completes with the new reference. Its task retains
three commitments: model admission, selected input reading, and selected goal interpretation.
This demonstrates active use of the induced correspondence, not just model-shaped metadata.

Separate regressions establish that model refitting invalidates a dependent task, and that
withdrawing the model selection during goal selection or a precondition callback blocks
the actual action. No supplied goal-selection policy means no automatic execution. Changed
non-reference structure is outside the learned template rather than an invitation to use
VerbNet or a domain refiner as an implicit fallback.

The initial frame, reference grounding, teacher labels/refinement, device semantics, and
explicit admission/selection policies are authored. The new reference association is induced
from those retained examples. This is not a natural-language inference benchmark, a visual
identity benchmark, or evidence that the teacher's intended meaning was independently
established. The stable-context tests likewise measure explicit reference structure, not
learned notions of rooms, devices, or context relevance.

Historical fitted teaching snapshots do not require their former task selections to remain
live forever. Current downstream model admission does: withdrawal or refit changes the
execution commitment without rewriting what was originally taught. Broader correction,
literal transformation, synonym induction, richer goal structures, and autonomous model
admission remain open. Source evidence, exact supported scope, and failure behavior take
precedence over a broad claim of learned cognition.

These models and their admission handles live inside the current agent. Restarting
the chat server does not restore them from its stored transcript. Reference
substitution also does not independently infer entity types or semantic compatibility;
predicted goals remain hypotheses under the supplied teaching context.

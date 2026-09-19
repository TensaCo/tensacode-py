# 34 — Cognitive primitives and implementation

> **Objective update, 2026-09-19:** [36 — The structured cognitive workspace](36-structured-cognitive-workspace.md)
> develops the governing objective beyond task and action foundations: unstructured input
> must become evidence-backed, revisable interpretations within the reasoning process.
> Stable mechanics should support extensible concepts, competing hypotheses, learning, and
> communicative/action realization. The proposals and first slice below are components of
> that objective, not a complete cognitive core.

*2026-09-19. Architectural assessment and implementation proposal. The historical measurements
in [33](33-the-cognitive-fronts.md) were not rerun for this assessment. Proposed structures and
acceptance criteria below are not claims of implemented agent behavior. The final section
separately records the first implemented slice and its limits.*

## What the failure inventory leaves unresolved

Document 33 names useful failures and preserves important evidence: explicit abstentions,
world-state grading, do-nothing controls, and plugins whose isolated machinery produced no
end-to-end improvement. The weakness is the repeated inference from “we can store it” to
“the remaining work is plumbing.” Several proposed remedies are the smallest demonstration
of a faculty, without the conceptual distinctions needed to extend it.

A goal is more than a verb's result state. A hypothesis is more than a proposition marked
`hypothesised`. Conversation is more than persistent references. Analogy is more than argument
substitution. Learning a word, revising a belief, repairing a reference, and acquiring a
procedure require different updates. The [ten-front reassessment](33-the-cognitive-fronts.md#architectural-reassessment-2026-09-19)
records these distinctions individually; this document develops their shared consequences.

## Task, interpretation, plan, and attempt

A durable task should identify what the user is trying to accomplish, the current interpretation
of that request, success criteria, restrictions, unresolved choices, and status. Plans are
proposals for achieving that task. Attempts record what actually happened. These should remain
separately inspectable and correctable.

Consider this episode:

> Make a hello-world project in scratch.
> Actually, use Documents, but keep the existing README.
> Why did you choose that file?
> Do the same for the other project.

The task identity must survive the destination correction. A revised interpretation can
invalidate part of a plan without erasing execution history. Preserving a README is a
restriction, not simply another desired final condition. An explanation relates a decision
to the request, assumptions, and observations; a receipt only establishes that an operation
ran. Reuse needs a distinction between a method and the concrete objects of its last attempt.
If work has already begun, a correction must account for the resulting state rather than
pretend it never happened or automatically undo completed work.

The proposed minimal task record needs:

- Stable identity, originating request, and revision history.
- A current interpreted specification and observable success criteria.
- Restrictions with scope and lifetime, plus unresolved alternatives or questions.
- Status and links to plans, observations, and execution attempts.
- Dependencies explaining which interpretations or observations justify which decisions.

This is a behavioral contract, not a mandate to add every field before shipping a small slice.
Persistence within an agent session and persistence across process restarts are separate
capabilities and must be reported separately.

## Decomposition needs domain knowledge and action models

“A Python hello-world project is a folder, a file, and a line of text” is an interpretation
supplied by the author. Generic regression over creation effects cannot derive that meaning.
A refinement method must supply the connection between an abstract task and concrete success
criteria, with its source and applicability conditions visible. Another project convention
might reasonably require different artifacts.

The inspected starting architecture ties `Goal` to a verb, verb class, result conditions, and
source frame in [`language/verbnet.py`](../../src/tensorcode/language/verbnet.py). A
[`Capability`](../../src/tensorcode/agent/plugin.py) declares parameters and effects but, at the
start of this work, has no declared preconditions. `choose_plan` in
[`agent/core.py`](../../src/tensorcode/agent/core.py) selects one capability. Sequences alone do
not repair the missing specification or action model.

Keep four questions distinct:

1. What would count as satisfying the request?
2. How can this kind of task be refined in this domain?
3. Which available actions can establish those conditions in the observed world?
4. Did the execution actually satisfy the request and its restrictions?

Action models should expose preconditions, expected effects, and observation or verification
requirements. A planner must distinguish unknown conditions from false ones, and must not
assert predicted effects as observed facts. Restriction checks need to cover intermediate
states and indirect effects; a correct final state alone cannot establish compliance.

## Lexical resources propose interpretations

VerbNet is useful evidence about language. Its lexical coverage should not determine the
space of intentions the agent can represent. At the inspected starting point, `goal_of` in
[`language/verbnet.py`](../../src/tensorcode/language/verbnet.py) cannot form a goal when the
verb class or result state is absent. Consequently, a missing lexical entry becomes a missing
cognitive ability, including for discourse requests.

Use a cognitive task specification independent of the resource that proposed it. VerbNet,
domain methods, context, demonstrations, and later learned interpreters should be able to
propose compatible specifications, with provenance and explicit uncertainty. Decoupling the
representation does not license guessing a meaning when the evidence is insufficient.

The knowledge-as-code audit in [32](32-knowledge-written-as-code.md) remains useful, but moving
a table into seed data or reading a curated resource does not establish conceptual adequacy.
It can improve maintainability and lexical coverage while preserving the same flawed model.
Evaluate the represented distinctions and behavior as well as the location of the knowledge.

## Semantic preservation is an interface requirement

N-ary [`Proposition`](../../src/tensorcode/records.py) records improve on flat triples: named
roles, nesting, polarity, modality, and scope are valuable. Arbitrary nested values do not,
however, supply shared semantics for binding, quantification, time, or nested attitudes.
Examples that expose the gap include:

- “Every file except the newest one”: a quantified set, an exception, and an ordering.
- “Keep this true until the task finishes”: a restriction over an interval.
- “Alice believes Bob might know”: distinct agents and nested attitudes.
- “Exactly one statement is false”: statements as objects and a cardinality constraint.
- “There were three plants; two more arrived”: collection identity and a change over time.

A predicate spelling is not enough; producers and consumers must agree on meaning. Add only
the semantics needed by the next episode, but make unsupported distinctions visible.

The inspected `to_propositions` in
[`language/semantics.py`](../../src/tensorcode/language/semantics.py) turns an `Entity` into a
reference without generally preserving or reporting its features. The quantity failure is
therefore evidence of a boundary problem: information can disappear while the output looks
complete. Projection should preserve supported meaning and provide a structured report for
unsupported or discarded distinctions. Downstream consumers must not treat such an output
as an unqualified equivalent of the original request.

Counting and arithmetic additionally need collection identity, overlap, ownership, rates,
measurements, and temporal changes. Constraint solving needs variables, domains, quantifiers,
logical scope, and completeness assumptions. Supplying numbers to an arithmetic routine or
assignments to a solver does not establish that the language was formalized correctly.

## Alternatives, clarification, and revision

An uncertainty score does not identify the uncertainty or establish that asking the user is
useful. Keep the alternatives explicit: two candidate files, two task interpretations, an
unknown precondition, or incompatible evidence. Record which decision depends on the choice.
Ask when an answer is available from the user and relevant to that decision; inspect the world
when it can resolve the issue more directly. Clarification should have an answer that can
actually update the pending task, not merely produce a question-shaped reply.

Corrections should revise the relevant interpretation and invalidate dependent decisions.
Reference repair is usually local to an exchange. A vocabulary definition may be scoped to a
conversation or domain. A belief update needs evidence and supersession. A learned procedure
needs applicability conditions and transfer evidence. They should not all silently become
permanent global knowledge.

Likewise, a reusable procedure must distinguish arguments from incidental details of its
original execution. Full analogy adds relational correspondence, adaptation, and recognition
of mismatches. Report parameterized reuse as parameterized reuse until those operations are
implemented and measured.

## Hypotheses must predict, and determination is relative

The hidden-machine probe is a useful bounded research task. Finite observations do not
uniquely determine an arbitrary hidden function. A conclusion can be forced relative to a
declared candidate family and assumptions, or a prediction can be shared by all remaining
candidates even while the full model is unresolved. These are different claims.

A first experiment loop should explicitly provide:

- A finite or otherwise tractable model family and stated assumptions.
- Predictions from each model and a rule for compatibility with observations.
- Surviving alternatives and the inputs on which they disagree.
- A legal probe budget, a selection objective, and a stopping rule.
- An answer separating excluded models, unresolved alternatives, common predictions, and
  any simplicity preference used to select a representative.

`ops.choose` can select among supplied probes once predictions and an objective exist. It
does not itself generate models, derive predictions, or decide what evidence means. Evaluate
held-out predictions and claims of determination separately. Include cases where the true
function is outside the declared family and where the remaining budget cannot discriminate
survivors. This remains a bounded research bet, not a prerequisite to the first task episode.

## Implementation sequence and acceptance evidence

The ten fronts remain an evaluation index. Implementation should develop a few shared
structures through working episodes, rather than create ten isolated modules.

| Stage | Deliverable | Evidence required before claiming the stage works |
|---|---|---|
| 0. Reproducible starting point | Record revision, worktree state, and a relevant baseline; preserve unrelated work | Reproduction commands and results, with dependency or environment blockers distinguished from behavioral failures |
| 1. Persistent task identity | Agent-owned task records, revision history, and links to interpretation and attempts | A request creates a task; a supported correction revises it; previous attempts remain inspectable; status changes follow real outcomes |
| 2. Meaning preserved at boundaries | Supported semantic features survive projection; losses are explicit | Pairs that differ only in quantity, negation, or scope remain distinguishable, or the unsupported distinction is reported |
| 3. Refinement and action models | A bounded domain method supplies a concrete specification; actions declare preconditions and effects | Multiple actions satisfy an explicit specification; missing preconditions and changed observations prevent unsupported success claims |
| 4. Integrated project episode | Project creation, destination correction, existing-file restriction, verification, and explanation | The agent handles the episode through its ordinary input path; the world grader verifies artifacts and preserved contents, including interrupted or partially completed variants |
| 5. Transfer and further fronts | Related tasks, useful clarification, bounded procedure reuse, then experiment design | Paraphrases, changed entities and initial states, and structurally relevant variations succeed without case-specific reply patches |

Stages 1 and 2 can begin independently; integration must make semantic uncertainty affect
interpretation and execution. A small first implementation is valuable even if it does not
complete the full episode. State exactly which stages and input paths are supported. An API
unit test proves an API behavior; it does not prove an English conversation works.

For the project episode, include an existing README with recognizable content, a different
initial destination, and a correction at a controlled interruption point. Specify which file
is the intended runnable artifact, rather than treating the evaluator's convention as an
unstated truth about all projects. Check that successful completion is established from the
world, the restricted file is preserved, and explanations cite recorded reasons. A later
reuse request must re-evaluate applicability and current state.

## Evaluation that exposes conceptual errors

Retain explicit failures, world verification, and do-nothing comparisons from 33. Extend
single showcase probes with paired and compositional cases:

- Paraphrases that preserve meaning, contrasted with small wording changes that alter it.
- Renamed entities and changed initial conditions, including pre-existing artifacts.
- Corrections and interruptions before and after partial execution.
- Ambiguities that matter to the action and ambiguities that do not.
- Restrictions on intermediate actions, not only final-state properties.
- Transfer to a related case and a superficially similar case where reuse is inappropriate.

Report interpretation accuracy, completion, constraint violations, wrong answers, abstentions,
unnecessary questions, and cost separately. Distinguish formal-model correctness from the
correctness of language-to-model translation. Label fixtures, author-supplied refinement
knowledge, held-out cases, and which benchmarks informed development.

Zero observed wrong answers describes a particular sample. It is not a universal invariant
and must not reward doing nothing. Improvements should demonstrate useful completed work
alongside error and restriction measurements, without hiding errors in an aggregate score.
No new benchmark gains are claimed by this proposal.

## First implementation slice, 2026-09-19

The implementation now begins stages 1–3. It does **not** complete the integrated project
episode or establish an improvement on the historical cognitive benchmarks.

[`goals.py`](../../src/tensorcode/goals.py) defines `Condition` and `GoalSpec` independently of
VerbNet. `verbnet.Condition` remains a compatible alias. A `GoalSpec` is a nonempty conjunction
of bound conditions, using the capability matcher's role conventions; it is not yet a
quantified or temporal goal language. `Agent.pursue` accepts it directly without lexical
interpretation, still selecting one capability. Every explicit condition must match, and
incompatible bindings cannot be silently merged into a successful plan.

[`Agent.tasks`](../../src/tensorcode/agent/tasks.py) keeps detached snapshots with stable IDs,
revision reasons, and execution attempts. Ordinary request outcomes also carry `task_id`,
and those records remain available across turns within the same agent instance. Revising a
task retains its earlier goal and attempt history. Completed tasks and attempts that may
already have changed the world require explicit revision before another attempt; a rejected
precondition can be checked again on the same task. This guards against accidental replay;
revision itself does not undo previous effects.

For example, suppose the caller has supplied a fictional `devices` plugin whose `enable`
capability has effect `enabled(undergoer=device)` and knows how to resolve the following
references. The caller, rather than a natural-language correction interpreter, revises the task:

```python
from tensorcode.agent import Agent, Condition, GoalSpec
from tensorcode.records import Ref

agent = Agent([devices])  # supplied plugin; not a built-in device integration
first_goal = GoalSpec((Condition("enabled", {"undergoer": Ref("device:a")}),))
first = agent.pursue(first_goal)

replacement = GoalSpec((Condition("enabled", {"undergoer": Ref("device:b")}),))
agent.tasks.revise(first.task_id, replacement, reason="use the second device instead")
second = agent.pursue(task_id=first.task_id)
record = agent.tasks.get(first.task_id)
assert second.task_id == first.task_id
assert record.revision == 2
```

This revision does not disable device A if it was already enabled. The example demonstrates
identity and history, not rollback, multi-step planning, or interpretation of “instead.”

[`Capability.preconditions`](../../src/tensorcode/agent/plugin.py) and
`Plugin.precondition_holds` make applicability explicit. A centralized check before dispatch
rejects false, unknown, or unbound requirements. An information capability cannot reveal its
answer after a rejected receipt. These checks guard the selected action; they do not yet
select an alternative plan when a precondition fails. Existing plugins opt in by declaring
requirements, so this adds no claim of general held-constraint coverage.

Semantic projection now reports paths to entity features it discards, including nested
features, instead of silently treating their disappearance as a lossless conversion. This
is loss reporting, **not preservation or interpretation** of quantity, scope, or other feature
semantics. Consumers still need to make those reports govern their decisions.

Focused regression coverage is in
[`test_task_records.py`](../../tests/test_task_records.py),
[`test_agent_tasks.py`](../../tests/test_agent_tasks.py),
[`test_capability_preconditions.py`](../../tests/test_capability_preconditions.py), and
[`test_projection_preservation.py`](../../tests/test_projection_preservation.py).
The task-path test supplies controlled interpretation fixtures; it proves ordinary request
integration and retention across turns, not natural-language correction coverage. Additional
cases cover incompatible bindings, omitted conjunction conditions, replay prevention,
precondition rejection, and execution receipts that fail to establish completion.

Remaining limits are explicit: no natural-language task correction, sequence or regression
planner, project refinement knowledge, or disk persistence. Verification still uses a fresh
capability-based observation, rather than an independent task-level verifier. These are
foundations with a bounded execution path, not completion of fronts 2, 3, or 8.

### Validation and repository baseline

The initial worktree was clean at `a73bbf3` on `computerworld-only`. At the owner's
request, local `main` was fast-forwarded from its ancestor `d9839e6`, preserving all
38 exploration commits. Uncommitted work was temporarily stashed and restored;
nothing was pushed. `AGENTS.md` records the main-only workflow for this phase.

- Baseline: `.venv/bin/python -m pytest -q` — **1,245 passed, 5 skipped** (86.28 s).
- Implementation suite: the same command — **1,275 passed, 5 skipped** (94.85 s).
- Two further regressions cover rejected information actions and unbound preconditions.
  The final `.venv/bin/python -m pytest -q tests/test_agent_tasks.py` run, including those
  additions and the existing task cases, passed **14 tests** (3.02 s).
- `git diff --check` passed. Historical cognitive assay scores were not rerun or changed.

## Follow-up implementation, 2026-09-19

The first slice above is a dated baseline. [35 — Model-based planning and inspectable
refinement](35-model-based-planning.md) supersedes its single-action and capability-only
verification limits for plugins that supply grounded action models. The implementation now
supports bounded multi-step search, independent task-condition observations, explicit held
conditions, step histories, and suspension/replanning. A real filesystem adapter and a
hand-authored, replaceable project recipe support a named Python project through the ordinary
language path. Structured callers can revise a destination after partial execution while
preserving a README and task identity.

Ordered modifiers now survive grammar/dependency interpretation, with tested grammar
realization/reparse preservation and unambiguous compatibility aliases. General semantic
projection still reports unsupported feature losses; preservation alone does not interpret
all modifier meanings. The example server exposes the filesystem adapter only with an
explicit existing root.

This advances stages 1–3 and the structured part of stage 4. It does not complete the ordinary
English correction episode, repair the original “make a python hello world project” parse,
or establish benchmark gains. The new document separates these input paths and includes an
executed API example, provenance, model assumptions, and remaining semantic debt.

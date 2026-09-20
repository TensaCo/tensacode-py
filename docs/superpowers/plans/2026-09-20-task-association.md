# Evidence-driven Task Association Implementation Plan

> **For agentic workers:** Use `superpowers:subagent-driven-development` for the
> independent tasks below; use `superpowers:executing-plans` for root integration.

**Goal:** Infer supported conversational task associations from retained complete
readings and task context, then use them before sentence dispatch.

**Architecture:** A shared exact structural correspondence mechanism supports
dedicated relation learners. Retained supervision and model admission connect
their proposals to authenticated conversation evidence. Sentence routing compares
revision targets, positively supported new-task interpretations, and unresolved
rivals before any task mutation.

**Tech Stack:** Existing Python package, interpretation workspace, task ledger,
cached learned reader, and pytest. No new runtime dependency or downloaded model.

**Spec:** [Evidence-driven task association](../../revival/82-evidence-driven-task-association.md).

## Global constraints

- Work on `main` only; do not create branches or worktrees.
- Keep the current correction implementation frozen until its running full suite
  finishes and its verified milestone is committed.
- Preserve complete ordered utterance frames, source evidence, uncertainty,
  reference equality, and task execution history.
- Do not introduce keyword routing, recency defaults, implicit first-candidate
  selection, or bundled semantic teaching.
- New-task intent requires positive evidence. Failed association is not evidence.
- Explicit teaching and grounding remain supplied; report them as such.
- Use `.venv/bin/pytest`; do not change the lockfile to run tests.
- Root owns commits and pushes. Workers own only their assigned files and must
  accommodate shared changes rather than revert them.

## Review focus

1. The intended task is older than a structurally similar distractor.
2. One supported target coexists with an unsupported or conflicting rival.
3. A rival is added or revised during comparison or adoption.
4. Multiple supervision pairs share one episode and must not inflate evidence.
5. A sentence contains multiple acts; none may create a partial task before routing.

## Task 1: Shared structure and learned pair relations

**Owner:** Pure-learning worker, after the correction checkpoint.

**Files:** Create `src/tensorcode/learning/structural_correspondence.py` and
`src/tensorcode/learning/task_association.py`; modify
`src/tensorcode/learning/goal_correspondence.py`,
`src/tensorcode/learning/speech_act.py`, and
`src/tensorcode/learning/scene_grounding.py` to use the shared encoding. Create
`tests/test_task_association_learning.py` and extend
`tests/test_goal_correspondence.py` with adapter regressions.

**Interfaces:**

```python
@dataclass(frozen=True)
class TaskAssociationExample:
    id: str
    episode_id: str
    incoming: tuple[Frame, ...]
    originating: tuple[Frame, ...]
    previous: GoalSpec
    relation: str  # supplied "revise" or "exclude"
    basis: tuple[str, ...] = ()

fit_task_associations(training, validation, *, max_pairs=256)
model.propose(incoming, originating, previous)
```

Proposals expose the learned relation plus template, training, validation, and
conflict evidence. Results expose all proposals, unresolved evidence, and search
completeness. Relations are not represented as executable goals. Context excludes
episode IDs, task IDs, revision numbers, labels, and timestamps; it includes all
conditions and invariants. Public model snapshots are detached.

- [ ] Write failing tests in which two independent training episodes and a third
  held-out episode teach matching protected-resource identity as `revise` and a
  different identity as `exclude`. Fresh references must transfer; missing,
  negated, or reordered incoming clauses must not match silently.
- [ ] Add conflicting teaching and shared-episode tests. Training pairs from one
  episode cannot serve as independent observations, and an episode cannot occur
  in both splits. Search exhaustion and unvalidated rivals remain visible.
- [ ] Run `.venv/bin/pytest -q tests/test_task_association_learning.py` and witness
  the missing behavior before implementation.
- [ ] Extract encoding, template fitting, and matching as structural operations.
  Preserve scalar types, output-reference availability checks, complete reference
  equality patterns, learned constants, split separation, and conflict tracking.
  Keep goal reconstruction in the goal adapter. Update internal consumers instead
  of adding compatibility fallbacks.
- [ ] Implement the relation adapter and run its tests together with goal,
  revision, speech, and scene-grounding learner tests. Root reviews any changes to
  exact model authentication required by the extraction before accepting them.

## Task 2: Complete task-context snapshots

**Owner:** Retained-context worker; `tasks.py` is exclusively owned by this worker.

**Files:** Modify `src/tensorcode/agent/tasks.py`; create
`src/tensorcode/agent/task_association_context.py` and
`tests/test_task_association_context.py`.

**Interfaces:**

```python
TaskLedger.comparison_basis()  # callback-free identities/revisions snapshot
capture_task_association_context(agent, group_id, candidate_id, *, basis)
validate_task_association_context(agent, context, *, historical=False)
```

Capture all ledger tasks in the agent's conversation, not a caller-supplied chosen
subset. Retain complete incoming reading, all task snapshots, and originating
interpretation dependencies. Tasks without usable origin evidence become explicit
unresolved members. Context validation returns `True` or `Unknown`.

- [ ] Write failures for an earlier target plus later distractor, an ungrounded
  rival, a task inserted after capture, and a rival revised during payload copy.
- [ ] Run `.venv/bin/pytest -q tests/test_task_association_context.py`.
- [ ] Add a lock-protected callback-free ledger membership/revision comparison;
  do not copy arbitrary payloads to obtain the final guard value.
- [ ] Capture readings through retained interpretation and goal provenance.
  Use historical originating readings but current goal constraints. Do not infer
  the origin by reparsing `Task.source` or inspecting mutable `agent.turns`.
- [ ] Revalidate all dependencies and membership after callback-bearing reads.
  Test withdrawal, missing origins, and comparisons with identical task contents
  but different identities. Candidate order must not affect semantic pair inputs.
- [ ] Separate live validation from historical authentication of an already
  retained comparison. Test that adding another teaching episode invalidates an
  old live routing comparison but does not erase its authenticated supervision.
  Historical mode must validate retained snapshots and ledger history, not accept
  a caller-created context merely because its task IDs still exist.

## Task 3: Retained supervision, model admission, and alternatives

**Owner:** Retained-model worker after Tasks 1 and 2 interfaces are verified.

**Files:** Create `src/tensorcode/agent/task_association_learning.py`,
`src/tensorcode/agent/task_association.py`,
`tests/test_task_association_models.py`, and
`tests/test_task_association_adoption.py`.
The same worker owns the association integration in
`src/tensorcode/agent/task_revision.py` and
`src/tensorcode/agent/task_revision_learning.py`; coordinate with root before
editing their already verified correction behavior.

**Interfaces:**

```python
retain_task_association_example(agent, context, task_id, relation, *, basis)
fit_task_association_model(agent, training, validation, *, group_id=None,
                           max_pairs=256)
admit_task_association_model(agent, handle, *, reason)
propose_task_association(agent, group_id, candidate_id, *, model, basis)
select_task_association(agent, association_group_id, decision, *, reason)
propose_task_revision(..., association=selected_association)
```

The retained example carries its authenticated context and episode source, not
just a freely assigned episode label. A selected association carries the task ID,
revision, and complete comparison dependency for use by correction proposal and
adoption. These operations do not execute, create tasks, or revise goals.

- [ ] Write and run failing tests for forged teaching, cross-split episodes,
  mutated models, withdrawal, stale membership, and partial publication.
- [ ] Implement historical retained supervision and separately admitted model
  versions. Authenticate nested pure-model state and retained fit sources.
- [ ] Retain each complete episode comparison before its pair labels are fitted.
  Verify a model can fit earlier teaching after later episodes add tasks. Use
  historical context authentication for these examples; require current membership
  for live proposals and commit guards. Keep split identity tied to the retained
  incoming episode source, never a freely relabeled pair ID.
- [ ] Evaluate every usable pair and retain every unresolved member. Represent
  supported exclusions distinctly from unsupported pairs. A singleton positive
  with an unresolved rival cannot be treated as a complete automatic decision.
- [ ] Require an exact explicit comparison when selecting; recheck the complete
  context in correction proposal and adoption, including the ledger's final
  `before_commit` callback. Retain the selected association in proposal provenance
  and resulting task dependencies. Test rival insertion and association withdrawal
  during final payload copying, not only before calling adoption.
- [ ] After successful adoption, authenticate the exact historical comparison and
  commit identity. Keep its reading, grounding, model, and interpretation support
  live. Do not require old rival revisions or membership to remain current after
  commit; unrelated later task creation is not evidence against the old decision.
  A later explicit interpretation challenge still invalidates its dependency.
- [ ] Run `.venv/bin/pytest -q tests/test_task_association_models.py
  tests/test_task_association_adoption.py` plus the current correction tests.

## Task 4: Real-input association and sentence routing

**Owner:** Root integration; only root edits `agent/core.py`.

**Files:** Create `tests/test_learned_task_association_inputs.py` and
`tests/test_sentence_task_routing.py`; modify `src/tensorcode/agent/core.py` and
extend the relation-learning and retained-model modules for positive new-task
evidence. Root also owns the whole-sentence goal path in
`src/tensorcode/agent/goal_interpretation.py` and
`src/tensorcode/agent/goal_learning.py`, plus guarded task creation in
`src/tensorcode/agent/tasks.py` after the context worker finishes that file.
Update the revival design and implementation reports.

- [ ] Begin with a failing actual-reader episode: two tasks, older target, newer
  distractor, complete incoming correction, held-out association, contextual
  revision, then explicit pursuit. Assert the target's README and receipts are
  preserved and the distractor's goal and files remain unchanged.
- [ ] Repeat with reversed candidate order and extra distractors; assert ambiguity
  for symmetric tasks and abstention for unsupported rivals. Check withdrawal
  before dispatch. Use supplied grounding and labels explicitly in the report.
- [ ] Add positive new-request supervision as a dedicated relation over the
  complete incoming reading. Retain its model evidence alongside task-target
  alternatives. Do not derive `new` from negative pairwise results or task count.
- [ ] Write routing tests for supported new requests, supported corrections,
  competing new/revise interpretations, empty context without teaching, and a
  multi-act sentence with unresolved routing. Assert unresolved cases create no
  task and perform no effects.
- [ ] Add one retained complete-frame goal proposal for a supported new multi-act
  sentence. Teach the whole sentence-to-goal relation; do not discard clauses or
  independently project them and silently invent composition semantics. Assert
  exactly one task with all taught constraints. Unsupported complete projection
  must abstain even when new-task intent itself has positive support.
- [ ] Extend `TaskLedger.create` with guarded creation after all input copies and
  before publication. Test task insertion during payload copy and model withdrawal
  in the final guard. Create the task before pursuit; no effect may occur before
  the routing and complete-goal comparison commits successfully.
- [ ] Retain a historical precreation comparison plus the new task identity, so
  creation does not invalidate its own membership dependency. Verify later
  unrelated tasks do not invalidate it, while reading/model/interpretation
  withdrawal still prevents further dispatch. Reuse the same historical/live
  distinction implemented for correction adoption.
- [ ] Route once per selected sentence before the per-act loop. Consume the
  complete comparison when dispatching the selected interpretation. Keep explicit
  structured task creation available to callers; remove unconditional task
  creation as the language path's implicit discourse decision. A failed association
  must never drop its handle and enter the supplied-task correction API instead.
- [ ] Migrate tests that intentionally supply authored routing through explicit
  test fixtures; do not add production semantic seeds to preserve old behavior.
- [ ] Run focused routing and real-input tests, obtain boundary review, then run
  the full suite once the integrated source is frozen. Commit and push verified
  milestones on `main`, reporting remaining grounding and discourse limitations.

## Milestone accounting

Tasks 1–3 establish a retained learned association mechanism. They are not a
claim that normal chat uses it. Task 4 establishes that active consumer and must
be completed before this plan is reported as implemented. Collective references,
unseen discourse structures, and grounding from raw scenes remain explicit
frontier work after this milestone.

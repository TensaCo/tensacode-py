# Paused task-association checkpoint — 2026-09-20

Status: **paused at the owner's explicit request; work in progress, not verified
for release.** The owner requested that all current work be documented and
committed. Do not resume implementation until asked. No claim of generalized
cognition or complete conversational routing is made.

## Last verified baseline

`c04f787` (`Learn retained contextual task corrections and realize symbolic
resources`) is committed and pushed to `origin/main`. Its full suite passed
**2,812 tests, 2 skipped, in 1,126.29 seconds**, exit 0. It implements contextual
correction with explicitly supplied task association and grounding; it does not
infer which existing task a new utterance concerns.

The demo at `http://127.0.0.1:8771/` was refreshed after that baseline. It remains
running; pausing development did not shut down the requested demo. The refresh
preserved both chats and all 30 messages. Server PID at pause: `2925280`; argv:
`.venv/bin/python -m examples.general_agent.server --port 8771 --no-open --reader learned`.
History database: `~/.cache/tensorcode/chat/chats.sqlite3`. WIP code has not received
a deliberate demo refresh; fresh imports by the existing worker can see files
in the shared checkout, so do not treat the live demo as an isolated release.

## Current implementation state

The governing objective is [doc 36](36-structured-cognitive-workspace.md).
The next design is [doc 82](82-evidence-driven-task-association.md); the detailed
[implementation plan](../superpowers/plans/2026-09-20-task-association.md) is not
complete.

1. **Shared structural learner and task-pair relations: implementation present,
   independent review pending.** New `learning/structural_correspondence.py`
   extracts exact structure/reference encoding, template fitting, and matching.
   `learning/task_association.py` learns positively taught `revise` or `exclude`
   relations from complete incoming/originating frames and current goals.
   Goal, speech, and scene learners use the shared encoding. Semantic inputs
   exclude task recency, task IDs, episode IDs, and revision metadata.

   Generic observations/templates use `structure` and `result`, not goal-shaped
   classification labels. `fit_templates(..., require_reference_variation=False)`
   permits independently taught exact/reference-free mappings; its default
   remains reference-substitution induction. This option supplies no new-task
   semantics. Goal model authentication retains the existing six-key state.
   The worker reports **104 focused tests passed**, then **46 authentication and
   evidence tests passed** after the final rename. These are worker-reported
   results; the entire WIP has not passed the full suite.

2. **Authenticated task contexts: implementation present, review fix pending.**
   `TaskLedger.comparison_basis()` captures identities/revisions without payload
   callbacks. `agent/task_association_context.py` retains all candidate tasks,
   complete incoming/originating readings, current goals, and unresolved members.
   Registered historical contexts support teaching; live contexts require the
   current complete candidate comparison. Worker verification: **13 focused
   tests**, **85 adjacent tests** passed before the review finding below.

3. **Retained association teaching, admission, selection, and correction
   integration: not implemented.** Task 3 of the plan must connect the pure
   learner/context APIs, preserve unresolved rivals, and carry selected
   association evidence through the final task-revision commit guard.

4. **Whole-sentence goals: partial implementation, review fixes pending.**
   `goal_interpretation.retain_taught_sentence_goal(...)` retains one supplied
   goal over all ordered request frames. `retain_sentence_goal_proposals(...)`
   requires an admitted learned goal model and retains a complete-input proposal.
   `goal_learning` authenticates that input when extracting supervision.
   Root observed **67 focused tests passed** before adding the failing review
   regressions. These APIs do not create tasks or establish new-task intent.

5. **Chat routing, positive new-task intent, and guarded creation: unfinished.**
   `Agent.turn` still dispatches individual request acts and `Agent.request`
   still creates tasks without an explicit discourse interpretation. The four
   new routing regressions intentionally expose that gap. No production fallback
   or supplied semantic seed was added to make them pass.

## Known defects and failing tests — fix these first on resume

There are **nine known failing regression cases**. They are committed normally,
not skipped or marked expected-failure. Further failures may exist because the
whole WIP has not had a full-suite run.

- `tests/test_sentence_task_routing.py`: **4 failed in 2.30 seconds**. Single-
  and multi-act requests, with and without an existing task, create tasks despite
  absent discourse evidence. Full routing integration is needed, not a blanket
  refusal that removes the positive path.
- `tests/test_sentence_goal_teaching.py`: **5 new review regressions failed,
  11 deselected, in 1.84 seconds**, using:

  ```bash
  .venv/bin/pytest -q tests/test_sentence_goal_teaching.py -k 'final_selection_callback or final_evidence_copy or proposal_comparison'
  ```

  The defects are:
  1. Final goal selection can return a concrete goal after a callback changes
     the parent sentence's content without changing comparison IDs/revision.
     Revalidate sentence snapshots after late selection/extraction callbacks,
     then finish with comparisons of all participating commitments.
  2. Taught retention can return success after its final evidence copy withdraws
     the parent. Recheck parent/support after those reads and before registration.
  3. Runtime publication can return an already-selected group or an unexpected
     extra candidate. Authenticate the expected source, candidate payloads,
     unselected state, and exact initial comparison before successful registration.

- **Additional context-review finding, not yet pinned by a regression:**
  `task_association_context._validate_context` excludes unresolved members from
  its final dependency checks even when they retain authenticated origin evidence.
  A later member's copy callback can withdraw an earlier unresolved member's
  origin goal group; live validation returns `True`, while an immediate second
  validation returns `Unknown`. Include retained origin dependencies of such
  unresolved members in the final checks. The reviewer reproduced this before
  the explicit pause; its report was interrupted before being written.

Detailed goal-boundary reproductions are preserved in the
[goal review](checkpoints/2026-09-20-task-association/task-4-goals-review.md).
The [context report](checkpoints/2026-09-20-task-association/task-2-report.md)
records its APIs and original test evidence. The
[goal implementation report](checkpoints/2026-09-20-task-association/task-4-goals-report.md)
predates the failing review regressions and must not be read as a clean verdict.
The [execution ledger](checkpoints/2026-09-20-task-association/execution-ledger.md)
preserves ownership and decisions beyond the ignored local scratch directory.

## Resume sequence and operating constraints

1. Read this handoff and inspect the current worktree. Remain on `main`; no
   exploration branches or worktrees. Preserve all checkpointed work.
2. Fix the three goal-boundary findings under their existing failing tests. Add
   and fix the unresolved-origin context regression. Obtain scoped re-review.
3. Independently review the structural/association learner, preserving exact
   reference relationships, episode-disjoint validation, conflict evidence, and
   explicit search exhaustion. Do not derive `new` from exclusions.
4. Implement plan Task 3 and then the remaining Task 4 integration. Carry
   association dependencies to final correction commit; teach positive new-task
   intent; project one goal for the complete sentence; create a guarded task
   before effects. Keep historical committed context distinct from live support.
5. Exercise actual-reader multi-task association, whole-sentence new requests,
   withdrawal, ambiguous rivals, and unchanged execution receipts. Then run a
   single full suite against frozen source before claiming a verified milestone.

Use `.venv/bin/pytest`, not `uv run`. Root owns commits and pushes. Completed,
verified milestones may be pushed under existing authorization; this paused WIP
is a local checkpoint and has not been represented as a verified milestone.

All implementation and review agents were stopped. The last regression process
finished before checkpointing; no test run remains intentionally active. On
resume, verify process state rather than relying on old PID/session records.

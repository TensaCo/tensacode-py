# SDD ledger — plan: docs/superpowers/plans/2026-09-20-task-association.md

Current state: implementation started after verified correction checkpoint.
Base c04f787 is committed and pushed to origin/main. Full suite: 2812 passed,
2 skipped in 1126.29 seconds, exit 0, /tmp/tensorcode-contextual-revision-suite.
Demo refreshed to PID 2925280; exact 2-chat / 30-message history preserved.

Ruling: Work on main and push verified milestones without a new approval gate —
explicit repository/user authorization overrides skill branch and push defaults —
the tradeoff is shared-main coordination, mitigated by file ownership and tests.

Ruling: Root owns commits; independent workers may run in parallel with disjoint
file ownership — the user explicitly requested appropriate parallel work —
incorrect ownership would risk conflicts, so overlapping tasks remain sequential.

## Preflight interface review

| Tasks | Shared boundary | Result |
|---|---|---|
| 1 / 2 | Semantic pair input vs retained task/reading snapshots | Separate source ownership; both preserve complete ordered frames and current goal |
| 1 / 3 | Pure model state vs admission authentication | Task 3 starts after Task 1 interface verification; authenticate nested state |
| 2 / 3 | Context capture/validation | Candidate membership and missing origins retained, not filtered |
| 2 / 4 | tasks.py | Context worker finishes before root adds guarded create |
| 3 / 4 | association learning and revision support | Sequential integration; association handle reaches final ledger guard |
| 1 / 4 | Positive new-task relation | Dedicated relation; no exclusion-to-new fallback |
| 1 | Pure tests vs extraction | Existing goal/speech/scene behavior must remain covered; no synthetic goals for labels |
| 2 | Snapshot tests vs ledger interface | Callback-free membership/revision basis covers rival mutation |
| 3 | Selection vs adoption | Plan amended to include task_revision files and final guard |
| 4 | Routing vs task creation | Plan amended for whole-sentence goals, create-before-pursue, historical postcommit context |

## Review corrections incorporated before implementation

- Association evidence reaches correction provenance, task dependencies, final commit.
- Positive new-task routing requires one complete-sentence goal, not per-act creation.
- New-task creation is guarded and happens before effects.
- Precommit current comparisons become authenticated historical context after commit;
  reading/model/interpretation support remains live.

Task 1: running — /root/association_learner, gpt-5.6-sol high; base c04f787
Task 2: running — /root/association_context, gpt-5.6-sol high; base c04f787
Task 3: pending
Task 4: pending

Preflight refinement: retained teaching uses authenticated historical episode
comparisons; live routing uses current task-set validation. Collecting later
episodes must not invalidate earlier supervision. Added explicit tests to Tasks
2 and 3 and amended the context validation interface.

Task 4: root began independent complete-sentence goal teaching and routing regressions.
- tests/test_sentence_task_routing.py: 4 expected RED failures in 2.30s;
  current turn creates one task per act without discourse evidence. These remain
  failing pending association/routing integration; do not claim suite green.
- tests/test_sentence_goal_teaching.py: 5 RED failures in 1.58s for absent API.
- Added retain_taught_sentence_goal in agent/goal_interpretation.py using complete
  ordered Request frames and authenticated original snapshot, no inferred clause
  composition. Existing explicit single-frame teaching remains supplied semantics.
- GREEN: .venv/bin/pytest -q tests/test_sentence_goal_teaching.py
  tests/test_goal_learning_evidence.py tests/test_agent_learned_goals.py
  tests/test_task_revision_learning.py -> 41 passed in 8.24s.
- Root owns goal_interpretation.py, both new sentence test files; no core.py change
  yet. Complete new-task runtime goal projection and routing remain unfinished.

## Explicit user pause, 2026-09-20

All implementation/review agents stopped; no further fixes authorized during pause.
Task 1: implementation present; worker reports final 104 + 46 focused passes;
root task review still pending. Generic fields renamed structure/result.
Task 2: implementation present; review identified final dependency-guard omission
for unresolved members that still retain authentic origin evidence. Fix pending.
Task 3: not started.
Task 4: whole-sentence goal teaching/proposal present; 3 review findings unfixed.
Latest RED command:
.venv/bin/pytest -q tests/test_sentence_goal_teaching.py -k 'final_selection_callback or final_evidence_copy or proposal_comparison'
5 failed, 11 deselected in 1.84s. Plus 4 existing routing RED cases = 9 known
failing regression cases. No full-suite rerun for this WIP.
Tracked handoff: docs/revival/83-paused-task-association-checkpoint.md.

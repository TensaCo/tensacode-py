# Whole-sentence goal boundary: partial Task 4 implementation

Scope: complete-sentence supplied goal teaching and learned retained goal proposals.
Not implemented here: discourse routing, task creation, association adoption, or
automatic pursuit. tests/test_sentence_task_routing.py has 4 expected RED failures
and is outside this review scope; those remain pending runtime integration.

APIs: retain_taught_sentence_goal(agent, goal, source_text,
parent_dependency=..., reason=...) and retain_sentence_goal_proposals(agent,
group_id, candidate_id, basis=...). Input is an authenticated selected complete
ordered Request reading; internal Frame request-sequence contains every frame.
No inferred clause composition; goal teaching remains supplied. Runtime path
requires an admitted learned model and never falls back to lexical projection.

TDD RED: 5 teaching tests failed for absent API in 1.58s, then 6 runtime tests
failed for absent API in 1.98s. GREEN command:
.venv/bin/pytest -q tests/test_sentence_goal_teaching.py tests/test_goal_learning_evidence.py tests/test_agent_learned_goals.py tests/test_task_revision_learning.py tests/test_goal_interpretation_workspace.py
Result: 67 passed in 9.64s. No full-suite rerun yet.

Files owned by root: agent/goal_interpretation.py, agent/goal_learning.py,
tests/test_sentence_goal_teaching.py. Root will handle integration fixes.
Review source fidelity, callbacks, publication/selection support, dependency
propagation, and regression of supplied single-frame teaching. This partial
component is not claimed as normal chat capability.

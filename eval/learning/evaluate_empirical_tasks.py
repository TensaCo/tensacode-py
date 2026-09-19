"""Actual Gym pause/resume and mid-execution revision audit; authored goals/choices."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time
from unittest.mock import patch

from eval.learning.empirical_planning_fixture import GOAL, make_setup
from eval.learning.evaluate_model_investigation import encode, sha
from tensorcode.agent.empirical_tasks import EmpiricalGoal


def choose_route(actions):
    actions = iter(actions)
    def choose(plan):
        action = next(actions)
        return next(call for call in plan.first_calls if dict(call.args)['action'] == action)
    return choose


def goal(setup, state=GOAL):
    return EmpiricalGoal(state, setup.calls, setup.model.id, setup.model.provider)


def evidence(setup, outcomes):
    ids = tuple(dict.fromkeys(sid for outcome in outcomes for sid in outcome.plan.observation_source_ids))
    return [setup.agent.interpretations.get_source(sid) for sid in ids]


def model_record(setup):
    return {'id': setup.model.id, 'provider': setup.model.provider, 'policy': setup.model.policy,
            'projection': {'name': setup.model.projection.name, 'provenance': setup.model.projection.provenance},
            'edges': setup.model.edges, 'examples': setup.model.examples,
            'training_attempt_ids': setup.training, 'evaluation_attempt_ids': setup.evaluation}


def paused_task():
    setup = make_setup()
    try:
        before = setup.plugin.sequence
        first = setup.agent.pursue_empirical(setup.model, goal=goal(setup), max_steps=2,
            choose=choose_route((2, 2)), max_depth=4)
        suspended = setup.agent.tasks.get(first.task_id)
        halfway = setup.plugin.sequence
        second = setup.agent.pursue_empirical(setup.model, task_id=first.task_id, max_steps=2,
            choose=choose_route((1, 1)), max_depth=4)
        task = setup.agent.tasks.get(first.task_id)
        return {'model': model_record(setup), 'first_outcome': first, 'suspended_task': suspended,
                'resumed_outcome': second, 'final_task': task, 'observations': evidence(setup, (first, second)),
                'checks': {'first_suspended': first.status == 'suspended' and suspended.status == 'suspended',
                           'same_task_id': second.task_id == first.task_id,
                           'one_goal_revision': task.revision == 1 and len(task.revisions) == 1,
                           'two_revision_one_attempts': len(task.attempts) == 2 and all(a.revision == 1 for a in task.attempts),
                           'two_plus_two_actions': halfway == before+2 and setup.plugin.sequence == before+4,
                           'old_attempt_unchanged': task.attempts[0] == suspended.attempts[0],
                           'fresh_proposals': set(first.plan.proposal_ids).isdisjoint(second.plan.proposal_ids),
                           'fresh_executions': set(first.plan.execution_ids).isdisjoint(second.plan.execution_ids),
                           'actual_goal_completed': second.status == task.status == 'done' and second.verified is True,
                           'no_beliefs': not tuple(setup.agent.store.claims())}}
    finally:
        setup.plugin.close()


def revised_task():
    setup = make_setup()
    try:
        task = setup.agent.tasks.create('audit: explicit goal revised during applied action', goal(setup))
        original = setup.plugin.execute
        revised_goal = goal(setup, (6, False, False))
        def execute(act, *, key=None):
            receipt = original(act, key=key)
            if receipt.status == 'applied':
                setup.agent.tasks.revise(task.id, revised_goal, reason='audit supplies a different goal during execution')
            return receipt
        before = setup.plugin.sequence
        with patch.object(setup.plugin, 'execute', execute):
            outcome = setup.agent.pursue_empirical(setup.model, task_id=task.id, max_steps=4,
                choose=choose_route((2, 2, 1, 1)), max_depth=4)
        current = setup.agent.tasks.get(task.id)
        return {'model': model_record(setup), 'original_task': task, 'outcome': outcome,
                'revised_task': current, 'observations': evidence(setup, (outcome,)),
                'checks': {'one_action_only': setup.plugin.sequence == before+1,
                           'revision_change_reported': outcome.reason == 'task_revision_changed',
                           'new_goal_ready': current.revision == 2 and current.status == 'ready' and current.goal == revised_goal,
                           'old_revision_attempt': len(current.attempts) == 1 and current.attempts[0].revision == 1,
                           'old_receipt_retained': len(outcome.steps) == 1 and outcome.steps[0].receipt.status == 'applied'
                                                   and current.attempts[0].steps == outcome.steps,
                           'old_goal_retained': current.revisions[0].goal == task.goal and outcome.goal == task.goal,
                           'new_goal_not_completed': outcome.status == 'unknown',
                           'no_beliefs': not tuple(setup.agent.store.claims())}}
    finally:
        setup.plugin.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=Path('eval/results/empirical_tasks.json'))
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__).resolve(), root/'eval/learning/empirical_planning_fixture.py',
             root/'eval/learning/evaluate_model_investigation.py', root/'examples/general_agent/gym_connection.py',
             *[root/'src/tensorcode'/name for name in (
                 'agent/core.py', 'agent/tasks.py', 'agent/empirical_tasks.py', 'agent/empirical_planning.py',
                 'agent/empirical_execution.py', 'agent/experience_planning.py', 'agent/interpretation.py',
                 'agent/plugin.py', 'learning/empirical_dynamics.py', 'learning/experience.py')]]
    hashes = {str(path): sha(path) for path in files}
    started = time.perf_counter()
    cases = {'pause_resume': paused_task(), 'revision_during_action': revised_task()}
    report = {'evaluation': 'revision_bound_actual_gym_empirical_tasks', 'source_hashes': hashes,
              'source_hashes_verified_after_run': {p: sha(p) == h for p, h in hashes.items()},
              'gymnasium_version': version('gymnasium'), 'elapsed_ms': (time.perf_counter()-started)*1000,
              'protocol': {'pause_budgets': [2, 2], 'pause_choices': [2, 2, 1, 1], 'max_depth': 4,
                           'revision_budget': 4, 'original_goal': GOAL, 'revised_goal': (6, False, False)},
              'cases': cases, 'limitations': ['Goals, projections, exploration, and route choices authored.',
                  'Mid-action revision is an explicit test callback, not autonomous goal formation.',
                  'In-memory lifetime only; no process restart, natural-language request, or chat-default execution.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(encode(report), indent=2, allow_nan=False)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()

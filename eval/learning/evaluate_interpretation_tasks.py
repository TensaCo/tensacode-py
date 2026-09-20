"""Audit explicit interpretation commitments controlling actual Gym task resumption."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from eval.learning.empirical_planning_fixture import make_setup
from eval.learning.evaluate_empirical_tasks import choose_route, evidence, goal, model_record
from eval.learning.evaluate_model_investigation import encode, sha
from tensorcode.agent.task_dependencies import validate_dependency


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=Path('eval/results/interpretation_tasks.json'))
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__).resolve(), *[root/name for name in (
        'eval/learning/empirical_planning_fixture.py', 'eval/learning/evaluate_empirical_tasks.py',
        'eval/learning/evaluate_model_investigation.py', 'examples/general_agent/gym_connection.py')],
        *[root/'src/tensorcode'/name for name in (
            'agent/core.py', 'agent/task_dependencies.py', 'agent/tasks.py', 'agent/empirical_tasks.py',
            'agent/empirical_execution.py', 'agent/empirical_planning.py', 'agent/experience_planning.py',
            'agent/interpretation.py', 'agent/plugin.py', 'learning/empirical_dynamics.py', 'learning/experience.py')]]
    hashes = {str(path): sha(path) for path in paths}
    started = time.perf_counter()
    setup = make_setup()
    try:
        agent, workspace = setup.agent, setup.agent.interpretations
        source = workspace.add_source('Audit supplied meaning: proceed to the designated lake goal.',
                                      provider='audit:authored-input-not-learned-language')
        group = workspace.create_group(source.id)
        candidate = workspace.propose(group.id, {'authored_meaning': 'designated goal', 'goal_state': (8, True, False)},
                                      provenance=('audit:authored-candidate',))
        workspace.select(group.id, candidate.id, reason='audit supplies selection, not inferred intent')
        old_dependency = agent.capture_task_dependency(group.id,
            basis=('Audit author associates this selected candidate with the explicit empirical goal',),
            evidence_ids=(source.id,))
        initial_group = workspace.get(group.id)
        sequence = setup.plugin.sequence
        first = agent.pursue_empirical(setup.model, goal=goal(setup), dependencies=(old_dependency,),
            max_steps=2, choose=choose_route((2, 2)), max_depth=4)
        paused = agent.tasks.get(first.task_id)
        halfway = setup.plugin.sequence
        unrelated_source = workspace.add_source('Unrelated authored comparison', provider='audit:unrelated')
        unrelated = workspace.create_group(unrelated_source.id)
        other = workspace.propose(unrelated.id, {'unrelated': True}, provenance=('audit:authored',))
        workspace.select(unrelated.id, other.id, reason='unrelated authored selection')
        unrelated_valid = validate_dependency(workspace, old_dependency) is True
        workspace.unset(group.id, reason='audit explicitly withdraws the supporting interpretation')
        withdrawn = agent.pursue_empirical(setup.model, task_id=first.task_id, max_steps=2,
            choose=choose_route((1, 1)), max_depth=4)
        after_withdrawn = setup.plugin.sequence
        withdrawn_group = workspace.get(group.id)
        workspace.select(group.id, candidate.id, reason='audit explicitly reselects the same candidate')
        reselected = agent.pursue_empirical(setup.model, task_id=first.task_id, max_steps=2,
            choose=choose_route((1, 1)), max_depth=4)
        after_reselected = setup.plugin.sequence
        fresh_dependency = agent.capture_task_dependency(group.id,
            basis=('Explicit renewed audit commitment to the current interpretation revision',),
            evidence_ids=(source.id,))
        before_revision = agent.tasks.get(first.task_id)
        revised = agent.tasks.revise(first.task_id, goal(setup), reason='explicit fresh interpretation commitment',
                                    dependencies=(fresh_dependency,))
        done = agent.pursue_empirical(setup.model, task_id=first.task_id, max_steps=2,
            choose=choose_route((1, 1)), max_depth=4)
        final = agent.tasks.get(first.task_id)
        report = {'evaluation': 'actual_gym_interpretation_dependent_tasks', 'source_hashes': hashes,
                  'source_hashes_verified_after_run': {p: sha(p) == h for p,h in hashes.items()},
                  'raw_source': source, 'initial_group': initial_group, 'withdrawn_group': withdrawn_group,
                  'final_group': workspace.get(group.id), 'unrelated_group': workspace.get(unrelated.id),
                  'old_dependency': old_dependency, 'fresh_dependency': fresh_dependency,
                  'paused_task': paused, 'task_before_revision': before_revision, 'revised_task': revised,
                  'final_task': final, 'outcomes': [first, withdrawn, reselected, done],
                  'observations': evidence(setup, (first, withdrawn, reselected, done)), 'model': model_record(setup),
                  'elapsed_ms': (time.perf_counter()-started)*1000,
                  'checks': {'initial_two_steps_suspended': first.status == 'suspended' and halfway == sequence+2,
                             'unrelated_group_did_not_invalidate': unrelated_valid,
                             'withdrawal_blocks_dispatch': withdrawn.reason == 'interpretation_dependency_changed' and after_withdrawn == halfway,
                             'same_candidate_reselection_still_blocks': reselected.reason == 'interpretation_dependency_changed' and after_reselected == halfway,
                             'fresh_dependency_new_revision_same_candidate': fresh_dependency.revision > old_dependency.revision and fresh_dependency.candidate_id == old_dependency.candidate_id,
                             'explicit_task_revision_replaces_dependency': revised.revision == 2 and revised.dependencies == (fresh_dependency,),
                             'old_revision_dependency_retained': final.revisions[0].dependencies == (old_dependency,),
                             'prior_attempts_unchanged': final.attempts[:len(before_revision.attempts)] == before_revision.attempts,
                             'completed_without_replay': done.status == final.status == 'done' and done.verified is True and setup.plugin.sequence == sequence+4,
                             'four_attributed_attempts': tuple(a.revision for a in final.attempts) == (1,1,1,2),
                             'same_task_id': len({o.task_id for o in (first,withdrawn,reselected,done)}) == 1,
                             'source_unchanged': workspace.get_source(source.id) == source,
                             'no_beliefs': not tuple(agent.store.claims())},
                  'limitations': ['Candidate, selection, goal association, projection, and tie choices authored.',
                                  'No learned-language reading or inferred meaning-to-goal entailment.',
                                  'Explicit fresh commitment required; no automatic replacement goal or action rollback.',
                                  'In-memory workspace/task history; no restart persistence.']}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(encode(report), indent=2, allow_nan=False)+'\n')
        print(args.output)
    finally:
        setup.plugin.close()


if __name__ == '__main__':
    main()

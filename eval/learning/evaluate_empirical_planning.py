"""Audit composed finite-state plans using actual FrozenLake reset/step evidence.

Projection, map, exploration prefixes, goal, and explicit tie choices are authored.
No transition-table access or direct environment state mutation supplies samples.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time

from eval.learning.empirical_planning_fixture import GOAL, latest, make_setup
from eval.learning.evaluate_model_investigation import encode, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=Path('eval/results/empirical_planning.json'))
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__).resolve(), root / 'eval/learning/empirical_planning_fixture.py',
             root / 'eval/learning/evaluate_model_investigation.py',
             root / 'examples/general_agent/gym_connection.py',
             *[root / 'src/tensorcode' / name for name in (
                 'agent/core.py', 'agent/empirical_planning.py', 'agent/empirical_execution.py',
                 'agent/experience_planning.py', 'agent/interpretation.py', 'agent/plugin.py',
                 'learning/empirical_dynamics.py', 'learning/experience.py')]]
    hashes = {str(path): sha(path) for path in paths}
    started = time.perf_counter()
    setup = make_setup()
    setup_ms = (time.perf_counter() - started) * 1000
    route = (2, 2, 1, 1)  # Predeclared explicit tie choices, not a discovered preference.
    try:
        initial_sequence = setup.plugin.sequence
        initial_claims = tuple(setup.agent.store.claims())
        steps = []
        for index, action in enumerate(route):
            before = time.perf_counter()
            sequence = setup.plugin.sequence
            proposal = setup.agent.propose_empirical_plan(
                setup.model, latest(setup.agent, setup.plugin).id, setup.calls, GOAL,
                max_depth=4-index, max_states=10000, max_edges=100000)
            no_action = setup.plugin.sequence == sequence
            call = next(c for c in proposal.plan.first_calls if dict(c.args)['action'] == action)
            result = setup.agent.execute_empirical_plan(proposal.id, call=call)
            after_sequence = setup.plugin.sequence
            replay = setup.agent.execute_empirical_plan(proposal.id, call=call)
            steps.append({'index': index, 'proposal': proposal, 'explicit_call': call,
                          'result': result, 'replay_reason': replay.reason,
                          'proposal_did_not_act': no_action,
                          'exactly_one_step': after_sequence == sequence + 1,
                          'replay_did_not_act': setup.plugin.sequence == after_sequence,
                          'latency_ms': (time.perf_counter()-before)*1000,
                          'observation_sources': [setup.agent.interpretations.get_source(sid)
                                                  for sid in result.source_ids]})
        model = setup.model
        report = {'evaluation': 'empirical_frozenlake_composed_route', 'source_hashes': hashes,
                  'source_hashes_verified_after_run': {p: sha(p) == h for p, h in hashes.items()},
                  'environment': {'gymnasium_version': version('gymnasium'), 'name': 'FrozenLake-v1',
                                  'map': ['SFF', 'FFF', 'FFG'], 'is_slippery': False, 'seed': 0},
                  'authored': ['map', 'state projection', 'goal', 'actions', 'exploration prefixes',
                               'sample support thresholds', 'tie choices'],
                  'goal': GOAL, 'executed_actions': route, 'exploration_episodes': setup.episodes,
                  'training_attempt_ids': setup.training, 'evaluation_attempt_ids': setup.evaluation,
                  'model': {'id': model.id, 'revision': model.revision, 'policy': model.policy,
                            'projection': {'name': model.projection.name, 'provenance': model.projection.provenance},
                            'states': model.states, 'calls': model.calls, 'edges': model.edges,
                            'examples': model.examples},
                  'steps': steps, 'setup_ms': setup_ms,
                  'checks': {'new_complete_episode': route not in setup.episodes,
                             'split_ids_disjoint': set(setup.training).isdisjoint(setup.evaluation),
                             'all_edges_eligible': all(edge.eligible for edge in model.edges),
                             'all_outcomes_two_train_one_validation': all(
                                 len(outcome.training_attempt_ids) == 2 and len(outcome.evaluation_attempt_ids) == 1
                                 for edge in model.edges for outcome in edge.outcomes),
                             'four_actual_steps': setup.plugin.sequence == initial_sequence + 4,
                             'all_proposals_action_free': all(s['proposal_did_not_act'] for s in steps),
                             'all_replays_blocked': all(s['replay_did_not_act'] and s['replay_reason'] == 'proposal_already_consumed' for s in steps),
                             'goal_observed': steps[-1]['result'].verification is True,
                             'intermediate_not_goal': all(s['result'].verification is False for s in steps[:-1]),
                             'no_beliefs_added': tuple(setup.agent.store.claims()) == initial_claims == (),
                             'no_capability_effects': not any(cap.effects for cap in setup.plugin.capabilities())},
                  'task_count': len(setup.agent.tasks),
                  'limitations': ['Novel complete route composes familiar edges; no unseen-state generalization.',
                                  'All outcome support is empirical, not exhaustive future possibility.',
                                  'Explicit shortest-action tie choices supplied; no learned preference.',
                                  'Independent proposal API; four steps require explicit fresh replanning calls.']}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(encode(report), indent=2, allow_nan=False)+'\n')
        print(args.output)
    finally:
        setup.plugin.close()


if __name__ == '__main__':
    main()

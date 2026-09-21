"""Measured feedback in an explicitly authored, tiny service-recovery simulation.

This is a mechanism demonstration, not real-world competence or causal inference.
The environment, actions, telemetry semantics, exploration and reward are fixtures.
Nothing here adds domain policy to the library core. Train/test scenario IDs and
telemetry strings are disjoint; both share the same three authored status classes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

from tensorcode import training
from tensorcode.runtime.action_loop import ActionOutcome
from tensorcode.runtime.planning import ExecutablePlan, PlanExecutor, PlanStep, PlanExecutionResult
from tensorcode.tools.planner import Planner

CANDIDATES = [{'id': name, 'text': name} for name in ('cool', 'reindex', 'serve')]
TRAIN = [{'scenario': f'train-{i}', 'status': status} for i, status in enumerate(
    ['hot', 'corrupt', 'ready', 'hot', 'corrupt', 'ready'])]
TEST = [{'scenario': f'test-{i}', 'status': status} for i, status in enumerate(
    ['hot', 'corrupt', 'hot', 'corrupt', 'hot', 'corrupt'])]


def registry():
    def transition(name):
        def action(state):
            state = deepcopy(state)
            before = state['status']
            done = name == 'serve' and before == 'ready'
            repaired = (name, before) in [('cool', 'hot'), ('reindex', 'corrupt')]
            if repaired:
                state['status'] = 'ready'
            if done:
                state['status'] = 'serving'
            reward = 1. if done else .5 if repaired else -.5
            return ActionOutcome(state, {'before': before, 'after': state['status'],
                                         'reward': reward, 'scenario': state['scenario']}, done)
        return action
    return {name: transition(name) for name in ('cool', 'reindex', 'serve')}


def inputs(state):
    return {'goal': 'restore service', 'plans': deepcopy(CANDIDATES),
            'evidence': [{'source_id': state['scenario'] + ':telemetry:' + state['status'],
                          'text': f"status {state['status']} scenario {state['scenario']}"}]}


def structured(candidate_id):
    if candidate_id not in {item['id'] for item in CANDIDATES}:
        raise ValueError('selected candidate is not registered; no fallback')
    return ExecutablePlan(candidate_id, (PlanStep(candidate_id),))


def run_scenario(model, state, *, baseline=False):
    def choose(current):
        return structured('cool' if baseline else model(inputs(current))['selected_id'])
    return PlanExecutor(actions=registry(), replan=lambda request: choose(request.state),
                        max_steps=2)(state, choose(state))


def evaluate(model, *, baseline=False):
    runs = [run_scenario(model, state, baseline=baseline) for state in TEST]
    return {'success_rate': sum(r.stop_reason == 'completed' for r in runs) / len(runs),
            'mean_reward': sum(sum(x.observation['reward'] for x in r.experiences) for r in runs) / len(runs),
            'scenarios': len(runs)}


def run(output, *, epochs=18, seed=12):
    if type(epochs) is not int or epochs < 1:
        raise ValueError('epochs must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = Planner({'vocabulary': ['restore', 'service', 'status', 'hot', 'corrupt', 'ready',
                                    'scenario', 'train', 'test', 'cool', 'reindex', 'serve'],
                     'dimensions': 12, 'slots': 2, 'steps': 1})
    trainer = training.ToolTrainer(model, optimizer=lambda p: torch.optim.Adam(p, lr=.008))
    before = evaluate(model)
    baseline = evaluate(model, baseline=True)
    sessions, observations = [], []
    # Explicit exploration: every label comes from a fresh actual transition.
    # Even with all candidates in the input, each trace labels only its executed ID.
    for state in TRAIN:
        for candidate in CANDIDATES:
            trajectory = PlanExecutor(actions=registry(), replan=lambda _: None, max_steps=1)(
                state, structured(candidate['id']))
            experience = trajectory.experiences[0]
            target = experience.to_target(experience.observation['reward'])
            session = trainer.capture(inputs(state), target, source=experience.source_id)
            name = f'experience-{len(sessions):02d}.json'
            session.save(output / name, operations=trainer.operations)
            trajectory.save(output / name.replace('experience', 'trajectory'))
            sessions.append(session)
            observations.append({'scenario': state['scenario'], 'candidate_id': candidate['id'],
                                 'source_id': experience.source_id, 'target': target, 'file': name})
    losses = []
    for _ in range(epochs):
        losses.append(sum(trainer.step(session) for session in sessions) / len(sessions))
    after = evaluate(model)
    model.save_pretrained(output / 'model')
    trainer.save_checkpoint(output / 'training', progress={'epochs': epochs})
    session = model.new_session()
    expected = session(inputs(TEST[0]))
    session.save(output / 'session.json')
    trajectory = run_scenario(model, TEST[0])
    trajectory.save(output / 'evaluation-trajectory.json')
    restored = Planner.from_pretrained(output / 'model', local_files_only=True)
    resumed = training.ToolTrainer(restored, optimizer=lambda p: torch.optim.Adam(p, lr=.008))
    resumed.load_checkpoint(output / 'training')
    loaded_trace = training.load(output / observations[0]['file'], operations=resumed.operations)
    # Session adds revision fields to its receipt; compare stable candidate scores.
    parity = restored(inputs(TEST[0]))['candidates'] == expected['candidates']
    session_parity = type(session).load(output / 'session.json', restored).history == session.history
    trajectory_parity = PlanExecutionResult.load(output / 'evaluation-trajectory.json') == trajectory
    expected_loss = trainer.step(sessions[0])
    actual_loss = resumed.step(loaded_trace)
    continuation = expected_loss == actual_loss and all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters()))
    try:
        model.loss(inputs(TRAIN[0]), {'candidate_id': 'unseen-action', 'outcome': 0.})
        unknown_rejected = False
    except ValueError:
        unknown_rejected = True
    frozen_feedback_runs = [PlanExecutor(actions=registry(),
        replan=lambda request: request.previous_plan, max_steps=2)(state,
        structured(restored(inputs(state))['selected_id'])) for state in TEST]
    report = {'scope': 'Authored deterministic simulation; shared status classes; no real-world competence or causal estimate.',
        'seed': seed, 'epochs': epochs, 'updates': epochs * len(sessions),
        'train_ids': [x['scenario'] for x in TRAIN], 'test_ids': [x['scenario'] for x in TEST],
        'before': before, 'after': after, 'fixed_cool_baseline': baseline,
        'frozen_feedback_success_rate': sum(x.stop_reason == 'completed' for x in frozen_feedback_runs) / len(TEST),
        'first_epoch_loss': losses[0], 'last_epoch_loss': losses[-1],
        'observations': len(observations), 'labels_per_observation': 1,
        'model_parity': parity, 'session_parity': session_parity, 'trajectory_parity': trajectory_parity,
        'optimizer_continuation_parity': continuation, 'unseen_label_rejected': unknown_rejected}
    (output / 'observations.json').write_text(json.dumps(observations, indent=2) + '\n')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=18)
    args = parser.parse_args()
    print(json.dumps(run(args.output, epochs=args.epochs), indent=2))

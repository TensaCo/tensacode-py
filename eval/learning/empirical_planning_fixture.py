"""Real FrozenLake executions; supplied measurements and exploration, learned edges.

The map and exploration routes are authored. The implementation never reads the
Gym transition table or mutates environment state. Every sample comes through
Agent._invoke. Complete exploration episodes are retained to audit route novelty.
"""
from dataclasses import dataclass
import gymnasium as gym
from examples.general_agent.gym_connection import GymPlugin
from tensorcode.agent import Agent
from tensorcode.agent.plugin import Call
from tensorcode.learning.experience import extract_transitions
from tensorcode.learning.empirical_dynamics import StateProjection, fit_dynamics


PROJECTION = StateProjection('discrete-gym-state-and-termination',
    lambda o: (int(o['transition']['observation']), o['transition']['terminated'], o['transition']['truncated']),
    ('Authored measurement of Gym discrete observation and terminal flags; no reward-to-goal convention',))
GOAL = (8, True, False)
# Separate prefixes reach each nonterminal state. They are exploration policy,
# not effects passed to the learner or routes supplied to the planner.
PREFIXES = ((), (2,), (2, 2), (1,), (2, 1), (1, 2, 2), (1, 1), (2, 1, 1))


def invoke(agent, plugin, capability, **args):
    events = []
    cap = next(c for c in plugin.capabilities() if c.name == capability)
    receipt = agent._invoke(plugin, cap, args, events)
    assert receipt.status == 'applied', receipt
    return next(e['attempt_id'] for e in events if e['type'] == 'receipt')


def latest(agent, plugin):
    return [s for s in agent.interpretations.sources()
            if s.provider == 'plugin:' + plugin.name and s.modality == 'observation'][-1]


@dataclass
class Setup:
    agent: Agent
    plugin: GymPlugin
    model: object
    calls: tuple
    episodes: tuple
    training: tuple
    evaluation: tuple


def make_setup():
    plugin = GymPlugin(gym.make('FrozenLake-v1', desc=['SFF', 'FFF', 'FFG'], is_slippery=False),
                       name='empirical-lake')
    agent = Agent([plugin])
    training, evaluation, episodes = [], [], []
    try:
        for trial in range(3):
            for prefix in PREFIXES:
                for action in range(4):
                    invoke(agent, plugin, 'reset', seed=0)
                    for step in prefix:
                        invoke(agent, plugin, 'step', action=step)
                    attempt = invoke(agent, plugin, 'step', action=action)
                    (training if trial < 2 else evaluation).append(attempt)
                    episodes.append((*prefix, action))
        rows = extract_transitions(agent.interpretations.sources(), provider='plugin:' + plugin.name).transitions
        ids = set(training + evaluation)
        model = fit_dynamics(tuple(r for r in rows if r.attempt_id in ids), projection=PROJECTION,
                             train_attempt_ids=training, evaluation_attempt_ids=evaluation)
        invoke(agent, plugin, 'reset', seed=0)
        calls = tuple(Call(plugin.name, 'step', (('action', action),)) for action in range(4))
        return Setup(agent, plugin, model, calls, tuple(episodes), tuple(training), tuple(evaluation))
    except BaseException:
        plugin.close()
        raise

"""Authored hidden wiring contexts, with rules fitted from executed observations.

The fixture supplies the environment, context partition, projection and candidate
bindings. Lamp outcomes and action associations are learned from actual receipts.
"""
from dataclasses import dataclass

from tensorcode.agent import Agent
from tensorcode.agent.plugin import Call, Capability, Param, Plugin
from tensorcode.learning.experience import Projection, extract_transitions, fit_transitions
from tensorcode.outcomes import Receipt, Unknown


PROJECTION = Projection(
    'switch-lamp-and-button/v1',
    lambda observation, action: {'lamp': observation['lamp'], 'button': action.arg('button')},
    lambda observation: observation['lamp'],
    ('Authored lamp/button projection; hidden wiring is excluded from observations',))


class Switch(Plugin):
    def __init__(self, *, mode=2, tie=False):
        super().__init__('switch')
        self.mode = mode
        self.tie = tie
        self.lamp = 0
        self.executions = []

    def capabilities(self):
        return (Capability('press', (Param('button', 'integer'),), description='Press a switch button'),)

    def observe_evidence(self):
        return {'lamp': self.lamp}

    def execute(self, act, *, key=None):
        button = act.arg('button')
        if act.plugin != self.name or act.capability != 'press' or type(button) is not int or button not in (0, 1, 2):
            return Receipt(act, 'rejected', error='invalid button call')
        self.lamp = self.mode if button == 1 or (button == 2 and self.tie) else 0
        self.executions.append(act)
        return Receipt(act, 'applied')


def invoke(agent, plugin, button):
    events = []
    receipt = agent._invoke(plugin, plugin.capabilities()[0], {'button': button}, events)
    assert receipt.status == 'applied'
    return next(event['attempt_id'] for event in events if event['type'] == 'receipt')


def latest(agent, plugin):
    return tuple(source for source in agent.interpretations.sources()
                 if source.provider == 'plugin:' + plugin.name and source.modality == 'observation')[-1]


def train_models(agent, plugin):
    models = []
    for mode in (1, 2):
        plugin.mode = mode  # Authored experimental context, never a model input.
        train, held = [], []
        for button in ((0, 1, 2) if plugin.tie else (0, 1)):
            # More nonzero-action support ensures its distinguishing association
            # can pay the inducer's description cost; no induced rule is edited.
            train_count = 8 if button == 0 else 12
            for trial in range(train_count + 2):
                invoke(agent, plugin, 0)
                attempt = invoke(agent, plugin, button)
                (train if trial < train_count else held).append(attempt)
        selected = set(train + held)
        rows = tuple(row for row in extract_transitions(
            agent.interpretations.sources(), provider='plugin:' + plugin.name).transitions
                     if row.attempt_id in selected)
        model = fit_transitions(rows, projection=PROJECTION, train_attempt_ids=train,
                                evaluation_attempt_ids=held)
        prediction = model.predict({'lamp': 0}, Call(plugin.name, 'press', (('button', 1),)))
        assert not isinstance(prediction, Unknown), prediction
        assert prediction.outcome == mode
        models.append(model)
    return tuple(models)


@dataclass
class Setup:
    agent: Agent
    plugin: Switch
    models: tuple
    group_id: str
    candidate_ids: tuple[str, ...]
    before_source_id: str
    calls: tuple[Call, ...]


def make_setup(*, mode=2, tie=False):
    plugin = Switch(mode=mode, tie=tie)
    agent = Agent([plugin])
    models = train_models(agent, plugin)
    source = agent.interpretations.add_source('Which learned wiring context applies here?', provider='test:authored-contexts')
    group = agent.interpretations.create_group(source.id)
    candidates = tuple(agent.interpretations.propose(group.id, {'context': number},
                       provenance=('authored alternative context partition',)) for number in (1, 2))
    plugin.mode = mode
    invoke(agent, plugin, 0)
    return Setup(agent, plugin, models, group.id, tuple(c.id for c in candidates),
                 latest(agent, plugin).id,
                 tuple(Call(plugin.name, 'press', (('button', button),))
                       for button in ((1, 2) if tie else (1,))))


def bindings(setup):
    from tensorcode.agent.experience_investigation import ModelApplicability
    return tuple(ModelApplicability(candidate_id, model,
                 ('Authored candidate-to-experimental-context correspondence',))
                 for candidate_id, model in zip(setup.candidate_ids, setup.models))


def unknown_model(setup):
    """Fit a baseline-only context; unfamiliar button evidence remains unknown."""
    examples = tuple(example for example in setup.models[0].examples if example.action.arg('button') == 0)
    train = tuple(example.attempt_id for example in examples if example.split == 'training')
    held = tuple(example.attempt_id for example in examples if example.split == 'evaluation')
    selected = set(train + held)
    rows = tuple(row for row in extract_transitions(setup.agent.interpretations.sources(),
                 provider=setup.models[0].provider).transitions if row.attempt_id in selected)
    return fit_transitions(rows, projection=PROJECTION, train_attempt_ids=train,
                           evaluation_attempt_ids=held)


def noisy_model(setup):
    """Learn an empirical {1, 2} context from an authored execution schedule.

    The experiment uses ten (1,1,1,2) training cycles and two independent
    validation cycles. A 75% validation threshold permits the learned dominant
    label while preserving the minority outcomes as evidence, not impossibility.
    """
    from tensorcode.learning.experience import ValidationPolicy
    train, held = [], []
    schedule = [(0, 1, 'training')] * 8 + [(0, 1, 'evaluation')] * 2
    schedule += [(1, mode, 'training') for mode in (1, 1, 1, 2) * 10]
    schedule += [(1, mode, 'evaluation') for mode in (1, 1, 1, 2) * 2]
    for button, mode, split in schedule:
        setup.plugin.mode = mode
        invoke(setup.agent, setup.plugin, 0)
        attempt = invoke(setup.agent, setup.plugin, button)
        (train if split == 'training' else held).append(attempt)
    selected = set(train + held)
    rows = tuple(row for row in extract_transitions(setup.agent.interpretations.sources(),
                 provider=setup.models[0].provider).transitions if row.attempt_id in selected)
    model = fit_transitions(rows, projection=PROJECTION, train_attempt_ids=train,
                           evaluation_attempt_ids=held, policy=ValidationPolicy(min_accuracy=0.75))
    prediction = model.predict({'lamp': 0}, setup.calls[0])
    assert not isinstance(prediction, Unknown), prediction
    assert prediction.outcome == 1
    return model


def make_noisy_setup(*, mode=2):
    setup = make_setup(mode=mode)
    setup.models = (noisy_model(setup), setup.models[1])
    setup.plugin.mode = mode
    invoke(setup.agent, setup.plugin, 0)
    setup.before_source_id = latest(setup.agent, setup.plugin).id
    return setup

"""Selected authored readings must not lose obligations at lexical goal projection."""
import pytest

from agent_test_support import selected_agent, select_unique_fixture_goal
from tensorcode.agent import Capability, Effect, Param, Plugin
from tensorcode.agent.operations import Transcript
from tensorcode.agent.understand import Act, Sentence
from tensorcode.language import Entity, Frame, Request, verbnet
from tensorcode.outcomes import Receipt
from tensorcode.records import Ref

TARGET = Ref('world:report')


class Eraser(Plugin):
    def __init__(self):
        super().__init__('eraser')
        self.calls = []

    def capabilities(self):
        return (Capability('erase', (Param('target', 'thing'),),
                           effects=(Effect('gone', {'undergoer': 'target'}),)),)

    def execute(self, call, *, key):
        self.calls.append(call)
        return Receipt(call, 'applied')

    def holds(self, capability, args):
        return True


def lexical_resource():
    # Explicit resource fixture, not a learned reading or learned lexical meaning.
    semantics = (verbnet.Pred('exists', (('Event', 'e1'), ('ThemRole', 'Theme'))),
                 verbnet.Pred('gone', (('Event', 'e2'), ('ThemRole', 'Theme'))))
    frame = verbnet.VFrame('NP V NP', (('NP', 'Agent'), ('VERB', ''), ('NP', 'Theme')), semantics)
    return {'erase': (verbnet.VerbClass('authored:erase', ('erase',), (frame,)),)}


def run(monkeypatch, frame, plugin=None, *, lexicon=None):
    from tensorcode.agent import core
    act = Act('request', Request(frame), frame)
    sentence = Sentence('authored request evidence', ('erase', 'report'), None, (act,))
    monkeypatch.setattr(core.ops, 'parse', lambda *a, **kw: Transcript((sentence,), 'authored fixture'))
    plugin = plugin or Eraser()
    agent = selected_agent([plugin], goal_selector=select_unique_fixture_goal)
    agent.verbs = lexical_resource() if lexicon is None else lexicon
    turn = agent.turn(sentence.text)
    return agent, plugin, turn.outcomes[0]


@pytest.mark.parametrize('feature,value', [('polarity', 'negative'), ('modality', 'possible'),
                                          ('count', 3), ('new-constraint', {'preserve': TARGET})])
def test_selected_request_frame_qualifiers_survive_without_dispatch(monkeypatch, feature, value):
    frame = Frame('erase', {'object': TARGET}, {'mood': 'imperative', feature: value})
    agent, plugin, outcome = run(monkeypatch, frame)
    assert outcome.status == 'unknown'
    assert outcome.verified.reason == 'unconsumed_request_semantics'
    assert feature in outcome.reason
    assert not plugin.calls
    task = agent.tasks.get(outcome.task_id)
    assert task.goal.frame == frame and task.dependencies
    # Resuming the retained lexical goal cannot evade the request boundary.
    resumed = agent.pursue(task_id=task.id)
    assert resumed.status == 'unknown' and not plugin.calls
    assert agent.plans(task.goal)[0] == []


def test_unmapped_preservation_role_cannot_be_dropped(monkeypatch):
    frame = Frame('erase', {'object': TARGET, 'preserve': Ref('world:notes')}, {'mood': 'imperative'})
    agent, plugin, outcome = run(monkeypatch, frame)
    assert outcome.status == 'unknown' and outcome.goal.reason == 'unresolved_goal_projection'
    group = agent.interpretations.get(outcome.goal_interpretation_id)
    assert group.selected_id is None
    assert any(candidate.payload.goal.unmapped_roles == ('preserve',) for candidate in group.candidates)
    assert not plugin.calls
    with pytest.raises(ValueError, match='interpreted goal'):
        agent.pursue(task_id=outcome.task_id)


def test_explicit_actor_cannot_be_replaced_with_implicit_addressee(monkeypatch):
    frame = Frame('erase', {'subject': Ref('agent:someone-else'), 'object': TARGET}, {'mood': 'imperative'})
    _, plugin, outcome = run(monkeypatch, frame)
    assert outcome.status == 'unknown' and 'subject' in outcome.reason
    assert not plugin.calls


@pytest.mark.parametrize('features', [{'count': 3}, {'modifiers': (('amod', 'temporary'),)}])
def test_selected_grounded_entity_qualifiers_are_retained(monkeypatch, features):
    entity = Entity('description', 'qualified report', features, TARGET)
    frame = Frame('erase', {'object': entity}, {'mood': 'imperative'})
    agent, plugin, outcome = run(monkeypatch, frame)
    assert outcome.status == 'declined'
    assert 'unconsumed entity features' in outcome.reason
    assert agent.tasks.get(outcome.task_id).goal.frame.roles['object'] == entity
    assert not plugin.calls


def test_refinement_cannot_silently_consume_negative_request(monkeypatch):
    class LossyRefiner(Eraser):
        def refine_goal(self, goal):
            pytest.fail('unsupported frame semantics reached a lossy refiner')
    frame = Frame('erase', {'object': TARGET}, {'mood': 'imperative', 'polarity': 'negative'})
    _, plugin, outcome = run(monkeypatch, frame, LossyRefiner())
    assert outcome.status == 'unknown' and not plugin.calls


def test_unqualified_grounded_request_still_executes(monkeypatch):
    frame = Frame('erase', {'object': TARGET}, {'mood': 'imperative'})
    _, plugin, outcome = run(monkeypatch, frame)
    assert outcome.status == 'done' and outcome.verified is True
    assert len(plugin.calls) == 1 and plugin.calls[0].arg('target') == TARGET


@pytest.mark.skipif(verbnet.find_verbnet() is None, reason='requires installed VerbNet')
@pytest.mark.parametrize('case', ['negative', 'preserve', 'count'])
def test_installed_lexical_resource_cannot_authorize_partial_deletion(monkeypatch, case):
    from test_agent_operations import Papers
    target = Entity('name', 'report.txt', {'count': 3} if case == 'count' else {}, Ref('path:/h/report.txt'))
    roles = {'object': target}
    features = {'mood': 'imperative'}
    if case == 'negative':
        features['polarity'] = 'negative'
    if case == 'preserve':
        roles['preserve'] = Ref('path:/h/notes.txt')
    frame = Frame('delete', roles, features)
    agent, plugin, outcome = run(monkeypatch, frame, Papers(), lexicon=verbnet.load())
    group = agent.interpretations.get(outcome.goal_interpretation_id)
    proposals = [candidate.payload for candidate in group.candidates
                 if isinstance(candidate.payload, verbnet.GoalProposal)]
    assert proposals and all(isinstance(proposal.goal, verbnet.Goal) for proposal in proposals)
    assert all(proposal.goal.frame == frame for proposal in proposals), 'retain actual lexical translation inputs'
    if case == 'preserve':
        assert group.selected is not None
        assert 'preserve' in group.selected.payload.goal.unmapped_roles
        assert outcome.goal == group.selected.payload.goal
        assert outcome.verified.reason == 'unconsumed_request_semantics'
        assert all('preserve' in proposal.goal.unmapped_roles for proposal in proposals)
    assert outcome.status in ('unknown', 'declined')
    assert not plugin.calls and plugin.fs == {'/h/report.txt', '/h/notes.txt'}

"""Selected interpretations cannot silently acquire additional semantic commitments."""
from tensorcode import ops
from tensorcode.agent import Agent, InterpretationDecision, Outcome
from tensorcode.agent.operations import Transcript
from tensorcode.agent.understand import Act, Sentence, act_of
from tensorcode.language import Entity, Frame, Question, Request
from tensorcode.records import Claim, Evidence, Ref
from tensorcode.outcomes import Unknown


def test_request_attachment_is_preserved():
    frame = Frame('make', {'object': Entity('noun', 'project'), 'name': Entity('name', 'hello')})
    act = act_of(Request(frame))
    assert act.frame == frame
    assert 'name' in act.frame.roles
    assert 'name' not in act.frame.roles['object'].features


def test_selected_candidate_is_not_resolved_again_after_selection(monkeypatch):
    frame = Frame('delete', {'object': Entity('pronoun', 'it', {'person': 3})})
    act = Act('request', Request(frame), frame)
    sentence = Sentence('delete it', ('delete', 'it'), None, (act,))
    monkeypatch.setattr(ops, 'parse', lambda *a, **kw: Transcript((sentence,), 'test'))
    agent = Agent(interpretation_selector=lambda group:
                  InterpretationDecision(group.candidates[0].id, 'caller supplies this unresolved candidate'))
    agent.context.observe(Entity('file', 'report', ref=Ref('file:report')))
    seen = []
    def handle(sentence, selected, events, **kwargs):
        seen.append(selected)
        return Outcome(selected, 'unknown', reason='unresolved referent')
    monkeypatch.setattr(agent, 'handle', handle)
    agent.turn('delete it')
    assert seen[0].frame == frame
    assert seen[0].frame.roles['object'].ref is None


def test_selected_location_is_not_rewritten_to_time(monkeypatch):
    agent = Agent()
    frame = Frame('be', {'subject': Entity('noun', 'meeting'), 'location': Entity('noun', 'Tuesday')})
    act = Act('tell', frame, frame)
    sentence = Sentence('the meeting is on Tuesday', (), None, (act,))
    seen = []
    def tell(sentence, selected, events):
        seen.append(selected.frame)
        return Outcome(selected, 'noted')
    monkeypatch.setattr(agent, 'tell', tell)
    agent.handle(sentence, act, [], requests_in_message=0)
    assert 'location' in seen[0].roles and 'time' not in seen[0].roles


def test_binary_claim_is_not_an_undirected_question_fallback():
    agent = Agent()
    first, second = Ref('entity:first'), Ref('entity:second')
    agent.store.tell(Claim(first, 'follows', second), Evidence(source=Ref('test:source'), observed_at=None))
    question = Question(Frame('follows', {'subject': Entity('noun', 'second', ref=second)}), 'object')
    from store_query_fixtures import taught_lookup
    from tensorcode.records import Proposition, Var
    assert isinstance(taught_lookup(agent, question, Proposition('follows', {'subject': second, 'object': Var('answer')})), Unknown)


def test_informing_capability_requires_declared_direction_and_answer():
    from tensorcode.agent import Capability, Informs, Param, Plugin
    from tensorcode.outcomes import Receipt
    from tensorcode.records import Proposition, Var

    owner, answer = Ref('entity:owner'), Ref('entity:answer')

    class Reports(Plugin):
        def __init__(self, *, reverse=False, declared=True):
            super().__init__('reports')
            self.reverse, self.declared = reverse, declared
            self.calls = 0
        def capabilities(self):
            query = Proposition('follows', {'subject': Var('owner'), 'object': Var('answer')})
            return (Capability('report', (Param('owner', 'entity'),),
                               informs=(Informs('follows', 'undergoer', 'owner',
                                                query=query if self.declared else None),),
                               effect_kind='read'),)
        def refer(self, description, param, *, context):
            return owner
        def execute(self, call, *, key):
            self.calls += 1
            return Receipt(call, 'applied')
        def reveal(self, cap, args, receipt):
            yield Claim(answer, 'follows', owner) if self.reverse else Claim(owner, 'follows', answer)

    frame = Frame('follows', {'subject': Entity('noun', 'owner', ref=owner)})
    act = Act('question', Question(frame, 'object'), frame)
    sentence = Sentence('supplied question', (), None, (act,))
    for reverse, declared, expected in ((False, True, 'answered'), (True, True, 'unknown'), (False, False, 'unknown')):
        plugin = Reports(reverse=reverse, declared=declared)
        agent = Agent([plugin])
        from informing_fixtures import teach_informing, selected_question_dependency
        from tensorcode.learning.informing import InformingPlan
        teach_informing(agent, act.meaning, InformingPlan(plugin.name, 'report', (('owner', owner),),
            Proposition('follows', {'subject': owner, 'object': Var('answer')}), 'answer'))
        outcome = agent.handle(sentence, act, [], requests_in_message=0,
            interpretation_dependency=selected_question_dependency(agent, act.meaning))
        assert outcome.status == expected
        if expected == 'answered':
            assert outcome.answer == [answer]
        else:
            assert not agent.store.claims()
        if not declared:
            assert plugin.calls == 0

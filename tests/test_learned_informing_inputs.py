"""Real syntax plus explicit teaching reaches a kind-restricted observation.

Syntax comes from local trained artifacts. Speech labels, occurrence identities,
measurement semantics, and candidate-selection policies are supplied explicitly.
Repeated question wording in distinct grounded contexts tests reference transfer,
not free paraphrase understanding or learned reference resolution.
"""
from pathlib import Path

import pytest

from tensorcode.agent import Agent, InterpretationDecision
from tensorcode.agent.understand import LearnedReader
from tensorcode.agent.grounding import MentionBinding, propose_grounding
from tensorcode.agent.quantity_plugin import QuantityPlugin
from tensorcode.language import Entity, Question
from tensorcode.language.deps_semantics import ProvisionalMeaning
from tensorcode.outcomes import Unknown
from tensorcode.quantity import Quantity, Unit
from tensorcode.records import Proposition, Ref, Var


@pytest.fixture(scope='module')
def actual_reader():
    root = Path.home() / '.cache/tensorcode/models'
    if not all((root / name).exists() for name in ('ud_ewt_parser.pickle', 'ud_ewt_segmenter.json')):
        pytest.skip('locally trained parsing and segmentation artifacts required')
    return LearnedReader(tag_beam_width=2, tag_max_candidates=2,
        parse_beam_width=4, parse_max_candidates=2, max_alternatives=8,
        segmentation_max_candidates=1)


def question_text(name='Shondra'):
    return f'how many plants does {name} have?'


def proper_question(frame, name):
    return (frame.predicate == 'have'
        and isinstance(frame.roles.get('subject'), Entity)
        and frame.roles['subject'].kind == 'name'
        and frame.roles['subject'].text == name
        and isinstance(frame.roles.get('object'), Entity)
        and frame.roles['object'].text == 'how many plants'
        and frame.roles['object'].features.get('noun') == 'plant')


def teach_actual_speech(agent):
    from tensorcode.agent.speech_act_learning import (
        retain_speech_act_example, fit_speech_act_model, admit_speech_act_model,
    )
    from tensorcode.learning.speech_act import SpeechActLabel
    records = []
    for name in ('Alex', 'Toni', 'Shondra'):
        message = agent.interpret(question_text(name))
        matches = []
        for gid in message.group_ids:
            group = agent.interpretations.get(gid)
            for candidate in group.candidates:
                for index, act in enumerate(candidate.payload.acts):
                    if type(act.meaning) is ProvisionalMeaning and proper_question(act.meaning.frame, name):
                        matches.append((group, candidate, index, act.meaning))
        assert len(matches) == 1
        group, candidate, index, meaning = matches[0]
        record = retain_speech_act_example(agent, group.id, candidate.id, index,
            SpeechActLabel('question', 'quantity', (), tuple(range(len(meaning.words)))),
            basis=('explicit question label over chosen actual trained syntax',))
        assert not isinstance(record, Unknown), record
        records.append(record)
    fitted = fit_speech_act_model(agent, records[:2], records[2:])
    assert not isinstance(fitted, Unknown), fitted
    admitted = admit_speech_act_model(agent, fitted, reason='explicit integration-test admission')
    assert not isinstance(admitted, Unknown), admitted
    agent.speech_act_model = admitted


def ground_question(agent, group, owner, kind):
    parents = [c for c in group.candidates if len(c.payload.acts) == 1
        and isinstance(c.payload.acts[0].meaning, Question)
        and proper_question(c.payload.acts[0].frame, 'Shondra')]
    assert len(parents) == 1
    evidence = agent.interpretations.add_source(
        'Explicit fixture identities for this context', provider='test occurrence binding',
        payload={'owner': owner, 'kind': kind})
    return propose_grounding(agent.interpretations, group.id, parents[0].id, (
        MentionBinding(('acts', 0, 'frame', 'roles', 'subject'), owner,
            (evidence.id,), 'supplied owner identity'),
        MentionBinding(('acts', 0, 'frame', 'roles', 'object'), kind,
            (evidence.id,), 'supplied counted-kind identity'),
    ))


def test_real_question_retains_counted_kind_through_learned_observation(actual_reader):
    from tensorcode.agent.informing_learning import (
        retain_informing_example, fit_informing_model, admit_informing_model,
    )
    from tensorcode.learning.informing import InformingPlan

    class RecordedQuantity(QuantityPlugin):
        def __init__(self):
            super().__init__()
            self.calls = []

        def execute(self, call, *, key=None):
            self.calls.append(call)
            return super().execute(call, key=key)

    plugin = RecordedQuantity()
    agent = Agent([plugin], reader=actual_reader)
    teach_actual_speech(agent)
    plant, coin = Ref('kind:plant'), Ref('kind:coin')
    records = []
    for context in ('Alpha', 'Beta', 'Gamma'):
        owner = Ref('owner:' + context)
        plugin.remember(owner, 'have', Quantity(3, Unit.of('plant')), kind=plant)
        message = agent.interpret(context + '. ' + question_text())
        groups = [agent.interpretations.get(gid) for gid in message.group_ids]
        matching = [g for g in groups if any(
            isinstance(a.meaning, Question) and proper_question(a.frame, 'Shondra')
            for c in g.candidates for a in c.payload.acts)]
        assert len(matching) == 1
        group = matching[0]
        child = ground_question(agent, group, owner, plant)
        plan = InformingPlan('quantity', 'amount_of_kind_have',
            (('owner', owner), ('kind', plant)),
            Proposition('total_kind:have', {'subject': owner, 'kind': plant, 'object': Var('answer')}),
            'answer')
        record = retain_informing_example(agent, group.id, child.id, 0, plan,
            basis=('explicit measurement correspondence; all question qualifiers retained',))
        assert not isinstance(record, Unknown), record
        records.append(record)
    model = fit_informing_model(agent, records[:2], records[2:])
    assert not isinstance(model, Unknown), model
    model = admit_informing_model(agent, model, reason='explicit retained measurement teaching')
    assert not isinstance(model, Unknown), model
    agent.informing_model = model
    fresh_owner = Ref('owner:heldout-execution')
    plugin.remember(fresh_owner, 'have', Quantity(7, Unit.of('plant')), kind=plant)
    plugin.remember(fresh_owner, 'have', Quantity(19, Unit.of('coin')), kind=coin)

    def select_reading(group):
        child = ground_question(agent, group, fresh_owner, plant)
        current = agent.interpretations.get(group.id)
        return InterpretationDecision(child.id, 'explicit selected grounded question',
            compared_revision=current.revision,
            compared_candidate_ids=tuple(c.id for c in current.candidates))

    def select_informing(group):
        assert len(group.candidates) == 1
        return InterpretationDecision(group.candidates[0].id, 'explicit measurement-plan selection',
            compared_revision=group.revision,
            compared_candidate_ids=tuple(c.id for c in group.candidates))

    agent.interpretation_selector = select_reading
    unselected = agent.turn(question_text())
    assert all(o.status == 'unknown' and o.receipt is None for o in unselected.outcomes)
    assert not plugin.calls
    agent.informing_selector = select_informing
    turn = agent.turn(question_text())
    assert len(turn.outcomes) == 1
    outcome = turn.outcomes[0]
    assert outcome.status == 'answered', (outcome.reason, outcome.verified, outcome.receipt)
    assert outcome.answer == [Quantity(7, Unit.of('plant'))]
    assert outcome.receipt.status == 'applied'
    assert outcome.act.frame.roles['object'].text == 'how many plants'
    assert outcome.act.frame.roles['object'].features['noun'] == 'plant'
    assert outcome.act.frame.roles['object'].ref == plant
    assert outcome.act.frame.roles['subject'].ref == fresh_owner
    assert dict(outcome.plan.plan.args) == {'owner': fresh_owner, 'kind': plant}
    assert len(plugin.calls) == 1
    from tensorcode.derivations import validate_record_support
    derived, = agent.store.propositions('total_kind:have')
    assert derived.proposition.roles == {'subject': fresh_owner, 'kind': plant, 'object': Quantity(7, Unit.of('plant'))}
    assert derived.evidence[0].method == 'authenticated-derivation-import'
    assert derived.evidence[0].derived_from
    assert validate_record_support(agent.store, derived.id) is True
    assert not agent.store.propositions('have')
    assert model.dependency in outcome.plan.dependencies
    agent.interpretations.unset(model.group_id, reason='withdraw measurement model')
    withdrawn = agent.turn(question_text())
    assert all(o.status == 'unknown' and o.receipt is None for o in withdrawn.outcomes)
    assert len(plugin.calls) == 1

import asyncio
import math

import pytest

from tensorcode.ops import text

from test_http_providers import server_factory  # noqa: F401  (fixture)
from test_owned_operations import foundation  # noqa: F401  (fixture)

VALUE = (text.Message('user', 'question'),)

LIKELIHOOD_CASES = [
    (text.Classify, {'labels': ['question', 'answer'], 'descriptions': {'answer': 'answer'}},
     {'label': 'answer', 'distribution': None, 'confidence': None, 'abstained': False}),
    (text.Decide, {'options': ['question', 'answer']},
     {'choice': 'answer', 'distribution': {'question': .25, 'answer': .75}, 'confidence': None, 'abstained': False}),
    (text.Score, {'rubric': ['question', 'answer']},
     {'score': 1, 'distribution': None, 'confidence': None, 'abstained': False}),
    (text.Retrieve, {'items': {'a': 'answer', 'b': 'question'}, 'limit': 1},
     {'keys': ['b'], 'scores': None, 'abstained': False}),
]


@pytest.mark.parametrize('cls,options,target', LIKELIHOOD_CASES)
def test_likelihood_decoding_scores_alternatives_in_one_encoder_pass(foundation, tmp_path, cls, options, target):
    import torch
    from tensorcode.training import Trainer

    op = cls.from_foundation(foundation, config={**options, 'decoding': 'likelihood'})
    calls = []
    hook = op.model.model.get_encoder().register_forward_pre_hook(lambda *args: calls.append(1))
    result = op(VALUE)
    hook.remove()
    assert calls == [1]
    if cls is text.Retrieve:
        assert set(result.scores) == set(options['items']) and len(result.keys) == 1
        assert result.keys[0] == max(result.scores, key=result.scores.get)
        assert result.distribution is None
    else:
        assert math.isclose(sum(result.distribution.values()), 1.0, abs_tol=1e-6)
        assert result.confidence == max(result.distribution.values())
        assert not result.abstained
        if cls is text.Score:
            assert math.isclose(result.value, sum(k * p for k, p in result.distribution.items()))
        else:
            assert result.value == max(result.distribution, key=result.distribution.get)

    loss = op.loss(VALUE, target)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in op.parameters())

    op.save_pretrained(tmp_path / 'op')
    restored = cls.from_pretrained(tmp_path / 'op')
    assert restored.decoding == 'likelihood'
    assert restored(VALUE) == result
    torch.testing.assert_close(restored.loss(VALUE, target), loss)

    trainer = Trainer.from_tool(restored, lr=.5)
    session = trainer.capture(VALUE, target, source='authored-test')
    before = float(restored.loss(VALUE, target).detach())
    for _ in range(5):
        trainer.step(session)
    assert float(restored.loss(VALUE, target).detach()) < before


def test_likelihood_configuration_is_strict(foundation):
    with pytest.raises(ValueError, match='decoding'):
        text.Classify.from_foundation(foundation, config={'labels': ['a'], 'decoding': 'beam'})
    with pytest.raises(ValueError, match='requires'):
        text.Classify.from_foundation(foundation, config={'labels': ['a'], 'likelihood_normalization': 'mean'})
    with pytest.raises(ValueError, match='descriptions'):
        text.Classify.from_foundation(foundation, config={'labels': ['a'], 'descriptions': {'b': 'x'}})
    with pytest.raises(ValueError, match='Unknown'):
        text.Classify.from_model(lambda request: None, labels=['a'], decoding='likelihood')
    with pytest.raises(ValueError, match='foundation config'):
        text.Transform.from_foundation(foundation, config={'decoding': 'likelihood'})
    same = text.Classify.from_foundation(foundation, config={'labels': ['yes', 'no'], 'decoding': 'likelihood'})
    with pytest.raises(ValueError, match='distinct'):
        same(VALUE)
    op = text.Classify.from_foundation(foundation, config={'labels': ['yes', 'no'], 'decoding': 'likelihood',
                                                           'likelihood_normalization': 'mean'})
    with pytest.raises(ValueError, match='abstention'):
        op.loss(VALUE, {'label': None, 'distribution': None, 'confidence': None, 'abstained': True})
    assert op.configuration()['likelihood_normalization'] == 'mean'


def test_default_generation_mode_is_unchanged(foundation):
    op = text.Classify.from_foundation(foundation, config={'labels': ['yes', 'no']})
    assert op.decoding == 'generate'
    assert 'decoding' not in op.configuration()


class QuestionProvider:
    def __init__(self):
        self.calls = []

    def _answer(self, request):
        if request.schema_name == 'tensorcode.score':
            return {'score': 1, 'distribution': None, 'confidence': None, 'abstained': False}
        return {'label': 'b', 'distribution': None, 'confidence': .5, 'abstained': False}

    def complete(self, request):
        self.calls.append(('complete', request.instructions))
        return text.ModelOutput(structured=self._answer(request))

    def complete_questions(self, requests):
        self.calls.append(('questions', tuple(requests)))
        return {name: text.ModelOutput(structured=self._answer(request)) for name, request in requests.items()}

    async def acomplete_questions(self, requests):
        return self.complete_questions(requests)


def test_ask_fuses_questions_for_one_shared_provider():
    provider = QuestionProvider()
    questions = {
        'topic': text.Classify.from_model(provider, labels=('a', 'b'), instructions='topic'),
        'urgency': text.Score.from_model(provider, rubric=('low', 'high'), instructions='urgency'),
    }
    answers = text.ask(VALUE, questions)
    assert provider.calls == [('questions', ('topic', 'urgency'))]
    assert answers['topic'].label == 'b' and answers['urgency'].value == 1
    with pytest.raises(TypeError):
        answers['topic'] = None
    assert asyncio.run(text.aask(VALUE, questions))['urgency'].value == 1


def test_ask_calls_each_operation_when_fusion_is_unavailable():
    import tensorcode

    provider = QuestionProvider()
    other = QuestionProvider()
    questions = {
        'first': text.Classify.from_model(provider, labels=('a', 'b'), instructions='first'),
        'second': text.Classify.from_model(other, labels=('a', 'b'), instructions='second'),
    }
    assert text.ask(VALUE, questions)['second'].label == 'b'
    assert provider.calls == [('complete', 'first')] and other.calls == [('complete', 'second')]

    shared = {'x': text.Classify.from_model(provider, labels=('a', 'b')),
              'y': text.Classify.from_model(provider, labels=('a', 'b'))}
    provider.calls.clear()
    with tensorcode.trace() as session:
        text.ask(VALUE, shared)
    assert [kind for kind, _ in provider.calls] == ['complete', 'complete']
    assert len(session.calls) == 2


def test_ask_validates_questions_and_answers():
    provider = QuestionProvider()
    with pytest.raises(TypeError, match='nonempty'):
        text.ask(VALUE, {})
    with pytest.raises(TypeError, match='Classify'):
        text.ask(VALUE, {'t': text.Transform.from_model(lambda messages: 'x')})

    class Wrong(QuestionProvider):
        def complete_questions(self, requests):
            return {'other': text.ModelOutput(structured={})}

    wrong = Wrong()
    with pytest.raises(text.InvalidModelOutput, match='names'):
        text.ask(VALUE, {'a': text.Classify.from_model(wrong, labels=('a', 'b'))})


def test_jev_fuses_questions_maps_noul_and_descriptions(server_factory):  # noqa: F811
    from tensorcode.integrations import JevModel, ProviderProtocolError

    response = {
        'model': 'jev-1.13.0',
        'answers': {
            'spam': {'type': 'noul', 'noul': .2},
            'route': {'type': 'choice', 'choice': 'billing', 'confidence': .6,
                      'probabilities': {'billing': .8, 'technical': .2}},
        },
        'usage': {'input_tokens': 12, 'output_tokens': 0},
    }
    server = server_factory(lambda request: (200, response, 0))
    model = JevModel(base_url=server.url, api_key='key')
    questions = {
        'spam': text.Classify.from_model(model, labels=('true', 'false'), instructions='Is this spam?',
                                         descriptions={'true': 'unsolicited advertising'}),
        'route': text.Decide.from_model(model, options=('billing', 'technical'), instructions='Route',
                                        descriptions={'billing': 'payments and charges'}),
    }
    answers = text.ask(VALUE, questions)
    assert len(server.requests) == 1
    sent = server.requests[0]['json']
    assert sent['state'] == [{'role': 'user', 'content': 'question'}]
    assert sent['questions'] == {
        'spam': {'type': 'noul', 'instructions': 'Is this spam?',
                 'criteria': {'true': 'unsolicited advertising', 'false': None}},
        'route': {'type': 'choice', 'instructions': 'Route',
                  'criteria': {'billing': 'payments and charges', 'technical': None}},
    }
    assert answers['spam'].label == 'false'
    assert answers['spam'].distribution == {'true': .2, 'false': .8}
    assert answers['spam'].confidence is None
    assert answers['route'].choice == 'billing' and answers['route'].confidence == .6

    other = (text.Message('user', 'different'),)
    requests = {name: op._request(VALUE if name == 'spam' else other, None) for name, op in questions.items()}
    with pytest.raises(ProviderProtocolError, match='same messages'):
        model.complete_questions(requests)
    assert len(server.requests) == 1

    single = server_factory(lambda request: (200, {'answers': {'result': {'type': 'noul', 'noul': 1.5}}}, 0))
    with pytest.raises(ProviderProtocolError, match='probability'):
        text.Classify.from_model(JevModel(base_url=single.url, api_key='key'),
                                 labels=('true', 'false'))(VALUE)

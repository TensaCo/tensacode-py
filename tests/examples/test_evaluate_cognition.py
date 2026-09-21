"""Authored records validate accounting only; no cognitive capability claim."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('evaluate_cognition', Path(__file__).parents[2] / 'examples/evaluate_cognition.py')
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_hotpot_target_is_not_injected_into_evidence():
    case = example.prepare_hotpot({'id': 'q', 'question': 'question', 'answer': 'SECRET',
        'supporting_facts': {'title': ['included']}, 'context': {'title': ['included', 'excluded'], 'sentences': [['source text'], ['distractor']]}})
    assert case['target'] == 'SECRET'
    assert len(case['evidence']) == 1
    assert case['evidence'][0]['text'] == 'included\nsource text'
    assert 'SECRET' not in str(case['evidence'])


def test_abstention_is_not_counted_as_fidelity_or_answer_accuracy():
    receipt = {'cognition': {'candidates': [], 'selected_id': None, 'abstained': True, 'policy': {'max_contradiction': .2}}}
    result = example.measure({'id': 'q', 'question': 'Where?', 'target': 'Paris'}, 'Insufficient evidence.', receipt)
    assert result['abstained']
    assert not result['selected_text_preserved']
    assert not result['answer_exact_match']
    assert not result['candidate_answer_substring_coverage']


def test_lexical_coverage_and_contradiction_are_explicit_diagnostics():
    receipt = {'cognition': {'candidates': [{'id': 'h', 'text': 'The answer is Paris.', 'verifications': [{'distribution': {'contradiction': .8}}]}],
               'selected_id': 'h', 'abstained': False, 'policy': {'max_contradiction': .2}}}
    result = example.measure({'id': 'q', 'question': 'Where?', 'target': 'Paris'}, 'The answer is Paris.', receipt)
    assert result['candidate_answer_substring_coverage']
    assert result['selected_text_preserved']
    assert result['contradicted_selection']
    assert not result['answer_exact_match']
    assert example.summarize([result])['selected_contradiction_veto_violations'] == 1


def test_model_failures_are_counted_without_becoming_abstentions():
    class Session:
        def __call__(self, value):
            raise ValueError('generator produced no hypotheses')
    class Bot:
        def new_session(self):
            return Session()
    report = example.evaluate(Bot(), [{'id': 'q', 'question': 'Where?', 'target': 'Paris', 'evidence': []}])
    metrics = report['real_data']['metrics']
    assert metrics['count'] == 1
    assert metrics['failed_calls'] == 1
    assert metrics['cognitive_abstention_rate'] == 0
    assert metrics['short_answer_exact_match_format_sensitive'] == 0


def test_fixed_control_subset_does_not_change_primary_denominator():
    calls = []
    class Session:
        def __call__(self, value):
            calls.append(value)
            raise ValueError('fixture model failure')
    class Bot:
        def new_session(self):
            return Session()
    cases = [{'id': str(i), 'question': 'Where?', 'target': 'Paris', 'evidence': []} for i in range(3)]
    report = example.evaluate(Bot(), cases, control_count=1)
    assert report['real_data']['metrics']['count'] == 3
    assert report['controls']['omission']['count'] == 1
    assert len(calls) == 5

"""Authored tiny fixtures isolate ownership and transactional cognitive plumbing."""
import copy
import json

import pytest
import torch

from tensorcode.tools.chatbot import Chatbot
from test_chatbot_model import tiny_config
from test_investigation import config as investigator_config


def config():
    result = tiny_config()
    result['max_input_tokens'] = 256
    result['cognition'] = {'investigator': investigator_config(), 'proposal_count': 1}
    return result


def prepared(monkeypatch, *, memory=False):
    settings = config()
    if memory:
        settings['cognition']['memory'] = {'capacity': 8, 'top_k': 2}
    model = Chatbot(settings).eval()
    def propose(inputs, **kwargs):
        return [{'id': 'h1', 'text': 'hello', 'origin': 'generated',
                 'generated_by': 'authored-test-output'}]
    monkeypatch.setattr(model.investigator, 'propose', propose)
    with torch.no_grad():
        model.investigator.verifier.model.classifier.weight.zero_()
        model.investigator.verifier.model.classifier.bias.copy_(torch.tensor([-5., -5., 5.]))
    monkeypatch.setattr(model, 'generate_batch', lambda inputs: ['realized answer'] * len(inputs))
    return model


INPUT = {'question': 'hello?', 'evidence': [{'id': 'e1', 'source_id': 'source-one', 'text': 'hello world'}]}


def test_complete_owned_cognitive_parameters_and_local_roundtrip(tmp_path):
    model = Chatbot(config()).eval()
    assert model.investigator.generator is not None and model.investigator.verifier is not None
    ids = {id(p) for p in model.parameters()}
    assert all(id(p) in ids for p in model.investigator.parameters())
    assert model.capabilities['persistent_cognitive_state']
    assert not Chatbot(tiny_config()).capabilities['source_verification']
    model.save_pretrained(tmp_path / 'model')
    restored = Chatbot.from_pretrained(tmp_path / 'model')
    for key, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, restored.state_dict()[key])
    assert restored.configuration() == model.configuration()
    assert any(key.startswith('investigator.') for key in model.operation_bindings())
    assert model.investigator.generator.investigator is None


def test_recursive_or_incomplete_config_is_rejected():
    recursive = config()
    recursive['cognition']['investigator']['generator']['cognition'] = config()['cognition']
    with pytest.raises(ValueError, match='Recursive'):
        Chatbot(recursive)
    incomplete = config()
    del incomplete['cognition']['investigator']['generator']
    with pytest.raises(ValueError, match='owned proposal'):
        Chatbot(incomplete)


def test_sources_state_and_session_are_independent(monkeypatch, tmp_path):
    model = prepared(monkeypatch)
    first, second = model.new_session(), model.new_session()
    assert first(copy.deepcopy(INPUT)) == 'realized answer'
    snapshot = first.cognition.snapshot()
    assert 'realized answer' not in json.dumps(snapshot)
    assert 'hello?' not in [row['text'] for row in snapshot['state']['evidence']]
    assert second.cognition.snapshot() != snapshot
    path = tmp_path / 'session.json'
    first.save(path)
    second.load(path)
    assert second.cognition.snapshot() == snapshot
    assert second.history == first.history


def test_question_is_not_evidence_and_abstention_is_authored(monkeypatch):
    model = prepared(monkeypatch)
    output = model('hello?')
    assert output == 'I do not have enough supported evidence to answer.'
    assert model.last_result['abstention_enforced']
    assert not model._session.cognition.snapshot()['state']['evidence']


def test_failed_decoder_does_not_commit_evidence_or_interpretations(monkeypatch):
    model = prepared(monkeypatch)
    before = model._session.cognition.snapshot()
    def fail(inputs):
        raise RuntimeError('decoder failed')
    monkeypatch.setattr(model, 'generate_batch', fail)
    with pytest.raises(RuntimeError, match='decoder failed'):
        model(copy.deepcopy(INPUT))
    assert model._session.cognition.snapshot() == before
    assert model.history == ()


def test_generation_receives_structured_interpretations_and_source_evidence(monkeypatch):
    model = prepared(monkeypatch)
    prompts = []
    def generate(inputs):
        prompts.extend(inputs)
        return ['answer']
    monkeypatch.setattr(model, 'generate_batch', generate)
    model(copy.deepcopy(INPUT))
    assert 'source-one' in prompts[0]
    assert 'Selected hypothesis' in prompts[0]
    assert 'hello world' in prompts[0]


def test_source_revision_retains_original_and_never_ingests_assistant(monkeypatch):
    model = prepared(monkeypatch)
    model(copy.deepcopy(INPUT))
    model({'question': 'hello?', 'revisions': [{'evidence_id': 'e1', 'text': 'world changed',
                                               'source_id': 'corrected-source'}]})
    snapshot = model._session.cognition.snapshot()
    texts = [row['text'] for row in snapshot['state']['evidence']]
    assert 'hello world' in texts and 'world changed' in texts
    assert 'realized answer' not in texts
    assert snapshot['active_evidence']['e1'] != 'e1'


def test_invalid_cognitive_session_load_is_transactional(monkeypatch, tmp_path):
    model = prepared(monkeypatch)
    model(copy.deepcopy(INPUT))
    before = model._session.cognition.snapshot()
    history = model.history
    path = tmp_path / 'session.json'
    model.save_session(path)
    value = json.loads(path.read_text())
    value['cognition']['active_evidence']['e1'] = 'missing'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        model.load_session(path)
    assert model._session.cognition.snapshot() == before
    assert model.history == history


def test_language_and_investigator_objectives_own_distinct_supervision():
    model = Chatbot(config())
    model.loss_batch(['hello'], ['world']).backward()
    assert any(p.grad is not None for p in model.foundation.parameters())
    assert all(p.grad is None for p in model.investigator.parameters())
    model.investigator.verification_loss([{'premise': 'hello', 'hypothesis': 'world'}], ['unknown']).backward()
    assert any(p.grad is not None for p in model.investigator.verifier.parameters())


def test_unsupported_realization_enforces_abstention_after_valid_selection(monkeypatch):
    model = prepared(monkeypatch)
    original = model.investigator.verifier.verify
    def verify(text, evidence):
        if text == 'realized answer':
            return [{'source_id': row['source_id'], 'distribution':
                     {'support': .01, 'contradiction': .98, 'unknown': .01}} for row in evidence]
        return original(text, evidence)
    monkeypatch.setattr(model.investigator.verifier, 'verify', verify)
    answer = model(copy.deepcopy(INPUT))
    assert not model.last_result['cognition']['abstained']
    assert model.last_result['abstention_enforced']
    assert answer == 'I do not have enough supported evidence to answer.'
    assert model.last_result['response_proposal'] == {
        'text': 'realized answer', 'origin': 'model_generation',
        'epistemic_status': 'unverified_proposal'}
    assert all(row['text'] != 'realized answer' for row in model.history)
    assert all(row.text != 'realized answer' for row in model.cognitive_state.evidence)
    assert model.cognitive_state is not None


def test_realization_budget_prioritizes_sources_and_reports_omissions(monkeypatch):
    model = prepared(monkeypatch)
    model.config['max_input_tokens'] = 64
    receipt = {'selected_id': 'h1', 'candidates': [{'id': 'h1', 'text': 'hello',
                'verifications': [{'evidence_id': 'e1', 'distribution': {'support': .99}}]}],
               'evidence': [{'id': 'e1', 'source_id': 'one', 'text': 'hello ' * 1000},
                            {'id': 'e2', 'source_id': 'two', 'text': 'world ' * 1000}]}
    prompt, visible, omitted = model._realization_input('hello?', receipt)
    assert len(model.tokenizer(prompt)['input_ids']) <= 64
    assert visible and visible[0]['id'] == 'e1'
    assert len(omitted) == 2
    assert 'distribution' not in prompt and 'verifications' not in prompt


def test_truncated_realization_verification_cannot_release_answer(monkeypatch):
    model = prepared(monkeypatch)
    original = model.investigator.verifier.verify
    def verify(text, evidence):
        result = original(text, evidence)
        if text == 'realized answer':
            for row in result:
                row['input_truncated'] = True
        return result
    monkeypatch.setattr(model.investigator.verifier, 'verify', verify)
    model(copy.deepcopy(INPUT))
    assert model.last_result['abstention_enforced']
    assert any(row['input_truncated'] for row in model.last_result['realization_verifications'])


def test_opaque_memory_retention_episode_retrieval_and_session_save(monkeypatch, tmp_path):
    model = prepared(monkeypatch, memory=True)
    model(copy.deepcopy(INPUT))
    assert model.last_result['retained_evidence_ids'] == ['e1']
    assert 'realized answer' not in json.dumps(model._session.cognition.snapshot()['memory'])
    model.new_episode()
    assert model.history == ()
    model('hello?')
    assert model.last_result['cognition']['retrieval']
    assert model.last_result['cognition']['evidence'][0]['source_id'] == 'source-one'
    path = tmp_path / 'session.json'
    model.save_session(path)
    session = model.new_session()
    session.load(path)
    assert session.cognition.snapshot() == model._session.cognition.snapshot()


def test_memory_is_not_committed_on_failed_decode(monkeypatch):
    model = prepared(monkeypatch, memory=True)
    before = model._session.cognition.snapshot()
    def fail(inputs):
        raise RuntimeError('decoder failed')
    monkeypatch.setattr(model, 'generate_batch', fail)
    with pytest.raises(RuntimeError):
        model(copy.deepcopy(INPUT))
    assert model._session.cognition.snapshot() == before


def test_memory_weights_change_requires_explicit_index_rebuild(monkeypatch):
    model = prepared(monkeypatch, memory=True)
    model(copy.deepcopy(INPUT))
    model.new_episode()
    with torch.no_grad():
        next(model.investigator.rank.encode.parameters()).add_(.1)
    with pytest.raises(ValueError, match='stale'):
        model('hello?')
    model.rebuild_memory()
    model('hello?')
    assert model.last_result['cognition']['retrieval']


@pytest.mark.parametrize('recall_before_revision', [False, True])
def test_cross_episode_source_correction_survives_session_reload(monkeypatch, tmp_path,
                                                               recall_before_revision):
    model = prepared(monkeypatch, memory=True)
    model(copy.deepcopy(INPUT))
    model.new_episode()
    if recall_before_revision:
        model('hello?')
    model({'question': 'hello?', 'revisions': [
        {'evidence_id': 'e1', 'text': 'corrected world', 'source_id': 'corrected-source'}]})
    receipt = model.last_result['cognition']
    assert [(row['text'], row['source_id']) for row in receipt['evidence']] == [
        ('corrected world', 'corrected-source')]
    assert {'hello world', 'corrected world'} <= {
        row.text for row in model.cognitive_state.evidence}
    model.new_episode()
    path = tmp_path / 'corrected-session.json'
    model.save_session(path)
    restored = model.new_session().load(path)
    restored('hello?')
    assert [row['text'] for row in restored.last_result['cognition']['evidence']] == ['corrected world']
    assert all(hit.evidence.text != 'hello world' for hit in restored.cognition.retrieve('hello?'))
    # The caller keeps the logical source ID through repeated corrections.
    restored({'question': 'hello?', 'revisions': [{'evidence_id': 'e1', 'text': 'latest world'}]})
    assert [row['text'] for row in restored.last_result['cognition']['evidence']] == ['latest world']


def test_failed_cross_episode_correction_does_not_commit_memory(monkeypatch):
    model = prepared(monkeypatch, memory=True)
    model(copy.deepcopy(INPUT))
    model.new_episode()
    before = model._session.cognition.snapshot()
    before_history = model.history
    def fail(inputs):
        raise RuntimeError('decoder failed after revision')
    monkeypatch.setattr(model, 'generate_batch', fail)
    with pytest.raises(RuntimeError, match='decoder failed after revision'):
        model({'question': 'hello?', 'revisions': [{'evidence_id': 'e1', 'text': 'corrected world'}]})
    assert model._session.cognition.snapshot() == before
    assert model.history == before_history


@pytest.mark.parametrize('field,value', [('text', 'conflicting world'), ('source_id', 'different-source')])
def test_conflicting_remembered_source_load_is_transactional(monkeypatch, tmp_path, field, value):
    model = prepared(monkeypatch, memory=True)
    model(copy.deepcopy(INPUT))
    path = tmp_path / 'inconsistent-session.json'
    model.save_session(path)
    payload = json.loads(path.read_text())
    payload['cognition']['memory']['records'][0]['evidence'][field] = value
    path.write_text(json.dumps(payload))
    before = model._session.cognition.snapshot()
    before_history = model.history
    with pytest.raises(ValueError, match='conflict'):
        model.load_session(path)
    assert model._session.cognition.snapshot() == before
    assert model.history == before_history


def test_repeated_source_is_idempotent_across_questions_and_episodes(monkeypatch):
    model = prepared(monkeypatch, memory=True)
    model(copy.deepcopy(INPUT))
    again = dict(copy.deepcopy(INPUT), question='world?')
    model(again)
    assert model.last_result['retained_evidence_ids'] == []
    model.new_episode()
    model(again)
    assert len(model._session.cognition.snapshot()['memory']['records']) == 1
    assert model.last_result['retained_evidence_ids'] == []


def test_empty_owned_generation_withdraws_selection_without_fake_hypothesis(monkeypatch):
    model = prepared(monkeypatch, memory=True)
    model(copy.deepcopy(INPUT))
    assert model.cognitive_state.selection
    # Restore the owned proposal pipeline; only its rendered output is an
    # authored empty fixture, representing a real observed generation failure.
    monkeypatch.delattr(model.investigator, 'propose')
    monkeypatch.setattr(model.investigator.generator.tokenizer, 'batch_decode',
                        lambda *args, **kwargs: [''])
    answer = model('world?')
    assert answer == 'I do not have enough supported evidence to answer.'
    assert model.last_result['abstention_enforced']
    assert model.last_result['cognition']['candidates'] == []
    assert model.cognitive_state.selection == ()
    assert [row.text for row in model.cognitive_state.evidence] == ['hello world']
    assert all(row.text.strip() for row in model.cognitive_state.hypotheses)
    assert len(model._session.cognition.snapshot()['memory']['records']) == 1


def test_pretrained_default_session_uses_loaded_memory_encoder(tmp_path, monkeypatch):
    settings = config()
    settings['cognition']['memory'] = {'capacity': 8, 'top_k': 2}
    model = Chatbot(settings).eval()
    model.save_pretrained(tmp_path / 'model')
    restored = Chatbot.from_pretrained(tmp_path / 'model')
    monkeypatch.setattr(restored.investigator, 'propose', lambda *args, **kwargs:
                        [{'id': 'h1', 'text': 'hello', 'origin': 'generated'}])
    monkeypatch.setattr(restored, 'generate_batch', lambda inputs: ['hello'])
    answer = restored(copy.deepcopy(INPUT))
    assert isinstance(answer, str)
    assert restored.last_result['retained_evidence_ids'] == ['e1']
    assert len(restored._session.cognition.snapshot()['memory']['records']) == 1
    history = restored.history
    restored.load_state_dict(Chatbot(settings).state_dict())
    with pytest.raises(ValueError, match='stale'):
        restored('world?')
    assert restored.history == history
    assert len(restored._session.cognition.snapshot()['memory']['records']) == 1


def test_joint_scope_screens_realization_and_persists_complete_model(monkeypatch, tmp_path):
    settings = config(); settings['cognition']['investigator']['verification_scope'] = 'joint'
    model = Chatbot(settings).eval()
    with torch.no_grad():
        model.investigator.verifier.model.classifier.weight.zero_()
        model.investigator.verifier.model.classifier.bias.copy_(torch.tensor([-5., -5., 5.]))
    model.save_pretrained(tmp_path / 'joint')
    model = Chatbot.from_pretrained(tmp_path / 'joint')
    assert model.configuration()['cognition']['investigator']['verification_scope'] == 'joint'
    monkeypatch.setattr(model.investigator, 'propose', lambda *a, **kw: [{'id': 'h', 'text': 'hello'}])
    monkeypatch.setattr(model, 'generate_batch', lambda inputs: ['realized answer'])
    original = model.investigator.verifier.verify_joint
    def joint(text, evidence):
        result = original(text, evidence)
        if text == 'realized answer':
            result['distribution'] = {'support': .01, 'contradiction': .01, 'unknown': .98}
        return result
    monkeypatch.setattr(model.investigator.verifier, 'verify_joint', joint)
    model(copy.deepcopy(INPUT))
    assert not model.last_result['cognition']['abstained']
    assert model.last_result['abstention_enforced']
    assert model.last_result['realization_joint_verification']['source_ids'] == ['e1']


def test_joint_realization_full_evidence_check_retains_omitted_conflict(monkeypatch):
    settings = config(); settings['cognition']['investigator']['verification_scope'] = 'joint'
    model = Chatbot(settings).eval()
    with torch.no_grad():
        model.investigator.verifier.model.classifier.weight.zero_()
        model.investigator.verifier.model.classifier.bias.copy_(torch.tensor([-5., -5., 5.]))
    monkeypatch.setattr(model.investigator, 'propose', lambda *a, **kw: [{'id': 'h', 'text': 'hello'}])
    monkeypatch.setattr(model, 'generate_batch', lambda inputs: ['realized answer'])
    def realization(question, interpretation):
        return 'hello', interpretation['evidence'][:1], [{'evidence_id': 'e2', 'included_characters': 0}]
    monkeypatch.setattr(model, '_realization_input', realization)
    original = model.investigator.verifier.verify
    def source_checks(text, evidence):
        checks = original(text, evidence)
        if text == 'realized answer':
            for row in checks:
                if row['source_id'] == 'e2':
                    row['distribution'] = {'support': .01, 'contradiction': .98, 'unknown': .01}
        return checks
    monkeypatch.setattr(model.investigator.verifier, 'verify', source_checks)
    value = copy.deepcopy(INPUT)
    value['evidence'].append({'id': 'e2', 'source_id': 'two', 'text': 'world'})
    model(value)
    assert not model.last_result['cognition']['abstained']
    assert model.last_result['abstention_enforced']
    assert model.last_result['realization_joint_verification']['source_ids'] == ['e1']
    assert model.last_result['full_realization_joint_verification']['source_ids'] == ['e1', 'e2']

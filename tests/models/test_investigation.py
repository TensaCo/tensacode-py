"""Tiny random models and explicitly authored outputs test mechanisms, not reasoning."""
import json

import pytest
import torch
from transformers import BertConfig

from tensorcode.tools.investigator import Investigator
from test_chatbot_model import tiny_config


def config():
    generator = tiny_config()
    return {'vocabulary': ['hello', 'world'], 'dimensions': 8, 'slots': 2, 'steps': 1,
            'generator': generator,
            'verifier_config': BertConfig(vocab_size=8, hidden_size=8, num_hidden_layers=1,
                num_attention_heads=2, intermediate_size=16, num_labels=3).to_dict(),
            'verifier_tokenizer_json': generator['tokenizer_json'],
            'verifier_tokenizer_special_tokens': generator['tokenizer_special_tokens'],
            'verifier_labels': {'support': 2, 'contradiction': 0, 'unknown': 1}}


INPUT = {'question': 'hello', 'evidence': [{'source_id': 'a', 'text': 'world'},
                                        {'source_id': 'b', 'text': 'hello'}]}


def test_proposals_owned_beams_provenance_dedup_and_no_leakage(monkeypatch):
    tool = Investigator(config()).train()
    tool.generator.workspace.eval()
    modes = [m.training for m in tool.generator.modules()]
    seen, contexts = [], []
    original = tool.generator.encode_workspace
    def encode(inputs):
        seen.extend(inputs)
        return original(inputs)
    monkeypatch.setattr(tool.generator, 'encode_workspace', encode)
    def decode(state, *, context):
        contexts.append(context)
        return torch.tensor([[5], [5], [6]])
    monkeypatch.setattr(tool.generator.decoder, 'forward', decode)
    outputs = tool.propose(dict(INPUT, hypotheses=[{'id': 'secret', 'text': 'LEAK'}], targets='SECRET'))
    assert len(outputs) == 2
    assert contexts[0]['num_beams'] == 3
    assert 'LEAK' not in seen[0] and 'SECRET' not in seen[0]
    assert outputs[0]['origin'] == 'generated' and outputs[0]['epistemic_status'] == 'hypothesis'
    assert outputs[0]['source_ids'] == ['a', 'b']
    assert outputs[0]['source_reference_kind'] == 'generation_context'
    assert outputs[0]['generator_configuration_fingerprint'] == tool.generator.fingerprint
    assert outputs[0]['proposal_template_version'] == 1
    assert outputs[0]['generated_by'] != tool.generator.fingerprint
    assert seen[0].startswith('Generate one declarative candidate explanation')
    assert modes == [m.training for m in tool.generator.modules()]
    for count in (0, -1, True, 17):
        with pytest.raises(ValueError, match='count'):
            tool.propose(INPUT, count=count)


def test_missing_capability_and_empty_or_malformed_outputs(monkeypatch):
    with pytest.raises(ValueError, match='not configured'):
        Investigator({'vocabulary': ['hello']}).propose(INPUT)
    tool = Investigator(config())
    monkeypatch.setattr(tool.generator.tokenizer, 'batch_decode', lambda *a, **k: ['', ' ', ''])
    assert tool.propose(INPUT) == []
    receipt = tool(INPUT)
    assert receipt['abstained'] and receipt['reason'] == 'no_hypotheses_generated'
    assert receipt['selected_id'] is None and receipt['candidates'] == []
    assert receipt['evidence'] == INPUT['evidence']
    monkeypatch.setattr(tool.generator.tokenizer, 'batch_decode', lambda *a, **k: [None])
    with pytest.raises(ValueError, match='malformed'):
        tool.propose(INPUT)


def test_explicit_verifier_mapping_and_contradiction_retained():
    bad = config()
    del bad['verifier_labels']
    with pytest.raises(ValueError, match='explicitly map'):
        Investigator(bad)
    tool = Investigator(config()).eval()
    with torch.no_grad():
        tool.verifier.model.classifier.weight.zero_()
        tool.verifier.model.classifier.bias.copy_(torch.tensor([4., 1., -2.]))
    inputs = dict(INPUT, hypotheses=[{'id': 'h', 'text': 'hello'}])
    receipt = tool.investigate(inputs)
    checks = receipt['candidates'][0]['verifications']
    assert [x['source_id'] for x in checks] == ['a', 'b']
    assert all(x['distribution']['contradiction'] > .9 for x in checks)
    assert all(sum(x['distribution'].values()) == pytest.approx(1) for x in checks)
    assert all(x['origin'] == 'model_inference' for x in checks)
    assert 'verifications' not in tool(inputs)['candidates'][0]


def test_supervision_gradients_target_separation_and_complete_roundtrip(tmp_path, monkeypatch):
    tool = Investigator(config()).eval()
    seen = []
    original = tool.generator.encode_workspace
    def encode(inputs, **kwargs):
        seen.extend(inputs)
        return original(inputs, **kwargs)
    monkeypatch.setattr(tool.generator, 'encode_workspace', encode)
    loss = tool.proposal_loss(dict(INPUT, targets='secret'), 'answer')
    loss.backward()
    assert all('answer' not in prompt.split('\n', 1)[1] and 'secret' not in prompt for prompt in seen)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in tool.generator.workspace.parameters())
    pairs = [{'premise': 'world', 'hypothesis': 'hello'}]
    tool.verification_loss(pairs, ['contradiction']).backward()
    assert tool.verifier.model.classifier.weight.grad.abs().sum() > 0
    expected = tool.verifier(pairs).detach()
    tool.save_pretrained(tmp_path / 'model')
    loaded = Investigator.from_pretrained(tmp_path / 'model')
    assert torch.equal(loaded.verifier(pairs), expected)
    assert torch.equal(loaded.proposal_loss(INPUT, 'answer'), loss)
    assert 'generator.decoder' in loaded.operation_bindings()
    assert 'verifier' in loaded.operation_bindings()
    manifest = json.loads((tmp_path / 'model/tensorcode_config.json').read_text())
    assert 'generator' in manifest['config'] and 'verifier_config' in manifest['config']


def test_calibration_owned_state_and_generated_session_roundtrip(tmp_path, monkeypatch):
    from tensorcode._internal.ranking import RankingSession
    tool = Investigator(config()).eval()
    monkeypatch.setattr(tool.generator.tokenizer, 'batch_decode', lambda *a, **k: ['hello', 'world', 'hello'])
    session = tool.new_session()
    receipt = session(INPUT)
    assert not receipt['candidates'][0]['verifications'][0]['calibrated']
    session.save(tmp_path / 'session.json')
    restored = RankingSession.load(tmp_path / 'session.json', tool)
    assert restored.history == session.history
    assert restored.history[0]['inputs']['hypotheses'][0]['origin'] == 'generated'
    tool.verifier.calibration.fit(torch.tensor([[5., 0., 0.], [5., 0., 0.]]), torch.tensor([0, 1]))
    assert tool.investigate(dict(INPUT, hypotheses=[{'id': 'h', 'text': 'hello'}]))['candidates'][0]['verifications'][0]['calibrated']
    tool.save_pretrained(tmp_path / 'model')
    loaded = Investigator.from_pretrained(tmp_path / 'model')
    assert bool(loaded.verifier.calibration.calibrated)
    assert torch.equal(tool.verifier.calibration.temperature, loaded.verifier.calibration.temperature)


@pytest.mark.parametrize('mode', ['rank', 'proposal', 'verification'])
def test_training_modes_durable_replay_and_optimizer_checkpoint(tmp_path, mode):
    from tensorcode import training
    tool = Investigator(config())
    trainer = training.ToolTrainer(tool)
    if mode == 'rank':
        inputs = dict(INPUT, hypotheses=[{'id': 'a', 'text': 'hello'}, {'id': 'b', 'text': 'world'}])
        targets = 'a'
    elif mode == 'proposal':
        inputs, targets = INPUT, 'answer'
    else:
        inputs, targets = [{'premise': 'world', 'hypothesis': 'hello'}], ['support']
    experience = trainer.capture({'mode': mode, 'inputs': inputs}, targets, source='authored-mechanism-fixture')
    assert trainer.step(experience) >= 0
    experience.save(tmp_path / 'experience.json', operations=trainer.operations)
    tool.save_pretrained(tmp_path / 'model')
    trainer.save_checkpoint(tmp_path / 'resume')
    restored = training.ToolTrainer(Investigator.from_pretrained(tmp_path / 'model'))
    restored.load_checkpoint(tmp_path / 'resume')
    loaded = training.load(tmp_path / 'experience.json', operations=restored.operations)
    assert restored.step(loaded) >= 0


def test_calibration_invalidated_by_supervision_and_weight_mutation():
    tool = Investigator(config())
    verifier = tool.verifier
    logits, labels = torch.tensor([[5., 0., 0.], [5., 0., 0.]]), torch.tensor([0, 1])
    verifier.calibration.fit(logits, labels)
    with torch.no_grad():
        verifier.model.classifier.bias.add_(1.)
    assert not verifier.verify('hello', INPUT['evidence'])[0]['calibrated']
    verifier.calibration.fit(logits, labels)
    tool.verification_loss([{'premise': 'world', 'hypothesis': 'hello'}], ['support'])
    assert not verifier.calibration.calibrated


@pytest.mark.parametrize('corruption', ['origin', 'source_ids', 'generated_by', 'verification_source',
                                      'distribution', 'nan', 'calibrated', 'sample_count', 'missing_checks', 'model'])
def test_generated_session_rejects_contradictory_provenance(tmp_path, monkeypatch, corruption):
    from tensorcode._internal.ranking import RankingSession
    tool = Investigator(config()).eval()
    monkeypatch.setattr(tool.generator.tokenizer, 'batch_decode', lambda *a, **k: ['hello'])
    session = tool.new_session()
    session(INPUT)
    path = tmp_path / 'session.json'
    session.save(path)
    payload = json.loads(path.read_text())
    candidate = payload['history'][0]['receipt']['candidates'][0]
    if corruption in ('origin', 'source_ids', 'generated_by'):
        candidate[corruption] = 'corrupted'
    elif corruption == 'missing_checks':
        del candidate['verifications']
    else:
        check = candidate['verifications'][0]
        if corruption == 'verification_source':
            check['source_id'] = 'missing-source'
        elif corruption == 'distribution':
            check['distribution'] = {'support': .2, 'contradiction': .2, 'unknown': .2}
        elif corruption == 'nan':
            check['distribution']['support'] = float('nan')
        elif corruption == 'calibrated':
            check['calibrated'] = True
        elif corruption == 'sample_count':
            check['calibration_sample_count'] = -1
        elif corruption == 'model':
            check['model'] = {'repository': 'wrong'}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='session|provenance'):
        RankingSession.load(path, tool)


def test_empty_generation_session_persists_abstention(tmp_path, monkeypatch):
    from tensorcode._internal.ranking import RankingSession
    tool = Investigator(config()).eval()
    monkeypatch.setattr(tool.generator.tokenizer, 'batch_decode', lambda *a, **k: ['', '  '])
    monkeypatch.setattr(tool.rank, 'receipt', lambda *a, **k: pytest.fail('empty generation must not rank'))
    session = tool.new_session()
    receipt = session(INPUT)
    assert receipt['abstained'] and receipt['candidates'] == []
    session.save(tmp_path / 'session.json')
    restored = RankingSession.load(tmp_path / 'session.json', tool)
    assert restored.history == session.history


def test_joint_session_roundtrip_and_source_order_tampering(tmp_path, monkeypatch):
    from tensorcode._internal.ranking import RankingSession
    settings = config(); settings['verification_scope'] = 'joint'
    tool = Investigator(settings).eval()
    monkeypatch.setattr(tool.generator.tokenizer, 'batch_decode', lambda *a, **k: ['hello'])
    session = tool.new_session(); session(INPUT)
    path = tmp_path / 'joint.json'; session.save(path)
    restored = RankingSession.load(path, tool)
    assert restored.history == session.history
    assert 'joint_verification' not in restored.history[0]['inputs']['hypotheses'][0]
    original = json.loads(path.read_text())
    for corruption in ('order', 'missing', 'scope', 'distribution', 'truncation', 'model', 'budget-missing', 'budget-bool', 'count-conflict'):
        payload = json.loads(json.dumps(original))
        candidate = payload['history'][0]['receipt']['candidates'][0]
        joint = candidate['joint_verification']
        if corruption == 'order': joint['source_ids'].reverse()
        elif corruption == 'missing': candidate.pop('joint_verification')
        elif corruption == 'scope': joint['scope'] = 'source'
        elif corruption == 'distribution': joint['distribution']['support'] = 9
        elif corruption == 'truncation': joint['input_truncated'] = 'false'
        elif corruption == 'budget-missing': joint.pop('max_tokens', None)
        elif corruption == 'budget-bool': joint['max_tokens'] = True
        elif corruption == 'count-conflict': joint['token_count'] = 3000
        else: joint['model'] = {'wrong': 'model'}
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match='joint|verification|provenance'):
            RankingSession.load(path, tool)


def test_question_prompt_version_is_owned_for_training_generation_and_reload(tmp_path, monkeypatch):
    from tensorcode._internal.proposals import proposal_prompt
    settings = config()
    settings['proposal_template_version'] = 2
    tool = Investigator(settings).eval()
    expected = proposal_prompt(INPUT, 'question', template_version=2)
    assert expected.startswith('Answer the question using only the supplied evidence.')
    assert json.loads(expected.split('\n', 1)[1]) == INPUT
    seen = []
    original = tool.generator.encode_workspace
    def encode(inputs, **kwargs):
        seen.extend(inputs)
        return original(inputs, **kwargs)
    monkeypatch.setattr(tool.generator, 'encode_workspace', encode)
    monkeypatch.setattr(tool.generator.tokenizer, 'batch_decode', lambda *a, **k: ['hello'])
    proposals = tool.propose(dict(INPUT, targets='SECRET'), count=1)
    loss = tool.proposal_loss(dict(INPUT, targets='SECRET'), 'answer')
    assert seen == [expected, expected]
    assert proposals[0]['proposal_template_version'] == 2
    assert 'SECRET' not in expected
    tool.save_pretrained(tmp_path / 'v2')
    restored = Investigator.from_pretrained(tmp_path / 'v2')
    assert restored.configuration()['proposal_template_version'] == 2
    assert torch.equal(restored.proposal_loss(INPUT, 'answer'), loss)
    for version in (0, 3, True, '2'):
        with pytest.raises(ValueError, match='template'):
            Investigator(dict(settings, proposal_template_version=version))

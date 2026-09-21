import json

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import T5Config

from tensorcode.tools.chatbot import Chatbot


def tiny_config():
    tokenizer = Tokenizer(models.WordLevel({'<pad>': 0, '</s>': 1, '<unk>': 2,
                                            'user': 3, ':': 4, 'hello': 5,
                                            'world': 6, 'answer': 7}, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    foundation = T5Config(vocab_size=8, d_model=16, d_ff=32, num_layers=1,
                          num_decoder_layers=1, num_heads=2, d_kv=8,
                          decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
                          dropout_rate=0.0)
    return {'foundation_config': foundation.to_dict(),
            'tokenizer_json': tokenizer.to_str(),
            'tokenizer_special_tokens': {'pad_token': '<pad>', 'eos_token': '</s>',
                                         'unk_token': '<unk>'},
            'workspace': {'slots': 3, 'steps': 2}, 'max_new_tokens': 3,
            'max_input_tokens': 32, 'max_turns': 2}


def test_owned_model_gradients_and_eager_parameters():
    model = Chatbot(tiny_config())
    params = {id(parameter) for parameter in model.parameters()}
    loss = model.loss_batch(['hello world'], ['answer'])
    loss.backward()
    assert loss.isfinite()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.workspace.parameters())
    assert params == {id(parameter) for parameter in model.parameters()}


def test_evidence_and_workspace_ablation_affect_computation():
    model = Chatbot(tiny_config()).eval()
    first = model.encode_workspace(['hello'])['conditioning']
    second = model.encode_workspace(['world'])['conditioning']
    assert not torch.allclose(first, second)
    zero = model.encode_workspace(['hello'], workspace_ablation='zero')['conditioning']
    assert torch.count_nonzero(zero) == 0
    assert not torch.allclose(model.loss_batch(['hello'], ['answer']),
                              model.loss_batch(['hello'], ['answer'], workspace_ablation='zero'))


def test_target_never_enters_encoder():
    model = Chatbot(tiny_config())
    seen = []
    original = model.encoder.forward
    def record(value, **kwargs):
        seen.append(value)
        return original(value, **kwargs)
    model.encoder.forward = record
    model.loss_batch(['hello'], ['answer'])
    assert seen == [['hello']]


def test_checkpoint_parity_and_separate_sessions(tmp_path):
    model = Chatbot(tiny_config()).eval()
    expected = model.loss_batch(['hello'], ['answer']).detach()
    model.save_pretrained(tmp_path / 'model')
    loaded = Chatbot.from_pretrained(tmp_path / 'model').eval()
    assert torch.equal(expected, loaded.loss_batch(['hello'], ['answer']).detach())
    a, b = loaded.new_session(), loaded.new_session()
    a('hello')
    assert len(a.history) == 2 and b.history == [] and loaded.history == ()
    a.save(tmp_path / 'session.json')
    b.load(tmp_path / 'session.json')
    assert b.history == a.history
    b.history[0]['text'] = 'world'
    assert a.history[0]['text'] == 'hello'
    assert 'history' not in json.loads((tmp_path / 'model/tensorcode_config.json').read_text())


def test_failed_decode_rolls_back_and_restores_mode():
    model = Chatbot(tiny_config()).train()
    def fail(*args, **kwargs):
        raise RuntimeError('decode failed')
    model.decoder.forward = fail
    with pytest.raises(RuntimeError, match='decode failed'):
        model('hello')
    assert model.history == () and model.last_result is None and model.training


def test_objective_training_protocol():
    model = Chatbot(tiny_config())
    loss = model.training_operation({'inputs': ['hello'], 'targets': ['answer']})
    assert loss.ndim == 0 and loss.requires_grad
    assert model.operation_bindings()['objective'] is model.training_operation
    assert list(model.training_operation.parameters()) == list(model.parameters())


def test_bypass_is_exact_foundation_memory_and_invalid_ablation_rejected():
    model = Chatbot(tiny_config()).eval()
    raw = model.encoder(['hello'])['encoded']
    bypass = model.encode_workspace(['hello'], workspace_ablation='bypass')
    assert torch.equal(raw, bypass['conditioning'])
    with pytest.raises(ValueError, match='ablation'):
        model.encode_workspace(['hello'], workspace_ablation='unknown')


def test_session_compatibility_and_capacity(tmp_path):
    model = Chatbot(tiny_config())
    model.generate_batch = lambda inputs: ['answer'] * len(inputs)
    for _ in range(4):
        model('hello')
    assert len(model.history) == 4
    assert model.history[-1]['source_id'] == 'turn-7'
    path = tmp_path / 'session.json'
    model.save_session(path)
    state = json.loads(path.read_text())
    state['model'] = 'different'
    path.write_text(json.dumps(state))
    before = model.history
    with pytest.raises(ValueError, match='incompatible'):
        model.load_session(path)
    assert model.history == before


def test_tool_trainer_durable_experience_and_resume(tmp_path):
    from tensorcode.training import ToolTrainer
    from tensorcode.training import load
    model = Chatbot(tiny_config())
    trainer = ToolTrainer(model)
    session = trainer.capture(['hello'], ['answer'], source='test authored target')
    session.save(tmp_path / 'experience.json', operations=trainer.operations)
    restored_session = load(tmp_path / 'experience.json', operations=trainer.operations)
    assert torch.isfinite(torch.tensor(trainer.step(restored_session)))
    trainer.save_checkpoint(tmp_path / 'resume', progress={'batch': 1})
    fresh = ToolTrainer(Chatbot(tiny_config()))
    assert fresh.load_checkpoint(tmp_path / 'resume') == {'batch': 1}
    assert fresh.steps == 1
    fresh_experience = load(tmp_path / 'experience.json', operations=fresh.operations)
    assert torch.isfinite(torch.tensor(fresh.step(fresh_experience)))


def test_corrupt_source_ids_rejected_transactionally(tmp_path):
    model = Chatbot(tiny_config())
    model.generate_batch = lambda inputs: ['answer'] * len(inputs)
    model('hello')
    path = tmp_path / 'session.json'
    model.save_session(path)
    state = json.loads(path.read_text())
    state['history'][0]['source_id'] = 'invalid'
    path.write_text(json.dumps(state))
    original = model.history
    with pytest.raises(ValueError, match='source IDs'):
        model.load_session(path)
    assert model.history == original


def test_concurrent_session_calls_commit_complete_turns():
    from concurrent.futures import ThreadPoolExecutor
    model = Chatbot(tiny_config())
    model.generate_batch = lambda inputs: ['answer'] * len(inputs)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(model, ['hello', 'world']))
    assert [row['source_id'] for row in model.history] == [f'turn-{i}' for i in range(4)]
    assert [row['role'] for row in model.history] == ['user', 'assistant'] * 2


def test_foundation_bootstrap_preserves_actual_untied_weights(monkeypatch, tmp_path):
    import transformers
    model = Chatbot(tiny_config())
    model.foundation.lm_head.weight = torch.nn.Parameter(model.foundation.lm_head.weight.detach().clone() + 1)
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **kw: model.tokenizer)
    monkeypatch.setattr(transformers.AutoModelForSeq2SeqLM, 'from_pretrained', lambda *a, **kw: model.foundation)
    restored = Chatbot.from_foundation(str(tmp_path))
    assert restored.foundation.config.tie_word_embeddings
    assert torch.equal(restored.foundation.shared.weight, model.foundation.shared.weight)
    assert torch.equal(restored.foundation.lm_head.weight, model.foundation.lm_head.weight)
    assert restored.foundation.shared.weight is not restored.foundation.lm_head.weight
    model.eval()
    restored.eval()
    inputs = model.tokenizer(['hello'], return_tensors='pt')['input_ids']
    target = model.tokenizer(['answer'], return_tensors='pt')['input_ids']
    expected = model.foundation(input_ids=inputs, labels=target).loss
    actual = restored.loss_batch(['hello'], ['answer'], workspace_ablation='bypass')
    assert torch.equal(expected, actual)


def test_encoder_fingerprint_distinguishes_tokenizer():
    model = Chatbot(tiny_config())
    config = tiny_config()
    token_config = json.loads(config['tokenizer_json'])
    vocab = token_config['model']['vocab']
    vocab['hello'], vocab['world'] = vocab['world'], vocab['hello']
    config['tokenizer_json'] = json.dumps(token_config)
    other = Chatbot(config)
    assert model.encoder.configuration() != other.encoder.configuration()


def test_generation_configuration_survives_checkpoint(tmp_path):
    model = Chatbot(tiny_config())
    model.foundation.generation_config.num_beams = 3
    original = model.decoder.configuration()
    model.save_pretrained(tmp_path / 'model')
    restored = Chatbot.from_pretrained(tmp_path / 'model')
    assert restored.foundation.generation_config.num_beams == 3
    assert restored.decoder.configuration() == original


def test_generation_preserves_mixed_module_modes():
    model = Chatbot(tiny_config()).train()
    model.foundation.eval()
    expected = {name: module.training for name, module in model.named_modules()}
    model.generate_batch(['hello'])
    assert expected == {name: module.training for name, module in model.named_modules()}

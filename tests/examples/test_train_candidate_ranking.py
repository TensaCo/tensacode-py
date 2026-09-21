import copy
import importlib.util
from pathlib import Path
import runpy

import pytest
import torch


def module():
    path = Path(__file__).parents[2] / '.development/experiments/train_candidate_ranking.py'
    spec = importlib.util.spec_from_file_location('candidate_ranking', path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def rows():
    base = {'question_id': 'q', 'question': 'Which city?', 'evidence': [{'id': 'e', 'source_id': 'document', 'text': 'Paris is the city.'}],
            'reference_answer': 'secret reference', 'rationale': 'secret rationale'}
    return [dict(base, id='good', candidate='Paris', targets=dict(support=True, completeness=True, constraints=True)),
            dict(base, id='bad', candidate='London', targets=dict(support=False, completeness=None, constraints=True)),
            dict(base, id='unknown', candidate='Maybe', targets=dict(support=True, completeness=None, constraints=True))]


def test_groups_exclude_unresolved_but_keep_explicit_failures_without_label_leakage():
    groups, excluded = module().group_rows(rows())
    assert excluded == ['unknown']
    assert groups[0]['mixed']
    assert groups[0]['targets'] == [0., 1.]
    assert set(groups[0]['inputs']) == {'question', 'evidence', 'hypotheses'}
    assert 'secret' not in str(groups[0]['inputs'])
    assert 'targets' not in str(groups[0]['inputs'])
    assert groups[0]['inputs']['evidence'] == [{'source_id': 'e', 'text': 'Paris is the city.'}]
    changed = rows(); changed[-1]['question'] = 'Different'
    with pytest.raises(ValueError, match='identical'):
        module().group_rows(changed)
    changed = rows(); changed[0]['targets']['support'] = 1
    with pytest.raises(ValueError, match='boolean'):
        module().group_rows(changed)


def test_overflow_checks_all_native_segments_without_truncating():
    from tensorcode.tools.investigator import Investigator
    model = Investigator({'vocabulary': ['city'], 'dimensions': 8, 'max_tokens': 5})
    calls = []
    class Tokenizer:
        def __call__(self, text, *, truncation):
            calls.append((text, truncation))
            return {'input_ids': list(range(len(text.split()) + 2))}
    model.rank.tokenizer = Tokenizer()
    group = module().group_rows(rows())[0][0]
    receipt = module().check_lengths(model.rank, group)
    assert receipt['overflow']
    assert len(calls) == 4 and all(flag is False for _, flag in calls)


def test_extract_only_rank_then_step_exact_continuation_and_reload(tmp_path):
    from tensorcode.tools.chatbot import Chatbot
    import sys
    models = str(Path(__file__).parents[1] / 'models')
    sys.path.insert(0, models)
    try:
        config = runpy.run_path(str(Path(models) / 'test_cognitive_chatbot.py'))['config']()
    finally:
        sys.path.remove(models)
    bot = Chatbot(config).eval()
    bot.save_pretrained(tmp_path / 'source')
    mod = module(); ranker = mod.extract_rank(tmp_path / 'source')
    assert ranker.generator is ranker.verifier is ranker.episodic_encoder is None
    for key, value in bot.investigator.rank.state_dict().items():
        assert torch.equal(ranker.rank.state_dict()[key], value)
    groups, _ = mod.group_rows(rows())
    before = copy.deepcopy(ranker.state_dict())
    report = mod.train(ranker, groups, tmp_path / 'run', mod.helper_module(), epochs=1)
    assert report['optimizer_continuation_exact'] and report['owned_reload_exact']
    assert report['changed_trainable_tensors'] > 0
    assert any(not torch.equal(before[k], v) for k, v in ranker.state_dict().items())
    all_bad = copy.deepcopy(groups[0]); all_bad['targets'] = None; all_bad['mixed'] = False
    for row in all_bad['rows']:
        row['good'] = False
    metrics = mod.evaluate(ranker, [all_bad])['summary']
    assert metrics['unanswerable_ranking_groups'] == 1
    assert metrics['active']['answerable']['rate'] is None


def test_extract_preserves_bart_tied_parameter_identity_and_saved_dtype(tmp_path):
    from tensorcode.tools.chatbot import Chatbot
    from transformers import BartConfig
    import sys
    models = str(Path(__file__).parents[1] / 'models')
    sys.path.insert(0, models)
    try:
        config = runpy.run_path(str(Path(models) / 'test_cognitive_chatbot.py'))['config']()
    finally:
        sys.path.remove(models)
    rank = config['cognition']['investigator']
    generator = rank['generator']
    rank.update(foundation_config=BartConfig(vocab_size=8, d_model=8,
        encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
        decoder_attention_heads=2, encoder_ffn_dim=16, decoder_ffn_dim=16,
        max_position_embeddings=32, pad_token_id=0, bos_token_id=1,
        eos_token_id=2, decoder_start_token_id=1).to_dict(),
        tokenizer_json=generator['tokenizer_json'],
        tokenizer_special_tokens=generator['tokenizer_special_tokens'],
        freeze_foundation=False)
    bot = Chatbot(config).eval()
    # Distinct component dtypes must survive without splitting tied Parameters.
    bot.investigator.rank.encode.module.model.double()
    bot.save_pretrained(tmp_path / 'source-bart')
    restored = module().extract_rank(tmp_path / 'source-bart')
    native = restored.rank.encode.module.model
    assert native.shared.weight is native.encoder.embed_tokens.weight
    assert native.shared.weight is native.decoder.embed_tokens.weight
    assert native.shared.weight.dtype == torch.float64
    assert native.shared.weight.requires_grad
    assert restored.rank.query.module.weight.dtype == torch.float32
    expected = bot.investigator.rank.state_dict()
    for name, value in restored.rank.state_dict().items():
        assert value.dtype == expected[name].dtype
        assert torch.equal(value, expected[name])
    # An optimizer sees the shared native embedding exactly once.
    assert sum(p is native.shared.weight for p in restored.parameters()) == 1
    before = native.shared.weight.detach().clone()
    native.shared.weight.grad = torch.ones_like(native.shared.weight)
    optimizer = torch.optim.SGD(restored.parameters(), lr=.125)
    optimizer.step()
    torch.testing.assert_close(native.shared.weight, before - .125, rtol=0, atol=0)

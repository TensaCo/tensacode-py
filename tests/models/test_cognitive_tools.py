import copy

import pytest
import torch

from tensorcode.tools.investigator import Investigator
from tensorcode.tools.planner import Planner

CONFIG = {'vocabulary': ['find', 'red', 'blue', 'evidence', 'choose'], 'dimensions': 8, 'slots': 2, 'steps': 1}
CASE = {'question': 'find red', 'evidence': [{'source_id': 'original', 'text': 'red evidence'}], 'hypotheses': [{'id': 'r', 'text': 'red'}, {'id': 'b', 'text': 'blue'}]}


def test_investigator_gradients_and_receipt():
    torch.manual_seed(11)
    tool = Investigator(CONFIG)
    parameter_ids = [id(p) for p in tool.parameters()]
    before = copy.deepcopy(CASE)
    loss = tool.objective({'inputs': CASE, 'targets': 'r'})
    loss.backward()
    assert tool.rank.encode.module.weight.grad.abs().sum() > 0
    assert tool.rank.workspace.queries.grad.abs().sum() > 0
    assert parameter_ids == [id(p) for p in tool.parameters()]
    receipt = tool(CASE)
    assert receipt['evidence'] == CASE['evidence']
    assert receipt['selected_id'] in {'r', 'b'}
    assert len(receipt['candidates']) == 2
    assert sum(item['probability'] for item in receipt['candidates']) == pytest.approx(1)
    assert len(receipt['attention'][0]) == len(receipt['attention_source_ids'])
    assert 'original' in receipt['attention_source_ids']
    assert CASE == before
    assert 'objective' in tool.operation_bindings()


def test_context_changes_candidate_scores_and_targets_do_not_enter_prediction():
    torch.manual_seed(4)
    tool = Investigator(CONFIG)
    other = copy.deepcopy(CASE)
    other['evidence'][0]['text'] = 'blue evidence'
    assert not torch.allclose(tool.rank(CASE), tool.rank(other))
    scores = tool.rank(CASE).detach().clone()
    tool.loss(CASE, 0)
    tool.loss(CASE, 1)
    assert torch.equal(scores, tool.rank(CASE))
    assert scores[0] != scores[1]


@pytest.mark.parametrize('cls', [Investigator, Planner])
def test_checkpoint_roundtrip(tmp_path, cls):
    tool = cls(CONFIG)
    case = CASE if cls is Investigator else {'goal': CASE['question'], 'evidence': CASE['evidence'], 'plans': CASE['hypotheses']}
    tool.save_pretrained(tmp_path)
    restored = cls.from_pretrained(tmp_path)
    assert torch.equal(tool.rank(case), restored.rank(case))
    assert restored(case) == tool(case)


def test_observed_plan_loss_does_not_fabricate_other_targets():
    tool = Planner(CONFIG)
    case = {'goal': 'choose red', 'evidence': CASE['evidence'], 'plans': CASE['hypotheses']}
    observed = {'candidate_id': 'r', 'outcome': 0.75}
    expected = (tool.rank(case)[0] - 0.75).square()
    assert torch.allclose(tool.loss(case, observed), expected)
    tool.loss(case, observed).backward()
    assert tool.rank.workspace.queries.grad.abs().sum() > 0
    assert all('outcome' not in candidate for candidate in tool(case)['candidates'])
    with pytest.raises(ValueError):
        tool.loss(case, {'candidate_id': 'missing', 'outcome': 1})


def test_invalid_inputs_rejected():
    tool = Investigator(CONFIG)
    bad = copy.deepcopy(CASE)
    bad['hypotheses'][1]['id'] = 'r'
    with pytest.raises(ValueError):
        tool(bad)
    with pytest.raises(ValueError):
        tool.loss(CASE, True)
    with pytest.raises(ValueError):
        Investigator({**CONFIG, 'vocabulary': ['red', 'red']})


def test_training_reduces_explicit_supervised_loss():
    torch.manual_seed(14)
    tool = Investigator(CONFIG)
    initial = float(tool.loss(CASE, 'r').detach())
    optimizer = torch.optim.Adam(tool.parameters(), lr=0.02)
    for _ in range(15):
        optimizer.zero_grad()
        loss = tool.loss(CASE, 'r')
        loss.backward()
        optimizer.step()
    assert float(tool.loss(CASE, 'r').detach()) < initial * 0.25
    assert tool(CASE)['selected_id'] == 'r'


def test_sessions_are_independent_transactional_and_restore(tmp_path):
    from tensorcode._internal.ranking import RankingSession
    tool = Investigator(CONFIG)
    one, two = tool.new_session(), tool.new_session()
    first = copy.deepcopy(CASE)
    first['hypotheses'] = first['hypotheses'][:1]
    receipt = one(first)
    assert receipt['revised'] is False
    first['evidence'][0]['text'] = 'mutated'
    receipt['evidence'].clear()
    assert one.history[0]['inputs']['evidence'][0]['text'] == 'red evidence'
    assert one.history[0]['receipt']['evidence']
    assert two.history == []
    before = one.history
    with pytest.raises(ValueError):
        one({'question': 'find red', 'hypotheses': []})
    assert one.history == before
    path = tmp_path / 'session.json'
    one.save(path)
    loaded = RankingSession.load(path, tool)
    second = copy.deepcopy(CASE)
    second['hypotheses'] = second['hypotheses'][1:]
    updated = loaded(second)
    assert updated['previous_selected_id'] == 'r'
    assert updated['selected_id'] == 'b'
    assert updated['revised'] is True
    assert one.history == before
    with pytest.raises(ValueError):
        RankingSession.load(path, Planner(CONFIG))


def test_workspace_ablations_are_explicit_and_differentiable():
    tool = Investigator(CONFIG)
    baseline = tool.rank(CASE)
    zero = tool.rank.compute(CASE, workspace_ablation='zero')[0]
    bypass = tool.rank.compute(CASE, workspace_ablation='bypass')[0]
    assert zero.shape == bypass.shape == baseline.shape
    assert not torch.equal(baseline, zero)
    assert not torch.equal(baseline, bypass)
    with pytest.raises(ValueError):
        tool(CASE, context={'ignored': True})


def test_distribution_targets_supervise_all_positive_candidates():
    tool = Investigator(CONFIG)
    logits = tool.rank(CASE)
    expected = -(logits.log_softmax(-1) * logits.new_tensor([0.5, 0.5])).sum()
    assert torch.allclose(tool.loss(CASE, [0.5, 0.5]), expected)
    for target in [[1, 1], [-1, 2], [float('nan'), 0], [1]]:
        with pytest.raises(ValueError):
            tool.loss(CASE, target)


def test_owned_native_foundation_reconstructs_without_download(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import BertConfig
    tokenizer = Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1, 'find': 2, 'red': 3, 'blue': 4, 'evidence': 5}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    config = {'foundation_config': BertConfig(vocab_size=6, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16).to_dict(), 'tokenizer_json': tokenizer.to_str(), 'tokenizer_special_tokens': {'unk_token': '[UNK]', 'pad_token': '[PAD]'}, 'dimensions': 8, 'slots': 2, 'steps': 1, 'max_tokens': 16, 'cache_records': 2}
    tool = Investigator(config)
    tool.train()
    assert not tool.rank.encode.module.model.training
    assert not any(p.requires_grad for p in tool.rank.encode.module.model.parameters())
    calls = []
    handle = tool.rank.encode.module.model.register_forward_hook(lambda *args: calls.append(True))
    first = tool.rank(CASE).detach()
    assert torch.equal(first, tool.rank(CASE))
    assert len(calls) == 1
    tool.load_state_dict(tool.state_dict())
    assert torch.equal(first, tool.rank(CASE))
    assert len(calls) == 2
    handle.remove()
    tool.rank.clear_encoding_cache()
    with torch.inference_mode():
        tool(CASE)
    loss = tool.loss(CASE, [0.5, 0.5])
    loss.backward()
    assert tool.rank.projection.module.weight.grad.abs().sum() > 0
    tool.save_pretrained(tmp_path)
    restored = Investigator.from_pretrained(tmp_path)
    assert torch.equal(tool.rank(CASE), restored.rank(CASE))
    unfrozen = Investigator(dict(config, freeze_foundation=False))
    unfrozen.loss(CASE, 0).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in unfrozen.rank.encode.module.model.parameters())

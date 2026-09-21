import pytest
import torch

from tensorcode._internal.workspace import Workspace


def test_source_links_and_masked_evidence():
    torch.manual_seed(12)
    model = Workspace(6, slots=3, steps=2)
    evidence = torch.randn(2, 5, 6)
    mask = torch.tensor([[True, True, False, True, False], [True, False, True, True, True]])
    result = model(evidence, mask)
    changed = evidence.clone()
    changed[~mask] = 1e20
    other = model(changed, mask)
    assert result['conditioning'].shape == (2, 3, 6)
    assert result['mask'].all()
    assert torch.equal(result['conditioning'], other['conditioning'])
    assert torch.equal(result['attention'].masked_select(~mask[:, None, :]), torch.zeros(9))
    torch.testing.assert_close(result['attention'].sum(-1), torch.ones(2, 3))
    torch.testing.assert_close(result['relations'].sum(-1), torch.ones(2, 3))
    changed[0, 0] += 4
    assert not torch.allclose(model(changed, mask)['conditioning'], result['conditioning'])


def test_evidence_permutation_preserves_state_and_reindexes_links():
    torch.manual_seed(11)
    model = Workspace(5, slots=3)
    evidence = torch.randn(2, 4, 5)
    mask = torch.tensor([[True, False, True, True], [False, True, True, False]])
    permutation = torch.tensor([2, 0, 3, 1])
    original = model(evidence, mask)
    shuffled = model(evidence[:, permutation], mask[:, permutation])
    torch.testing.assert_close(original['conditioning'], shuffled['conditioning'])
    torch.testing.assert_close(original['relations'], shuffled['relations'])
    torch.testing.assert_close(original['attention'][:, :, permutation], shuffled['attention'])


def test_gradients_and_eager_parameters_roundtrip():
    torch.manual_seed(27)
    model = Workspace(6, slots=3)
    before = {name: id(parameter) for name, parameter in model.named_parameters()}
    evidence = torch.randn(2, 4, 6, requires_grad=True)
    mask = torch.tensor([[True, False, True, True], [True, True, True, False]])
    output = model(evidence, mask)
    (output['conditioning'] * torch.randn_like(output['conditioning'])).sum().backward()
    assert before == {name: id(parameter) for name, parameter in model.named_parameters()}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert evidence.grad[mask].abs().sum() > 0
    assert torch.equal(evidence.grad[~mask], torch.zeros_like(evidence.grad[~mask]))
    restored = Workspace(6, slots=3)
    restored.load_state_dict(model.state_dict())
    assert restored.configuration() == model.configuration()
    torch.testing.assert_close(restored(evidence, mask)['conditioning'], output['conditioning'])


@pytest.mark.parametrize('kwargs', [{'dimensions': 0}, {'dimensions': True}, {'dimensions': 3, 'steps': 0}, {'dimensions': 3, 'slots': 1.5}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        Workspace(**kwargs)


@pytest.mark.parametrize('evidence,mask', [
    (torch.empty(0, 2, 3), None), (torch.empty(2, 0, 3), None),
    (torch.ones(2, 2, 4), None), (torch.ones(2, 3), None),
    (torch.ones(2, 2, 3, dtype=torch.long), None),
    (torch.full((2, 2, 3), float('nan')), None),
    (torch.ones(2, 2, 3), torch.zeros(2, 2, dtype=torch.bool)),
    (torch.ones(2, 2, 3), torch.ones(2, 2)),
    (torch.ones(2, 2, 3), torch.ones(2, 3, dtype=torch.bool)),
])
def test_invalid_evidence(evidence, mask):
    with pytest.raises(ValueError):
        Workspace(3)(evidence, mask)

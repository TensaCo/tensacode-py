import pytest
import torch
from tensorcode.tools.scene import Scene


def sample():
    return {'question': 'object left of other', 'source_id': 'photo:1', 'pixels': torch.rand(3, 16, 16), 'candidates': [{'id': 'yes', 'text': 'supported'}, {'id': 'no', 'text': 'unsupported'}]}


def model():
    return Scene({'vocabulary': ['object', 'left', 'of', 'other', 'supported', 'unsupported'], 'dimensions': 8, 'slots': 3})


def test_spatial_sources_and_complete_weight_roundtrip(tmp_path):
    torch.manual_seed(4)
    tool, inputs = model(), sample()
    result = tool(inputs)
    assert result['patch_coordinates'] == [[4., 4.], [4., 12.], [12., 4.], [12., 12.]]
    assert result['attention_source_ids'] == ['photo:1'] * 4 + [None] * 4
    assert len(result['relations']) == 3
    assert sum(c['probability'] for c in result['candidates']) == pytest.approx(1.)
    tool.save_pretrained(tmp_path / 'scene')
    restored = Scene.from_pretrained(tmp_path / 'scene')
    assert restored(inputs) == result
    assert 'objective' in tool.operation_bindings()


def test_owned_visual_workspace_gradients_and_input_dependence():
    torch.manual_seed(3)
    tool, inputs = model(), sample()
    parameters = {name: id(p) for name, p in tool.named_parameters()}
    first = tool.rank(inputs)
    changed = dict(inputs, pixels=torch.zeros_like(inputs['pixels']))
    assert not torch.allclose(first, tool.rank(changed))
    tool.training_operation({'inputs': inputs, 'targets': 'yes'}).backward()
    for name, p in tool.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    assert tool.rank.image.module.weight.grad.abs().sum() > 0
    assert tool.rank.workspace.queries.grad.abs().sum() > 0
    assert parameters == {name: id(p) for name, p in tool.named_parameters()}


def test_authored_fixture_optimization_is_not_pretrained_claim():
    torch.manual_seed(12)
    tool, inputs = model(), sample()
    optimizer = torch.optim.Adam(tool.parameters(), lr=.02)
    before = tool.loss(inputs, 'no').item()
    for _ in range(15):
        optimizer.zero_grad()
        loss = tool.loss(inputs, 'no')
        loss.backward()
        optimizer.step()
    assert tool.loss(inputs, 'no').item() < before * .5


@pytest.mark.parametrize('change', [
    {'source_id': ''}, {'question': ''}, {'pixels': torch.zeros(1, 16, 16)},
    {'pixels': torch.ones(3, 16, 16) * 2}, {'pixels': torch.full((3, 16, 16), float('nan'))},
    {'pixels': torch.zeros(3, 3, 3)}, {'candidates': []},
    {'candidates': [{'id': 'x', 'text': 'a'}, {'id': 'x', 'text': 'b'}]},
])
def test_invalid_inputs(change):
    with pytest.raises(ValueError):
        model()(dict(sample(), **change))


def test_relational_text_order_is_preserved():
    torch.manual_seed(5)
    tool, inputs = model(), sample()
    inputs['candidates'] = [{'id': 'a', 'text': 'object left other'}, {'id': 'b', 'text': 'other left object'}]
    scores = tool.rank(inputs)
    assert not torch.isclose(scores[0], scores[1])
    swapped = dict(inputs, question='other left of object')
    assert not torch.allclose(tool.rank(swapped), scores)


def test_scene_feedback_replays_after_weight_reload(tmp_path):
    from tensorcode.training import ToolTrainer, load
    tool, inputs = model(), sample()
    trainer = ToolTrainer(tool)
    experience = trainer.capture(inputs, 'yes', source='authored mechanism fixture')
    experience.save(tmp_path / 'experience.json', operations=trainer.operations)
    tool.save_pretrained(tmp_path / 'model')
    restored = Scene.from_pretrained(tmp_path / 'model')
    resumed = ToolTrainer(restored)
    loaded = load(tmp_path / 'experience.json', operations=resumed.operations)
    before = restored.rank.image.module.weight.detach().clone()
    resumed.step(loaded)
    assert not torch.equal(before, restored.rank.image.module.weight)

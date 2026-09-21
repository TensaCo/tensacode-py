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


def tiny_foundation_model(image_size=16):
    import hashlib
    import json
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import CLIPConfig
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0, '[BOS]': 1, 'object': 2, 'left': 3, 'of': 4, 'other': 5, 'supported': 6, 'unsupported': 7, '[EOS]': 15}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(single='[BOS] $A [EOS]', special_tokens=[('[BOS]', 1), ('[EOS]', 15)])
    serialized = tokenizer.to_str()
    config = CLIPConfig(text_config={'vocab_size': 16, 'hidden_size': 8, 'intermediate_size': 16, 'num_hidden_layers': 1, 'num_attention_heads': 2, 'max_position_embeddings': 16, 'eos_token_id': 15, 'bos_token_id': 1, 'pad_token_id': 0}, vision_config={'image_size': image_size, 'patch_size': 8, 'hidden_size': 8, 'intermediate_size': 16, 'num_hidden_layers': 1, 'num_attention_heads': 2}, projection_dim=8)
    return Scene({'vocabulary': ['<foundation>'], 'dimensions': 8, 'slots': 3, 'foundation_config': json.loads(json.dumps(config.to_dict())), '_tokenizer_json': serialized, 'tokenizer_sha256': hashlib.sha256(serialized.encode()).hexdigest(), 'image_mean': [.5] * 3, 'image_std': [.5] * 3, 'patch_size': 8})


def test_owned_foundation_assets_gradients_cache_and_reload(tmp_path):
    pytest.importorskip('transformers')
    tool, inputs = tiny_foundation_model(), sample()
    tool.train()
    assert not tool.rank.foundation.training
    result = tool(inputs)
    assert len(result['patch_coordinates']) == 4
    assert result['attention_source_ids'][:4] == ['photo:1'] * 4
    tool.loss(inputs, 'yes').backward()
    assert tool.rank.image_projection.module.weight.grad.abs().sum() > 0
    assert tool.rank.workspace.queries.grad.abs().sum() > 0
    assert all(p.grad is None for p in tool.rank.foundation.parameters())
    changed = dict(inputs, pixels=torch.zeros_like(inputs['pixels']))
    assert not torch.allclose(tool.rank(inputs), tool.rank(changed))
    tool.save_pretrained(tmp_path / 'foundation')
    assert (tmp_path / 'foundation' / 'tokenizer.json').is_file()
    restored = Scene.from_pretrained(tmp_path / 'foundation')
    assert not restored.rank._vision_cache
    assert restored(inputs) == result


def test_foundation_requires_matching_assets():
    pytest.importorskip('transformers')
    tool = tiny_foundation_model()
    with pytest.raises(ValueError, match='tokenizer'):
        Scene(tool.configuration())


def test_from_foundation_preserves_owned_weights(monkeypatch):
    pytest.importorskip('transformers')
    from types import SimpleNamespace
    from transformers import CLIPModel, CLIPTokenizerFast, CLIPImageProcessor
    original = tiny_foundation_model()
    monkeypatch.setattr(CLIPModel, 'from_pretrained', lambda *args, **kwargs: original.rank.foundation)
    monkeypatch.setattr(CLIPTokenizerFast, 'from_pretrained', lambda *args, **kwargs: SimpleNamespace(backend_tokenizer=original.rank.tokenizer))
    monkeypatch.setattr(CLIPImageProcessor, 'from_pretrained', lambda *args, **kwargs: SimpleNamespace(image_mean=(.5, .5, .5), image_std=(.5, .5, .5)))
    imported = Scene.from_foundation('explicit/test-foundation', revision='pinned', dimensions=8)
    for name, weight in original.rank.foundation.state_dict().items():
        torch.testing.assert_close(imported.rank.foundation.state_dict()[name], weight, rtol=0, atol=0)
    assert imported.configuration()['foundation_source'] == {'repo_id': 'explicit/test-foundation', 'revision': 'pinned'}


def test_foundation_cache_invalidates_on_weight_and_dtype_changes():
    pytest.importorskip('transformers')
    first, second, inputs = tiny_foundation_model(), tiny_foundation_model(), sample()
    first(inputs)
    first.load_state_dict(second.state_dict())
    assert not first.rank._vision_cache and not first.rank._text_cache
    torch.testing.assert_close(first.rank(inputs), second.rank(inputs), rtol=0, atol=0)
    first.double()
    assert not first.rank._vision_cache and not first.rank._text_cache


def test_foundation_inference_cache_can_be_used_for_training_and_bfloat_pixels():
    pytest.importorskip('transformers')
    tool, inputs = tiny_foundation_model(), sample()
    inputs['pixels'] = inputs['pixels'].to(torch.bfloat16)
    with torch.inference_mode():
        tool(inputs)
    tool.loss(inputs, 'yes').backward()
    assert tool.rank.image_projection.module.weight.grad.abs().sum() > 0


def test_foundation_patch_coordinates_preserve_dropped_border():
    pytest.importorskip('transformers')
    tool, inputs = tiny_foundation_model(image_size=18), sample()
    coordinates = torch.tensor(tool(inputs)['patch_coordinates'])
    expected = torch.tensor([[4., 4.], [4., 12.], [12., 4.], [12., 12.]]) * 16 / 18
    torch.testing.assert_close(coordinates, expected)


def test_foundation_cache_hash_distinguishes_equal_bytes_different_dtypes():
    pytest.importorskip('transformers')
    tool, inputs = tiny_foundation_model(), sample()
    pixels = torch.full((3, 16, 16), .5, dtype=torch.float16)
    other = pixels.view(torch.bfloat16)
    tool.rank(dict(inputs, pixels=pixels))
    tool.rank(dict(inputs, pixels=other))
    assert len(tool.rank._vision_cache) == 2

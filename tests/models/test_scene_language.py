"""Small random local configurations verify mechanics, not visual competence."""
import hashlib
import json
from pathlib import Path
import tempfile

import pytest
import torch

pytest.importorskip('transformers')
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Idefics3Config, Idefics3Processor, Idefics3ImageProcessor, GenerationConfig

from tensorcode.tools.scene import Scene


def tiny_scene():
    vocabulary = {'[UNK]': 0, '[BOS]': 1, '[EOS]': 2, 'describe': 3, 'left': 4, 'right': 5, 'object': 6, 'user': 7, 'assistant': 8, ':': 9, '<image>': 10, '<fake_token_around_image>': 11, '<end_of_utterance>': 12, '<global-img>': 13}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', bos_token='[BOS]', eos_token='[EOS]', pad_token='[UNK]', additional_special_tokens=['<image>', '<fake_token_around_image>', '<end_of_utterance>', '<global-img>'])
    template = "{% for message in messages %}{{ message['role'] }}: {% for part in message['content'] %}{% if part['type'] == 'image' %}<image>{% else %}{{ part['text'] }}{% endif %}{% endfor %}{% endfor %}{% if add_generation_prompt %} assistant:{% endif %}"
    processor = Idefics3Processor(Idefics3ImageProcessor(do_resize=False, do_image_splitting=False, max_image_size={'longest_edge': 8}, size={'longest_edge': 8}), tokenizer, image_seq_len=4, chat_template=template)
    with tempfile.TemporaryDirectory() as directory:
        processor.save_pretrained(directory)
        assets = {p.name: p.read_text() for p in Path(directory).iterdir() if p.is_file()}
    config = Idefics3Config(vision_config={'hidden_size': 8, 'intermediate_size': 16, 'num_hidden_layers': 1, 'num_attention_heads': 2, 'image_size': 8, 'patch_size': 2}, text_config={'model_type': 'llama', 'hidden_size': 8, 'intermediate_size': 16, 'num_hidden_layers': 1, 'num_attention_heads': 2, 'num_key_value_heads': 2, 'vocab_size': 16, 'pad_token_id': 0}, pad_token_id=0, scale_factor=2, image_token_id=10)
    generation = GenerationConfig(bos_token_id=1, eos_token_id=2, pad_token_id=0, suppress_tokens=[10, 11, 12, 13])
    return Scene({'mode': 'language', 'language_config': json.loads(json.dumps(config.to_dict())), 'generation_config': generation.to_dict(), '_language_assets': assets, 'processor_hashes': {k: hashlib.sha256(v.encode()).hexdigest() for k, v in assets.items()}, 'workspace_dimensions': 4, 'workspace_slots': 2, 'max_new_tokens': 8}).eval()


def inputs():
    return {'pixels': torch.rand(3, 8, 8), 'question': 'describe object', 'source_id': 'fixture:image'}


def test_local_scene_interpretation_artifact_and_provenance(tmp_path):
    torch.manual_seed(7)
    tool, value = tiny_scene(), inputs()
    result = tool.interpret(value, max_new_tokens=2)
    assert result['verification'] == 'unverified'
    assert result['uncertainty'] == {'status': 'uncalibrated', 'confidence': None}
    assert result['source']['kind'] == 'full-image'
    assert result['source']['shape'] == [3, 8, 8]
    assert result['source']['source_id'] == 'fixture:image'
    assert not result['workspace']['active']
    assert result['workspace']['visual_tokens'] == 4
    assert 'boxes' not in result and 'claims' not in result
    tool.save_pretrained(tmp_path / 'scene')
    loaded = Scene.from_pretrained(tmp_path / 'scene')
    assert loaded.interpret(value, max_new_tokens=2) == result
    assert (tmp_path / 'scene' / 'processor' / 'tokenizer.json').exists()


def test_teacher_feedback_trains_residual_without_entering_workspace():
    tool, value = tiny_scene(), inputs()
    captured = []
    handle = tool.language.workspace.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    a = tool.loss(value, 'left object')
    tool.loss(value, 'right object')
    handle.remove()
    torch.testing.assert_close(captured[0], captured[1], rtol=0, atol=0)
    a.backward()
    assert tool.language.gate.grad is not None
    assert tool.language.gate.grad.abs() > 0
    assert all(p.grad is None for p in tool.language.model.parameters())
    tool.zero_grad()
    with torch.no_grad():
        tool.language.gate.fill_(.2)
    tool.loss(value, 'left object').backward()
    assert tool.language.workspace.queries.grad.abs().sum() > 0


def test_zero_residual_preserves_visual_states():
    tool, value = tiny_scene(), inputs()
    batch, _, _ = tool.language.prepare(value)
    assert torch.isfinite(tool.language.model(**batch).logits).all()
    original = batch['image_hidden_states'].detach().clone()
    with torch.no_grad():
        tool.language.gate.fill_(.5)
    revised, _, _ = tool.language.prepare(value)
    assert not torch.equal(original, revised['image_hidden_states'])


@pytest.mark.parametrize('replacement', [{'question': ''}, {'source_id': ''}, {'pixels': torch.zeros(1, 8, 8)}, {'pixels': torch.full((3, 8, 8), float('nan'))}])
def test_invalid_scene_language_inputs(replacement):
    tool = tiny_scene()
    with pytest.raises(ValueError):
        tool.interpret(dict(inputs(), **replacement))


def test_processor_asset_integrity(tmp_path):
    tool = tiny_scene()
    tool.save_pretrained(tmp_path / 'scene')
    path = tmp_path / 'scene' / 'processor' / 'tokenizer.json'
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='checksum'):
        Scene.from_pretrained(tmp_path / 'scene')


def test_language_feedback_replay_and_fresh_loading(tmp_path):
    from tensorcode.training import ToolTrainer, load
    tool, value = tiny_scene(), inputs()
    trainer = ToolTrainer(tool, lr=.1)
    experience = trainer.capture(value, 'left object', source='authored mechanism fixture')
    experience.save(tmp_path / 'experience.json', operations=trainer.operations)
    tool.save_pretrained(tmp_path / 'scene')
    restored = Scene.from_pretrained(tmp_path / 'scene')
    replay = ToolTrainer(restored, lr=.1)
    record = load(tmp_path / 'experience.json', operations=replay.operations)
    replay.step(record)
    assert restored.language.gate.abs() > 0


def test_zero_gate_preserves_foundation_image_encoding(monkeypatch):
    tool, value = tiny_scene(), inputs()
    original = tool.language.model.model.get_image_features
    captured = []
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result.pooler_output.detach().clone())
        return result
    monkeypatch.setattr(tool.language.model.model, 'get_image_features', capture)
    batch, _, _ = tool.language.prepare(value)
    torch.testing.assert_close(batch['image_hidden_states'], captured[0], rtol=0, atol=0)


def test_processor_paths_rejected_before_read(tmp_path):
    tool = tiny_scene()
    tool.save_pretrained(tmp_path / 'scene')
    path = tmp_path / 'scene' / 'tensorcode_config.json'
    config = json.loads(path.read_text())
    config['config']['processor_hashes'] = {'../../outside.json': 'invalid'}
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='asset names'):
        Scene.from_pretrained(tmp_path / 'scene')


def test_generation_dictionary_setting_supported():
    tool = tiny_scene()
    tool.language.model.generation_config.return_dict_in_generate = True
    assert isinstance(tool.interpret(inputs(), max_new_tokens=1)['interpretation'], str)


def test_named_processor_templates_preserved(monkeypatch, tmp_path):
    from transformers import Idefics3ForConditionalGeneration
    original = tiny_scene()
    processor = original.language.processor
    processor.chat_template = {'default': processor.chat_template, 'alternative': processor.chat_template + ' alternative'}
    monkeypatch.setattr(Idefics3ForConditionalGeneration, 'from_pretrained', lambda *args, **kwargs: original.language.model)
    original_load = Idefics3Processor.from_pretrained
    def load_processor(path, **kwargs):
        return processor if path == 'explicit/foundation' else original_load(path, **kwargs)
    monkeypatch.setattr(Idefics3Processor, 'from_pretrained', load_processor)
    owned = Scene.from_language_foundation('explicit/foundation', revision='pinned')
    assert owned.language.processor.chat_template == processor.chat_template
    owned.save_pretrained(tmp_path / 'scene')
    loaded = Scene.from_pretrained(tmp_path / 'scene')
    assert loaded.language.processor.chat_template == processor.chat_template


def test_scene_evaluation_rejects_same_content_shuffle():
    import importlib.util
    path = Path(__file__).parents[2] / 'examples' / 'evaluate_scene_language.py'
    spec = importlib.util.spec_from_file_location('scene_language_evaluation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = inputs()['pixels']
    with pytest.raises(ValueError, match='distinct image'):
        module.evaluate(None, [({'source_id': 'a'}, value), ({'source_id': 'b'}, value.clone())], max_new_tokens=1)


def test_inference_then_learning_preserves_native_gradients():
    tool, value = tiny_scene(), inputs()
    with torch.inference_mode():
        tool.interpret(value, max_new_tokens=1)
    tool.loss(value, 'left object').backward()
    assert tool.language.gate.grad is not None and torch.isfinite(tool.language.gate.grad)

import json

import pytest
import torch

from tensorcode._internal.pretrained import PretrainedTool


class Tiny(PretrainedTool):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = torch.nn.Linear(config['width'], config['width'], bias=False)
        self.decoder = torch.nn.Linear(config['width'], config['width'], bias=False)
        self.decoder.weight = self.encoder.weight
        self.session = []

    def forward(self, value):
        return self.decoder(self.encoder(value))


class Other(Tiny):
    pass


def test_roundtrip_shared_weights_and_independent_session(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError('network used')
    monkeypatch.setattr('huggingface_hub.snapshot_download', no_network)
    config = {'width': 3}
    model = Tiny(config)
    config['width'] = 9
    model.session.append('private message')
    value = torch.randn(2, 3)
    model.save_pretrained(tmp_path / 'model')
    restored = Tiny.from_pretrained(tmp_path / 'model', local_files_only=True)
    assert restored.config == {'width': 3}
    assert not restored.training
    assert restored.decoder.weight is restored.encoder.weight
    assert restored.session == []
    torch.testing.assert_close(restored(value), model(value))
    assert 'private message' not in (tmp_path / 'model' / 'tensorcode_config.json').read_text()
    assert model.configuration() == {'width': 3}


@pytest.mark.parametrize('change', [
    {'version': 999}, {'tool': 'untrusted.Other'}, {'config': []}, {'format': 'other'},
])
def test_rejects_incompatible_manifest(tmp_path, change):
    Tiny({'width': 2}).save_pretrained(tmp_path)
    path = tmp_path / 'tensorcode_config.json'
    data = json.loads(path.read_text())
    data.update(change)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        Tiny.from_pretrained(tmp_path)


def test_rejects_wrong_class_and_shape(tmp_path):
    Tiny({'width': 2}).save_pretrained(tmp_path)
    with pytest.raises(ValueError, match='tool'):
        Other.from_pretrained(tmp_path)
    path = tmp_path / 'tensorcode_config.json'
    data = json.loads(path.read_text())
    data['config']['width'] = 3
    path.write_text(json.dumps(data))
    with pytest.raises((RuntimeError, ValueError)):
        Tiny.from_pretrained(tmp_path)


def test_config_must_be_json():
    with pytest.raises(ValueError):
        Tiny({'width': 2, 'value': float('nan')})


def test_failed_save_preserves_previous_model(tmp_path, monkeypatch):
    model = Tiny({'width': 2})
    model.save_pretrained(tmp_path)
    previous = (tmp_path / 'model.safetensors').read_bytes()
    def fail(*args, **kwargs):
        raise RuntimeError('disk failure')
    monkeypatch.setattr(model, '_save_pretrained_assets', fail)
    with pytest.raises(RuntimeError, match='disk failure'):
        model.save_pretrained(tmp_path)
    assert (tmp_path / 'model.safetensors').read_bytes() == previous
    Tiny.from_pretrained(tmp_path)


def test_hub_resolution_preserves_revision_and_offline_options(tmp_path, monkeypatch):
    Tiny({'width': 2}).save_pretrained(tmp_path)
    observed = {}
    def download(**kwargs):
        observed.update(kwargs)
        return str(tmp_path)
    monkeypatch.setattr('huggingface_hub.snapshot_download', download)
    loaded = Tiny.from_pretrained('owner/model', revision='immutable-sha',
        local_files_only=True, cache_dir='/cache/models', token='token')
    assert loaded.config == {'width': 2}
    assert observed == {'repo_id': 'owner/model', 'revision': 'immutable-sha',
        'local_files_only': True, 'cache_dir': '/cache/models', 'token': 'token'}


def test_missing_local_directory_does_not_become_hub_request(tmp_path, monkeypatch):
    monkeypatch.setattr('huggingface_hub.snapshot_download', lambda **kw: pytest.fail('network'))
    with pytest.raises(FileNotFoundError):
        Tiny.from_pretrained(tmp_path / 'missing')


def test_push_publishes_only_model_assets(tmp_path, monkeypatch):
    uploaded = {}
    class API:
        def __init__(self, token):
            assert token == 'token'
        def create_repo(self, **kwargs):
            assert kwargs['repo_id'] == 'owner/model'
            assert kwargs['private']
        def upload_folder(self, **kwargs):
            from pathlib import Path
            folder = Path(kwargs['folder_path'])
            uploaded['names'] = {p.name for p in folder.iterdir()}
            assert kwargs['revision'] == 'main'
            restored = Tiny.from_pretrained(folder)
            assert restored.session == []
            return 'commit-url'
    monkeypatch.setattr('huggingface_hub.HfApi', API)
    model = Tiny({'width': 2})
    model.session.append('private')
    assert model.push_to_hub('owner/model', private=True, revision='main', token='token') == 'commit-url'
    assert uploaded['names'] == {'tensorcode_config.json', 'model.safetensors', 'README.md'}


def test_extra_weight_is_rejected(tmp_path):
    from safetensors.torch import load_file, save_file
    Tiny({'width': 2}).save_pretrained(tmp_path)
    path = tmp_path / 'model.safetensors'
    weights = load_file(path)
    weights['unexpected'] = torch.zeros(1)
    save_file(weights, path)
    with pytest.raises(RuntimeError, match='Unexpected key'):
        Tiny.from_pretrained(tmp_path)


def test_operation_bindings_find_vector_operations_and_deduplicate_aliases():
    from tensorcode.ops.vec import Transform
    model = Tiny({'width': 2})
    model.interpret = Transform(torch.nn.Linear(2, 2))
    model.alias = model.interpret
    bindings = model.operation_bindings()
    assert bindings == {'interpret': model.interpret}


def test_roundtrip_preserves_double_dtype_and_aliases(tmp_path):
    model = Tiny({'width': 2}).double()
    model.save_pretrained(tmp_path)
    loaded = Tiny.from_pretrained(tmp_path)
    assert loaded.encoder.weight.dtype == torch.float64
    assert loaded.decoder.weight is loaded.encoder.weight
    value = torch.randn(1, 2, dtype=torch.float64)
    torch.testing.assert_close(loaded(value), model(value), rtol=0, atol=0)


class Mixed(Tiny):
    def __init__(self, config):
        super().__init__(config)
        self.register_buffer('scale', torch.ones(2))
        self.register_buffer('count', torch.ones(2, dtype=torch.int64))
        self.other = torch.nn.Linear(2, 2)


def test_roundtrip_preserves_mixed_parameter_and_buffer_dtypes(tmp_path):
    model = Mixed({'width': 2})
    model.encoder.double()
    model.other.half()
    model.scale = model.scale.bfloat16()
    model.save_pretrained(tmp_path)
    loaded = Mixed.from_pretrained(tmp_path)
    assert loaded.decoder.weight is loaded.encoder.weight
    for name, expected in model.state_dict().items():
        actual = loaded.state_dict()[name]
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class SharedStorage(Tiny):
    def __init__(self, config):
        super().__init__(config)
        self.decoder.weight = torch.nn.Parameter(self.encoder.weight.detach())


def test_dtype_restoration_preserves_distinct_parameter_shared_storage(tmp_path):
    model = SharedStorage({'width': 2})
    # Module.double() can break distinct-parameter storage ties; establish a
    # genuine double-precision tied checkpoint explicitly.
    model.encoder.double()
    model.decoder.weight = torch.nn.Parameter(model.encoder.weight.detach())
    model.save_pretrained(tmp_path)
    loaded = SharedStorage.from_pretrained(tmp_path)
    assert loaded.encoder.weight is not loaded.decoder.weight
    assert loaded.encoder.weight.data_ptr() == loaded.decoder.weight.data_ptr()
    assert loaded.encoder.weight.dtype == torch.float64
    torch.testing.assert_close(loaded.encoder.weight, model.encoder.weight, rtol=0, atol=0)


def test_save_pretrained_writes_factual_card_and_preserves_custom_card(tmp_path):
    model = Tiny({'width': 2})
    model.save_pretrained(tmp_path)
    card = tmp_path / 'README.md'
    assert 'library_name: tensorcode' in card.read_text()
    assert 'from_pretrained' in card.read_text()
    assert 'training' in card.read_text().lower()
    card.write_text('# Authored model card\n')
    model.save_pretrained(tmp_path)
    assert card.read_text() == '# Authored model card\n'


def test_push_model_card_is_explicit_and_validated_before_network(monkeypatch):
    calls = []
    class API:
        def __init__(self, token):
            calls.append('api')
        def create_repo(self, **kwargs):
            pass
        def upload_folder(self, **kwargs):
            from pathlib import Path
            assert (Path(kwargs['folder_path']) / 'README.md').read_text() == '# Evaluated model\n'
            return 'published'
    monkeypatch.setattr('huggingface_hub.HfApi', API)
    model = Tiny({'width': 2})
    with pytest.raises(TypeError, match='model_card'):
        model.push_to_hub('owner/model', model_card=123)
    assert calls == []
    assert model.push_to_hub('owner/model', model_card='# Evaluated model\n') == 'published'

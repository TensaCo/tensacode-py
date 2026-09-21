import json
import pytest
import torch
from tensorcode.ops import vec
from tensorcode.ops.vec.encode import VocabularyEncoder, PatchEncoder


def test_specialized_encoders_construct_from_config_and_restore(tmp_path):
    configs = [
        (VocabularyEncoder, {'vocabulary':['hello','world'],'dimensions':4,
          'output_space':vec.Space('words',4).configuration()}, 'hello world'),
        (PatchEncoder, {'patch_size':2,'in_channels':3,
          'output_space':vec.Space('patches',4,organization='spatial').configuration()}, torch.rand(3,4,4)),
    ]
    for cls,config,value in configs:
        model=cls(config)
        expected=model(value)
        directory=tmp_path/cls.__name__
        model.save_pretrained(directory)
        restored=cls.from_pretrained(directory)
        assert type(restored) is cls
        assert restored.configuration()==model.configuration()
        torch.testing.assert_close(restored(value).tensor,expected.tensor,atol=0,rtol=0)
        assert json.loads((directory/'tensorcode_config.json').read_text())['tool']==f'{cls.__module__}.{cls.__name__}'


def test_parameter_free_selector_configuration_round_trip(tmp_path):
    for cls,config in [(vec.Decide,{'largest':False}),(vec.Retrieve,{'k':2,'largest':False})]:
        selector=cls(config)
        selector.save_pretrained(tmp_path/cls.__name__)
        assert cls.from_pretrained(tmp_path/cls.__name__).configuration()==selector.configuration()
        with pytest.raises((ValueError,TypeError)):
            cls({'unexpected':True})


def test_graph_stubs_accept_only_declared_config():
    from tensorcode.ops.graph import Transform
    graph=Transform({})
    with pytest.raises(NotImplementedError):
        graph(None)
    with pytest.raises(NotImplementedError):
        Transform.from_pretrained('unused')
    with pytest.raises(ValueError):
        Transform({'model':'implicit'})

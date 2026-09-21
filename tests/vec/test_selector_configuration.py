"""Selectors retain exact selection settings in data-only artifacts."""
import pytest
from tensorcode.ops.vec import Decide, Retrieve


@pytest.mark.parametrize('cls,config', [(Decide, {'largest': False}),
                                      (Retrieve, {'k': 2, 'largest': False})])
def test_config_artifacts_are_weightless_and_preserve_settings(cls, config, tmp_path):
    operation = cls(config)
    config['largest'] = True
    assert operation.largest is False
    assert list(operation.parameters()) == []
    operation.save_pretrained(tmp_path)
    assert [path.name for path in tmp_path.iterdir()] == ['tensorcode_config.json']
    restored = cls.from_pretrained(tmp_path)
    assert restored.configuration() == operation.configuration()
    assert restored.largest is False


@pytest.mark.parametrize('bad', [0, 1, 'true', None, [], {}])
@pytest.mark.parametrize('cls', [Decide, Retrieve])
def test_largest_requires_actual_boolean(cls, bad):
    config = {'largest': bad}
    if cls is Retrieve:
        config['k'] = 1
    with pytest.raises((TypeError, ValueError), match='largest'):
        cls(config)


@pytest.mark.parametrize('config', [None, {}, {'k': True}, {'k': 0}, {'k': 1.5}])
def test_retrieve_requires_positive_integer_count(config):
    with pytest.raises(ValueError, match='positive integer'):
        Retrieve(config)


def test_selector_artifact_cannot_be_loaded_as_different_operation(tmp_path):
    Decide().save_pretrained(tmp_path)
    with pytest.raises(ValueError, match='identity'):
        Retrieve.from_pretrained(tmp_path)

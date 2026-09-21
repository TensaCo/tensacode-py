"""Public pretrained operations reject misspelled and obsolete configuration."""
import pytest
from tensorcode.ops.vec.encode import TextEncoder, ImageEncoder
from tensorcode.ops.vec.decode import TextDecoder, ImageDecoder


@pytest.mark.parametrize('cls',[TextEncoder,TextDecoder,ImageEncoder,ImageDecoder])
def test_native_operation_unknown_keys_fail_before_model_construction(cls):
    with pytest.raises(ValueError,match='Unknown configuration'):
        cls({'misspelled_option': True})


@pytest.mark.parametrize('cls',[TextEncoder,TextDecoder,ImageEncoder,ImageDecoder])
@pytest.mark.parametrize('config',[lambda value: value, [('foundation','example')]])
def test_native_operation_rejects_non_json_constructor(config,cls):
    with pytest.raises((ValueError,TypeError),match='config|JSON'):
        cls(config)

"""The developer contract follows operations, independent of backend layout."""
import importlib
import importlib.util
import subprocess
import sys


def test_vector_classes_have_public_operation_identities():
    from tensorcode.ops import vec
    for module, names in {
        'encode': ('TextEncoder', 'ImageEncoder', 'VocabularyEncoder', 'PatchEncoder'),
        'decode': ('Decode', 'TextDecoder', 'ImageDecoder'),
        'transform': ('Transform',),
        'classify': ('Classify',),
        'score': ('Score',),
    }.items():
        public = importlib.import_module(f'tensorcode.ops.vec.{module}')
        for name in names:
            cls = getattr(public, name)
            assert cls is getattr(vec, name)
            assert cls.__module__ == public.__name__
            assert cls.__qualname__ == name


def test_backend_modules_and_tool_only_encoder_are_not_public():
    for name in ('image', 'text_model', 'vision_model', 'diffusion'):
        assert importlib.util.find_spec(f'tensorcode.ops.vec.{name}') is None
    from tensorcode.ops.vec import encode
    assert not hasattr(encode, 'SequenceEncoder')
    assert not hasattr(encode, 'tokenize')


def test_operation_imports_do_not_load_foundation_libraries():
    result = subprocess.run([sys.executable, '-c', '''
import sys
from tensorcode.ops.vec.encode import TextEncoder, ImageEncoder, VocabularyEncoder, PatchEncoder
from tensorcode.ops.vec.decode import TextDecoder, ImageDecoder
assert 'transformers' not in sys.modules
assert 'diffusers' not in sys.modules
'''], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_specialized_encoders_declare_their_output_space():
    import torch
    from tensorcode.ops.vec import Space
    from tensorcode.ops.vec.encode import VocabularyEncoder, PatchEncoder
    text_space = Space('vocabulary', 4)
    text = VocabularyEncoder({'vocabulary': ['hello'], 'dimensions': 4, 'output_space': text_space.configuration()})
    assert text.output_space == text('hello').space == text_space
    assert text._tool_identity() == 'tensorcode.ops.vec.encode.VocabularyEncoder'
    assert text.configuration()['output_space'] == text_space.configuration()
    image_space = Space('patches', 4, organization='spatial')
    image = PatchEncoder({'patch_size': 2, 'in_channels': 3, 'output_space': image_space.configuration()})
    assert image.output_space == image(torch.zeros(3, 4, 4)).space == image_space
    assert image._tool_identity() == 'tensorcode.ops.vec.encode.PatchEncoder'
    assert image.configuration()['output_space'] == image_space.configuration()

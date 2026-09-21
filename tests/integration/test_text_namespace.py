"""Public text operations remain usable without optional model dependencies."""
import subprocess
import sys


def test_text_namespace_composes_without_loading_optional_dependencies():
    result = subprocess.run(
        [sys.executable, "-c", """
import sys
from tensorcode.ops import text, graph
from tensorcode.ops.text import decode
from tensorcode.runtime import JsonMemory, DecisionPipeline
messages = text.TextEncoder()('hello')
response = text.Transform.from_model(lambda messages: 'answer')(messages)
assert text.TextDecoder()(response) == 'answer'
assert not hasattr(text, 'Decode')
assert not hasattr(decode, 'Decode')
assert graph.Transform({}).configuration()['implementation'] == 'unimplemented'
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
"""], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_removed_llm_namespace_is_not_an_importable_compatibility_path():
    result = subprocess.run(
        [sys.executable, "-c", """
import importlib.util
assert importlib.util.find_spec('tensorcode.ops.llm') is None
"""], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

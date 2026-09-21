"""Symbolic API declarations must never masquerade as implemented cognition."""
import asyncio
import importlib.util

import pytest

from tensorcode.ops import graph
from tensorcode.ops.base import Operation


@pytest.mark.parametrize("name,value", [
    ("Encode", object()),
    ("TextEncode", "Evidence with two competing interpretations."),
    ("Decode", graph.Graph(nodes=("a",))),
    ("TextDecode", graph.Graph(nodes=("a",))),
    ("Transform", graph.Graph(nodes=("a",))),
    ("Score", graph.Graph(nodes=("a",))),
    ("Retrieve", graph.Graph(nodes=("a",))),
    ("Classify", graph.Graph(nodes=("a",))),
    ("Decide", graph.ChoiceInput(graph.Graph(nodes=("a",)), (graph.Graph(nodes=("b",)),))),
])
def test_symbolic_contracts_fail_explicitly_in_sync_and_async_calls(name, value):
    operation = getattr(graph, name)()
    assert isinstance(operation, Operation)
    assert operation.replayable is False
    assert operation.configuration()["implementation"] == "unimplemented"
    with pytest.raises(NotImplementedError, match="symbolic semantics are not implemented"):
        operation(value)
    with pytest.raises(NotImplementedError, match="symbolic semantics are not implemented"):
        asyncio.run(operation.acall(value))


def test_symbolic_transform_does_not_accept_callback_compatibility():
    with pytest.raises(TypeError):
        graph.Transform(lambda value, context: value)


def test_neural_graph_implementation_and_json_operation_aliases_are_removed():
    assert importlib.util.find_spec("tensorcode.ops.graph.neural") is None
    assert not hasattr(graph, "JSONEncoder")
    assert not hasattr(graph, "JSONDecoder")


def test_choice_input_requires_actual_graph_alternatives():
    objective = graph.Graph(nodes=("goal",))
    with pytest.raises(ValueError, match="at least one Graph"):
        graph.ChoiceInput(objective, ())
    with pytest.raises(TypeError, match="objective"):
        graph.ChoiceInput("goal", (objective,))


@pytest.mark.parametrize('name', ['Encode', 'TextEncode', 'Decode', 'TextDecode',
                                  'Transform', 'Score', 'Retrieve', 'Classify', 'Decide'])
def test_graph_configuration_does_not_enable_artifact_or_foundation_loading(name, tmp_path):
    cls = getattr(graph, name)
    operation = cls({})
    assert operation.configuration() == cls().configuration()
    with pytest.raises(ValueError, match='Unknown configuration'):
        cls({'model': 'implicit'})
    with pytest.raises(NotImplementedError):
        cls.from_foundation('unused')
    with pytest.raises(NotImplementedError):
        cls.from_pretrained('unused')
    with pytest.raises(NotImplementedError):
        operation.save_pretrained(tmp_path / name)
    assert not (tmp_path / name).exists()

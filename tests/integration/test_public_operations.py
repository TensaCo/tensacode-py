import pytest
from tensorcode.ops import text as text_ops, graph


def test_messages_preserve_context_roles_and_caller_state():
    observed = []
    def model(messages):
        observed.append(messages)
        return 'answer'
    encode = text_ops.TextEncoder()
    respond = text_ops.Transform(model)
    original = encode('hello')
    result = respond(original, context={'policy': encode('be brief')})
    assert original == (text_ops.Message('user', 'hello'),)
    assert result[-1] == text_ops.Message('assistant', 'answer')
    assert any('be brief' in m.content for m in observed[0])
    assert observed[0][-1].content == 'hello'
    assert text_ops.TextDecoder()(result) == 'answer'


def test_graph_stub_preserves_supplied_representation_without_inference():
    value = graph.Graph(nodes=('a', 'b', 'c'), edges=(('a', 'next', 'b'),), sources=('document:1',))
    with pytest.raises(NotImplementedError, match='symbolic semantics'):
        graph.Transform()(value)
    assert value.neighbors('a', relation='next') == ('b',)
    assert value.sources == ('document:1',)
    with pytest.raises(ValueError):
        graph.Graph(nodes=('a',), edges=(('a', 'next', 'missing'),))


def test_invalid_model_response_is_not_silently_converted_to_text():
    with pytest.raises(TypeError):
        text_ops.Transform(lambda messages: None)(text_ops.TextEncoder()('hello'))


def test_graph_rejects_mutable_node_and_source_payloads():
    with pytest.raises((TypeError, ValueError)):
        graph.Graph(nodes=('a',), sources=(['mutable'],))

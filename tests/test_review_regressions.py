from dataclasses import dataclass
import pytest
import torch
from tensorcode.ops import Operation, vec
from tensorcode.tools.agents import Chatbot
import tensorcode as tc


def test_dataclass_operand_keeps_encoder_dependency_and_gradient():
    @dataclass(frozen=True)
    class Box:
        tensor: torch.Tensor
    class Consume(Operation):
        replayable = True
        def forward(self, value, *, context=None):
            return value.tensor * 2
    encoder = vec.Transform(torch.nn.Linear(2, 1))
    with tc.trace() as episode:
        output = Consume()(Box(encoder(torch.ones(2))))
    port = episode.ref(output)
    assert len(episode.example(port).calls) == 2
    episode.replay(port).sum().backward()
    assert encoder.module.weight.grad is not None


def test_container_tensor_replacement_is_a_mutation():
    class Make(Operation):
        replayable = True
        def forward(self, value, *, context=None):
            return [value * 1]
    class Consume(Operation):
        replayable = True
        def forward(self, value, *, context=None):
            return value[0] * 2
    with tc.trace():
        result = Make()(torch.tensor([1.]))
        result[0] = torch.tensor([10.])
        with pytest.raises(ValueError, match='mutat'):
            Consume()(result)


def test_two_chatbot_turns_in_one_trace_preserve_history():
    bot = Chatbot(model=lambda messages: 'reply')
    with tc.trace() as episode:
        assert bot('one') == 'reply'
        assert bot('two') == 'reply'
    assert len(bot.history) == 4
    assert len(episode.calls) == 6


def test_explicit_reference_does_not_bypass_mutation_check():
    op = vec.Transform(torch.nn.Linear(1, 1))
    with tc.trace() as episode:
        result = op(torch.ones(1))
        port = episode.ref(result)
        result.add_(1)
        with pytest.raises(ValueError, match='mutat'):
            op(port)


def test_tracing_does_not_replace_primary_input_container():
    class Append(Operation):
        def forward(self, value, *, context=None):
            value.append('changed')
            return len(value)
    items = []
    with tc.trace():
        assert Append()(items) == 1
    assert items == ['changed']


def test_inference_mode_traces_and_detects_inference_tensor_mutation():
    op = vec.Transform(torch.nn.Linear(1, 1))
    with torch.inference_mode():
        with tc.trace() as episode:
            result = op(torch.ones(1))
            op(result)
            result.add_(1)
            with pytest.raises(ValueError, match='mutat'):
                op(result)
    assert len(episode.calls) == 2

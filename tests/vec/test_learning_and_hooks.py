import torch
import pytest
import tensorcode as tc
from tensorcode.ops import vec


def test_native_gradient_reaches_both_modules_through_trace_and_replay():
    torch.manual_seed(1)
    encoder = vec.Transform(torch.nn.Linear(2, 3))
    classifier = vec.Classify(torch.nn.Linear(3, 2), labels=('left', 'right'))
    x = torch.tensor([[1., 0.], [0., 1.]])
    y = torch.tensor([0, 1])
    optimizer = torch.optim.SGD(list(encoder.parameters()) + list(classifier.parameters()), lr=.2)
    with tc.trace() as episode:
        prediction = classifier(encoder(x))
    initial = torch.nn.functional.cross_entropy(prediction.logits, y).item()
    target = episode.ref(prediction)
    assert len(episode.example(target).inputs) == 1
    for _ in range(40):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(episode.replay(target).logits, y)
        loss.backward()
        assert encoder.module.weight.grad is not None
        assert classifier.module.weight.grad is not None
        optimizer.step()
    assert loss.item() < initial * .4


def test_vector_invocation_keeps_module_hooks_and_context_gradients():
    op = vec.Transform(torch.nn.Identity(), combine=lambda value, context: value + context['bias'])
    calls = []
    op.register_forward_hook(lambda module, args, result: calls.append(result))
    x = torch.ones(2, requires_grad=True)
    bias = torch.ones(2, requires_grad=True)
    result = op(x, context={'bias': bias})
    result.sum().backward()
    assert calls[0] is result
    assert torch.equal(bias.grad, torch.ones(2))


def test_mutated_intermediate_tensor_is_rejected_by_trace():
    op = vec.Transform(torch.nn.Identity())
    with tc.trace():
        result = op(torch.tensor([1.]))
        result.add_(1)
        with pytest.raises(ValueError, match='mutat'):
            op(result)


def test_classifier_validates_label_count_and_keeps_logits():
    op = vec.Classify(torch.nn.Identity(), labels=('a', 'b'))
    result = op(torch.tensor([1., 3.]))
    assert result.value == 'b'
    assert torch.allclose(result.probabilities.sum(), torch.tensor(1.))
    with pytest.raises(ValueError, match='labels'):
        op(torch.ones(3))


def test_shared_backbone_is_registered_once_and_receives_both_path_gradients():
    backbone = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        backbone.weight.fill_(2.)
    a, b = vec.Transform(backbone), vec.Transform(backbone)
    modules = torch.nn.ModuleList((a, b))
    assert len(list(modules.parameters())) == 1
    with tc.trace():
        output = b(a(torch.ones(1)))
    output.sum().backward()
    assert torch.equal(backbone.weight.grad, torch.tensor([[4.]]))

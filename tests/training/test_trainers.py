from tensorcode._internal.training.checkpoint import save_checkpoint, load_checkpoint
import pytest
import torch
from tensorcode import trace, training
from tensorcode.ops.vec import Transform, Classify


def test_shared_parameters_deduplicated_and_checkpoint_aliases(tmp_path):
    module = torch.nn.Linear(2, 2)
    operations = {'a': Transform.from_module(module), 'b': Transform.from_module(module)}
    trainer = training.Trainer.from_ops(operations)
    assert len(trainer.parameters) == 2
    assert len(trainer.optimizer.param_groups[0]['params']) == 2
    path = tmp_path / 'checkpoint.json'
    save_checkpoint(path, operations=operations, optimizer=trainer.optimizer)
    expected = module.weight.detach().clone()
    with torch.no_grad():
        module.weight.add_(5)
    load_checkpoint(path, operations=operations, optimizer=trainer.optimizer)
    assert torch.equal(module.weight, expected)
    separate = {'a': Transform.from_module(torch.nn.Linear(2, 2)), 'b': Transform.from_module(torch.nn.Linear(2, 2))}
    with pytest.raises(ValueError, match='alias'):
        load_checkpoint(path, operations=separate)


def test_supervision_held_out_improvement_and_nondifferentiable_rejection():
    torch.manual_seed(3)
    head = Classify.from_module(torch.nn.Linear(1, 2), labels=('negative', 'positive'))
    experiences = []
    for value in [-3., -1., 1., 3.]:
        with trace() as session:
            output = head(torch.tensor([value]))
        session.supervise(output, 'negative' if value < 0 else 'positive')
        experiences.append(session)
    held_out = torch.tensor([[-2.], [2.]])
    targets = torch.tensor([0, 1])
    before = torch.nn.functional.cross_entropy(head(held_out).logits, targets).item()
    losses = training.Trainer.from_ops({'head': head}, lr=0.1).fit(experiences, epochs=15)
    after = torch.nn.functional.cross_entropy(head(held_out).logits, targets).item()
    assert after < before * 0.5
    assert losses[-1] < losses[0]
    with pytest.raises(ValueError, match='source'):
        experiences[0].supervise(experiences[0].calls[0].output, 0, source='')
    detached = Transform.from_module(torch.nn.Identity())
    with trace() as session:
        out = detached(torch.tensor([1.]))
    session.supervise(out, torch.tensor([2.]), loss='mse')
    with pytest.raises(ValueError, match='parameter|differentiable'):
        training.Trainer.from_ops({'head': detached}).step(session)


def test_shared_parameter_single_optimizer_update():
    module = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(1.)
    first, second = Transform.from_module(module), Transform.from_module(module)
    with trace() as session:
        out = second(first(torch.tensor([2.])))
    session.supervise(out, torch.tensor([0.]), loss='mse')
    training.Trainer.from_ops({'a': first, 'b': second}, lr=0.01).step(session)
    # L=(2*w*w)^2, dL/dw=16 at w=1; one update reaches .84.
    assert module.weight.item() == pytest.approx(.84)


def test_custom_loss_and_optimizer_validation():
    head = Transform.from_module(torch.nn.Linear(1, 1))
    with trace() as session:
        out = head(torch.tensor([1.]))
    session.supervise(out, torch.tensor([0.]), loss='absolute', source='test:observed')
    trainer = training.Trainer.from_ops({'head': head}, losses={'absolute': lambda actual, target: (actual - target).abs().mean()})
    assert trainer.step(session) >= 0
    unrelated = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match='exactly'):
        training.Trainer.from_ops({'head': head}, optimizer=torch.optim.SGD(unrelated.parameters(), lr=.1))


def test_checkpoint_restores_optimizer_momentum(tmp_path):
    head = Transform.from_module(torch.nn.Linear(1, 1))
    trainer = training.Trainer.from_ops({'head': head}, optimizer=lambda params: torch.optim.SGD(params, lr=.1, momentum=.9))
    with trace() as session:
        out = head(torch.tensor([1.]))
    session.supervise(out, torch.tensor([0.]), loss='mse')
    trainer.step(session)
    path = tmp_path / 'momentum.json'
    save_checkpoint(path, operations={'head': head}, optimizer=trainer.optimizer)
    expected = {p: v['momentum_buffer'].clone() for p, v in trainer.optimizer.state.items()}
    trainer.step(session)
    load_checkpoint(path, operations={'head': head}, optimizer=trainer.optimizer)
    for parameter, value in expected.items():
        assert torch.equal(trainer.optimizer.state[parameter]['momentum_buffer'], value)


def test_nonfinite_custom_loss_rejects_update():
    head = Transform.from_module(torch.nn.Linear(1, 1))
    with trace() as session:
        out = head(torch.tensor([1.]))
    session.supervise(out, 0, loss='broken')
    initial = head.module.weight.detach().clone()
    trainer = training.Trainer.from_ops({'head': head}, losses={'broken': lambda out, target: out.sum() * float('nan')})
    with pytest.raises(ValueError, match='finite'):
        trainer.step(session)
    assert torch.equal(initial, head.module.weight)


def test_checkpoint_rejects_malformed_optimizer_slots_before_mutation(tmp_path):
    import json
    from tensorcode._internal.training.persistence import Codec
    head = Transform.from_module(torch.nn.Linear(1, 1))
    trainer = training.Trainer.from_ops({'head': head}, optimizer=lambda params: torch.optim.SGD(params, lr=.1, momentum=.9))
    with trace() as session:
        out = head(torch.tensor([1.]))
    session.supervise(out, torch.tensor([0.]), loss='mse')
    trainer.step(session)
    path = tmp_path / 'bad-optimizer.json'
    save_checkpoint(path, operations={'head': head}, optimizer=trainer.optimizer)
    payload = json.loads(path.read_text())
    codec = Codec()
    state = codec.decode(payload['optimizer']['state'])
    state['state'][0]['momentum_buffer'] = torch.ones(7)
    payload['optimizer']['state'] = codec.encode(state)
    path.write_text(json.dumps(payload))
    with torch.no_grad():
        head.module.weight.add_(10)
    expected = head.module.weight.detach().clone()
    with pytest.raises(ValueError, match='optimizer.*shape'):
        load_checkpoint(path, operations={'head': head}, optimizer=trainer.optimizer)
    assert torch.equal(expected, head.module.weight)

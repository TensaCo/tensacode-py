"""Explicit gradient training for captured, supervised tensor paths."""
from __future__ import annotations


def parameters(operations):
    result, seen = [], set()
    for operation in operations.values():
        for parameter in getattr(operation, 'parameters', lambda: ())():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                result.append(parameter)
    return result


def validate_optimizer(optimizer, params):
    actual = [p for group in optimizer.param_groups for p in group['params']]
    if len({id(p) for p in actual}) != len(actual):
        raise ValueError('Optimizer contains duplicate shared parameters')
    if {id(p) for p in actual} != {id(p) for p in params}:
        raise ValueError('Optimizer must own exactly the deduplicated trainable parameters')


class Trainer:
    """Train with current bound parameters and explicit target provenance.

    External operations are treated as recorded constant boundaries. No gradient
    is claimed across them or across ordinary Python outside captured operations.
    Custom losses are explicit in-process callbacks mapped by supervision name.
    """
    def __init__(self, operations, *, optimizer=None, lr=0.01, losses=None):
        import torch
        if not isinstance(operations, dict) or not operations:
            raise ValueError('Trainer needs named operation bindings')
        self.operations = dict(operations)
        self.parameters = parameters(operations)
        if not self.parameters:
            raise ValueError('No trainable parameters in supplied operations')
        self.optimizer = torch.optim.SGD(self.parameters, lr=lr) if optimizer is None else (
            optimizer(self.parameters) if callable(optimizer) else optimizer)
        validate_optimizer(self.optimizer, self.parameters)
        self.losses = dict(losses or {})
        if any(not isinstance(k, str) or not callable(v) for k, v in self.losses.items()):
            raise TypeError('Custom losses must map names to callables')

    def _loss(self, output, supervision):
        import torch
        if supervision.loss in self.losses:
            loss = self.losses[supervision.loss](output, supervision.target)
        else:
            prediction = getattr(output, 'logits', output)
            if not isinstance(prediction, torch.Tensor) or not prediction.requires_grad:
                raise ValueError('Supervision target is not a differentiable tensor path')
            if supervision.loss == 'cross_entropy':
                target = supervision.target
                labels = getattr(output, 'labels', None)
                if isinstance(target, str):
                    if labels is None or target not in labels:
                        raise ValueError('Target label is absent from prediction labels')
                    target = labels.index(target)
                elif isinstance(target, (list, tuple)) and target and isinstance(target[0], str):
                    if labels is None or any(t not in labels for t in target):
                        raise ValueError('Target labels are absent from prediction labels')
                    target = [labels.index(t) for t in target]
                raw = torch.as_tensor(target, device=prediction.device)
                if raw.dtype.is_floating_point or raw.dtype == torch.bool:
                    raise ValueError('Cross-entropy targets must be integer indices or named labels')
                loss = torch.nn.functional.cross_entropy(prediction, raw.long())
            elif supervision.loss == 'mse':
                target = torch.as_tensor(supervision.target, dtype=prediction.dtype, device=prediction.device)
                if prediction.shape != target.shape:
                    raise ValueError('MSE target shape must exactly match prediction')
                loss = torch.nn.functional.mse_loss(prediction, target)
            else:
                raise ValueError(f'Unknown loss: {supervision.loss}')
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not loss.requires_grad:
            raise ValueError('Loss must be a differentiable scalar tensor')
        if not torch.isfinite(loss):
            raise ValueError('Loss must be finite')
        return loss

    def step(self, session):
        import torch
        if not session.supervisions:
            raise ValueError('Experience has no explicit supervision')
        allowed = {id(op) for op in self.operations.values()}
        for supervision in session.supervisions:
            for index in session.example(supervision.output).calls:
                if id(session.calls[index].operation) not in allowed:
                    raise ValueError('Experience uses an operation outside Trainer bindings')
        validate_optimizer(self.optimizer, self.parameters)
        self.optimizer.zero_grad(set_to_none=True)
        try:
            losses = [self._loss(session.replay(s.output, boundary='recorded'), s) for s in session.supervisions]
            loss = torch.stack(losses).mean()
            loss.backward()
            if not any(p.grad is not None for p in self.parameters):
                raise ValueError('Loss has no differentiable path to bound parameters')
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.parameters):
                raise ValueError('Nonfinite gradients; optimizer update rejected')
            self.optimizer.step()
            return float(loss.detach())
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            raise

    def fit(self, sessions, *, epochs=1):
        if type(epochs) is not int or epochs < 1:
            raise ValueError('epochs must be a positive integer')
        sessions = list(sessions)
        if not sessions:
            raise ValueError('No experiences supplied')
        return [self.step(session) for _ in range(epochs) for session in sessions]

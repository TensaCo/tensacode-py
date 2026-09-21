"""Empirical calibration on explicit held-out scores and reviewed labels.

These utilities never fit source-model weights. Calibration describes the supplied
sample; it is neither a statistical guarantee nor general epistemic probability.
Recalibrate after changing source weights or the deployment distribution.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _validated(logits, labels):
    if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
        raise TypeError('logits must be a floating-point tensor')
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2:
        raise ValueError('logits must have nonempty shape [samples, classes >= 2]')
    if not torch.isfinite(logits).all():
        raise ValueError('logits must be finite')
    if not isinstance(labels, torch.Tensor) or labels.dtype not in (torch.int32, torch.int64):
        raise TypeError('labels must be integer class indices')
    if labels.shape != (logits.shape[0],):
        raise ValueError('labels must have one class index per sample')
    if ((labels < 0) | (labels >= logits.shape[1])).any():
        raise ValueError('label index outside class range')
    return logits.detach().to(device='cpu', dtype=torch.float64), labels.detach().to(device='cpu', dtype=torch.long)


def evaluate_calibration(logits, labels, *, n_bins=15):
    """Return empirical NLL, multiclass Brier, accuracy and equal-width ECE."""
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError('n_bins must be a positive integer')
    scores, targets = _validated(logits, labels)
    probabilities = scores.softmax(-1)
    confidence, predictions = probabilities.max(-1)
    correct = predictions.eq(targets).double()
    bins = (confidence * n_bins).long().clamp(max=n_bins - 1)
    ece = 0.
    for index in range(n_bins):
        selected = bins == index
        if selected.any():
            ece += float(selected.double().mean() * (confidence[selected].mean() - correct[selected].mean()).abs())
    return {'nll': float(torch.nn.functional.cross_entropy(scores, targets)),
            'brier': float((probabilities - torch.nn.functional.one_hot(targets, scores.shape[1])).square().sum(-1).mean()),
            'accuracy': float(correct.mean()), 'ece': ece, 'sample_count': len(targets)}


class TemperatureCalibration(nn.Module):
    """Positive bounded scalar temperature fitted only to held-out logits.

    Forward preserves argmax and input gradients. ``calibrated`` is false until
    explicit ``fit``; it records fit provenance, not a correctness guarantee.
    Buffers exist at construction and persist in ordinary module state dicts.
    """

    def __init__(self, *, min_temperature=0.05, max_temperature=100., iterations=64):
        super().__init__()
        if not (math.isfinite(min_temperature) and math.isfinite(max_temperature)
                and 0 < min_temperature <= 1 <= max_temperature):
            raise ValueError('temperature bounds must be finite, positive and contain 1')
        if isinstance(iterations, bool) or not isinstance(iterations, int) or not 1 <= iterations <= 256:
            raise ValueError('iterations must be an integer between 1 and 256')
        self.min_temperature = float(min_temperature)
        self.max_temperature = float(max_temperature)
        self.iterations = iterations
        self.register_buffer('temperature', torch.tensor(1., dtype=torch.float64))
        self.register_buffer('calibrated', torch.tensor(False))
        self.register_buffer('sample_count', torch.tensor(0, dtype=torch.long))

    def configuration(self):
        return {'min_temperature': self.min_temperature, 'max_temperature': self.max_temperature,
                'iterations': self.iterations}

    def forward(self, logits):
        if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
            raise TypeError('logits must be a floating-point tensor')
        if logits.ndim < 1 or logits.shape[-1] < 2 or not torch.isfinite(logits).all():
            raise ValueError('logits must be finite with at least two classes')
        return logits / self.temperature.to(device=logits.device, dtype=logits.dtype)

    @torch.no_grad()
    def fit(self, logits, labels):
        """Minimize held-out NLL by deterministic convex inverse-temperature search.

        Pass held-out model scores and explicit class labels, never training scores
        as evidence of held-out quality. Inputs are detached; no gradients accrue.
        """
        scores, targets = _validated(logits, labels)
        before = evaluate_calibration(scores, targets)
        lower, upper = 1 / self.max_temperature, 1 / self.min_temperature
        target_scores = scores.gather(1, targets[:, None]).squeeze(1)
        for _ in range(self.iterations):
            midpoint = (lower + upper) / 2
            derivative = ((scores * midpoint).softmax(-1) * scores).sum(-1).sub(target_scores).mean()
            if derivative > 0:
                upper = midpoint
            else:
                lower = midpoint
        # Include the identity and exact bounds so fit never worsens identity NLL.
        candidates = [1., self.min_temperature, self.max_temperature, 1 / ((lower + upper) / 2)]
        temperature = min(candidates, key=lambda value: float(torch.nn.functional.cross_entropy(scores / value, targets)))
        after = evaluate_calibration(scores / temperature, targets)
        if not all(math.isfinite(after[key]) for key in ('nll', 'brier', 'ece')):
            raise ValueError('calibration computation produced nonfinite metrics')
        self.temperature.fill_(temperature)
        self.calibrated.fill_(True)
        self.sample_count.fill_(len(targets))
        return {'before': before, 'after': after, 'temperature': temperature}


def fit_threshold(scores, labels, *, max_error):
    """Select maximal empirical coverage subject to supplied sample error limit.

    ``scores`` are confidence values in [0, 1]; ``labels`` are explicit boolean
    correctness labels. Accept scores >= threshold, or abstain on all when the
    threshold is None. Ties stay together. There is no finite-sample guarantee
    for new inputs: selection and error reporting use the same calibration data.
    """
    if not isinstance(max_error, (int, float)) or not math.isfinite(max_error) or not 0 <= max_error <= 1:
        raise ValueError('max_error must be finite and in [0, 1]')
    if not isinstance(scores, torch.Tensor) or not scores.is_floating_point():
        raise TypeError('scores must be floating-point confidence values')
    if scores.ndim != 1 or not scores.numel() or not torch.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('scores must be nonempty finite confidence values in [0, 1]')
    if not isinstance(labels, torch.Tensor) or labels.dtype != torch.bool:
        raise TypeError('labels must be explicit boolean correctness labels')
    if labels.shape != scores.shape:
        raise ValueError('one correctness label is required per confidence score')
    values = scores.detach().cpu().double()
    correct = labels.detach().cpu()
    order = values.argsort(descending=True, stable=True)
    ordered = values[order]
    errors = (~correct[order]).long().cumsum(0)
    count = 0
    for index in range(len(values)):
        if index + 1 < len(values) and ordered[index] == ordered[index + 1]:
            continue
        if float(errors[index]) / (index + 1) <= max_error:
            count = index + 1
    return {'threshold': float(ordered[count - 1]) if count else None,
            'accepted_count': count, 'sample_count': len(values),
            'coverage': count / len(values),
            'error': float(errors[count - 1]) / count if count else None,
            'max_error': float(max_error), 'empirical': True}

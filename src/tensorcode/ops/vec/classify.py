"""Owned trainable label heads over vector inputs."""
from dataclasses import dataclass
import torch
from .transform import Transform


@dataclass(frozen=True)
class Prediction:
    """Label logits for one item ``(labels,)`` or a batch ``(batch, labels)``."""
    logits: torch.Tensor
    labels: tuple[str, ...]

    @property
    def probabilities(self):
        """Softmax over labels; uncalibrated unless separately calibrated."""
        return self.logits.softmax(dim=-1)

    @property
    def value(self):
        """Top label for a single (unbatched) prediction."""
        if self.logits.ndim != 1:
            raise ValueError('Use values for batched predictions')
        return self.labels[int(self.logits.argmax())]

    @property
    def values(self):
        """Top label per row for a batched prediction."""
        if self.logits.ndim != 2:
            raise ValueError('values requires a batch of predictions')
        return tuple(self.labels[i] for i in self.logits.argmax(dim=-1).tolist())


class Classify(Transform):
    """Owned trainable label head; native transformer bridges start untrained."""
    kind = 'classify'

    def forward(self, value, *, context=None):
        """Return a ``Prediction`` over the configured labels."""
        logits = self._tensor(value, context)
        if not isinstance(logits, torch.Tensor) or logits.ndim not in (1, 2) or logits.shape[-1] != len(self.labels):
            raise ValueError('Model logits must match the labels (single item or batch)')
        return Prediction(logits, self.labels)


__all__ = ['Classify', 'Prediction']

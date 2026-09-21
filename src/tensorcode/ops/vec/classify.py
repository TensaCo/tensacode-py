from dataclasses import dataclass
import torch
from .transform import Transform


@dataclass(frozen=True)
class Prediction:
    logits: torch.Tensor
    labels: tuple[str, ...]

    @property
    def probabilities(self):
        return self.logits.softmax(dim=-1)

    @property
    def value(self):
        if self.logits.ndim != 1:
            raise ValueError('Use values for batched predictions')
        return self.labels[int(self.logits.argmax())]

    @property
    def values(self):
        if self.logits.ndim != 2:
            raise ValueError('values requires a batch of predictions')
        return tuple(self.labels[i] for i in self.logits.argmax(dim=-1).tolist())


class Classify(Transform):
    """Owned trainable label head; native transformer bridges start untrained."""
    kind = 'classify'

    def forward(self, value, *, context=None):
        logits = self._tensor(value, context)
        if not isinstance(logits, torch.Tensor) or logits.ndim not in (1, 2) or logits.shape[-1] != len(self.labels):
            raise ValueError('Model logits must match the labels (single item or batch)')
        return Prediction(logits, self.labels)

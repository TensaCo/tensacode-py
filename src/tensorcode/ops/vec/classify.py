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
    def __init__(self, module, *, labels, combine=None):
        super().__init__(module, combine=combine)
        self.labels = tuple(labels)
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError('labels must be nonempty and unique')

    def forward(self, value, *, context=None):
        logits = super().forward(value, context=context)
        if not isinstance(logits, torch.Tensor) or logits.ndim not in (1, 2) or logits.shape[-1] != len(self.labels):
            raise ValueError('Model logits must match the labels (single item or batch)')
        return Prediction(logits, self.labels)

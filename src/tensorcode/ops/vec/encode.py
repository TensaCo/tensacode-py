"""Small trainable text encoder with an explicit, caller-supplied vocabulary.

Lowercase regex tokenization and mean pooling are authored mechanics. The word
embeddings are learned parameters; this is not a pretrained language model.
"""
import re
import torch
from torch import nn
from .transform import Transform


def tokenize(text):
    if not isinstance(text, str):
        raise TypeError('TextEncoder expects strings')
    return re.findall(r"\w+|[^\w\s]", text.lower())


class TextEncoder(Transform):
    def __init__(self, *, vocabulary, dimensions=64):
        nn.Module.__init__(self)
        self.vocabulary = tuple(vocabulary)
        if len(set(self.vocabulary)) != len(self.vocabulary) or not all(isinstance(w, str) for w in self.vocabulary):
            raise ValueError('Vocabulary must contain unique strings')
        self.lookup = {word: index + 1 for index, word in enumerate(self.vocabulary)}
        self.embedding = nn.EmbeddingBag(len(self.vocabulary) + 1, dimensions, mode='mean')

    def forward(self, value, *, context=None):
        if context:
            raise ValueError('TextEncoder does not consume context')
        single = isinstance(value, str)
        texts = (value,) if single else tuple(value)
        if not texts:
            raise ValueError('Empty batch')
        indices, offsets = [], []
        for text in texts:
            offsets.append(len(indices))
            indices.extend([self.lookup.get(t, 0) for t in tokenize(text)] or [0])
        device = self.embedding.weight.device
        result = self.embedding(
            torch.tensor(indices, dtype=torch.long, device=device),
            torch.tensor(offsets, dtype=torch.long, device=device),
        )
        return result[0] if single else result

    def get_extra_state(self):
        return {'vocabulary': self.vocabulary}

    def set_extra_state(self, state):
        if tuple(state['vocabulary']) != self.vocabulary:
            raise ValueError('Checkpoint vocabulary differs from encoder configuration')

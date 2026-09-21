"""Small trainable text encoder with an explicit, caller-supplied vocabulary.

Lowercase regex tokenization and mean pooling are authored mechanics. The word
embeddings are learned parameters; this is not a pretrained language model.
"""
import re
import torch
from torch import nn
from tensorcode.ops.vec.latent import Latent, Space
from tensorcode._internal.latent_ops import LatentOperation
from tensorcode._internal.operation_config import validated_config


def tokenize(text):
    if not isinstance(text, str):
        raise TypeError('VocabularyEncoder expects strings')
    return re.findall(r"\w+|[^\w\s]", text.lower())


class VocabularyEncoder(LatentOperation):
    def __init__(self, config):
        config = validated_config(config, {'vocabulary', 'dimensions', 'output_space'},
                                  {'dimensions': 64, 'output_space': None})
        vocabulary = config.get('vocabulary')
        dimensions = config['dimensions']
        if not isinstance(vocabulary, list) or not all(isinstance(w, str) for w in vocabulary):
            raise ValueError('Vocabulary must be a list of unique strings')
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError('Vocabulary must contain unique strings')
        if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions <= 0:
            raise ValueError('dimensions must be a positive integer')
        output_space = None if config['output_space'] is None else Space(**config['output_space'])
        if output_space is not None and output_space.dimensions != dimensions:
            raise ValueError('VocabularyEncoder output_space dimensions must match dimensions')
        super().__init__(config)
        self.vocabulary = tuple(vocabulary)
        self.lookup = {word: index + 1 for index, word in enumerate(self.vocabulary)}
        self.embedding = nn.EmbeddingBag(len(self.vocabulary) + 1, dimensions, mode='mean')
        self.output_space = output_space

    def forward(self, value, *, context=None):
        if context:
            raise ValueError('VocabularyEncoder does not consume context')
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
        result = result[0] if single else result
        return result if self.output_space is None else Latent(result, self.output_space)

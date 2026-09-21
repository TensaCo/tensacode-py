"""Small trainable text encoder with an explicit, caller-supplied vocabulary.

Lowercase regex tokenization and mean pooling are authored mechanics. The word
embeddings are learned parameters; this is not a pretrained language model.
"""
import re
import torch
from torch import nn
from ._configuration import module_configuration, qualified_name, space_configuration
from .latent import Latent, Space
from .transform import Transform
from ..base import Operation


def tokenize(text):
    if not isinstance(text, str):
        raise TypeError('TextEncoder expects strings')
    return re.findall(r"\w+|[^\w\s]", text.lower())


class TextEncoder(Transform):
    def __init__(self, *, vocabulary, dimensions=64, space: Space | None = None):
        nn.Module.__init__(self)
        self.vocabulary = tuple(vocabulary)
        if len(set(self.vocabulary)) != len(self.vocabulary) or not all(isinstance(w, str) for w in self.vocabulary):
            raise ValueError('Vocabulary must contain unique strings')
        self.lookup = {word: index + 1 for index, word in enumerate(self.vocabulary)}
        self.embedding = nn.EmbeddingBag(len(self.vocabulary) + 1, dimensions, mode='mean')
        if space is not None and space.dimensions != dimensions:
            raise ValueError('TextEncoder space dimensions must match dimensions')
        self.space = space

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
        result = result[0] if single else result
        return result if self.space is None else Latent(result, self.space)

    def get_extra_state(self):
        return {
            'vocabulary': self.vocabulary,
            'space': space_configuration(self.space),
        }

    def set_extra_state(self, state):
        if tuple(state['vocabulary']) != self.vocabulary:
            raise ValueError('Checkpoint vocabulary differs from encoder configuration')
        if state.get('space') != space_configuration(self.space):
            raise ValueError('Checkpoint space differs from encoder configuration')

    def configuration(self):
        return {
            'operation': qualified_name(self),
            'vocabulary': list(self.vocabulary),
            'dimensions': self.embedding.embedding_dim,
            'space': space_configuration(self.space),
            'module': module_configuration(self.embedding),
            'tokenization': 'lowercase-regex-word-or-punctuation-v1',
            'pooling': 'mean',
        }


class SequenceEncoder(Operation):
    """Tokenize text and run an owner's local pretrained sequence encoder.

    Ownership stays with the enclosing model; this callable exposes its encoding
    boundary without registering the shared encoder/decoder weights twice.
    """
    replayable = True

    def __init__(self, model, tokenizer, *, max_tokens=512):
        import weakref
        self._model = weakref.ref(model)
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        import hashlib
        import json
        specification = json.loads(tokenizer.backend_tokenizer.to_str())
        specification["padding"] = None
        specification["truncation"] = None
        self.tokenizer_sha256 = hashlib.sha256(
            json.dumps(specification, sort_keys=True).encode()).hexdigest()

    def forward(self, value, *, context=None):
        if context:
            raise ValueError('SequenceEncoder does not consume context')
        model = self._model()
        if model is None:
            raise RuntimeError('SequenceEncoder owner has been released')
        batch = self.tokenizer(value, padding=True, truncation=True,
                               max_length=self.max_tokens, return_tensors='pt')
        batch = {key: tensor.to(next(model.parameters()).device)
                 for key, tensor in batch.items() if key in ('input_ids', 'attention_mask')}
        encoded = model.get_encoder()(**batch, return_dict=True).last_hidden_state
        return {'encoded': encoded, 'mask': batch['attention_mask']}

    def configuration(self):
        import hashlib
        import json
        model = self._model()
        return {'operation': 'tensorcode.ops.vec.encode.SequenceEncoder',
                'max_tokens': self.max_tokens,
                'model': json.loads(model.config.to_json_string()),
                'tokenizer_sha256': self.tokenizer_sha256,
                'special_tokens': self.tokenizer.special_tokens_map}

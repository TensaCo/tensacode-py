"""Owned sentence embeddings with an explicit masked-mean/L2 model contract.

Only foundations documented for this pooling contract are compatible. Loading a
plain language encoder does not confer contrastively learned retrieval ability.
"""
from __future__ import annotations

import copy
import math
import json
from pathlib import Path
import threading

import torch
from torch.nn import functional as F

from .pretrained import PretrainedTool
from .ranking import FoundationEncoding
from ..ops.vec import Transform


class _RetrievalTransform(Transform):
    """Public Transform execution with an explicit native-model identity.

    Transformers keeps non-JSON runtime caches on child modules. Its complete
    native configuration and registered tensor schemas describe this owned
    architecture without serializing those implementation caches.
    """
    def configuration(self):
        return {'operation': type(self).__module__ + '.' + type(self).__qualname__,
                'module': self.module.configuration(),
                'parameters': [{'name': name, 'shape': list(value.shape),
                                'dtype': str(value.dtype), 'requires_grad': value.requires_grad}
                               for name, value in self.named_parameters()],
                'buffers': [{'name': name, 'shape': list(value.shape), 'dtype': str(value.dtype)}
                            for name, value in self.named_buffers()]}


class RetrievalEncoder(PretrainedTool):
    """A complete encoder/tokenizer artifact, without an added projection.

    Configuration construction initializes random weights. ``from_foundation``
    explicitly imports weights; the caller must select a foundation whose model
    card specifies masked mean pooling and L2 normalization at ``max_tokens``.
    """
    def __init__(self, config):
        config = self._validated_config(json.loads(json.dumps(config, allow_nan=False)))
        allowed = {'foundation_config', 'tokenizer_json', 'tokenizer_special_tokens',
                   'pooling', 'normalize', 'max_tokens', 'freeze_foundation', 'foundation'}
        if set(config) - allowed:
            raise ValueError('unsupported retrieval encoder configuration fields')
        if config.get('pooling') != 'masked_mean' or config.get('normalize') is not True:
            raise ValueError('retrieval requires explicit pooling=masked_mean and normalize=True')
        if not isinstance(config.get('foundation_config'), dict) or not isinstance(config.get('tokenizer_json'), str):
            raise ValueError('retrieval requires complete native encoder and fast tokenizer configuration')
        config.setdefault('max_tokens', 256)
        config.setdefault('freeze_foundation', False)
        config.setdefault('tokenizer_special_tokens', {})
        if type(config['max_tokens']) is not int or config['max_tokens'] < 1:
            raise ValueError('max_tokens must be a positive integer')
        if type(config['freeze_foundation']) is not bool:
            raise ValueError('freeze_foundation must be boolean')
        if not isinstance(config['tokenizer_special_tokens'], dict):
            raise ValueError('tokenizer_special_tokens must be an object')
        if 'foundation' in config and not isinstance(config['foundation'], dict):
            raise ValueError('foundation provenance must be an object')
        native = config['foundation_config']
        if native.get('is_encoder_decoder') or native.get('is_decoder'):
            raise ValueError('retrieval currently supports encoder-only foundations')
        super().__init__(config)
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(config['tokenizer_json']),
                                                **config['tokenizer_special_tokens'])
        if self.tokenizer.pad_token_id is None:
            raise ValueError('retrieval tokenizer requires a padding token')
        self.encode = _RetrievalTransform(FoundationEncoding(config))
        self._lock = threading.RLock()

    @property
    def metadata(self):
        return {'encoder': 'owned_retrieval_encoder', 'pooling': 'masked_mean',
                'normalized': True, 'max_tokens': self.config['max_tokens'],
                'dimensions': self.encode.module.model.config.hidden_size,
                'foundation': copy.deepcopy(self.config.get('foundation', {'initialization': 'configured_weights'})),
                'compatibility': 'requires a foundation trained/documented for masked-mean pooling followed by L2 normalization',
                'semantics': 'embedding proximity, not truth or calibrated evidence support'}

    @staticmethod
    def _validate_texts(texts):
        if not isinstance(texts, (list, tuple)) or not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError('retrieval input must be a nonempty sequence of nonempty strings')

    def forward(self, inputs, *, context=None):
        if context:
            raise ValueError('retrieval encoder does not accept context')
        self._validate_texts(inputs)
        tokens = self.tokenizer(list(inputs), padding=True, truncation=True,
                                max_length=self.config['max_tokens'], return_tensors='pt')
        device = next(self.encode.parameters()).device
        tokens = {key: tensor.to(device) for key, tensor in tokens.items()}
        hidden = self.encode(tokens).float()
        mask = tokens['attention_mask'].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return F.normalize(pooled, p=2, dim=-1)

    def receipt(self, texts):
        """Inference vectors and explicit source truncation/pooling metadata."""
        self._validate_texts(texts)
        with self._lock:
            modes = [(module, module.training) for module in self.modules()]
            try:
                self.eval()
                with torch.no_grad():
                    vectors = self(texts).cpu().tolist()
            finally:
                for module, training in modes:
                    module.training = training
        lengths = [len(self.tokenizer(text, truncation=False)['input_ids']) for text in texts]
        return dict(self.metadata, embeddings=vectors,
                    input_truncated=[length > self.config['max_tokens'] for length in lengths],
                    input_token_counts=lengths)

    def contrastive_loss(self, queries, documents, positive_mask, *, temperature=.05):
        """Explicit multi-positive contrastive supervision; no inferred positives."""
        self._validate_texts(queries); self._validate_texts(documents)
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be finite and positive')
        positives = torch.as_tensor(positive_mask, device=next(self.parameters()).device)
        if positives.dtype != torch.bool or positives.shape != (len(queries), len(documents)) or not positives.any(dim=1).all():
            raise ValueError('positive_mask must be boolean [queries, documents] with a positive for each query')
        logits = self(queries) @ self(documents).T / temperature
        weights = positives.to(logits.dtype) / positives.sum(dim=1, keepdim=True)
        return -(weights * logits.log_softmax(dim=-1)).sum(dim=-1).mean()

    loss = contrastive_loss

    @classmethod
    def from_foundation(cls, repo, *, pooling, normalize, revision=None, max_tokens=256,
                        freeze_foundation=False, local_files_only=False):
        """Load an explicitly selected compatible model; never download in __init__.

        ``pooling='masked_mean', normalize=True`` is a compatibility declaration,
        not automatic discovery of arbitrary SentenceTransformers module graphs.
        CLS, weighted pooling, prefixes and learned post-pooling projections are
        unsupported. Pin remote revisions and use the model's documented limit.
        """
        if pooling != 'masked_mean' or normalize is not True:
            raise ValueError('only masked_mean followed by L2 normalization is supported')
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision,
                    local_files_only=local_files_only, use_fast=True, trust_remote_code=False)
        native = AutoModel.from_pretrained(repo, revision=revision,
                    local_files_only=local_files_only, trust_remote_code=False)
        if not hasattr(tokenizer, 'backend_tokenizer'):
            raise ValueError('retrieval foundation requires a serializable fast tokenizer')
        resolved = getattr(native.config, '_commit_hash', None) or revision
        if not Path(repo).is_dir() and not resolved:
            raise ValueError('retrieval foundation provenance requires a resolved revision')
        config = {'foundation_config': native.config.to_dict(),
                  'tokenizer_json': tokenizer.backend_tokenizer.to_str(),
                  'tokenizer_special_tokens': {key: str(value) for key, value in tokenizer.special_tokens_map.items() if isinstance(value,str)},
                  'pooling': pooling, 'normalize': normalize, 'max_tokens': max_tokens,
                  'freeze_foundation': freeze_foundation,
                  'foundation': {'repository': str(repo), 'revision': resolved,
                                 'pooling_contract': 'caller_declared_masked_mean_l2',
                                 'weights': 'loaded_foundation'}}
        result = cls(config)
        result.encode.module.model.load_state_dict(native.state_dict())
        result.eval()
        return result
